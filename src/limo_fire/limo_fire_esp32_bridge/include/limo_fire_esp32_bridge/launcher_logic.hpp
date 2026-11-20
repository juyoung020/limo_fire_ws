// 발사대 레버 상태 기계 — ROS·시리얼과 떨어진 순수 로직(test_launcher_logic 이 시간을 넣어 가며 시험한다).
// 펌웨어 명령·응답은 firmware/limo_fire_esp32/include/launcher.h. 다리(esp32_bridge_node.cpp)가 줄을 넣고, 보낼 줄·끝난 일을 콜백으로 받는다.
//
// 지키는 것 (예전 Python 다리 pitch_serial_bridge 의 안전 규칙을 이어받음)
//   - firing_enabled=false 면 켜기·장전·발사를 받지 않는다(끄기는 늘 받음).
//   - 이미 장전됐으면 장전·장전+발사를 거절, 장전 안 됐으면 발사를 거절.
//   - 레버가 꺼져 있으면 동작 전에 e 로 켜고 LA,on 을 받은 뒤 동작 명령을 보낸다.
//   - 응답 기한(켜기 ack_timeout_s, 동작 action_timeout_s)을 넘기면 고장 — x 로 끄고 다시 시도하지 않는다.
//   - 동작 중 펌웨어 재시작(LA,boot)·막힘·오류·순서 틀린 응답·연결 끊김 → 고장.
//   - 고장은 사람이 끄기(set_enabled false → x → LA,off)를 해야 풀린다. 그 전엔 켜기·동작을 받지 않는다.
//   - 동작 취소 = x (펌웨어가 움직이던 것도 바로 멈추고 쌓인 명령을 버림). 장전 여부를 모르므로 고장으로 둔다.
// 응답은 펌웨어가 명령을 끝냈다는 뜻이지 레버·공의 실제 움직임을 잰 것이 아니다.
#pragma once

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "limo_fire_esp32_bridge/protocol.hpp"

namespace limo_fire_esp32_bridge {

enum class LaunchMode { kCock = 1, kFire = 2, kShot = 3 };
enum class LauncherState { kDisconnected, kOff, kReady, kBusy, kCocked, kFault };
enum class GoalEnd { kSucceeded, kAborted, kCanceled };

inline const char* state_name(LauncherState s) {
  switch (s) {
    case LauncherState::kDisconnected: return "disconnected";
    case LauncherState::kOff: return "off";
    case LauncherState::kReady: return "ready";
    case LauncherState::kBusy: return "busy";
    case LauncherState::kCocked: return "cocked";
    case LauncherState::kFault: return "fault";
  }
  return "fault";
}

struct LauncherConfig {
  bool firing_enabled = false;
  double ack_timeout_s = 3.0;      // e·x·? 응답 기한
  double action_timeout_s = 8.0;   // 동작 명령(c·f·s) 보낸 뒤 마지막 응답까지 (펌웨어 s ≈ 2.5 s)
  double query_period_s = 1.0;     // 펌웨어를 아직 모를 때·쉴 때 ? 간격
};

class LauncherLogic {
 public:
  struct Callbacks {
    std::function<bool(const std::string&)> send;                // 명령 한 줄(\n 없이). 쓰기 실패면 false
    std::function<void(bool ok, const std::string&)> enable_done;  // set_enabled 결과
    std::function<void(const std::string& phase)> feedback;      // enabling · cocking · firing
    std::function<void(GoalEnd, const std::string&)> goal_done;
    std::function<void()> changed;                               // 상태·글이 바뀜
    std::function<void(double deg)> lever;                       // 레버 서보 명령 각(펌웨어가 알려 준 값)
  };

  LauncherLogic(LauncherConfig cfg, Callbacks cb) : cfg_(cfg), cb_(std::move(cb)) {}

  // ---- 읽기
  LauncherState state() const {
    if (!fault_.empty()) return LauncherState::kFault;
    if (!link_ || !identified_) return LauncherState::kDisconnected;
    if (op_ != Op::kNone) return LauncherState::kBusy;
    if (cocked_) return LauncherState::kCocked;
    return lever_on_ ? LauncherState::kReady : LauncherState::kOff;
  }
  bool cocked() const { return cocked_; }
  bool lever_on() const { return lever_on_; }
  bool goal_active() const { return op_ == Op::kGoal || op_ == Op::kCancel; }
  const std::string& fault() const { return fault_; }
  const std::string& last_result() const { return last_; }
  int reboots() const { return reboots_; }
  bool firing_enabled() const { return cfg_.firing_enabled; }

  // ---- 시리얼 연결
  void link_up(double now) {
    link_ = true;
    identified_ = false;
    lever_on_ = false;
    last_query_ = -1e9;
    tick(now);
    note("ESP32 포트 열림 — 펌웨어 응답 기다림");
  }
  void link_down(double /*now*/) {
    const bool was = link_;
    link_ = false;
    identified_ = false;
    if (op_ != Op::kNone || lever_on_ || cocked_) fail("ESP32 연결 끊김 (레버가 켜져 있었거나 동작 중)");
    lever_on_ = false;
    if (was) note("ESP32 연결 끊김");
  }

  // ---- 요청 (빈 글 = 받음, 아니면 거절 까닭)
  std::string set_enabled(bool on, double now) {
    if (!on) {
      if (!link_) return "ESP32 포트가 열려 있지 않음";
      if (op_ == Op::kDisable || op_ == Op::kCancel) return "이미 끄는 중";
      if (op_ == Op::kGoal) end_goal(GoalEnd::kAborted, "끄기 요청으로 동작 중단 — 장전 여부를 눈으로 확인");
      if (op_ == Op::kEnable) finish_enable(false, "끄기 요청으로 켜기 취소");
      start_op(Op::kDisable, now);
      if (!send("x")) return "시리얼 쓰기 실패";
      return "";
    }
    if (const auto r = refuse_common(); !r.empty()) return r;
    start_op(Op::kEnable, now);
    if (!send("e")) return "시리얼 쓰기 실패";
    return "";
  }

  std::string start_goal(LaunchMode mode, double now) {
    if (const auto r = refuse_common(); !r.empty()) return r;
    if (mode == LaunchMode::kFire && !cocked_) return "장전 응답을 받은 적이 없음 — 먼저 장전";
    if (mode != LaunchMode::kFire && cocked_) return "이미 장전됨 — 발사하거나 확인한 뒤 다시 장전";
    mode_ = mode;
    expect_.clear();
    if (mode != LaunchMode::kFire) expect_.push_back(LauncherReply::kCocked);
    if (mode != LaunchMode::kCock) expect_.push_back(LauncherReply::kFired);
    start_op(Op::kGoal, now);
    if (!lever_on_) {
      enabling_ = true;
      phase("enabling");
      if (!send("e")) return "";   // send 실패가 이미 고장·끝을 알림
    } else {
      motion(now);
    }
    return "";
  }

  void cancel_goal(double now) {
    if (op_ != Op::kGoal) return;
    start_op(Op::kCancel, now);
    send("x");
  }

  // ---- 펌웨어 응답
  void reply(const LauncherReply& r, double now) {
    const bool first = !identified_;
    identified_ = true;
    if (r.kind == LauncherReply::kBoot) {
      ++reboots_;
      const bool risky = op_ != Op::kNone || lever_on_ || cocked_;
      lever_on_ = false;
      if (risky && !first) {
        fail("펌웨어가 다시 켜짐(레버 꺼짐) — 장전 여부를 눈으로 확인한 뒤 끄기로 풀기");
        return;
      }
      if (risky && first) {
        // 포트를 열면 칩이 다시 켜진다 — 연결 직후 boot 는 정상. 그래도 장전 기억은 믿지 않는다
        cocked_ = false;
      }
      note("펌웨어 켜짐 — 레버 꺼짐");
      return;
    }
    if (r.kind == LauncherReply::kState) {
      if (cb_.lever) cb_.lever(r.lever_deg);
      if (op_ == Op::kNone) {
        if (lever_on_ && !r.on) { fail("펌웨어 상태는 레버 꺼짐인데 다리는 켜짐으로 알고 있었음"); return; }
        lever_on_ = r.on;
      }
      if (first) note(std::string("펌웨어 확인 — 레버 ") + (r.on ? "켜짐" : "꺼짐"));
      return;
    }
    if (r.kind == LauncherReply::kLever || r.kind == LauncherReply::kHome) {
      if (r.kind == LauncherReply::kLever && cb_.lever) cb_.lever(r.lever_deg);
      return;   // 이 다리는 h·l 을 보내지 않는다(사람이 시리얼로 친 것)
    }
    if (r.kind == LauncherReply::kOff) {
      const bool was_on = lever_on_;
      lever_on_ = false;
      if (op_ == Op::kDisable) {
        op_ = Op::kNone;
        fault_.clear();
        finish_enable(true, "레버 꺼짐");
        query(now);
        return;
      }
      if (op_ == Op::kCancel) {
        op_ = Op::kNone;
        cocked_ = false;
        end_goal(GoalEnd::kCanceled, "취소 — 레버 끔(x). 장전 여부를 눈으로 확인한 뒤 끄기로 풀기");
        fault_ = "동작 취소됨 — 장전 여부 모름";
        changed();
        return;
      }
      if (op_ != Op::kNone || was_on) fail("기다리지 않던 레버 꺼짐 응답");
      return;
    }
    if (r.kind == LauncherReply::kOn) {
      if (op_ == Op::kEnable) {
        lever_on_ = true;
        op_ = Op::kNone;
        finish_enable(true, "레버 켜짐(쉼 자리)");
        query(now);
        return;
      }
      if (op_ == Op::kGoal && enabling_) {
        lever_on_ = true;
        enabling_ = false;
        motion(now);
        return;
      }
      lever_on_ = true;
      return;
    }
    if (r.kind == LauncherReply::kBlocked || r.kind == LauncherReply::kError) {
      const std::string what = std::string(r.kind == LauncherReply::kBlocked ? "막힘 " : "오류 ") + r.detail;
      if (op_ != Op::kNone) fail("펌웨어 " + what);
      else note("펌웨어 " + what + " (기다리던 명령 없음)");
      return;
    }
    // cocked · fired
    if (op_ != Op::kGoal || enabling_) {
      if (op_ == Op::kCancel) return;   // 취소 전에 끝난 응답
      fail(std::string("기다리지 않던 응답 ") + (r.kind == LauncherReply::kCocked ? "cocked" : "fired"));
      return;
    }
    if (expect_.empty() || expect_.front() != r.kind) { fail("순서가 틀린 발사대 응답"); return; }
    expect_.erase(expect_.begin());
    if (r.kind == LauncherReply::kCocked) cocked_ = true;
    if (r.kind == LauncherReply::kFired) cocked_ = false;
    if (!expect_.empty()) { phase("firing"); return; }
    op_ = Op::kNone;
    const char* what = mode_ == LaunchMode::kCock ? "장전" : mode_ == LaunchMode::kFire ? "발사" : "장전+발사";
    end_goal(GoalEnd::kSucceeded, std::string(what) + " 끝 (펌웨어 응답, 실제 움직임은 재지 않음)");
    query(now);
  }

  // ---- 주기 (기한·질의)
  void tick(double now) {
    if (!link_) return;
    if (op_ != Op::kNone) {
      const bool motion_wait = op_ == Op::kGoal && !enabling_;
      const double limit = motion_wait ? cfg_.action_timeout_s : cfg_.ack_timeout_s;
      if (now - op_at_ > limit) {
        const Op was = op_;
        if (was == Op::kCancel) {
          op_ = Op::kNone;
          end_goal(GoalEnd::kCanceled, "취소 — x 응답을 받지 못함. 레버 전원을 직접 확인");
          fail("x(끄기) 응답 기한 넘김");
        } else if (was == Op::kDisable) {
          fail("x(끄기) 응답 기한 넘김");
        } else {
          fail(motion_wait ? "발사대 동작 응답 기한 넘김 — 다시 시도하지 않음, 기구 확인"
                           : "레버 켜기(e) 응답 기한 넘김");
          send("x");   // 멈추게만 한다(다시 시도 아님)
        }
      }
      return;
    }
    if (now - last_query_ >= cfg_.query_period_s) query(now);
  }

 private:
  enum class Op { kNone, kEnable, kDisable, kGoal, kCancel };

  std::string refuse_common() const {
    if (!cfg_.firing_enabled) return "launcher.firing_enabled=false — 설정에서 켜야 함";
    if (!link_) return "ESP32 포트가 열려 있지 않음";
    if (!identified_) return "펌웨어 발사대 응답을 아직 받지 못함";
    if (!fault_.empty()) return "고장 상태(" + fault_ + ") — 확인 뒤 끄기(set_enabled false)로 풀기";
    if (op_ != Op::kNone) return "다른 발사대 명령이 진행 중";
    return "";
  }

  void start_op(Op op, double now) {
    op_ = op;
    op_at_ = now;
    if (op != Op::kGoal) enabling_ = false;
    changed();
  }
  void motion(double now) {
    op_at_ = now;   // 동작 기한은 동작 명령을 보낸 때부터
    phase(mode_ == LaunchMode::kFire ? "firing" : "cocking");
    send(mode_ == LaunchMode::kCock ? "c" : mode_ == LaunchMode::kFire ? "f" : "s");
  }
  void query(double now) {
    last_query_ = now;
    if (link_) cb_.send("?");
  }
  bool send(const std::string& line) {
    if (cb_.send && cb_.send(line)) return true;
    fail("시리얼 쓰기 실패");
    return false;
  }
  void phase(const char* p) {
    if (cb_.feedback) cb_.feedback(p);
  }
  void fail(const std::string& why) {
    const Op was = op_;
    op_ = Op::kNone;
    enabling_ = false;
    fault_ = why;
    cocked_ = false;   // Python 다리와 같게: 고장 뒤엔 장전 기억을 버린다(사람이 확인)
    if (was == Op::kGoal || was == Op::kCancel) end_goal(GoalEnd::kAborted, "고장: " + why);
    if (was == Op::kEnable || was == Op::kDisable) finish_enable(false, "고장: " + why);
    note("고장: " + why);
  }
  void end_goal(GoalEnd e, const std::string& msg) {
    if (op_ == Op::kGoal || op_ == Op::kCancel) op_ = Op::kNone;
    enabling_ = false;
    last_ = msg;
    if (cb_.goal_done) cb_.goal_done(e, msg);
    changed();
  }
  void finish_enable(bool ok, const std::string& msg) {
    last_ = msg;
    if (cb_.enable_done) cb_.enable_done(ok, msg);
    changed();
  }
  void note(const std::string& msg) {
    last_ = msg;
    changed();
  }
  void changed() {
    if (cb_.changed) cb_.changed();
  }

  LauncherConfig cfg_;
  Callbacks cb_;
  bool link_ = false, identified_ = false, lever_on_ = false, cocked_ = false, enabling_ = false;
  Op op_ = Op::kNone;
  double op_at_ = 0.0, last_query_ = -1e9;
  LaunchMode mode_ = LaunchMode::kShot;
  std::vector<LauncherReply::Kind> expect_;
  std::string fault_, last_;
  int reboots_ = 0;
};

}  // namespace limo_fire_esp32_bridge
