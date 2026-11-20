// 발사대 조작 터미널 — 사람이 따로 연 창에서 글자 하나씩 친다. 시리얼은 ESP32 다리(esp32_bridge)가 혼자 열고, 여기는 ROS 로만 말한다.
//   a  표적 찾기(없으면 프론티어 탐사를 켜고, 찾으면 끄고 정지 확인) → 접근·조준·장전+발사 임무 한 번 (approach_aim auto_fire_enabled 일 때만)
//   r  공 다시 넣음 확인 — 자동 임무가 발사를 요청했을 수 있으면 r 전에는 다음 a 를 받지 않는다(미리 친 a 도)
//   c · f · s  장전 · 발사 · 장전+발사 (액션 /launcher/launch)     x  레버 끔(고장 풀기도 이것)
//   ?  상태     q  끝
// Ctrl+C: 탐사 중이면 탐사를 끄고, 임무 중이면 임무 취소를, 수동 동작 중이면 동작 취소(x)를 보낸 뒤 끝낸다. 응답을 못 받을 수 있으니 정지는 현장에서 확인.
// 옵션: --log <파일> (JSONL 기록) · --search-timeout-s <초, 기본 180> · --no-search (지금 보이는 표적만)
// 예전 Python launcher_control_terminal.py 를 옮겼다.
#include <signal.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <deque>
#include <fstream>
#include <iostream>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "limo_fire_msgs/action/launch.hpp"
#include "limo_fire_msgs/msg/fire_observation.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_srvs/srv/set_bool.hpp"
#include "std_srvs/srv/trigger.hpp"

using namespace std::chrono_literals;
using Launch = limo_fire_msgs::action::Launch;
using Fire = limo_fire_msgs::msg::FireObservation;
using SetBool = std_srvs::srv::SetBool;
using Trigger = std_srvs::srv::Trigger;

namespace {
std::atomic<bool> g_int{false};
void on_sigint(int) { g_int = true; }
struct Interrupted {};

double mono() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
void sleep_s(double s) { std::this_thread::sleep_for(std::chrono::duration<double>(s)); }

std::string json_escape(const std::string& s) {
  std::string o;
  for (char c : s) {
    if (c == '"' || c == '\\') o += '\\', o += c;
    else if (c == '\n') o += "\\n";
    else if (static_cast<unsigned char>(c) < 0x20) o += ' ';
    else o += c;
  }
  return o;
}

// 이 터미널이 허용하는 조준 노드 상태 (임무가 끝났거나 쉼)
const std::set<std::string> kAimIdle = {"idle",      "preview_ready",    "aim_ready", "shot_done", "shot_rejected",
                                        "shot_unconfirmed", "failed", "cancelled", "stop_unconfirmed"};
const std::set<std::string> kAimEnd = {"preview_ready", "aim_ready", "shot_done", "shot_rejected", "shot_unconfirmed",
                                       "failed", "cancelled", "stop_unconfirmed"};
const std::set<std::string> kShotStates = {"wait_shot", "shot_done", "shot_rejected", "shot_unconfirmed"};

class Terminal {
 public:
  Terminal(const std::string& log, double search_timeout, bool no_search)
      : node_(std::make_shared<rclcpp::Node>("launcher_terminal")), search_timeout_(search_timeout), no_search_(no_search) {
    if (!log.empty()) log_.open(log, std::ios::app);
    const auto latched = rclcpp::QoS(10).transient_local().reliable();
    subs_.push_back(node_->create_subscription<std_msgs::msg::String>(
        "/launcher/approach_aim/state", latched, [this](std_msgs::msg::String::SharedPtr m) {
          std::lock_guard<std::mutex> l(mu_);
          aim_state_ = m->data;
          aim_states_.push_back(m->data);
        }));
    subs_.push_back(node_->create_subscription<std_msgs::msg::String>(
        "/launcher/approach_aim/status", latched, [this](std_msgs::msg::String::SharedPtr m) {
          std::lock_guard<std::mutex> l(mu_);
          aim_status_ = m->data;
          aim_statuses_.push_back(m->data);
        }));
    subs_.push_back(node_->create_subscription<std_msgs::msg::String>(
        "/launcher/state", latched, [this](std_msgs::msg::String::SharedPtr m) {
          std::lock_guard<std::mutex> l(mu_);
          launcher_state_ = m->data;
        }));
    subs_.push_back(node_->create_subscription<std_msgs::msg::String>(
        "/launcher/status", latched, [this](std_msgs::msg::String::SharedPtr m) {
          std::lock_guard<std::mutex> l(mu_);
          launcher_status_ = m->data;
        }));
    subs_.push_back(node_->create_subscription<std_msgs::msg::String>(
        "/explore/frontier_explorer/state", latched, [this](std_msgs::msg::String::SharedPtr m) {
          std::lock_guard<std::mutex> l(mu_);
          explorer_.emplace_back(m->data, mono());
          if (explorer_.size() > 20) explorer_.pop_front();
        }));
    fire_sub_ = node_->create_subscription<Fire>("/fire/observations", 10, [this](Fire::SharedPtr m) {
      std::lock_guard<std::mutex> l(mu_);
      fires_.push_back(m);
      if (fires_.size() > 40) fires_.pop_front();
    });
    odom_sub_ = node_->create_subscription<nav_msgs::msg::Odometry>(
        "/odometry/filtered", rclcpp::SensorDataQoS(), [this](nav_msgs::msg::Odometry::SharedPtr m) {
          std::lock_guard<std::mutex> l(mu_);
          odom_ = m;
        });
    launch_ = rclcpp_action::create_client<Launch>(node_, "/launcher/launch");
    lever_ = node_->create_client<SetBool>("/launcher/enable");
    start_ = node_->create_client<Trigger>("/launcher/approach_aim/start");
    cancel_ = node_->create_client<Trigger>("/launcher/approach_aim/cancel");
    explore_ = node_->create_client<SetBool>("/explore/frontier_explorer/enable");
    params_ = std::make_shared<rclcpp::AsyncParametersClient>(node_, "/launcher/approach_aim");
    exec_.add_node(node_);
    spin_ = std::thread([this] { exec_.spin(); });
  }
  ~Terminal() {
    exec_.cancel();
    if (spin_.joinable()) spin_.join();
  }

  int run() {
    say("발사대 터미널: a=찾기·접근·조준·발사, r=공 다시 넣음 확인(다음 a 허용), c=장전, f=발사, s=장전+발사, x=레버 끔, ?=상태, q=끝");
    say("ESP32 다리가 USB 포트를 혼자 엽니다. 명령 완료는 펌웨어 응답이며 공의 비행을 재지 않습니다. 발사대 각은 15° 고정.");
    try {
      std::string line;
      while (rclcpp::ok()) {
        std::cout << "launcher> " << std::flush;
        if (!std::getline(std::cin, line)) break;   // EOF 또는 Ctrl+C(EINTR)
        if (g_int) break;
        const auto b = line.find_first_not_of(" \t\r"), e = line.find_last_not_of(" \t\r");
        const std::string cmd = b == std::string::npos ? "" : line.substr(b, e - b + 1);
        if (cmd.empty()) continue;
        if (cmd == "q") break;
        if (cmd == "?") show();
        else if (cmd == "r") {
          spent_ = false;
          record("reload_confirmed");
          say("공 다시 넣음 확인. 다음 a 한 번을 받습니다.");
        } else if (cmd == "a") mission();
        else if (cmd == "c" || cmd == "f" || cmd == "s") manual(cmd[0]);
        else if (cmd == "x") lever_off();
        else say("a, r, c, f, s, x, ?, q 중 하나를 치세요.");
      }
    } catch (const Interrupted&) {
      say("Ctrl+C — 끝냅니다.");
    }
    return 0;
  }

 private:
  void say(const std::string& s) { std::cout << s << std::endl; }
  void check_int() {
    if (g_int) throw Interrupted{};
  }
  void record(const std::string& event, const std::vector<std::pair<std::string, std::string>>& f = {}) {
    if (!log_) return;
    char ts[32];
    const std::time_t t = std::time(nullptr);
    std::strftime(ts, sizeof ts, "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t));
    log_ << "{\"time_utc\":\"" << ts << "\",\"event\":\"" << json_escape(event) << "\"";
    for (const auto& [k, v] : f) log_ << ",\"" << json_escape(k) << "\":\"" << json_escape(v) << "\"";
    log_ << "}" << std::endl;
  }
  std::string aim_state() {
    std::lock_guard<std::mutex> l(mu_);
    return aim_state_;
  }
  void show() {
    sleep_s(0.2);
    std::lock_guard<std::mutex> l(mu_);
    say("조준: " + (aim_status_.empty() ? std::string("상태 없음 (approach_aim 이 꺼져 있음?)") : aim_status_));
    say("발사대: " + (launcher_state_.empty() ? std::string("상태 없음 (esp32_bridge 가 꺼져 있음?)")
                                              : launcher_state_ + " — " + launcher_status_));
  }

  // 같은 표적을 서로 다른 영상 시각으로 세 번(0.25 m 안) 봤고 마지막이 1 초 안
  bool target_visible() {
    std::lock_guard<std::mutex> l(mu_);
    if (fires_.empty()) return false;
    const auto now = node_->now();
    auto valid = [&](const Fire& m) {
      const double age = (now - rclcpp::Time(m.header.stamp, now.get_clock_type())).seconds();
      return m.position_valid && m.header.frame_id == "map" && std::isfinite(m.confidence) && m.confidence >= 0.5 &&
             std::isfinite(m.position.x) && std::isfinite(m.position.y) && std::isfinite(m.position.z) && age >= -0.1 &&
             age <= 1.0;
    };
    const auto& t = *fires_.back();
    if (!valid(t)) return false;
    std::set<std::pair<int32_t, uint32_t>> stamps;
    for (auto it = fires_.rbegin(); it != fires_.rend(); ++it) {
      const auto& m = **it;
      if (m.id != t.id || !valid(m)) continue;
      if (std::hypot(m.position.x - t.position.x, m.position.y - t.position.y, m.position.z - t.position.z) > 0.25) continue;
      stamps.insert({m.header.stamp.sec, m.header.stamp.nanosec});
      if (stamps.size() >= 3) return true;
    }
    return false;
  }
  bool robot_stopped() {
    std::lock_guard<std::mutex> l(mu_);
    if (!odom_) return false;
    const auto now = node_->now();
    const double age = (now - rclcpp::Time(odom_->header.stamp, now.get_clock_type())).seconds();
    const auto& tw = odom_->twist.twist;
    return age >= -0.1 && age <= 0.5 && std::isfinite(tw.linear.x) && std::isfinite(tw.linear.y) &&
           std::isfinite(tw.angular.z) && std::hypot(tw.linear.x, tw.linear.y) <= 0.01 && std::fabs(tw.angular.z) <= 0.02;
  }
  std::optional<std::pair<std::string, double>> explorer_latest() {
    std::lock_guard<std::mutex> l(mu_);
    if (explorer_.empty()) return std::nullopt;
    return explorer_.back();
  }
  bool explorer_seen(const std::string& s, double since) {
    std::lock_guard<std::mutex> l(mu_);
    for (const auto& [st, t] : explorer_)
      if (st == s && t >= since) return true;
    return false;
  }

  template <class Srv>
  typename Srv::Response::SharedPtr call(typename rclcpp::Client<Srv>::SharedPtr c, typename Srv::Request::SharedPtr req,
                                         double wait_s, double reply_s) {
    if (!c->wait_for_service(std::chrono::duration<double>(wait_s))) return nullptr;
    auto f = c->async_send_request(req);
    if (f.wait_for(std::chrono::duration<double>(reply_s)) != std::future_status::ready) return nullptr;
    return f.get();
  }
  void set_exploration(bool on) {
    auto req = std::make_shared<SetBool::Request>();
    req->data = on;
    const auto r = call<SetBool>(explore_, req, 2.0, 3.0);
    if (!r || !r->success)
      throw std::runtime_error(std::string("프론티어 탐사 ") + (on ? "시작" : "정지") + " 응답을 확인하지 못함");
    record(on ? "exploration_enabled" : "exploration_disabled", {{"response", r->message}});
  }

  // 표적이 보이면 바로, 아니면 탐사를 켜고 찾은 뒤 끄고 정지를 확인한다. 실패는 runtime_error, Ctrl+C 는 Interrupted (둘 다 탐사를 끄고 나감)
  void search_until_target() {
    // 최근 표적은 1 초 이상 듣고 본다. 탐사 상태(TRANSIENT_LOCAL)는 다른 프로세스라 찾는 데 1 초 넘게 걸릴 수 있어 5 초까지 기다린다
    const double t0 = mono();
    while (mono() - t0 < 5.0) {
      check_int();
      sleep_s(0.1);
      if (mono() - t0 >= 1.0 && (no_search_ || explorer_latest())) break;
    }
    const auto ex = explorer_latest();
    if (ex && ex->first == "running") throw std::runtime_error("이미 다른 프론티어 탐사가 진행 중 — 탐사를 멈춘 뒤 다시");
    if (target_visible()) return;
    if (no_search_) throw std::runtime_error("최근 1 초 안의 지도 좌표 표적이 없음");
    if (!ex || (ex->first != "idle" && ex->first != "done")) throw std::runtime_error("프론티어 탐사 상태를 확인하지 못함");
    bool exploring = false;
    try {
      const double enable_at = mono();
      set_exploration(true);
      exploring = true;
      say("표적을 찾을 때까지 프론티어 탐사를 합니다.");
      const double deadline = mono() + search_timeout_, running_deadline = mono() + 3.0;
      bool seen_running = false, found = false;
      // 서비스 응답과 상태 토픽은 따로 온다 — 이번 실행의 running 을 본 뒤에만 idle/done 을 끝으로 본다
      while (rclcpp::ok() && mono() < deadline) {
        check_int();
        sleep_s(0.1);
        if (!seen_running) seen_running = explorer_seen("running", enable_at);
        if (target_visible()) { found = true; break; }
        const auto l = explorer_latest();
        if (seen_running && l && l->second >= enable_at && (l->first == "done" || l->first == "idle"))
          throw std::runtime_error("탐사가 끝났지만 쓸 만한 표적을 찾지 못함");
        if (!seen_running && mono() > running_deadline) throw std::runtime_error("프론티어 탐사 running 상태를 확인하지 못함");
      }
      if (!found) throw std::runtime_error("찾기 제한 시간이 지났거나 ROS 가 끝남");
      const double stop_at = mono();
      set_exploration(false);
      exploring = false;
      std::optional<double> still;
      const double handoff = mono() + 8.0;
      while (rclcpp::ok() && mono() < handoff) {
        check_int();
        sleep_s(0.1);
        const auto l = explorer_latest();
        if (l && l->first == "idle" && l->second >= stop_at && robot_stopped()) {
          if (!still) still = mono();
          if (mono() - *still >= 0.5) {
            if (!target_visible()) throw std::runtime_error("정지 기다리는 동안 표적 관측이 오래됨");
            record("exploration_handoff");
            return;
          }
        } else {
          still.reset();
        }
      }
      throw std::runtime_error("탐사 정지와 차체 정지를 확인하지 못함");
    } catch (...) {
      if (exploring) {
        try {
          set_exploration(false);
        } catch (const std::exception& e) {
          say(std::string("탐사 정지 확인 실패: ") + e.what());
          record("exploration_stop_unconfirmed", {{"reason", e.what()}});
        }
      }
      throw;
    }
  }

  void mission() {
    if (spent_) {
      say("앞선 자동 임무가 발사를 요청했을 수 있습니다. 공과 발사기를 확인한 뒤 r을 입력하세요.");
      record("auto_mission_refused", {{"reason", "reload_not_confirmed"}});
      return;
    }
    // 조준 노드에 닿는지 + 자동 발사가 켜졌는지 (파라미터로 확인 — 노드가 살아 있다는 증거도 됨)
    if (!params_->wait_for_service(5s)) return say("5 초 안에 조준 노드(/launcher/approach_aim)에 닿지 못함. 실행과 ROS 도메인을 확인하세요.");
    auto pf = params_->get_parameters({"auto_fire_enabled"});
    if (pf.wait_for(3s) != std::future_status::ready || pf.get().empty())
      return say("조준 노드 파라미터를 읽지 못함.");
    if (!pf.get()[0].as_bool()) return say("자동 발사가 꺼져 있습니다. launch 에서 auto_fire_enabled:=true 를 주세요.");
    if (!start_->wait_for_service(2s)) return say("조준 임무 서비스가 없습니다.");
    for (int i = 0; i < 20 && aim_state().empty(); ++i) sleep_s(0.1);
    if (!kAimIdle.count(aim_state())) return say("다른 조준 임무가 진행 중입니다 (" + aim_state() + ").");
    try {
      search_until_target();
    } catch (const std::runtime_error& e) {
      say(std::string("자동 임무 시작 안 함: ") + e.what());
      record("search_failed", {{"reason", e.what()}});
      return;
    }
    size_t seen_states, seen_statuses;   // 요청 뒤에 온 것만 본다
    {
      std::lock_guard<std::mutex> l(mu_);
      seen_states = aim_states_.size();
      seen_statuses = aim_statuses_.size();
    }
    const auto reply = call<Trigger>(start_, std::make_shared<Trigger::Request>(), 2.0, 3.0);
    if (!reply) {
      record("mission_unconfirmed");
      return say("임무 요청 응답을 확인하지 못함. 상태를 먼저 확인하세요.");
    }
    record("mission_request", {{"accepted", reply->success ? "true" : "false"}, {"message", reply->message}});
    if (!reply->success) return say("임무 거부: " + reply->message);
    say("자동 임무 시작. 접근·조준·발사 상태를 기다립니다.");
    const double deadline = mono() + 130.0;
    try {
      while (rclcpp::ok() && mono() < deadline) {
        check_int();
        sleep_s(0.05);
        std::vector<std::string> new_states, new_statuses;
        {
          std::lock_guard<std::mutex> l(mu_);
          new_states.assign(aim_states_.begin() + seen_states, aim_states_.end());
          new_statuses.assign(aim_statuses_.begin() + seen_statuses, aim_statuses_.end());
          seen_states = aim_states_.size();
          seen_statuses = aim_statuses_.size();
        }
        for (const auto& t : new_statuses) {
          say("조준: " + t);
          record("mission_status", {{"status", t}});
        }
        bool ended = false;
        for (const auto& st : new_states) {
          if (kShotStates.count(st)) spent_ = true;
          if (kAimEnd.count(st)) ended = true;
        }
        if (ended) {
          sleep_s(0.1);   // 끝 상태의 글이 따라오게
          std::lock_guard<std::mutex> l(mu_);
          for (size_t i = seen_statuses; i < aim_statuses_.size(); ++i) say("조준: " + aim_statuses_[i]);
          return;
        }
      }
      spent_ = true;
      say("임무 끝을 확인하지 못함. 다시 하기 전에 로봇과 발사기를 점검하세요.");
      record("mission_unconfirmed");
    } catch (const Interrupted&) {
      spent_ = true;
      say("Ctrl+C: 조준 임무 취소를 요청합니다. 로봇 정지를 직접 확인하세요.");
      const auto r = call<Trigger>(cancel_, std::make_shared<Trigger::Request>(), 1.0, 2.0);
      say("취소 응답: " + (r ? r->message : std::string("확인하지 못함")));
      record("mission_cancel_requested", {{"accepted", r && r->success ? "true" : "false"}});
      throw;
    }
  }

  void manual(char c) {
    if (!kAimIdle.count(aim_state()) && !aim_state().empty())
      return say("자동 조준 임무가 진행 중입니다. 수동 장전·발사를 보내지 않습니다.");
    if (!launch_->wait_for_action_server(2s)) return say("발사대 액션(/launcher/launch)이 없습니다. ESP32 다리를 확인하세요.");
    Launch::Goal g;
    g.mode = c == 'c' ? Launch::Goal::COCK : c == 'f' ? Launch::Goal::FIRE : Launch::Goal::SHOT;
    const std::string action = c == 'c' ? "cock" : c == 'f' ? "fire" : "shot";
    rclcpp_action::Client<Launch>::SendGoalOptions opt;
    opt.feedback_callback = [this](rclcpp_action::ClientGoalHandle<Launch>::SharedPtr,
                                   const std::shared_ptr<const Launch::Feedback> fb) { say("  … " + fb->phase); };
    auto gf = launch_->async_send_goal(g, opt);
    if (gf.wait_for(3s) != std::future_status::ready || !gf.get()) {
      record("unconfirmed", {{"command", action}});
      return say("발사대가 목표를 받지 않음. 상태를 확인하세요.");
    }
    auto gh = gf.get();
    record("request", {{"command", action}});
    auto rf = launch_->async_get_result(gh);
    const double deadline = mono() + 15.0;
    try {
      while (rf.wait_for(50ms) != std::future_status::ready) {
        check_int();
        if (mono() > deadline) {
          record("unconfirmed", {{"command", action}});
          return say("응답 시간 초과. 기구를 확인하기 전에는 같은 명령을 되풀이하지 마세요.");
        }
      }
    } catch (const Interrupted&) {
      say("Ctrl+C: 발사대 동작 취소(x)를 보냅니다.");
      launch_->async_cancel_goal(gh);
      sleep_s(0.5);
      throw;
    }
    const auto w = rf.get();
    const char* code = w.code == rclcpp_action::ResultCode::SUCCEEDED ? "completed"
                       : w.code == rclcpp_action::ResultCode::CANCELED ? "canceled"
                                                                        : "error";
    say(std::string(code) + " " + (w.result ? w.result->message : ""));
    record("result", {{"command", action}, {"state", code}, {"message", w.result ? w.result->message : ""}});
  }

  void lever_off() {
    auto req = std::make_shared<SetBool::Request>();
    req->data = false;
    const auto r = call<SetBool>(lever_, req, 2.0, 5.0);
    say(r ? std::string(r->success ? "레버 꺼짐: " : "끄기 실패: ") + r->message : "레버 끄기 응답을 받지 못함");
    record("lever_off", {{"ok", r && r->success ? "true" : "false"}});
  }

  rclcpp::Node::SharedPtr node_;
  rclcpp::executors::SingleThreadedExecutor exec_;
  std::thread spin_;
  std::vector<rclcpp::Subscription<std_msgs::msg::String>::SharedPtr> subs_;
  rclcpp::Subscription<Fire>::SharedPtr fire_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp_action::Client<Launch>::SharedPtr launch_;
  rclcpp::Client<SetBool>::SharedPtr lever_, explore_;
  rclcpp::Client<Trigger>::SharedPtr start_, cancel_;
  std::shared_ptr<rclcpp::AsyncParametersClient> params_;
  std::mutex mu_;
  std::string aim_state_, aim_status_, launcher_state_, launcher_status_;
  std::vector<std::string> aim_states_, aim_statuses_;   // 받은 차례대로 전부 (빨리 지나간 상태도 놓치지 않게)
  std::deque<std::pair<std::string, double>> explorer_;
  std::deque<Fire::SharedPtr> fires_;
  nav_msgs::msg::Odometry::SharedPtr odom_;
  double search_timeout_;
  bool no_search_;
  bool spent_ = false;   // 자동 임무가 발사를 요청했을 수 있음 → r 전엔 a 거절
  std::ofstream log_;
};
}  // namespace

int main(int argc, char** argv) {
  std::string log;
  double search_timeout = 180.0;
  bool no_search = false;
  const auto args = rclcpp::remove_ros_arguments(argc, argv);
  for (size_t i = 1; i < args.size(); ++i) {
    if (args[i] == "--log" && i + 1 < args.size()) log = args[++i];
    else if (args[i] == "--search-timeout-s" && i + 1 < args.size()) search_timeout = std::atof(args[++i].c_str());
    else if (args[i] == "--no-search") no_search = true;
    else {
      std::fprintf(stderr, "사용: launcher_terminal [--log 파일] [--search-timeout-s 초] [--no-search]\n");
      return 2;
    }
  }
  if (!std::isfinite(search_timeout) || search_timeout <= 0) {
    std::fprintf(stderr, "--search-timeout-s 는 양수\n");
    return 2;
  }
  // rclcpp 기본 신호 처리기는 Ctrl+C 에 바로 컨텍스트를 닫아 탐사 끄기·임무 취소를 보낼 수 없다 → 우리 처리기(SA_RESTART 없음: 입력 대기도 깨움)
  rclcpp::init(argc, argv, rclcpp::InitOptions(), rclcpp::SignalHandlerOptions::None);
  struct sigaction sa {};
  sa.sa_handler = on_sigint;
  sigaction(SIGINT, &sa, nullptr);
  sigaction(SIGTERM, &sa, nullptr);
  int code;
  {
    Terminal t(log, search_timeout, no_search);
    code = t.run();
  }
  rclcpp::shutdown();
  return code;
}
