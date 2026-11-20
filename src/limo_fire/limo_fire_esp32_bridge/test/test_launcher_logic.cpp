// 발사대 상태 기계(launcher_logic.hpp): 시간을 직접 넣어 응답·기한·재시작·취소 규칙을 확인한다 (ROS·시리얼 없음)
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "limo_fire_esp32_bridge/launcher_logic.hpp"

using namespace limo_fire_esp32_bridge;

namespace {
LauncherReply R(LauncherReply::Kind k) { return LauncherReply{k}; }
LauncherReply State(bool on) {
  LauncherReply r{LauncherReply::kState};
  r.lever_deg = 100.0;
  r.on = on;
  return r;
}

struct Rig {
  std::vector<std::string> sent, feedback;
  std::vector<std::pair<GoalEnd, std::string>> goals;
  std::vector<std::pair<bool, std::string>> enables;
  std::vector<double> lever;
  bool write_ok = true;
  LauncherLogic logic;

  explicit Rig(bool firing = true)
      : logic(LauncherConfig{firing, 3.0, 8.0, 1.0},
              LauncherLogic::Callbacks{
                  [this](const std::string& l) { sent.push_back(l); return write_ok; },
                  [this](bool ok, const std::string& m) { enables.emplace_back(ok, m); },
                  [this](const std::string& p) { feedback.push_back(p); },
                  [this](GoalEnd e, const std::string& m) { goals.emplace_back(e, m); },
                  [] {},
                  [this](double d) { lever.push_back(d); }}) {}

  // 연결 → 부팅 응답 → 상태 응답(레버 꺼짐) 까지
  void connect(double t = 0.0) {
    logic.link_up(t);
    logic.reply(R(LauncherReply::kBoot), t + 0.1);
    logic.reply(State(false), t + 0.2);
  }
  std::string last() const { return sent.empty() ? "" : sent.back(); }
};
}  // namespace

TEST(LauncherLogic, QueriesUntilFirmwareAnswers) {
  Rig r;
  r.logic.link_up(0.0);
  EXPECT_EQ(r.logic.state(), LauncherState::kDisconnected);
  EXPECT_EQ(r.last(), "?");
  r.sent.clear();
  r.logic.tick(0.5);
  EXPECT_TRUE(r.sent.empty());
  r.logic.tick(1.01);
  EXPECT_EQ(r.last(), "?");
  r.logic.reply(State(false), 1.1);
  EXPECT_EQ(r.logic.state(), LauncherState::kOff);
  ASSERT_EQ(r.lever.size(), 1u);
  EXPECT_DOUBLE_EQ(r.lever[0], 100.0);
}

TEST(LauncherLogic, ShotEnablesLeverFirstThenCocksAndFires) {
  Rig r;
  r.connect();
  r.sent.clear();
  EXPECT_EQ(r.logic.start_goal(LaunchMode::kShot, 1.0), "");
  EXPECT_EQ(r.sent, (std::vector<std::string>{"e"}));
  EXPECT_EQ(r.logic.state(), LauncherState::kBusy);
  r.logic.reply(R(LauncherReply::kOn), 1.01);
  EXPECT_EQ(r.last(), "s");
  r.logic.reply(R(LauncherReply::kCocked), 3.0);
  EXPECT_TRUE(r.goals.empty());
  r.logic.reply(R(LauncherReply::kFired), 3.5);
  ASSERT_EQ(r.goals.size(), 1u);
  EXPECT_EQ(r.goals[0].first, GoalEnd::kSucceeded);
  EXPECT_EQ(r.feedback, (std::vector<std::string>{"enabling", "cocking", "firing"}));
  EXPECT_FALSE(r.logic.cocked());
  EXPECT_EQ(r.logic.state(), LauncherState::kReady);
  EXPECT_EQ(r.last(), "?");   // 끝난 뒤 레버 각을 다시 물음
}

TEST(LauncherLogic, CockThenFireAndCockRefusals) {
  Rig r;
  r.connect();
  EXPECT_NE(r.logic.start_goal(LaunchMode::kFire, 1.0), "");   // 장전 전 발사 거절
  EXPECT_EQ(r.logic.start_goal(LaunchMode::kCock, 1.0), "");
  r.logic.reply(R(LauncherReply::kOn), 1.0);
  EXPECT_EQ(r.last(), "c");
  r.logic.reply(R(LauncherReply::kCocked), 3.0);
  EXPECT_TRUE(r.logic.cocked());
  EXPECT_EQ(r.logic.state(), LauncherState::kCocked);
  EXPECT_NE(r.logic.start_goal(LaunchMode::kCock, 3.1), "");   // 두 번 장전 거절
  EXPECT_NE(r.logic.start_goal(LaunchMode::kShot, 3.1), "");
  r.sent.clear();
  EXPECT_EQ(r.logic.start_goal(LaunchMode::kFire, 3.2), "");
  EXPECT_EQ(r.sent, (std::vector<std::string>{"f"}));          // 레버가 켜져 있으면 바로
  r.logic.reply(R(LauncherReply::kFired), 3.7);
  EXPECT_FALSE(r.logic.cocked());
  EXPECT_EQ(r.goals.back().first, GoalEnd::kSucceeded);
}

TEST(LauncherLogic, FiringDisabledRefusesMotionButAllowsOff) {
  Rig r(false);
  r.connect();
  EXPECT_NE(r.logic.start_goal(LaunchMode::kShot, 1.0), "");
  EXPECT_NE(r.logic.set_enabled(true, 1.0), "");
  EXPECT_EQ(r.logic.set_enabled(false, 1.0), "");
  EXPECT_EQ(r.last(), "x");
  r.logic.reply(R(LauncherReply::kOff), 1.01);
  ASSERT_EQ(r.enables.size(), 1u);
  EXPECT_TRUE(r.enables[0].first);
}

TEST(LauncherLogic, ReplyDeadlineFaultsStopsAndDoesNotRetry) {
  Rig r;
  r.connect();
  r.logic.start_goal(LaunchMode::kShot, 1.0);
  r.logic.reply(R(LauncherReply::kOn), 1.0);
  r.sent.clear();
  r.logic.tick(8.9);
  EXPECT_TRUE(r.goals.empty());
  r.logic.tick(9.1);   // 동작 명령 뒤 8 s
  ASSERT_EQ(r.goals.size(), 1u);
  EXPECT_EQ(r.goals[0].first, GoalEnd::kAborted);
  EXPECT_EQ(r.sent, (std::vector<std::string>{"x"}));   // 끄기만, 다시 s 를 보내지 않음
  EXPECT_EQ(r.logic.state(), LauncherState::kFault);
  r.sent.clear();
  r.logic.tick(20.0);
  for (const auto& s : r.sent) EXPECT_NE(s, "s");
  EXPECT_NE(r.logic.start_goal(LaunchMode::kShot, 20.0), "");   // 고장이면 거절
  // 사람이 끄기로 푼다
  EXPECT_EQ(r.logic.set_enabled(false, 21.0), "");
  r.logic.reply(R(LauncherReply::kOff), 21.01);
  EXPECT_EQ(r.logic.state(), LauncherState::kOff);
  EXPECT_EQ(r.logic.start_goal(LaunchMode::kShot, 22.0), "");
}

TEST(LauncherLogic, EnableDeadlineFaults) {
  Rig r;
  r.connect();
  EXPECT_EQ(r.logic.set_enabled(true, 1.0), "");
  r.logic.tick(4.1);
  ASSERT_EQ(r.enables.size(), 1u);
  EXPECT_FALSE(r.enables[0].first);
  EXPECT_EQ(r.logic.state(), LauncherState::kFault);
}

TEST(LauncherLogic, RebootWhileRunningFaults) {
  Rig r;
  r.connect();
  r.logic.start_goal(LaunchMode::kCock, 1.0);
  r.logic.reply(R(LauncherReply::kOn), 1.0);
  r.logic.reply(R(LauncherReply::kBoot), 1.5);
  ASSERT_EQ(r.goals.size(), 1u);
  EXPECT_EQ(r.goals[0].first, GoalEnd::kAborted);
  EXPECT_EQ(r.logic.state(), LauncherState::kFault);
  EXPECT_FALSE(r.logic.lever_on());
  EXPECT_EQ(r.logic.reboots(), 2);
}

TEST(LauncherLogic, RebootWhileIdleAndOffIsOnlyNoted) {
  Rig r;
  r.connect();
  r.logic.reply(R(LauncherReply::kBoot), 5.0);
  EXPECT_EQ(r.logic.state(), LauncherState::kOff);
  EXPECT_EQ(r.logic.reboots(), 2);
}

TEST(LauncherLogic, RebootWhileLeverOnFaults) {
  Rig r;
  r.connect();
  r.logic.set_enabled(true, 1.0);
  r.logic.reply(R(LauncherReply::kOn), 1.0);
  EXPECT_EQ(r.logic.state(), LauncherState::kReady);
  r.logic.reply(R(LauncherReply::kBoot), 5.0);
  EXPECT_EQ(r.logic.state(), LauncherState::kFault);
}

TEST(LauncherLogic, CancelSendsXAndLeavesFaultUntilAcknowledged) {
  Rig r;
  r.connect();
  r.logic.start_goal(LaunchMode::kShot, 1.0);
  r.logic.reply(R(LauncherReply::kOn), 1.0);
  r.logic.cancel_goal(1.5);
  EXPECT_EQ(r.last(), "x");
  r.logic.reply(R(LauncherReply::kOff), 1.51);
  ASSERT_EQ(r.goals.size(), 1u);
  EXPECT_EQ(r.goals[0].first, GoalEnd::kCanceled);
  EXPECT_EQ(r.logic.state(), LauncherState::kFault);
  r.logic.set_enabled(false, 2.0);
  r.logic.reply(R(LauncherReply::kOff), 2.01);
  EXPECT_EQ(r.logic.state(), LauncherState::kOff);
}

TEST(LauncherLogic, FirmwareErrorsAndOutOfOrderRepliesFault) {
  {
    Rig r;
    r.connect();
    r.logic.start_goal(LaunchMode::kCock, 1.0);
    LauncherReply e{LauncherReply::kError};
    e.detail = "attach";
    r.logic.reply(e, 1.01);
    EXPECT_EQ(r.goals.back().first, GoalEnd::kAborted);
    EXPECT_EQ(r.logic.state(), LauncherState::kFault);
  }
  {
    Rig r;
    r.connect();
    r.logic.start_goal(LaunchMode::kShot, 1.0);
    r.logic.reply(R(LauncherReply::kOn), 1.0);
    r.logic.reply(R(LauncherReply::kFired), 1.2);   // cocked 전에 fired
    EXPECT_EQ(r.goals.back().first, GoalEnd::kAborted);
    EXPECT_EQ(r.logic.state(), LauncherState::kFault);
  }
  {
    Rig r;
    r.connect();
    r.logic.start_goal(LaunchMode::kShot, 1.0);
    r.logic.reply(R(LauncherReply::kOn), 1.0);
    LauncherReply b{LauncherReply::kBlocked};
    b.detail = "off";
    r.logic.reply(b, 1.1);
    EXPECT_EQ(r.logic.state(), LauncherState::kFault);
  }
}

TEST(LauncherLogic, BusyRefusesSecondRequest) {
  Rig r;
  r.connect();
  EXPECT_EQ(r.logic.start_goal(LaunchMode::kShot, 1.0), "");
  EXPECT_NE(r.logic.start_goal(LaunchMode::kCock, 1.0), "");
  EXPECT_NE(r.logic.set_enabled(true, 1.0), "");
}

TEST(LauncherLogic, LinkLossDuringGoalFaults) {
  Rig r;
  r.connect();
  r.logic.start_goal(LaunchMode::kShot, 1.0);
  r.logic.link_down(1.2);
  EXPECT_EQ(r.goals.back().first, GoalEnd::kAborted);
  EXPECT_EQ(r.logic.state(), LauncherState::kFault);
}

TEST(LauncherLogic, WriteFailureFaults) {
  Rig r;
  r.connect();
  r.write_ok = false;
  r.logic.set_enabled(true, 1.0);
  EXPECT_EQ(r.logic.state(), LauncherState::kFault);
}

TEST(LauncherLogic, UnexpectedOffWhileReadyFaults) {
  Rig r;
  r.connect();
  r.logic.set_enabled(true, 1.0);
  r.logic.reply(R(LauncherReply::kOn), 1.0);
  r.logic.reply(R(LauncherReply::kOff), 2.0);   // 누가 시리얼로 x 를 친 경우 등
  EXPECT_EQ(r.logic.state(), LauncherState::kFault);
}
