// 지시 클래스 묻기 시험 (class_ask.h): colcon test --packages-select limo_fire_explore
#include <gtest/gtest.h>

#include <cstdio>
#include <deque>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "limo_fire_explore/class_ask.h"

using limo_fire_explore::ClassAsk;

namespace {
// 보낸 것을 모아 두고, 넣어 준 답을 내놓는 가짜 보내는 쪽
struct Fake : lxd::Transport {
  std::vector<lxd::Req> sent;
  std::deque<lxd::Answer> ans;
  void send(const lxd::Req& r) override { sent.push_back(r); }
  bool poll(lxd::Answer* a) override {
    if (ans.empty()) return false;
    *a = ans.front();
    ans.pop_front();
    return true;
  }
};
lxd::Answer answer(long id, const std::string& choice) {
  lxd::Answer a;
  a.id = id;
  a.ok = true;
  a.choice = choice;
  return a;
}
}  // namespace

TEST(ClassAsk, BodyHasAllClassesAndNone) {
  const ClassAsk ask({"person", "potted_plant", "tv"});
  const auto j = nlohmann::json::parse(ask.body("초록 식물 있는 데로 가"));
  EXPECT_EQ(j["state"]["instruction"], "초록 식물 있는 데로 가");
  const auto& c = j["questions"]["cls"];
  EXPECT_EQ(c["type"], "choice");
  EXPECT_EQ(c["criteria"].size(), 4u);
  EXPECT_TRUE(c["criteria"].contains("none"));
  EXPECT_TRUE(c["criteria"].contains("potted plant"));   // 밑줄은 띄어 씀
}

TEST(ClassAsk, AnswerToClass) {
  Fake tr;
  ClassAsk ask({"person", "potted_plant", "tv"});
  ask.ask(tr, "초록 식물 있는 데로 가", 10);
  ASSERT_EQ(tr.sent.size(), 1u);
  EXPECT_EQ(tr.sent[0].qid, "cls");
  EXPECT_TRUE(ask.pending());
  EXPECT_FALSE(ask.poll(tr, 11, 80));
  tr.ans.push_back(answer(tr.sent[0].id - 1, "tv"));            // 옛 물음의 답은 버림
  tr.ans.push_back(answer(tr.sent[0].id, "potted plant"));
  EXPECT_TRUE(ask.poll(tr, 12, 80));
  EXPECT_EQ(ask.cls(), "potted_plant");
  EXPECT_FALSE(ask.pending());
  EXPECT_FALSE(ask.poll(tr, 13, 80));                           // 한 번만
}

TEST(ClassAsk, NoneFailureAndTimeout) {
  Fake tr;
  ClassAsk ask({"person", "chair"});
  ask.ask(tr, "피아노 찾아", 0);
  tr.ans.push_back(answer(tr.sent.back().id, "none"));
  EXPECT_TRUE(ask.poll(tr, 1, 80));
  EXPECT_EQ(ask.cls(), "");
  ask.ask(tr, "의자로 가", 5);
  lxd::Answer bad;
  bad.id = tr.sent.back().id;
  bad.err = "answer shape";
  tr.ans.push_back(bad);
  EXPECT_TRUE(ask.poll(tr, 6, 80));
  EXPECT_EQ(ask.cls(), "");
  EXPECT_EQ(ask.err(), "answer shape");
  ask.ask(tr, "의자로 가", 100);
  EXPECT_FALSE(ask.poll(tr, 180, 80));
  EXPECT_TRUE(ask.poll(tr, 181, 80));
  EXPECT_EQ(ask.err(), "timeout");
}

TEST(ClassAsk, EmptyListOrInstructionAnswersAtOnce) {
  Fake tr;
  ClassAsk none_list(std::vector<std::string>{});
  none_list.ask(tr, "의자로 가", 0);
  EXPECT_TRUE(tr.sent.empty());
  EXPECT_TRUE(none_list.poll(tr, 0, 80));
  EXPECT_EQ(none_list.cls(), "");
  ClassAsk ask({"chair"});
  ask.ask(tr, "", 0);
  EXPECT_TRUE(tr.sent.empty());
  EXPECT_TRUE(ask.poll(tr, 0, 80));
}

TEST(ClassAsk, ParseAnswerQuestion) {
  const std::string raw = R"({"answers":{"cls":{"type":"choice","choice":"dining table","confidence":0.9}}})";
  const auto a = lxd::parse_answer(3, raw, "cls");
  EXPECT_TRUE(a.ok);
  EXPECT_EQ(a.choice, "dining table");
  EXPECT_FALSE(lxd::parse_answer(3, raw).ok);   // pick 질문이 없음
  EXPECT_EQ(ClassAsk({"dining_table"}).class_of(a), "dining_table");
}

TEST(ClassAsk, LoadClassFiles) {
  const std::string a = testing::TempDir() + "/cls_a.yaml", b = testing::TempDir() + "/cls_b.yaml";
  std::ofstream(a) << "/perception/object_detector:\n  ros__parameters:\n    class_names: [person, chair]\n/perception/object_map:\n  ros__parameters:\n    min_score: 0.5\n";
  std::ofstream(b) << "/perception/object_map:\n  ros__parameters:\n    class_names: [fire, person]\n";
  const auto c = limo_fire_explore::load_class_files({a, b}, "/nowhere");
  EXPECT_EQ(c, (std::vector<std::string>{"person", "chair", "fire"}));
  std::remove(a.c_str());
  std::remove(b.c_str());
}
