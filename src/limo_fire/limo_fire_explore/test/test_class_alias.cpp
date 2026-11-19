// 지시 글 풀기 시험 (class_alias.h — 클래스 힌트 class_of · 그림·글 모델에 물을 명사구 query_phrase): colcon test --packages-select limo_fire_explore
#include <gtest/gtest.h>

#include "limo_fire_explore/class_alias.h"

using limo_fire_explore::class_of;
using limo_fire_explore::parse_aliases;
using limo_fire_explore::query_phrase;

namespace {
// config/explorer.yaml decider_sem_strip 와 같은 목록
const std::vector<std::string> kStrip = {"있는 데로 가", "데로 가", "으로 가", "로 가", "가져와", "찾아줘", "찾아", "어디 있어", "가줘", "take me to ", "go to the ",
                                         "go to a ", "go to ", "find the ", "find a ", "find ", "where is the ", "is there any "};
}  // namespace

TEST(ClassAlias, ClassOf) {
  const auto al = parse_aliases({"의자=chair", "소화기=fire_extinguisher"});
  EXPECT_EQ(class_of("가장 먼 의자로 가서 멈춰", al, {}), "chair");
  EXPECT_EQ(class_of("불 끄는 거 찾아", al, {}), "");
  EXPECT_EQ(class_of("tv 앞으로", al, {"tv"}), "tv");
}

// 영문 클래스 이름 지시는 그 클래스가 물체 지도에 들어온 뒤에야 풀림 → 노드 refresh_class 가 물체 후보를 줄 때마다 다시 풂
TEST(ClassAlias, EnglishNeedsKnownClass) {
  const auto al = parse_aliases({"침대=bed"});
  EXPECT_EQ(class_of("find the bed", al, {}), "");
  EXPECT_EQ(class_of("find the bed", al, {"chair", "bed"}), "bed");
  EXPECT_EQ(class_of("go to the potted plant", al, {"potted_plant"}), "potted_plant");
}

TEST(ClassAlias, QueryPhrase) {
  EXPECT_EQ(query_phrase("의자로 가", kStrip), "의자");
  EXPECT_EQ(query_phrase("앉을 수 있는 곳으로 가", kStrip), "앉을 수 있는 곳");
  EXPECT_EQ(query_phrase("냉장고 어디 있어?", kStrip), "냉장고");
  EXPECT_EQ(query_phrase("곰인형 찾아줘", kStrip), "곰인형");
  EXPECT_EQ(query_phrase("티비 있는 데로 가", kStrip), "티비");
  EXPECT_EQ(query_phrase("불 끄는 거 가져와", kStrip), "불 끄는 거");
  EXPECT_EQ(query_phrase("Find the fire extinguisher.", kStrip), "fire extinguisher");
  EXPECT_EQ(query_phrase("Where is the TV?", kStrip), "TV");
  EXPECT_EQ(query_phrase("찾아", kStrip), "찾아");   // 다 떼면 원래 글
  EXPECT_EQ(query_phrase("빨간 의자", {}), "빨간 의자");
}
