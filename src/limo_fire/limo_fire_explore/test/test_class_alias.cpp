// 지시 글 풀기 시험 (class_alias.h — 클래스 힌트 class_of): colcon test --packages-select limo_fire_explore
#include <gtest/gtest.h>

#include "limo_fire_explore/class_alias.h"

using limo_fire_explore::class_of;
using limo_fire_explore::parse_aliases;

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
