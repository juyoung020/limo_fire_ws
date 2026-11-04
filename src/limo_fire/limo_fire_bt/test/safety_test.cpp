#include "limo_fire_bt/safety.hpp"
#include <cassert>
#include <limits>
int main() {
  using namespace limo_fire_bt;
  assert(classify(.2, true, .2, .3) == Size::Small);
  assert(classify(.3, true, .2, .3) == Size::Large);
  assert(classify(.25, true, .2, .3) == Size::Unknown);
  assert(classify(.1, false, .2, .3) == Size::Unknown);
  assert(classify(std::numeric_limits<double>::quiet_NaN(), true, .2, .3) ==
         Size::Unknown);
  assert(classify(-1, true, .2, .3) == Size::Unknown);
  std::vector<Circle> z{{0, 0, 1}};
  assert(
      !segment_safe(-2, 0, 2, 0, z)); // endpoints alone would miss the hazard
  assert(!segment_safe(-2, 1, 2, 1, z));
  assert(segment_safe(-2, 1.01, 2, 1.01, z));
  assert(!segment_safe(0, 0, 0, 0, z));
}
