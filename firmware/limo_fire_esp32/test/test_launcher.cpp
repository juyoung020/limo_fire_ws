// 호스트 시험: 레버 서보 명령이 loop() 를 막지 않고 정한 응답·순서·속도로 움직이는지 (test/run.sh)
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "launcher.h"

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("실패 %s:%d  %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

struct Said { uint32_t t; std::string body; };
static std::vector<Said> said;
static std::vector<int> lever_us;
static int other_pin_writes = 0;
static bool attached = false;
static uint32_t clock_ms = 0;

static bool hw_attach(int pin) { if (pin != kLeverPin) ++other_pin_writes; attached = true; return true; }
static void hw_detach(int) { attached = false; }
static void hw_write(int pin, int us) { if (pin == kLeverPin) lever_us.push_back(us); else ++other_pin_writes; }
static void hw_say(const char* b) { said.push_back({clock_ms, b}); }

static void run(launcher::Launcher& L, uint32_t ms) {   // loop() 가 1 ms 마다 돈다고 보고
  for (uint32_t i = 0; i < ms; ++i) { ++clock_ms; L.update(clock_ms); }
}
static int find(const std::string& s) {
  for (size_t i = 0; i < said.size(); ++i) if (said[i].body == s) return static_cast<int>(i);
  return -1;
}

int main() {
  launcher::Launcher L({hw_attach, hw_detach, hw_write, hw_say});
  L.boot();
  CHECK(said.back().body == "LA,boot");

  // 켜기 전: 움직임 명령은 막힘, 펄스 없음
  for (const char* c : {"c", "f", "s", "h", "l90"}) { L.line(c); CHECK(said.back().body == "LA,blocked,off"); }
  CHECK(lever_us.empty() && !L.busy());
  L.line("?");
  CHECK(said.back().body == "LA,state,100.0,off,0");

  // 켜면 쉼 자리 100°/180° → 1611 us 한 번
  L.line("e");
  CHECK(said.back().body == "LA,on" && attached && lever_us.size() == 1 && lever_us.back() == 1611);

  // 장전: 바로 돌아오고(막지 않음) 약 2 s 뒤 cocked, 장전 각 30° = 833 us 까지, 한 걸음 ≤ 6° (300°/s × 20 ms)
  said.clear();
  const uint32_t t0 = clock_ms;
  L.line("c");
  CHECK(L.busy() && said.empty());
  L.line("?");                        // 움직이는 중 상태 질문은 쌓였다가 끝난 뒤
  run(L, 3000);
  CHECK(said.size() == 2 && said[0].body == "LA,cocked" && said[1].body == "LA,state,100.0,on,0");
  if (!said.empty()) { const uint32_t dt = said[0].t - t0; std::printf("c %u ms\n", dt); CHECK(dt > 1900 && dt < 2100); }
  int min_us = 9999;
  for (size_t i = 1; i < lever_us.size(); ++i) {
    min_us = std::min(min_us, lever_us[i]);
    CHECK(std::abs(lever_us[i] - lever_us[i - 1]) <= 68);
  }
  CHECK(min_us == 833 && lever_us.back() == 1611);

  // 발사: 125° = 1889 us 까지, 약 0.5 s
  said.clear(); lever_us.clear();
  const uint32_t t1 = clock_ms;
  L.line("f");
  run(L, 1000);
  CHECK(find("LA,fired") == 0);
  if (!said.empty()) { const uint32_t dt = said[0].t - t1; std::printf("f %u ms\n", dt); CHECK(dt > 400 && dt < 600); }
  int max_us = 0;
  for (int us : lever_us) max_us = std::max(max_us, us);
  CHECK(max_us == 1889);

  // 한 발: cocked 다음 fired
  said.clear();
  const uint32_t t2 = clock_ms;
  L.line("s");
  run(L, 3500);
  CHECK(said.size() == 2 && said[0].body == "LA,cocked" && said[1].body == "LA,fired");
  if (said.size() == 2) { const uint32_t dt = said[1].t - t2; std::printf("s %u ms\n", dt); CHECK(dt > 2300 && dt < 2600); }

  // 레버 각 직접 · 범위 밖 · 모르는 명령
  said.clear();
  L.line("l90"); run(L, 500);
  CHECK(find("LA,lever,90.0") >= 0 && std::fabs(L.lever() - 90.0f) < 1e-3f);
  L.line("l181"); CHECK(said.back().body == "LA,error,bad");
  L.line("zz");   CHECK(said.back().body == "LA,error,bad");
  L.line("e5");   CHECK(said.back().body == "LA,error,bad");   // 각도 서보 명령은 없음
  L.line("a20");  CHECK(said.back().body == "LA,error,bad");
  L.line("h"); run(L, 500);
  CHECK(said.back().body == "LA,home" && std::fabs(L.lever() - kLeverRest) < 1e-3f);

  // x 는 움직이는 중에도 바로: 쌓인 것 버리고 끄고 그 뒤 응답 없음
  said.clear();
  L.line("s"); L.line("c");
  run(L, 50);
  L.line("x");
  CHECK(said.size() == 1 && said[0].body == "LA,off" && !L.busy() && !attached);
  run(L, 4000);
  CHECK(said.size() == 1);

  // 쌓을 자리(4) 넘침
  L.line("e"); said.clear();
  L.line("l30");
  for (int i = 0; i < 5; ++i) L.line("?");
  CHECK(find("LA,error,busy") >= 0);

  CHECK(other_pin_writes == 0);   // 각도 서보 핀은 한 번도 안 건드림
  std::printf(fails ? "발사대 시험 %d 실패\n" : "발사대 시험 통과\n", fails);
  return fails ? 1 : 0;
}
