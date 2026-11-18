// 가짜 ESP32 — 가상 시리얼(pty)에 펌웨어와 같은 줄을 흘린다. 실물 ESP32·HC-SR04 없이 다리(esp32_bridge)를 끝까지 시험:
//   fake_esp32 --link /tmp/fake_esp32 --scenario sweep [--duration 0]    (0 = 계속)
//   ros2 run limo_fire_esp32_bridge esp32_bridge --ros-args --params-file <share>/config/esp32_bridge.yaml -p port:=/tmp/fake_esp32
// 장면은 include/limo_fire_esp32_bridge/fake_stream.hpp 에. 다리를 먼저 켜도 된다(포트가 생길 때까지 1 초마다 다시 엶).
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdlib.h>
#include <termios.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>

#include "limo_fire_esp32_bridge/fake_stream.hpp"

using namespace limo_fire_esp32_bridge;

namespace {
volatile sig_atomic_t g_stop = 0;
void on_signal(int) { g_stop = 1; }
}  // namespace

int main(int argc, char** argv) {
  std::string link = "/tmp/fake_esp32", sc_name = "sweep";
  double duration = 0.0;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string k = argv[i];
    if (k == "--link") link = argv[i + 1];
    else if (k == "--scenario") sc_name = argv[i + 1];
    else if (k == "--duration") duration = std::atof(argv[i + 1]);
    else { std::fprintf(stderr, "모르는 인자 %s\n사용: fake_esp32 [--link 경로] [--scenario sweep|noecho|corrupt|stall|sensors] [--duration 초]\n", k.c_str()); return 2; }
  }
  Scenario sc;
  if (!parse_scenario(sc_name, sc)) { std::fprintf(stderr, "모르는 장면 %s (sweep|noecho|corrupt|stall|sensors)\n", sc_name.c_str()); return 2; }

  const int master = posix_openpt(O_RDWR | O_NOCTTY);
  if (master < 0 || grantpt(master) != 0 || unlockpt(master) != 0) { std::perror("pty"); return 1; }
  const char* slave = ptsname(master);
  termios t{};
  const int sfd = ::open(slave, O_RDWR | O_NOCTTY);   // 슬레이브를 열어 둬야 읽는 쪽이 없을 때도 쓰기가 막히지 않는다
  if (sfd >= 0 && tcgetattr(sfd, &t) == 0) { cfmakeraw(&t); tcsetattr(sfd, TCSANOW, &t); }
  ::unlink(link.c_str());
  if (::symlink(slave, link.c_str()) != 0) { std::perror("symlink"); return 1; }
  std::printf("가짜 ESP32: %s -> %s, 장면 %s\n", link.c_str(), slave, sc_name.c_str());
  std::fflush(stdout);
  signal(SIGINT, on_signal);
  signal(SIGTERM, on_signal);

  const FakeParams p;
  const auto t0 = std::chrono::steady_clock::now();
  for (int64_t step = 0; !g_stop; ++step) {
    const auto due = t0 + std::chrono::microseconds(static_cast<int64_t>(step * p.slot_s * 1e6));
    std::this_thread::sleep_until(due);
    if (duration > 0.0 && std::chrono::duration<double>(due - t0).count() > duration) break;
    for (const auto& l : lines_at(sc, step, p)) {
      pollfd pf{master, POLLOUT, 0};
      if (poll(&pf, 1, 0) > 0 && (pf.revents & POLLOUT)) { const ssize_t w = ::write(master, l.data(), l.size()); (void)w; }
    }
  }
  ::unlink(link.c_str());
  return 0;
}
