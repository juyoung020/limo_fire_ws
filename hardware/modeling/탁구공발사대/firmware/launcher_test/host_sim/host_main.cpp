// 발사기 펌웨어(src/main.cpp)를 수정 없이 리눅스에서 실행해 가상 시리얼(pty)에 연결한다.
// ROS pitch_serial_bridge 와 실제 펌웨어 명령 처리·서보 속도 제한 타이밍을 맞춰 보는 시험용이다.
// 빌드: g++ -std=c++17 -O1 -I. host_main.cpp -o launcher_host_sim -lutil
// 실행: ./launcher_host_sim <pty 경로를 적을 파일>   (PWM 명령은 표준 출력에 한 줄씩)
#include "Arduino.h"

#include <chrono>
#include <csignal>
#include <cstdlib>
#include <fcntl.h>
#include <pty.h>
#include <termios.h>
#include <thread>
#include <unistd.h>

#include "../src/main.cpp"

namespace {
using Clock = std::chrono::steady_clock;
const Clock::time_point kStart = Clock::now();
double now_s() { return std::chrono::duration<double>(Clock::now() - kStart).count(); }

class PtyStream : public Stream {
 public:
  explicit PtyStream(int fd) : fd_(fd) {}
  int available() override {
    if (has_) return 1;
    char c;
    if (::read(fd_, &c, 1) == 1) { peek_ = c; has_ = true; return 1; }
    return 0;
  }
  int read() override {
    if (!available()) return -1;
    has_ = false;
    return static_cast<unsigned char>(peek_);
  }
  void write_raw(const char* data, size_t n) override {
    while (n > 0) {
      const ssize_t w = ::write(fd_, data, n);
      if (w <= 0) return;
      data += w; n -= static_cast<size_t>(w);
    }
  }

 private:
  int fd_;
  char peek_ = 0;
  bool has_ = false;
};

class NullStream : public Stream {
 public:
  int available() override { return 0; }
  int read() override { return -1; }
  void write_raw(const char*, size_t) override {}
};

int g_master = -1;
volatile std::sig_atomic_t g_stop = 0;
NullStream g_null;
}  // namespace

Stream& serial_ref();
Stream& Serial = serial_ref();
Stream& Serial0 = g_null;
Stream& serial_ref() {
  static PtyStream* s = nullptr;
  if (!s) {
    int slave = -1;
    if (openpty(&g_master, &slave, nullptr, nullptr, nullptr) != 0) { std::perror("openpty"); std::_Exit(2); }
    termios t{};
    tcgetattr(slave, &t);
    cfmakeraw(&t);
    tcsetattr(slave, TCSANOW, &t);
    fcntl(g_master, F_SETFL, fcntl(g_master, F_GETFL) | O_NONBLOCK);
    s = new PtyStream(g_master);
    static char name[128];
    std::snprintf(name, sizeof(name), "%s", ttyname(slave));
    // slave 는 열어 둔다(브리지가 닫아도 master 가 EIO 로 끊기지 않게).
    ::setenv("LAUNCHER_HOST_SIM_PTY", name, 1);
  }
  return *s;
}

void delay(unsigned long ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }
void pinMode(int, int) {}
void digitalWrite(int pin, int value) { std::printf("%.3f gpio %d %s\n", now_s(), pin, value ? "HIGH" : "LOW"); std::fflush(stdout); }
bool ledcAttach(int pin, int freq, int res) {
  std::printf("%.3f attach %d %dHz %dbit\n", now_s(), pin, freq, res);
  std::fflush(stdout);
  return true;
}
void ledcWrite(int pin, uint32_t duty) {
  static int last_us[64] = {0};
  const int us = static_cast<int>(std::lround(duty * 20000.0 / ((1u << 14) - 1)));
  if (pin >= 0 && pin < 64 && std::abs(us - last_us[pin]) < 1) return;
  if (pin >= 0 && pin < 64) last_us[pin] = us;
  std::printf("%.3f pwm %d %dus\n", now_s(), pin, us);
  std::fflush(stdout);
}
void ledcDetach(int pin) { std::printf("%.3f detach %d\n", now_s(), pin); std::fflush(stdout); }

int main(int argc, char** argv) {
  std::signal(SIGINT, [](int) { g_stop = 1; });
  std::signal(SIGTERM, [](int) { g_stop = 1; });
  (void)Serial;
  const char* name = std::getenv("LAUNCHER_HOST_SIM_PTY");
  if (argc > 1) {
    FILE* f = std::fopen(argv[1], "w");
    if (f) { std::fprintf(f, "%s\n", name); std::fclose(f); }
  }
  std::printf("pty %s\n", name);
  std::fflush(stdout);
  // 실제 보드처럼 부팅 배너는 포트가 열리기 전에 나갈 수 있다. 브리지는 ? 로 식별해야 한다.
  setup();
  while (!g_stop) {
    loop();
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  return 0;
}
