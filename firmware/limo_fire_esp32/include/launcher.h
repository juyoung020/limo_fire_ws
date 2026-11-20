// 탁구공 발사대 레버 서보(GPIO 4) — 센서를 읽는 loop() 를 막지 않게 한 걸음(20 ms)씩 움직인다.
// 각도 서보는 쓰지 않는다(발사대 각은 기구로 15° 고정, GPIO 5 는 늘 LOW — config.h).
//
// 명령 (호스트 → ESP32, 글자 한 줄. 사람이 시리얼 모니터에서 쳐도 됨)
//   e  레버 PWM 켜기(쉼 자리로)   x  끄기(핀 LOW, 움직이던 것도 멈추고 쌓인 명령 버림 — 늘 바로 들음)
//   c  장전   f  발사   s  장전 + 발사   h  쉼 자리로   l<0..180>  레버 서보 각(각 찾을 때)   ?  상태
// 응답 (ESP32 → 호스트, 센서 줄과 같은 $ 틀 + 체크섬 — protocol.h. 여기서는 몸통만 만든다)
//   LA,boot  LA,on  LA,off  LA,cocked  LA,fired  LA,home  LA,lever,<각>  LA,state,<각>,<on|off>,<busy 0|1>
//   LA,blocked,off (꺼진 채 움직임 명령)  LA,error,busy (쌓을 자리 넘침)  LA,error,bad (모르는 명령·범위 밖)
//   s 는 LA,cocked 다음 LA,fired. 움직이는 명령은 끝났을 때 응답한다(움직임 자체는 재지 않음).
// 움직이는 동안 들어온 명령은 4 개까지 쌓았다가 차례로 실행한다.
// 하드웨어는 Hw 로만 만진다 — 호스트 시험(test/test_launcher.cpp)이 가짜 Hw 로 같은 코드를 돌린다.
#pragma once
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "config.h"

namespace launcher {

struct Hw {
  bool (*attach)(int pin);              // PWM 켜기 (50 Hz)
  void (*detach_low)(int pin);          // PWM 끄고 핀 LOW
  void (*write_us)(int pin, int us);    // 펄스 폭
  void (*say)(const char* body);        // 응답 몸통 하나 ("LA,..." — 틀·체크섬은 Hw 가 씌움)
};

constexpr int kUsMin = 500, kUsMax = 2500;
constexpr uint32_t kStepMs = 20;        // 서보 한 걸음 = PWM 한 주기

class Launcher {
 public:
  explicit Launcher(const Hw& hw) : hw_(hw) {}

  bool busy() const { return n_ > 0; }
  float lever() const { return lever_; }
  bool on() const { return on_; }

  void boot() { hw_.say("LA,boot"); }

  // 한 줄 받기. 움직이는 중이면 쌓아 둔다(x 는 바로)
  void line(const char* raw) {
    char cmd[kLineMax];
    trim(raw, cmd);
    if (!cmd[0]) return;
    if (strcmp(cmd, "x") == 0) { n_ = head_ = 0; nq_ = 0; run(cmd); return; }
    if (busy() || nq_) {
      if (nq_ == kQueueMax) { hw_.say("LA,error,busy"); return; }
      memcpy(queue_[(qh_ + nq_) % kQueueMax], cmd, kLineMax);
      ++nq_;
      return;
    }
    run(cmd);
  }

  // loop() 마다
  void update(uint32_t now) {
    while (true) {
      if (!busy()) {
        if (!nq_) return;
        char cmd[kLineMax];
        memcpy(cmd, queue_[qh_], kLineMax);
        qh_ = (qh_ + 1) % kQueueMax;
        --nq_;
        run(cmd);
        continue;
      }
      Step& s = steps_[head_];
      if (s.kind == Step::kSay) { hw_.say(s.text); pop(); continue; }
      if (s.kind == Step::kWait) {
        if (!s.started) { s.started = true; s.t0 = now; }
        if (static_cast<int32_t>(now - s.t0) < static_cast<int32_t>(s.ms)) return;
        pop(); continue;
      }
      // kMove: 20 ms 마다 deg_per_s × 0.02 씩
      if (!s.started) { s.started = true; s.t0 = now - kStepMs; }
      if (static_cast<int32_t>(now - s.t0) < static_cast<int32_t>(kStepMs)) return;
      s.t0 += kStepMs;
      if (static_cast<int32_t>(now - s.t0) > static_cast<int32_t>(kStepMs)) s.t0 = now;   // 늦었으면 몰아 걷지 않음(속도 제한 지킴)
      const float step = s.deg_per_s * kStepMs * 0.001f, d = s.target - lever_;
      lever_ += fabsf(d) <= step ? d : (d > 0 ? step : -step);
      write_deg(lever_);
      if (fabsf(s.target - lever_) <= 0.01f) { lever_ = s.target; pop(); continue; }
      return;
    }
  }

 private:
  static constexpr int kLineMax = 16, kQueueMax = 4, kStepsMax = 12;
  struct Step {
    enum Kind : uint8_t { kMove, kWait, kSay } kind;
    float target, deg_per_s;
    uint32_t ms;
    const char* text;
    bool started;
    uint32_t t0;
  };

  static void trim(const char* in, char* out) {
    while (*in == ' ' || *in == '\t') ++in;
    size_t n = strnlen(in, kLineMax - 1);
    while (n && (in[n - 1] == ' ' || in[n - 1] == '\t' || in[n - 1] == '\r')) --n;
    memcpy(out, in, n);
    out[n] = '\0';
  }
  static bool number(const char* s, float& v) {
    if (!s[0]) return false;
    char* end = nullptr;
    v = strtof(s, &end);
    return end != s && *end == '\0' && isfinite(v);
  }

  void write_deg(float deg) {
    deg = deg < 0 ? 0 : (deg > kLeverTravelDeg ? kLeverTravelDeg : deg);
    hw_.write_us(kLeverPin, kUsMin + static_cast<int>((kUsMax - kUsMin) * deg / kLeverTravelDeg + 0.5f));
  }

  void push(Step s) {
    if (head_ + n_ >= kStepsMax) return;   // 명령 하나가 넣는 걸음은 9 개 이하(s) — 넘칠 일 없음(시험이 확인)
    s.started = false;
    steps_[head_ + n_++] = s;
  }
  void pop() {
    ++head_;
    if (--n_ == 0) head_ = 0;
  }
  void move(float target, float deg_per_s) { push({Step::kMove, target, deg_per_s, 0, nullptr, false, 0}); }
  void wait(uint32_t ms) { push({Step::kWait, 0, 0, ms, nullptr, false, 0}); }
  void say(const char* t) { push({Step::kSay, 0, 0, 0, t, false, 0}); }

  // 장전: 장전 각까지 빠르게 → 끝에서 버팀(걸쇠가 내려앉을 시간) → 쉼 자리로 천천히 → 공 자리 잡기
  void cock_steps() {
    move(kLeverCock, kLeverCockDegPerS);
    wait(kCockHoldMs);
    move(kLeverRest, kLeverReturnDegPerS);
    wait(kCockSettleMs);
    say("LA,cocked");
  }
  // 발사: 손가락이 걸쇠 꼬리를 눌러 풀고 쉼 자리로
  void fire_steps() {
    move(kLeverFire, kLeverCockDegPerS);
    wait(kFireHoldMs);
    move(kLeverRest, kLeverReturnDegPerS);
    say("LA,fired");
  }

  void run(const char* cmd) {
    float v = 0;
    switch (cmd[0]) {
      case 'e':
        if (cmd[1]) { hw_.say("LA,error,bad"); break; }
        if (!on_) {
          if (!hw_.attach(kLeverPin)) { hw_.say("LA,error,attach"); break; }
          on_ = true;
          write_deg(lever_);   // 켤 때 기록된 자리(처음엔 쉼)로
        }
        hw_.say("LA,on");
        break;
      case 'x':
        if (on_) hw_.detach_low(kLeverPin);
        on_ = false;
        hw_.say("LA,off");
        break;
      case 'c': case 'f': case 's': case 'h':
        if (cmd[1]) { hw_.say("LA,error,bad"); break; }
        if (!on_) { hw_.say("LA,blocked,off"); break; }
        if (cmd[0] == 'c' || cmd[0] == 's') cock_steps();
        if (cmd[0] == 'f' || cmd[0] == 's') fire_steps();
        if (cmd[0] == 'h') { move(kLeverRest, kLeverReturnDegPerS); say("LA,home"); }
        break;
      case 'l':
        if (!number(cmd + 1, v) || v < 0 || v > kLeverTravelDeg) { hw_.say("LA,error,bad"); break; }
        if (!on_) { hw_.say("LA,blocked,off"); break; }
        move(v, kLeverManualDegPerS);
        snprintf(lever_reply_, sizeof lever_reply_, "LA,lever,%.1f", v);
        say(lever_reply_);
        break;
      case '?': {
        char out[48];
        snprintf(out, sizeof out, "LA,state,%.1f,%s,%d", lever_, on_ ? "on" : "off", busy() ? 1 : 0);
        hw_.say(out);
        break;
      }
      default: hw_.say("LA,error,bad"); break;
    }
  }

  const Hw hw_;
  Step steps_[kStepsMax];
  int head_ = 0, n_ = 0;
  char queue_[kQueueMax][kLineMax];
  int qh_ = 0, nq_ = 0;
  char lever_reply_[24];   // say() 가 나중에 읽으므로 명령 동안 살아 있어야 함
  float lever_ = kLeverRest;
  bool on_ = false;
};

}  // namespace launcher
