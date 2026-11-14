// 탁구공 발사대 시험 펌웨어 — ESP32-S3 + 서보 2개
//
// 배선
//   레버 서보 DS3218 PRO  신호 → GPIO 4      (장전·발사: 한쪽으로 돌면 장전, 반대쪽으로 돌면 걸쇠를 밀어 발사)
//   각도 서보 MG996R      신호 → GPIO 5      (발사대 기울기 5~30°)
//   두 서보 전원(빨강·갈색)은 6 V 전원(XL4015, 5 A 이상)에서. ESP32 5V 핀에서 끌어 쓰지 말 것.
//   6 V 전원의 - 와 ESP32 GND 를 꼭 같이 묶는다(신호 기준).
//
// 시리얼 명령 (115200, 줄 끝 Enter)
//   s        한 발: 장전 → 레버 복귀 → 발사 → 레버 복귀
//   c        장전만(레버를 장전 각으로 돌린 뒤 쉼 위치로)
//   f        발사만(레버를 발사 각으로 돌려 걸쇠를 민 뒤 쉼 위치로)
//   a<각도>  발사대 각도(5~30), 예: a15
//   h        쉼 자세(레버 쉼, 각도 15°)
//   l<각도>  레버 서보 각을 직접(0~180) — 혼 끼운 뒤 각도 맞출 때
//   t<각도>  각도 서보 각을 직접(0~180)
//   ?        지금 값 보기
//
// 서보 각(아래 상수)은 실물에서 맞춘다: 혼을 끼운 뒤 l/t 명령으로 쉼·장전·발사 자리를 찾아 고친다.
#include <Arduino.h>

// ---------- 핀 ----------
constexpr int kLeverPin = 4;
constexpr int kTiltPin = 5;

// ---------- 서보 펄스 ----------
// 0° = 500 us, 180° = 2500 us (DS3218 PRO 180°, MG996R 보통 범위). 끝에서 서보가 웅웅거리면 범위를 좁힌다.
constexpr int kUsMin = 500, kUsMax = 2500;

// ---------- 레버 각(서보 각, 도) — 설계값, 실물에서 맞출 것 ----------
// 쉼 = 레버가 핀 앞에서 비켜 있는 자리. 장전 = 쉼에서 66° 돌린 자리(핀이 걸쇠 뒤로 넘어감).
// 발사 = 쉼에서 반대쪽으로 15° 돌린 자리(레버 손가락이 걸쇠 꼬리를 누름).
float kLeverRest  = 90.0f;
float kLeverCock  = 90.0f + 66.0f + 3.0f;   // 3° 여유(혼 톱니 오차)
float kLeverFire  = 90.0f - 15.0f - 3.0f;
// ---------- 각도 서보: 발사대 각(도) → 서보 각 ----------
float kTiltServoAt15 = 90.0f;     // 발사대 15° 일 때 서보 각(실물에서 맞춤)
float kTiltSign = 1.0f;           // 서보 각이 늘 때 총구가 들리면 +1, 내려가면 -1

// ---------- 속도 제한(시뮬에서 정한 순서 지키기) ----------
constexpr float kLeverCockDegPerS   = 300.0f;   // 장전 쪽은 서보 최대 속도에 가깝게
constexpr float kLeverReturnDegPerS = 150.0f;   // 레버 복귀는 200°/s 이하(빠르면 공이 차여 굴러 나감)
constexpr float kTiltDegPerS        = 30.0f;    // 각도 바꾸기는 40°/s 이하
constexpr float kCockMinPitch       = 15.0f;    // 장전은 15° 이상에서(낮으면 공 두 개가 같이 나감)

float leverNow = kLeverRest, tiltServoNow = kTiltServoAt15, pitchNow = 15.0f;

// ---------- PWM ----------
constexpr int kFreq = 50, kRes = 14;            // 20 ms 주기, 14비트
#if ESP_ARDUINO_VERSION_MAJOR >= 3
void pwmAttach(int pin, int ch) { (void)ch; ledcAttach(pin, kFreq, kRes); }
void pwmWrite(int pin, int ch, uint32_t duty) { (void)ch; ledcWrite(pin, duty); }
#else
void pwmAttach(int pin, int ch) { ledcSetup(ch, kFreq, kRes); ledcAttachPin(pin, ch); }
void pwmWrite(int pin, int ch, uint32_t duty) { (void)pin; ledcWrite(ch, duty); }
#endif
uint32_t usToDuty(int us) { return (uint32_t)((uint64_t)us * ((1u << kRes) - 1) / 20000u); }
void writeDeg(int pin, int ch, float deg) {
  deg = constrain(deg, 0.0f, 180.0f);
  int us = kUsMin + (int)((kUsMax - kUsMin) * deg / 180.0f + 0.5f);
  pwmWrite(pin, ch, usToDuty(us));
}

// 목표까지 속도 제한으로 천천히(20 ms 마다 한 걸음)
void moveServo(int pin, int ch, float& now, float target, float degPerS) {
  const float step = degPerS * 0.02f;
  while (fabsf(target - now) > 0.01f) {
    float d = target - now;
    now += (fabsf(d) < step) ? d : (d > 0 ? step : -step);
    writeDeg(pin, ch, now);
    delay(20);
  }
}
void lever(float target, float speed) { moveServo(kLeverPin, 0, leverNow, target, speed); }
float pitchToServo(float p) { return kTiltServoAt15 + kTiltSign * (p - 15.0f); }
void pitch(float p) {
  p = constrain(p, 5.0f, 30.0f);
  moveServo(kTiltPin, 1, tiltServoNow, pitchToServo(p), kTiltDegPerS);
  pitchNow = p;
}

// ---------- 동작 ----------
void cock() {
  float target = pitchNow;
  if (pitchNow < kCockMinPitch) pitch(kCockMinPitch);       // 장전은 15° 이상에서
  lever(kLeverCock, kLeverCockDegPerS);
  delay(250);                                               // 걸쇠가 내려앉을 시간
  lever(kLeverRest, kLeverReturnDegPerS);                   // 천천히 복귀(핀은 걸쇠에 걸려 있음)
  delay(300);                                               // 공이 플런저 앞에 자리 잡을 시간
  if (target != pitchNow) pitch(target);                    // 원래 각도로 천천히
}
void fire() {
  lever(kLeverFire, kLeverCockDegPerS);                     // 손가락이 걸쇠 꼬리를 눌러 발사
  delay(200);
  lever(kLeverRest, kLeverReturnDegPerS);
}

void printState(Stream& s) {
  s.printf("lever %.1f (rest %.1f cock %.1f fire %.1f) | pitch %.1f deg, tilt servo %.1f\n",
           leverNow, kLeverRest, kLeverCock, kLeverFire, pitchNow, tiltServoNow);
}

void handle(String cmd, Stream& out) {
  cmd.trim();
  if (cmd.isEmpty()) return;
  char k = cmd[0];
  float v = cmd.length() > 1 ? cmd.substring(1).toFloat() : 0;
  switch (k) {
    case 's': out.println("shot: cock"); cock(); out.println("shot: fire"); fire(); out.println("done"); break;
    case 'c': cock(); out.println("cocked"); break;
    case 'f': fire(); out.println("fired"); break;
    case 'a': pitch(v); out.printf("pitch %.1f\n", pitchNow); break;
    case 'h': lever(kLeverRest, kLeverReturnDegPerS); pitch(15); out.println("home"); break;
    case 'l': moveServo(kLeverPin, 0, leverNow, v, 60); out.printf("lever servo %.1f\n", leverNow); break;
    case 't': moveServo(kTiltPin, 1, tiltServoNow, v, kTiltDegPerS); out.printf("tilt servo %.1f\n", tiltServoNow); break;
    case '?': printState(out); break;
    default: out.println("명령: s c f a<5~30> h l<0~180> t<0~180> ?"); break;
  }
}

String buf0, buf1;
void poll(Stream& s, String& buf) {
  while (s.available()) {
    char ch = (char)s.read();
    if (ch == '\n' || ch == '\r') { if (buf.length()) { handle(buf, s); buf = ""; } }
    else buf += ch;
  }
}

void setup() {
  Serial.begin(115200);    // 네이티브 USB(USB 포트)
  Serial0.begin(115200);   // USB-UART(COM 포트)
  pwmAttach(kLeverPin, 0);
  pwmAttach(kTiltPin, 1);
  writeDeg(kLeverPin, 0, leverNow);
  writeDeg(kTiltPin, 1, tiltServoNow);
  delay(500);
  Serial.println("launcher_test ready — ? 로 상태, s 로 한 발");
  Serial0.println("launcher_test ready — ? 로 상태, s 로 한 발");
}

void loop() {
  poll(Serial, buf0);
  poll(Serial0, buf1);
}
