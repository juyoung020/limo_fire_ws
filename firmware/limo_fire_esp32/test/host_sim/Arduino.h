// 호스트(리눅스) 시험용 최소 Arduino 대체 헤더 — 펌웨어 src/main.cpp(센서 + 발사대)를 고치지 않고 PC 에서 돌리기 위함.
// 센서 UART 는 비어 있고(열화상·레이더 줄 없음) 초음파는 늘 메아리 없음(0) — 발사대 명령·응답·PWM 과 줄 섞임만 본다.
// 실물 보드·서보 동작을 증명하지 않는다. 시리얼 프로토콜·타이밍·PWM 명령값만 확인한다.
#pragma once
#include <cctype>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#define ESP_ARDUINO_VERSION_MAJOR 3
#define OUTPUT 1
#define INPUT 0
#define SERIAL_8N1 0
#define LOW 0
#define HIGH 1
#define constrain(x, lo, hi) ((x) < (lo) ? (lo) : ((x) > (hi) ? (hi) : (x)))

inline bool isDigit(char c) { return std::isdigit(static_cast<unsigned char>(c)) != 0; }
using std::isfinite;

class String {
 public:
  String() = default;
  String(const char* s) : s_(s ? s : "") {}
  String(const std::string& s) : s_(s) {}
  void trim() {
    const auto b = s_.find_first_not_of(" \t\r\n");
    const auto e = s_.find_last_not_of(" \t\r\n");
    s_ = b == std::string::npos ? "" : s_.substr(b, e - b + 1);
  }
  bool isEmpty() const { return s_.empty(); }
  unsigned length() const { return static_cast<unsigned>(s_.size()); }
  const char* c_str() const { return s_.c_str(); }
  char operator[](unsigned i) const { return i < s_.size() ? s_[i] : '\0'; }
  bool operator==(const char* o) const { return s_ == o; }
  String& operator+=(char c) { s_ += c; return *this; }
  String& operator=(const char* o) { s_ = o; return *this; }
  friend String operator+(const char* a, const String& b) { return String(std::string(a) + b.s_); }

 private:
  std::string s_;
};

class Stream {
 public:
  virtual ~Stream() = default;
  virtual int available() = 0;
  virtual int read() = 0;
  virtual void write_raw(const char* data, size_t n) = 0;
  void begin(unsigned long) {}
  size_t write(const uint8_t* d, size_t n) { write_raw(reinterpret_cast<const char*>(d), n); return n; }
  void print(const char* s) { write_raw(s, std::strlen(s)); }
  void println(const char* s) { print(s); write_raw("\r\n", 2); }
  void println(const String& s) { println(s.c_str()); }
  int printf(const char* fmt, ...) __attribute__((format(printf, 2, 3))) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    const int n = std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    write_raw(buf, std::strlen(buf));
    return n;
  }
};

// 센서 UART(열화상·레이더): 아무것도 안 들어온다
class HardwareSerial : public Stream {
 public:
  int available() override { return 0; }
  int read() override { return -1; }
  void write_raw(const char*, size_t) override {}
  void setRxBufferSize(size_t) {}
  void begin(unsigned long, int, int, int) {}
};

// host_main.cpp 가 정의한다.
extern Stream& Serial;
extern HardwareSerial Serial1, Serial2;
uint32_t millis();
void delay(unsigned long ms);
inline void delayMicroseconds(unsigned) {}
inline unsigned long pulseIn(int, int, unsigned long) { return 0; }
void pinMode(int pin, int mode);
void digitalWrite(int pin, int value);
bool ledcAttach(int pin, int freq, int resolution);
void ledcWrite(int pin, uint32_t duty);
void ledcDetach(int pin);
