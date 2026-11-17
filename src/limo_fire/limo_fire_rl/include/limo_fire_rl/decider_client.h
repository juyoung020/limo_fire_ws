// 위층 판단 모델 요청을 로봇 안의 llama-server 로 보내는 것 (low_rl upper:=decider — scripts/low_decider.h 의 Transport).
// 스레드 없음: send 는 줄에 넣고 비차단 연결만 시작하고, poll (노드가 걸음마다 부름)이 비차단 소켓을 한 단계씩 진행한다
// (연결 → HTTP/1.1 POST 보내기 → 답 받기 → 해석, 연결마다 닫음). 라이다 콜백은 막지 않고, 노드는 한 스레드로 돈다.
// 로봇에는 libcurl 개발 헤더가 없어서 POSIX 소켓으로 직접 쓴다 (localhost · 평문 HTTP 만 — url 은 http://호스트:포트).
#pragma once
#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <deque>
#include <stdexcept>
#include <string>
#include <vector>

#include "low_decider.h"

namespace limo_fire_rl {

class DeciderClient : public lxd::Transport {
 public:
  // url = http://호스트:포트 (경로는 /v1/systemone 을 붙임), timeout_ms = 연결부터 답 끝까지 한도
  DeciderClient(const std::string& url, int timeout_ms) : timeout_ms_(timeout_ms) {
    const std::string pre = "http://";
    if (url.compare(0, pre.size(), pre) != 0) throw std::runtime_error("decider_url 은 http://호스트:포트 꼴: " + url);
    std::string hp = url.substr(pre.size());
    const size_t sl = hp.find('/');
    if (sl != std::string::npos) hp = hp.substr(0, sl);
    const size_t co = hp.rfind(':');
    host_ = co == std::string::npos ? hp : hp.substr(0, co);
    port_ = co == std::string::npos ? "80" : hp.substr(co + 1);
    addrinfo hints{}, *res = nullptr;   // 주소는 한 번만 풂 (요청마다 막히지 않게)
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host_.c_str(), port_.c_str(), &hints, &res) == 0 && res) {
      for (addrinfo* p = res; p; p = p->ai_next) {
        Addr a;
        a.family = p->ai_family; a.socktype = p->ai_socktype; a.proto = p->ai_protocol; a.len = p->ai_addrlen;
        std::memcpy(&a.sa, p->ai_addr, p->ai_addrlen);
        addrs_.push_back(a);
      }
      freeaddrinfo(res);
    }
  }
  ~DeciderClient() override { if (fd_ >= 0) close(fd_); }
  void send(const lxd::Req& r) override {
    keep_png(r.body);
    todo_.push_back({r.id, r.body});
    step();
  }
  // 마지막으로 보낸 요청의 첫 그림(지도 PNG 바이트)과 그 번호 — 노드가 RViz 로 보여 줌 (~/decider_image)
  const std::vector<uint8_t>& last_png() const { return last_png_; }
  long n_png() const { return n_png_; }
  bool poll(lxd::Answer* a) override {
    step();
    if (done_.empty()) return false;
    *a = done_.front();
    done_.pop_front();
    return true;
  }

 private:
  // 요청 본문의 첫 "data:image/png;base64,…" 를 풀어 둠 (보는 용도만 — 요청에는 손대지 않음)
  void keep_png(const std::string& body) {
    static const std::string tag = "data:image/png;base64,";
    const size_t b = body.find(tag);
    if (b == std::string::npos) return;
    const size_t s = b + tag.size(), e = body.find('"', s);
    if (e == std::string::npos) return;
    auto val = [](char c) -> int {
      if (c >= 'A' && c <= 'Z') return c - 'A';
      if (c >= 'a' && c <= 'z') return c - 'a' + 26;
      if (c >= '0' && c <= '9') return c - '0' + 52;
      if (c == '+') return 62;
      if (c == '/') return 63;
      return -1;
    };
    std::vector<uint8_t> out;
    out.reserve((e - s) * 3 / 4);
    unsigned acc = 0;
    int bits = 0;
    for (size_t i = s; i < e; ++i) {
      const int v = val(body[i]);
      if (v < 0) continue;   // '=' 채움 · 이스케이프 글자 건너뜀
      acc = (acc << 6) | (unsigned)v;
      bits += 6;
      if (bits >= 8) {
        bits -= 8;
        out.push_back((uint8_t)((acc >> bits) & 0xFF));
      }
    }
    last_png_.swap(out);
    ++n_png_;
  }
  std::vector<uint8_t> last_png_;
  long n_png_ = 0;
  struct Job {
    long id;
    std::string body;
  };
  struct Addr {
    int family, socktype, proto;
    socklen_t len;
    sockaddr_storage sa;
  };
  enum State { kIdle, kConnect, kSend, kRecv };
  using Clock = std::chrono::steady_clock;

  bool late() const { return Clock::now() >= t_end_; }
  // 다음 주소로 비차단 연결 시작 (없으면 거짓)
  bool open_next() {
    for (; ai_ < addrs_.size(); ++ai_) {
      const Addr& a = addrs_[ai_];
      fd_ = socket(a.family, a.socktype | SOCK_NONBLOCK | SOCK_CLOEXEC, a.proto);
      if (fd_ < 0) continue;
      if (connect(fd_, (const sockaddr*)&a.sa, a.len) == 0 || errno == EINPROGRESS) { st_ = kConnect; return true; }
      close(fd_);
      fd_ = -1;
    }
    return false;
  }
  void finish(lxd::Answer a) {
    if (fd_ >= 0) { close(fd_); fd_ = -1; }
    a.ms = std::chrono::duration<double, std::milli>(Clock::now() - t0_).count();
    done_.push_back(std::move(a));
    st_ = kIdle;
  }
  void fail(const std::string& err) {
    lxd::Answer a;
    a.id = cur_.id;
    a.err = err;
    finish(std::move(a));
  }
  // 막지 않고 할 수 있는 만큼 진행 (send · poll 이 부름)
  void step() {
    for (;;) {
      if (st_ == kIdle) {
        if (todo_.empty()) return;
        cur_ = std::move(todo_.front());
        todo_.pop_front();
        t0_ = Clock::now();
        t_end_ = t0_ + std::chrono::milliseconds(timeout_ms_);
        ai_ = 0;
        if (addrs_.empty()) { fail("주소를 못 찾음 " + host_); continue; }
        if (!open_next()) { fail("연결 안 됨 " + host_ + ":" + port_ + " (llama-server 가 떠 있나)"); continue; }
      }
      if (st_ == kConnect) {
        pollfd pf{fd_, POLLOUT, 0};
        const int pr = ::poll(&pf, 1, 0);
        if (pr == 0) { if (late()) fail("시간 끝 (보내기)"); return; }
        int e = 0;
        socklen_t el = sizeof e;
        if (pr < 0 || getsockopt(fd_, SOL_SOCKET, SO_ERROR, &e, &el) != 0 || e != 0) {   // 이 주소는 안 됨 → 다음 주소
          close(fd_);
          fd_ = -1;
          ++ai_;
          if (!open_next()) fail("연결 안 됨 " + host_ + ":" + port_ + " (llama-server 가 떠 있나)");
          continue;
        }
        req_ = "POST /v1/systemone HTTP/1.1\r\nHost: " + host_ + ":" + port_ + "\r\nContent-Type: application/json\r\nContent-Length: " + std::to_string(cur_.body.size()) +
               "\r\nConnection: close\r\n\r\n";
        req_ += cur_.body;
        off_ = 0;
        st_ = kSend;
      }
      if (st_ == kSend) {
        while (off_ < req_.size()) {
          const ssize_t n = ::send(fd_, req_.data() + off_, req_.size() - off_, MSG_NOSIGNAL | MSG_DONTWAIT);
          if (n > 0) { off_ += (size_t)n; continue; }
          if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) { if (late()) fail("시간 끝 (보내기)"); return; }
          fail("보내기 실패");
          break;
        }
        if (st_ != kSend) continue;
        resp_.clear();
        st_ = kRecv;
      }
      if (st_ == kRecv) {
        char buf[16384];
        for (;;) {
          const ssize_t n = ::recv(fd_, buf, sizeof buf, MSG_DONTWAIT);
          if (n > 0) { resp_.append(buf, (size_t)n); continue; }
          if (n == 0) {   // 서버가 닫음 = 답 끝
            std::string raw, err;
            lxd::Answer a;
            if (parse_http(resp_, &raw, &err)) a = lxd::parse_answer(cur_.id, raw);
            else { a.id = cur_.id; a.err = err; }
            finish(std::move(a));
            break;
          }
          if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) { if (late()) fail("시간 끝 (답)"); return; }
          fail("받기 실패");
          break;
        }
      }
    }
  }
  // HTTP 답 → 본문 (HTTP 200 일 때만 *out)
  static bool parse_http(const std::string& resp, std::string* out, std::string* err) {
    const size_t he = resp.find("\r\n\r\n");
    if (he == std::string::npos || resp.compare(0, 5, "HTTP/") != 0) { *err = "HTTP 꼴 아님"; return false; }
    const size_t sp = resp.find(' ');
    const int code = sp != std::string::npos ? std::atoi(resp.c_str() + sp + 1) : 0;
    std::string head = resp.substr(0, he), b = resp.substr(he + 4);
    for (auto& ch : head) ch = (char)std::tolower((unsigned char)ch);
    if (head.find("transfer-encoding: chunked") != std::string::npos) {   // 조각 나눔 풀기
      std::string d;
      size_t p = 0;
      for (;;) {
        const size_t e = b.find("\r\n", p);
        if (e == std::string::npos) break;
        const long len = std::strtol(b.c_str() + p, nullptr, 16);
        if (len <= 0) break;
        d.append(b, e + 2, (size_t)len);
        p = e + 2 + (size_t)len + 2;
      }
      b.swap(d);
    }
    if (code != 200) { *err = "http " + std::to_string(code); return false; }
    *out = std::move(b);
    return true;
  }

  std::string host_, port_;
  int timeout_ms_;
  std::vector<Addr> addrs_;
  size_t ai_ = 0;
  std::deque<Job> todo_;
  std::deque<lxd::Answer> done_;
  State st_ = kIdle;
  Job cur_{};
  int fd_ = -1;
  std::string req_, resp_;
  size_t off_ = 0;
  Clock::time_point t0_, t_end_;
};

}  // namespace limo_fire_rl
