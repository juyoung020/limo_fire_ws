// 위층 판단 모델 요청을 로봇 안의 llama-server 로 보내는 일꾼 (low_rl upper:=decider — scripts/low_decider.h 의 Transport).
// 라이다 콜백은 막지 않는다: send 는 줄에 넣고 돌아오고, 일꾼 스레드가 HTTP/1.1 POST (소켓, 연결마다 닫음)로 보내 답을 받아 줄에 넣고, poll 이 꺼내 간다.
// 로봇에는 libcurl 개발 헤더가 없어서 POSIX 소켓으로 직접 쓴다 (localhost · 평문 HTTP 만 — url 은 http://호스트:포트).
#pragma once
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

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
    th_ = std::thread([this] { run(); });
  }
  ~DeciderClient() override {
    {
      std::lock_guard<std::mutex> l(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    if (th_.joinable()) th_.join();
  }
  void send(const lxd::Req& r) override {
    {
      std::lock_guard<std::mutex> l(mu_);
      todo_.push_back({r.id, r.body});
    }
    cv_.notify_all();
  }
  bool poll(lxd::Answer* a) override {
    std::lock_guard<std::mutex> l(mu_);
    if (done_.empty()) return false;
    *a = done_.front();
    done_.pop_front();
    return true;
  }

 private:
  struct Job {
    long id;
    std::string body;
  };
  void run() {
    for (;;) {
      Job j;
      {
        std::unique_lock<std::mutex> l(mu_);
        cv_.wait(l, [this] { return stop_ || !todo_.empty(); });
        if (stop_) return;
        j = std::move(todo_.front());
        todo_.pop_front();
      }
      const auto t0 = std::chrono::steady_clock::now();
      std::string raw, err;
      lxd::Answer a;
      if (post(j.body, &raw, &err)) a = lxd::parse_answer(j.id, raw);
      else { a.id = j.id; a.err = err; }
      a.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      std::lock_guard<std::mutex> l(mu_);
      done_.push_back(a);
    }
  }
  // POST /v1/systemone — 성공(HTTP 200)이면 본문을 *out
  bool post(const std::string& body, std::string* out, std::string* err) const {
    const auto t_end = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms_);
    const auto left_ms = [&]() { return (int)std::max<long>(0, std::chrono::duration_cast<std::chrono::milliseconds>(t_end - std::chrono::steady_clock::now()).count()); };
    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host_.c_str(), port_.c_str(), &hints, &res) != 0 || !res) { *err = "주소를 못 찾음 " + host_; return false; }
    int fd = -1;
    for (addrinfo* p = res; p && fd < 0; p = p->ai_next) {
      fd = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
      if (fd < 0) continue;
      if (connect(fd, p->ai_addr, p->ai_addrlen) != 0) { close(fd); fd = -1; }
    }
    freeaddrinfo(res);
    if (fd < 0) { *err = "연결 안 됨 " + host_ + ":" + port_ + " (llama-server 가 떠 있나)"; return false; }
    std::string req = "POST /v1/systemone HTTP/1.1\r\nHost: " + host_ + ":" + port_ + "\r\nContent-Type: application/json\r\nContent-Length: " + std::to_string(body.size()) +
                      "\r\nConnection: close\r\n\r\n";
    req += body;
    size_t off = 0;
    while (off < req.size()) {
      pollfd pf{fd, POLLOUT, 0};
      if (::poll(&pf, 1, left_ms()) <= 0) { close(fd); *err = "시간 끝 (보내기)"; return false; }
      const ssize_t n = ::send(fd, req.data() + off, req.size() - off, MSG_NOSIGNAL);
      if (n <= 0) { close(fd); *err = "보내기 실패"; return false; }
      off += (size_t)n;
    }
    std::string resp;
    char buf[16384];
    for (;;) {
      pollfd pf{fd, POLLIN, 0};
      const int pr = ::poll(&pf, 1, left_ms());
      if (pr <= 0) { close(fd); *err = "시간 끝 (답)"; return false; }
      const ssize_t n = ::recv(fd, buf, sizeof buf, 0);
      if (n < 0) { close(fd); *err = "받기 실패"; return false; }
      if (n == 0) break;
      resp.append(buf, (size_t)n);
    }
    close(fd);
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
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<Job> todo_;
  std::deque<lxd::Answer> done_;
  bool stop_ = false;
  std::thread th_;
};

}  // namespace limo_fire_rl
