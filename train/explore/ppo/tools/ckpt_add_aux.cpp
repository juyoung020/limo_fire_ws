// 아래층 순환 체크포인트에 보조 예측 머리를 더한다 (ppo/README "보조 예측 머리", mdp/obs_layout.h AUX_OUT) — GPU 없이 파일 → 파일.
//   ckpt_add_aux 입력.bin 출력.bin
// 하는 일: 변수 끝(log_std 뒤)에 블록 aux.w [AUX_OUT][H] · aux.b [AUX_OUT] 을 0 으로 붙임 (network/actor_critic.cu 와 같은 자리·초기값). Adam m·v 0,
//   나머지는 바이트 그대로 → 정책·가치는 옛 것과 같다 (머리는 정책 출력에 안 들어감).
// 형식은 ppo.cu save/load (형식 3, fp32), 블록 이름은 network/actor_critic.cu · gru.cu.
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../mdp/obs_layout.h"

namespace {
constexpr uint32_t kMagic = 0x4F505058, kVersion = 3;
struct Blk { std::string name; int64_t n; };
}  // namespace

int main(int argc, char** argv) try {
  if (argc != 3 || std::string(argv[1]) == argv[2]) {
    std::fprintf(stderr, "사용: ckpt_add_aux 입력.bin 출력.bin (입력과 다른 경로)\n");
    return 2;
  }
  std::ifstream f(argv[1], std::ios::binary);
  if (!f) throw std::runtime_error(std::string("체크포인트 없음: ") + argv[1]);
  auto r = [&](void* d, size_t n) {
    f.read((char*)d, (std::streamsize)n);
    if (!f) throw std::runtime_error("체크포인트가 잘림");
  };
  uint32_t hdr[3];
  r(hdr, 12);
  if (hdr[0] != kMagic || hdr[1] != kVersion || hdr[2] != 4) throw std::runtime_error("형식 3 · fp32 체크포인트가 아님");
  int32_t nb;
  r(&nb, 4);
  std::vector<Blk> blk(nb);
  int64_t P = 0;
  for (auto& b : blk) {
    int32_t ln;
    r(&ln, 4);
    b.name.assign(ln, '\0');
    r(b.name.data(), ln);
    r(&b.n, 8);
    P += b.n;
  }
  int64_t P0;
  r(&P0, 8);
  if (P0 != P) throw std::runtime_error("변수 수가 층 크기 합과 다름");
  std::vector<float> p(P), m(P), v(P);
  r(p.data(), P * 4);
  r(m.data(), P * 4);
  r(v.data(), P * 4);
  int64_t adam_t;
  r(&adam_t, 8);
  int32_t D;
  r(&D, 4);
  std::vector<float> mean(D), var(D);
  r(mean.data(), D * 4);
  r(var.data(), D * 4);
  char tail[8 + 4 + 4 + 8 + 8];
  r(tail, sizeof tail);
  int64_t H = 0;
  for (auto& b : blk) {
    if (b.name == "actor.gru.wh") H = (int64_t)(std::sqrt((double)b.n / 3.0) + 0.5);
    if (b.name.rfind("aux.", 0) == 0) throw std::runtime_error("이미 보조 머리가 있음");
  }
  if (H <= 0) throw std::runtime_error("순환 핵(actor.gru) 이 없는 체크포인트");
  if (blk.empty() || blk.back().name != "log_std") throw std::runtime_error("마지막 블록이 log_std 가 아님");
  const int64_t K = mdp::low::AUX_OUT, add = K * H + K;
  blk.push_back({"aux.w", K * H});
  blk.push_back({"aux.b", K});
  for (auto* q : {&p, &m, &v}) q->insert(q->end(), add, 0.f);
  const int64_t P2 = P + add;
  const std::string tmp = std::string(argv[2]) + ".tmp";
  {
    std::ofstream g(tmp, std::ios::binary);
    auto w = [&](const void* d, size_t n) { g.write((const char*)d, (std::streamsize)n); };
    w(hdr, 12);
    const int32_t nb2 = (int32_t)blk.size();
    w(&nb2, 4);
    for (auto& b : blk) {
      int32_t ln = (int32_t)b.name.size();
      w(&ln, 4);
      w(b.name.data(), ln);
      w(&b.n, 8);
    }
    w(&P2, 8);
    w(p.data(), P2 * 4);
    w(m.data(), P2 * 4);
    w(v.data(), P2 * 4);
    w(&adam_t, 8);
    w(&D, 4);
    w(mean.data(), D * 4);
    w(var.data(), D * 4);
    w(tail, sizeof tail);
    if (!g) throw std::runtime_error("쓰기 실패");
  }
  if (std::rename(tmp.c_str(), argv[2]) != 0) throw std::runtime_error("이름 바꾸기 실패");
  std::printf("%s: 보조 머리 aux.w [%lld][%lld] · aux.b 더함 (0), 변수 %lld → %lld\n", argv[2], (long long)K, (long long)H, (long long)P, (long long)P2);
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "ckpt_add_aux: %s\n", e.what());
  return 1;
}
