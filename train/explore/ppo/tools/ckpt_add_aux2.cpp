// (ckpt_add_aux2) 아래층 체크포인트의 선형 보조 머리(aux.w · aux.b)를 두 층(aux.w1 [hid][H] · aux.b1 · aux.w2 [AUX_OUT][hid] · aux.b2)으로 바꾼다.
//   ckpt_add_aux2 입력.bin 출력.bin [hid=64] — W1 은 U(±1/√H) (고정 씨앗), 나머지 0, Adam 0. 정책·가치 변수는 바이트 그대로 (머리는 정책 출력에 안 들어감).
// (옛 머리 학습값은 버림 — 꼴이 달라 못 옮김)
// 원래 머리말: 아래층 순환 체크포인트에 보조 예측 머리를 더한다 (ppo/README "보조 예측 머리", mdp/obs_layout.h AUX_OUT) — GPU 없이 파일 → 파일.
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

#include "../../mdp/ckpt_cfg.h"   // 설정 꼬리: 있으면 바뀐 열쇠만 고쳐 다시 붙임 (없으면 없이)
#include "../../mdp/obs_layout.h"

namespace {
constexpr uint32_t kMagic = 0x4F505058, kVersion = 3;
struct Blk { std::string name; int64_t n; };
}  // namespace

int main(int argc, char** argv) try {
  if ((argc != 3 && argc != 4) || std::string(argv[1]) == argv[2]) {
    std::fprintf(stderr, "사용: ckpt_add_aux2 입력.bin 출력.bin [은닉=64] (입력과 다른 경로)\n");
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
  const int64_t HH = argc == 4 ? std::atoi(argv[3]) : 64;
  if (HH <= 0 || HH % 32) throw std::runtime_error("은닉은 32 의 배수");
  int64_t H = 0;
  for (auto& b : blk)
    if (b.name == "actor.gru.wh") H = (int64_t)(std::sqrt((double)b.n / 3.0) + 0.5);
  if (blk.size() < 2 || blk[blk.size() - 2].name != "aux.w" || blk.back().name != "aux.b") throw std::runtime_error("끝 두 블록이 선형 보조 머리(aux.w · aux.b)가 아님");
  const int64_t K = blk.back().n, nold = blk[blk.size() - 2].n + K;
  if (blk[blk.size() - 2].n != K * H) throw std::runtime_error("aux.w 모양이 이상함");
  blk.pop_back(); blk.pop_back();
  for (auto* q : {&p, &m, &v}) q->resize(q->size() - nold);
  P -= nold;
  blk.push_back({"aux.w1", HH * H});
  blk.push_back({"aux.b1", HH});
  blk.push_back({"aux.w2", K * HH});
  blk.push_back({"aux.b2", K});
  const int64_t add = HH * H + HH + K * HH + K;
  const size_t w1 = p.size();
  for (auto* q : {&p, &m, &v}) q->insert(q->end(), add, 0.f);
  uint64_t st = 0x9E3779B97F4A7C15ull;   // 고정 씨앗 (splitmix64) — 같은 입력이면 같은 출력
  const float bd = 1.f / std::sqrt((float)H);
  for (int64_t i = 0; i < HH * H; ++i) {
    st += 0x9E3779B97F4A7C15ull;
    uint64_t z = st;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    p[w1 + i] = bd * (2.f * (float)((z >> 40) * (1.0 / 16777216.0)) - 1.f);
  }
  const int64_t P2 = P + add;
  ckcfg::Kv ck = ckcfg::read(argv[1]);   // 설정 꼬리 (mdp/ckpt_cfg.h): 보조 머리 · 보조 재료
  if (!ck.empty()) {
    ckcfg::set(ck, "aux_out", std::to_string(K));
    ckcfg::set(ck, "SIM_LOW_AUX", "1");
  }
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
    if (!ck.empty()) {
      const std::string t = ckcfg::trailer(ck);
      w(t.data(), t.size());
    }
    if (!g) throw std::runtime_error("쓰기 실패");
  }
  if (std::rename(tmp.c_str(), argv[2]) != 0) throw std::runtime_error("이름 바꾸기 실패");
  std::printf("%s: 보조 머리 선형 → 두 층 (은닉 %lld, 입력 %lld, 출력 %lld), 변수 %lld → %lld\n", argv[2], (long long)HH, (long long)H, (long long)K, (long long)(P + nold), (long long)P2);
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "ckpt_add_aux2: %s\n", e.what());
  return 1;
}
