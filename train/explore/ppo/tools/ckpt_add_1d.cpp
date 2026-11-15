// 아래층 체크포인트의 1D 관측 뒤에 칸을 더한다 (움직이는 것 칸 넷 — mdp/obs_layout.h N1D_MOV4) — GPU 없이 파일 → 파일.
//   ckpt_add_1d 입력.bin 출력.bin [더할 칸 수, 기본 N1D_MOV4 − N1D]
// 하는 일: 1D 를 읽는 층의 1D 열 뒤에 0 열을 끼움 — actor·critic MLP 첫 층 [out][F | h | 1D] (행마다 끝), 순환 핵 입력 Wx [3H][F | 1D] (행마다 끝).
//   정규화 평균·분산 뒤에 (0, 1) 칸을 더함. 새 자리 Adam m·v 0, 나머지는 바이트 그대로 → 새 칸 값과 무관하게 처음 정책·가치는 옛 것과 같다.
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
  if (argc < 3 || argc > 4 || std::string(argv[1]) == argv[2]) {
    std::fprintf(stderr, "사용: ckpt_add_1d 입력.bin 출력.bin [더할 칸 수] (입력과 다른 경로)\n");
    return 2;
  }
  const int ext = argc == 4 ? std::atoi(argv[3]) : mdp::low::N1D_MOV4 - mdp::low::N1D;
  if (ext <= 0) throw std::runtime_error("더할 칸 수 > 0");
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
  // 블록 크기·이름으로 모양 알기: h = actor.gru.wh 의 √(n/3), 출력 = 다음 블록(편향) 크기
  int64_t H = 0;
  for (auto& b : blk)
    if (b.name == "actor.gru.wh") H = (int64_t)(std::sqrt((double)b.n / 3.0) + 0.5);
  std::vector<Blk> nblk;
  std::vector<float> np, nm, nv;
  int64_t off = 0, added = 0;
  for (int i = 0; i < nb; ++i) {
    const Blk& b = blk[i];
    const float* bp = p.data() + off;
    const float* bm = m.data() + off;
    const float* bv = v.data() + off;
    const bool mlp0 = b.name == "actor.mlp0.w" || b.name == "critic.mlp0.w";
    const bool wx = b.name == "actor.gru.wx" || b.name == "critic.gru.wx";
    if (mlp0 || wx) {
      const int64_t rows = wx ? 3 * H : blk[i + 1].n, in = b.n / rows;
      if (rows <= 0 || rows * in != b.n) throw std::runtime_error(b.name + " 모양이 이상함");
      if (in < D) throw std::runtime_error(b.name + ": 입력 열 < 정규화 칸 수");
      for (int64_t j = 0; j < rows; ++j) {   // 행마다 끝 = 1D 열의 끝 → 그 뒤에 0 열
        for (auto* pr : {&np, &nm, &nv}) {
          const float* src = pr == &np ? bp : pr == &nm ? bm : bv;
          pr->insert(pr->end(), src + j * in, src + (j + 1) * in);
          pr->insert(pr->end(), ext, 0.f);
        }
      }
      nblk.push_back({b.name, rows * (in + ext)});
      added += rows * ext;
    } else {
      np.insert(np.end(), bp, bp + b.n);
      nm.insert(nm.end(), bm, bm + b.n);
      nv.insert(nv.end(), bv, bv + b.n);
      nblk.push_back(b);
    }
    off += b.n;
  }
  if (added == 0) throw std::runtime_error("1D 를 읽는 층을 못 찾음 (아래층 체크포인트?)");
  const int64_t P2 = (int64_t)np.size();
  const int32_t D2 = D + ext;
  mean.insert(mean.end(), ext, 0.f);
  var.insert(var.end(), ext, 1.f);
  ckcfg::Kv ck = ckcfg::read(argv[1]);   // 설정 꼬리 (mdp/ckpt_cfg.h): 1D 칸 수 · 움직이는 것 칸 넷
  if (!ck.empty()) {
    ckcfg::set(ck, "d1d", std::to_string(D2));
    if (D2 == mdp::low::N1D_MOV4) ckcfg::set(ck, "SIM_LOW_MOV4", "1");
  }
  const std::string tmp = std::string(argv[2]) + ".tmp";
  {
    std::ofstream g(tmp, std::ios::binary);
    auto w = [&](const void* d, size_t n) { g.write((const char*)d, (std::streamsize)n); };
    w(hdr, 12);
    const int32_t nb2 = (int32_t)nblk.size();
    w(&nb2, 4);
    for (auto& b : nblk) {
      int32_t ln = (int32_t)b.name.size();
      w(&ln, 4);
      w(b.name.data(), ln);
      w(&b.n, 8);
    }
    w(&P2, 8);
    w(np.data(), P2 * 4);
    w(nm.data(), P2 * 4);
    w(nv.data(), P2 * 4);
    w(&adam_t, 8);
    w(&D2, 4);
    w(mean.data(), D2 * 4);
    w(var.data(), D2 * 4);
    w(tail, sizeof tail);
    if (!ck.empty()) {
      const std::string t = ckcfg::trailer(ck);
      w(t.data(), t.size());
    }
    if (!g) throw std::runtime_error("쓰기 실패");
  }
  if (std::rename(tmp.c_str(), argv[2]) != 0) throw std::runtime_error("이름 바꾸기 실패");
  std::printf("%s: 1D %d → %d 칸, 변수 %lld → %lld (+%lld, 모두 0)\n", argv[2], D, D2, (long long)P, (long long)P2, (long long)added);
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "ckpt_add_1d: %s\n", e.what());
  return 1;
}
