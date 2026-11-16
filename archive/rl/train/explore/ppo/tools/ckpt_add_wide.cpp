// 아래층 체크포인트(explore_spec_low())에 넓은 지도 가지(explore_spec_low(true) — mdp/obs_layout.h WIDE_*)를 더한다 — GPU 없이 파일 → 파일.
//   ckpt_add_wide 입력.bin 출력.bin [씨앗] [--gru H]
//   입력에 넓은 지도 가지가 이미 있으면(넓은 지도만으로 학습한 체크포인트) --gru 로 순환 핵만 더한다 (넓은 지도 쪽은 바이트 그대로).
// 하는 일 (actor·critic 인코더 둘 다 — 나뉜 인코더 enc·enc_c):
//   · 새 가지 cnn3 의 합성곱 앞 세 층은 Net::init 과 같은 무작위(U(±1/√(cin·4))), 넷째 층(꼬리) 가중치·편향은 0 → 새 가지 특징 = ELU(0) = 0
//   · MLP 첫 층(actor.mlp0.w · critic.mlp0.w) [out][in] 의 합성곱 특징 열 뒤(1D 열 앞)에 512 열을 끼움 — U(±1/√새 in) 무작위
//   · Adam m·v 는 새 자리 0, 나머지(옛 가중치·m·v·Adam 걸음 수·정규화·반복·학습률·난수 자리)는 바이트 그대로
// --gru H (explore_spec_low(true, H) — network/gru.cuh): 순환 핵도 더한다. MLP 첫 층 입력이 [F | h | 1D] 가 되게 넓은 지도 열 뒤에 H 열을 0 으로 끼우고,
//   actor.gru · critic.gru 변수(wx [3H][F + 1D] · bx · wh [H][3H] · bh)를 PyTorch nn.GRU 기본 U(±1/√H) 로 critic MLP 뒤(log σ 앞)에 — h 열이 0 이라 처음 정책은 그대로.
// 새 가지 특징이 0 이라 정책·가치 출력은 옛 체크포인트와 같다 (끼운 열은 0 을 곱함 — 더하는 차례가 바뀌는 GEMM 은 반올림 몫만큼 다를 수 있음).
// 처음 기울기: 꼬리 가중치는 (MLP 새 열 · 앞 층 출력)으로 바로 받고, 앞 세 층은 꼬리가 0 이 아니게 된 뒤부터.
// 형식은 ppo.cu save/load 와 같다 (형식 3, fp32). 이름·크기는 network/actor_critic.cu Net::build 규칙 ("<pfx>.cnn<가지>.conv<층>.w|b", W [cout][4][cin]).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <utility>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../mdp/ckpt_cfg.h"   // 설정 꼬리: 있으면 바뀐 열쇠만 고쳐 다시 붙임 (없으면 없이)
#include "../../mdp/obs_layout.h"

namespace {
constexpr uint32_t kMagic = 0x4F505058, kVersion = 3;
constexpr int KK = 4;                                  // 2 × 2 합성곱 자식 수
constexpr int kCh[4] = {16, 32, 32, 32};               // explore_spec_low conv_ch
constexpr int kOld = 3;                                // 옛 가지 수 (가까운 지도 · 라이다 · 깊이) — 새 가지 번호
constexpr int kFeatNew = 32 * 4 * 4;                   // 64 × 64 → 4 층 → 4 × 4 × 32
struct Blk { std::string name; int64_t n; };
}  // namespace

int main(int argc, char** argv) try {
  int gru_h = 0;
  std::vector<std::string> pos;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--gru" && i + 1 < argc) gru_h = std::atoi(argv[++i]);
    else pos.push_back(argv[i]);
  }
  if (pos.size() < 2 || pos.size() > 3 || pos[0] == pos[1] || gru_h < 0) {
    std::fprintf(stderr, "사용: ckpt_add_wide 입력.bin 출력.bin [씨앗] [--gru H] (입력과 다른 경로)\n");
    return 2;
  }
  argv[1] = (char*)pos[0].c_str();
  argv[2] = (char*)pos[1].c_str();
  const uint64_t seed = pos.size() == 3 ? std::strtoull(pos[2].c_str(), nullptr, 10) : 1;
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
  char tail[8 + 4 + 4 + 8 + 8];   // 정규화 개수 · 반복 · 학습률 · act 호출 수 · 씨앗 — 그대로 옮김
  r(tail, sizeof tail);
  if (D != mdp::low::N1D) throw std::runtime_error("아래층 체크포인트가 아님 (정규화 칸 수 " + std::to_string(D) + ")");
  bool has_wide = false;
  for (auto& b : blk) {
    if (b.name.find(".cnn" + std::to_string(kOld) + ".") != std::string::npos) has_wide = true;
    if (b.name.find(".gru.") != std::string::npos) throw std::runtime_error("이미 순환 핵이 있음: " + b.name);
  }
  if (has_wide && gru_h == 0) throw std::runtime_error("이미 넓은 지도 가지가 있음 (순환 핵만 더하려면 --gru H)");
  const int64_t feat_new = has_wide ? 0 : kFeatNew;   // 이번에 끼울 넓은 지도 열 수

  std::mt19937_64 rng(seed);
  auto uni = [&](std::vector<float>& o, int64_t n, float bound) {   // Net::init 과 같은 꼴 (float 로 뽑음)
    std::uniform_real_distribution<float> U(-bound, bound);
    for (int64_t i = 0; i < n; ++i) o.push_back(U(rng));
  };
  std::vector<Blk> nblk;
  std::vector<float> np, nm, nv;
  int64_t off = 0, added_conv = 0, added_mlp = 0, added_gru = 0;
  for (int i = 0; i < nb; ++i) {
    const Blk& b = blk[i];
    const float* bp = p.data() + off;
    const float* bm = m.data() + off;
    const float* bv = v.data() + off;
    const bool is_mlp0 = b.name == "actor.mlp0.w" || b.name == "critic.mlp0.w";
    if (is_mlp0) {
      const int64_t out = blk[i + 1].n, in = b.n / out;
      if (blk[i + 1].name != b.name.substr(0, b.name.size() - 1) + "b" || in * out != b.n) throw std::runtime_error(b.name + " 모양이 이상함");
      const int64_t nf = in - mdp::low::N1D, in2 = in + feat_new + gru_h;
      if (nf != (has_wide ? 1024 + kFeatNew : 1024))
        throw std::runtime_error(b.name + ": 합성곱 특징 열 " + std::to_string(nf) + (has_wide ? " (1536 이어야 — 가지 4 개)" : " (1024 이어야 — 옛 가지 3 개)"));
      const float bd = 1.f / std::sqrt((float)in2);
      for (int64_t j = 0; j < out; ++j) {
        np.insert(np.end(), bp + j * in, bp + j * in + nf);
        uni(np, feat_new, bd);
        np.insert(np.end(), gru_h, 0.f);   // h 열 0 — 처음 정책 그대로
        np.insert(np.end(), bp + j * in + nf, bp + (j + 1) * in);
        for (int q = 0; q < 2; ++q) {   // Adam m · v: 새 열 0
          std::vector<float>& o = q ? nv : nm;
          const float* src = q ? bv : bm;
          o.insert(o.end(), src + j * in, src + j * in + nf);
          o.insert(o.end(), feat_new + gru_h, 0.f);
          o.insert(o.end(), src + j * in + nf, src + (j + 1) * in);
        }
      }
      nblk.push_back({b.name, out * in2});
      added_mlp += out * (feat_new + gru_h);
    } else {
      np.insert(np.end(), bp, bp + b.n);
      nm.insert(nm.end(), bm, bm + b.n);
      nv.insert(nv.end(), bv, bv + b.n);
      nblk.push_back(b);
    }
    off += b.n;
    // 순환 핵: critic MLP 마지막 블록 뒤 (다음 블록이 critic.mlp 가 아니면)
    if (gru_h > 0 && b.name.rfind("critic.mlp", 0) == 0 && (i + 1 == nb || blk[i + 1].name.rfind("critic.mlp", 0) != 0)) {
      const int64_t H = gru_h, nin = 1024 + kFeatNew + mdp::low::N1D;   // GRU 입력 = [F (옛 1024 + 넓은 지도 512) | 1D]
      const float bd = 1.f / std::sqrt((float)H);
      for (const char* side : {"actor", "critic"}) {
        const std::pair<const char*, int64_t> parts[4] = {{".gru.wx", 3 * H * nin}, {".gru.bx", 3 * H}, {".gru.wh", 3 * H * H}, {".gru.bh", 3 * H}};
        for (auto& pt : parts) {
          uni(np, pt.second, bd);
          nm.insert(nm.end(), pt.second, 0.f);
          nv.insert(nv.end(), pt.second, 0.f);
          nblk.push_back({std::string(side) + pt.first, pt.second});
          added_gru += pt.second;
        }
      }
    }
    // 인코더의 마지막 옛 가지 꼬리 편향 뒤에 새 가지
    for (const char* pfx : {"enc", "enc_c"}) {
      if (has_wide) break;
      if (b.name != std::string(pfx) + ".cnn" + std::to_string(kOld - 1) + ".conv3.b") continue;
      int cin = mdp::low::WIDE_C;
      for (int l = 0; l < 4; ++l) {
        const std::string nmz = std::string(pfx) + ".cnn" + std::to_string(kOld) + ".conv" + std::to_string(l);
        const int64_t nw = (int64_t)kCh[l] * cin * KK;
        const float bd = 1.f / std::sqrt((float)cin * KK);
        if (l < 3) {
          uni(np, nw, bd);
          uni(np, kCh[l], bd);
        } else {
          np.insert(np.end(), nw + kCh[l], 0.f);   // 꼬리 0 — 처음 정책 = 옛 정책
        }
        nm.insert(nm.end(), nw + kCh[l], 0.f);
        nv.insert(nv.end(), nw + kCh[l], 0.f);
        nblk.push_back({nmz + ".w", nw});
        nblk.push_back({nmz + ".b", kCh[l]});
        added_conv += nw + kCh[l];
        cin = kCh[l];
      }
    }
  }
  const int64_t P2 = (int64_t)np.size();
  if ((int64_t)nm.size() != P2 || (int64_t)nv.size() != P2 || P2 != P + added_conv + added_mlp + added_gru) throw std::runtime_error("새 변수 수 계산이 안 맞음");
  if ((!has_wide && added_conv == 0) || added_mlp == 0 || (gru_h > 0 && added_gru == 0)) throw std::runtime_error("인코더 꼬리나 MLP 첫 층을 못 찾음 (아래층 체크포인트?)");

  ckcfg::Kv ck = ckcfg::read(argv[1]);   // 설정 꼬리 (mdp/ckpt_cfg.h): 넓은 지도 · 순환 핵
  if (!ck.empty()) {
    ckcfg::set(ck, "SIM_LOW_WIDE", "1");
    if (gru_h > 0) ckcfg::set(ck, "gru_h", std::to_string(gru_h));
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
  std::printf("%s: 층 %d → %zu, 변수 %lld → %lld (합성곱 +%lld · MLP 첫 층 +%lld · 순환 핵 +%lld, H %d), 씨앗 %llu\n", argv[2], nb, nblk.size(), (long long)P,
              (long long)P2, (long long)added_conv, (long long)added_mlp, (long long)added_gru, gru_h, (unsigned long long)seed);
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "ckpt_add_wide: %s\n", e.what());
  return 1;
}
