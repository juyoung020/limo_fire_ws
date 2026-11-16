// 옛 체크포인트(가치 전용 시뮬 참값 32 칸이 가치 입력에 있던 것, 2026-11-14 전)를 지금 구조로 바꾼다 — GPU 없이 파일 → 파일.
//   ckpt_drop_priv 입력.bin 출력.bin
// 하는 일: 가치 MLP 첫 층 가중치 [out][in] 에서 마지막 32 열(참값 입력)을 가중치·Adam m·v 에서 빼고, 정규화 μ·σ² 의 마지막 32 칸을 뺀다.
//   나머지(인코더·actor·log σ·Adam 걸음 수·반복·학습률·난수 자리)는 바이트 그대로 → 정책 출력은 옛 체크포인트와 비트까지 같다.
//   가치는 뺀 열의 몫(Σ w·정규화 참값)만큼 달라진다. 정규화한 참값은 평균이 0 이라 가치 평균은 그대로이고, 몇 반복 학습하면 다시 맞는다(ppo/README).
// 형식은 ppo.cu save/load 와 같다 (형식 3, fp32).
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr uint32_t kMagic = 0x4F505058, kVersion = 3;
constexpr int kPriv = 32;   // 뺀 참값 칸 수 (옛 mdp::obs::PRIV)
struct Blk { std::string name; int64_t n, off; };
}  // namespace

int main(int argc, char** argv) try {
  if (argc != 3 || std::string(argv[1]) == argv[2]) {
    std::fprintf(stderr, "사용: ckpt_drop_priv 입력.bin 출력.bin (입력과 다른 경로)\n");
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
  int64_t off = 0;
  for (auto& b : blk) {
    int32_t ln;
    r(&ln, 4);
    b.name.assign(ln, '\0');
    r(b.name.data(), ln);
    r(&b.n, 8);
    b.off = off;
    off += b.n;
  }
  int64_t P;
  r(&P, 8);
  if (P != off) throw std::runtime_error("변수 수가 층 크기 합과 다름");
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

  // 가치 MLP 첫 층: "critic.mlp0.w" (그래프 모드는 "gcritic.v.mlp0.w"), 편향 블록 크기 = out
  int wi = -1;
  for (int i = 0; i + 1 < nb; ++i)
    if (blk[i].name == "critic.mlp0.w" || blk[i].name == "gcritic.v.mlp0.w") wi = i;
  if (wi < 0) throw std::runtime_error("가치 MLP 첫 층을 못 찾음");
  const int64_t out = blk[wi + 1].n, in = blk[wi].n / out;
  if (blk[wi + 1].name != blk[wi].name.substr(0, blk[wi].name.size() - 1) + "b" || in * out != blk[wi].n) throw std::runtime_error("가치 첫 층 모양이 이상함");
  // 옛 구조인지: 가치 첫 층 입력 = actor 첫 층 입력 + 32 (그래프 모드는 actor MLP 가 없고 정규화가 참값 32 칸뿐)
  int64_t actor_in = -1;
  for (int i = 0; i + 1 < nb; ++i)
    if (blk[i].name == "actor.mlp0.w") actor_in = blk[i].n / blk[i + 1].n;
  if (actor_in >= 0 ? in != actor_in + kPriv || D < kPriv : D != kPriv || in <= kPriv) throw std::runtime_error("참값 입력이 없는 체크포인트 (이미 바꾼 것?)");
  const int64_t in2 = in - kPriv, cut = out * kPriv, w0 = blk[wi].off;
  double sum_abs_kept = 0, sum_abs_cut = 0;
  for (auto* a : {&p, &m, &v}) {
    std::vector<float> o;
    o.reserve(P - cut);
    o.insert(o.end(), a->begin(), a->begin() + w0);
    for (int64_t j = 0; j < out; ++j) {
      const float* row = a->data() + w0 + j * in;
      o.insert(o.end(), row, row + in2);
      if (a == &p)
        for (int64_t k = 0; k < in; ++k) (k < in2 ? sum_abs_kept : sum_abs_cut) += row[k] < 0 ? -row[k] : row[k];
    }
    o.insert(o.end(), a->begin() + w0 + blk[wi].n, a->end());
    a->swap(o);
  }
  blk[wi].n = out * in2;
  P -= cut;
  D -= kPriv;

  const std::string tmp = std::string(argv[2]) + ".tmp";
  {
    std::ofstream g(tmp, std::ios::binary);
    auto w = [&](const void* d, size_t n) { g.write((const char*)d, (std::streamsize)n); };
    w(hdr, 12);
    w(&nb, 4);
    for (auto& b : blk) {
      int32_t ln = (int32_t)b.name.size();
      w(&ln, 4);
      w(b.name.data(), ln);
      w(&b.n, 8);
    }
    w(&P, 8);
    w(p.data(), P * 4);
    w(m.data(), P * 4);
    w(v.data(), P * 4);
    w(&adam_t, 8);
    w(&D, 4);
    w(mean.data(), D * 4);
    w(var.data(), D * 4);
    w(tail, sizeof tail);
    if (!g) throw std::runtime_error("쓰기 실패");
  }
  if (std::rename(tmp.c_str(), argv[2]) != 0) throw std::runtime_error("이름 바꾸기 실패");
  std::printf("%s: %s [%lld][%lld → %lld], 정규화 %d → %d 칸, 변수 %lld 개\n", argv[2], blk[wi].name.c_str(), (long long)out, (long long)in, (long long)in2, D + kPriv, D,
              (long long)P);
  std::printf("뺀 열 |w| 평균 %.5f (남은 열 %.5f)\n", sum_abs_cut / (double)cut, sum_abs_kept / (double)(out * in2));
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "ckpt_drop_priv: %s\n", e.what());
  return 1;
}
