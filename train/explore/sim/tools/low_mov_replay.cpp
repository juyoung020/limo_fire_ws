// 아래층 "움직이는 것" 칸 실물 CPU 판 == GPU 확인 (녹화한 추적 입력으로). GPU 없이 돈다 — 실물 Orin 에서도 (aarch64, -ffp-contract=off).
//   env_verify --low --mov-dump 녹화.bin … (SIM_EXACT=1 SIM_LOW_OBS_MOV=1 SIM_LOW_MOV4=1) 로 GPU 를 녹화하고
//   low_mov_replay 녹화.bin
// 걸음 t 마다 환경마다:
//   추적: 걸음 t−1 뒤 GPU 의 추적 상태(LowRl)·라이다 링·EnvState 를 실물 판(limo_fire_rl/low_mov.h LowMov::track)에 넣고 한 걸음 → 걸음 t 뒤 GPU 추적 상태와
//         추적 칸(tk_head ~ tk_y0) 비트 비교. 그 걸음에 정책이 안 몰았거나(LowRl::pol 0 — 시뮬은 추적을 안 돌림) 판이 새로 시작했으면(step 0 — 추적 비움) 건너뜀
//   칸:   걸음 t 뒤 GPU 추적 상태·EnvState 로 LowMov::slots → GPU 아래층 1D 의 kLoMov 6 칸 · kLoMov4 24 칸과 비트 비교
// 실물 판 한 번 부르는 시간(추적 한 걸음 · 칸 채우기, µs 평균)도 적는다.
// 끝줄: 다름 0 이면 종료 코드 0. 움직이는 것이 잡힌 걸음 수(칸 1·2·3·4 개 찬 것)를 같이 적는다 (빈 칸만 견주면 뜻 없음)
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "env.h"
#include "limo_fire_rl/low_mov.h"

namespace {
struct Rec {
  sim::EnvState e;
  sim::LowRl R;
  std::vector<float> ring, o1d;
};
bool same_trk(const sim::LowRl& a, const sim::LowRl& b) {
  const size_t o0 = offsetof(sim::LowRl, tk_head), o1 = offsetof(sim::LowRl, tk_y0) + sizeof(a.tk_y0);
  return std::memcmp(reinterpret_cast<const char*>(&a) + o0, reinterpret_cast<const char*>(&b) + o0, o1 - o0) == 0;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "low_mov_replay 녹화.bin   (env_verify --low --mov-dump 녹화.bin 이 만든 것)\n");
    return 2;
  }
  FILE* f = std::fopen(argv[1], "rb");
  if (!f) { std::perror(argv[1]); return 2; }
  try {
    int32_t hd[8];
    if (std::fread(hd, sizeof(hd), 1, f) != 1 || hd[0] != 0x564F4D4C) throw std::runtime_error("녹화 머리가 아님");
    if (hd[1] != (int32_t)sizeof(sim::EnvCfg) || hd[2] != (int32_t)sizeof(sim::EnvState) || hd[3] != (int32_t)sizeof(sim::LowRl) ||
        hd[4] != sim::kLidarRing * mdp::obs::LIDAR_BINS)
      throw std::runtime_error("구조체 크기가 녹화와 다름 (소스 판이 다름?)");
    const int n1 = hd[5], n_env = hd[6], n_rec = hd[7], rn = hd[4];
    sim::EnvCfg c;
    if (std::fread(&c, sizeof(c), 1, f) != 1) throw std::runtime_error("EnvCfg 읽기");
    if (!c.lo_mov) throw std::runtime_error("녹화가 SIM_LOW_OBS_MOV=1 이 아님 (추적이 안 돎)");
    const bool m4 = c.lo_mov4 != 0;
    if (n1 != (m4 ? sim::kLow1dMov4 : sim::kLow1d)) throw std::runtime_error("1D 칸 수가 lo_mov4 와 안 맞음");
    const auto rd = [&](Rec& r) {
      r.ring.resize((size_t)rn);
      r.o1d.resize((size_t)n1);
      if (std::fread(&r.e, sizeof(r.e), 1, f) != 1 || std::fread(&r.R, sizeof(r.R), 1, f) != 1 ||
          std::fread(r.ring.data(), sizeof(float), (size_t)rn, f) != (size_t)rn || std::fread(r.o1d.data(), sizeof(float), (size_t)n1, f) != (size_t)n1)
        throw std::runtime_error("녹화가 짧음");
    };
    std::vector<Rec> prev((size_t)n_env), cur((size_t)n_env);
    sim::EnvBatch b{};
    b.cfg = c;
    b.n_env = 1;
    long n_trk = 0, bad_trk = 0, n_idle = 0, bad_idle = 0, n_slot = 0, bad_mov = 0, bad_mov4 = 0, skip_reset = 0;
    long filled[5] = {};
    int first_bad = -1;
    double us_trk = 0, us_slot = 0;
    using clk = std::chrono::steady_clock;
    float mov[6], mov4[mdp::low::MOV_SLOTS * mdp::low::MOV_W];
    for (int t = 0; t < n_rec; ++t) {
      for (int i = 0; i < n_env; ++i) rd(cur[(size_t)i]);
      for (int i = 0; i < n_env; ++i) {
        const Rec& C = cur[(size_t)i];
        if (t > 0) {
          const Rec& P = prev[(size_t)i];
          if (C.e.step == 0) {
            ++skip_reset;
          } else if (C.R.pol) {
            limo_fire_rl::LowMov m;
            m.state_mut() = P.R;
            b.lidar_ring = const_cast<float*>(P.ring.data());
            const auto t0 = clk::now();
            m.track(b, P.e);
            us_trk += std::chrono::duration<double, std::micro>(clk::now() - t0).count();
            ++n_trk;
            if (!same_trk(m.state(), C.R)) {
              ++bad_trk;
              if (first_bad < 0) first_bad = t;
            }
          } else {
            ++n_idle;
            bad_idle += !same_trk(P.R, C.R);
          }
        }
        limo_fire_rl::LowMov m;
        m.state_mut() = C.R;
        b.lidar_ring = nullptr;
        const auto t1 = clk::now();
        m.slots(b, C.e, mov, m4 ? mov4 : nullptr);
        us_slot += std::chrono::duration<double, std::micro>(clk::now() - t1).count();
        ++n_slot;
        if (std::memcmp(mov, C.o1d.data() + sim::kLoMov, sizeof(mov)) != 0) {
          ++bad_mov;
          if (first_bad < 0) first_bad = t;
        }
        int nf = 0;
        if (m4) {
          if (std::memcmp(mov4, C.o1d.data() + sim::kLoMov4, sizeof(mov4)) != 0) {
            ++bad_mov4;
            if (first_bad < 0) first_bad = t;
          }
          for (int k = 0; k < mdp::low::MOV_SLOTS; ++k) nf += C.o1d[(size_t)sim::kLoMov4 + (size_t)mdp::low::MOV_W * k] > 0.5f;
        } else {
          nf = C.o1d[sim::kLoMov] > 0.5f;
        }
        filled[nf] += 1;
      }
      std::swap(prev, cur);
    }
    std::printf("녹화: 환경 %d · 걸음 %d · 1D %d (움직이는 것 칸 넷 %s)\n", n_env, n_rec - 1, n1, m4 ? "켬" : "끔");
    std::printf("추적 한 걸음 (실물 LowMov::track vs GPU): %ld 번 중 다름 %ld · 정책 안 몬 걸음 %ld 번 중 GPU 추적이 바뀐 것 %ld · 판 새로 시작 건너뜀 %ld\n",
                n_trk, bad_trk, n_idle, bad_idle, skip_reset);
    std::printf("칸 (실물 LowMov::slots vs GPU 아래층 1D): %ld 번 중 kLoMov 다름 %ld · kLoMov4 다름 %ld\n", n_slot, bad_mov, bad_mov4);
    std::printf("시간: 추적 한 걸음 %.2f µs · 칸 채우기 %.2f µs (평균)\n", n_trk ? us_trk / n_trk : 0.0, n_slot ? us_slot / n_slot : 0.0);
    std::printf("움직이는 것이 찬 칸 수별 걸음: 0 개 %ld · 1 개 %ld · 2 개 %ld · 3 개 %ld · 4 개 %ld\n", filled[0], filled[1], filled[2], filled[3], filled[4]);
    const bool ok = bad_trk == 0 && bad_idle == 0 && bad_mov == 0 && bad_mov4 == 0 && n_trk > 0;
    std::printf("%s%s\n", ok ? "같음 (비트)" : "다름", first_bad >= 0 ? (" — 처음 다른 걸음 " + std::to_string(first_bad)).c_str() : "");
    std::fclose(f);
    return ok ? 0 : 1;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "low_mov_replay: %s\n", ex.what());
    return 2;
  }
}
