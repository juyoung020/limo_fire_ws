// 기울기 검사 결과 (정밀도와 무관한 평범한 형) — fp32·fp64 두 빌드가 같이 쓴다.
#pragma once

#include <string>
#include <vector>

struct GcBlock {
  long off, n;
  int kind;  // PKind
  std::string name;
};
struct GcOut {
  double loss = 0;
  long n_params = 0;
  std::vector<double> ga;     // 해석 기울기 (전부)
  std::vector<GcBlock> blocks;
  std::vector<long> fd_ids;   // 차분한 변수
  std::vector<double> fd;     // 차분 기울기
  bool proj_ok = false;       // σ 범위 투영: 범위 밖에서 update 한 번 뒤 log σ 가 [proj_lo, proj_hi] 안인가
  double proj_lo = 0, proj_hi = 0;
  std::vector<double> proj_ls;
};

namespace xppo {
GcOut gc_run(bool full, bool negative, bool do_fd, int K, int mlp);
}
namespace xppo64 {
GcOut gc_run(bool full, bool negative, bool do_fd, int K, int mlp);
}
