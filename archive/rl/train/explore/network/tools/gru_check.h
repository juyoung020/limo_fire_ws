// gru_check 공유: 크기·자료 (fp32·fp64 번역 단위가 같이 씀)
#pragma once
#include <string>
#include <vector>
namespace gruchk {
constexpr int NF = 10, ND = 5, H = 32, T = 5, B = 3, LD = NF + H + ND, OFFD = NF + H, NR = T * B;
struct Data {
  std::vector<double> x, h0, reset, w, params;
};
struct Blk { std::string name; long off, n; };
struct Out {
  long n = 0, wx = 0, bx = 0, wh = 0, bh = 0;
  std::vector<Blk> blocks;
  std::vector<double> h, grad, dxf;
  double step_diff = 0;   // step 걸음마다 vs forward_seq 최대 차
};
}  // namespace gruchk
// 각 정밀도 번역 단위가 정의: 변수 수·자리만 (params 를 채우기 전), 그다음 전체 실행
namespace xnet { gruchk::Out gru_layout(); gruchk::Out gru_run(const gruchk::Data& d, bool negative); }
namespace xnet64 { gruchk::Out gru_layout(); gruchk::Out gru_run(const gruchk::Data& d, bool negative); }
