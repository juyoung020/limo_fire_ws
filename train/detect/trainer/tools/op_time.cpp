// 연산별 시간(정방향·역방향) — 무엇을 빠르게 할지 고르기용. 사용: op_time [배치 32] [반복 10]
#include <algorithm>
#include <cstdio>
#include <map>
#include <string>

#include "net.h"

int main(int argc, char** argv) {
  int B = argc > 1 ? std::stoi(argv[1]) : 32, it = argc > 2 ? std::stoi(argv[2]) : 10;
  cudaStream_t s;
  CK(cudaStreamCreate(&s));
  cudnnHandle_t dnn;
  CDN(cudnnCreate(&dnn));
  CDN(cudnnSetStream(dnn, s));
  cublasHandle_t blas;
  CBL(cublasCreate(&blas));
  CBL(cublasSetStream(blas, s));
  Net net;
  NetCfg c;
  c.batch = B;
  net.build(c, dnn, blas, s);
  net.params.sync_w16(s);
  for (int i = 0; i < it + 2; ++i) {
    if (i == 2)
      for (auto& op : net.ops()) op->ms_f = op->ms_b = 0;
    net.zero_act_grads();
    net.forward_timed(true);
    net.backward_timed();
  }
  // 종류별 합
  std::map<std::string, std::pair<double, double>> kind;
  double tf = 0, tb = 0;
  for (auto& op : net.ops()) {
    std::string d = op->desc(), k = d.substr(0, d.find(' '));
    kind[k].first += op->ms_f / it;
    kind[k].second += op->ms_b / it;
    tf += op->ms_f / it;
    tb += op->ms_b / it;
  }
  std::printf("배치 %d: 정방향 %.1f ms, 역방향 %.1f ms\n", B, tf, tb);
  for (auto& [k, v] : kind) std::printf("  %-10s 정 %6.2f  역 %6.2f\n", k.c_str(), v.first, v.second);
  std::vector<Op*> ops;
  for (auto& op : net.ops()) ops.push_back(op.get());
  std::sort(ops.begin(), ops.end(), [](Op* a, Op* b) { return a->ms_f + a->ms_b > b->ms_f + b->ms_b; });
  std::printf("느린 연산 25:\n");
  for (int i = 0; i < 25 && i < (int)ops.size(); ++i)
    std::printf("  %6.2f + %6.2f ms  %s\n", ops[i]->ms_f / it, ops[i]->ms_b / it, ops[i]->desc().c_str());
  return 0;
}
