#include "fenv_api.h"

namespace fe {

CpuEnvs cpu_create(const fm::View& hv, uint32_t max_cells, int n, uint64_t seed) {
  CpuEnvs e;
  e.n = n; e.know_stride = max_cells; e.view = hv;
  e.core.resize(n); e.know.assign((size_t)max_cells * n, 0);
  e.action.assign(n, 0); e.reward.assign(n, 0.f); e.done.assign(n, 0); e.ends.assign(n, EpisodeEnd{});
  for (int i = 0; i < n; ++i) init_one(e.core[i], seed, i);
  return e;
}
void cpu_reset_all(CpuEnvs& e) {
  for (int i = 0; i < e.n; ++i) reset(e.core[i], e.view, e.know.data() + (size_t)i * e.know_stride);
}
void cpu_step(CpuEnvs& e) {
  for (int i = 0; i < e.n; ++i)
    step_one(e.core[i], e.action[i], e.view, e.know.data() + (size_t)i * e.know_stride, &e.reward[i], &e.done[i], &e.ends[i]);
}

}  // namespace fe
