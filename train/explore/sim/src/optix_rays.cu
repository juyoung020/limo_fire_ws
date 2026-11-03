// OptiX 장치 프로그램 (optix-ir 로 컴파일 — CMake). include/optix_rt.h 설명
#include <optix.h>

#include "optix_rt.h"
#include "scene.h"

extern "C" __constant__ sim::RtParams params;

extern "C" __global__ void __raygen__rays() {
  const int i = optixGetLaunchIndex().x;
  if (i >= params.n || (params.n_dev && params.off + i >= *params.n_dev)) return;
  const float4 o = params.o[i], d = params.d[i];
  const int m = __float_as_int(d.w);
  if (m < 0) return;   // 안 쏘는 칸 (out_tri 는 준비 커널이 쓴 그대로)
  unsigned int p0 = __float_as_uint(o.w), p1 = (unsigned)params.tri0[i], p2 = (unsigned)m;
  unsigned int ox = __float_as_uint(o.x), oy = __float_as_uint(o.y), oz = __float_as_uint(o.z);
  unsigned int dx = __float_as_uint(d.x), dy = __float_as_uint(d.y), dz = __float_as_uint(d.z);
  optixTrace((OptixTraversableHandle)params.gas[m], make_float3(o.x, o.y, o.z), make_float3(d.x, d.y, d.z), 0.f, o.w, 0.f,
             OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, 0, 1, 0, p0, p1, p2, ox, oy, oz, dx, dy,
             dz);
  const int tri = (int)p1;
  params.out_tri[i] = tri;
  params.out_t[i] = tri != -1 ? __uint_as_float(p0) : sim::kInf;
}

extern "C" __global__ void __miss__none() {}

extern "C" __global__ void __intersection__tri() {
  const int prim = (int)optixGetPrimitiveIndex();
  const sim::V3 o{__uint_as_float(optixGetPayload_3()), __uint_as_float(optixGetPayload_4()), __uint_as_float(optixGetPayload_5())};
  const sim::V3 d{__uint_as_float(optixGetPayload_6()), __uint_as_float(optixGetPayload_7()), __uint_as_float(optixGetPayload_8())};
  const sim::Tri* tris = (const sim::Tri*)params.tris[optixGetPayload_2()];
  const float t = sim::ray_tri(o, d, sim::load_tri(tris, prim));
  const float best = __uint_as_float(optixGetPayload_0());
  const int bt = (int)optixGetPayload_1();
  if (t < best || (t == best && bt >= 0 && prim < bt)) {   // scene.h trav_tri 와 같은 규칙
    optixSetPayload_0(__float_as_uint(t));
    optixSetPayload_1((unsigned)prim);
    if (t < best) optixReportIntersection(t, 0);   // 하드웨어가 tmax 를 줄여 먼 상자를 자름 (답은 payload)
  }
}
