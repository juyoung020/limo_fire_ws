// OptiX 호스트 쪽: 맵마다 GAS(삼각형마다 키운 상자, compaction), 모듈·파이프라인·SBT. include/optix_rt.h
#include "optix_rt.h"

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>

#include "scene_host.h"

namespace sim {
namespace {
void ox(OptixResult r, const char* what) {
  if (r != OPTIX_SUCCESS) throw std::runtime_error(std::string("OptiX ") + what + " 실패 " + std::to_string((int)r));
}
template <class T>
T* dalloc(size_t n) {
  void* p = nullptr;
  if (cudaMalloc(&p, std::max<size_t>(1, n * sizeof(T))) != cudaSuccess) throw std::runtime_error("cudaMalloc 실패 (OptiX)");
  return (T*)p;
}
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Rec {
  char h[OPTIX_SBT_RECORD_HEADER_SIZE];
};
}  // namespace

struct OptixRt::Impl {
  OptixDeviceContext ctx = nullptr;
  OptixModule mod = nullptr;
  OptixProgramGroup pg[kRtProgs + 3] = {};   // raygen kRtProgs 개 · miss · 정확한 길 hit · 빠른 길 hit
  OptixPipeline pipe = nullptr;
  OptixShaderBindingTable sbt = {};
  CUdeviceptr raygen[kRtProgs] = {};
  std::vector<void*> bufs;
  unsigned long long* gas = nullptr;
  unsigned long long* gas_fast = nullptr;   // 하드웨어 삼각형
  const void** tris = nullptr;
  RtParams* dparams = nullptr;   // prog 마다 한 칸 (kRtProgs)
  RtParams hparams[kRtProgs] = {};   // 장치 칸에 올라 있는 값 (같으면 다시 안 올림)
  bool hvalid[kRtProgs] = {};
};

bool OptixRt::available() {
  cudaFree(0);
  return optixInit() == OPTIX_SUCCESS;
}

OptixRt::OptixRt(const std::vector<const Scene*>& scenes, const std::vector<const void*>& dev_tris) : impl_(new Impl) {
  Impl& I = *impl_;
  cudaFree(0);
  ox(optixInit(), "init");
  OptixDeviceContextOptions co = {};
  ox(optixDeviceContextCreate(0, &co, &I.ctx), "context");
  std::vector<unsigned long long> handles, fast;
  auto build_gas = [&](const OptixBuildInput& bi) {   // GAS 하나 (compaction)
    OptixAccelBuildOptions ao = {};
    ao.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    ao.operation = OPTIX_BUILD_OPERATION_BUILD;
    OptixAccelBufferSizes sz;
    ox(optixAccelComputeMemoryUsage(I.ctx, &ao, &bi, 1, &sz), "GAS 크기");
    char* tmp = dalloc<char>(sz.tempSizeInBytes);
    char* out = dalloc<char>(sz.outputSizeInBytes);
    size_t* csz = dalloc<size_t>(1);
    OptixAccelEmitDesc em = {};
    em.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    em.result = (CUdeviceptr)csz;
    OptixTraversableHandle h;
    ox(optixAccelBuild(I.ctx, 0, &ao, &bi, 1, (CUdeviceptr)tmp, sz.tempSizeInBytes, (CUdeviceptr)out, sz.outputSizeInBytes, &h, &em, 1), "GAS");
    size_t cs = 0;
    cudaMemcpy(&cs, csz, sizeof(size_t), cudaMemcpyDeviceToHost);
    char* comp = dalloc<char>(cs);
    ox(optixAccelCompact(I.ctx, 0, h, (CUdeviceptr)comp, cs, &h), "GAS compaction");
    cudaDeviceSynchronize();
    cudaFree(tmp);
    cudaFree(out);
    cudaFree(csz);
    I.bufs.push_back(comp);
    return (unsigned long long)h;
  };
  for (const Scene* s : scenes) {
    const size_t nt = s->tris.size();
    // 삼각형 상자를 1 mm + 상대 1e−5 키움: 상자 시험 반올림으로 ray_tri 가 맞히는 삼각형을 놓치지 않게
    std::vector<OptixAabb> bb(nt);
    for (size_t k = 0; k < nt; ++k) {
      const Tri& T = s->tris[k];
      float lo[3] = {T.a.x, T.a.y, T.a.z}, hi[3] = {T.a.x, T.a.y, T.a.z};
      for (const V3& v : {T.b, T.c}) {
        const float p[3] = {v.x, v.y, v.z};
        for (int q = 0; q < 3; ++q) { lo[q] = std::min(lo[q], p[q]); hi[q] = std::max(hi[q], p[q]); }
      }
      for (int q = 0; q < 3; ++q) {
        const float pad = 1e-3f + 1e-5f * std::max(std::fabs(lo[q]), std::fabs(hi[q]));
        lo[q] -= pad;
        hi[q] += pad;
      }
      bb[k] = OptixAabb{lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]};
    }
    OptixAabb* dbb = dalloc<OptixAabb>(nt);
    cudaMemcpy(dbb, bb.data(), nt * sizeof(OptixAabb), cudaMemcpyHostToDevice);
    CUdeviceptr vb = (CUdeviceptr)dbb;
    unsigned flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
    OptixBuildInput bi = {};
    bi.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    bi.customPrimitiveArray.aabbBuffers = &vb;
    bi.customPrimitiveArray.numPrimitives = (unsigned)nt;
    bi.customPrimitiveArray.flags = &flags;
    bi.customPrimitiveArray.numSbtRecords = 1;
    handles.push_back(build_gas(bi));
    cudaFree(dbb);
    // 빠른 길: 같은 삼각형을 하드웨어 삼각형으로 (꼭짓점 3 개씩, 번호 = 삼각형 번호)
    float* dv = dalloc<float>(nt * 9);
    cudaMemcpy(dv, s->tris.data(), nt * 36, cudaMemcpyHostToDevice);
    CUdeviceptr tv = (CUdeviceptr)dv;
    OptixBuildInput ti = {};
    ti.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    ti.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    ti.triangleArray.vertexStrideInBytes = 12;
    ti.triangleArray.numVertices = (unsigned)(nt * 3);
    ti.triangleArray.vertexBuffers = &tv;
    ti.triangleArray.flags = &flags;
    ti.triangleArray.numSbtRecords = 1;
    fast.push_back(build_gas(ti));
    cudaFree(dv);
  }

  I.gas = dalloc<unsigned long long>(handles.size());
  cudaMemcpy(I.gas, handles.data(), handles.size() * 8, cudaMemcpyHostToDevice);
  I.gas_fast = dalloc<unsigned long long>(fast.size());
  cudaMemcpy(I.gas_fast, fast.data(), fast.size() * 8, cudaMemcpyHostToDevice);
  I.tris = dalloc<const void*>(dev_tris.size());
  cudaMemcpy(I.tris, dev_tris.data(), dev_tris.size() * sizeof(void*), cudaMemcpyHostToDevice);
  std::ifstream f(SIM_OPTIX_PTX, std::ios::binary);
  if (!f) throw std::runtime_error(std::string("OptiX 모듈 파일 없음: ") + SIM_OPTIX_PTX);
  const std::string ir((std::istreambuf_iterator<char>(f)), {});
  OptixModuleCompileOptions mo = {};
  OptixPipelineCompileOptions po = {};
  po.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
  po.numPayloadValues = 9;
  po.numAttributeValues = 2;
  po.pipelineLaunchParamsVariableName = "params";
  po.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM | OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
  char log[4096];
  size_t ll = sizeof(log);
  ox(optixModuleCreate(I.ctx, &mo, &po, ir.data(), ir.size(), log, &ll, &I.mod), "모듈");
  OptixProgramGroupOptions pgo = {};
  constexpr int R = kRtProgs, NG = kRtProgs + 3;
  OptixProgramGroupDesc ds[NG] = {};
  const char* rg_name[kRtProgs] = {"__raygen__rays", "__raygen__depth", "__raygen__depth_side", "__raygen__beam", "__raygen__beam_side"};
  for (int k = 0; k < R; ++k) {
    ds[k].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    ds[k].raygen.module = I.mod;
    ds[k].raygen.entryFunctionName = rg_name[k];
  }
  ds[R + 0].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
  ds[R + 0].miss.module = I.mod;
  ds[R + 0].miss.entryFunctionName = "__miss__none";
  ds[R + 1].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  ds[R + 1].hitgroup.moduleIS = I.mod;
  ds[R + 1].hitgroup.entryFunctionNameIS = "__intersection__tri";
  ds[R + 2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;   // 빠른 길: 하드웨어 삼각형 + 가장 가까운 맞음
  ds[R + 2].hitgroup.moduleCH = I.mod;
  ds[R + 2].hitgroup.entryFunctionNameCH = "__closesthit__tri";
  ll = sizeof(log);
  ox(optixProgramGroupCreate(I.ctx, ds, NG, &pgo, log, &ll, I.pg), "프로그램 묶음");
  OptixPipelineLinkOptions plo = {};
  plo.maxTraceDepth = 1;
  ll = sizeof(log);
  ox(optixPipelineCreate(I.ctx, &po, &plo, I.pg, NG, log, &ll, &I.pipe), "파이프라인");
  // 스택 크기: 프로그램마다 잰 값으로 (optix_stack_size.h). 기본값은 raygen 이 광선 여러 개·결과 뒤 단계까지 하는 판에서 모자라 발사가 깨졌다(people_check)
  {
    OptixStackSizes ss = {};
    for (int k = 0; k < NG; ++k) ox(optixUtilAccumulateStackSizes(I.pg[k], &ss, I.pipe), "스택 크기");
    unsigned dc_trav = 0, dc_state = 0, cont = 0;
    ox(optixUtilComputeStackSizes(&ss, plo.maxTraceDepth, 0, 0, &dc_trav, &dc_state, &cont), "스택 크기 계산");
    ox(optixPipelineSetStackSize(I.pipe, dc_trav, dc_state, cont, 1), "스택");
  }
  Rec rec[NG];
  for (int k = 0; k < NG; ++k) ox(optixSbtRecordPackHeader(I.pg[k], &rec[k]), "SBT");
  Rec* dr = dalloc<Rec>(NG);
  cudaMemcpy(dr, rec, sizeof(rec), cudaMemcpyHostToDevice);
  I.bufs.push_back(dr);
  for (int k = 0; k < R; ++k) I.raygen[k] = (CUdeviceptr)&dr[k];
  I.sbt.raygenRecord = I.raygen[0];
  I.sbt.missRecordBase = (CUdeviceptr)&dr[R];
  I.sbt.missRecordStrideInBytes = sizeof(Rec);
  I.sbt.missRecordCount = 1;
  I.sbt.hitgroupRecordBase = (CUdeviceptr)&dr[R + 1];
  I.sbt.hitgroupRecordStrideInBytes = sizeof(Rec);
  I.sbt.hitgroupRecordCount = 2;   // 0 정확한 길(교차 프로그램), 1 빠른 길(하드웨어 삼각형)
  I.dparams = dalloc<RtParams>(kRtProgs);
}

OptixRt::~OptixRt() {
  Impl& I = *impl_;
  if (I.pipe) optixPipelineDestroy(I.pipe);
  for (auto& g : I.pg) if (g) optixProgramGroupDestroy(g);
  if (I.mod) optixModuleDestroy(I.mod);
  if (I.ctx) optixDeviceContextDestroy(I.ctx);
  for (void* p : I.bufs) cudaFree(p);
  cudaFree(I.gas);
  cudaFree(I.gas_fast);
  cudaFree(I.tris);
  cudaFree(I.dparams);
  delete impl_;
}

void OptixRt::launch(const RtParams& p0, cudaStream_t st) {
  Impl& I = *impl_;
  RtParams p = p0;
  p.gas = p.fast ? I.gas_fast : I.gas;
  p.tris = I.tris;
  if (p.n <= 0) return;
  // 발사 인자는 prog 마다 장치 칸 하나에 둔다. 시뮬 걸음의 발사(prog 1 ~ 4)는 인자가 Env 하나 동안 안 바뀌므로 처음 한 번만 올린다
  // (전에는 발사마다 페이지 메모리 → 장치 복사 한 번 — 걸음마다 4 번, 호스트가 복사를 기다림. CUDA 그래프 캡처도 막았음)
  const int slot = p.prog >= 0 && p.prog < kRtProgs ? p.prog : 0;
  const RtParams& h = I.hparams[slot];
  const bool same = I.hvalid[slot] && h.prog == p.prog && h.batch == p.batch && h.o == p.o && h.d == p.d && h.tri0 == p.tri0 && h.gas == p.gas && h.tris == p.tris &&
                    h.out_t == p.out_t && h.out_tri == p.out_tri && h.n == p.n && h.n_dev == p.n_dev && h.off == p.off && h.fast == p.fast;
  if (!same) {   // 앞서 넣은 발사가 이 칸을 아직 읽을 수 있으므로 같은 스트림 차례로 올린다 (호스트 사본은 객체 안 — 복사가 끝날 때까지 살아 있음)
    I.hparams[slot] = p;
    I.hvalid[slot] = true;
    cudaMemcpyAsync(I.dparams + slot, &I.hparams[slot], sizeof(RtParams), cudaMemcpyHostToDevice, st);
  }
  const unsigned dim = (unsigned)std::min(p.n, 1 << 19);   // 발사 칸은 많아야 2^19 개 — raygen 이 칸마다 여러 일을 차례로
  OptixShaderBindingTable sbt = I.sbt;
  sbt.raygenRecord = I.raygen[p.prog];
  ox(optixLaunch(I.pipe, st, (CUdeviceptr)(I.dparams + slot), sizeof(RtParams), &sbt, dim, 1, 1), "launch");
}

}  // namespace sim
