// OptiX 호스트 쪽: 맵마다 GAS(삼각형마다 키운 상자, compaction), 모듈·파이프라인·SBT. include/optix_rt.h
#include "optix_rt.h"

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>

#include "scene_host.h"

namespace sim {
extern const unsigned char kOptixRaysPtx[];   // 빌드가 만든 optix_rays_ptx.cpp (PTX 바이트)
extern const std::size_t kOptixRaysPtxLen;
}  // namespace sim

namespace sim {
namespace {
void ox(OptixResult r, const char* what) {
  if (r != OPTIX_SUCCESS) {   // CUDA 쪽 오류(7900)면 그 CUDA 오류 글도 붙임
    const cudaError_t ce = cudaPeekAtLastError();
    throw std::runtime_error(std::string("OptiX ") + what + " 실패 " + std::to_string((int)r) + " (" + optixGetErrorName(r) + ", CUDA: " + cudaGetErrorString(ce) + ")");
  }
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
  bool has_exact = false, has_fast = false;   // 지은 길
  const void** tris = nullptr;
  const void** tris2 = nullptr;   // [맵] 덧 삼각형 (없으면 nullptr)
  int32_t* nb = nullptr;          // [맵] 바탕 삼각형 수
  RtParams* dparams = nullptr;   // prog 마다 한 칸 (kRtProgs)
  RtParams hparams[kRtProgs] = {};   // 장치 칸에 올라 있는 값 (같으면 다시 안 올림)
  bool hvalid[kRtProgs] = {};
  cudaStream_t up = nullptr;   // 캡처 중 인자 올리기용
};

bool OptixRt::available() {
  cudaFree(0);
  return optixInit() == OPTIX_SUCCESS;
}

OptixRt::OptixRt(const std::vector<const Scene*>& scenes, const std::vector<const void*>& dev_tris, bool want_exact, bool want_fast,
                 const std::vector<const Scene*>* extras, const std::vector<const void*>* dev_tris2) : impl_(new Impl) {
  if (!want_exact && !want_fast) throw std::runtime_error("OptixRt: 정확한 길·빠른 길 중 하나는 지어야 함");
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
  auto build_pair = [&](const Scene* s) -> std::pair<unsigned long long, unsigned long long> {   // (정확한 길, 빠른 길) GAS — 안 짓는 길은 0
    unsigned long long hx = 0ull;
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
    unsigned flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
    if (want_exact) {
    OptixAabb* dbb = dalloc<OptixAabb>(nt);
    cudaMemcpy(dbb, bb.data(), nt * sizeof(OptixAabb), cudaMemcpyHostToDevice);
    CUdeviceptr vb = (CUdeviceptr)dbb;
    OptixBuildInput bi = {};
    bi.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    bi.customPrimitiveArray.aabbBuffers = &vb;
    bi.customPrimitiveArray.numPrimitives = (unsigned)nt;
    bi.customPrimitiveArray.flags = &flags;
    bi.customPrimitiveArray.numSbtRecords = 1;
    hx = build_gas(bi);
    cudaFree(dbb);
    }
    if (!want_fast) return {hx, 0ull};
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
    const unsigned long long hf = build_gas(ti);
    cudaFree(dv);
    return {hx, hf};
  };

  // IAS 하나: 바탕 GAS(인스턴스 번호 0) + 덧 GAS(1), 같은 자리 (덧붙인 형상 맵)
  auto build_ias = [&](unsigned long long h0, unsigned long long h1) -> unsigned long long {
    OptixInstance ins[2] = {};
    const unsigned long long hs[2] = {h0, h1};
    for (int k = 0; k < 2; ++k) {
      const float tf[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
      std::memcpy(ins[k].transform, tf, sizeof(tf));
      ins[k].instanceId = (unsigned)k;
      ins[k].visibilityMask = 255;
      ins[k].sbtOffset = 0;
      ins[k].flags = OPTIX_INSTANCE_FLAG_NONE;
      ins[k].traversableHandle = hs[k];
    }
    OptixInstance* di = dalloc<OptixInstance>(2);
    cudaMemcpy(di, ins, sizeof(ins), cudaMemcpyHostToDevice);
    OptixBuildInput ib = {};
    ib.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    ib.instanceArray.instances = (CUdeviceptr)di;
    ib.instanceArray.numInstances = 2;
    const unsigned long long h = build_gas(ib);   // 같은 짓기·compaction 길 (입력만 인스턴스)
    I.bufs.push_back(di);   // IAS 는 인스턴스 배열을 짓는 동안만 읽지만 남겨 둠 (작음)
    return h;
  };
  std::map<const void*, std::pair<unsigned long long, unsigned long long>> base_gas;   // 같은 바탕은 GAS 하나
  bool any_extra = false;
  std::vector<const void*> t2(scenes.size(), nullptr);
  std::vector<int32_t> nbv(scenes.size(), 0);
  for (size_t i = 0; i < scenes.size(); ++i) {
    auto it = base_gas.find(dev_tris[i]);
    if (it == base_gas.end()) it = base_gas.emplace(dev_tris[i], build_pair(scenes[i])).first;
    nbv[i] = (int32_t)scenes[i]->tris.size();
    const Scene* x = extras ? (*extras)[i] : nullptr;
    if (!x) {
      handles.push_back(it->second.first);
      fast.push_back(it->second.second);
      continue;
    }
    any_extra = true;
    t2[i] = (*dev_tris2)[i];
    const auto xp = build_pair(x);
    handles.push_back(want_exact ? build_ias(it->second.first, xp.first) : 0ull);
    fast.push_back(want_fast ? build_ias(it->second.second, xp.second) : 0ull);
  }
  I.tris2 = dalloc<const void*>(scenes.size());
  cudaMemcpy(I.tris2, t2.data(), t2.size() * sizeof(void*), cudaMemcpyHostToDevice);
  I.nb = dalloc<int32_t>(scenes.size());
  cudaMemcpy(I.nb, nbv.data(), nbv.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
  I.has_exact = want_exact;
  I.has_fast = want_fast;
  I.gas = dalloc<unsigned long long>(handles.size());
  cudaMemcpy(I.gas, handles.data(), handles.size() * 8, cudaMemcpyHostToDevice);
  I.gas_fast = dalloc<unsigned long long>(fast.size());
  cudaMemcpy(I.gas_fast, fast.data(), fast.size() * 8, cudaMemcpyHostToDevice);
  I.tris = dalloc<const void*>(dev_tris.size());
  cudaMemcpy(I.tris, dev_tris.data(), dev_tris.size() * sizeof(void*), cudaMemcpyHostToDevice);
  // 바이너리에 넣어 둔 PTX (sim/cmake/embed_ptx.cmake — 빌드 폴더 파일을 읽지 않음)
  const std::string ir(reinterpret_cast<const char*>(kOptixRaysPtx), kOptixRaysPtxLen);
  OptixModuleCompileOptions mo = {};
  OptixPipelineCompileOptions po = {};
  po.traversableGraphFlags = any_extra ? OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY : OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;   // 덧붙인 형상 맵은 IAS
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
    ox(optixPipelineSetStackSize(I.pipe, dc_trav, dc_state, cont, any_extra ? 2 : 1), "스택");   // 순회 깊이: IAS → GAS 면 2
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
  cudaFree(I.tris2);
  cudaFree(I.nb);
  cudaFree(I.dparams);
  delete impl_;
}

void OptixRt::launch(const RtParams& p0, cudaStream_t st) {
  Impl& I = *impl_;
  RtParams p = p0;
  if (p.fast ? !I.has_fast : !I.has_exact) throw std::runtime_error(p.fast ? "OptixRt: 빠른 길 GAS 를 안 지음" : "OptixRt: 정확한 길 GAS 를 안 지음 (SIM_EXACT=1 로 Env 를 만들 것)");
  p.gas = p.fast ? I.gas_fast : I.gas;
  p.tris = I.tris;
  p.tris2 = I.tris2;
  p.nb = I.nb;
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
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(st, &cs);
    if (cs == cudaStreamCaptureStatusNone) {
      cudaMemcpyAsync(I.dparams + slot, &I.hparams[slot], sizeof(RtParams), cudaMemcpyHostToDevice, st);
    } else {
      // CUDA Graph 캡처 중 (학습 롤아웃 그래프): 페이지 메모리 복사는 캡처할 수 없다 → 따로 스트림에서 지금 올리고 기다린다.
      // 이 칸을 읽는 앞선 발사가 없을 때만 (캡처 전에 같은 칸을 쓴 발사가 아직 돌 수 있으면 안전하지 않음 — 시뮬 걸음 발사 1 ~ 4 는 Env 마다 처음 한 번만 올림)
      if (!I.up) cudaStreamCreateWithFlags(&I.up, cudaStreamNonBlocking);
      cudaMemcpyAsync(I.dparams + slot, &I.hparams[slot], sizeof(RtParams), cudaMemcpyHostToDevice, I.up);
      cudaStreamSynchronize(I.up);
    }
  }
  const unsigned dim = (unsigned)std::min(p.n, 1 << 19);   // 발사 칸은 많아야 2^19 개 — raygen 이 칸마다 여러 일을 차례로
  OptixShaderBindingTable sbt = I.sbt;
  sbt.raygenRecord = I.raygen[p.prog];
  ox(optixLaunch(I.pipe, st, (CUdeviceptr)(I.dparams + slot), sizeof(RtParams), &sbt, dim, 1, 1), "launch");
  if (std::getenv("SIM_KSYNC")) {   // 진단 (env_kernel.cu ksync 와 같은 스위치): 발사마다 기다려 OptiX 프로그램의 잘못된 접근을 그 자리에서
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
      std::fprintf(stderr, "SIM_KSYNC: optixLaunch(prog %d) 뒤 CUDA 오류: %s\n", p.prog, cudaGetErrorString(e));
      std::abort();
    }
  }
}

}  // namespace sim
