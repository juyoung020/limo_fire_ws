// OptiX 호스트 쪽: 맵마다 GAS(삼각형마다 키운 상자, compaction), 모듈·파이프라인·SBT. include/optix_rt.h
#include "optix_rt.h"

#include <optix.h>
#include <optix_function_table_definition.h>
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
  OptixProgramGroup pg[3] = {};
  OptixPipeline pipe = nullptr;
  OptixShaderBindingTable sbt = {};
  std::vector<void*> bufs;
  unsigned long long* gas = nullptr;
  const void** tris = nullptr;
  RtParams* dparams = nullptr;
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
  std::vector<unsigned long long> handles;
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
    cudaFree(dbb);
    I.bufs.push_back(comp);
    handles.push_back((unsigned long long)h);
  }
  I.gas = dalloc<unsigned long long>(handles.size());
  cudaMemcpy(I.gas, handles.data(), handles.size() * 8, cudaMemcpyHostToDevice);
  I.tris = dalloc<const void*>(dev_tris.size());
  cudaMemcpy(I.tris, dev_tris.data(), dev_tris.size() * sizeof(void*), cudaMemcpyHostToDevice);
  std::ifstream f(SIM_OPTIXIR, std::ios::binary);
  if (!f) throw std::runtime_error(std::string("OptiX 모듈 파일 없음: ") + SIM_OPTIXIR);
  const std::string ir((std::istreambuf_iterator<char>(f)), {});
  OptixModuleCompileOptions mo = {};
  OptixPipelineCompileOptions po = {};
  po.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
  po.numPayloadValues = 9;
  po.numAttributeValues = 2;
  po.pipelineLaunchParamsVariableName = "params";
  po.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;
  char log[4096];
  size_t ll = sizeof(log);
  ox(optixModuleCreate(I.ctx, &mo, &po, ir.data(), ir.size(), log, &ll, &I.mod), "모듈");
  OptixProgramGroupOptions pgo = {};
  OptixProgramGroupDesc ds[3] = {};
  ds[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  ds[0].raygen.module = I.mod;
  ds[0].raygen.entryFunctionName = "__raygen__rays";
  ds[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
  ds[1].miss.module = I.mod;
  ds[1].miss.entryFunctionName = "__miss__none";
  ds[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  ds[2].hitgroup.moduleIS = I.mod;
  ds[2].hitgroup.entryFunctionNameIS = "__intersection__tri";
  ll = sizeof(log);
  ox(optixProgramGroupCreate(I.ctx, ds, 3, &pgo, log, &ll, I.pg), "프로그램 묶음");
  OptixPipelineLinkOptions plo = {};
  plo.maxTraceDepth = 1;
  ll = sizeof(log);
  ox(optixPipelineCreate(I.ctx, &po, &plo, I.pg, 3, log, &ll, &I.pipe), "파이프라인");
  Rec rec[3];
  for (int k = 0; k < 3; ++k) ox(optixSbtRecordPackHeader(I.pg[k], &rec[k]), "SBT");
  Rec* dr = dalloc<Rec>(3);
  cudaMemcpy(dr, rec, sizeof(rec), cudaMemcpyHostToDevice);
  I.bufs.push_back(dr);
  I.sbt.raygenRecord = (CUdeviceptr)&dr[0];
  I.sbt.missRecordBase = (CUdeviceptr)&dr[1];
  I.sbt.missRecordStrideInBytes = sizeof(Rec);
  I.sbt.missRecordCount = 1;
  I.sbt.hitgroupRecordBase = (CUdeviceptr)&dr[2];
  I.sbt.hitgroupRecordStrideInBytes = sizeof(Rec);
  I.sbt.hitgroupRecordCount = 1;
  I.dparams = dalloc<RtParams>(1);
}

OptixRt::~OptixRt() {
  Impl& I = *impl_;
  if (I.pipe) optixPipelineDestroy(I.pipe);
  for (auto& g : I.pg) if (g) optixProgramGroupDestroy(g);
  if (I.mod) optixModuleDestroy(I.mod);
  if (I.ctx) optixDeviceContextDestroy(I.ctx);
  for (void* p : I.bufs) cudaFree(p);
  cudaFree(I.gas);
  cudaFree(I.tris);
  cudaFree(I.dparams);
  delete impl_;
}

void OptixRt::launch(const RtParams& p0, cudaStream_t st) {
  Impl& I = *impl_;
  RtParams p = p0;
  p.gas = I.gas;
  p.tris = I.tris;
  if (p.n <= 0) return;
  cudaMemcpyAsync(I.dparams, &p, sizeof(p), cudaMemcpyHostToDevice, st);
  ox(optixLaunch(I.pipe, st, (CUdeviceptr)I.dparams, sizeof(RtParams), &I.sbt, (unsigned)p.n, 1, 1), "launch");
}

}  // namespace sim
