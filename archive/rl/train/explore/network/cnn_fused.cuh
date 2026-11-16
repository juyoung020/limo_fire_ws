// 합성곱 가지 하나(2×2 stride 2 합성곱 3 층 + 편향 + ELU)를 커널 하나로 — 정방향·역방향. fp32 학습 빌드 전용.
//
// 3 층 뒤 출력 픽셀 하나는 입력 8×8 조각 하나(겹치지 않음)에만 기대므로, 블록이 조각 TP 개(타일)를 맡아
// u8 입력 → 층 0 → 층 1 → 층 2 를 공유 메모리·레지스터 안에서 끝내고 평탄화 결과만 전역(X0)에 쓴다.
// 역방향은 같은 타일을 다시 정방향해(중간 활성값을 전역에 안 둠) 기울기를 구하고, 가중치·편향 기울기는
// 블록이 레지스터에 쌓아 블록별 부분합 → 블록 순서대로 더함(원자 연산 없음, 결정적). 첫 층 입력 기울기는 안 구함(u8).
// 가중치 배치는 기존과 같다: 층마다 W [cout][자식 2×2][cin], 편향 [cout], 가지 안에서 w0 b0 w1 b1 w2 b2 연속.
#pragma once

#include <cstdint>

#include "common.cuh"

namespace XNET_NS {

// 입력 채널 담음꼴 (2026-11-15, 롤아웃 그림 줄여 담기): 0 = u8 판 (채널 판이 그 자리에 H·W 바이트), 1 = 비트 판 (값 0/255 만 — 칸 y·W + x 가 비트, 바이트 안 낮은 비트부터),
// 2 = 늘 0 (안 담음). 기본(packed 거짓)은 옛 배치: 채널 c 가 img_off + c·H·W 의 u8 판
enum : int8_t { kChU8 = 0, kChBit = 1, kChZero = 2 };
struct CnnBranch {
  int cin, h, w, c0, c1, c2;  // 입력 채널·크기, 층 채널
  long img_off;               // 환경 하나 u8 관측 안 오프셋
  int img_stride;
  long feat, flat_off;        // X0 행 길이, 이 가지 평탄화 시작 열
  bool packed = false;        // 참이면 채널마다 담음꼴 ch_kind·ch_off (샘플 행 시작 기준 바이트 오프셋) — img_stride 는 담은 행 길이
  int8_t ch_kind[8] = {};
  int32_t ch_off[8] = {};
};

#ifndef XNET_FP64
// 이 모양을 융합 커널이 지원하나 (3 층, h·w 가 8 의 배수, 채널 조합이 인스턴스에 있음)
bool cnn_fused_supported(const CnnBranch& b, int nlayers);
// params = 가지 첫 변수(w0) 위치. X0 의 이 가지 열에 ELU 출력을 쓴다. 샘플 i 의 입력 = img 의 idx[i] 행 (idx = nullptr 이면 i).
// tc = 텐서 코어 판(bf16 입력·fp32 누산; 채널이 16 의 배수일 때만, 아니면 fp32 판)
bool cnn_fused_tc_supported(const CnnBranch& b);
void cnn_fused_forward(const CnnBranch& b, const float* params, const uint8_t* img, const int* idx, int n, float* X0,
                       cudaStream_t st, bool tc);
// dZ3 = 출력 기울기에 ELU'(출력)을 곱한 것 (X0 와 같은 배치 — 넷째 층 역방향이 z_out 으로 씀). grads(가지 첫 변수 위치)에 더한다.
// part = 작업 공간(cnn_fused_part_floats).
void cnn_fused_backward(const CnnBranch& b, const float* params, const uint8_t* img, const int* idx, int n, const float* dZ3,
                        float* grads, float* part, cudaStream_t st, bool broken_elu, bool tc);
long cnn_fused_part_floats(const CnnBranch& b);
#endif

}  // namespace XNET_NS
