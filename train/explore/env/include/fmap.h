// 맵 묶음(Map bank): .fmap 여러 개를 한 덩어리로 올린다. 호스트 로더 + 장치/호스트 공용 보기(View).
//
// .fmap 형식은 train/maps/src/lib.rs 머리말과 같다. 여기서는 칸 값에 표시를 하나 더 얹는다:
//   bit0-1  0 바깥 · 1 빈 곳(도달) · 2 벽          (lib.rs 값 그대로)
//   bit2    덮을 표면(TARGET) — 빈 곳과 8-이웃으로 맞닿은 벽·바깥 칸. "갈 수 있는 곳에서 보이는 표면"
//   bit3    시작 가능(START) — 벽에서 START_CLEAR 이상 떨어진 빈 칸(로봇이 처음부터 부딪히지 않게)
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "detmath.h"

namespace fm {

constexpr uint8_t OUTSIDE = 0, FREE = 1, WALL = 2;
constexpr uint8_t KIND_MASK = 3, TARGET = 4, START = 8;
constexpr float RES = 0.05f;
constexpr float START_CLEAR = 0.25f;
constexpr int MAX_FAC = 64;   // 맵 하나의 소화기·비상구·감지기 각각 최대

DEV bool blocks(uint8_t c) { return (c & KIND_MASK) != FREE; }   // 광선·몸통을 막는 칸(바깥·벽)

// 장치·호스트 공용 보기. 포인터는 같은 배치의 장치 메모리이거나 호스트 메모리.
struct View {
  const uint8_t* cell;      // 모든 맵의 칸을 이어 붙임
  const uint64_t* off;      // 맵 m 의 첫 칸 위치
  const uint32_t* w;
  const uint32_t* h;
  const float* ox;
  const float* oy;
  const uint32_t* free_off; // 빈 칸 목록(불 놓을 곳) 시작
  const uint32_t* free_n;
  const uint32_t* start_off; // 시작 가능 칸 목록 시작
  const uint32_t* start_n;
  const uint32_t* idx;      // 빈 칸·시작 칸 목록(맵 안 칸 번호)
  const uint32_t* n_target; // 덮을 표면 칸 수
  const float* area;        // 도달 넓이 m²
  int n_maps;
};

// 호스트 쪽 묶음
struct Bank {
  std::vector<std::string> names;
  std::vector<uint8_t> cell;
  std::vector<uint64_t> off;
  std::vector<uint32_t> w, h, free_off, free_n, start_off, start_n, idx, n_target;
  std::vector<float> ox, oy, area;
  // 시설(맵마다 MAX_FAC 칸, 남는 칸은 NaN): 학습 관측·보상에는 아직 안 쓰고 뷰어·평가용으로 싣는다
  std::vector<float> ext_xy, exit_xy, det_xy;
  std::vector<uint32_t> n_ext, n_exit, n_det;
  uint32_t max_cells = 0;

  View host_view() const;
};

// index.tsv 가 있는 폴더(fmap_build --out)에서 source(houseexpo|procthor|duct|"" = 전부) 맵을 최대 limit 개 읽는다.
// skip 개를 건너뛰고(학습/시험 나누기), 순서는 index.tsv 의 정렬 순서.
Bank load_bank(const std::string& dir, const std::string& source, size_t limit, size_t skip = 0);

// 장치 사본. free_device 로 놓는다.
struct DeviceBank {
  View view{};
  void* blocks[16] = {};
  int n_blocks = 0;
};
DeviceBank upload(const Bank& b);
void free_device(DeviceBank& d);

}  // namespace fm
