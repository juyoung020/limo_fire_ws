// 위층 판단 모델(scripts/low_decider.h — 실물 노드 /explore/explorer · 시뮬 low_explore_check · 라벨 dec_rollout)에 보내는 로봇 지도 사진.
// 요청 본문(JSON)과 보내기는 low_decider.h 와 Transport(실물 limo_fire_explore/decider_client.h) 쪽에 있다 — 여기는 사진(PNG)과 data URI 만.
// (combine_run --upper decider 의 옛 요청 꼴 · 카메라 그림 · libcurl 동시 요청 · 녹화 · 끝 선언 채점은 2026-11-18 지움)
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace dec {

// 요청 입력 꼴 (low_decider.h Params::input): 후보 글만 · 지도 사진만 · 둘 다
enum Input { kText = 0, kImage = 1, kBoth = 2 };
inline int input_of(const std::string& s) { return s == "text" ? kText : s == "image" ? kImage : s == "both" ? kBoth : -1; }

// 사진에 그릴 후보 하나 (글자 key 꼬리표)
struct Cand {
  char key = 'A';
  int slot = 0;        // 요청 안 후보 번호
  bool object = false; // 물체 후보 (low_decider 물체 찾기 지시) — 지도 그림에서 노란 점 대신 하늘색 네모 (프론티어 후보와 구별)
  int px = 0, py = 0;  // 지도 사진 안 자리 (render_png_mem 이 채움)
};

// 기억을 얹은 로봇 지도 사진 (판단 모델 위층 꼴 A): 건드린 칸 범위(+ 로봇·후보·길·목표)를 정사각형으로 잘라 px × px 로. 흰색 = 빈 곳, 검정 = 막힘, 회색 = 모름,
// 파란 세모 = 로봇(머리 방향), 노란 점 + 글자 꼬리표 = 후보, 지나온 길(점, 오래될수록 옅은 보라)과 지난 목표(닿음 초록 ✓ · 실패 빨강 × · 바꿈 회색 고리 · 지금 목표 파란 고리).
// 위가 +y. lmap 은 환경 하나의 지도 [size²] (로그 오즈 × 1000, 모름 −32768), occ_l 이상이 막힘. cells 는 후보 칸 (x, y) 쌍 [cand.size() · 2].
// 좌표는 지도 칸 (cx, cy — 칸 가운데는 +0.5). 돌려주는 값은 PNG 바이트
struct MapMem {
  struct P { float cx, cy, age_s; };
  struct G { float cx, cy; int outcome; };   // 0 지금 목표 · 1 닿음 · 2 실패 · 3 바꿈 (다른 목표로)
  std::vector<P> trail;
  std::vector<G> goals;
  struct N { float cx, cy; int num; };   // 결정 기록 번호 (1 = 가장 최근 결정의 목표)
  std::vector<N> nums;
  float trail_s = 60.f;   // 길 색의 시간 폭 (s)
  bool clear_tags = false;   // 꼬리표를 글자마다 다른 색 · 또렷한 글꼴로 (후보 모드)
};
std::vector<uint8_t> render_png_mem(const int16_t* lmap, int size, int gw, int gh, int16_t unknown_l, int16_t occ_l, int kb[4], float rob_cx, float rob_cy, float rob_th,
                                    std::vector<Cand>& cand, const std::vector<int>& cells, int px, const MapMem& mem);
// 꼴 B: 한 사진(px × px, 토큰 수 그대로)을 2 × 2 칸으로 — 왼쪽 위 = 지금 지도(후보 · 길 · 목표 표시, 위 render_png_mem 과 같은 그림을 반 크기로),
// 나머지 = 지난 지도 past[k] (past_age_s[k] 초 전, 그때 로봇 자세 past_pose[3k..3k+2] 칸 좌표·rad). 모든 칸이 지금과 같은 범위·축척이라 자리가 맞음. 칸마다 왼쪽 위에 시각 글 (NOW · -5s …)
std::vector<uint8_t> render_png_tiles(const int16_t* lmap, const std::vector<const int16_t*>& past, const std::vector<float>& past_age_s, const std::vector<float>& past_pose,
                                      int size, int gw, int gh, int16_t unknown_l, int16_t occ_l, int kb[4], float rob_cx, float rob_cy, float rob_th, std::vector<Cand>& cand,
                                      const std::vector<int>& cells, int px, const MapMem& mem);
std::string png_uri(const std::vector<uint8_t>& png);   // "data:image/png;base64,…" (요청 images 칸)

}  // namespace dec
