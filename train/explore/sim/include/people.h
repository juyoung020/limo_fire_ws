// 움직이는 사람 (동적 장애물, mdp/README "움직이는 사람"): 형태·상태·설정. 동작은 people_sim.h, 호스트(설정 읽기·통로 너비)는 src/people_host.cpp.
//
// 사람 = 세로 원기둥(반지름 radius, 키 height, 판마다 사람마다 무작위). 환경마다 판 시작 때 0 ~ k_max 명(고르게) — k_max 는 환경마다
// 판 시작 인자(Env::set_people_max, 커리큘럼이 바꿈. 바꾸면 다음 판부터). 0 이면 사람 코드는 아무 일도 안 한다(없던 때와 비트 같음).
//   - 서 있기(p_stand) 또는 걷기 0~speed m/s: 갈 수 있는 바닥 위 무작위 목표점(시작 칸 후보)으로 곧게. 막히면(벽·다른 사람) 잠깐 서고 새 목표.
//     벽: 물리 걸음마다 새 자리의 "가장 가까운 막힌 칸까지 거리"(맵마다 미리 잰 표, World::clear)가 반지름 + 칸 대각보다 커야 움직인다 →
//     원기둥이 막힌 칸에 닿는 일이 없다(한 걸음 ≤ 1.3 m/s × 0.02 s = 2.6 cm < 칸 5 cm 라 건너뛰지도 못함). 갈 수 있는 바닥 칸 밖으로도 안 나감.
//   - 로봇 무시(p_ignore): 로봇이 있어도 그대로 걷는다 — 정책이 피해야만 한다. 나머지는 로봇(참 자세 중심)이 yield_dist 안이면
//     로봇에서 멀어지는 쪽으로 yield_speed 로 비켜서고, 못 비키면(벽) 선다.
//   - 사람끼리: 새 자리가 다른 사람과 겹치면서 더 가까워지면 안 움직인다(멀어지는 건 됨).
//   - 놓기: 판 시작에 로봇 출발점에서 spawn_keep 안엔 안 놓고, 서로 안 겹치게. 자리가 안 나면 그만큼 적게 놓는다.
// 센서: 라이다·깊이 광선이 원기둥을 맞힌다(scene.h raycast, SceneView::cyl). 라이다 재질은 "보통"(빠짐 없음 + 물체 σ 배율) — 잠정,
//   옷 재질 세기·빠짐을 실측하면 바꾼다. 지도는 라이다 그대로 쌓아 사람이 흔적을 남긴다(실물과 같음). 덮음(벽 면 칸 분자·분모)엔 안 든다.
// 충돌: 로봇 몸통 사각형과 원기둥이 겹치면 벽 충돌처럼 판 끝(collided) + hit_person. 가장 가까운 사람까지 몸통 간격은 near_d.
// 난수는 판 world_seed 에서 따로 뽑은 줄기(People::rng) — 사람을 켜도 로봇 쪽 난수 순서는 그대로다.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "people_geom.h"

namespace sim {

constexpr int kMaxPeople = 8;          // 환경마다 사람 수 상한 (k_max ≤ 이것)
constexpr float kNoPerson = 99.f;      // near_d: 사람이 없을 때

// 사람 모양·걸음 (config/env.yaml people). 수는 여기 없다 — 판 시작 인자
struct PeopleCfg {
  float r_lo, r_hi, h_lo, h_hi;        // 반지름·키 범위 (m)
  float v_lo, v_hi;                    // 걷는 사람 속도 범위 (m/s)
  float p_stand, p_ignore;             // 서 있기만 하는 사람·로봇 무시하는 사람 비율
  float yield_dist, yield_speed;       // 로봇 중심이 이 안이면 비켜섬 (m), 그 속도
  float spawn_keep;                    // 로봇 출발점에서 이 안엔 안 놓음 (m)
  float pause;                         // 목표에 닿으면 0 ~ 이 시간 섬 (s)
};

enum : int32_t { kPersonWalk = 1, kPersonIgnore = 2 };   // People::mode 비트

// 환경 한 판의 사람들 (EnvState::people). 위치는 세계 좌표(참값)
struct People {
  Cyl cyl[kMaxPeople];                 // x, y, r, h — 광선(SceneView::cyl)이 이 배열을 그대로 본다
  float gx[kMaxPeople], gy[kMaxPeople];   // 걷는 목표점
  float speed[kMaxPeople];             // 걷는 속도 (m/s)
  float wait[kMaxPeople];              // 남은 서 있을 시간 (s)
  int32_t mode[kMaxPeople];            // kPersonWalk | kPersonIgnore
  int32_t n;                           // 이 판 사람 수 (0 ~ k_max)
  int32_t k_max;                       // 판 시작 인자: 다음 판 사람 수 상한 (Env::set_people_max). 리셋이 안 지움
  int32_t hit_person;                  // 이번 행동 중 로봇 몸통이 사람과 겹침 (collided 도 1). 다음 행동 시작에 지움 —
                                       //   판 끝 리셋 뒤에도 남아 끝 종류를 밖에서 볼 수 있다
  float near_d;                        // 로봇 몸통(사각형) ~ 가장 가까운 사람 원기둥 간격 (m, 겹치면 음수, 없으면 kNoPerson). 보상용
  uint64_t rng;                        // 사람 난수 줄기
};

// 호스트 (src/people_host.cpp)
// env.yaml 의 people 절 (없으면 예외)
PeopleCfg load_people_cfg(const std::string& env_yaml);
// 칸마다 가장 가까운 막힌 칸(kCellBlocked)까지 칸 가운데끼리 거리 (cm, 255 에서 자름). 갈 수 있는 바닥 칸만 재고 나머지는 0
std::vector<uint8_t> people_clearance(const std::vector<uint8_t>& grid, int W, int H, float res);

}  // namespace sim
