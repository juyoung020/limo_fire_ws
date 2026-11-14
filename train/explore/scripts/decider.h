// 위층을 바깥 판단 모델 서버에 맡기는 길 (combine_run --upper decider). 측정 전용, 호스트(CPU)에서만 돈다.
// 서버는 llama.cpp llama-server 의 POST /v1/systemone (TypeSafe System One API — tools/server/README.md): 상태(state)·사진(images)·질문(questions)을 주면
// 글을 쓰지 않고 선택지마다 확률을 돌려준다. 여기서는 결정마다 choice 질문 하나(선택지 = 유효 후보)를 묻는다. 모델은 미세조정 없이 그대로(zero-shot).
//   입력 text  : 후보마다 글 한 줄 (아래 Cand — 위층 신경망이 보는 후보 특징 mdp::hier::F_* 과 같은 값)
//        image : 로봇 지도 사진(후보를 글자 동그라미로 표시) + 과제 설명만
//        both  : 사진 + 글
//        all   : 로봇 카메라 자리에서 본 장면 그림 + 지도 사진 + 글 (아래 render_cam_png · build_body_all) — "다 주고 돌리기"
//   방식 choice : choice 질문 하나로 후보를 고름
//        prune  : 규칙 차례는 그대로 두고 앞쪽 후보마다 noul 질문 "그 길이 막혔나"만 물어 막혔다는 후보를 건너뜀 (모델은 그 밖에 차례를 못 바꿈)
//   카메라 그림은 진짜 RGB 가 아니다: 시뮬 장면에는 삼각형과 물체 번호만 있고 색·재질이 없어서, 광선을 쏴 거리·면 방향·물체 번호로 칠한 그림이다.
//   프롬프트 판(prompt 1 ~ 3)은 decider.cpp build_body 에 있다 — 실험용이라 코드 안에 둔다.
// 판단에 걸린 시간은 시뮬 시계에 안 들어간다 (서버가 답할 때까지 시뮬이 멈춰 기다림).
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace dec {

enum Input { kText = 0, kImage = 1, kBoth = 2, kAll = 3 };
enum Mode { kChoice = 0, kPrune = 1, kScore = 2 };
inline int input_of(const std::string& s) { return s == "text" ? kText : s == "image" ? kImage : s == "both" ? kBoth : s == "all" ? kAll : -1; }

// 유효 후보 하나 (글자 key 로 물음)
struct Cand {
  char key = 'A';
  int slot = 0;        // 후보 번호 0 … K − 1
  float path_m = 0;    // 로봇 → 덩어리 가장 가까운 칸 길 거리 (F_DIST × kFrontNorm, 51.2 m 에서 자름)
  float bearing = 0;   // 그 칸 방향, 로봇 머리 기준 (rad, 왼쪽 +) — atan2(F_SIN, F_COS)
  float size_m = 0;    // 덩어리 크기 (F_SIZE × kFrontSizeNorm, 10 m 에서 자름)
  float util = 0;      // 주변 프론티어 칸 수 (F_UTIL × kUtilNorm — 후보 v2 면 예상 관측량 0 ~ 1)
  bool cur = false;    // 지금 목표가 이 덩어리 안 (F_CUR)
  int px = 0, py = 0;  // 지도 사진 안 자리 (render_png 가 채움)
  bool in_cam = false; // 카메라 그림 안에 보임 (render_cam_png 가 채움 — 그 칸 바닥 점이 시야 안이고 안 가려짐)
  int cu = 0, cv = 0;  // 카메라 그림 안 자리
  int tried = -1, abandoned = 0;   // 이 판에서 이 후보 자리(반경 안)를 목표로 삼은 횟수 · 가다가 포기한 횟수 (−1 = 안 적음 — 끝 선언을 물을 때만 채움)
};

// 로봇 지도 사진: 건드린 칸 범위(+ 로봇·후보)를 정사각형으로 잘라 px × px 로. 흰색 = 빈 곳, 검정 = 막힘, 회색 = 모름, 파란 화살표 = 로봇(머리 방향),
// 빨간 동그라미 글자 = 후보. 위가 +y. lmap 은 환경 하나의 지도 [size²] (로그 오즈 × 1000, 모름 −32768), occ_l 이상이 막힘.
// cells 는 후보 칸 (x, y) 쌍 [cand.size() · 2]. 돌려주는 값은 PNG 바이트
std::vector<uint8_t> render_png(const int16_t* lmap, int size, int gw, int gh, int16_t unknown_l, int16_t occ_l, int kb[4], float rob_cx, float rob_cy, float rob_th,
                                std::vector<Cand>& cand, const std::vector<int>& cells, int px);

// 로봇 카메라 자리에서 본 장면 그림 (진짜 RGB 아님 — 머리말). 광선 함수 ray(o, d, t, n, obj): 맞으면 true, 거리 t (d 단위)·면 법선 n·물체 번호.
// 바닥(위를 보는 면, z < 2 cm)은 모래색, 나머지는 물체 번호마다 다른 옅은 색 × 면 방향 밝기, 멀수록 어둡게. 못 맞으면 어두운 남색.
// 후보 칸의 바닥 점이 시야 안이고 안 가려지면 그 자리에 점 + 글자 꼬리표 (wxy = 후보 칸 가운데 세계 좌표 쌍)
struct CamSpec {
  int w = 320, h = 240;                       // 그림 크기 (px)
  float fx = 0, fy = 0, cx = 0, cy = 0;       // 그 크기에서의 내부값
  float mx = 0, my = 0, mz = 0, cp = 1, sp = 0;   // base_link 기준 자리 (m) · 아래로 숙인 각의 cos · sin
};
using RayFn = std::function<bool(const float o[3], const float d[3], float* t, float n[3], int* obj)>;
std::vector<uint8_t> render_cam_png(const CamSpec& C, float x, float y, float th, const RayFn& ray, std::vector<Cand>& cand, const std::vector<float>& wxy,
                                    std::vector<uint8_t>* rgb_out = nullptr);   // rgb_out 을 주면 PNG 에 넣은 픽셀(RGB, w × h)을 그대로 돌려줌 (녹화용)

// 요청 본문 (JSON). png 가 비면 사진 없이. v2 면 util 을 예상 관측량으로 적음
std::string build_body(int input, int prompt, const std::vector<Cand>& cand, const std::vector<uint8_t>& png, bool v2);
// 입력 all 의 본문: 사진 둘(카메라 그림 먼저, 지도 다음) + 글. mode choice 면 pick 질문 하나, prune 이면 앞 n_ask 후보마다 noul 질문 blk_<글자>.
// prompt 1 ~ 2 는 decider.cpp 에 (실험용)
// 끝 선언 선택지 (combine_run --decider-finish): 모델이 "탐사를 끝낼까"를 판단하는 데 쓰는 상태 — 모두 로봇이 스스로 아는 값만 (정답 지도·정답 덮음 없음)
struct ScoreCfg;
struct FinishInfo {
  int n_valid = 0;      // 지금 고를 수 있는 후보 수 (로봇 지도의 프론티어 덩어리 가운데 피하는 칸이 아닌 것)
  int n_skipped = 0;    // 지금 후보 가운데 "가다가 포기한 칸" 둘레라 뺀 수
  bool closed = false;  // 로봇 라이다 지도가 닫힘 (EnvState::map_closed — mdp/frontier.h 닫힘 규칙)
  float idle_s = 0;     // 지도가 마지막으로 늘어난 뒤 지난 시간 (EnvState::since_occ × 걸음 시간 — 로봇 지도의 새 막힘 칸 기준)
  int n_abandoned = 0;  // 이 판에서 가다가 포기한 목표 수 (막힘 신호·경로 없음으로 피하는 칸에 넣은 수)
  float t_s = 0;        // 판 시작 뒤 지난 시간
  const ScoreCfg* cfg = nullptr;   // 임무 문장 · "남은 탐사" 질문 글과 단계 (scripts/config/decider.yaml — 코드에 박지 않음)
};
// fin 을 주면 (choice 방식의 끝 선언) 같은 요청에 score 질문 하나(answers.left — "남은 탐사가 얼마나 되나")를 더한다. choice 질문은 후보만(FINISH 선택지 없음), 후보가 2 개보다 적으면 choice 질문은 뺌.
// 상태 글에 exploration_status 가 붙고 임무 문장은 fin->cfg->mission
std::string build_body_all(int mode, int prompt, const std::vector<Cand>& cand, const std::vector<uint8_t>& cam_png, const std::vector<uint8_t>& map_png, bool v2, int n_ask, const FinishInfo* fin = nullptr);

// 점수 방식 (--decider-mode score) 설정 — scripts/config/decider.yaml 에서 읽음 (글·단계 수·값을 코드에 박지 않음)
struct ScoreCfg {
  std::string mission;              // 임무 문장 (mission)
  std::string question;             // 후보마다 묻는 글 (score_question — "{id}" 자리에 후보 글자)
  std::vector<std::string> levels;  // 단계 설명, 낮은 것부터 (score_level_0, score_level_1, … — 2 ~ 10 개)
  float margin = 0;                 // score_margin: 기대 점수가 최고에서 이 안이면 비긴 것으로 봄 (지금 목표 먼저, 다음은 가까운 후보)
  float finish_below = -1;          // score_finish_below: 끝 = 모든 후보 기대 점수 < 이 값. 음수면 "모든 후보의 가장 그럴듯한 단계가 0"
  // choice 방식의 끝 선언: 같은 요청의 score 질문 하나 "남은 탐사가 얼마나 되나"
  std::string left_question;             // left_question
  std::vector<std::string> left_levels;  // left_level_0, … (낮은 것 = 남은 게 없음)
  float left_finish_below = -1;          // left_finish_below: "남은 게 없음" = 기대 점수 < 이 값. 음수면 가장 그럴듯한 단계가 0
  int left_hold = 1;                     // left_hold: 그 답이 이 수만큼 이어진 결정에서 끝 선언
};
ScoreCfg load_score_cfg(const std::string& path);
// 요청 하나에 후보마다 score 질문 하나 (answers.s_<글자>). 사진 · 상태 글은 질문들이 같이 씀
std::string build_body_score(const std::vector<Cand>& cand, const std::vector<uint8_t>& cam_png, const std::vector<uint8_t>& map_png, bool v2, const ScoreCfg& sc);

struct Reply {

  bool ok = false;
  int choice = -1;        // cand 안 번호
  float conf = 0, p = 0;  // 서버가 준 confidence, 고른 선택지 확률
  std::vector<float> probs;     // choice: 후보마다 확률 (cand 차례, 서버가 안 주면 −1)
  std::vector<float> blocked;   // prune: 앞 n_ask 후보마다 "막혔다" 확률 (프롬프트가 거꾸로 물었으면 1 − 답)
  std::vector<float> lv, exp;   // score: 후보 × 단계 확률 (cand 차례로 이어 붙임) · 후보마다 기대 점수 Σ 단계 × 확률. choice + 끝 선언: "남은 탐사" 질문의 단계 확률 · 기대 점수 하나
  double ms = 0;          // 요청을 보낸 때부터 답을 다 받은 때까지 (같이 보낸 요청과 서버를 나눠 쓴 시간 포함)
  std::string err, raw;   // 실패 사유 · 답 본문
};

// 요청 묶음을 동시에 parallel 개씩 보낸다 (libcurl multi). 실패(연결·시간 끝·HTTP 오류·답 꼴)는 ok = false 로 돌려줌 — 부른 쪽이 0 번 후보로 떨어지고 센다
class Client {
 public:
  Client(const std::string& url, int timeout_ms, int parallel);
  ~Client();
  // n_ask 가 비면 choice 답(answers.pick)을 읽고, 있으면 요청마다 앞 n_ask[q] 후보의 noul 답(answers.blk_<글자>)을 읽는다. invert 면 막힘 = 1 − 답
  // finish 면 (choice) "남은 탐사" score 답(answers.left, 단계 score_levels 개)도 읽는다 — 그때 후보가 2 개보다 적은 요청은 choice 답이 없음 (choice = −1)
  std::vector<Reply> ask(const std::vector<std::string>& bodies, const std::vector<std::vector<Cand>>& cands, const std::vector<int>& n_ask = {}, bool invert = false, bool finish = false, int score_levels = 0);   // finish 가 아니고 score_levels > 0 이면 후보마다 score 답(answers.s_<글자>)을 읽음

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};


// ── 녹화 (combine_run --video): 환경 하나의 판을 걸음마다 한 장씩 mp4 로. 화면 1280 × 720:
//   왼쪽 = 위에서 본 로봇 지도(판 내내 같은 범위) + 로봇·지나온 길·후보(점 + 글자 꼬리표)·고른 후보(초록 고리)·모델이 막혔다 한 후보(자홍 X)·정답으로 막힌 후보(주황 네모)·
//          지금 목표(파란 십자)와 계획 길(파란 선)·라이다 높이에 안 걸리는 낮은 장애물 칸(옅은 주황 — 정답, 로봇 지도엔 없음)
//   오른쪽 위 = 판단 모델이 마지막으로 받은 카메라 그림 (요청에 넣은 그 픽셀 그대로, 확대만)
//   오른쪽 아래 = 글: 판단 모델 출력(선택·확률 / 막힘 확률)과 아래층 RL 출력(v, ω 명령)
// 글은 글꼴이 필요해 ASS 자막 파일로 쓰고 ffmpeg(libass)가 얹는다: 그림은 ffmpeg 에 rawvideo 로 흘려 <prefix>.raw.mp4 → 끝나면 <prefix>.ass 를 얹어 <prefix>.mp4
struct VidCand {
  char key = 'A';
  int cx = 0, cy = 0;      // 후보 칸
  bool chosen = false, veto = false, gt_blk = false;
};
struct VidFrame {
  const int16_t* lmap = nullptr;   // 로봇 지도 [size²]
  int size = 0, gw = 0, gh = 0;
  int16_t unknown_l = 0, occ_l = 0;
  const uint8_t* low = nullptr;    // [gw · gh] 1 = 낮은 장애물 칸 (없으면 nullptr)
  float rob_cx = 0, rob_cy = 0, rob_th = 0;   // 칸 좌표
  const std::vector<float>* trail = nullptr;  // 지나온 자리 (칸 좌표 쌍)
  const std::vector<VidCand>* cand = nullptr;
  int goal_cx = -1, goal_cy = -1;
  const int16_t* wp = nullptr;     // 계획 길 웨이포인트 (칸 x, y 쌍)
  int wp_n = 0, wp_i = 0;
  const std::vector<uint8_t>* cam = nullptr;  // 지금 자세에서 본 카메라 그림 RGB (cam_w × cam_h, 걸음마다 새로), 없으면 빈 칸
  int cam_w = 0, cam_h = 0;
  const std::vector<uint8_t>* sent = nullptr; // 모델에 실제로 보낸 그림 RGB (같은 크기) — 카메라 칸 오른쪽 위에 작게. 없으면 안 그림
  bool sent_flash = false;                    // 방금 보낸 걸음들: 작은 그림 테두리를 초록으로 굵게
  std::string sent_label;                     // 작은 그림 아래 글 ("sent to model at t=…")
  std::string text;                // 오른쪽 아래 글 (줄바꿈 \n, ASS 색 꼬리표 가능)
};
class Video {
 public:
  Video(const std::string& prefix, double fps, const std::string& map_title, const std::string& cam_title);   // ffmpeg 을 연다 (없으면 예외)
  ~Video();
  void frame(const VidFrame& f);
  void hold(double sec, const std::string& text);   // 마지막 장을 sec 동안 더 (글만 바꿈) — 판 끝 사유
  void close();                                     // 자막을 얹어 <prefix>.mp4
 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

// 끝 선언 채점 (combine_run --decider-finish): ROS nav_msgs/OccupancyGrid 꼴 격자만으로 "아직 넓힐 수 있는 프론티어가 남았나"를 센다 — 정답 지도 없이, 실물 SLAM 지도에 그대로 쓸 수 있게.
//   occ: 줄 차례(row-major) [w · h], −1 = 모름, 0 … 100 = 막힘 확률 (occ_thr 이상이 막힘, 그 밖의 아는 칸은 갈 수 있음)
//   설 수 있는 칸 = 갈 수 있음이고 반지름 radius_cells(칸) 안에 막힘이 없는 칸. 로봇 칸 (rx, ry) 에서 가장 가까운 설 수 있는 칸(반경 start_cells 안)부터 4 이웃으로 번짐.
//   프론티어 칸 = 번진 칸 가운데 4 이웃에 모름이 있는 칸. 8 이웃으로 묶어 min_cells 칸 이상인 덩어리만 셈 (explore_lite min_frontier_size 와 같은 뜻).
//   판정: n_block == 0 (그리고 start ≥ 0) 이면 "닫힘 = 끝 선언이 맞음", 아니면 "이른 선언".
struct GridOpen {
  int start = -1;       // 번지기 시작 칸 (없으면 −1 — 그때는 닫힘이 아님)
  int n_reach = 0;      // 번진 칸 수
  int n_frontier = 0;   // 번진 칸 안 프론티어 칸 수 (덩어리 크기와 무관)
  int n_clusters = 0;   // min_cells 이상 덩어리 수
  int n_block = 0;      // 그 덩어리들의 칸 수 합 (0 이면 닫힘)
};
// 설정 파일(scripts/config/decider.yaml)에서 "key: 값" 한 줄을 읽음 (따옴표는 벗김). 없으면 예외
std::string config_text(const std::string& path, const std::string& key);
bool config_find(const std::string& path, const std::string& key, std::string* out);   // 없으면 false
GridOpen grid_open_frontier(const int8_t* occ, int w, int h, int rx, int ry, float radius_cells, int min_cells, int occ_thr = 65, int start_cells = 10);

void write_file(const std::string& path, const void* data, size_t n);

}  // namespace dec
