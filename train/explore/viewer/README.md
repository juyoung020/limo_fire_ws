# viewer — 탐사 실행 폴더 뷰어 (trainview)

> 탐사 강화학습은 2026-11-16 에 그만뒀다([archive/rl](../../../archive/rl/README.md)). 뷰어는 프론티어 기준선(`scripts/frontier_run.cpp`)이 쓰는 실행 폴더를 보는 데 쓴다.

실행 폴더를 브라우저로 본다. 탭 셋:

| 탭 | 보는 것 |
|---|---|
| **Run**(처음 여는 탭) | 실행 하나: 머리줄(상태 불·성공률·경과), 판 표(맵 × 맵 넓이 칸 × …, `/api/table`), 판 분포(Logged episodes, 펼칠 때만 `/api/episodes`), 평가 표(`evals/*.json` — frontier_run 은 장면마다 한 줄), 사건(`events.txt`), `run.json` |
| **Replay** | 판 하나를 2D 로(`.xrp`) |
| **Compare** | 실행 여럿의 progress 키 두 개를 나란히, 같은 group 은 평균 ± 표준편차 띠 + 끝 10 % 평균 표 |

**2026-11-18 Training 탭 지움.** 강화학습을 그만둬 PPO 학습 실행을 더 쓰지 않는다. 지운 것: 큰 카드 4 장·More stats 카드(KL·엔트로피·선언 머리·GPU·로그 버림 …), 건강 검사 9 개(보상 꼼수·이른 선언·탐색 죽음·KL …), 곡선 묶음(`chartsSpec` — 결과·탐사·보상·PPO·커리큘럼·속도), X 축·평활·Step cursor 막대(과 그것에 묶인 표·Replay 의 "near cursor" 거르기), 첫 판 안내, `Plot` 의 기준선·커서·클릭 고르기, 그 화면만 쓰던 CSS. progress 키에서는 PPO·보상·커리큘럼·자원 키를 뺐다(아래 표). 서버 경로는 하나도 안 지웠다 — 남은 탭이 모두 쓴다. 판 표·평가 표·사건·run.json 은 새 **Run** 탭으로 옮겼다(frontier_run 이 `episodes.jsonl`·`evals/frontier.json`·`events.txt` 를 쓴다).

- 서버는 Rust(std `TcpListener` + 연결마다 스레드, `serde_json`·`flate2`). **읽기 전용**: 실행 폴더의 파일만 읽는다. 화면은 순수 JS(빌드 단계 없음, 바이너리에 넣음).
- 형식의 원본은 `trainfmt/`(Rust) 한 곳. 쓰는 쪽은 C++ 헤더 `include/runfolder.h`(표준 라이브러리 + POSIX 만). 둘이 어긋나면 `runcheck` 가 잡는다.
- 가짜 실행 `fake_run`(C++)도 `runfolder.h` 로 쓴다 — 진짜 러너(`frontier_run`)와 같은 길.

## 출처

[juyoung020/object-memory-robot](https://github.com/juyoung020/object-memory-robot) (Apache-2.0, 같은 저자) 커밋 `690d0bd08558856db8a2d215af720e6af076bce8` 의 `training/viewer/` (설계: 그 저장소 `docs/map_vla/TRAIN_VIEWER.md`).

| 가져온 것 | 바꾼 것 |
|---|---|
| `src/http.rs`, `tail.rs`(이어 읽기·되감기 거르기·`max_points` 솎기), `live.rs`(SSE) | 옛 키 별칭(`keys::canon`)·sgview 역프록시 머리 뺌 |
| `src/runs.rs`(실행 찾기·줄기·학습 상태 불·latest) | 교사/학생·archive·BEHAVIOR 갈래 뺌, 상태 판정은 `ended` 먼저 |
| `src/table.rs`(표본 표) | 탐사 칸: 맵 × 단계 × 맵 넓이 칸(`area_bins`) × 끝 사유 |
| `src/replay.rs`, `main.rs` | `.trp`/`.sg` → `.xrp`(아래). 경로는 탐사에 필요한 것만 |
| `assets/charts.js`, `app.js`, `style.css`, `index.html` | 처음엔 카드·건강 검사·곡선 묶음을 탐사 PPO 로 옮겼다가 2026-11-18 에 그 Training 탭을 지움(위). Compare·판 표·평가 표만 남김 |
| `assets/replay.js` | 새로 씀: 2D 캔버스 재생 |
| `trainfmt` | `RunWriter`(Rust 쓰개)·`replay_hook`·`.trp` 를 빼고 `keys`·`xrp`(형식·검사) 만. 쓰개는 C++ `runfolder.h` |
| `src/bin/fake_run.rs` | C++ `tools/fake_run.cpp` 로 새로 씀(탐사 곡선 + 2D 시뮬로 만든 재생 판) |

지운 것: sgview 연동 Replay(`sg.rs`, `sgpanel.js`, `inputs.js`, 역프록시, three.js·로봇 GLB), OmniGibson/BEHAVIOR(`tools/og*`, `scene_mesh`, `/api/scene_mesh`), teacher/student·pipelines·DAgger·BC 갈래, FP8 감시, 재생 자동 기록(`replay_hook`, `tools/record_replay`, REC 배지), `csv2run`, `--archive-before` 보관함, `build_deps.sh`, Python 파일 전부.

## 빌드·실행

```bash
cd train/explore/viewer
cargo build --release -j 4                       # → target/release/trainview, runcheck (rustc 1.75 에서 확인)
cmake -S . -B build && cmake --build build -j 4  # → build/fake_run (C++17)

target/release/trainview --root ../runs          # 기본 포트 8095, bind 127.0.0.1 (원격은 SSH 터널 또는 --bind 0.0.0.0)
target/release/runcheck ../runs/<실험>            # 실행 폴더가 규약대로인지(오류면 끝 코드 1)
```

- `--root` 여러 개. 뿌리 밑 깊이 4 까지에서 `run.json` 또는 `progress.jsonl` 이 있는 폴더가 실행(이름 = `<뿌리 이름>/<상대 경로>`). `--root` 없으면 `train/explore/runs`.
- 기본으로 여는 실행: 뿌리의 `latest.txt`. 가짜 자료(`"synthetic": true`)는 진짜 실행이 하나라도 있으면 기본 후보가 아니고, 실행 목록에서 "Test data" 로 따로 접히며 빨간 띠가 뜬다.

### 가짜 실행으로 화면 보기

```bash
build/fake_run --out /tmp/tv/runs/fake_s1 --seed 1 --group fake            # 300 이터, 한 번에(수 초)
build/fake_run --out /tmp/tv/runs/fake_s2 --seed 2 --group fake            # 같은 group → Compare 에서 평균 ± 표준편차
build/fake_run --out /tmp/tv/runs/live_s1 --live 0.15 --iters 400          # 실행 중 흉내(초록 불 Running, SSE)
build/fake_run --out /tmp/tv/runs/live_s1 --resume 120 --iters 220         # 이터 120 체크포인트에서 이어 쓰기(되감기)
target/release/trainview --root /tmp/tv/runs
```

`train/explore/runs/` 에는 가짜 실행을 만들지 않는다(진짜 실험 자리).

## 실행 폴더 규약

```
<root>/latest.txt                                  기본으로 열 실행 이름 (러너가 시작할 때 씀)
<root>/<이름>/run.json                             설정·상수(schema 1) — 이름 바꾸기(.tmp → rename)로 씀
              progress.jsonl                       업데이트마다 → ≤ 1 줄/s 로 합침(time/iterations_merged). 모집단 — 덧붙이기
              episodes.jsonl [.1.jsonl …]          판마다 한 줄. 표본(log_envs 개 환경의 끝난 판 전부) — 덧붙이기, 1 GiB 마다 다음 번호
              events.txt                           "<시각> <글>" 사건(시작·재개·단계·체크포인트·평가) — 덧붙이기
              evals/<이름>.json                     평가 표 — 이름 바꾸기
              replays/ep_<판 6 자리>_<끝 사유>.xrp   판 하나 2D 재생 — 이름 바꾸기
              s_<줄기>/episodes_<줄기>.jsonl, s_<줄기>/replays/   집계에 섞지 않는 판(평가 = s_eval)
              replays_rewound/, evals_rewound/     재개로 되감긴 구간의 재생·평가 표(지우지 않고 비켜 둠, 뷰어는 안 읽음)
```

- **잰 것만 쓴다**: 그 줄 동안 판이 하나도 안 끝났으면 비율 키를 **뺀다**(0·null 을 쓰지 않음). `Row::mean_w(k, v, 판 수)` 가 판 수 0 이면 알아서 뺀다.
- **run.json**: `schema`, `kind`(러너가 꼭 넣음 — 없으면 `RunFolder` 가 `std::runtime_error`. frontier_run `baseline`, fake_run `fake`), `name`, `pid`, `pid_start`(`/proc/<pid>/stat` 22 번째 칸), `started`, `segments`(재개마다 `{started, pid, from_iter, from_steps, trimmed_progress_lines, trimmed_episode_lines, set_aside_files}` 덧붙임), `git`(commit·dirty), 끝나면 `ended`. 러너가 넣는 칸 중 화면이 읽는 것: `group`, `seed`, `mode`(머리줄), `baseline`·`trainer`·`n_envs`·`ray_path`(머리줄 둘째 줄), `log_envs`(있으면 판 표에 "표본" 표시), `area_bins`(표의 맵 넓이 칸 경계, m²), `robot{sim{footprint, lidar, lidar_fov, lidar_range, camera, camera_hfov, camera_range}, urdf{base_box, wheels, wheel_diameter, wheel_width, laser, camera, source} | urdf_error}`(재생의 위에서 본 로봇 — sim 충돌 몸통 실선·URDF 차체 점선·바퀴·센서 시야, 없으면 원), `synthetic`. 나머지(`params`, `scenes`, 정의 글 …)는 Run 탭 맨 아래 `run.json` 에 그대로 보인다.
- **재개(resume)**: `Options.resume = true; resume_iter = <체크포인트 이터>`. 열 때 progress·모든 줄기의 episodes 에서 `time/iterations`·`iter` 가 그 값 이상인 줄(과 해석 못 하는 반쪽 줄)을 지우고(`.tmp` 후 rename), 그 구간의 재생·평가 표를 `*_rewound/` 로 옮긴다. 판 번호는 지운 줄까지 포함한 가장 큰 번호 다음부터(파일 이름이 안 겹침), `time/time_elapsed` 는 앞 구간에 이어 센다. 뷰어도 `total_timesteps` 가 줄어드는 줄을 한 번 더 거르고(`rewound`), 파일이 새로 쓰이면 SSE `reset` 으로 화면이 처음부터 다시 받는다. (지금 쓰는 곳은 fake_run 의 되감기 시험뿐 — frontier_run 은 안 씀)
- **실행 상태 불**(목록·머리줄, SSE `status` 2 초마다): `training`(화면 글 Running) 초록 = `pid` 가 살아 있고 `/proc/<pid>/stat` 시작 시각이 `pid_start` 와 같고(pid 재사용 거름) 60 s 안에 기록, `stalled` 노랑 = 살아 있는데 60 s 넘게 기록 없음, `finished` 회색 = `ended` 있음, `crashed` 빨강 = 프로세스 없음 + `ended` 없음, `unknown` = pid 없음.

### progress 키 (`trainfmt/src/keys.rs` = `runfolder.h` `rf::key`)

이름 꼴은 Stable-Baselines3 logger(`<묶음>/<이름>`)를 따랐다. 출처: [SB3 logger](https://stable-baselines3.readthedocs.io/en/master/common/logger.html). 2026-11-18 에 frontier_run 이 쓰거나 뷰어가 읽는 키만 남겼다 — 옛 RL 실행을 `runcheck` 하면 빠진 키는 경고(모르는 키)로만 나온다.

| 묶음 | 키 |
|---|---|
| `time/` | `iterations`(필수 — frontier_run 은 장면 번호), `total_timesteps`(필수, 단조 증가), `time_elapsed`(안 넣으면 runfolder 가 넣음), `iterations_merged`(runfolder 가 넣음) |
| `rollout/` 모집단 | `n_episodes`(합 — 가중치 분모), `ep_len_mean`, `success_rate`, `collision_rate`, `truncation_rate`, `declare_rate`, `declare_short_rate` |
| `rollout/` 탐사 지표 | `coverage_mean`(판 끝 덮은 비율, 정답 맵 기준), `coverage_at_declare_mean`(선언한 판만), `t95_mean`(95 % 덮음 도달 시간 s, 도달한 판만), `t95_reached_rate`, `path_len_mean`(m), `known_area_mean`(지도에서 안 넓이 m²) |
| `eval/` | `frontier_t95_mean`(프론티어 기준선 95 % 도달 시간 — frontier_run 이 씀) |
| 그 밖 | `ts`(runfolder 가 넣음) |

뺀 키(2026-11-18): `time/fps`·`iter_ms`·`rollout_ms`·`update_ms`, `rollout/ep_rew_mean`·`step_reward_mean`·`person_collision_rate`·`no_progress_rate`, `reward/<항>`, `train/*`(PPO 손실·KL·엔트로피·σ·선언 머리 …), `eval/*`(`frontier_t95_mean` 말고 전부), `eval_reward/<항>`, `curriculum/*`, `gpu/*`, `log/*_dropped`, 그리고 `runfolder.h` 의 `reward_key`·`eval_reward_key`·`std_key`·`entropy_key`·`dropped_key`.

### episodes.jsonl 한 줄

```json
{"iter":120,"env_steps":11796480,"env":3,"ep":61,"map":"rooms_medium","stage":"medium","outcome":"success",
 "success":true,"declared":true,"collided":false,"truncated":false,"t":74.1,"steps":741,"ret":212.4,
 "r":{"coverage":215.3,"time":-1.2,"slip":-0.4,"collision":0,"declare":-1.3},"coverage":0.93,"coverage_at_declare":0.93,
 "t95":70.2,"path_len":41.5,"known_m2":131.0,"area_m2":143.0,"replay":"ep_000061_success.xrp","ts":1793717407.6}
```

- `outcome`: `success`(선언, 기준 이상) · `declared_short`(선언, 기준 미달) · `collision`(충돌, 종료) · `truncated`(안전 시간, 끊김) · `no_progress`(새 칸을 오래 못 봄, 종료). 도달 못 한 `t95`, 선언 안 한 `coverage_at_declare` 는 키를 뺀다.
- `r`(보상 항)·`ret` 는 옛 RL 판과 fake_run 만 쓴다(frontier_run 은 안 씀 — 대신 `floor_coverage`·`tail_s`·`end_reason`·`baseline{…}` 을 더 씀). 있으면 `ret` = `r` 의 합인지 서버·`runcheck` 가 확인한다.

### `.xrp` 재생 파일 (형식 원본 `trainfmt/src/xrp.rs`)

```
"XRP1"  u32 head_len  head(JSON)  [0 채움 → 8 정렬] = 데이터 시작 D, 섹션은 D + off (head.sections)
frames : n_frames × n_cols × f32     열 이름 = head.cols. 필수 t x y yaw. 선택 sx sy syaw v vy w a_v a_vy a_w p_end a_end value r_<항> ret coverage known_m2 ev h<k>_x h<k>_y h<k>_r(사람 k 원기둥, 없으면 NaN)
gt     : w × h × u8                  정답 맵 한 번: 0 빈 곳, 1 벽, 2 맵 밖
obs    : [u32 frame][i32 x0 y0 x1 y1][i8 칸…] 이어짐   관측 지도(−1 모름, 0–100 %): frame 0 전체 한 번 + 그 뒤 16×16 타일마다 바뀐 칸 사각형
cov    : [u32 frame][u32 n][u32 칸 × n] 이어짐          새로 덮인 칸(늘기만 함)
scan   : n_frames × lidar.n × u16 mm (0xFFFF 반사 없음)  선택
```

head: `meta`(그 판의 episodes 줄), `dt`, `n_frames`, `cols`, `grid{w,h,res,ox,oy}`(칸 (i,j) 가운데 = (ox+(i+½)res, oy+(j+½)res), map 좌표), `lidar{n,a0,da,rmax}`, `limits{v,vy,w}`, `reward_terms`. `ev` 비트: 1 충돌, 2 선언, 4 끊김, 8 95 % 도달. 크기(가짜 판으로 잼): 120 s 판(1200 프레임, 210 × 160 칸) 약 0.8 MB — 대부분 라이다(248 빔 × 10 Hz ≈ 0.6 MB).

## C++ 러너에서 쓰기 (frontier_run 꼴)

```cpp
#include "runfolder.h"   // -I train/explore/viewer/include, C++17

rf::Json meta = rf::Json::object();
meta.set("kind", "baseline").set("mode", mode).set("group", "frontier_" + mode).set("seed", seed).set("n_envs", N)
    .set("ctrl_hz", 10).set("success_coverage", 0.95).set("t_safe_s", 600);    // kind 는 꼭 넣는다
rf::RunFolder::Options o;
o.every_s = 0;                                                   // 줄을 합치지 않음(장면마다 한 줄)
rf::RunFolder run("train/explore/runs/frontier_" + mode, meta, o);   // run.json·latest.txt·pid·segments

for (size_t wi = 0; wi < scenes.size(); ++wi) {
  // … 장면 하나에서 판 N 개 … 판이 끝날 때마다:
  //   rf::Json line = rf::Json::object(); line.set("iter", (int)wi).set("env_steps", steps).set("ep", run.new_episode_id())
  //       .set("map", scene).set("stage", mode).set("outcome", rf::outcome::SUCCESS) /* t, steps, coverage, t95 … */;
  //   run.episode(line);
  auto& r = run.row();
  r.last(rf::key::ITER, (double)wi);  r.last(rf::key::STEPS, total_steps);
  r.sum(rf::key::N_EPISODES, n);
  r.mean_w(rf::key::SUCCESS_RATE, succ / n, n);                  // 판 수 0 → 키 빠짐
  r.mean_w(rf::key::COVERAGE_MEAN, cov / n, n);
  if (t95n > 0) r.mean_w(rf::key::T95_MEAN, t95s / t95n, t95n);
  run.end_update();
  run.event("scene " + scene + " done");
}
run.eval("frontier", ev);                                        // evals/frontier.json — Run 탭 Evaluation
run.finish();                                                    // ended → 뷰어 불 = finished
```

재생 판 하나:

```cpp
rf::ReplayWriter rw(cols /* "t","x","y","yaw",… */, 0.1, grid, gt_cells);
rw.set_lidar(248, -1.92f, 3.84f / 247, 8.0f);  rw.set_limits(1.0f, 1.0f, 3.14f);  rw.set_reward_terms(terms);
for (each step) { rw.frame(row); rw.obs(obs_grid_int8); rw.covered(cov_mask_u8); rw.scan(ranges_m); }
long long ep = run.new_episode_id();
std::string path = run.replay_path(ep, outcome /*, "eval"*/);
line.set("ep", ep).set("replay", rf::basename_of(path));
rw.finish(path, line);  run.episode(line);
```

- 한 스레드(러너의 로그 스레드)에서만 부른다. 실패는 `std::runtime_error`. 학습 고리 안에서 파일을 쓰지 않게, GPU 합계를 가져온 뒤의 로그 경로에서 부른다.
- 리틀 엔디언 기계(x86·Jetson ARM)를 전제로 float·u16 을 그대로 쓴다.

## 확인한 것 (2026-11-18, Training 탭을 지운 뒤)

| 무엇 | 결과 |
|---|---|
| 빌드 | `cargo build --release -j 4 --offline` 경고 0, `cargo test --workspace` 4 개 통과, `fake_run` g++ `-Wall -Wextra` 경고 0, `frontier_run`(baselines/frontier CMake) 빌드 — runfolder.h 에서 나는 경고 0 |
| 규약 | `fake_run` 3 실행(씨앗 1·2, 이터 120 재개 1) `runcheck` OK, progress 키 18 개 전부 표에 있음(경고 0). `frontier_run --cpu --modes nav2 --episodes 1`(small-house) 실행 폴더 `runcheck` OK(키 15 개, 경고 0) |
| API | `/` `app.js` `charts.js` `replay.js` `style.css` `/api/runs` `meta` `progress`(머리줄 키·키 목록·Compare 키) `table` `episodes` `evals` `events` `replays`(main·eval) `replay` 전부 200, `/api/live` SSE 200(`progress`·`episodes`·`replay`·`status` 이벤트) — 가짜 실행과 frontier 실행 둘 다 |
| 화면 | 옛 링크 `#tab=train` → Run 탭으로 열림. frontier 실행: 머리줄·판 표·평가 표·판 분포 4 장, Replay 판 열림, Compare 에 frontier + 가짜 group 띠·표. 콘솔 오류·경고 0 |

## 확인한 것 (2026-11-03, 가짜 실행으로 — 실제 학습 결과 아님, Training 탭이 있던 때)

| 무엇 | 결과 |
|---|---|
| 빌드 | `cargo build --release`(rustc 1.75, 경고 0), `cargo test --workspace` 4 개 통과, `fake_run` g++ 11 `-Wall -Wextra` 경고 0 |
| 규약 | `fake_run` 5 실행 모두 `runcheck` OK(진행 줄 단조, 필수 키, 키 철자, ret = Σr, 판 번호 중복 없음, `.xrp` 전부 해석·범위 검사) |
| API | `/api/runs` `meta` `progress`(`max_points` 솎기) `episodes` `table` `replays` `replay` `evals` `events` `live`(SSE) 응답, `../` 경로 거부 |
| 화면 | Training·Replay·Compare 콘솔 오류 0, 420 px 폭에서 가로 넘침 없음 |
| 실행 상태 불 | 실행 중 `training` → SIGSTOP 60 s 뒤 `stalled` → SIGKILL `crashed` → 정상 끝 `finished`. 같은 pid 에 `pid_start` 가 다르면 `crashed`(재사용 거름) |
| 재개 | 이터 60·120 에서 재개: progress·episodes 의 그 뒤 줄 지움, 재생·평가 표 `*_rewound/` 로, 판 번호 이어 셈, 화면은 SSE `reset` 으로 다시 받아 이터 단조 |

## 남은 일

- 지금 실행 폴더를 쓰는 것은 프론티어 기준선 `frontier_run` 뿐이다(RL 러너는 2026-11-16 에 그만둠).
- RL 때 남은 것 중 아직 둔 것: Replay 의 보상 항 띠·가치·선언 확률 줄(`.xrp` 의 `r_<항>`·`value`·`p_end` 열 — frontier_run 은 안 씀, fake_run 만 씀), 판 표의 `return` 칸·episodes `r`/`ret` 검사, 재개(resume) 되감기. 쓰는 곳이 fake_run 뿐이니 다음 정리 때 함께 뺄지 정한다.
- 라이다를 매 걸음 다 싣는 것이 재생 파일 크기의 대부분이다. 판이 많아지면 몇 걸음마다로 줄일지 정한다.
