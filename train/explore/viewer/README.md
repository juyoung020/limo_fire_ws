# viewer — 탐사 RL 학습 뷰어 (trainview)

학습이 도는 동안 브라우저로 본다. 탭 셋: **Training**(카드·건강 검사·곡선·표·평가), **Replay**(판 하나를 2D 로), **Compare**(실행 여럿, 같은 group 은 평균 ± 표준편차 띠).

- 서버는 Rust(std `TcpListener` + 연결마다 스레드, `serde_json`·`flate2`). **읽기 전용**: 실행 폴더의 파일만 읽는다. 화면은 순수 JS(빌드 단계 없음, 바이너리에 넣음).
- 형식의 원본은 `trainfmt/`(Rust) 한 곳. 쓰는 쪽은 C++ 헤더 `include/runfolder.h`(표준 라이브러리 + POSIX 만). 둘이 어긋나면 `runcheck` 가 잡는다.
- 가짜 실행 `fake_run`(C++)도 `runfolder.h` 로 쓴다 — 진짜 러너와 같은 길.

## 출처

[juyoung020/object-memory-robot](https://github.com/juyoung020/object-memory-robot) (Apache-2.0, 같은 저자) 커밋 `690d0bd08558856db8a2d215af720e6af076bce8` 의 `training/viewer/` (설계: 그 저장소 `docs/map_vla/TRAIN_VIEWER.md`).

| 가져온 것 | 바꾼 것 |
|---|---|
| `src/http.rs`, `tail.rs`(이어 읽기·되감기 거르기·`max_points` 솎기), `live.rs`(SSE) | 옛 키 별칭(`keys::canon`)·sgview 역프록시 머리 뺌 |
| `src/runs.rs`(실행 찾기·줄기·학습 상태 불·latest) | 교사/학생·archive·BEHAVIOR 갈래 뺌, 상태 판정은 `ended` 먼저 |
| `src/table.rs`(표본 표) | 탐사 칸: 맵 × 단계 × 맵 넓이 칸(`area_bins`) × 끝 사유 |
| `src/replay.rs`, `main.rs` | `.trp`/`.sg` → `.xrp`(아래). 경로는 탐사에 필요한 것만 |
| `assets/charts.js`, `app.js`, `style.css`, `index.html` | 카드·건강 검사·곡선 묶음(`chartsSpec`)을 탐사 과제로, 실행 종류는 `explore_ppo` 하나 |
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
build/fake_run --out /tmp/tv/runs/live_s1 --live 0.15 --iters 400          # 학습 중 흉내(초록 불, SSE)
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
- **run.json**: `schema`, `kind`(`explore_ppo`), `name`, `pid`, `pid_start`(`/proc/<pid>/stat` 22 번째 칸), `started`, `segments`(재개마다 `{started, pid, from_iter, from_steps, trimmed_progress_lines, trimmed_episode_lines, set_aside_files}` 덧붙임), `git`(commit·dirty), 끝나면 `ended`. 러너가 넣는 칸: `group`, `seed`, `n_envs`, `rollout_T`, `log_envs`, `ctrl_hz`, `gamma`, `target_kl`, `reward_terms`, `curriculum.stages[{name}]`, `refs`(곡선 기준선, 키 → 값), `success_coverage`, `t_safe_s`, `action_limits{v,vy,w}`, `robot{sim{footprint, lidar, lidar_fov, lidar_range, camera, camera_hfov, camera_range}, urdf{base_box, wheels, wheel_diameter, wheel_width, laser, camera, source} | urdf_error}`(재생의 위에서 본 로봇 — sim 충돌 몸통 실선·URDF 차체 점선·바퀴·센서 시야, 없으면 원), `area_bins`(표의 맵 넓이 칸 경계, m²), `checks`(건강 검사 문턱 덮어쓰기). 화면의 기준선·이름은 전부 여기서 읽는다.
- **재개(resume)**: `Options.resume = true; resume_iter = <체크포인트 이터>`. 열 때 progress·모든 줄기의 episodes 에서 `time/iterations`·`iter` 가 그 값 이상인 줄(과 해석 못 하는 반쪽 줄)을 지우고(`.tmp` 후 rename), 그 구간의 재생·평가 표를 `*_rewound/` 로 옮긴다. 판 번호는 지운 줄까지 포함한 가장 큰 번호 다음부터(파일 이름이 안 겹침), `time/time_elapsed` 는 앞 구간에 이어 센다. 뷰어도 `total_timesteps` 가 줄어드는 줄을 한 번 더 거르고(`rewound`), 파일이 새로 쓰이면 SSE `reset` 으로 화면이 처음부터 다시 받는다.
- **학습 상태 불**(목록·머리줄, SSE `status` 2 초마다): `training` 초록 = `pid` 가 살아 있고 `/proc/<pid>/stat` 시작 시각이 `pid_start` 와 같고(pid 재사용 거름) 60 s 안에 기록, `stalled` 노랑 = 살아 있는데 60 s 넘게 기록 없음, `finished` 회색 = `ended` 있음, `crashed` 빨강 = 프로세스 없음 + `ended` 없음, `unknown` = pid 없음.

### progress 키 (`trainfmt/src/keys.rs` = `runfolder.h` `rf::key`)

표준 RL 기록 이름(Stable-Baselines3 logger 꼴 `<묶음>/<이름>`)에 탐사 지표를 더했다. 출처: [SB3 logger](https://stable-baselines3.readthedocs.io/en/master/common/logger.html).

| 묶음 | 키 |
|---|---|
| `time/` | `iterations`(필수), `total_timesteps`(필수, 단조 증가), `time_elapsed`(안 넣으면 runfolder 가 넣음), `fps`, `iter_ms`, `rollout_ms`, `update_ms`, `iterations_merged`(runfolder 가 넣음) |
| `rollout/` 모집단 | `n_episodes`(합 — 가중치 분모), `ep_rew_mean`, `ep_len_mean`, `success_rate`, `collision_rate`, `truncation_rate`, `declare_rate`, `declare_short_rate`, `step_reward_mean` |
| `rollout/` 탐사 지표 | `coverage_mean`(판 끝 덮은 비율, 정답 맵 기준), `coverage_at_declare_mean`(선언한 판만), `t95_mean`(95 % 덮음 도달 시간 s, 도달한 판만), `t95_reached_rate`, `path_len_mean`(m), `known_area_mean`(정책 지도에서 안 넓이 m²) |
| `reward/<항>` | 판당 그 보상 항 합 평균. 항 이름은 `mdp/rewards.h` 가 정한다(아직 미정 — 무엇이든 받음). `run.json reward_terms` 에 같은 이름 |
| `train/` | `policy_gradient_loss`, `value_loss`, `entropy_loss`(= −엔트로피), `approx_kl`, `clip_fraction`, `explained_variance`, `grad_norm`, `learning_rate`, `loss`, `std`, `log_std`, `std/<v\|vy\|w>`, `entropy/<v\|vy\|w\|declare>`, `advantage_std`, `advantage_mean`, `value_mean`, `declare_logit_mean`, `declare_prob_mean` |
| `eval/` | `n_episodes`, `mean_reward`, `mean_ep_length`, `success_rate`, `collision_rate`, `truncation_rate`, `declare_short_rate`, `coverage_mean`, `coverage_at_declare_mean`, `t95_mean`, `t95_reached_rate`, `path_len_mean`, `frontier_t95_mean`(같은 맵 프론티어 기준선); `eval_reward/<항>` |
| 그 밖 | `curriculum/stage`, `curriculum/<무엇>/<단계>`, `gpu/mem_used_mb`, `log/<무엇>_dropped`(0 이어야 함), `ts`(runfolder 가 넣음) |

### episodes.jsonl 한 줄

```json
{"iter":120,"env_steps":11796480,"env":3,"ep":61,"map":"rooms_medium","stage":"medium","outcome":"success",
 "success":true,"declared":true,"collided":false,"truncated":false,"t":74.1,"steps":741,"ret":212.4,
 "r":{"coverage":215.3,"time":-1.2,"slip":-0.4,"collision":0,"declare":-1.3},"coverage":0.93,"coverage_at_declare":0.93,
 "t95":70.2,"path_len":41.5,"known_m2":131.0,"area_m2":143.0,"replay":"ep_000061_success.xrp","ts":1793717407.6}
```

- `outcome`: `success`(선언, 기준 이상) · `declared_short`(선언, 기준 미달) · `collision`(충돌, 종료) · `truncated`(안전 시간, 끊김). `ret` = `r` 의 합(서버가 확인해 어긋난 줄 수를 적는다). 도달 못 한 `t95`, 선언 안 한 `coverage_at_declare` 는 키를 뺀다.

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

## C++ 러너에서 쓰기

```cpp
#include "runfolder.h"   // -I train/explore/viewer/include, C++17

rf::Json meta = rf::Json::object();
meta.set("group", "explore_base").set("seed", seed).set("n_envs", N).set("rollout_T", T).set("log_envs", 8)
    .set("ctrl_hz", 10).set("gamma", 0.995).set("target_kl", 0.02).set("success_coverage", 0.9).set("t_safe_s", 120)
    .set("reward_terms", rf::Json::array(std::vector<std::string>{"coverage", "time", "slip", "collision", "declare"}));
rf::RunFolder::Options o;
if (resume) { o.resume = true; o.resume_iter = ckpt_iter; o.resume_steps = ckpt_steps; }
rf::RunFolder run("train/explore/runs/" + name, meta, o);       // run.json·latest.txt·pid·segments, 재개면 줄 정리

for (int it = start; it < iters; it++) {
  // … 롤아웃·갱신 …
  auto& r = run.row();
  r.last(rf::key::ITER, it);  r.last(rf::key::STEPS, steps);
  r.sum(rf::key::N_EPISODES, n_done);
  r.mean_w(rf::key::SUCCESS_RATE, n_success / std::max(1, n_done), n_done);   // n_done == 0 → 키 빠짐
  r.mean_w(rf::key::COVERAGE_MEAN, cov_mean, n_done);
  r.mean_w(rf::key::T95_MEAN, t95_mean, n_reached95);
  for (auto& [term, v] : reward_sums) r.mean_w(rf::reward_key(term), v, n_done);
  r.mean(rf::key::APPROX_KL, kl);  r.mean(rf::key::ENTROPY_LOSS, -entropy);
  run.end_update();                                              // 1 초 지났으면 한 줄
  for (auto& e : finished_sample_episodes) {                     // log_envs 환경에서 끝난 판 전부
    rf::Json ep = rf::Json::object();
    ep.set("iter", it).set("env_steps", steps).set("ep", run.new_episode_id()).set("outcome", rf::outcome::SUCCESS) /* … */;
    run.episode(ep);                                             // 평가 판은 run.episode(ep, "eval")
  }
  if (it % ckpt_every == 0) run.event("checkpoint ckpt_" + std::to_string(it) + ".bin");
}
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

## 확인한 것 (2026-11-03, 가짜 실행으로 — 실제 학습 결과 아님)

| 무엇 | 결과 |
|---|---|
| 빌드 | `cargo build --release`(rustc 1.75, 경고 0), `cargo test --workspace` 4 개 통과, `fake_run` g++ 11 `-Wall -Wextra` 경고 0 |
| 규약 | `fake_run` 5 실행 모두 `runcheck` OK(진행 줄 단조, 필수 키, 키 철자, ret = Σr, 판 번호 중복 없음, `.xrp` 전부 해석·범위 검사) |
| API | `/api/runs` `meta` `progress`(`max_points` 솎기) `episodes` `table` `replays` `replay` `evals` `events` `live`(SSE) 응답, `../` 경로 거부 |
| 화면 | Training·Replay·Compare 콘솔 오류 0, 420 px 폭에서 가로 넘침 없음 |
| 학습 상태 불 | 학습 중 `training` → SIGSTOP 60 s 뒤 `stalled` → SIGKILL `crashed` → 정상 끝 `finished`. 같은 pid 에 `pid_start` 가 다르면 `crashed`(재사용 거름) |
| 재개 | 이터 60·120 에서 재개: progress·episodes 의 그 뒤 줄 지움, 재생·평가 표 `*_rewound/` 로, 판 번호 이어 셈, 화면은 SSE `reset` 으로 다시 받아 이터 단조 |

## 남은 일

- 진짜 러너(`ppo/runner.cpp`)가 아직 없다. 보상 항 이름·덮음 정의·성공 기준·안전 시간은 `mdp` 에서 정해지면 run.json 에 싣는다(뷰어 코드는 안 바뀜).
- 프론티어 기준선(`eval/frontier_t95_mean`, 평가 표 `ref_t95`)은 기준선 코드가 생기면 러너가 채운다.
- 라이다를 매 걸음 다 싣는 것이 재생 파일 크기의 대부분이다. 판이 많아지면 몇 걸음마다로 줄일지 정한다.
