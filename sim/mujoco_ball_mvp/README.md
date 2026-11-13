# MuJoCo 공 발사 → 화재 모형 전도 MVP

고정 힌지 표적에 공을 발사하는 물리 시뮬레이션이다. **힌지 각도의 절댓값이 45° 이상인 상태를 0.20초 연속 유지**해야 성공한다. 판정 기준은 원본 그대로이며 충돌만으로 성공 처리하지 않는다. 실제 장치 제어, ROS 2/Nav2, MPC는 포함하지 않는다.`r`n`r`n이 폴더는 개인 저장소 [`choejunhyeok2005/robot-programming`](https://github.com/choejunhyeok2005/robot-programming/tree/2c905d3/simulation/mujoco_ball_mvp)의 `main` 커밋 `2c905d3`에서 복사한 독립 데모 스냅샷이다. MuJoCo 장면에는 LIMO 시각 메시가 있지만 이동 베이스·센서·ROS 2 브리지는 없으므로, 여기의 데모는 팀의 Nav2·탐지·임무 BT 통합 결과가 아니다. 변경은 양쪽 저장소에 각각 반영해야 한다.

## 실행 환경과 사용법

2026-11-09 실제 검증: Windows 11, CPython 3.12.14 (64-bit), MuJoCo 3.17.0. 정확한 의존성은 [requirements-lock.txt](experiments/2026-11-09/requirements-lock.txt)에 저장했다. 저장소 루트에서:

```powershell
cd sim/mujoco_ball_mvp
# py가 없으면 설치된 Windows CPython 실행 파일의 절대 경로를 사용한다.
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r experiments/2026-11-09/requirements-lock.txt
.\.venv\Scripts\python.exe launch.py
```

이 PC에서는 기본 `python`이 MSYS2 Python이고 기존 사용자 venv도 실행 불가능하여 프로젝트 전용 `.venv`를 만들었다. 최초 생성에는 `C:\Users\choej\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`를 사용했다. 이후에는 위 `.venv\Scripts\python.exe` 명령을 사용한다.

```powershell
# 기본 성공 조건, GUI 없이 시계열과 요약 저장
.\.venv\Scripts\python.exe launch.py --headless --csv output/trace.csv --summary output/summary.json
# 원래 물리 조건의 실패 재현
.\.venv\Scripts\python.exe launch.py --headless --pivot-height 0.02 --duration 3
# GUI: 6초 시뮬레이션 후 최종 화면 5초 표시하고 자동 종료
.\.venv\Scripts\python.exe launch.py --auto-close --view-hold 5
# 결과 보존을 위해 재실행은 별도 폴더 사용
.\.venv\Scripts\python.exe experiments.py --output experiments/local-rerun
.\.venv\Scripts\python.exe verify.py --output experiments/local-verification
# 실제 GUI 자동 종료 검사 포함 (데스크톱/그래픽 드라이버 필요)
.\.venv\Scripts\python.exe verify.py --gui --output experiments/local-gui-verification
```

기본 발사값은 속도 5m/s, Pitch 20°, Yaw 0°, 표적 피벗 x=1.25m이다. 공 시작 x=0.05m이므로 `--distance`는 공에서 표적까지의 순수 거리와 다르다. x는 0.7~3.0m 범위다. 기존 `--speed`, `--pitch`, `--yaw`, `--distance`, `--duration`, `--csv`를 유지했다. `--pivot-height`, `--target-mass`, `--damping`, `--frictionloss`, `--timestep`은 비교 실험용이다. 질량은 XML 컴파일 전에 변경하여 형상에 맞는 관성도 함께 계산한다.

## 원본 재현과 실패 원인

원본 커밋: `6cfdb08d5164c694605235083d3f11b7942d80de`. [baseline](experiments/2026-11-09/baseline/)에 원본 `launch.py`, `scene.xml`, 재현 스크립트, 오류 로그, import 수정본과 CSV를 보존했다.

원본 실행은 `main()` 내부 `import mujoco.viewer`가 `mujoco`를 지역 변수로 만들어 `UnboundLocalError`로 중단됐다. 먼저 이 import만 수정하고 원본 물리값·3초 실행을 유지하여 다음 결과를 실제 재현했다:

| 조건 | 공 접촉 | 최대 기울기 | 최종 기울기 | 전도 |
|---|---:|---:|---:|---|
| 원본 물리값, 힌지 0.02m | 0.230초(원본 표기) | 19.415° | 15.529° | 실패 |
| 수정 기본값, 힌지 0.08m | 0.228초 | 94.331° | 90.629° | **0.532초에 성공** |

주원인은 **힌지와 바닥의 기하학적 간섭**이다. 표적은 0.15 × 0.18 × 0.46m이며 바닥 쪽 모서리의 회전 반경은 0.075m인데 피벗 높이가 0.02m였다. 0~90° 구간에서 가장 낮은 모서리 높이는 `h - 0.075 sin(θ)`이므로 `asin(0.02/0.075) = 15.466°`에서 바닥에 걸린다. 실제 표적-바닥 첫 접촉은 0.270초, 정착 각도는 15.529°였다. 순간 최대 19.415°는 MuJoCo의 부드러운 접촉 모델에서 관측된 침투/반동이며 자유 회전이 가능하다는 뜻이 아니다.

수치와 비교 실험은 [diagnostic.json](experiments/2026-11-09/baseline/diagnostic.json), [ablations.csv](experiments/2026-11-09/ablations.csv)에 있다.

- 공 질량 0.10kg, 반지름 0.055m, 발사 에너지 1.25J. 충돌 좌표는 약 `(1.175625, 0, 0.432685)m`, 충돌 직전 속도는 `(4.698463, 0, -0.526579)m/s`다. 기본 Pitch에서는 하강 중 표적 상부에 맞는다.
- 표적 질량 0.14kg, 질량중심의 y축 관성은 0.00273117kg·m², 힌지 기준 관성은 평행축 정리에 따라 약 0.01013717kg·m²다. 피벗을 올려도 질량·관성은 유지된다. 수정 후 충돌점은 피벗보다 약 0.353m 높다.
- 원래 힌지에서 속도 3/4/6m/s, 질량 0.07/0.28kg, 감쇠 0/0.08N·m·s/rad, 마찰손실 0/0.01N·m를 각각 변경해도 성공하지 않았다. 접촉한 경우 최대각은 약 16~21°였다. 속도 8m/s 또는 Pitch 30/40°는 기본 거리에서 표적 위로 빗나갔다.
- **기본 물리 변경은 힌지 높이 0.02 → 0.08m 하나다.** 반두께 0.075m보다 5mm 높아 90°까지 바닥 간섭 없이 회전하는 고정 축 받침을 가정한다. 받침 세부 형상은 모델링하지 않았다. 바닥 충돌, 표적 형상/질량, 공, 힌지 감쇠 0.02와 마찰손실 0.002는 유지했다.
- 높이 0.04m는 실패, 0.06m는 45° 기준을 만족하지만 바닥에 막혀 최대 약 58°였다. 완전히 눕는 동작을 위해 0.08m를 택했다. 실제 장치가 반드시 2cm 피벗이어야 한다면 바닥 여유/힌지 배치를 별도로 재설계해야 한다.

이 표적은 질량중심이 힌지 위에 있는 이상적인 역진자다. 중력이 전도를 돕고 래치/복원 스프링/자유 지지면은 없다. 실물 표적의 충격 임계값이나 현장 성공률을 예측하는 모델은 아니다.

## 반복 실험

모든 아래 실험은 실제 MuJoCo 엔진으로 실행했다. [runs.log](experiments/2026-11-09/runs.log), [환경 및 집계](experiments/2026-11-09/environment.json)에 기록했다.

| 실험 | 결과 |
|---|---|
| 거리 0.8/1.25/1.75/2.25m × 속도 4/5/6m/s × Pitch 0/10/20/30/40° × 3회 | **141/180 성공, 78.33%** |
| 위 실험 중 바닥 선접촉 없는 성공 | **99/180, 55.00%** |
| 공이 먼저 바닥을 맞은 뒤 표적을 쓰러뜨린 경우 | **42/180, 23.33%** |
| 나머지 | 39/180, 표적 미접촉으로 실패 |
| 기본값 주변 속도 ±0.25m/s, Pitch ±2°, Yaw ±1°, 거리 ±0.05m 균등 오차, seed=20261109 | **100/100 성공** |
| timestep 2/1/0.5ms | 모두 성공, 판정 0.532/0.531/0.531초 |
| 무발사 / Yaw 65° 빗나감 / 실행 0.3초 | 모두 실패 |

동일 조건 3회는 결정론적 반복 재현 확인이다. 78.33%는 선택한 격자의 비율이며 현장 신뢰도가 아니다. 100회 오차 실험도 지정한 오차 범위에만 해당한다. 바닥 후 충돌도 원래 판정상 성공이지만 직접 발사 성능과 구별해야 한다.

- [grid.csv](experiments/2026-11-09/grid.csv): 180회 각각의 성공 여부, 충돌 시각, 최대각, 성공 판정 시각, 물리 파라미터, 바닥 선접촉 여부.
- [success_rates.csv](experiments/2026-11-09/success_rates.csv): 조건별 성공률, 직접 성공/바닥 선접촉 후 성공 횟수.
- [perturbations.csv](experiments/2026-11-09/perturbations.csv): 100회 오차 실험의 실제 입력과 결과.
- [checks.csv](experiments/2026-11-09/checks.csv), [traces](experiments/2026-11-09/traces/): timestep/음성 대조 및 대표 성공·실패 시계열. 모든 격자의 시계열을 중복 저장하지 않고 각 실행의 지표를 저장했다.
- [tests.log](experiments/2026-11-09/tests.log): 6개 회귀 테스트 통과. CSV 각도로 연속 0.2초 조건을 독립 검사했다. 45°를 넘었어도 유지 시간이 부족한 0.45초 실행은 실패인지, 질량 변경 시 관성도 변하는지 검증했다.

CSV의 `time_s`는 적분 후 상태 시각이다. MuJoCo `mj_step`의 접촉은 해당 적분 시작 시점의 값이므로 `contact_sample_time_s`와 요약의 접촉 시각은 적분 전 시각이다. 원본 접촉 표시 0.230초가 현재 0.228초로 바뀐 것은 궤적 변경이 아니다. 성공 판정은 적분 후 각도를 사용한다. 시각은 timestep 단위이며 부동소수점 때문에 duration보다 한 스텝 길 수 있다. 접촉/성공이 없으면 해당 CSV 시각은 빈 값이다.

기본 접촉 모델의 유연성 때문에 수정 표적의 바닥 착지 시 최대 침투 약 29.5mm가 관측됐다. 0.5ms에서도 약 28.5mm가 남는다. 정착각은 약 90.6°이고 45° 통과는 착지 전에 일어난다. **성공은 timestep을 줄여도 유지되지만 최대 착지각/침투/힘은 정밀한 실물값으로 해석하지 않는다.**

## GUI와 종료 조건

- 원본은 3초 뒤 즉시 닫혔다. 현재 기본 실행은 **6초**, 계산 후 로그/CSV/JSON을 저장하고 **사용자가 창을 닫을 때까지 최종 화면을 유지**한다. 유지 중에는 물리를 적분하지 않는다.
- `--auto-close --view-hold 5`는 계산 후 5초 유지하고 닫는다. `--view-hold`는 자동 종료에만 적용된다. headless는 duration까지 계산한 후 즉시 끝난다.
- `--hide-panels`는 passive GUI를 띄울 때 좌우 MuJoCo 패널을 처음부터 숨긴다.
- GUI는 약 60Hz로 갱신하며 경과 시간에 맞춰 물리 스텝을 수행한다. 계산 중 창을 닫으면 `completed=false`, `termination=viewer_closed_early`가 기록된다. 이 경로는 코드로 확인했으며 수동 창 닫기 조작은 실제 검증하지 못했다.
- 실제 기본 GUI의 [gui.json](experiments/2026-11-09/gui.json)과 [로그](experiments/2026-11-09/gui.log)에서 성공 및 계산 완료를 확인했다. 6초 뒤에도 MuJoCo 창이 남아 있음을 창 목록으로 확인했다.
- [자동 종료 검증](experiments/2026-11-09/gui_verification.json): 정상 종료 코드 0, 전체 경과 약 12.41초, 시뮬레이션 6초 및 최종 화면 유지 5초. **computer-use의 Python 창 접근이 자동 승인 검토에서 거절되어 스크린샷을 통한 시각 검수는 하지 못했다.**

## PR 준비 검토 (2026-11-09)

검토 후 `verify.py`의 출력 경로를 호출 위치 기준 절대 경로로 바꿨다. 이전에는 저장소 루트에서 상대 `--output`으로 GUI 검증을 실행하면 자식 프로세스의 작업 디렉터리와 달라 결과 파일을 찾지 못할 수 있었다. GUI 검증 기록에 고정되어 있던 과거 접근 거절 사유도 제거했다. 자동 검증은 렌더링된 화면을 검사하지 않는다고 명시한다.

- [테스트 로그](experiments/pr-review-2026-11-09/tests.log): **7개 통과**. 가짜 뷰어로 실행 도중 창이 닫히는 상황을 만들어 `viewer_closed_early`, 미완료 상태, CSV/JSON 저장을 검증했다. 실제 수동 창 닫기/시각 검수와는 구분한다.
- [GUI 검증](experiments/pr-review-2026-11-09/gui_verification.json): 저장소 루트에서 상대 출력 경로를 사용하여 실제 실행, 정상 종료, 약 12.50초 경과를 확인했다.
- [실험 재현 비교](experiments/pr-review-2026-11-09/reproduction.json): 별도 임시 폴더에서 전체 실험을 재실행하고 제출 CSV와 SHA-256을 비교했다. 원래 실험 증거는 덮어쓰지 않았다.

실제 장치에 8cm 고정 힌지를 적용할 수 있는지, 착지 시 접촉 유연성이 허용되는지, GUI 화면이 보기 좋은지는 여전히 리뷰 대상이다. 이번 검토에서 발사 기본값, 물리 모델, 45°·0.20초 판정은 변경하지 않았다.

## 변경 파일

- `scene.xml`: 힌지 높이와 설명 주석만 변경.
- `launch.py`: import 오류 수정, 기존 발사/판정 보존, 파라미터 옵션, 접촉/최대각/완료 상태 기록, GUI 유지/자동 종료.
- `experiments.py`, `test_physics.py`, `verify.py`: 반복 실험, 회귀/독립 판정 검사, 실제 GUI 자동 종료 검증.
- `.gitignore`: venv와 Python 캐시 제외.
- `experiments/2026-11-09/`: 원본과 실제 실험 증거. 재실행은 `--output`으로 별도 폴더를 권장한다.

작업 브랜치: `feat/mujoco-topple-validation`. main에 직접 변경을 올리지 않는다.

[MuJoCo 적분/접촉 설명](https://github.com/google-deepmind/mujoco/blob/main/doc/programming/simulation.rst) · [공식 passive viewer 설명](https://github.com/google-deepmind/mujoco/blob/main/doc/python.rst) · [팀 시뮬레이션 구조](../../docs/simulation.md)


## CAD launcher scene (2026-11-09)

The original `scene.xml` remains the marker-only baseline. `scene_cad.xml` adds four visual-only meshes from the referenced XRP ball-shooter STEP assembly; the CAD ball is represented by the MuJoCo sphere so the existing launch and collision detector operate on the exact 40 mm, 2.7 g projectile. Run the reproducible headless example from this directory:

```powershell
.\.venv\Scripts\python.exe launch.py --headless --scene scene_cad.xml --speed 4.2 --pitch 30 --duration 6 --csv experiments/2026-11-09/cad-demo/trace.csv --summary experiments/2026-11-09/cad-demo/summary.json
```

Verified with MuJoCo 3.17.0: contact at 0.304 s; topple confirmed at 1.202 s after the target remained at least 45 degrees tilted for 0.20 s; maximum tilt 92.913 degrees; final tilt 90.629 degrees. The exact run output is in [run.log](experiments/2026-11-09/cad-demo/run.log), [summary.json](experiments/2026-11-09/cad-demo/summary.json), and [trace.csv](experiments/2026-11-09/cad-demo/trace.csv).

The meshes are display geometry only (`contype=0`, `conaffinity=0`). The launcher is fixed in the scene: flywheel rotation, ball feeding, motor forces, and launcher recoil are not modeled. The simulation initializes the ball at the CAD outlet and applies its initial velocity directly, so this confirms the projectile leaves the muzzle location in the simulation; it does not validate the physical flywheel launcher. GUI runs can use the same scene with `--scene scene_cad.xml`; the existing six-second duration and final-view hold behavior remain in effect.

## Elastic launcher and flame target (2026-11-11)

Run the new mechanism sequence from this directory:

```powershell
.\.venv\Scripts\python.exe launch.py --elastic-launcher --csv experiments/elastic-launcher/trace.csv --summary experiments/elastic-launcher/summary.json
```

The GUI shows the sequence and runs for 12 simulated seconds by default so the target's return is visible. Add `--headless` to run without the viewer.

To see a single-window live aiming demo, run `launch.py --elastic-launcher --aim-demo --yaw 0 --pitch 15`. The LIMO rotates from the deliberately offset starting yaw (`--aim-start-yaw`, default 20 degrees), the launcher elevates from `--aim-start-pitch` (default 5 degrees), and the held ball follows the muzzle. It then runs the launcher sequence and fires. `--aim-time` controls how long the smooth aim adjustment takes.

This preset uses the attached design's 40 mm, 2.7 g ball; 15-degree elevation; nominal 4.20 m/s launch speed (the three documented 15-degree simulation samples are 4.26, 4.06, and 4.28 m/s); and a 1.275-second mechanism cycle. The visible sequence animates DS3218 lever +66 degrees, 0.60 seconds loaded, return to rest over 0.33 seconds, 0.10 seconds for sear capture, release to -18 degrees, then the documented 0.055-second ball-exit lag. The target uses the provided flame and base STL visuals and a collision panel at the STL's measured 82.7 x 169.1 x 6 mm envelope, with its broad face aligned toward the launcher.

The scene uses x-forward, y-lateral, z-up coordinates: `--yaw` turns the LIMO chassis and mounted launcher together around vertical z, while `--pitch` tilts the launcher around lateral y. Pitch transforms both the muzzle position and the ball's launch vector.

The launcher body, cradle, hopper, cap, plunger, lever, and sear use the supplied print STL meshes, positioned from the documented millimeter dimensions. The LIMO chassis and four wheels use the visual meshes and poses from AgileX Robotics' official [limo_ros2](https://github.com/agilexrobotics/limo_ros2/tree/humble/limo_description) description; they are static visual geometry. The elastic-band/plunger forces are not dynamically solved. The ball is held at the muzzle during the animation, then receives the documented MuJoCo launch speed at release. The target's 40 g moving mass is an initial estimate: 30 g is assigned to the return weight and 10 g to the panel. The weight center is 28.3 mm below the hinge and hinge damping/friction are set low so the panel can tip and return. These values need replacement with measurements of the assembled target. MuJoCo collides against a rectangular panel proxy rather than the concave flame silhouette.

The retained nominal run contacts the target at 1.574 s, reaches 114.7 degrees, passes the existing 45-degree / 0.20-second success condition at 1.922 s, and is back within 5.6 degrees of upright after 12 s. Ball-to-target contact is normal-only so the thin panel does not impart an artificial sideways kick. These are outputs from the estimated design model, not measured hardware results.

CAD source: [XRP ball shooter, CAD reference](https://app.notion.com/p/CAD-40f454b08de281f59782e2f8e31add9a). The source model is attributed to Tyler Jacques on Printables (model 951454), licensed CC BY 4.0. The OBJ files in `cad_assets/` are tessellated visual derivatives of the named STEP assembly components.


GUI verification (2026-11-09): launched `scene_cad.xml` with `--duration 6 --auto-close --view-hold 5`. MuJoCo completed the full six simulated seconds, recorded contact and the 45-degree / 0.20-second topple success, then exited after its five-second final-view hold. The GUI run's [summary](experiments/2026-11-09/cad-demo/gui-summary.json) reports `completed: true` and `termination: duration`; its [CSV trace](experiments/2026-11-09/cad-demo/gui-trace.csv) is retained separately from the headless trace.

## CAD-size ball, distance sweep and repeatability (2026-11-09)

The original `experiments/2026-11-09/` results above use the older 100 g sphere. The following new experiments explicitly use `scene_cad.xml`, with the 40 mm, 2.7 g ball. The original scene and its historical result files are unchanged.

The current regression suite has 9 passing tests, including a CAD-scene check for projectile dimensions, visual-only launcher meshes, direct contact, optional aerodynamic drag, and the unchanged 45-degree / 0.20-second hold. Its [test log](experiments/2026-11-09/cad-launcher-grid/regression/tests.log) is saved with the experiment results.

Re-run the 150-condition sweep and then the selected operating-point repeatability run:

```powershell
.\.venv\Scripts\python.exe experiment_cad_launcher.py --output experiments/2026-11-09/cad-launcher-grid
.\.venv\Scripts\python.exe verify_cad_operating_points.py --output experiments/2026-11-09/cad-launcher-grid --repeats 30
```

The grid covered 6 target-hinge x coordinates (0.8, 1.25, 1.75, 2.25, 2.75, 3.0 m), 5 speeds (3, 4, 4.2, 5, 6 m/s) and 5 pitches (0, 10, 20, 30, 40 degrees), with 3 repeats per condition and small seeded setup jitter. The CLI's `--distance` is the target pivot's x coordinate; the ball starts at x=0.05 m. The grid recorded **141 contacts and 124 topple successes in 450 trials (27.56%)**. Every successful case was a direct ball-to-target hit before the ball touched the ground. This rate includes deliberately poor settings and misses; it is not the success probability of the recommended settings.

Seven selected operating points were then run 30 times each with the same small seeded jitter, for **209 topple successes in 210 trials (99.52%)**. At the 3.0 m CLI limit, target-position jitter is clipped to the valid range:

| Target pivot x | Speed | Pitch | Successes | Median contact | Median confirmation |
|---:|---:|---:|---:|---:|---:|
| 0.80 m | 4.2 m/s | 20° | 30/30 | 0.166 s | 1.086 s |
| 1.25 m | 4.2 m/s | 30° | 30/30 | 0.304 s | 1.201 s |
| 1.75 m | 5.0 m/s | 30° | 30/30 | 0.376 s | 1.227 s |
| 2.25 m | 6.0 m/s | 20° | 30/30 | 0.376 s | 1.248 s |
| 2.75 m | 6.0 m/s | 30° | 29/30 | 0.510 s | 1.520 s |
| 2.90 m | 6.0 m/s | 30° | 30/30 | 0.535 s | 1.355 s |
| 3.00 m | 6.0 m/s | 30° | 30/30 | 0.551 s | 1.394 s |

The single failure at 2.75 m was a miss with no contact, not a relaxed or bypassed topple test. Detailed actual inputs, contact times, maximum tilts and success-confirmation times are in [operating_point_trials.csv](experiments/2026-11-09/cad-launcher-grid/operating_point_trials.csv). The full grid is in [trials.csv](experiments/2026-11-09/cad-launcher-grid/trials.csv), with [condition rates](experiments/2026-11-09/cad-launcher-grid/success_rates.csv) and [run summary](experiments/2026-11-09/cad-launcher-grid/summary.json).

A wider setup-error stress pass used ±0.10 m target-position, ±0.30 m/s speed, ±2° Pitch and ±2° Yaw variation (30 repeats at each selected pivot coordinate): **196/210 toppled (93.33%)**, with 197 contacts. Reproduce it with:

```powershell
.\.venv\Scripts\python.exe verify_cad_operating_points.py --output experiments/2026-11-09/cad-launcher-grid/stress --repeats 30 --seed 20261112 --distance-jitter 0.10 --speed-jitter 0.30 --pitch-jitter 2 --yaw-jitter 2
```

The larger variations reduced reliability mainly through misses: all 30 runs succeeded at 0.8, 1.25 and 2.25 m; results were 27/30 at 1.75 m, 22/30 at 2.75 m, 28/30 at 2.90 m and 29/30 at the 3.0 m limit. At 2.75 m there were seven misses and one hit that did not topple the target. Full [stress trials](experiments/2026-11-09/cad-launcher-grid/stress/operating_point_trials.csv) and [summary](experiments/2026-11-09/cad-launcher-grid/stress/operating_points.json) retain each perturbed input and result. This is a bounded simulation stress test, not measured physical launch error.

## Reproducible mobile-search video

The generated [21.57-second, 30 fps video](experiments/2026-11-09/cad-launcher-grid/mobile-demo/mobile_search_aim_fire.mp4) is committed with the [mission path](experiments/2026-11-09/cad-launcher-grid/mobile-demo/mission.csv), [events](experiments/2026-11-09/cad-launcher-grid/mobile-demo/events.json), [shot trace](experiments/2026-11-09/cad-launcher-grid/mobile-demo/shot.csv) and [recording metadata](experiments/2026-11-09/cad-launcher-grid/mobile-demo/recording.json). The launcher, rover stop pose and actual flight trajectory now share the same coordinates; the orange ball trail is rendered from saved MuJoCo states. Flight is shown at 0.1× playback for visibility.

To regenerate it on this PC, install the pinned rendering dependencies and run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-video.txt
.\.venv\Scripts\python.exe record_mobile_cad_demo.py
```

The script writes the MP4 and its raw artifacts to `%USERPROFILE%\Videos\MuJoCo\mobile_demo`. The moving rover still follows a prescribed kinematic route and the detector uses the target's true position and a simulated field of view; this is a visual demo, not autonomous navigation or camera recognition. CAD geometry is display-only, and the projectile receives an initial velocity at the muzzle rather than being accelerated by simulated flywheels.

## Optional aerodynamic drag sensitivity (2026-11-09)

The legacy runs above use vacuum flight (`--drag-coefficient 0`), preserving the original reproducible baseline. `launch.py` now accepts `--drag-coefficient` and `--air-density`; for a nonzero coefficient it applies quadratic drag to the 40 mm CAD ball from its instantaneous translational velocity. The detector remains unchanged: the target must stay at least 45 degrees tilted for 0.20 seconds. This model currently omits ball spin, Magnus lift, and changing drag coefficient.

A 2020 table-tennis-ball trajectory study reports that neglecting air resistance can produce substantial trajectory error ([Flight and bounce of spinning sports balls](https://orbit.dtu.dk/en/publications/flight-and-bounce-of-spinning-sports-balls/), DOI 10.1119/10.0001659). A 2022 CFD study describes the quadratic drag formulation and air-property assumptions ([J-STAGE paper PDF](https://www.jstage.jst.go.jp/article/tjsst/14/1/14_36/_pdf/-char/en)). Its coefficient results cover faster 10?40 m/s and spinning-ball conditions, so this project's constant `Cd=0.45` at 4?10 m/s is explicitly a sensitivity assumption, not a coefficient calibrated to this launcher or ball.

Reproduce the single-shot comparison and drag-aware experiments with:

```powershell
.\.venv\Scripts\python.exe launch.py --headless --scene scene_cad.xml --speed 4.2 --pitch 30 --duration 6 --drag-coefficient 0.45 --air-density 1.2044 --csv experiments/2026-11-09/cad-demo/drag-045.csv --summary experiments/2026-11-09/cad-demo/drag-045.json
.\.venv\Scripts\python.exe experiment_cad_launcher.py --output experiments/2026-11-09/cad-launcher-grid/drag-cd045 --drag-coefficient 0.45 --air-density 1.2044
.\.venv\Scripts\python.exe verify_cad_operating_points.py --output experiments/2026-11-09/cad-launcher-grid/drag-cd045/operating-points --repeats 30 --seed 20261113 --drag-coefficient 0.45 --air-density 1.2044 --operating-point 0.8 4 20 --operating-point 1.25 4 30 --operating-point 1.75 6 20 --operating-point 2.25 7 20 --operating-point 2.75 8 20 --operating-point 3.0 8 20
```

At the 1.25 m, 4.2 m/s, 30-degree baseline, adding this drag assumption changed contact from 0.304 s to 0.328 s and topple confirmation from 1.202 s to 1.298 s; the target still toppled. The extended 630-trial grid produced 171 topple successes (27.14%) across deliberately varied settings. Six retuned points reached 180/180 with small setup jitter. Under the wider +/-0.10 m position, +/-0.30 m/s speed, +/-2-degree pitch and yaw perturbations, those same six points achieved **178/180 successes (98.89%)**; there were 179 contacts. The two failures were one miss and one contact that did not satisfy the topple rule. Per-trial input, contact time, maximum tilt, and confirmation time are in [wide-jitter CSV](experiments/2026-11-09/cad-launcher-grid/drag-cd045/stress/operating_point_trials.csv); the [summary JSON](experiments/2026-11-09/cad-launcher-grid/drag-cd045/stress/operating_points.json) records the seed, model assumptions and success totals. The 450-trial and extended 630-trial grids, including all misses, are retained under [drag-cd045](experiments/2026-11-09/cad-launcher-grid/drag-cd045/) and [drag-cd045-extended](experiments/2026-11-09/cad-launcher-grid/drag-cd045-extended/).

The mobile-search video can now be regenerated with the same drag-aware physics used by these experiments:

```powershell
.\.venv\Scripts\python.exe record_mobile_cad_demo.py --drag-coefficient 0.45 --air-density 1.2044
```

The video HUD and metadata state the selected coefficient. Set `--drag-coefficient 0` to recreate the vacuum baseline. This remains a constant-coefficient, unspun-ball sensitivity demonstration; it does not validate the real launcher's flywheel speed or the physical ball's aerodynamic coefficients.

To compare the coefficient assumption with identical seeded setup perturbations across all cases, run:

```powershell
.\.venv\Scripts\python.exe experiment_drag_sensitivity.py --output experiments/2026-11-09/cad-launcher-grid/drag-sensitivity --repeats 30 --seed 20261115 --coefficients 0 0.2 0.35 0.45 0.6
```

At the baseline 1.25 m / 4.2 m/s / 30-degree operating point, this paired 150-trial study had 30/30 successes at every tested Cd. Median contact time increased from 0.308 s at Cd=0 to 0.340 s at Cd=0.6; median topple confirmation increased from 1.209 s to 1.343 s. The [per-trial CSV](experiments/2026-11-09/cad-launcher-grid/drag-sensitivity/trials.csv) and [summary](experiments/2026-11-09/cad-launcher-grid/drag-sensitivity/summary.json) preserve the shared perturbations and measurements.

## LIMO search-to-fire demo with the elastic launcher (2026-11-11)

`record_mobile_cad_demo.py` now builds its scene from `scene_elastic_launcher.xml`, so the moving rover is the AgileX LIMO visual model with the provided printed launcher and flame target. The patrol, target cue, approach, stop, aim, launcher sequence, shot, and topple are rendered in one video. After stopping, the LIMO body yaws toward the target and the launcher pitches from -5 to 15 degrees; the ball stays at the muzzle until release. The shot reuses the elastic-launcher MuJoCo physics and the existing 45-degree / 0.20-second topple rule.

```powershell
.\.venv\Scripts\python.exe record_mobile_cad_demo.py --output experiments/elastic-launcher/mobile-demo
```

The generated [22.27-second, 30 fps video](experiments/elastic-launcher/mobile-demo/mobile_search_aim_fire.mp4), [mission timeline](experiments/elastic-launcher/mobile-demo/mission.csv), [events](experiments/elastic-launcher/mobile-demo/events.json), [shot trace](experiments/elastic-launcher/mobile-demo/shot.csv), and [metadata](experiments/elastic-launcher/mobile-demo/metadata.json) are saved together. This run detected at 5.55 s, stopped at 9.95 s, fired at 12.45 s, contacted at 14.05 s, and confirmed the topple at 14.466 s; the shot used 4.2 m/s, 15-degree pitch, and Cd=0.45.

The earlier “YOLO” label referred to a visual demonstration only: repository inspection found no YOLO weights, image inference code, or camera pipeline. The current cue is still based on target truth position, a 2.3 m range limit, and a simulated 50-degree field of view; the rover follows prescribed kinematic waypoints without wheel dynamics or obstacle avoidance. The LIMO and launcher meshes are visual geometry, and the elastic forces are not simulated. This video does not claim real YOLO recognition or autonomous navigation.

### Integrated movement-to-topple GUI playback

`play_mobile_gui.py` replays the same saved patrol timeline and MuJoCo shot trace inside a panel-free MuJoCo viewer. It shows the LIMO moving along the prescribed route, simulated target detection, approach and chassis/launcher aiming, the launcher sequence, projectile contact, and the existing topple confirmation. The post-trigger physics trace plays at 0.2× so the launch and target motion remain visible. This is a GUI replay of recorded mission and physics states; it is not a live perception or wheel-dynamics controller.

Regenerate the playback inputs and run the GUI from the project directory:

```powershell
.\.venv\Scripts\python.exe record_mobile_cad_demo.py --preview --output experiments/agile-demo-2026-11-16/gui-recording/gui-mobile-replay-data
.\.venv\Scripts\python.exe play_mobile_gui.py --data-dir experiments/agile-demo-2026-11-16/gui-recording/gui-mobile-replay-data --summary experiments/agile-demo-2026-11-16/gui-recording/gui-integration-summary.json --wait-for-space --hold 15
```

The panel-free GUI waits at the initial pose until Space is pressed. For a screen recording, start the recorder first, then press Space. The GUI playback uses the same mission events as the integrated video: simulated detection at 5.55 s, stop at 9.95 s, fire trigger at 12.45 s, ball release at 13.725 s, contact at 14.048 s, and topple confirmation at 14.466 s. The success rule remains at least 45 degrees for 0.20 seconds.

The recorded [36-second panel-free GUI video](experiments/agile-demo-2026-11-16/gui-recording/mujoco-gui-demo.mp4) shows the LIMO's prescribed patrol, simulated detection and approach, chassis aiming, shot, target impact, and topple. The [GUI playback summary](experiments/agile-demo-2026-11-16/gui-recording/gui-integration-summary.json) records the original event times and unchanged 45-degree / 0.20-second rule.

### Wego navigation + MuJoCo workflow overview

The [49-second combined overview](experiments/agile-demo-2026-11-16/gui-recording/wego-nav-mujoco-workflow-overview.mp4) opens with the measured Wego/Nav2 MPPI waypoint results, shows the MuJoCo GUI replay with a persistent scope label, and closes with the remaining integration gap. It combines evidence from two separate runs; it is **not** a single live end-to-end run. The Wego segment is a result card because no Wego screen recording was captured. The source video can be regenerated with `create_combined_demo.py` from the project venv; it uses Pillow and the bundled `imageio-ffmpeg` executable.

## Exploration, detection, and approach: current demo and next design

The current `record_mobile_cad_demo.py` is a presentation replay, not a perception/navigation implementation. It follows three fixed waypoints at up to 0.45 m/s with a 1.2 rad/s heading-rate limit. A target is treated as visible from its simulator ground-truth position when it is within 2.3 m and within a 50-degree field of view; 0.25 s of continuous visibility triggers detection. The rover finishes its preset route, stops, interpolates chassis yaw and launcher pitch for 1.2 s, and emits the fire trigger 2.5 s after stopping. The detector does not process camera images, and the route has no wheel dynamics, map, or obstacle avoidance.

### Design direction informed by the team repository

The team repository [`juyoung020/limo_fire_ws`](https://github.com/juyoung020/limo_fire_ws) separates mission logic from navigation and perception. Its current design describes Nav2 Behavior Tree orchestration, mapped patrol for routine inspection, frontier-based exploration for first-map building, YOLO instance segmentation for fire/smoke/equipment/person/door/stair observations, and object-map detections as the mission interface. It proposes measuring fire size from depth, then LiDAR or a ground-plane estimate as fallbacks. The exploration roadmap selects among frontier candidates at 1 Hz and sends chassis velocity actions at 10 Hz; safe path planning and map-building remain rule-based. Its approach/aim choice is still an experiment: compare learned aiming with PID plus a distance correction table. These are team design directions, not capabilities of this MuJoCo demo.

For this project, evolve the scene in reviewable stages:

1. **Explore:** replace the fixed route with a small occupancy grid and frontier selection. For an initial deterministic demo, rank reachable frontiers by information gain divided by path cost, then navigate using waypoint following with collision checks. Keep the current replay available as the visual baseline.
2. **Detect:** first replace the ground-truth cue with a camera observation interface carrying class, confidence, bounding box/mask, and timestamp. Use a simulated ground-truth detector only as a labeled test adapter; later connect the team's YOLO segmenter. Require temporally consistent observations and reject stale detections.
3. **Approach:** estimate target range from depth at the fire mask, with LiDAR association and a documented ground-plane estimate as fallbacks. Transform the target into the map frame, choose a reachable stand-off pose with line of sight, and let a local planner avoid obstacles. Stop within a tested position/heading tolerance before aiming.
4. **Aim and fire:** compute yaw from the target bearing and pitch from the range/trajectory model, then close the loop with camera feedback. Compare PID plus distance correction against a learned policy only after collecting repeatable miss, contact, and time-to-confirmation metrics. Preserve the existing 45-degree / 0.20-second success condition.

Suggested state flow: `PATROL_OR_FRONTIER → CANDIDATE_DETECTED → CONFIRM_AND_LOCALIZE → PLAN_STANDOFF → APPROACH → ALIGN → FIRE → VERIFY_TOPPLE`. Add explicit timeout, stale-sensor, unreachable-target, and retry/fallback transitions. For the next demo milestone, implement only the camera-observation interface and a deterministic frontier/standoff controller in MuJoCo; keep YOLO, ROS 2/Nav2, and learned policies as later integrations rather than implying they are already running here.

### Wego Nav2 controller trial (2026-11-13)

Separately from this local MuJoCo MVP, a remote Webots run on Wego exercised Nav2 waypoint navigation in the team's `limo_fire_presentation` world. The active MPPI controller completed two waypoint goals (16 seconds each); a Regulated Pure Pursuit frontier candidate stalled with collision/progress reports and was canceled. This experiment did **not** test camera detection, frontier exploration end-to-end, aiming, firing, or target topple. It does not add ROS 2/Nav2 or MPC to this MVP. See the [Wego run report](experiments/wego-nav-2026-11-13/README.md) and [per-goal CSV](experiments/wego-nav-2026-11-13/navigation_trials.csv) for setup, limitations, and results.

The existing integrated Webots/Nav2/YOLO/fire-manager/MuJoCo launcher was exercised on Wego in isolated ROS domain 43. A localized fire observation triggered the actual MuJoCo shot, direct hit, and topple confirmation after a small fire-manager fix. Two later launches failed on Nav2 progress before detection; one of three post-fix starts completed the firing flow. The original 45-degree-for-0.20-second rule remained unchanged. See the [integrated Wego report](experiments/wego-integrated-2026-11-13/README.md) for the cause, change, test evidence, CSVs, and explicit video limitations; the initial pre-fix failure is preserved in [its diagnostic](experiments/wego-nav-2026-11-13/integrated-fire-attempt.md).
