# scenes — 탐사 RL 학습·시험 맵 묶음

옛 ROADMAP(`archive/explore_roadmap_1104`) S2. 장면 파일(`.scene`, 형식 2 — `sim/include/scene_host.h`)과 목록은 다시 만들 수 있는 생성물이라 저장소에 넣지 않는다(.gitignore). 아래 명령으로 같은 씨앗이면 같은 묶음이 나온다.

| 폴더 | 내용 |
|---|---|
| `train/*.scene` + `train/list.tsv` | 학습 맵 520 개 (씨앗 1) = 가구 없는 판 260 + 같은 집·같은 출발점에 가구를 놓은 판 260 (`<이름>_f.scene`). `list.tsv` = path · floor_m2(갈 수 있는 바닥 넓이) · wall_cells(벽 면 칸 = 덮음 분모, 가구 면 포함) · grade · gw · gh · furnished(0/1) |
| `train/rejected.tsv` | build_world 닫힘 검사에서 버린 맵과 까닭(새는 곳 좌표). 씨앗 1 은 0 개 |
| `test/` | 시험 맵 — AWS RoboMaker small-house·bookstore (`w3d2scene` 결과를 복사). **학습 목록에 넣지 않는다** |
| `grids/` | 중간 격자(Nav2 map_server 형식 PGM + YAML) + `manifest.tsv`(출발점) + `skipped.tsv`(격자 단계에서 버린 것) |
| `preview/` | 눈 확인용 build_world 격자 그림(흰 바닥, 검정 벽 면 칸, 회색 그 밖 막힘, 진회색 점 출발점) |
| `src/procthor_10k/` | 받은 데이터 (아래) |
| `hssd_train/`·`hssd_test/` | HSSD 3D 메시 집 (`hssd_<id>_l<층>.scene` + `list.tsv` 같은 꼴). 학습 목록에 넣을지는 따로 정함 — 아래 "HSSD 메시 장면" |

## 크기 등급 (커리큘럼이 고르는 값)

갈 수 있는 바닥 넓이(`build_world` 의 `n_floor × 0.05²`) 기준. 문턱은 `sim/tools/scene_list.cpp` `grade_of`.

| 등급 | 넓이 | 씨앗 1 개수 | 넓이 범위 (m²) | 평균 벽 면 칸 |
|---|---|---|---|---|
| 0 작은 방 | < 30 m² | 67 (ProcTHOR 42 + 절차 방 25) | 6.7 ~ 29.2 | 408 |
| 1 작은 집 | 30 ~ 80 | 66 (ProcTHOR 56 + 절차 방 10) | 30.1 ~ 79.9 | 944 |
| 2 보통 집 | 80 ~ 150 | 69 (ProcTHOR) | 80.8 ~ 149.6 | 1670 |
| 3 큰 집 | ≥ 150 | 58 (ProcTHOR) | 150.2 ~ 299.1 | 2928 |

위 표는 가구 없는 판. 가구 판(furnished = 1, 260 개)은 가구가 바닥을 차지해 넓이가 줄고 벽 면 칸이 늘어 등급이 내려가기도 한다:

| 등급 | 가구 판 개수 | 넓이 범위 (m²) | 평균 벽 면 칸 |
|---|---|---|---|
| 0 | 75 | 5.0 ~ 29.9 | 559 |
| 1 | 77 | 30.2 ~ 79.4 | 1508 |
| 2 | 73 | 80.3 ~ 148.6 | 3090 |
| 3 | 34 | 151.0 ~ 264.1 | 4782 |

`rooms_g0_0009_f` 는 6.7 m² 방이라 놓을 자리가 없어 가구 0 개 — 목록에 furnished = 0 으로 적힌다(scene_list 는 이름이 아니라 장면 안 가구 물체로 가림). 그래서 furnished 0/1 = 261/259.
가구 없는 판(등급마다 약 절반)은 쉬운 단계용으로 남긴다.

시험 맵 넓이: small-house 147.8 m²(벽 면 칸 2594), bookstore 145.3 m²(3206) — 둘 다 등급 2.

## 만드는 법

1. 데이터 받기 (ProcTHOR-10K, Git LFS 파일을 직접. 저장소의 `sim/maps/procthor_10k/*.jsonl.gz` 는 LFS 포인터라 내용이 없다)
   ```bash
   mkdir -p train/explore/scenes/src/procthor_10k && cd train/explore/scenes/src/procthor_10k
   curl -sSLO https://media.githubusercontent.com/media/allenai/procthor-10k/main/train.jsonl.gz
   sha256sum train.jsonl.gz   # ee3c4aa14b4d8f0895fecfb5fdaca59395427ca1018b2f9aeeedbc61e5824587 (LFS 포인터의 oid 와 같아야 함)
   ```
2. 도구 빌드: `train/maps` 에서 `cargo build --release` (`scene_grids`, `pgm2png`), `train/explore/sim` 에서
   `cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86 && cmake --build build --target map2scene scene_list` (아키텍처는 기기에 맞게).
3. 격자 → 장면 → 닫힘 검사·목록 (`train/explore/scenes` 에서, `M=../../maps/target/release`, `B=../sim/build`)
   ```bash
   $M/scene_grids --out grids --procthor src/procthor_10k/train.jsonl.gz --procthor-limit 1500 \
       --rooms 0:25 --rooms 1:10 --per-grade 65 --seed 1 --furnish
   mkdir -p train && tail -n +2 grids/manifest.tsv | awk -F'\t' '{print $1, $6, $7, $9}' | \
       xargs -P 10 -n 4 sh -c "if [ \"\$3\" = - ]; then H=; else H=\"--heights grids/\$3\"; fi; \
         $B/map2scene grids/\$0.yaml train/\$0.scene --height 2.4 --start \$1 \$2 \$H > /dev/null"
   mkdir -p preview && $B/scene_list --list train/list.tsv --rejected train/rejected.tsv --png preview --png-every 40 train/*.scene
   $M/pgm2png preview/*.pgm
   mkdir -p test && cp small-house.scene bookstore.scene test/
   ```

## 무엇을 하는가

- `scene_grids` (Rust, `train/maps/src/bin/scene_grids.rs`): ProcTHOR 집 한 줄 → `fmap::procthor::load`(방 바닥 다각형, 벽 0.10 m, 안쪽 문 구멍; 바깥 문은 막힘) → `keep_reachable`(로봇이 갈 수 있는 가장 큰 영역).
  절차 방(`train/maps/src/rooms.rs`)은 직사각 외곽을 BSP 로 나눈 방·복도(1.0~1.6 m 띠)에 문 0.8~1.1 m, 큰 방에 가구 상자 0~2 개.
  도달 영역 = 빈 칸(254), 나머지 = 막힘(0)인 격자로 쓰고, 막힘 아닌 칸의 바운딩 상자 + 4 칸으로 자른다. 한 변 > 510 칸이면 버림(sim 지도 512 칸 = 25.6 m 상한, build_world 는 칸 + 1).
  출발점은 도달 영역 중 벽면에서 0.4 m 이상인 칸에서 무작위. 넓이로 어림한 등급마다 `--per-grade` 개 — 절차 방 먼저, 나머지는 ProcTHOR 를 씨앗 섞은 차례로.
- `map2scene` (C++, 기존): 격자 등고선을 따라 바닥부터 2.4 m 벽면, 바닥 상자, 출발점을 넣은 `.scene`. 4 이웃으로 붙은 막힌 칸 덩어리 = 물체 하나(가구 없는 판은 벽 전체가 하나).
- `scene_list` (C++, `train/explore/sim/tools/scene_list.cpp`): 학습 환경과 같은 `build_world` 로 닫힘(갈 수 있는 바닥이 맵 테두리에 안 닿음)·넓이·벽 면 칸 수를 재고, 닫힌 맵만 `list.tsv` 에.
  닫힘 검사가 실제로 잡는지: 절차 방 하나의 격자에 외벽을 뚫는 0.45 m 통로를 내자 "집이 닫히지 않음 … 가장 좁은 곳 (3.17, 1.93) — 벽까지 0.11 m" 로 버렸고, 원본은 통과.

격자 단계 기록(씨앗 1): 작업 1535 개(ProcTHOR 1500 + 절차 35) 중 격자 1515 개, 버림 20 개(19 개 격자 상한 25.6 m 초과, 1 개 도달 영역 < 3 m²). 그중 260 개를 골라 장면으로 만들었고 닫힘 검사 버림 0 개.


## 절차 가구 (`--furnish`, `train/maps/src/furnish.rs`)

시험 맵(AWS 작은 집·서점)은 가구가 많은데 ProcTHOR 집은 벽·문뿐이라, 고른 맵마다 같은 출발점으로 가구 판 `<이름>_f` 를 하나 더 만든다. 씨앗 = `--seed` 와 맵 이름(맵마다 고정 — 판마다 바뀌지는 않음).

- 종류: 벽에 붙은 상자(소파·책장·침대 0.4~2.0 × 0.4~1.0 m, 높이 0.4~2.0 m), 방 가운데 탁자(0.7~1.6 × 0.6~1.0 m) + 의자 0~4 개(0.4~0.5 m) — **다리만**(5~10 cm 기둥, 탁자 높이 0.75 m·의자 0.45 m), 작은 물체(0.1~0.3 m 상자, 높이 0.15~0.4 m).
- 밀도: 맵마다 배율 0.5~1.5, 방(ProcTHOR 방 번호)마다 넓이에 비례 — 벽 상자 넓이/7 m², 탁자 넓이/25 m², 작은 물체 넓이/6 m² × 배율.
- 물체: 가구 하나(다리 하나)가 벽·다른 가구와 한 칸(5 cm) 이상 떨어져 map2scene 에서 따로 덩어리 = 따로 물체가 된다. 라이다 재질은 학습 시뮬(`lidar_real.h material_of`)이 물체마다·판마다 기존 분포대로 뽑는다(바닥 투영 1.5 m 넘는 것은 구조물로 보통 재질 — `sensor_model.yaml structure_size`).
  높이는 격자와 같은 크기의 높이 PGM(`<이름>_f.h.pgm`, cm)으로 map2scene `--heights` 에 넘기고, 그 덩어리 이름이 `furniture_<번호>` 가 된다.
- 막지 않는 곳: 출발점 0.6 m 안, 좁은 곳(가로·세로 빈 칸 줄이 1.3 m 미만 — 문 틈·좁은 복도)에서 0.5 m 안.
- 고립: 놓을 때마다 로봇이 설 칸(막힘까지 0.15 m 이상)을 출발점에서 다시 번져, 전에 닿던 칸 중 아직 설 수 있는데 못 닿는 칸이 생기면 그 가구를 뺀다.
- 씨앗 1: 벽 상자 3426, 탁자 870, 의자 1948, 작은 물체 3999 개, 고립시켜 뺀 것 120 개. 가구 판 260 개 모두 닫힘 검사 통과. 가구 없는 판 260 개는 이 옵션을 켜도 바이트까지 같다.

## HSSD 메시 장면 (`mesh2scene`, `sim/tools/mesh2scene.cpp`)

학습 맵이 바닥 평면도를 2.4 m 로 세운 벽뿐이라, 실제 3D 메시 집(가구·의자 다리·창틀·계단 포함)을 넣으려고 HSSD(Habitat Synthetic Scenes Dataset, `huggingface.co/datasets/hssd/hssd-hab`)를 바꾼다.

1. 데이터 받기 (gated — HF 계정에서 라이선스 동의 후 `hf auth login`). 장면이 쓰는 물체만 받으면 약 8.5 GB(렌더 glb + object_config). 원본은 저장소에 넣지 않는다
   ```bash
   hf download hssd/hssd-hab --repo-type dataset --local-dir ~/datasets/hssd-hab \
       --include "stages/*" --include "scenes/*" --include "metadata/*" --include "scene_splits.yaml" --include "*.scene_dataset_config.json"
   # 물체: scenes/*.scene_instance.json 의 template_name 에 해당하는 objects/**/<이름>.glb·.object_config.json (전체 objects/ 는 10.4 GB)
   ```
2. 빌드: `cmake --build build --target mesh2scene scene_list`
3. 바꾸기·검사 (HSSD 자체 나눔 `scene_splits.yaml`: train 125 → 학습, val 43 → 시험. small-house·bookstore 는 그대로 시험)
   ```bash
   D=~/datasets/hssd-hab; B=../sim/build
   for s in train val; do d=$([ $s = train ] && echo hssd_train || echo hssd_test); mkdir -p $d
     awk -v s=$s '/^[a-z]+:/{k=$1} /^- /&&k==s":"{gsub(/[-\x27 ]/,""); print}' $D/scene_splits.yaml | \
       xargs -P 8 -n 4 $B/mesh2scene --hssd $D --out $d --seed 1 >> hssd_convert.tsv
     $B/scene_list --list $d/list.tsv --rejected $d/rejected.tsv --png preview --png-every 1 $d/*.scene; done
   ```

무엇을 하는가 (자세한 규칙은 `mesh2scene.cpp` 머리 주석):
- 무대(stage) glb + `scene_instance.json` 물체마다 번역·회전(쿼터니언 [w, x, y, z])·크기를 곱하고 glTF y 위 → z 위. glTF 는 `sim/tools/cgltf/cgltf.h`(MIT, v1.15)로 읽는다.
- 층: 무대의 위를 보는 수평면 넓이를 5 cm 높이 칸으로 모은 봉우리(≥ 8 m², 1.5 m 넘게 떨어짐). 층마다 장면 `hssd_<id>_l<층>` — 바닥 −0.05 ~ +2.0 m 만 남기고(삼각형을 잘라) 바닥을 z = 0 으로.
- 바닥 ±5 cm 안에 다 든 삼각형(바닥·문턱·깔개)은 빼고 평평한 바닥 판으로 — 작은 턱이 몸통 높이(0.03 ~ 0.25 m)에 걸려 막힘이 되지 않게.
- 닫기: 갈 수 있는 바닥(바닥 칸 중 막힘 아닌 가장 큰 덩어리)에 닿은 "바닥도 막힘도 아닌 칸"(계단 구멍, 바깥 문, 메시 없는 유리)에 2.4 m 가상 벽(`virtual_wall`). 문(metadata `main_category = door`, 닫힌 문짝)은 빼서 방 사이 문을 연다 — 바깥 문 자리는 바깥에 바닥이 없어 가상 벽이 막는다.
  둘레 중 가상 벽 몫이 0.5 를 넘는 층(지붕·천장 윗면)은 버린다(문턱 잠정).
- 25.5 m 를 넘으면 갈 수 있는 바닥이 가장 많은 25.5 m 창을 고르고 잘린 가장자리도 가상 벽. 모든 삼각형을 창으로 잘라 BVH 상자 = 창.
- 출발점: 서는 칸의 가장 큰 덩어리 중 막힘(가상 벽 포함)에서 0.4 m 넘는 칸을 씨앗·이름 해시로.
- 물체: 무대 = `hssd_stage` 하나, HSSD 물체 인스턴스마다 `furniture_<번호>_<틀 이름>`(라이다 재질을 물체마다 뽑음, `scene_list` 의 furnished = 1).
- 마당 빼기: 8 방향 중 6 방향 이상이 20 m 안에서 벽 높이(0.3 ~ 2.0 m) 형상에 닿는 바닥 칸만 갈 수 있는 바닥으로(잠정 값). 천장 유무로 가리면 HSSD 위층(무대에 지붕이 없는 집이 많음)이 통째로 빠져 둘러싸임으로 본다.
- 삼각형 줄이기: 물체마다 꼭짓점을 칸으로 모음(vertex clustering) — 바닥 위 0.35 m 아래는 4 cm, 위는 12 cm. 물체의 몸통 높이 발자국이 한 칸 넘게 바뀌면(가는 다리가 사라짐) 2 cm·1 cm, 그래도 바뀌면 원래 메시.
- 끝에 학습과 같은 `build_world` 로 닫힘을 다시 재 닫힌 것만 저장. 표준 출력은 층마다 TSV 한 줄(머리줄 `mesh2scene --header`).

씨앗 1 결과 (2026-11-11, 168 집 모두 받음 — 장면 JSON 이 가리키는 물체 중 저장소에 없는 틀 3 개(`3759`·`3760`·`5804`, 인스턴스 13 개)만 빠짐):

| | 학습 (train 125 집) | 시험 (val 43 집) |
|---|---|---|
| 장면(층) | 129 | 44 |
| 등급 0/1/2/3 | 4 / 25 / 56 / 44 | 1 / 8 / 13 / 22 |
| 넓이 범위 (m²) | 15.7 ~ 601.2 | 21.2 ~ 571.7 |
| `scene_list` 닫힘 버림 | 0 | 0 |

- 층 후보 329 개 중 장면 173, 버림 156 = 갈 수 있는 바닥 < 3 m² 151(천장·지붕 윗면 높이) + 열린 층 5. 모든 집이 장면 하나 이상, 두 층 이상 5 집.
- 25.5 m 창으로 자른 장면 38 개. 문 물체 뺀 인스턴스 486 개. 꼭짓점 모으기에서 q/2·q/4 로 내린 물체 1232, 원래 메시 89.
- 삼각형 장면당 중앙값 7.2 만·최대 23.8 만, 합 1380 만, 파일 train 566 MB·test 209 MB (ProcTHOR 판은 장면당 수천 개 — GPU 메모리·광선 비용이 늘어남).
- CPU 참조판(`env_bench --cpu --threads 6`, 환경 32 × 100 걸음): HSSD 학습 목록 54.0 ms/걸음, ProcTHOR 학습 목록 23.6 ms/걸음.
- 같은 인자로 다시 돌리면 장면 파일이 바이트까지 같다.

## 낮은 장애물 변형 (`scene_lowobs`, `sim/tools/scene_lowobs.cpp`)

판단 모델 실험([scripts/README](../scripts/README.md) "위층을 판단 모델에 맡기기")의 시험 장면. 시험 맵에 높이 0.09 m 턱 · 상자를 더한다.
윗면이 라이다 평면보다 낮아 로봇 지도에는 안 찍히지만 몸통과는 부딪히고 카메라에는 보인다 — "지도에는 길이 뚫려 보이는데 실제로는 막힌 곳".
상자 자리는 `sim/config/lowobs/<장면>.boxes.tsv`(손으로 고른 값, 잠정)에 있고, 변형 장면 파일은 저장소에 넣지 않는다. 다시 만들려면:

```bash
cd train/explore/sim
<빌드>/sim/scene_lowobs --in ../scenes/test/small-house.scene --boxes config/lowobs/small-house.boxes.tsv \
  --out <폴더>/small-house-lowobs.scene --max-lost 8 --ppm <폴더>/small-house-lowobs.ppm --ppm-scale 3
```

상자 윗면이 라이다 높이보다 낮고 몸통 충돌 띠 안인지, 집이 여전히 닫혀 있는지, 갈 수 있는 바닥이 `--max-lost`(m²) 넘게 줄지 않았는지 확인하고 하나라도 어기면 파일을 쓰지 않는다.
`--boxes` 없이 부르면 원본 보고만 한다. `--ppm` 은 확인용 그림(파랑 = 더한 상자, 옅은 선 1 m 눈금).
학습 맵용 무작위 변형: `scene_lowobs --in a.scene --random N --seed S --out b.scene` — 갈 수 있는 바닥 칸에서 상자 N 개(0.3~0.8 × 0.25~0.5 m, 높이 0.05~0.09 m, 방향 고르게, 출발점 1 m 안은 뺌 — 잠정).
같은 확인(라이다에 안 보임 · 막힌 칸 · 길을 끊지 않음 `--max-lost`)을 못 넘으면 실패하므로 부른 쪽이 씨앗을 바꿔 다시 부른다. 판정용 장면에는 쓰지 않는다.

## 라이선스

| 출처 | 라이선스 | 비고 |
|---|---|---|
| ProcTHOR-10K (`github.com/allenai/procthor-10k`, Deitke et al., NeurIPS 2022) | Apache-2.0 (저장소 `LICENSE`, `sim/maps/procthor_10k/LICENSE` 사본) | 생성 장면은 이 데이터의 2 차 산출물 — 재배포 시 Apache-2.0 고지·출처 표기 |
| 절차 방 | 우리 코드 | |
| AWS RoboMaker small-house·bookstore (시험) | MIT-0 | `sim/maps/README.md` |
| HSSD (`huggingface.co/datasets/hssd/hssd-hab`, Khanna et al., "Habitat Synthetic Scenes Dataset", CVPR 2024) | CC BY-NC 4.0 | 비영리 연구라 바꾼 `hssd_*.scene` 을 팀 안에서 써도 되나 출처 표기 필수·상업 이용 금지. 원본 데이터는 저장소에 안 넣음 |
| cgltf (`github.com/jkuhlmann/cgltf` v1.15) | MIT | `sim/tools/cgltf/LICENSE` |
| HouseExpo | 코드 MIT, 데이터 SUNCG 유래 | **이 묶음에 안 씀** — SUNCG 는 재배포 금지라 로컬 학습만 가능. ProcTHOR 만으로 등급마다 충분해서 넣지 않았다 |

## 한계

- ProcTHOR 가구(`objects`)는 크기 정보가 없어 쓰지 않고, 가구 판은 위 절차 가구다(축에 맞춘 직사각만, 비스듬한 가구 없음).
- 벽 전체는 덩어리 하나라 재질이 하나(큰 구조물 = 보통). 탁자·의자 다리는 다리마다 따로 물체라 재질도 다리마다 뽑힌다(실물은 한 가구의 다리가 같은 재질).
- 탁자 상판·가구 윗면은 만들지 않는다(map2scene 은 옆면만) — 라이다 평면에는 영향 없고, 깊이 카메라에는 다리 기둥·뚜껑 없는 상자로 보인다.
- 등급 3 은 25.6 m 상한 안의 집만이라 300 m² 를 넘는 집은 없다 (HSSD 는 창으로 잘라 600 m² 까지).
- HSSD: 둘러싸임 검사가 담·생울타리·큰 가구에 둘러싸인 마당·테라스를 집 안으로 남길 때가 있다(큰 저택 몇 개). 둘러싸임 경계가 광선 꼴이라 테라스 쪽 가상 벽이 지그재그가 된다.
  계단 위층으로 가는 계단은 몸통 높이에 걸려 막힘(층 사이 이동 없음). 문은 모두 열림(뺌).
