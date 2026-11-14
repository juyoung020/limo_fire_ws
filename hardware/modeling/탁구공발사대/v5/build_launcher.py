"""탁구공 발사대 v4: 고무줄 + 서보 2개(레버 DS3218 = 장전·발사, 각도 MG996R), PLA 출력.

좌표(발사대 자기 좌표): x = 발사 방향(총신 축), y = 옆(+y 레버·걸쇠 쪽, -y 각도 서보 쪽), z = 총신에 수직 위. 단위 mm.
발사대는 몸통 아래 축(AX)에 걸려 각도 서보로 5~30° 기운다. 받침대(cradle)는 LIMO 지붕에 나사로 고정.

y 층(+y 쪽): 총신 바깥 23.2 | 레버 24.6~29.6 | 걸쇠 이빨 30.6~34.6 | 고무줄 35.6~40.6 | 핀 머리 41.2~43.7 | 레버 서보 벽 45.9~49.9

동작
 1. 레버(축이 총신 위 48 mm)가 뒤로 돌며(+, 66°) 핀(+y 쪽)을 밀어 고무줄을 늘린다(장전). 공 통 맨 아래 공이 총신으로 떨어진다.
 2. 핀이 걸쇠 이빨의 비탈을 눌러 이빨이 내려갔다가, 핀이 지나가면 걸쇠 고무줄이 이빨을 다시 올린다(저절로 걸림).
 3. 레버가 천천히 0° 로 돌아오면 핀이 이빨에 닿아 멈춘다.
 4. 레버를 0° 보다 앞(-)으로 더 돌리면 레버 둘째 팔(손가락)이 걸쇠 꼬리를 밀어 이빨이 내려가고 발사된다.
 5. 각도 서보가 축을 돌려 5~30°. 장전은 15° 이상에서 하고 목표 각도로 천천히 내린다(설계_v4.md 순서).

서보 치수는 공개 STEP(ref/servo)에서 잰 값(플랜지·구멍 중심 간격·윗면 높이·축 위치·혼).
조립 순서는 설계_v4.md."""
import os, math, json
import numpy as np, manifold3d as m3, trimesh

D = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(D, "stl"); os.makedirs(OUT, exist_ok=True)

# ---------------- 도우미 ----------------
def box(x0, x1, y0, y1, z0, z1):
    return m3.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])
def cyl_z(r, z0, z1, x=0, y=0, n=96):
    return m3.Manifold.cylinder(z1 - z0, r, r, n).translate([x, y, z0])
def cyl_x(r, x0, x1, y=0, z=0, n=96):
    return m3.Manifold.cylinder(x1 - x0, r, r, n).rotate([0, 90, 0]).translate([x0, y, z])
def cyl_y(r, y0, y1, x=0, z=0, n=64):
    return m3.Manifold.cylinder(y1 - y0, r, r, n).rotate([-90, 0, 0]).translate([x, y0, z])
def tear_x(r, x0, x1, top=None, n=96):
    """x 축 원기둥 + 위로 45° 지붕(출력 시 천장 서포트 없음)."""
    return (cyl_x(r, x0, x1, 0, 0, n) + box(x0, x1, -0.01, 0.01, r * math.sqrt(2) - 0.02, r * math.sqrt(2))).hull()
def tear_y(r, y0, y1, x, z, n=48):
    """y 축 구멍용: 위로 45° 지붕."""
    return (cyl_y(r, y0, y1, x, z, n) + box(x - 0.01, x + 0.01, y0, y1, z + r * math.sqrt(2) - 0.02, z + r * math.sqrt(2))).hull()
def tear_y_down(r, y0, y1, x, z, n=48):
    """y 축 원기둥 + 아래로 뾰족한 용골(옆으로 튀어나온 고리 밑면 서포트 없음)."""
    return (cyl_y(r, y0, y1, x, z, n) + box(x - 0.01, x + 0.01, y0, y1, z - 2.5 * r, z - 2.5 * r + 0.02)).hull()
def tear_y_down45(r, y0, y1, x, z, n=48):
    """y 축 원기둥 + 아래로 45° 뾰족(깊이 r√2) — 아래로 짧아 다른 고무줄 길을 덜 막음."""
    return (cyl_y(r, y0, y1, x, z, n) + box(x - 0.01, x + 0.01, y0, y1, z - r * math.sqrt(2), z - r * math.sqrt(2) + 0.02)).hull()
def prism_xz(pts, y0, y1):
    """(x, z) 다각형(반시계)을 y0~y1 로 밀어낸 기둥."""
    return m3.Manifold.extrude(m3.CrossSection([pts]), y1 - y0).rotate([90, 0, 0]).translate([0, y1, 0])
def to_t(mf):
    me = mf.to_mesh(); return trimesh.Trimesh(me.vert_properties[:, :3], me.tri_verts)

# ---------------- 서보 (STEP 실측) ----------------
# 서보 자기 좌표: 원점 = 출력축 중심선이 플랜지 윗면(혼 쪽 면)을 지나는 점, w = 혼 쪽, u = 몸통 길이 방향(축 → 먼 끝), v = 폭.
SERVO_L = dict(name="DS3218MG", body_l=40.1, body_w=20.1, flange_l=54.7, flange_t=3.2, hole_dl=49.4, hole_dw=10.0,
               hole_d=2.6, case_top=9.5, spline_end=14.5, below_h=27.7, shaft_off=10.4, horn_r=10.0,
               rib=dict(u=(-16.6, -10.4), v=0.8, h=2.0))             # 축 쪽 끝 플랜지 윗면 리브(폭 1.6, 길이 6.2, 높이 2)
SERVO_K = dict(name="MG996R", body_l=40.5, body_w=20.0, flange_l=53.8, flange_t=2.2, hole_dl=48.6, hole_dw=10.4,
               hole_d=2.6, case_top=8.0, spline_end=13.8, below_h=28.4, shaft_off=10.5, horn_r=10.5, rib=None)
HORN_TOP = 1.6       # 축 끝 → 혼 윗면(DS3218 동봉 원판 혼 STEP 실측), 혼 두께 3
HORN_T = 3.0
SERVO_CLR = 0.4

def servo_solid(S, horn=True, clr=0.0):
    """서보 덩어리(플랜지·위 케이스·아래 몸통·축 보스·혼·리브). 서보 자기 좌표 (u, v, w)."""
    c = clr
    u0, u1 = -S["shaft_off"], S["body_l"] - S["shaft_off"]
    ex = (S["flange_l"] - S["body_l"]) / 2
    w2 = S["body_w"] / 2
    s = box(u0 - ex - c, u1 + ex + c, -w2 - c, w2 + c, -S["flange_t"], 0)                    # 플랜지
    s += box(u0 - c, u1 + c, -w2 - c, w2 + c, 0, S["case_top"])                                 # 위 케이스
    s += box(u0 - c, u1 + c, -w2 - c, w2 + c, -S["flange_t"] - S["below_h"], -S["flange_t"])    # 아래 몸통
    s += cyl_z(6.0 + c, 0, S["spline_end"] - HORN_T + HORN_TOP, 0, 0, 48)                      # 축 보스·혼 허브
    s += cyl_z(3.0 + c, 0, S["spline_end"], 0, 0, 32)
    if horn:
        s += cyl_z(S["horn_r"] + c, S["spline_end"] + HORN_TOP - HORN_T, S["spline_end"] + HORN_TOP, 0, 0, 64)
    if S.get("rib"):
        r = S["rib"]; s += box(r["u"][0], r["u"][1], -r["v"], r["v"], 0, r["h"])
    return s

def servo_holes(S, depth=30):
    hs = m3.Manifold()
    uc = S["body_l"] / 2 - S["shaft_off"]
    for du in (-1, 1):
        for dv in (-1, 1):
            hs += cyl_z(S["hole_d"] / 2, -depth, depth, uc + du * S["hole_dl"] / 2, dv * S["hole_dw"] / 2, 24)
    return hs

def place(m, origin, u, v, w):
    """서보 자기 좌표 (u, v, w) → 발사대 좌표. u, v, w 는 단위 벡터(축 정렬)."""
    M = np.array([u, v, w], float).T            # 열 = u, v, w
    return m.transform(np.c_[M, np.asarray(origin, float)].tolist())

# ---------------- 치수 ----------------
BALL = 40.0
BORE_R = 20.8
WALL = 2.4
BARREL_R = BORE_R + WALL
DRAW = 50.0
PIN_REST, PIN_COCK = -30.0, -30.0 - DRAW
PIN_R = 3.0; SLOT_W = 6.6
PIN_OVER = 5.3                       # 검사·시뮬에서 장전 위치보다 더 들어가 보는 여유
PIN_Y = (-42.0, 43.7)                # 핀 y 범위: +y 쪽은 레버(24.6~29.6)·걸쇠 이빨(30.6~34.6)·고무줄 홈, -y 쪽은 고무줄 홈
PIN_HEAD = (-44.5, -42.0, 4.0)       # -y 끝 머리(y0, y1, r): -y 쪽에서 끝부터 꽂는다. +y 끝은 벽(45.9)이 막는다
BAND_Y = (35.6, 40.6)                # 고무줄 홈 |y| (걸쇠 바깥). 레버를 총신 가까이 둬야 플런저가 비틀려 걸리지 않음
PLUNGER_L = 50.0                     # 총신 안 안내 길이. 레버가 핀 한쪽 끝만 밀어 생기는 비틀림 마찰(2 μ a / L)을 줄임 — 40 이면 μ 0.5 에서 걸림(sim_full)
PLUNGER_R = BORE_R - 0.4
PLUNGER_WALL = 0.9                    # 2줄(0.45) — 26-11-12 1.2 에서 줄여 가볍게(공 속도)
PLUNGER_FACE = 1.2                   # 앞판 두께(v5)
PLUNGER_CHAMF = 4.0                  # 앞면 테두리 45° 모따기
PLUNGER_BOSS_IN = 4.0                # 핀 받침이 벽 안쪽으로 나온 길이(가운데는 비움)
PLUNGER_SIDE_WIN = ((-17.0, -6.0), (6.0, 14.0))   # v5 옆벽 창 z 범위: 귀 띠(|z|<6)·윗판(z>14)·앞뒤 고리만 남김
PIN_BORE_R = 1.0                     # 속 빈 핀 Ø6 속 Ø2: 고무줄 홈(Ø4.5) 벽 1.25 (Ø3 이면 0.75 로 얇아 11-12 키움)
ROLLER = (5.0, 3.25, 24.9, 29.3)     # v5 레버 층 핀 롤러(바깥 반지름, 안 반지름, y0, y1): 레버-핀 미끄럼 → 구름
PLUNGER_WIN_Y, PLUNGER_REAR_RING, PLUNGER_FRONT_KEEP = 16.0, 3.5, 5.0   # v5   # 아래 창 반폭, 뒤 고리 폭, 앞판 쪽 남김
RAMP = (30.0, 2.6)                   # 윗면 앞 30 mm 를 2.6 mm 낮춘 완만한 비탈(약 5°): 장전된 공 위에 얹혀 1.6 mm 내려온
                                     # 다음 공(과 그 위 공들)을 쏠 때 세게 차지 않고 천천히 들어 올림(시뮬: 안 하면 속도 -11 %)
X_REAR, X_MUZ = -DRAW - PLUNGER_L - 4.0, 22.0
# 플런저 귀(lug): 핀 구멍을 감싼 덩어리가 홈으로 나와(|y| ≤ 23.4) 홈 앞끝에 45° 면으로 닿아 멈춘다.
# 핀(Ø6)이 직접 부딪히면 굽힘 ~70 MPa·접촉 ~200 MPa 로 PLA 한계를 넘는다(durability.py). 귀는 면으로 받아 압축 ~40 MPa.
# 홈은 귀가 들어가도록 총신 뒤끝까지 트여 있다(플런저를 뒤에서 넣음).
LUG_C = -1.6                         # 귀 앞면·홈 앞끝 평면: x + |y| = LUG_C (플런저 쉴 때)
LUG_X0, LUG_R, LUG_H = -37.0, BARREL_R + 0.2, SLOT_W / 2 - 0.3
SLOT_X0, SLOT_X1 = X_REAR - 1.0, LUG_C - BORE_R   # 홈 x 범위(앞끝은 총신 안면 기준)
HOP_X = (-DRAW + 10.0, 2.0)          # 공 통 x 범위. 걸쇠에 걸린 공(가운데 -24.5)보다 통 가운데(-19)가 5.5 mm 앞이라
HOP_CX = (HOP_X[0] + HOP_X[1]) / 2   # 위 공이 장전된 공을 플런저 쪽으로 눌러 둠 → 낮은 각도에서도 굴러 나가지 않고, 첫 발·끝 발 속도 차 1~3 %
N_HOPPER = 5                         # 공 통 5개 = 5발. 총구로 넣지 않는다
HOP_TOP = BORE_R + N_HOPPER * 40.4 + 4.0
HOP_WALL = 1.6
HOP_SPLIT_Z = 72.0                   # 공 통을 여기서 나눔(몸통 쪽 짧은 통 + 따로 출력한 긴 통). 판 높이 256 안에 들게
HOP_SOCK = 10.0                      # 긴 통 아래 소켓 깊이(짧은 통 바깥에 끼움, 틈 0.2)
BAND_FREE = 31.7                     # 유니아트 4cm 고리가 늘림 0 일 때 핀-고리 거리(시뮬 기준값)
BAND_S0 = 18.0                       # v5: 쉴 때 늘림(mm). 클수록 힘이 고르게 → 같은 최대 힘에서 에너지 큼
HOOK_XS = (PIN_REST + BAND_FREE + BAND_S0, 12.5, 5.0)   # 고무줄 고리 x: 첫째 = 유니아트 4cm 기준 설계값(핀에서 49.7),
                                     # 둘째·셋째 = 센 고무줄(식당 고무줄 8자 두 겹 등)용 가까운 고리(42.5, 35). 실측해 고른다
HOOK_ZS = (0.0, 11.0, -9.0)          # 고리 높이: 먼 고리로 가는 고무줄이 가까운 고리 기둥을 스치지 않게 엇갈림
X_MUZ = max(X_MUZ, max(HOOK_XS) + 6.0)         # 고리가 앞으로 가면 총신을 늘려 고리 기둥을 붙인다

# 레버(DS3218): 축 y 방향, 혼은 -y(총신 쪽). 축은 핀 길보다 48 mm 위.
# 미는 면(팔 아래 면)이 축 중심을 지나는 직선이라 핀을 미는 방향이 팔에 수직 → 총신 축과 이루는 각이 장전 내내 30° 이하.
# (축이 핀 길 16 mm 아래였던 앞 설계는 처음 미는 각이 68° 라 마찰 0.2 이상에서 플런저가 걸려 멈췄다 — sim_full 확인)
LEVER_PIV = (-70.0, 42.0)            # (x, z) — v5 최적화(설계_v5.md). v4 (-57, 48)
LEVER_Y = (24.6, 29.6)               # 레버 팔 y(총신 바깥면 23.2 에서 1.4 띄움)
LEVER_W, LEVER_T = 9.0, 5.0
LEVER_GAP_DEG, LEVER_OVER = 10.0, 3.0   # 쉴 때 레버 면이 핀보다 앞선 각, 걸림 자리 뒤로 더 미는 거리
LEVER_STANDOFF = 4.2                 # 허브 받침: 레버 팔(29.6) → 혼 윗면(33.8). 걸쇠 이빨 층을 레버와 고무줄 사이에 넣느라 서보가 바깥으로 감
LEVER_SPLINE_Y = LEVER_Y[1] + LEVER_STANDOFF + HORN_TOP  # 축 끝 y
LEVER_FLANGE_Y = LEVER_SPLINE_Y + SERVO_L["spline_end"]  # 플랜지 윗면 y
LEVER_PLATE_T = 4.0
LEVER_TRIP_DEG = -24.0               # 발사 명령(앞으로 도는 쪽 -): 손가락이 걸쇠 꼬리를 밀어 이빨을 내림
LEVER_WALL_X = (LEVER_PIV[0] - 21.0, max(LEVER_PIV[0] + 41.0, -16.0))   # 레버 서보 판(세운 벽) x 범위(서보 몸통 둘레), 바닥부터 섬
LEVER_WALL_TOP = LEVER_PIV[1] + 12.0
LEVER_INSERT_DEG = 40.0              # 서보+레버를 위에서 내려 끼울 때 레버 각(팔이 아래로)
# 손가락(레버 둘째 팔): 쉴 때 걸쇠 꼬리 바로 뒤(시계 쪽). 레버가 앞(-)으로 돌면 꼬리를 민다. 장전 때는 꼬리에서 멀어진다.
FINGER_W = 7.0

# 걸쇠(sear): 고무줄로 닫히는 쪽으로 당겨진 수동 걸쇠. 축 y 방향(벽 밖에서 꽂는 출력 핀), 총신 뒤 (-100, 2).
# 레버가 쓸고 지나가는 원(축에서 반지름 62, 각 -35°~-124°) 밖이다.
SEAR_PIV = (-100.0, 2.0)             # (x, z). 핀을 잡는 힘선(z 0)이 축보다 2 아래 → 핀 힘이 걸쇠를 닫는 쪽으로 돌림
SEAR_Y = (30.6, 34.6)                # 이빨 층(레버와 고무줄 사이)
SEAR_YF = (LEVER_Y[0], SEAR_Y[1])    # 허브·꼬리·갈고리 팔은 두 층 다(출력 때 바닥에서 위로 쌓임)
SEAR_GAP = 3.5                       # v5: 장전 끝(-80) 핀 앞면 ↔ 이빨 걸림면. 행정을 거의 다 쓴다(v4 6.5 → 검토: +11 % 속도)
SEAR_FACE_X = PIN_COCK + PIN_R + SEAR_GAP    # 걸림면 x (-71.5)
SEAR_TOP = 2.0                       # 이빨 끝 z(핀 가운데 0 보다 2 위)
SEAR_ARM_Z = (-9.0, -4.0)            # 이빨 팔(핀 길 아래, 핀 밑면 -3 에서 1 띄움)
SEAR_TAIL = (30.0, 7.0)              # 꼬리: 축에서 레버 축 쪽으로 길이·폭
# ---- v5: 레버 축 위치에서 정해지는 값(손으로 맞추던 v4 값과 같은 규칙) ----
def _ang(x): return math.degrees(math.atan2(-LEVER_PIV[1], x - LEVER_PIV[0]))       # 축에서 핀(x, 0) 방향
SEAR_HOLD_X = PIN_COCK + SEAR_GAP                                                   # 걸렸을 때 핀 가운데 x
LEVER_REST_DEG = _ang(PIN_REST) + LEVER_GAP_DEG                                     # 쉴 때 미는 면 방향(핀보다 GAP 앞)
LEVER_COCK_DEG = round(_ang(PIN_REST) - _ang(SEAR_HOLD_X - LEVER_OVER) + LEVER_GAP_DEG + 1.0, 1)   # 장전 명령 각
LEVER_LEN = max(math.hypot(PIN_REST - LEVER_PIV[0], LEVER_PIV[1]), math.hypot(SEAR_HOLD_X - LEVER_OVER - 3 - LEVER_PIV[0], LEVER_PIV[1])) + 2.0
FINGER_OFF = 12.5                    # 쉴 때 손가락-꼬리 각 여유(축이 꼬리에 가까우면 손가락 폭이 큰 각이라 v4 9.5 보다 크게)
_ux, _uz = LEVER_PIV[0] - SEAR_PIV[0], LEVER_PIV[1] - SEAR_PIV[1]; _ul = math.hypot(_ux, _uz)
_TT = (SEAR_PIV[0] + _ux / _ul * SEAR_TAIL[0], SEAR_PIV[1] + _uz / _ul * SEAR_TAIL[0])   # 걸쇠 꼬리 끝
FINGER_DEG = math.degrees(math.atan2(_TT[1] - LEVER_PIV[1], _TT[0] - LEVER_PIV[0])) - FINGER_OFF   # 손가락: 꼬리 끝 방향보다 FINGER_OFF 뒤
FINGER_R = (11.0, math.hypot(_TT[0] - LEVER_PIV[0], _TT[1] - LEVER_PIV[1]) + 5.0)
SEAR_PEG = (-102.0, -15.0)           # 걸쇠 고무줄 갈고리(-y 로 튀어나옴, y 20.6~24.6)
SEAR_POST = (-55.0, -15.0)           # 걸쇠 고무줄 몸통 기둥(총신 옆, y ~24.2 까지)
SEAR_PIN_R = 2.5                     # 걸쇠 축 핀 Ø5(벽 밖에서 꽂음, 벽 구멍은 같은 지름 = 꽉 끼움, 걸쇠 구멍 +0.25)
SEAR_NOSE = (-111.0, -104.5, 1.0)    # 걸쇠 코(축 뒤, x0, x1, 밑면 z): 닫힐 때 내려와 멈춤 턱 윗면(z 1)에 앉는다
SEAR_STOP_X = (-111.0, -108.0)       # 닫힘 멈춤 턱(벽 → 이빨 층) x 범위. 윗면이 멈춤 면이라 출력 때 위를 봄(서포트 없음)
SEAR_WALL_X0 = -112.0                # 레버 서보 벽을 걸쇠 축 뒤까지 늘림

# 각도 축(몸통 아래)
AX = (-40.0, -45.0)                  # (x, z)
KEEL_X = (-70.0, -2.0)
KEEL_Y = 16.0                        # 몸통 축받이 |y|
KEEL_Z0 = AX[1] - 11.0
AXLE_R = 4.0                         # +y 축 핀 Ø8

def lever_servo_place(m): return place(m, (LEVER_PIV[0], LEVER_FLANGE_Y, LEVER_PIV[1]), (1, 0, 0), (0, 0, 1), (0, -1, 0))     # 몸통이 축 앞(+x), 오른손 좌표(u x v = w)

# ---------------- 본체 ----------------
def lever_plate():
    """DS3218 판: 바닥(축받이 밑면)부터 서는 벽(y = 플랜지 윗면 - 4 ~ 플랜지 윗면). 위가 트인 U 라서
    서보+혼+레버를 밖에서 다 조립해 위에서 내려 끼우고 바깥에서 플랜지 나사를 조인다.
    벽 앞쪽은 아래(축받이)와 위(공 통)로 이어진다. 둘 다 레버가 도는 자리 앞(x > -16)이고 고무줄 길(z ±5)을 피한다."""
    S = SERVO_L; y0, y1 = LEVER_FLANGE_Y - LEVER_PLATE_T, LEVER_FLANGE_Y
    xa, xb = LEVER_WALL_X
    p = box(xa, xb, y0, y1, KEEL_Z0, LEVER_WALL_TOP)
    sv = lever_servo_place(servo_solid(S, horn=True, clr=SERVO_CLR))
    slab = sv ^ box(-300, 300, y0 - 0.5, y1 + 0.5, -300, 300)
    p -= m3.Manifold.hull(slab + slab.translate([0, 0, 80]))          # 위에서 내려 끼우는 길
    p -= sv
    p -= lever_servo_place(servo_holes(S))
    p -= tear_y(8.0, y0 - 1, y1 + 1, AX[0], AX[1])                    # 축 핀·각도 서보 혼 나사 드라이버가 지나는 구멍
    fx0, fx1 = xb - 6.0, xb
    # 아래: 바닥 띠 + 45° 받침(축받이 옆면 → 벽)
    p += box(fx0, fx1, KEEL_Y - 1, y1, KEEL_Z0, KEEL_Z0 + 6)
    p += m3.Manifold.hull(box(fx0, fx1, KEEL_Y - 1, KEEL_Y, KEEL_Z0, -BARREL_R) + box(fx0, fx1, y0 - 0.5, y0, KEEL_Z0, KEEL_Z0 + 0.1))
    # 위: 공 통 옆면 → 벽, 밑면 45°
    tx0, tx1 = -16.0, -8.0
    p += m3.Manifold.hull(box(tx0, tx1, 14.0, 15.0, LEVER_WALL_TOP - 6 - (y0 - 15.0), LEVER_WALL_TOP) +
                          box(tx0, tx1, y0 - 0.5, y1, LEVER_WALL_TOP - 6, LEVER_WALL_TOP))
    # 뒤로 늘린 벽: 걸쇠 축 받침(벽 → 이빨 층 바로 바깥)과 닫힘 멈춤 턱(벽 → 이빨 층)
    sx, sz = SEAR_PIV
    p += m3.Manifold.hull(box(SEAR_WALL_X0, xa + 1, y0, y1, sz - 18.0, sz + 12.0) + box(SEAR_WALL_X0, xa + 1, y1 - 0.01, y1, sz - 22.0, sz + 12.0))
    p += tear_y_down(5.0, SEAR_Y[1] + 0.2, y0 + 0.5, sx, sz)
    zt = SEAR_NOSE[2]; sy0 = SEAR_Y[0] + 0.2
    p += box(SEAR_STOP_X[0], SEAR_STOP_X[1], sy0, SEAR_Y[1] + 0.3, zt - 2.0, zt)
    p += m3.Manifold.hull(box(SEAR_STOP_X[0], SEAR_STOP_X[1], SEAR_Y[1] + 0.2, SEAR_Y[1] + 0.21, zt - 2.0, zt) +
                          box(SEAR_STOP_X[0], SEAR_STOP_X[1], y0, y0 + 0.5, zt - 2.0 - (y0 - SEAR_Y[1] - 0.2), zt))      # 밑면 45°(걸쇠 층 밖에서부터)
    p -= cyl_y(SEAR_PIN_R, SEAR_Y[1], y1 + 1, sx, sz, 32)            # 걸쇠 축 핀 구멍(꽉 끼움)
    return p

def build_body():
    tube = tear_x(BARREL_R, X_REAR, X_MUZ)
    hop = cyl_z(BORE_R + HOP_WALL, 0, HOP_SPLIT_Z, HOP_CX)
    hooks = m3.Manifold()
    for hx, hz in zip(HOOK_XS, HOOK_ZS):
        for s in (-1, 1):
            y0, y1 = sorted((s * (math.sqrt(BARREL_R ** 2 - hz ** 2) - 1.5), s * (BAND_Y[1] + 1.0)))   # 기둥은 그 높이의 총신 겉면에서 시작
            hooks += tear_y_down45(3.0, y0, y1, hx, hz)
            y0, y1 = sorted((s * (BAND_Y[1] - 0.5), s * (BAND_Y[1] + 2.0)))
            hooks += tear_y_down45(4.5, y0, y1, hx, hz)
    keel = box(KEEL_X[0], KEEL_X[1], -KEEL_Y, KEEL_Y, KEEL_Z0, -BARREL_R + 4)
    # 총신 앞쪽 밑면 서포트 없애기: 축받이 앞끝에서 45° 로 올라가는 V 지느러미(총신 밑 V 능선 z -R√2 까지)
    rz = -BARREL_R * math.sqrt(2)
    vfin = m3.Manifold.hull(cyl_x(BARREL_R, KEEL_X[1] - 1, X_MUZ, 0, 0, 96) + box(KEEL_X[1] - 1, X_MUZ, -0.01, 0.01, rz, rz + 0.02))
    vfin ^= m3.Manifold.hull(box(KEEL_X[1] - 1, KEEL_X[1], -40, 40, KEEL_Z0, 40) + box(X_MUZ, X_MUZ + 0.01, -40, 40, KEEL_Z0 + (X_MUZ - KEEL_X[1]), 40))
    keel += vfin
    sx, sz = SEAR_PIV
    # 걸쇠가 -y 로 밀려 레버 층에 들어가지 않게 받는 덧살(총신 옆, 홈 위): 걸쇠 허브 면(24.6)에서 0.2
    pad = m3.Manifold.hull(box(sx - 3.5, sx + 4.0, 21.0, LEVER_Y[0] - 0.2, sz + 3.5, sz + 7.0) + box(sx - 3.5, sx + 4.0, 21.0, 21.1, sz, sz + 7.0))
    # 걸쇠 고무줄 기둥(총신 옆, 끝에 턱). 밑은 용골(서포트 없음)
    px, pz = SEAR_POST
    post = tear_y_down(2.5, 16.0, LEVER_Y[0] - 1.0, px, pz) + tear_y_down(3.3, LEVER_Y[0] - 1.0, LEVER_Y[0] - 0.4, px, pz)
    body = tube + hop + hooks + keel + lever_plate() + pad + post
    body -= tear_x(BORE_R, X_REAR - 1, X_MUZ + 1)
    body -= cyl_z(BORE_R, 0, HOP_SPLIT_Z + 1, HOP_CX)
    body -= m3.Manifold.cylinder(0.61, BORE_R, BORE_R + 0.6, 96).translate([HOP_CX, 0, HOP_SPLIT_Z - 0.6])   # 안쪽 윗모서리 모따기(공이 턱에 안 걸림)
    for s in (-1, 1):
        # 홈: 뒤끝은 트임, 앞끝은 45° 면(x + |y| = LUG_C), 천장은 벽 두께 가운데로 45° 지붕
        yi, yo = BORE_R - 1.0, BARREL_R + 1.0
        ym, h = (yi + yo) / 2, (yo - yi) / 2
        wy = lambda y: box(SLOT_X0, LUG_C - y, s * y - 0.005, s * y + 0.005, -SLOT_W / 2, SLOT_W / 2)
        ridge = box(SLOT_X0, LUG_C - ym, s * ym - 0.005, s * ym + 0.005, SLOT_W / 2 + h - 0.02, SLOT_W / 2 + h)
        body -= (wy(yi) + wy(yo) + ridge).hull()
    # 각도 축: +y 축 핀 구멍, -y 쪽은 각도 서보 혼을 나사로 붙이는 면(가운데 Ø7, 혼 나사 홈 r6~10)
    body -= tear_y(AXLE_R + 0.2, 0, KEEL_Y + 1, AX[0], AX[1])
    body -= cyl_y(3.5, -KEEL_Y - 1, -KEEL_Y + 6, AX[0], AX[1], 32)
    body -= tear_y(2.6, -KEEL_Y + 5, 1.0, AX[0], AX[1])          # 각도 서보 혼 가운데 나사: 축 핀 구멍(+y)으로 드라이버가 지나는 길
    for a in range(4):
        th = math.radians(45 + 90 * a)
        for r in np.arange(6.0, 10.01, 0.5):
            body -= cyl_y(1.0, -KEEL_Y - 1, -KEEL_Y + 8, AX[0] + r * math.cos(th), AX[1] + r * math.sin(th), 12)
    body -= lever_sweep()
    parts = [q for q in body.decompose() if q.volume() > 1.0]          # 레버 쓸림 자리를 깎고 남는 부스러기(0.1 mm³) 제거
    body = parts[0]
    for q in parts[1:]: body += q
    return body

def lever_sweep(margin=1.0):
    """레버(팔·손가락)가 발사~장전+8° 에서 쓸고 지나가는 자리(+margin). 몸통에서 깎는다."""
    out = m3.Manifold()
    angs = np.arange(LEVER_TRIP_DEG - 4.0, LEVER_COCK_DEG + 8.01, 1.0)
    for pc in lever_pieces(margin):
        for a0, a1 in zip(angs[:-1], angs[1:]):
            out += m3.Manifold.hull(lever_at(a0, pc) + lever_at(a1, pc))
    return out

# ---------------- 움직이는 부품 ----------------
def build_plunger():
    """면이 x=0, 뒤로 PLUNGER_L. 컵(가볍게), 핀 구멍 x=-30, 앞면 테두리 모따기."""
    p = cyl_x(PLUNGER_R, -PLUNGER_L, 0)
    # 속 빈 곳 앞끝도 앞면 모따기와 같은 기울기로 깎는다(벽 두께 유지). 예전엔 모따기(4)가 앞판(2.4)보다 깊어
    # 앞판과 옆벽이 0.2 mm 띠로만 이어졌고, 출력물에서 앞판이 통째로 떨어졌다(26-11-12).
    s = (PLUNGER_CHAMF + 1) / PLUNGER_CHAMF                       # 바깥 모따기 기울기(반지름/축)
    r_o = lambda d: PLUNGER_R - PLUNGER_CHAMF + s * d               # 앞면에서 d 뒤의 바깥 모따기 반지름
    off = PLUNGER_WALL * math.hypot(1, s)                           # 같은 벽 두께만큼 안쪽
    cone_in = m3.Manifold.cylinder(12 - PLUNGER_FACE, r_o(12) - off, r_o(PLUNGER_FACE) - off, 96).rotate([0, 90, 0]).translate([-12, 0, 0])
    p -= (cyl_x(PLUNGER_R - PLUNGER_WALL, -PLUNGER_L - 1, -12) + (cyl_x(PLUNGER_R - PLUNGER_WALL, -12, -PLUNGER_FACE) ^ cone_in))
    # 핀 받침: 벽 안쪽 짧은 토막만(핀이 가운데를 건너간다). 통째 원기둥(3.8 cm³)은 무게만 늘려 공 속도를 깎았다.
    yb = PLUNGER_R - PLUNGER_WALL - PLUNGER_BOSS_IN
    boss = cyl_y(5.5, -PLUNGER_R + 0.5, PLUNGER_R - 0.5, PIN_REST, 0) - cyl_y(6, -yb, yb, PIN_REST, 0)
    p += boss ^ cyl_x(PLUNGER_R - 0.2, -PLUNGER_L, 0)
    # 아래 창: 윗면(공 통 막는 문)·옆(귀·홈 따라 미끄러짐)·앞판·뒤 고리는 그대로, 아래쪽 벽만 덜어 낸다
    # 출력 때 위가 되는 뒤 끝은 45° 지붕(V)으로 — 넓은 다리 놓기 없이 서포트 없이 뽑힌다
    wx0, wx1 = -PLUNGER_L + PLUNGER_REAR_RING, -PLUNGER_FRONT_KEEP
    p -= (box(wx0 + PLUNGER_WIN_Y, wx1, -PLUNGER_WIN_Y, PLUNGER_WIN_Y, -PLUNGER_R - 2, -9.0) +
          box(wx0, wx0 + 0.01, -0.01, 0.01, -PLUNGER_R - 2, -9.0)).hull()
    x0w, x1w = -PLUNGER_L + PLUNGER_REAR_RING, -PLUNGER_FRONT_KEEP - 6.0
    for zl, zh in PLUNGER_SIDE_WIN:      # 옆벽 창(출력 때 위가 되는 뒤 끝은 45° 지붕)
        for s_ in (-1, 1):
            y0, y1 = sorted((s_ * 8.0, s_ * 30.0))
            p -= (box(x0w + (zh - zl) / 2, x1w, y0, y1, zl, zh) + box(x0w, x0w + 0.01, y0, y1, (zl + zh) / 2 - 0.01, (zl + zh) / 2 + 0.01)).hull()
    for s in (-1, 1):                # 귀: 앞면 45°(x + |y| = LUG_C) — 출력 때 아래를 향하는 면이 45° 라 서포트 없음
        p += (box(LUG_X0, LUG_C - 15.0, s * 15.0 - 0.005, s * 15.0 + 0.005, -LUG_H, LUG_H) +
              box(LUG_X0, LUG_C - LUG_R, s * LUG_R - 0.005, s * LUG_R + 0.005, -LUG_H, LUG_H)).hull()
    p -= cyl_y(PIN_R + 0.2, -30, 30, PIN_REST, 0, 32)
    cone = m3.Manifold.cylinder(PLUNGER_CHAMF + 0.01, PLUNGER_R + 1, PLUNGER_R - PLUNGER_CHAMF, 96).rotate([0, 90, 0]).translate([-PLUNGER_CHAMF, 0, 0])
    p = (p - box(-PLUNGER_CHAMF, 1, -30, 30, -30, 30)) + (p ^ cone)
    ang = math.degrees(math.atan2(RAMP[1], RAMP[0]))
    p -= box(-200, 50, -40, 40, 0, 60).rotate([0, ang, 0]).translate([0, 0, PLUNGER_R - RAMP[1]])
    return p

def build_pin():
    """핀 Ø6, y -42~43.7 + -y 끝 머리 Ø8. 고무줄 홈 |y| 35.6~40.6 은 Ø4.5. 바닥 0.5 mm 평면(눕혀 출력)."""
    pin = cyl_y(PIN_R, PIN_Y[0], PIN_Y[1], 0, 0, 48)
    pin += cyl_y(PIN_HEAD[2], PIN_HEAD[0], PIN_HEAD[1], 0, 0, 48)
    for s in (-1, 1):
        y0, y1 = sorted((s * BAND_Y[0], s * BAND_Y[1]))
        pin -= cyl_y(PIN_R + 1, y0, y1, 0, 0, 48) - cyl_y(2.25, y0, y1, 0, 0, 48)
    pin -= box(-6, 6, -60, 60, -6, -PIN_R + 0.5)
    pin -= cyl_y(PIN_BORE_R, PIN_HEAD[0] - 1, PIN_Y[1] + 1, 0, 0, 24)   # 속 빈 핀(Ø3 구멍은 다리 놓기로 뽑힘)
    return pin

def build_roller():
    """레버 층 핀 롤러: 핀(Ø6)에 끼워 레버가 굴리며 민다. 세워 출력."""
    ro, ri, y0, y1 = ROLLER
    return cyl_z(ro, 0, y1 - y0, 0, 0, 64) - cyl_z(ri, -1, y1 - y0 + 1, 0, 0, 48)

def horn_mount(r_hub, t):
    """혼 위 허브: 가운데 Ø7(혼 나사 머리), 혼 나사 홈 4개(폭 2.2, r 6~10, 레버 방향과 45° 어긋남)."""
    hub = cyl_z(r_hub, 0, t, 0, 0, 64)
    hub -= cyl_z(3.5, -1, t + 1, 0, 0, 32)
    for a in range(4):
        sl = box(6, 10, -1.1, 1.1, -1, t + 1) + cyl_z(1.1, -1, t + 1, 6, 0, 16) + cyl_z(1.1, -1, t + 1, 10, 0, 16)
        hub -= sl.rotate([0, 0, 45 + 90 * a])
    return hub

def lever_pieces(margin=0.0):
    """레버의 볼록 조각들(쉴 때 방향, 레버 자기 좌표: x-y 평면 = 발사대 x-z, z 0~T = y 29.6→24.6)."""
    m = margin
    ra = LEVER_REST_DEG
    arm = box(0, LEVER_LEN, -m, LEVER_W + m, -m, LEVER_T + m).rotate([0, 0, ra])
    tip = cyl_z(LEVER_W / 2 + m, -m, LEVER_T + m, LEVER_LEN, LEVER_W / 2, 32).rotate([0, 0, ra])
    fg = m3.Manifold.hull(box(0, FINGER_R[1], -FINGER_W - m, m, -m, LEVER_T + m) +
                          cyl_z(FINGER_W / 2 + m, -m, LEVER_T + m, FINGER_R[1], -FINGER_W / 2, 32))
    return [arm, tip, fg.rotate([0, 0, FINGER_DEG])]

def build_lever():
    """혼 허브(받침 LEVER_STANDOFF 포함) + 미는 팔 + 손가락. 미는 면(팔 아래 면)이 축 중심을 지난다.
    출력: 팔 면(자기 좌표 z = T, 총신 쪽 면)이 판에 닿게 뒤집어 놓는다."""
    t = LEVER_T + LEVER_STANDOFF
    hub = horn_mount(12.5, t).translate([0, 0, -LEVER_STANDOFF])
    for a in range(4):                                  # 혼 나사 머리 자리: 혼 쪽 3 mm 만 남기고 팔 쪽에서 판다
        th = math.radians(45 + 90 * a)
        hub -= m3.Manifold.hull(cyl_z(2.6, -LEVER_STANDOFF + 3.0, LEVER_T + 1, 6 * math.cos(th), 6 * math.sin(th), 24) +
                                cyl_z(2.6, -LEVER_STANDOFF + 3.0, LEVER_T + 1, 10 * math.cos(th), 10 * math.sin(th), 24))
    arms = m3.Manifold()
    for pc in lever_pieces():
        arms += pc
    arms -= cyl_z(12.4, -1, LEVER_T + 1, 0, 0, 64)
    return arms + hub

def build_sear():
    """걸쇠(닫힌 자세, 발사대 좌표). 이빨 팔은 이빨 층만, 허브·꼬리·갈고리 팔은 레버 층까지(두께 10).
    출력: 이빨 층 바깥 면(y 34.6)이 판에 닿고 -y 로 쌓인다(서포트 없음)."""
    sx, sz = SEAR_PIV
    y0, y1 = SEAR_YF; t0, t1 = SEAR_Y
    hub = cyl_y(5.0, y0, y1, sx, sz, 48)
    ux, uz = LEVER_PIV[0] - sx, LEVER_PIV[1] - sz; L = math.hypot(ux, uz); ux, uz = ux / L, uz / L
    tl, tw = SEAR_TAIL; nx, nz = -uz, ux
    pts = [(sx - nx * tw / 2, sz - nz * tw / 2), (sx + ux * tl - nx * tw / 2, sz + uz * tl - nz * tw / 2),
           (sx + ux * tl + nx * tw / 2, sz + uz * tl + nz * tw / 2), (sx + nx * tw / 2, sz + nz * tw / 2)]
    tail = prism_xz(pts, y0, y1)
    if tail.volume() < 1: tail = prism_xz(pts[::-1], y0, y1)
    gx, gz = SEAR_PEG
    tab = m3.Manifold.hull(box(gx - 3.0, gx + 3.0, y0, y1, gz, sz) + cyl_y(3.0, y0, y1, gx, gz, 32)) + box(sx - 4.0, sx + 1.0, y0, y1, gz, sz)
    peg = cyl_y(2.5, 21.7, y0 + 0.01, gx, gz, 32)
    peg += m3.Manifold.cylinder(0.8, 3.3, 2.5, 32).rotate([-90, 0, 0]).translate([gx, 20.9, gz])     # 45° 턱(고무줄 빠짐 막이)
    peg += cyl_y(3.3, 20.6, 20.91, gx, gz, 32)
    fx, top = SEAR_FACE_X, SEAR_TOP; a0, a1 = SEAR_ARM_Z
    arm = prism_xz([(sx - 1.0, a0), (fx + 8.0, a0), (fx + 8.0, a1), (fx + 2.0, top), (fx, top), (fx, a1), (sx - 1.0, a1)], t0, t1)
    nose = box(SEAR_NOSE[0], SEAR_NOSE[1], y0, y1, SEAR_NOSE[2], sz + 3.0)
    s = hub + tail + tab + peg + arm + nose
    s -= cyl_y(SEAR_PIN_R + 0.25, y0 - 1, y1 + 1, sx, sz, 32)
    return s

def build_sear_pin():
    """걸쇠 축 핀 Ø5: 벽 바깥에서 꽂아 벽(꽉)·걸쇠(+0.25)를 지나 y 25 까지. 머리 Ø8 x 2.5. 눕혀 출력(바닥 0.5 평면)."""
    y1 = LEVER_FLANGE_Y
    p = cyl_y(SEAR_PIN_R, 25.0, y1, 0, 0, 32) + cyl_y(4.0, y1, y1 + 2.5, 0, 0, 32)
    return p - box(-6, 6, 20, y1 + 5, -6, -SEAR_PIN_R + 0.5)

def build_axle_pin():
    """+y 축 핀: Ø8 x (받침대 볼 4 + 틈 1 + 몸통 10), 머리 Ø14 x 3. 눕혀 출력(바닥 0.5 평면)."""
    p = cyl_y(AXLE_R, 0, 15.0, 0, 0, 48) + cyl_y(7.0, 15.0, 18.0, 0, 0, 48)
    return p - box(-8, 8, -1, 19, -8, -AXLE_R + 0.5)

def build_hopper_tube():
    """공 통 위쪽(따로 출력): z HOP_SPLIT_Z-HOP_SOCK ~ HOP_TOP, 발사대 좌표 그대로. 아래 소켓이 몸통 짧은 통 바깥에 끼인다(틈 0.2).
    소켓 천장은 45° 원뿔(서포트 없음, 짧은 통 윗모서리에 앉음). 출력: 소켓 쪽이 판."""
    ri, ro = BORE_R, BORE_R + HOP_WALL
    z0, zs = HOP_SPLIT_Z - HOP_SOCK, HOP_SPLIT_Z
    rs = ro + 0.2; rso = rs + 1.6
    t = cyl_z(ro, zs, HOP_TOP, 0, 0, 96) + cyl_z(rso, z0, zs + 0.2, 0, 0, 96)
    # 바구니: 소켓 위 8 mm 와 맨 위 뚜껑 자리 12 mm 는 통, 그 사이는 세로 기둥 8개(폭 5) + 고리(높이 3, 간격 32).
    # 창 크기 약 12 x 29 라 공(Ø40)이 빠지지 않고, 기둥·고리 안쪽 면은 통 안면(Ø41.6) 그대로라 공이 걸리지 않는다.
    # 창 위쪽은 45° 지붕이라 서포트·다리가 없다.
    za, zb = zs + 8.0, HOP_TOP - 12.0
    n_win = max(1, int(round((zb - za) / 32.0)))
    pitch = (zb - za) / n_win
    for i in range(8):
        for j in range(n_win):
            w0, w1 = za + j * pitch + (3.0 if j else 0.0), za + (j + 1) * pitch
            win = m3.Manifold.hull(box(0, ro + 2, -6.0, 6.0, w0, w1 - 6.0) + box(0, ro + 2, -0.01, 0.01, w0, w1)).rotate([0, 0, 22.5 + 45 * i])   # 창 위는 45° 지붕
            t -= win
    t += m3.Manifold.cylinder(rso - ro, rso, ro, 96).translate([0, 0, zs + 0.2])          # 바깥 45°
    t -= cyl_z(ri, z0 - 1, HOP_TOP + 1, 0, 0, 96)
    t -= cyl_z(rs, z0 - 1, zs, 0, 0, 96)
    t -= m3.Manifold.cylinder(rs - ri + 0.01, rs, ri, 96).translate([0, 0, zs - 0.005])      # 소켓 천장 45°
    return t.translate([HOP_CX, 0, 0])

def build_hopper_cap():
    """공 통 뚜껑: 통 바깥에 끼움(틈 0.3), 깊이 8, 윗판 2."""
    ro = BORE_R + HOP_WALL
    cap = cyl_z(ro + 0.3 + 1.6, 0, 10.0, 0, 0, 96) - cyl_z(ro + 0.3, -1, 8.0, 0, 0, 96)
    return cap

# ---------------- 조립 자세 ----------------
def lever_at(th, lever=None):
    lever = lever or build_lever()
    return lever.rotate([90, 0, 0]).rotate([0, th, 0]).translate([LEVER_PIV[0], LEVER_Y[1], LEVER_PIV[1]])
def sear_at(s_deg, sear=None):
    """걸쇠를 s_deg 만큼 여는 쪽(+, 옆에서 보아 시계 = 이빨이 내려감)으로 돌린 자세."""
    sear = sear or build_sear()
    sx, sz = SEAR_PIV
    return sear.translate([-sx, 0, -sz]).rotate([0, s_deg, 0]).translate([sx, 0, sz])
def sear_pin_at(pin=None):
    return (pin or build_sear_pin()).translate([SEAR_PIV[0], 0, SEAR_PIV[1]])

if __name__ == "__main__":
    body = build_body()
    parts = {"body": body, "plunger": build_plunger(), "pin": build_pin(), "lever": build_lever(), "sear": build_sear(),
             "sear_pin": build_sear_pin(), "axle_pin": build_axle_pin(), "hopper_tube": build_hopper_tube(), "hopper_cap": build_hopper_cap(), "roller": build_roller()}
    meta = {}
    for n, mf in parts.items():
        tm = to_t(mf)
        meta[n] = dict(pieces=len(tm.split()), watertight=bool(tm.is_watertight), vol_cm3=round(tm.volume / 1000, 2),
                       ext=[round(v, 1) for v in tm.extents])
    # 출력 방향: 몸통은 축받이 바닥이 판, 플런저는 면이 판, 핀·축 핀·걸쇠 핀은 평면이 판, 레버는 팔 면(총신 쪽)이 판,
    # 걸쇠는 이빨 층 바깥 면(+y)이 판, 뚜껑은 윗판이 판
    pr = {"body": body, "plunger": parts["plunger"].rotate([0, 90, 0]), "pin": parts["pin"], "lever": parts["lever"].rotate([180, 0, 0]),
          "sear": parts["sear"].rotate([-90, 0, 0]), "sear_pin": parts["sear_pin"], "axle_pin": parts["axle_pin"], "hopper_tube": parts["hopper_tube"],
          "hopper_cap": parts["hopper_cap"].rotate([180, 0, 0]), "roller": parts["roller"]}
    for n, mf in pr.items():
        tm = to_t(mf); tm.apply_translation([0, 0, -tm.bounds[0][2]]); tm.export(os.path.join(OUT, f"print_{n}.stl"))
    to_t(body).export(os.path.join(OUT, "local_body.stl"))
    meta["AX"] = AX
    json.dump(meta, open(os.path.join(D, "build_meta.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    for k, v in meta.items(): print(k, v)
