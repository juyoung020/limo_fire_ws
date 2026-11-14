"""열화상(GY-MCU90640) + 레이더(HLK-LD2450) 팔 v4 — "등뼈와 눈썹".

부품 3 개 + 볼트 2 개:
- 팔(build): 지붕 앞 M3 구멍 2 개 → 닫힌 뿌리 상자(밑판·기둥·옆 웹 2) → 속 빈 상자 다리 → 경사 → 카메라 위 판
  → 가로대(열화상 뒤 받침, 속 빈 12×12) → 레이더 쪽 혀(오각형, 아래가 45° V). 옆으로 눕혀 출력(−y 면이 판).
- 레이더 받침(radar_holder): 15° 뒤로 누운 칼날. 머리의 굴(아래가 45° V)이 혀에 +y 쪽에서 끼워지고 뚜껑 볼트 기둥이 지붕 구멍에 꽂혀 못 빠짐.
  윗면(z 68.3)을 판에 대고 뒤집어 출력.
- 뚜껑(caps): 눈썹 전체를 덮는 챙. 앞 치마(열화상 창)·옆 볼·뒤 턱이 열화상 보드를 감싸고, 챙 밑면이 두 보드 윗변을 누름.
  윗면을 판에 대고 뒤집어 출력. 볼트는 일자 홈 둥근 머리(F.pan_bolt), 자리파기에 파묻힘.

좌표 = 지붕 좌표(x 앞, y 왼, z 위). 치수 출처:
- 지붕 앞 구멍 (49.4, ±9.5), 그 자리 지붕 윗면 z 2.23 (render/asm/limo.stl 단면)
- 머리: 뒷벽 x 53.7~53.9, 윗면 z 23.03 (x 56~78), 경사 (80, 28.1) (82.6, 32) (83.9, 36~41) (86.1, 44), 카메라 윗면 z 46.9 (x 88~108.1)
- GY-MCU90640 보드 15.5 × 25 (사용자 매뉴얼), 두께 1.6·렌즈 통 Ø9.2 × 6 가정. 렌즈가 긴 변 어디에 있는지는 매뉴얼에 치수가 없음(사진은 한쪽 끝) → 창은 |y| ≤ 11.3 전부 엶
- HLK-LD2450 보드 44 × 15 × 1.1 (매뉴얼 4.1), 소켓은 뒷면 한쪽 끝(아래로 둠), 시야 ±60° / ±35°
"""
import math
import os
import sys

import manifold3d as m3
import numpy as np
import trimesh

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(D, "..", "lib"))
import fasteners as F
from shape import fillet as _fillet

fillet = lambda A, r: _fillet(A, r, 32)
OUT = os.path.join(D, "stl"); os.makedirs(OUT, exist_ok=True)
PRN = os.path.join(D, "print"); os.makedirs(PRN, exist_ok=True)
SEG = 48


# ---------------- 도우미 ----------------
def box(x0, x1, y0, y1, z0, z1):
    return m3.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])


def cyl_z(r, z0, z1, x, y):
    return m3.Manifold.cylinder(z1 - z0, r, r, SEG).translate([x, y, z0])


def to_t(mf):
    me = mf.to_mesh()
    return trimesh.Trimesh(me.vert_properties[:, :3], me.tri_verts)


def _cs(pts):
    p = np.asarray(pts, float)
    a = 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
    return m3.CrossSection([p if a > 0 else p[::-1]])


def ext_y(cs, y0, y1):
    """(x, z) 단면을 y0 → y1 로 밀기."""
    if not isinstance(cs, m3.CrossSection): cs = _cs(cs)
    return cs.extrude(y1 - y0).transform([[1, 0, 0, 0], [0, 0, -1, y1], [0, 1, 0, 0]])


def ext_x(cs, x0, x1):
    """(y, z) 단면을 x0 → x1 로 밀기."""
    if not isinstance(cs, m3.CrossSection): cs = _cs(cs)
    return cs.extrude(x1 - x0).transform([[0, 0, 1, x0], [1, 0, 0, 0], [0, 1, 0, 0]])


def ext_z(cs, z0, z1):
    if not isinstance(cs, m3.CrossSection): cs = _cs(cs)
    return cs.extrude(z1 - z0).translate([0, 0, z0])


def louvre(c, half_len=6.0, half_w=1.5):
    """비늘창 홈 단면(긴 축이 첫째 좌표): 폭 3, 길이 12, 양 끝 45° 뾰족."""
    u, v = c
    return [(u - half_len, v), (u - half_len + half_w, v + half_w), (u + half_len - half_w, v + half_w),
            (u + half_len, v), (u + half_len - half_w, v - half_w), (u - half_len + half_w, v - half_w)]


# ---------------- 치수 ----------------
ROOF_Z = 2.25                         # 구멍 둘레 지붕 윗면(메시 2.23)
HOLES = [(49.4, s * 9.5) for s in (-1, 1)]
HEAD_TOP, PILL_TOP = 23.0, 46.8
W = 15.0                              # 등뼈 반폭. 옆면은 곧음(눕혀 출력하는 판 면·웹과 드라이버 길 때문에 4° 기울임은 안 함)
T = 4.0                               # 밑판·카메라 위 판 두께
TOP_Z0 = PILL_TOP + 2.0               # 카메라 위 판 아랫면 48.8
ZT = TOP_Z0 + T                       # 판 윗면 52.8 = 열화상 보드 아래 끝
COL_X0, COL_X1 = 40.5, 43.7           # 기둥(두께 3.2)
LEG_Z0, LEG_Z1 = HEAD_TOP + 2.0, HEAD_TOP + 9.0   # 상자 다리 25 ~ 32
M5 = (5.0, 1.25, 0.4)

# 열화상: 세워서 정면(+x), 25 가로(y) × 15.5 세로(z)
TH_W, TH_H, TH_T = 25.0, 15.5, 1.6
BEZEL_X = 104.0                       # 눈썹 앞면(치마 겉면)
TH_X = BEZEL_X - 1.75 - 0.25 - TH_T   # 보드 뒷면 x 100.4 (렌즈 쪽 면 102.0)
TH_Z0 = 0.0                           # 보드 아래 끝 = 판 윗면
Z_TOP = ZT + TH_Z0 + TH_H             # 두 보드 윗변·뚜껑 밑면 68.3
TH_LENS_Z = ZT + TH_Z0 + TH_H / 2
BAR_X0, BAR_X1 = 83.4, 95.4           # 가로대(열화상 보드 뒤 5 mm 가 앞면)
BAR_Z1 = TOP_Z0 + 12.0                # 60.8
TB_X = 88.0                           # 열화상 쪽 볼트 축
BOSS_TOP = 66.0                       # 볼트 기둥·혀 윗면

# 레이더: 긴 변을 15° 뒤로 눕힘, 패치 면이 앞·위
RD_L, RD_W, RD_T = 44.0, 15.0, 1.1
RD_TILT = 15.0
RD_Y = 53.0
_c, _s = math.cos(math.radians(RD_TILT)), math.sin(math.radians(RD_TILT))
RD_N = np.array([_c, 0, _s])          # 보드 면 법선(앞·위)
RD_U = np.array([-_s, 0, _c])         # 보드 긴 축(위·뒤)
RD_PT = np.array([BEZEL_X - 0.6, RD_Y, Z_TOP])        # 보드 앞면 윗변 가운데
RD_PB = RD_PT - RD_L * RD_U                           # 보드 앞면 아랫변 가운데 (114.79, 53, 25.80)
RD_X = float(RD_PB[0])
RD_Z0 = float(RD_PB[2])
RB_X = (BAR_X0 + BAR_X1) / 2          # 레이더 쪽 볼트 축 89.4
TG_Y0, TG_Y1 = 43.3, 62.4             # 혀
HD_Y0, HD_Y1 = 43.6, 62.4             # 받침(폭 18.8)
TONGUE = [(BAR_X0, BOSS_TOP), (BAR_X0, 61.0), (RB_X, 55.0), (BAR_X1, 61.0), (BAR_X1, BOSS_TOP)]   # (x, z)
FIT = 0.25                            # 혀 ↔ 굴 한쪽 틈

# 뚜껑·볼트
CAP_T = 2.0
CAP_TOP = Z_TOP + CAP_T               # 70.3
HEAD_D, HEAD_H = 9.0, 2.4
HEAD_Z = CAP_TOP - HEAD_H - 0.2       # 볼트 머리 밑면 67.7
BOLT_L = 9.5                          # 나사 길이: 67.7 → 58.2 (혀의 V 면 위 0.7 에서 멈춤, 물림 7.8)
THREAD_D = 8.5                        # 암나사 깊이(66 → 57.5)
BOLT_POS = [(TB_X, 0.0), (RB_X, RD_Y)]
CAP_X0 = 81.55
CAP_OUT = [(CAP_X0, -W), (BEZEL_X + 2.0, -W), (BEZEL_X + 2.0, 20.0), (BEZEL_X + 0.4, 40.0), (BEZEL_X + 0.4, HD_Y1), (CAP_X0, HD_Y1)]


def rd_local(m):
    """레이더 국소 좌표(a = 법선, b = y − 53, c = 긴 축, 원점 = 보드 앞면 아랫변 가운데) → 지붕 좌표."""
    return m.rotate([0, -RD_TILT, 0]).translate([float(v) for v in RD_PB])


def lbox(a0, a1, b0, b1, c0, c1):
    return box(a0, a1, b0, b1, c0, c1)


# ---------------- 팔 ----------------
RAMP = [(77.0, LEG_Z0), (83.0, TOP_Z0), (BAR_X0 + 0.1, TOP_Z0), (BAR_X0 + 0.1, BAR_Z1), (80.36, BAR_Z1), (71.67, 26.34)]      # 경사 판 두께 5.5(비틀림 강성)


def _arm_envelope():
    b = box(COL_X0, 53.0, -W, W, ROOF_Z, ROOF_Z + T)                         # 밑판
    b += box(COL_X0, COL_X1, -W, W, ROOF_Z, LEG_Z1)                          # 기둥
    b += box(COL_X0, 77.0, -W, W, LEG_Z0, LEG_Z1)                            # 다리(속은 나중에 비움)
    for s in (-1, 1):                                                        # 옆 웹: 뿌리 C 를 상자로
        b += box(COL_X1 - 0.2, 52.5, min(s * 13.0, s * W), max(s * 13.0, s * W), ROOF_Z + T - 0.2, LEG_Z0 + 0.2)   # 맞닿는 면은 0.2 겹쳐 틈이 안 생기게
    b += ext_y(RAMP, -W, W)                                                  # 경사 + 가로대 뒤 받침 살
    b += box(79.47, BEZEL_X, -W, W, TOP_Z0, ZT)                              # 카메라 위 판
    b += box(BAR_X0, BAR_X1, -W, TG_Y0, TOP_Z0, BAR_Z1)                      # 가로대
    b += ext_x([(TG_Y0 - 5.4, BAR_Z1 - 0.2), (TG_Y0 - 5.2, BAR_Z1), (TG_Y0, BOSS_TOP), (TG_Y0, BAR_Z1 - 0.2)], BAR_X0, BAR_X1)   # 혀 높이로 45° 오름
    b += ext_y(TONGUE, TG_Y0 - 0.3, TG_Y1)                                  # 혀
    b += box(TB_X - 5.2, TB_X + 5.2, -5.2, 5.2, BAR_Z1 - 0.2, BOSS_TOP)     # 열화상 쪽 볼트 기둥
    b += ext_x([(-5.0, BAR_Z1 - 0.2), (-5.0, BOSS_TOP), (-5.2, BOSS_TOP), (-5.4 - (BOSS_TOP - BAR_Z1), BAR_Z1 - 0.2)], TB_X - 5.2, TB_X + 5.2)  # 기둥 밑 45° 용골(출력 때 아래)
    return b


# 등 경사 덮개: 다리 윗면(수평) → 가파른 경사 판 → 가로대로 이어지던 계단을 경사면 한 장으로 덮는다(앞유리 53° 와 비슷한 약 49°).
# 속 빈 쐐기(경사 판 2.4, 옆벽 2.0). 비늘창 3 줄이 숨구멍을 겸함.
FAIR_X0, FAIR_X1 = 55.0, 80.5          # 드라이버 구멍(x ≤ 53) 앞에서 시작
FAIR_Y = W                             # 등뼈 옆면과 같은 면(눕혀 출력할 때 바닥 쪽 옆벽이 판에 닿게; 2 안쪽이면 판 위 2 mm 에 뜸)
FAIR_T, FAIR_WALL = 2.4, 2.0
_fdx, _fdz = FAIR_X1 - FAIR_X0, BAR_Z1 - LEG_Z1
FAIR_ANG = math.degrees(math.atan2(_fdz, _fdx))
FAIR_LEN = math.hypot(_fdx, _fdz)


def _fair_tri(shift=0.0):
    """(x, z) 세모: 빗변을 안쪽으로 shift 만큼 평행 이동."""
    sx = shift / math.sin(math.radians(FAIR_ANG))
    return [(FAIR_X0 + sx, LEG_Z1 - 0.2), (FAIR_X1 + 0.3, BAR_Z1 - sx * _fdz / _fdx), (FAIR_X1 + 0.3, LEG_Z1 - 0.2)] if shift else            [(FAIR_X0, LEG_Z1 - 0.2), (FAIR_X1, BAR_Z1), (FAIR_X1 + 0.3, BAR_Z1), (FAIR_X1 + 0.3, LEG_Z1 - 0.2)]


def build():
    env = _arm_envelope()
    b = fillet(env + ext_y(_fair_tri(), -FAIR_Y, FAIR_Y), 0.8)
    b -= ext_y(_fair_tri(FAIR_T), -(FAIR_Y - FAIR_WALL), FAIR_Y - FAIR_WALL) - env        # 쐐기 속 비움(다리·경사 판은 그대로)
    th = math.radians(FAIR_ANG)
    for sc in (FAIR_LEN / 2 - 6.0, FAIR_LEN / 2, FAIR_LEN / 2 + 6.0):                       # 비늘창 3 줄(긴 축 y, 6 피치)
        slot = _cs([(a, c) for (c, a) in louvre((0.0, 0.0), 8.0, 1.5)]).extrude(8.0).translate([0, 0, -6.0])
        b -= slot.rotate([0, -FAIR_ANG, 0]).translate([FAIR_X0 + sc * math.cos(th), 0, LEG_Z1 + sc * math.sin(th)])
    # 열화상 보드 자리: 바닥 앞 턱·뒤 턱(핀 줄을 피해 |y| ≥ 7)
    b += box(TH_X + TH_T + 0.25, BEZEL_X, -W, W, ZT - 0.2, ZT + 1.2)
    for s in (-1, 1):
        y0, y1 = sorted((s * 7.0, s * W))
        b += box(TH_X - 1.5, TH_X - 0.25, y0, y1, ZT - 0.2, ZT + 1.0)
    # 속 비우기: 다리 상자, 가로대(볼트 기둥 밑 |y| ≤ 6 은 속 참). 끝은 45° 지붕
    b -= box(45.5, 70.0, -13.0, 13.0, LEG_Z0 + 1.8, LEG_Z1 - 1.6)
    cx0, cx1, cz0, cz1 = BAR_X0 + 1.6, BAR_X1 - 1.6, TOP_Z0 + 1.6, BAR_Z1 - 1.6
    zm = (cz0 + cz1) / 2; h = (cz1 - cz0) / 2
    for ya, yb in ((-W + 1.6, -6.0), (6.0, TG_Y0 - 0.3)):
        b -= m3.Manifold.hull(box(cx0, cx1, ya, yb - h, cz0, cz1) + box(cx0, cx1, yb - 0.02, yb, zm - 0.01, zm + 0.01))
    for yv in (-10.0, 25.0):                                                 # 빈 속이 닫힌 방이 되지 않게 Ø2 숨구멍(앞면)
        b -= m3.Manifold.cylinder(4.0, 1.0, 1.0, 24).rotate([0, 90, 0]).translate([BAR_X1 - 3.0, yv, zm])
    # 지붕 M3: 관통 구멍, 머리 자리, 드라이버 구멍
    for (x, y) in HOLES:
        b -= cyl_z(1.7, ROOF_Z - 1, ROOF_Z + T + 1, x, y)
        b -= cyl_z(3.1, ROOF_Z + T - 1.6, ROOF_Z + T + 0.01, x, y)
        b -= cyl_z(3.6, LEG_Z0 - 1, LEG_Z1 + 1, x, y)
    # 선 묶는 곳: 기둥 뒤 비늘창 홈 3 줄(이웃한 두 줄 사이 살에 타이), 경사 판 양 옆 타이 홈(폭 3, 깊이 1.2)
    for zc in (11.0, 17.0, 23.0):
        b -= ext_x(louvre((0.0, zc)), COL_X0 - 1, COL_X1 + 1)
    # (경사 판 옆 타이 홈은 등 경사 덮개가 덮어서 없앰 — 선은 덮개 비늘창 두 줄 사이 살에 타이로 묶음)
    # 뚜껑 볼트 암나사 M5 (위에서 깊이 10)
    for (x, y) in BOLT_POS:
        b -= F.female_cut(M5[0], THREAD_D, M5[1], M5[2]).translate([x, y, BOSS_TOP - THREAD_D])
    return b


# ---------------- 레이더 받침 ----------------
A_BACK0, A_BACK1 = -7.0, -5.4          # 뒤판(국소 a)
A_LIP = 0.6                            # 앞 턱이 보드 면보다 나온 높이
B_OUT, B_SLOT, B_LIP = 9.4, 7.8, 6.3   # 바깥 반폭, 홈 반폭, 앞 턱 안쪽(보드 가장자리 1.2 덮음)
C_TIP = -6.0                           # 칼날 끝(앞 턱 쪽)


def _rear_x(z):
    """뒤판 뒷면(a = −7)의 x 를 높이 z 에서."""
    c = (z - RD_PB[2] - A_BACK0 * RD_N[2]) / RD_U[2]
    return RD_PB[0] + A_BACK0 * RD_N[0] + c * RD_U[0]


def _trim(h):
    h = h.trim_by_plane([0, 0, -1], -Z_TOP)                                             # 윗면 z ≤ 68.3
    nrm = (RD_N + RD_U) / math.sqrt(2)                                                  # 국소 a + c ≥ C_TIP + A_LIP: 수평에서 30° 로 자른 아래 끝
    return h.trim_by_plane([float(v) for v in nrm], float(nrm @ RD_PB) + (C_TIP + A_LIP) / math.sqrt(2))


def _lip_cut(s, t60):
    """국소 (a, b) 단면 삼각형을 c 0.8 ~ 43.2 로 민 것: a ≥ (|b| − 6.1) / tan60 인 곳."""
    tri = [(0.0, s * 6.1), (1.2, s * 6.1), (1.2, s * (6.1 + 1.2 * t60))]
    return rd_local(ext_z(tri, 0.8, RD_L - 0.8))


def radar_holder():
    tg = _cs(TONGUE)
    outer = tg.offset(FIT + 1.6, m3.JoinType.Miter, 4.0) + m3.CrossSection.square([BAR_X1 - BAR_X0 + 2 * (FIT + 1.6), Z_TOP - 61.0]).translate([BAR_X0 - FIT - 1.6, 61.0])
    h = rd_local(lbox(A_BACK0, A_LIP, -B_OUT, B_OUT, C_TIP - 1, RD_L + 3))               # 칼날
    h += ext_y(outer, HD_Y0, HD_Y1)                                                     # 머리(굴 둘레)
    sx = BAR_X1 + FIT + 1.0
    for y0, y1 in ((HD_Y0, HD_Y0 + B_OUT - B_SLOT), (HD_Y1 - (B_OUT - B_SLOT), HD_Y1)):  # 머리와 칼날을 잇는 옆판
        h += ext_y([(sx, Z_TOP), (sx, 49.5), (_rear_x(49.5) + 0.5, 49.5), (_rear_x(Z_TOP) + 0.5, Z_TOP)], y0, y1)
    h = fillet(_trim(h), 0.5)
    h -= rd_local(lbox(-1.4, 0.15, -B_SLOT, B_SLOT, C_TIP - 2, RD_L + 5))               # 보드 홈(위에서 꽂음)
    h -= rd_local(lbox(A_BACK1, -1.4, -B_LIP, B_LIP, C_TIP - 2, RD_L + 5))              # 보드 뒤 부품 자리
    h -= rd_local(lbox(0.15, A_LIP + 1, -B_LIP, B_LIP, C_TIP - 2, RD_L + 5))            # 패치 면 전부 엶
    # 앞 턱 안쪽을 60° 로 비스듬히 깎음: 가운데 12 mm 띠 끝(|b| 6)에서 ±60° 로 나가는 전파를 가리지 않게. 네 귀(c < 0.8, c > 43.2)는 그대로 두어 보드를 잡음
    t60 = math.tan(math.radians(60))
    for s in (-1, 1):
        h -= _lip_cut(s, t60)
    # 뒤판 선 구멍(국소 c 4 ~ 21, 아래 끝은 뾰족), 모서리 R1.5
    hole = _cs([(-5.0, 21.0), (-5.0, 9.0), (0.0, 4.0), (5.0, 9.0), (5.0, 21.0)]).offset(-1.5, m3.JoinType.Round).offset(1.5, m3.JoinType.Round, 2.0, 24)
    h -= rd_local(ext_x(hole, A_BACK0 - 1, A_BACK1 + 0.5))
    h -= ext_y(tg.offset(FIT, m3.JoinType.Miter, 4.0), HD_Y0 - 1, HD_Y1 + 1)             # 굴
    h -= cyl_z(6.55, BOSS_TOP - 0.5, Z_TOP + 1, RB_X, RD_Y)                              # 뚜껑 볼트 기둥이 꽂히는 지붕 구멍
    # 보드 아래 모서리 받침(50° 빗면 2 개, 가운데는 플러그·선 길)
    for s in (-1, 1):
        g = [(s * (B_SLOT + 0.1), 1.2 * 0.3 + 0.12), (s * (B_SLOT + 0.1), -3.5), (s * (RD_W / 2 - 3.5 / 1.2), -3.5)]
        h += rd_local(ext_x(g, -3.0, A_LIP))
    return _trim(h)


# ---------------- 뚜껑·볼트 ----------------
def cap_bolt(phase=0.0):
    return F.pan_bolt(M5[0], BOLT_L, M5[1], M5[2], head_d=HEAD_D, head_h=HEAD_H, slot_d=1.4).rotate([0, 0, phase])


def _bolt_at(x, y, phase):
    return cap_bolt(phase).rotate([180, 0, 0]).translate([x, y, HEAD_Z])


def best_phase(host, x, y, step=5):
    return min(((_bolt_at(x, y, a) ^ host).volume(), a) for a in range(0, 360, step))[1]


def caps():
    """챙 뚜껑 1 개 + 볼트 2 개(조립 자세)."""
    lo = ext_z(_cs(CAP_OUT), Z_TOP, Z_TOP + 0.8)
    c = lo + m3.Manifold.hull(ext_z(_cs(CAP_OUT), Z_TOP + 0.7, Z_TOP + 0.8) + ext_z(_cs(CAP_OUT).offset(-0.7, m3.JoinType.Miter, 4.0), CAP_TOP - 0.01, CAP_TOP))   # 둘레 30° 모따기
    c += box(BEZEL_X - 1.75, BEZEL_X, -W, W + 0.5, ZT + 1.5, Z_TOP + 0.2)              # 앞 치마(열화상)
    c += box(BEZEL_X - 1.75, BEZEL_X, W + 0.3, TG_Y0, TOP_Z0, Z_TOP + 0.2)              # 앞 치마(가운데, 선 길을 가림)
    for s in (-1, 1):                                                                    # 옆 볼
        y0, y1 = sorted((s * (TH_W / 2 + 0.3), s * W))
        c += box(TH_X - 3.4, BEZEL_X - 1.0, y0, y1, ZT + 1.5, Z_TOP + 0.2)
    c = fillet(c, 0.8)
    for s in (-1, 1):                                                                    # 뒤 턱: 보드 짧은 변 끝 1.2 를 뒤에서 받침
        y0, y1 = sorted((s * (TH_W / 2 - 1.2), s * (TH_W / 2 + 0.5)))
        c += box(TH_X - 2.25, TH_X - 0.25, y0, y1, ZT + 2.2, Z_TOP + 0.2)
    # 열화상 창: |y| ≤ 11.3, 아래로 열림, 겉에 1 mm × 45° 테
    wy, wz = TH_W / 2 - 1.2, TH_LENS_Z + 5.0
    win = _cs([(-wy, ZT - 4), (wy, ZT - 4), (wy, wz), (-wy, wz)]).offset(-1.5, m3.JoinType.Round).offset(1.5, m3.JoinType.Round, 2.0, 24)   # 안쪽 모서리 R1.5
    c -= ext_x(win, BEZEL_X - 1.8, BEZEL_X + 0.1)
    c -= m3.Manifold.hull(ext_x(win, BEZEL_X - 1.0, BEZEL_X - 0.99) + ext_x(win.offset(1.0, m3.JoinType.Round, 2.0, 24), BEZEL_X, BEZEL_X + 0.01))
    # 볼트 기둥(자리파기 밑 1.7), 구멍, 자리파기
    for (x, y), gap in zip(BOLT_POS, (0.0, 0.4)):
        c += cyl_z(6.25, BOSS_TOP + gap, Z_TOP + 0.2, x, y)
    for (x, y) in BOLT_POS:
        c -= cyl_z(M5[0] / 2 + 0.3, BOSS_TOP - 1, CAP_TOP + 1, x, y)
        c -= F.counterbore(HEAD_D, HEAD_H, 0.5, 0.2).rotate([180, 0, 0]).translate([x, y, HEAD_Z])
    # 바람 비늘창 3 줄(가운데 선 길 위)
    for yc in (23.5, 29.5, 35.5):
        c -= ext_z(louvre((92.5, yc)), Z_TOP - 1, CAP_TOP + 1)
    a, h = build(), radar_holder()
    bolts = _bolt_at(*BOLT_POS[0], best_phase(a, *BOLT_POS[0])) + _bolt_at(*BOLT_POS[1], best_phase(a, *BOLT_POS[1]))
    return c, bolts


# ---------------- 보드 모형 ----------------
TH_LENS_Y = (-6.5, 0.0, 6.5)           # 렌즈 중심 후보(매뉴얼에 치수 없음: 가운데 또는 어느 한쪽 끝)


def dummies():
    z0 = ZT + TH_Z0
    th = box(TH_X, TH_X + TH_T, -TH_W / 2, TH_W / 2, z0, z0 + TH_H)
    th += box(TH_X - 4.0, TH_X, -TH_W / 2 + 2, TH_W / 2 - 2, z0 + 1.0, z0 + TH_H - 1.0)                 # 뒷면 부품
    can = m3.Manifold.cylinder(11.25, 4.65, 4.65, SEG).rotate([0, 90, 0])   # MLX90640 BAB(55°) 통 Ø9.3 × 11.25 (Melexis 선택 안내서, 주문한 옵션)
    th += m3.Manifold.hull(can.translate([TH_X + TH_T, TH_LENS_Y[0], TH_LENS_Z]) + can.translate([TH_X + TH_T, TH_LENS_Y[-1], TH_LENS_Z]))   # 렌즈 통(위치 후보 전부)
    rd = lbox(-RD_T, 0, -RD_W / 2, RD_W / 2, 0, RD_L)
    rd += lbox(-RD_T - 3.5, -RD_T, -6.0, 6.0, 2.0, RD_L - 2.0)                                           # 뒷면 부품·소켓
    rd += lbox(-RD_T - 3.8, -RD_T, -4.0, 4.0, -6.0, 2.0)                                                 # 1.5 mm 간격 플러그 + 선(아래로)
    return th, rd_local(rd)


def print_pose(name, m):
    t = to_t(m)
    if name == "arm":          # 옆으로 눕힘(−y 면이 판)
        t.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    else:                      # 받침·뚜껑: 윗면이 판
        t.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
    t.apply_translation(-t.bounds[0])
    return t


if __name__ == "__main__":
    a = build(); h = radar_holder(); cap, bolts = caps(); th, rd = dummies()
    for nm, m in (("asm_arm", a), ("asm_arm_holder", h), ("asm_arm_caps", cap), ("asm_arm_bolts", bolts), ("dum_thermal", th), ("dum_radar", rd)):
        to_t(m).export(os.path.join(OUT, nm + ".stl"))
    vol = lambda x: round(x.volume(), 2)
    for nm, m in (("arm", a), ("holder", h), ("cap", cap)):
        t = to_t(m)
        print(f"{nm:7s} bodies {len(m.decompose())}  vol {t.volume / 1000:.2f} cm3  {t.volume * 1.24 / 1000:.1f} g  bounds {t.bounds.round(1).tolist()}")
    print("total g", round((a.volume() + h.volume() + cap.volume()) * 1.24 / 1000, 1))
    parts = {"arm": a, "holder": h, "cap": cap, "bolts": bolts, "thermal": th, "radar": rd}
    ks = list(parts)
    print("overlaps:", {f"{p}^{q}": vol(parts[p] ^ parts[q]) for i, p in enumerate(ks) for q in ks[i + 1:] if vol(parts[p] ^ parts[q]) > 0.005})
    for nm, m, fn in (("arm", a, "print_arm"), ("holder", h, "print_arm_holder"), ("cap", cap, "print_cap")):
        t = print_pose(nm, m)
        for d in (OUT, PRN): t.export(os.path.join(d, fn + ".stl"))
    pb = to_t(cap_bolt()); pb.apply_translation([0, 0, -pb.bounds[0][2]])
    for d in (OUT, PRN): pb.export(os.path.join(d, "print_arm_bolt.stl"))
