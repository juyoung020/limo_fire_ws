"""LIMO 앞 초음파 받침 v4 (HC-SR04 ×2, 좌우 ±10°, 위로 5°).

좌표 = 지붕 좌표(탁구공발사대 build_cradle.py 와 같음): 원점 = 지붕 65×52 구멍 사각형 중심의 지붕 윗면,
x 앞, y 왼, z 위, mm. 바닥(땅) z = -140.5, 차체 바닥판 아랫면 z = -118.0 (팀 메시).
밑면 앞쪽 확장 구멍(메시): (140, ±27.5), (148, ±27.5), Ø2.5. 공식 도면은 앞뒤 간격 10 → 뒤 구멍은 x 138~140 홈.
HC-SR04 (Handson 4.2 + Printables 1136823 실측): 보드 45×20×1.6, 원통 Ø16 간격 26, 원통 높이 12(보드 앞면부터).

LIMO 앞 범퍼(render/asm/limo.stl 실측):
  아래 범퍼 앞면 x 173.0(z −76) → 162.5(z −116), 14.8° 기욺. |y| 34.5 밖은 평면에서 방위 62° 로 뒤로 꺾임.
  위 범퍼(z −70 위)는 |y| < 35.5 뿐이고 x 171.6 → 165.5(z −42). 그 밖(|y| > 36, z > −70)은 x 138 까지 빈 곳.
  바닥판 아랫면 z −118 은 x 160 까지.

모양: 센서 띠(두 면, 앞판 3 mm, 위에 처마) + 기둥 3(가운데·양 끝, 범퍼 아랫면과 나란히 15°) + 밑판(막힌 판 3 mm).
강성 손계산(앞에서 30 N 을 띠 높이 z −62 에, PLA E 3000 MPa):
  기둥 발(z −118)까지 팔 56 mm → 1680 N·mm. 기둥 발 단면 10 × 7 둘 + 가운데 10 × 3.5(선 홈) → I 607 mm⁴,
  굽힘 응력 9.7 MPa, 기둥 처짐 ≤ 1.0 mm. 밑판(차체 끝 x 160 ~ 기둥 발 x 165, 폭 101 − 타이 홈 24, 두께 3):
  단면계수 116 mm³ → 14.5 MPa(층과 나란한 방향), 기울어짐으로 띠가 0.9 mm 더 밀림 → 합 ≈ 1.9 mm (범퍼까지 2.2 ~ 3.1).
  M3 나사 당김: 뒤 쌍 30 N, 앞 쌍 17 N.
보드는 앞판 뒷면에 대고 원통이 구멍에 꽂힘. 누름쇠 3 개(가운데 1 이 두 보드 안쪽 끝, 바깥 2 가 바깥 끝)를
기둥 머리(띠 아래)에 뒤에서 M5 둥근머리 일자 볼트로 조임(머리는 누름쇠에 파묻힘).
"""
import os, math
import numpy as np
import manifold3d as m3
import sys
import trimesh

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(D, "..", "lib"))
import fasteners as F
from shape import fillet as _fillet
fillet = lambda A, r: _fillet(A, r, 32)   # 구 분할 32: 필렛 면이 매끈하게
OUT = os.path.join(D, "stl"); os.makedirs(OUT, exist_ok=True)
SEG = 64
T5, S5 = math.tan(math.radians(5)), math.sin(math.radians(5))

# ---------------- LIMO 붙는 곳 ----------------
Z_BOT = -118.0          # 차체 바닥판 아랫면
T_BASE = 3.0            # 밑판 두께 → 밑면 z −121.0 (차체 가장 낮은 곳 −121.1 위)
HOLES_F = [(148.0, s * 27.5) for s in (-1, 1)]
HOLES_R = [(139.0, s * 27.5) for s in (-1, 1)]   # 138~140 홈 가운데
M3_CLR = 3.4

# ---------------- 센서 ----------------
BOARD = (45.0, 20.0, 1.6)   # 가로(y) × 세로(z) × 두께
CAN_D, CAN_GAP, CAN_H = 16.0, 26.0, 12.0
YAW, PITCH = 10.0, 5.0      # 바깥으로 10°, 위로 5°
SENSOR_Z = -62.0            # 원통 중심 높이(땅 위 78.5)
SENSOR_Y = 25.0             # 센서 중심 |y|
BEZEL_T = 3.0               # 앞판 두께
PLATE_X = 184.0             # 앞판 앞면(센서 가운데) x. 가운데 누름쇠 뒷면이 범퍼 꼭대기(173.0)에서 3 mm
XPCB = -BEZEL_T - BOARD[2]  # 보드 뒷면(국소 x)
PKY, PKZ = BOARD[0] / 2 + 0.3, BOARD[1] / 2 + 0.3      # 보드 둘레 틈 0.3

# ---------------- 띠 ----------------
ZB, ZT = -13.0, 14.3        # 앞판 아래·위(국소 z) → 지붕 좌표 z −75.0 … −47.7
BROW_D = 9.0                # 처마 깊이(앞판 뒷면에서 뒤로): 바로 위에서 볼 때 누름쇠 뒷면까지 가림
BROW_UA = 50.0              # 처마 밑면 기울기(국소, 수평에서). 위로 5° 숙이면 실제 45°
BROW_TA = 35.0              # 처마 윗면 기울기(국소) → 실제 30°
HAUNCH = 10.0               # 띠 아랫변 45° 받침 삼각(기둥 머리 옆): 출력 때 띠 아랫변 다리 길이를 줄임
END_AZ = 62.0               # 띠 끝을 자르는 면의 법선 방위(범퍼 모서리와 나란히)
END_Y = 24.6                # 앞면에서 띠 끝(국소 y). 보드 끝 22.5 + 틈 0.3 + 살 1.8
Y_MAX = 52.6                # 범퍼 폭

# ---------------- 나사(출력 M5 둥근 나사산) ----------------
M5 = (5.0, 1.25, 0.4)
HEAD_D, HEAD_H = 8.0, 3.0   # 둥근머리 일자 볼트 머리
BOSS_D = 8.5                # 바깥 기둥 머리 깊이(국소 x −8.5 … 0)
THR_CUT = 6.0               # 암나사 깎는 길이(female_cut 은 양쪽 1 mm 더 깎음 → 실제 깊이 7.0, 앞면 1.5 남음)
PAD_T = 5.0                 # 누름쇠 판 두께 = 머리 자리 3.2 + 머리 밑 1.8
FLOOR = PAD_T - (HEAD_H + 0.2)
TAB_L = FLOOR + 6.5         # 볼트 나사 길이 8.3 (누름쇠 1.8 + 암나사 6.5)
TAB_Y = 21.0                # 바깥 볼트 축(국소 |y|). 범퍼 모서리(|y| 34.5 밖에서 뒤로 꺾임)에서 멀리, 폭 52.6 안에서 가장 바깥
TAB_Z = -18.3               # 바깥 볼트 축(국소 z): 띠 아래
TAB_X0 = -BOSS_D - FLOOR    # 바깥 볼트 머리 밑면(국소 x)
PAD_W = 11.0                # 누름쇠 볼트 자리 폭(가운데). 바깥 누름쇠는 안쪽 5.5 · 바깥쪽 4.3(머리 자리가 옆으로 열림)
BUMP_REF = (167.2, 39.0)    # 범퍼 모서리 면 위 한 점(z −76, 메시 실측; 면은 방위 62° 로 뒤로 꺾임)
BUMP_CL = 2.3               # 바깥 누름쇠 뒤 안쪽 모서리를 이 면에서 이만큼 띄워 깎음(실제 거리는 check_sonar 로 잼)
PAD_Z0, PAD_Z1 = -23.6, -13.0
LIP_Z = -23.8               # 기둥 머리 뒤 턱 윗면(누름쇠 밑변 0.2 아래): 누름쇠가 볼트 축으로 돌지 않게
BOSS_Z0, BOSS_Z1 = -25.3, -11.0
BOSS_HW = 4.5               # 기둥 머리 반 폭
FIN = (17.5, 22.3)          # 보드 뒷면에 부품이 없는 양 끝 띠(국소 |y|)
# 가운데(지붕 좌표)
XB = PLATE_X - 2.9          # 가운데 기둥 머리 뒷면 x (181.1)
CP_Z1 = -81.0               # 가운데 볼트 축 z
CB_X = XB - FLOOR           # 가운데 볼트 머리 밑면 x
CB_L = TAB_L
CT_T = PAD_T
CPAD_Z0, CPAD_Z1 = CP_Z1 - 5.2, CP_Z1 + 5.2
CBOSS_Z0, CBOSS_Z1 = CPAD_Z0 - 1.6, -73.5
CBOSS_HW = 4.7

# ---------------- 기둥·밑판 ----------------
RAKE = 0.265                # 범퍼 아랫면 기울기 dx/dz (14.8°)
STRUT_GAP = 3.2             # 기둥 뒷면 ↔ 범퍼 앞면(가운데 단면) 수평 거리 → 면에 수직으로 3.1
STRUT_W = 10.0
STRUT_T0, STRUT_T1 = 7.0, 4.0   # 기둥 두께: 밑(z −118) 7, 위(z −76) 4
STRUT_Y = (39.5, 49.5)      # 바깥 기둥 |y|
SKID_A = 30.0               # 밑판 앞 끝 썰매 모따기(수평에서)
TIE_Y = (13.0, -31.0)       # 케이블 타이 홈 쌍 2 곳(y 가운데)
TIE_X = (164.0, 170.0)      # 홈 쌍의 x (3 mm 폭, 6 mm 간격)


def box(x0, x1, y0, y1, z0, z1):
    x0, x1 = sorted((x0, x1)); y0, y1 = sorted((y0, y1)); z0, z1 = sorted((z0, z1))
    return m3.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])


def cyl_x(r, x0, x1, y, z):
    return m3.Manifold.cylinder(x1 - x0, r, r, SEG).rotate([0, 90, 0]).translate([x0, y, z])


def cyl_z(r, z0, z1, x, y):
    return m3.Manifold.cylinder(z1 - z0, r, r, SEG).translate([x, y, z0])


def _cs(pts):
    a = sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1] for i in range(len(pts)))
    return m3.CrossSection([pts if a > 0 else pts[::-1]])


def prism_y(pts_xz, y0, y1):
    """(x, z) 다각형을 y0 → y1 로 밀어낸 기둥."""
    y0, y1 = sorted((y0, y1))
    return _cs(pts_xz).extrude(y1 - y0).rotate([90, 0, 0]).translate([0, y1, 0])


def prism_x(pts_yz, x0, x1):
    """(y, z) 다각형을 x0 → x1 로 밀어낸 기둥."""
    x0, x1 = sorted((x0, x1))
    return _cs(pts_yz).extrude(x1 - x0).rotate([90, 0, 90]).translate([x0, 0, 0])


def prism_z(pts_xy, z0, z1):
    return _cs(pts_xy).extrude(z1 - z0).translate([0, 0, z0])


def pose(mf, side):
    """센서 국소 좌표(앞판 앞면 가운데 = 원점, +x 앞) → 지붕 좌표. side +1 = 왼쪽(+y)."""
    return (mf.rotate([0, -PITCH, 0]).rotate([0, 0, side * YAW])
            .translate([PLATE_X, side * SENSOR_Y, SENSOR_Z]))


def unpose(mf, side):
    """지붕 좌표 → 센서 국소 좌표(pose 의 거꾸로)."""
    return (mf.translate([-PLATE_X, -side * SENSOR_Y, -SENSOR_Z]).rotate([0, 0, -side * YAW]).rotate([0, PITCH, 0]))


def bumper_keepout(side):
    """범퍼 모서리 면에서 BUMP_CL 안쪽(지붕 좌표): 바깥 누름쇠가 들어가면 안 되는 곳."""
    a = math.radians(END_AZ)
    return (box(-300, 0, -300, 300, -300, 300).rotate([0, 0, side * END_AZ])
            .translate([BUMP_REF[0] + BUMP_CL * math.cos(a), side * (BUMP_REF[1] + BUMP_CL * math.sin(a)), 0]))


def half_side(side):
    """y·side ≥ 0 반쪽."""
    return box(0, 400, 0, side * 400, -400, 400)


def front_env():
    """두 앞판 앞면 뒤쪽(두 면이 만드는 볼록한 앞 껍질)."""
    h = box(-400, 0, -400, 400, -400, 400)
    return pose(h, 1) ^ pose(h, -1)


def end_cut(side):
    """띠 끝 바깥쪽(잘라낼 부피): 법선 방위 62° 인 세로 면 밖 + |y| > 52.6."""
    p = pose(box(-0.01, 0, side * END_Y - 0.01, side * END_Y, -0.01, 0), side).bounding_box()
    c = box(0, 300, -300, 300, -300, 300).rotate([0, 0, side * END_AZ]).translate([p[0], p[1], 0])
    return c + box(0, 400, side * Y_MAX, side * 400, -400, 400)


def aperture_cs():
    """원통 구멍 단면(y, z): Ø16.4, 위는 45° 지붕을 원 위 0.8 에서 잘라 5.2 mm 폭 평평한 다리(거의 둥글게 보임)."""
    r = (CAN_D + 0.4) / 2
    cs = (m3.CrossSection.circle(r, SEG) + _cs([(-r * 0.7071, r * 0.7071), (r * 0.7071, r * 0.7071), (0, r * 1.4142)])).hull()
    return cs ^ m3.CrossSection.square([40, 40]).translate([-20, -40 + r + 0.8])


def aperture_cut():
    """구멍 + 앞면 1 mm × 45° 파인 테두리(국소)."""
    cs = aperture_cs()
    ex = lambda c, x0, x1: c.extrude(x1 - x0).rotate([90, 0, 90]).translate([x0, 0, 0])
    hole = ex(cs, -BEZEL_T - 1, 1)
    bez = m3.Manifold.hull(ex(cs.offset(1.0, m3.JoinType.Round), 0, 1) + ex(cs, -1.0, -0.99))
    out = m3.Manifold()
    for s in (-1, 1):
        out += (hole + bez).translate([0, s * CAN_GAP / 2, 0])
    return out


def xb(z):
    """기둥 뒷면 x (범퍼 아랫면과 나란히, 3.2 앞)."""
    return 173.0 + STRUT_GAP + RAKE * (z + 76.0)


def st(z):
    return STRUT_T1 + (STRUT_T0 - STRUT_T1) * (-76.0 - z) / 42.0


def strut(y0, y1, zt):
    """밑판(z −118)에서 zt 까지 곧은 기둥 + zt 의 얇은 윗단면."""
    top = box(xb(zt), xb(zt) + st(zt), y0, y1, zt - 0.02, zt)
    foot = box(xb(Z_BOT), xb(Z_BOT) + STRUT_T0, y0, y1, Z_BOT - 0.5, Z_BOT + 0.02)
    return m3.Manifold.hull(foot + top), top


def band_local(e):
    """한쪽 띠(국소, 바깥 끝이 e 쪽): 앞판 + 처마 + 받침 삼각 + 바깥 기둥 머리."""
    tu, tt = math.tan(math.radians(BROW_UA)), math.tan(math.radians(BROW_TA))
    xr = -BEZEL_T - BROW_D
    sec = [(0, ZB), (0, ZT), (xr, ZT - tt * xr), (xr, PKZ + tu * BROW_D), (-BEZEL_T, PKZ), (-BEZEL_T, ZB)]
    b = prism_y(sec, -e * 30.0, e * 34.0)
    yi = -(SENSOR_Y - CBOSS_HW) / math.cos(math.radians(YAW))      # 가운데 기둥 머리 옆(국소 y, 안쪽)
    tri = [[(e * (yi + HAUNCH), ZB + 0.01), (e * yi, ZB - HAUNCH), (e * -30.0, ZB - HAUNCH), (e * -30.0, ZB + 0.01)],
           [(e * (TAB_Y - BOSS_HW - HAUNCH), ZB + 0.01), (e * (TAB_Y - BOSS_HW), ZB - HAUNCH), (e * TAB_Y, ZB - HAUNCH), (e * TAB_Y, ZB + 0.01)],
           [(e * TAB_Y, ZB + 0.01), (e * TAB_Y, ZB - 8.0), (e * (TAB_Y + BOSS_HW), ZB - 8.0), (e * (TAB_Y + BOSS_HW + 8.0), ZB + 0.01)]]   # 기둥 머리 안으로 겹쳐 둠(면이 딱 맞닿으면 필렛에서 실금)
    for t in tri:
        b += prism_x(t, -BEZEL_T, 0)
    b += box(-BOSS_D, 0, e * (TAB_Y - BOSS_HW), e * (TAB_Y + BOSS_HW), BOSS_Z0, BOSS_Z1)
    return b


def boss_foot_local(e):
    """바깥 기둥 머리 맨 아래 층(뒤 턱 포함) — 기둥과 볼록하게 이어 붙이는 데 씀."""
    return box(-BOSS_D - 3.0, 0, e * (TAB_Y - BOSS_HW), e * (TAB_Y + BOSS_HW), BOSS_Z0, LIP_Z)


def louvre(xc, yc, length=12.0, w=3.0):
    """바람·타이 홈: 폭 3, 길이 12(y 방향), 양 끝 45° 뾰족. 밑판을 세로로 뚫는 부피."""
    h = length / 2
    pts = [(xc - w / 2, yc - h + w / 2), (xc, yc - h), (xc + w / 2, yc - h + w / 2),
           (xc + w / 2, yc + h - w / 2), (xc, yc + h), (xc - w / 2, yc + h - w / 2)]
    return prism_z(pts, Z_BOT - T_BASE - 1, Z_BOT + 1)


def base_plate():
    xf = xb(Z_BOT) + STRUT_T0                       # 기둥 발 앞면
    nose = T_BASE / math.tan(math.radians(SKID_A))
    yb, yf = 34.0, STRUT_Y[1] + 1.0
    xn = xf + nose * (T_BASE - 1.6) / T_BASE        # 앞 끝은 두께 1.6 에서 끊음(칼날 끝은 필렛이 지움)
    pts = [(134.0, -yb), (156.0, -yb), (164.0, -yf), (xn, -yf), (xn, yf), (164.0, yf), (156.0, yb), (134.0, yb)]
    cs = _cs(pts).offset(-3.0, m3.JoinType.Round).offset(3.0, m3.JoinType.Round)      # 평면 모서리 R3
    b = cs.extrude(T_BASE).translate([0, 0, Z_BOT - T_BASE])
    b -= prism_y([(xf, Z_BOT - T_BASE - 0.01), (xf + nose + 1, Z_BOT - T_BASE - 0.01), (xf + nose + 1, Z_BOT + T_BASE / nose)], -80, 80)
    return b


def _build():
    fe = front_env()
    body = base_plate()
    for side in (-1, 1):
        e = side
        body += (pose(band_local(e), side) ^ half_side(side)) - end_cut(side)
        s, top = strut(side * STRUT_Y[0], side * STRUT_Y[1], -96.0)
        body += s + (m3.Manifold.hull(top + pose(boss_foot_local(e), side)) - end_cut(side))
    # 가운데 기둥: 머리(암나사) + 뒤 턱 + 45° 뱃머리 + 기둥
    body += box(XB, 200, -CBOSS_HW, CBOSS_HW, CBOSS_Z0, CBOSS_Z1) ^ fe
    cfoot = box(XB - 3.0, 200, -CBOSS_HW, CBOSS_HW, CBOSS_Z0, CPAD_Z0 - 0.2) ^ fe
    s, top = strut(-STRUT_W / 2, STRUT_W / 2, -107.0)
    body += s + m3.Manifold.hull(top + cfoot)
    # 기둥 발·밑판 평면의 오목한 모서리 R1.5 (닫힘 연산: 불렸다 깎음. z −108 아래만) — 굽힘이 가장 큰 기둥 발의 응력 집중을 줄임
    sp = m3.Manifold.sphere(1.5, 16)
    body += (body ^ box(100, 220, -90, 90, Z_BOT - T_BASE, -108.0)).minkowski_sum(sp).minkowski_difference(sp)
    # 가운데 기둥 발 뒤 선 지나가는 홈(왼쪽 센서 선이 오른쪽으로 건너감): 깊이 3.5 + 범퍼 틈, 천장 45°
    x0 = xb(Z_BOT) + 3.5
    body -= prism_y([(x0 - 12, Z_BOT - 0.01), (x0, Z_BOT - 0.01), (x0, Z_BOT + 5.0), (x0 - 5.0, Z_BOT + 10.0), (x0 - 12, Z_BOT + 10.0)], -7, 7)
    for side in (-1, 1):
        body -= pose(aperture_cut(), side)
        # 보드·듀폰을 뒤에서 곧게 넣는 길(틈 0.3)
        body -= pose(box(-BEZEL_T - 25, -BEZEL_T + 0.01, -PKY, PKY, -PKZ, PKZ), side)
        body -= pose(box(-BEZEL_T - 25, -BEZEL_T + 0.01, -6.5, 6.5, -BOARD[1] / 2 - 21, -BOARD[1] / 2 + 0.5), side)
    for yc in TIE_Y:
        for xc in TIE_X:
            body -= louvre(xc, yc)
    body = fillet(body, 0.8)                            # 바깥 모서리 R0.8 (나사산·정밀 구멍은 이 뒤에)
    for side in (-1, 1):                                # 바깥 기둥 머리 M5 암나사(뒤에서, 막힌 구멍)
        body -= pose(F.female_cut(M5[0], THR_CUT, M5[1], M5[2]).rotate([0, 90, 0]).translate([-BOSS_D, side * TAB_Y, TAB_Z]), side)
    body -= F.female_cut(M5[0], THR_CUT, M5[1], M5[2]).rotate([0, 90, 0]).translate([XB, 0, CP_Z1])
    for (x, y) in HOLES_F:
        body -= cyl_z(M3_CLR / 2, Z_BOT - 20, Z_BOT + 1, x, y)
    for (x, y) in HOLES_R:
        body -= (cyl_z(M3_CLR / 2, Z_BOT - 20, Z_BOT + 1, x - 1.0, y) + cyl_z(M3_CLR / 2, Z_BOT - 20, Z_BOT + 1, x + 1.0, y)
                 + box(x - 1.0, x + 1.0, y - M3_CLR / 2, y + M3_CLR / 2, Z_BOT - 20, Z_BOT + 1))
    return body


_CACHE = {}


def build():
    """받침(지붕 좌표, 조립 자세). 한 번 만든 것은 다시 쓴다(필렛이 40 초쯤 걸림)."""
    k = id(fillet)                      # verify_fillet_loss.py 가 fillet 을 바꿔 다시 부르면 새로 만듦
    if k not in _CACHE:
        _CACHE[k] = _build()
    return _CACHE[k]


# ---------------- 누름쇠·볼트 ----------------
def _cbore():
    """머리 자리(축 +x, 머리 밑면 x 0, 뒤(−x)로 열림) + 볼트 지나는 구멍."""
    return F.counterbore(HEAD_D, HEAD_H).rotate([0, 90, 0]) + cyl_x(M5[0] / 2 + 0.3, -1, FLOOR + 1, 0, 0)


def tab_local(e):
    """바깥 누름쇠(국소, 보드 끝 e 쪽). 볼트 자리 판(기둥 머리 뒤) + 세로 띠 + 손가락(보드 뒷면)."""
    x0 = -BOSS_D - PAD_T
    t = box(x0, -BOSS_D, e * (TAB_Y - 5.5), e * (TAB_Y + 4.3), PAD_Z0, PAD_Z1)
    t += box(x0, -BOSS_D, e * FIN[0], e * FIN[1], PAD_Z1 - 0.01, 6.0)
    t += box(-BOSS_D - 0.01, XPCB, e * FIN[0], e * FIN[1], -6.0, 6.0)          # 손가락 끝 = 보드 뒷면
    t -= unpose(bumper_keepout(e), e)                                          # 뒤 안쪽 모서리: 범퍼 모서리와 나란히 깎음
    t = fillet(t, 0.8)
    return t - _cbore().translate([TAB_X0, e * TAB_Y, TAB_Z])


def pan(L, a=0.0):
    return F.pan_bolt(M5[0], L, M5[1], M5[2], head_d=HEAD_D, head_h=HEAD_H).rotate([0, 0, a])


def slot_bolt(L):
    """둥근머리 일자 홈 M5 볼트(머리 z −3~0, 나사 +z). 받침 볼트 3 개 모두 이것."""
    return pan(L)


def outer_bolt(side, a=0.0):
    """바깥 누름쇠 볼트(지붕 좌표), 나사 위상 a°."""
    return pose(pan(TAB_L, a).rotate([0, 90, 0]).translate([TAB_X0, side * TAB_Y, TAB_Z]), side)


def center_bolt(a=0.0):
    return pan(CB_L, a).rotate([0, 90, 0]).translate([CB_X, 0, CP_Z1])


def tab_bolt_local(e):
    return pan(TAB_L).rotate([0, 90, 0]).translate([TAB_X0, e * TAB_Y, TAB_Z])


def center_tab():
    """두 보드 안쪽 끝을 함께 누르는 누름쇠(지붕 좌표). 볼트 자리 판 + 세로 띠 + 손가락 2.
    끼우기: 보드를 넣은 뒤 뒤에서 앞(+x)으로 곧게 밀어 넣고 뒤에서 볼트."""
    x0 = XB - PAD_T
    t = box(x0, XB, -PAD_W / 2, PAD_W / 2, CPAD_Z0, CPAD_Z1)
    t += box(x0, XB, -7.0, 7.0, CPAD_Z1 - 0.01, SENSOR_Z + 6.5)
    for side in (-1, 1):
        pad = pose(box(XPCB - 2.0, XPCB, -side * FIN[1], -side * FIN[0], -5.5, 5.5), side)
        root = box(XB - 0.5, XB, side * 1.0, side * 7.0, SENSOR_Z - 6.5, SENSOR_Z + 6.5)
        t += m3.Manifold.hull(pad + root)
    # 장애물을 +x 로 14 mm 끌어 늘린 볼록 덩어리를 뺌 → 뒤(−x)로 곧게 빼고 넣을 수 있음
    sweep = lambda m: m3.Manifold.hull(m + m.translate([14.0, 0, 0]))
    for side in (-1, 1):
        obs = [box(XPCB, -BEZEL_T + 5, -BOARD[0] / 2, BOARD[0] / 2, -BOARD[1] / 2, BOARD[1] / 2),          # 보드(딱 닿음)
               box(XPCB - 4.3, -BEZEL_T, -15.3, 15.3, -BOARD[1] / 2 + 0.7, BOARD[1] / 2 - 0.7),            # 뒷면 부품 0.3
               box(XPCB - 2.9, -BEZEL_T, -5.8, 5.8, -BOARD[1] / 2 - 20.3, -BOARD[1] / 2 + 0.3)]            # 핀·듀폰 0.3
        for o in obs:
            t -= sweep(pose(o, side))
    t = fillet(t, 0.6)
    return t - _cbore().translate([CB_X, 0, CP_Z1])


def tabs_world():
    t = center_tab(); b = center_bolt()
    for side in (-1, 1):
        t += pose(tab_local(side), side); b += outer_bolt(side)
    return t, b


def to_t(mf):
    me = mf.to_mesh()
    return trimesh.Trimesh(me.vert_properties[:, :3], me.tri_verts)


def dummy_sensor_local():
    """HC-SR04 실물 부피: 보드(뒤로 1.6) + 뒷면 부품 4 + 원통 + 아래로 핀·듀폰 커넥터(보드 아랫변에서 20)."""
    pcb = box(-BEZEL_T - 1.6, -BEZEL_T, -BOARD[0] / 2, BOARD[0] / 2, -BOARD[1] / 2, BOARD[1] / 2)
    comp = box(-BEZEL_T - 1.6 - 4.0, -BEZEL_T - 1.6, -15, 15, -BOARD[1] / 2 + 1, BOARD[1] / 2 - 1)
    cans = sum((cyl_x(CAN_D / 2, -BEZEL_T, -BEZEL_T + CAN_H, s * CAN_GAP / 2, 0) for s in (-1, 1)), m3.Manifold())
    plug = box(-BEZEL_T - 1.6 - 2.6, -BEZEL_T - 1.6 + 0.0, -5.5, 5.5, -BOARD[1] / 2 - 20, -BOARD[1] / 2)
    return pcb + comp + cans + plug


def wire_bend_local():
    """듀폰 커넥터 아래 선이 꺾이는 자리(커넥터 끝에서 10 mm 더, 폭 11 × 두께 2.6)."""
    return box(-BEZEL_T - 1.6 - 2.6, -BEZEL_T - 1.6, -5.5, 5.5, -BOARD[1] / 2 - 30, -BOARD[1] / 2 - 20)


def print_parts():
    """출력 자세(판 z 0): 받침 = 밑판이 바닥, 누름쇠 = 뒷면(머리 자리 쪽)이 바닥, 볼트 = 머리가 바닥."""
    def down(t):
        t.apply_translation(-t.bounds[0]); return t
    back = trimesh.transformations.rotation_matrix(-np.pi / 2, [0, 1, 0])     # 국소 −x → 아래
    p = {"print_sonar.stl": to_t(build())}
    for e, nm in ((1, "print_sonar_tab.stl"), (-1, "print_sonar_tab_r.stl")):      # 왼쪽 / 오른쪽(거울상)
        tt = to_t(tab_local(e)); tt.apply_transform(back); p[nm] = tt
    tc = to_t(center_tab()); tc.apply_transform(back); p["print_sonar_ctab.stl"] = tc
    p["print_sonar_bolt.stl"] = to_t(pan(TAB_L)); p["print_sonar_cbolt.stl"] = to_t(pan(CB_L))
    return {k: down(v) for k, v in p.items()}


if __name__ == "__main__":
    d = pose(dummy_sensor_local(), 1) + pose(dummy_sensor_local(), -1)
    to_t(d).export(os.path.join(OUT, "dum_hcsr04.stl"))
    b = build()
    t = to_t(b)
    t.export(os.path.join(OUT, "asm_sonar.stl"))
    print("bounds", t.bounds.round(1).tolist(), "bodies", len(b.decompose()), "vol", round(b.volume()), "mass g", round(b.volume() * 1.24e-3, 1))
    tw, bw = tabs_world()
    to_t(tw).export(os.path.join(OUT, "asm_sonar_tabs.stl")); to_t(bw).export(os.path.join(OUT, "asm_sonar_bolts.stl"))
    v = lambda m: round(m.volume(), 2)
    print("body∩dummy", v(b ^ d), "tabs∩body", v(tw ^ b), "tabs∩dummy", v(tw ^ d), "bolts∩tabs", v(bw ^ tw),
          "tabs bodies", len(tw.decompose()), "tabs+bolts g", round((tw.volume() + bw.volume()) * 1.24e-3, 1))
    for k, m in print_parts().items():
        m.export(os.path.join(OUT, k))
    print("전체 검사는 check_sonar.py")
