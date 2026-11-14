"""LIMO 앞 초음파 띠 v5 (HC-SR04 ×2, 좌우 ±10°, 위로 5°) — 나사·기둥·밑판 없이 양면 폼 테이프로 붙임.
이전(나사로 조이는) 설계는 build_sonar_v4.py / check_sonar_v4.py.

좌표 = 지붕 좌표: x 앞, y 왼, z 위, mm. 땅 z −140.5.

LIMO 코(render/asm/limo.stl 실측, 26-11-13):
  위 앞면: x = 156.391 − 0.2174·z (위로 12.26° 누움), z −72.0 … −36.2, 평평한 폭 |y| ≤ 30 (그 밖은 R3.5 로 옆벽 y ±33.2~33.65).
           LIMO 글자는 0.8 mm 파인 글자(튀어나오지 않음), z −60.3 … −47.6, |y| ≤ 24.
  코 상자 윗면 z −36.0 (그 위가 라이다 틈).  옆벽: y ≈ 33.56 − 0.0083(x−150) + 0.0088(z+42).
  꼭대기 띠: (171.75, −72.25 홈) → (173.0, −74.5). 아래 앞면: x = 172.2 + 0.269(z+78) (아래로 15°), 평평한 폭 |y| ≤ 33.
  어깨(아래 범퍼 윗면, 코 상자 옆): z −69.80 평면, y 34.0~37.5, x 149~163.

붙이는 법: 1 mm 아크릴 폼 테이프 두 곳 — 위 패드(위 앞면, 글자 위 z −47.2 … −41.2) + 아래 패드(아래 앞면 z −89 … −78.5).
  두 면이 꼭대기 띠를 사이에 둔 V 자라 뒤(−x)로 누르면 두 패드가 함께 닿는다.
자리 잡기(출력물 모양): x = 두 패드(테이프 1.0), y = 지붕 밑 귀 2 개가 코 상자 옆벽을 0.3 틈으로 낌,
  z = 귀에서 내려온 다리 2 개가 어깨 윗면(z −69.8)에 얹힘(틈 0.3). 위(+z)·앞(+x)은 테이프만 잡는다.
센서: 보드 앞면 ↔ 앞판 안쪽 면 사이 1 mm 테이프(원통 사이). 보드 둘레는 0.3 틈 테두리(위 덩이·양옆 턱·아래 턱받이 빗면)가 자리를 정함.
  보드는 띠를 LIMO 에 붙이기 전에: 바깥 끝 쪽에서 15 mm 뒤로 물린 채 옆으로 밀어 넣고 앞으로 15 mm 눌러 붙임.
출력: 지붕(z −41.2 평면)을 판에 대고 뒤집어서. 서포트 없음(턱받이 47°, 원통 구멍 눈물방울은 아래로).
"""
import os, math
import numpy as np
import manifold3d as m3
import sys
import trimesh

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(D, "..", "lib"))
from shape import fillet as _fillet
fillet = lambda A, r: _fillet(A, r, 32)   # 구 분할 32
OUT = os.path.join(D, "stl"); os.makedirs(OUT, exist_ok=True)
SEG = 64
C5, S5 = math.cos(math.radians(5)), math.sin(math.radians(5))

# ---------------- 센서 ----------------
BOARD = (45.0, 20.0, 1.6)   # 가로(y) × 세로(z) × 두께
CAN_D, CAN_GAP, CAN_H = 16.0, 26.0, 12.0
YAW, PITCH = 10.0, 5.0      # 바깥으로 10°, 위로 5°
SENSOR_Z = -57.5            # 원통 중심 높이(땅 위 83.0). 보드 윗변 뒤 모서리 z −47.9 가 위 패드 밑변(−47.2) 아래로 지나감
SENSOR_Y = 25.0             # 센서 중심 |y|
BEZEL_T = 3.0               # 앞판 앞면 ↔ 보드 앞면 = 앞판 2.0 + 테이프 1.0
WALL = 2.0                  # 앞판 두께
TAPE = 1.0                  # 폼 테이프 두께
PLATE_X = 181.5             # 앞판 앞면(센서 가운데) x
PKY, PKZ = BOARD[0] / 2 + 0.3, BOARD[1] / 2 + 0.3      # 보드 둘레 틈 0.3
XPCB = -BEZEL_T - BOARD[2]  # 보드 뒷면(국소 x)

# ---------------- 띠 ----------------
Z_ROOF = -41.2              # 지붕 윗면(라이다 한계 −41 아래). 출력 때 판에 닿는 면
T_ROOF = 1.8
T_CHIN = 2.4                # 턱받이(밑을 막는 빗면) 두께
A_CHIN = 42.0               # 턱받이 기울기(국소, 수평에서). 위로 5° 숙이면 실제 47° → 출력 때 43° 오버행
TA, CA = math.tan(math.radians(A_CHIN)), math.cos(math.radians(A_CHIN))
RIM_Z0 = -(BOARD[1] / 2 + 0.4)   # 턱받이 윗면이 보드 앞 아랫모서리 밑 0.4 를 지남(면에 수직으로 0.3)
X_LOC = -9.0                # 지붕·턱받이가 뒤로 가는 깊이(국소 x): 뒷면 부품(−8.6)을 위아래에서 가림
BROW = (2.5, 3.0)           # 지붕 앞 모서리 차양 모따기(뒤로 2.5, 아래로 3.0 → 실제 47°)
END_AZ = 62.0               # 띠 끝을 자르는 면의 법선 방위(범퍼 모서리와 나란히)
END_Y = 23.6                # 앞면에서 띠 끝(국소 y)
T_CAP = 1.8                 # 끝 막이 두께
CAP_X = -5.0                # 끝 막이 깊이(국소 x): 보드를 15 물린 채 옆으로 밀 때 원통 앞(−6)이 그 뒤로 지나감
Y_MAX = 52.6                # 범퍼 폭

# ---------------- LIMO 코(실측) ----------------
UP_A, UP_B = 156.391, -0.2174            # 위 앞면 x = UP_A + UP_B·z
LOW_PROF = [(171.75, -72.0), (172.1, -72.4), (172.6, -73.2), (172.95, -74.0), (173.05, -74.5), (173.007, -75.0)]
LOW_X0, LOW_Z0, LOW_S = 172.2, -78.0, 0.269   # 아래 앞면 x = LOW_X0 + LOW_S·(z − LOW_Z0)
NOSE_HW = 33.5              # 코 상자 반 폭(막는 부피용; 귀 안쪽 면은 따로 옆벽 실측에 맞춤)
SHO_Z = -69.80              # 어깨 윗면
SHO_PLAN = [(173.05, 34.5), (171.7, 36.3), (171.7, 36.9), (171.2, 37.25), (169.3, 38.25), (162.6, 41.75), (158.8, 43.75), (147.3, 49.8), (140.0, 54.0)]   # 아래 범퍼 모서리 윤곽(z −80 … −71.5 가운데 가장 앞)
xf_up = lambda z: UP_A + UP_B * z
xf_low = lambda z: LOW_X0 + LOW_S * (z - LOW_Z0)

# ---------------- 패드·귀 ----------------
T_PAD = 1.8
PADA_Z = (-47.2, Z_ROOF)    # 위 패드(글자 위 끝 −47.6 위)
PADA_HW = 29.0
PADC_Z = (-89.0, -74.5)     # 아래 패드 판(테이프는 −78.5 아래: 그 위는 핀 지나는 홈·꼭대기 띠)
PADC_TAPE_Z = (-89.0, -78.5)
PADC_HW = 32.0
EAR_Y0 = 33.9               # 귀(지붕 밑 날개) 안쪽 면 |y|: 옆벽 33.40~33.58 에서 0.3~0.5
EAR_T = 3.5
EAR_X = (149.0, 176.5)
LEG_X = (149.0, 154.3)      # 다리: 15 물린 보드의 뒷면 부품 뒤, 어깨 평면(x 149~163) 위
LEG_GAP = 0.3               # 다리 밑 ↔ 어깨
wall_y = lambda z: 33.58 + 0.0088 * (z + 42.0)      # x 148.5 에서 옆벽 |y|
PLUG_HW = 6.0               # 핀·듀폰 홈 반 폭(커넥터 11 + 틈 0.5)
TIE = (15.5, 4.5)           # 타이 홈: 가운데 국소 y, 턱받이 뒤 끝에서 거리


def box(x0, x1, y0, y1, z0, z1):
    x0, x1 = sorted((x0, x1)); y0, y1 = sorted((y0, y1)); z0, z1 = sorted((z0, z1))
    return m3.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])


def cyl_x(r, x0, x1, y, z):
    return m3.Manifold.cylinder(x1 - x0, r, r, SEG).rotate([0, 90, 0]).translate([x0, y, z])


def _cs(pts):
    a = sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1] for i in range(len(pts)))
    return m3.CrossSection([pts if a > 0 else pts[::-1]])


def cs_y(cs, y0, y1):
    """(x, z) 단면을 y0 → y1 로 밀어냄."""
    y0, y1 = sorted((y0, y1))
    return cs.extrude(y1 - y0).rotate([90, 0, 0]).translate([0, y1, 0])


def prism_y(pts_xz, y0, y1):
    return cs_y(_cs(pts_xz), y0, y1)


def prism_x(pts_yz, x0, x1):
    x0, x1 = sorted((x0, x1))
    return _cs(pts_yz).extrude(x1 - x0).rotate([90, 0, 90]).translate([x0, 0, 0])


def prism_z(pts_xy, z0, z1):
    return _cs(pts_xy).extrude(z1 - z0).translate([0, 0, z0])


def pose(mf, side):
    """센서 국소 좌표(앞판 앞면 가운데 = 원점, +x 앞) → 지붕 좌표. side +1 = 왼쪽(+y)."""
    return (mf.rotate([0, -PITCH, 0]).rotate([0, 0, side * YAW])
            .translate([PLATE_X, side * SENSOR_Y, SENSOR_Z]))


def unpose(mf, side):
    return (mf.translate([-PLATE_X, -side * SENSOR_Y, -SENSOR_Z]).rotate([0, 0, -side * YAW]).rotate([0, PITCH, 0]))


def half_side(side):
    return box(-100, 400, 0, side * 400, -400, 400)


def end_cut(side):
    """띠 끝 바깥쪽(잘라낼 부피): 법선 방위 62° 인 세로 면 밖 + |y| > 52.6."""
    p = pose(box(-0.01, 0, side * END_Y - 0.01, side * END_Y, -0.01, 0), side).bounding_box()
    c = box(0, 300, -300, 300, -300, 300).rotate([0, 0, side * END_AZ]).translate([p[0], p[1], 0])
    return c + box(-100, 400, side * Y_MAX, side * 400, -400, 400)


def end_slab(side):
    """끝 면에서 안쪽으로 T_CAP 두께의 판(끝 막이)."""
    p = pose(box(-0.01, 0, side * END_Y - 0.01, side * END_Y, -0.01, 0), side).bounding_box()
    return box(-T_CAP, 0, -300, 300, -300, 300).rotate([0, 0, side * END_AZ]).translate([p[0], p[1], 0])


# ---------------- LIMO 를 피하는 부피 ----------------
def _low_cs():
    return _cs([(100.0, -72.0)] + LOW_PROF + [(xf_low(-125.0), -125.0), (100.0, -125.0)])


def _up_cs():
    return _cs([(100.0, -36.0), (xf_up(-36.0), -36.0), (xf_up(-72.0), -72.0), (100.0, -72.0)])


def _sho_cs():
    pts = [(100.0, NOSE_HW), (173.05, NOSE_HW)] + SHO_PLAN + [(100.0, 54.0)]     # 코 상자 옆(|y| ≥ 33.5)만
    return _cs(pts) + _cs([(x, -y) for x, y in pts])


def limo_keepout():
    """코 앞면에서 1.0(테이프), 어깨 위 0.3, 범퍼 모서리 둘레 0.5 안쪽."""
    rd = m3.JoinType.Round
    ko = cs_y(_up_cs().offset(TAPE, rd), -NOSE_HW, NOSE_HW)
    ko += cs_y(_low_cs().offset(TAPE, rd), -34.5, 34.5)
    ko += _sho_cs().offset(0.5, rd).extrude(60.0).translate([0, 0, SHO_Z + LEG_GAP - 60.0])
    return ko


def pad_a():
    """위 패드 판(위 앞면과 나란히, 테이프 1.0 띄움)."""
    k = 1.0 / math.cos(math.atan(-UP_B)); a, b = TAPE * k, (TAPE + T_PAD) * k
    z0, z1 = PADA_Z
    return prism_y([(xf_up(z1) + a, z1), (xf_up(z1) + b, z1), (xf_up(z0) + b, z0), (xf_up(z0) + a, z0)], -PADA_HW, PADA_HW)


def pad_c():
    """아래 패드 판(아래 앞면과 나란히 + 꼭대기 띠를 감싸는 윗부분). 테이프 쪽은 limo_keepout 으로 깎임."""
    p = cs_y(_low_cs().offset(TAPE + T_PAD, m3.JoinType.Round), -PADC_HW, PADC_HW)
    return p ^ box(150, 200, -PADC_HW, PADC_HW, PADC_Z[0], PADC_Z[1])


def tape_dummy():
    """폼 테이프 1.0 mm (붙인 뒤 두께 그대로로 봄): 위 패드, 아래 패드, 보드 2 장 × 2 조각."""
    k = 1.0 / math.cos(math.atan(-UP_B)); z0, z1 = PADA_Z
    ta = prism_y([(xf_up(z1), z1), (xf_up(z1) + TAPE * k, z1), (xf_up(z0) + TAPE * k, z0), (xf_up(z0), z0)], -PADA_HW, PADA_HW)
    k2 = 1.0 / math.cos(math.atan(LOW_S)); z0, z1 = PADC_TAPE_Z
    tc = prism_y([(xf_low(z1), z1), (xf_low(z1) + TAPE * k2, z1), (xf_low(z0) + TAPE * k2, z0), (xf_low(z0), z0)], -PADC_HW, PADC_HW)
    tb = m3.Manifold()
    for side in (-1, 1):
        tb += pose(board_tape_local(), side)
    return ta, tc, tb


def board_tape_local():
    """보드 앞면 ↔ 앞판 안쪽 면 테이프: 두 원통 사이, 수정 진동자 창 위·아래."""
    return box(-BEZEL_T, -WALL, -4.8, 4.8, 3.0, 9.6) + box(-BEZEL_T, -WALL, -4.8, 4.8, -9.2, -6.0)


# ---------------- 띠 ----------------
def R(x):
    """지붕 윗면(지붕 좌표 수평면)의 국소 z."""
    return (Z_ROOF - SENSOR_Z - x * S5) / C5


def U(x):
    """턱받이 윗면의 국소 z."""
    return RIM_Z0 + (x + BEZEL_T) * TA


def shell_local(xr):
    """한쪽 띠 껍질(국소 단면을 y 로 밀어냄): 앞판 + 지붕 + 턱받이, 뒤로 xr 까지. 보드 테두리 층(보드 두께만큼) 포함."""
    lw = lambda x: U(x) - T_CHIN / CA
    ri = lambda x: R(x) - T_ROOF / C5
    out = prism_y([(0, lw(0)), (0, R(0) - BROW[1]), (-BROW[0], R(-BROW[0])), (xr, R(xr)), (xr, lw(xr))], -45, 45)
    cav = prism_y([(-WALL, U(-WALL)), (-WALL, ri(-WALL)), (xr - 1, ri(xr - 1)), (xr - 1, U(xr - 1))], -46, 46)
    rim = (out ^ box(XPCB, -WALL + 0.01, -45, 45, -60, 60)) - box(XPCB - 1, -WALL + 0.02, -PKY, PKY, -60, PKZ)
    return (out - cav) + rim, out, cav


def aperture_cs():
    """원통 구멍 단면(y, z): Ø16.4. 뒤집어 출력하므로 45° 지붕은 아래(−z)로, 원 밖 0.8 에서 잘라 5.2 mm 다리."""
    r = (CAN_D + 0.4) / 2
    cs = (m3.CrossSection.circle(r, SEG) + _cs([(-r * 0.7071, -r * 0.7071), (r * 0.7071, -r * 0.7071), (0, -r * 1.4142)])).hull()
    return cs ^ m3.CrossSection.square([40, 40]).translate([-20, -r - 0.8])


def aperture_cut():
    """구멍 + 앞면 1 mm × 45° 파인 테두리 + 두 구멍 사이 수정 진동자 창(국소)."""
    cs = aperture_cs()
    ex = lambda c, x0, x1: c.extrude(x1 - x0).rotate([90, 0, 90]).translate([x0, 0, 0])
    hole = ex(cs, -BEZEL_T - 1, 1)
    bez = m3.Manifold.hull(ex(cs.offset(1.0, m3.JoinType.Round), 0, 1) + ex(cs, -1.0, -0.99))
    out = m3.Manifold()
    for s in (-1, 1):
        out += (hole + bez).translate([0, s * CAN_GAP / 2, 0])
    # 수정 진동자(앞면 가운데 11 x 4.7 x 3.5): 두 구멍을 잇는 12 x 5 창. 깊이 4 > 테이프 1 + 앞판 2 라 앞판을 뚫음
    out += box(-WALL - 0.01, 1, -6, 6, -2.5, 2.5)
    return out


def plug_slot_local():
    """핀·듀폰이 지나는 홈(국소): 커넥터 자리(아래로 열림) + 보드를 뒤에서 넣을 때 핀이 지나는 길."""
    return (box(-8.0, -4.1, -PLUG_HW, PLUG_HW, -60, -BOARD[1] / 2 + 0.3)
            + box(-60, -4.1, -PLUG_HW, PLUG_HW, -19.6, -BOARD[1] / 2 + 0.3))


def tie_slot_local(e):
    """케이블 타이 홈 3 × 12, 양 끝 45° 뾰족(턱받이 면에 수직으로 뚫음). 턱받이 뒤 끝에서 TIE[1]."""
    w, h = 3.0, 6.0
    hexa = _cs([(-w / 2, -h + w / 2), (0, -h), (w / 2, -h + w / 2), (w / 2, h - w / 2), (0, h), (-w / 2, h - w / 2)])
    xc = X_LOC + TIE[1] * CA
    zc = U(xc) - T_CHIN / CA / 2
    return hexa.extrude(8.0).translate([0, 0, -4.0]).rotate([0, 180 - A_CHIN, 0]).translate([xc, e * TIE[0], zc])


def ear(side):
    """지붕 밑 귀(코 상자 옆벽을 0.3 틈으로 낌) + 어깨에 얹히는 다리. 지붕 좌표."""
    z0, z1 = PADA_Z
    yi = lambda z: wall_y(z) + 0.3
    arm = box(EAR_X[0], EAR_X[1], EAR_Y0, EAR_Y0 + EAR_T, z0, z1)
    leg = prism_x([(yi(z0 + 0.01), z0 + 0.01), (EAR_Y0 + EAR_T, z0 + 0.01), (EAR_Y0 + EAR_T, -80.0), (yi(-80.0), -80.0)], LEG_X[0], LEG_X[1])
    e = arm + leg
    return e if side > 0 else e.mirror([0, 1, 0])


def _half():
    """왼쪽(+y) 반. 오른쪽은 거울상."""
    side = 1
    sh, out, cav = shell_local(-34.0)
    loc = box(X_LOC, 10, -60, 60, -80, 60)
    cavw = pose(cav, side)
    b = pose(sh ^ loc, side)                                                    # 어디서나: 뒤로 X_LOC 까지
    b += pose(sh, side) ^ box(100, 300, -NOSE_HW, NOSE_HW, -200, 0)             # 코 앞(|y| ≤ 33.5): 지붕은 위 패드까지, 턱받이는 아래 패드까지
    b += (pose(out ^ box(CAP_X, 0, -60, 60, -80, 60), side) ^ end_slab(side))   # 끝 막이
    b += pad_a() + (pad_c() - cavw)
    b += ear(side) ^ box(100, 300, -100, 100, -200, Z_ROOF)
    b += box(xf_up(Z_ROOF) + 1.0, 200, -1.0, 1.0, PADA_Z[0], Z_ROOF) ^ pose(out, side)   # 가운데 갈빗대(위 패드 ↔ 앞판)
    b = (b ^ half_side(side)) - end_cut(side)
    b -= pose(plug_slot_local(), side)
    b -= pose(tie_slot_local(side), side)
    return b


def _build():
    h = _half()
    body = h + h.mirror([0, 1, 0])
    ko = limo_keepout()
    body ^= box(xf_up(Z_ROOF) + 1.0, 300, -100, 100, -200, Z_ROOF) + box(100, 300, EAR_Y0, 100, -200, Z_ROOF) + box(100, 300, -100, -EAR_Y0, -200, Z_ROOF)
    body -= ko
    body = fillet(body, 0.8)                            # 바깥 모서리 R0.8
    # 패드 판은 필렛 뒤에 다시 붙임: 테이프 면 가장자리가 둥글어져 넓이가 줄지 않게(뒤에 숨는 모서리)
    cavs = pose(shell_local(-34.0)[2], 1) + pose(shell_local(-34.0)[2], -1)
    slots = pose(plug_slot_local(), 1) + pose(plug_slot_local(), -1)
    body += pad_a() + ((pad_c() - cavs - slots - ko) ^ box(150, 200, -PADC_HW, PADC_HW, PADC_Z[0], PADC_TAPE_Z[1] + 0.8))
    for side in (-1, 1):
        body -= pose(aperture_cut(), side)
    return body


_CACHE = {}


def build():
    """띠(지붕 좌표, 조립 자세). 한 번 만든 것은 다시 쓴다."""
    k = id(fillet)
    if k not in _CACHE:
        _CACHE[k] = _build()
    return _CACHE[k]


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


def board_only_local():
    """듀폰을 꽂기 전의 보드: 보드 + 뒷면 부품 + 원통 + ㄱ자 핀(보드 아랫변에서 9) + 수정 진동자."""
    pcb = box(-BEZEL_T - 1.6, -BEZEL_T, -BOARD[0] / 2, BOARD[0] / 2, -BOARD[1] / 2, BOARD[1] / 2)
    comp = box(-BEZEL_T - 1.6 - 4.0, -BEZEL_T - 1.6, -15, 15, -BOARD[1] / 2 + 1, BOARD[1] / 2 - 1)
    cans = sum((cyl_x(CAN_D / 2, -BEZEL_T, -BEZEL_T + CAN_H, s * CAN_GAP / 2, 0) for s in (-1, 1)), m3.Manifold())
    pins = box(-BEZEL_T - 1.6 - 2.6, -BEZEL_T - 1.6, -5.5, 5.5, -BOARD[1] / 2 - 9, -BOARD[1] / 2)
    xtal = box(-BEZEL_T, -BEZEL_T + 3.5, -5.5, 5.5, -2.35, 2.35)
    return pcb + comp + cans + pins + xtal


def plug_local():
    """듀폰 커넥터(핀 아래 14) + 선 꺾임 10."""
    return box(-BEZEL_T - 1.6 - 2.6, -BEZEL_T - 1.6, -5.5, 5.5, -BOARD[1] / 2 - 30, -BOARD[1] / 2 - 6)


def wire_bend_local():
    return box(-BEZEL_T - 1.6 - 2.6, -BEZEL_T - 1.6, -5.5, 5.5, -BOARD[1] / 2 - 30, -BOARD[1] / 2 - 20)


def print_parts():
    """출력 자세(판 z 0): 지붕이 바닥(x 축으로 180° 뒤집음)."""
    t = to_t(build())
    t.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
    t.apply_translation(-t.bounds[0])
    return {"print_sonar.stl": t}


OBSOLETE = ["asm_sonar_tabs.stl", "asm_sonar_bolts.stl", "print_sonar_tab.stl", "print_sonar_tab_r.stl",
            "print_sonar_ctab.stl", "print_sonar_bolt.stl", "print_sonar_cbolt.stl"]

if __name__ == "__main__":
    d = pose(dummy_sensor_local(), 1) + pose(dummy_sensor_local(), -1)
    to_t(d).export(os.path.join(OUT, "dum_hcsr04.stl"))
    ta, tc, tb = tape_dummy()
    to_t(ta + tc + tb).export(os.path.join(OUT, "dum_sonar_tape.stl"))
    b = build()
    t = to_t(b)
    t.export(os.path.join(OUT, "asm_sonar.stl"))
    print("bounds", t.bounds.round(1).tolist(), "bodies", len(b.decompose()), "vol", round(b.volume()), "mass g", round(b.volume() * 1.24e-3, 1))
    v = lambda m: round(m.volume(), 2)
    print("body∩dummy", v(b ^ d), "body∩tape", v(b ^ (ta + tc + tb)), "tape∩dummy", v((ta + tc + tb) ^ d))
    for k, m in print_parts().items():
        m.export(os.path.join(OUT, k))
    for f in OBSOLETE:
        p = os.path.join(OUT, f)
        if os.path.exists(p):
            os.remove(p); print("지움", f)
    print("전체 검사는 check_sonar.py")
