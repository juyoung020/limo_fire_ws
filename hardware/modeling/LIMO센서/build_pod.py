"""LIMO 옆판 전장 상자 2 개 — ESP32·배선판(오른쪽 +y), 강압 모듈 2 개(왼쪽 −y). "각진 안장 가방" 꼴.

옆판(AGILE·X 판) 바깥면은 아래로 갈수록 바깥으로 기욺: 기울기 0.253 (약 14.2°).
구멍(팀 메시): z −23, x −36.9 / −22.9 / 22.1 / 36.1, Ø2.5 (M3 나사산 자리), 수평(y) 방향.
판 국소 좌표: u = x, v = 옆판 따라 위(구멍 높이 0), w = 바깥(판 뒷면 0).
  ※ (u, v, w) 는 +y 쪽(side +1)에서 왼손 좌표다 → to_world(·, +1) 은 거울상, to_world(·, −1) 은 그대로.
     그래서 오른쪽(+y) 출력 파일은 국소 모양을 u 로 뒤집어 내보내고, 오른쪽 암나사·볼트는 국소에서 왼나사로 만든다(조립하면 오른나사).

옆모습(u–v): 아래 z −33, 위 z +10 수평, 뒤 끝은 화면과 나란히(수직에서 31°, 위가 앞), 앞 끝은 수직에서 37°(위가 뒤), 모서리 R3.
덮개: 벽·천장 1.6, 위 바깥 모서리 30° 모따기(폭 8), 아래 바깥 모서리 15° 모따기(폭 5), 바깥 필렛 R0.8.
고정: 덮개 아랫변 갈고리 혀 2 개 → 판 홈, 위쪽 출력 M5 둥근머리 볼트 1 개(머리는 천장에 묻힘) → 판 기둥.
높이 쌓기(판 앞면 기준): 받침 4 + 기판 1.6 + 핀 헤더 몸통 2.54 + 듀폰 하우징 14 + 선 꺾임 8 = 30.14 → 안쪽 높이 31.
"""
import os, math, sys
import numpy as np
import manifold3d as m3
import trimesh

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(D, "..", "lib"))
import fasteners as F
from shape import fillet as _fillet
fillet = lambda A, r: _fillet(A, r, 32)   # 구 분할 32
OUT = os.path.join(D, "stl"); os.makedirs(OUT, exist_ok=True)
SEG = 48
K = 0.253
A = math.atan(K)
CA, SA = math.cos(A), math.sin(A)
Y_HOLE, Z_HOLE = 63.5, -23.0
HOLES_X = (-36.9, -22.9, 22.1, 36.1)
T = 3.0                                    # 판 두께
Z_BOT, Z_TOP = -33.0, 10.0
V0, V1 = (Z_BOT - Z_HOLE) / CA, (Z_TOP - Z_HOLE) / CA
RND = m3.JoinType.Round

# ---- 높이 쌓기 ----
PED, PCB_T = 4.0, 1.6                      # 보드 받침 높이, 기판 두께
HDR_H, DUP_L, BEND = 2.54, 14.0, 8.0       # 핀 헤더 몸통, 듀폰 하우징 길이, 선 꺾임 자리
WALL, ROOF = 1.2, 1.2      # 덮개는 힘을 안 받는 껍질: 0.4 노즐 3 줄(1.6 → 1.2, 덮개 2 개에서 약 18 g)
H_IN = float(math.ceil(PED + PCB_T + HDR_H + DUP_L + BEND))      # 30.14 → 31
W_ROOF0 = T + H_IN                         # 천장 아랫면 w
W_OUT = W_ROOF0 + ROOF                     # 덮개 바깥면 w
W_PCB0, W_PCB1 = T + PED, T + PED + PCB_T  # 기판 아랫면·윗면
W_DUP = W_PCB1 + HDR_H + DUP_L             # 듀폰 하우징 끝

# ---- 옆모습 ----
SR = 0.594                                 # 뒤 끝 기울기 du/dv (화면 받침 앞면을 limo.stl 에서 잰 값: 수직에서 31°)
SF = math.tan(math.radians(37.0)) * CA     # 앞 끝 기울기
UR_AT0 = -101.6                            # 뒤 끝 바깥선 u(v) = UR_AT0 + SR·v  (화면 받침까지 3 mm)
UR = lambda v: UR_AT0 + SR * v
WALLU_R, WALLU_F = WALL * math.hypot(1, SR), WALL * math.hypot(1, SF)
CT_V, CT_W = 8 * math.sin(math.radians(30)), 8 * math.cos(math.radians(30))    # 위 모따기: w 축에서 30°, 폭 8
CB_V, CB_W = 5 * math.sin(math.radians(15)), 5 * math.cos(math.radians(15))    # 아래 모따기: 15°, 폭 5

# ---- 보드 자리 (국소) ----
V_BRD0 = 2.0                               # 보드 아랫변: 그 아래 띠는 M3 드라이버 길·선 길
BRD_V = (V1 - WALL - 0.5 - 30.0, V1 - WALL - 0.5)                 # 배선판 50 × 30 (19 × 11 칸)
BRD_U = (UR(BRD_V[1]) + WALLU_R + 0.5, UR(BRD_V[1]) + WALLU_R + 0.5 + 50.0)
R_BOSS_D = 9.5
BOSS_U, BOSS_V = BRD_U[1] + 0.3 + R_BOSS_D / 2, 18.5              # 볼트 기둥: 배선판과 ESP32 사이, 위쪽
ESP_LEN, ESP_WID = 68.0, 26.0              # ESP32-S3-DevKitC-1 (프로젝트 문서의 "약 68 × 26"; 공식 도면은 62.9 × 25.4 — 실물 확인 필요)
ESP_U = (BOSS_U + R_BOSS_D / 2 + 0.3, BOSS_U + R_BOSS_D / 2 + 0.3 + ESP_LEN)
ESP_V = (V_BRD0, V_BRD0 + ESP_WID)
ESP_VC = sum(ESP_V) / 2
ESP_ROWS = (ESP_VC - 11.43, ESP_VC + 11.43)                        # 핀 줄 간격 22.86
HDR_U = (ESP_U[0] + 7.0, ESP_U[0] + 7.0 + 22 * 2.54)               # 22 핀 헤더 몸통(55.88)
USB_OFF = 6.75                             # USB 단자 2 개의 중심(공식 DXF 글자 위치 4.1·17.6 → ±6.75)
UF_REF = ESP_U[1] + 0.6 + WALLU_F
UF = lambda v: UF_REF + SF * (ESP_V[1] - v)                        # 앞 끝 바깥선
OUTLINE_PTS = [(UR(V0), V0), (UF(V0), V0), (UF(V1), V1), (UR(V1), V1)]
XL_U = (UR(V_BRD0 + 26.0) + WALLU_R + 0.5, UR(V_BRD0 + 26.0) + WALLU_R + 0.5 + 52.0)   # XL4015 52 × 26
XL_V = (V_BRD0, V_BRD0 + 26.0)
LM_U = (-15.0, 28.0)                       # LM2596 43 × 21
LM_V = (V_BRD0, V_BRD0 + 21.0)
MOD_PCB = W_PCB1

# ---- 볼트·갈고리 ----
M5 = (5.0, 1.25, 0.4)
HEAD_D, HEAD_H = 9.0, 3.0
LID_BOLT_L = 11.0
BOLT_W = W_OUT - HEAD_H - 0.2              # 볼트 머리 밑면 w (머리 윗면이 바깥면보다 0.2 안)
BOSS_W1 = BOLT_W - 3.0                     # 기둥 윗면 = 덮개 볼트 자리 밑면 (볼트가 덮개 자리를 기둥에 누름)
THREAD_D = 9.0                             # 암나사 깊이 (물림 8)
R_BOSS = [(BOSS_U, (BOSS_V - R_BOSS_D / 2, BOSS_V + R_BOSS_D / 2))]
L_BOSS = [(BOSS_U, (BOSS_V - R_BOSS_D / 2, BOSS_V + R_BOSS_D / 2))]
HOOK_U = (-84.0, 52.0)                     # 갈고리 혀 중심
HOOK_W, HOOK_T, TOE = 12.0, 1.6, 2.0
HV = V0 + 3.2                              # 판 홈 바깥 벽 v (홈 2.4: HV ~ HV + 2.4)
BRIDGE_W0, BRIDGE_W1 = 2.2, 3.8            # 발가락 위 다리: 밑 2.2 ~ 위 3.8 (두께 1.6)

# ---- 선 구멍(테두리로 열린 U 홈) ----
ARM_U, ARM_D = (0.8, 10.8), 7.0            # 팔 묶음(열화상 4 + 레이더 4 + 서보 6 + 앰프 5 = 19 가닥) 윗벽
PWR_V, PWR_D = (4.0, 16.0), 9.0            # 전원(19 V 2 + 강압 입출력 8 = 10 가닥, 0.5~0.75 mm²) 뒷벽
SON_U, SON_D = (-28.5, -22.5), 6.0         # 초음파 묶음 8 가닥, 아랫벽
USB_V, USB_D = (ESP_VC - 13.5, ESP_VC + 14.5), 6.5                # USB 단자 2 개 전체 폭


def box(u0, u1, v0, v1, w0, w1):
    return m3.Manifold.cube([u1 - u0, v1 - v0, w1 - w0]).translate([u0, v0, w0])


def cyl_w(r, w0, w1, u, v, seg=SEG):
    return m3.Manifold.cylinder(w1 - w0, r, r, seg).translate([u, v, w0])


def prism_uv(cs, w0, w1):
    return cs.extrude(w1 - w0).translate([0, 0, w0])


def _cs(pts):
    """점 목록 → 단면(도는 방향이 거꾸로면 뒤집음)."""
    if isinstance(pts, m3.CrossSection):
        return pts
    a = sum(p[0] * q[1] - q[0] * p[1] for p, q in zip(pts, pts[1:] + pts[:1]))
    return m3.CrossSection([pts if a > 0 else pts[::-1]])


def prism_vw(pts, u0, u1):
    """(v, w) 다각형을 u 로 밀어냄."""
    cs = _cs(pts)
    return cs.extrude(u1 - u0).transform([[0, 0, 1, u0], [1, 0, 0, 0], [0, 1, 0, 0]])


def prism_uw(pts, v0, v1):
    """(u, w) 다각형을 v 로 밀어냄."""
    cs = _cs(pts)
    return cs.extrude(v1 - v0).rotate([90, 0, 0]).translate([0, v1, 0])


def rrect(a0, a1, b0, b1, r=1.5):
    """모서리 R 의 사각형 단면."""
    return m3.CrossSection([[(a0, b0), (a1, b0), (a1, b1), (a0, b1)]]).offset(-r, RND).offset(r, RND, 2.0, 32)


def louvre(length, wid=3.0):
    """루버 홈 단면: 폭 3, 양 끝 45° 뾰족. 긴 축이 x."""
    h, a = wid / 2, length / 2
    return m3.CrossSection([[(-a, 0), (-a + h, -h), (a - h, -h), (a, 0), (a - h, h), (-a + h, h)]])


def to_world(mf, side):
    """국소 → 지붕 좌표. side +1 = +y 쪽(거울상), −1 = −y 쪽."""
    M = [[1, 0, 0, 0],
         [0, -SA * side, CA * side, Y_HOLE * side],
         [0, CA, SA, Z_HOLE]]
    return mf.transform(M)


def cyl_horiz(r, t0, t1, x):
    """옆판 나사 구멍 축(수평 y): 국소에서 e = (0, −sinA, cosA), 구멍 중심 (x, 0, 0). t 는 e 따라 거리."""
    return (m3.Manifold.cylinder(t1 - t0, r, r, SEG).translate([0, 0, t0])
            .rotate([math.degrees(A), 0, 0]).translate([x, 0, 0]))


CB_R = 3.5                                              # 머리 자리 Ø7 (머리 Ø6)
SPOT_T0 = (1.5 + CB_R * SA) / CA                        # 머리 밑 판이 가장 얇은 곳 1.5 mm
SEAT_W1 = (SPOT_T0 + 2.6) * CA + CB_R * SA + 0.05       # 자리 깊이 2.6 이 온 둘레에 나오는 돋움 높이
SEAT_R, SEAT_V = 5.25, -1.3


def outline():
    return m3.CrossSection([OUTLINE_PTS]).offset(-3.0, RND).offset(3.0, RND, 2.0, 48)       # 모서리 R3


def _profile(w_lo):
    """덮개 바깥 옆단면 (v, w): 위 30° 모따기, 아래 15° 모따기."""
    return m3.CrossSection([[(V0, w_lo), (V1, w_lo), (V1, W_OUT - CT_W), (V1 - CT_V, W_OUT),
                             (V0 + CB_V, W_OUT), (V0, W_OUT - CB_W)]])


_cache = {}


def _clean(mf):
    """계산 찌꺼기(부피 0 조각) 버림."""
    out = m3.Manifold()
    for q in mf.decompose():
        if abs(q.volume()) > 0.05:
            out += q
    return out


def lid_outer():
    """덮개 바깥 덩어리(볼록): 옆모습 × 옆단면, 필렛 R0.8. w T−6 부터(테두리 모서리는 둥글리지 않음)."""
    if "o" not in _cache:
        o = prism_uv(outline(), T - 6, W_OUT + 1) ^ prism_vw(_profile(T - 6), -200, 200)
        _cache["o"] = fillet(o, 0.8)
    return _cache["o"]


def lid_inner():
    if "i" not in _cache:
        pin = _profile(T - 12).offset(-WALL, m3.JoinType.Miter)
        _cache["i"] = prism_uv(outline().offset(-WALL, RND, 2.0, 48), T - 6, W_OUT) ^ prism_vw(pin, -200, 200)
    return _cache["i"]


# ---------------- 판 ----------------
def bracket(rect, edge, a, b, ledge=2.0, stop=True, gap=0.3, st=2.0):
    """보드 받침 턱(기판 밑 ledge 만큼) + 바깥 멈춤 턱(기판 윗면 +0.6 까지). edge: v0 v1 u0 u1, a~b 는 변을 따라."""
    u0, u1, v0, v1 = rect
    hs = W_PCB1 + 0.6
    if edge == "v0":
        m = box(a, b, v0 - gap - 0.01, v0 + ledge, T - 1.2, W_PCB0)
        s = box(a, b, v0 - gap - st, v0 - gap, T - 1.2, hs)
    elif edge == "v1":
        m = box(a, b, v1 - ledge, v1 + (gap + 0.01 if stop else gap - 0.1), T - 1.2, W_PCB0)
        s = box(a, b, v1 + gap, v1 + gap + st, T - 1.2, hs)
    elif edge == "u0":
        m = box(u0 - gap - 0.01, u0 + ledge, a, b, T - 1.2, W_PCB0)
        s = box(u0 - gap - st, u0 - gap, a, b, T - 1.2, hs)
    else:
        m = box(u1 - ledge, u1 + gap + 0.01, a, b, T - 1.2, W_PCB0)
        s = box(u1 + gap, u1 + gap + st, a, b, T - 1.2, hs)
    return (m + s) if stop else m


def bracket_keep(rect, edge, a, b, ledge=2.0, stop=True, gap=0.3, st=2.0):
    u0, u1, v0, v1 = rect
    o = (gap + st) if stop else gap
    return {"v0": (a, b, v0 - o, v0 + ledge), "v1": (a, b, v1 - ledge, v1 + o),
            "u0": (u0 - o, u0 + ledge, a, b), "u1": (u1 - ledge, u1 + o, a, b)}[edge]


def tie_cut(uc, vc, along):
    """케이블 타이 자리: 관통 홈 2 개(1.8 × 3.4, 중심 간격 9) + 뒷면 홈(깊이 1.4)으로 타이가 판 뒤로 돌아 나옴."""
    if along == "u":      # 선이 v 로 지나감 → 홈이 u 로 떨어져 있음
        c = box(uc - 5.4, uc - 3.6, vc - 1.7, vc + 1.7, -1, T + 1) + box(uc + 3.6, uc + 5.4, vc - 1.7, vc + 1.7, -1, T + 1)
        c += box(uc - 5.4, uc + 5.4, vc - 1.7, vc + 1.7, -1, 1.4)
        keep = (uc - 5.4, uc + 5.4, vc - 1.7, vc + 1.7)
    else:
        c = box(uc - 1.7, uc + 1.7, vc - 5.4, vc - 3.6, -1, T + 1) + box(uc - 1.7, uc + 1.7, vc + 3.6, vc + 5.4, -1, T + 1)
        c += box(uc - 1.7, uc + 1.7, vc - 5.4, vc + 5.4, -1, 1.4)
        keep = (uc - 1.7, uc + 1.7, vc - 5.4, vc + 5.4)
    return c, keep


GRID_P, RIB = 18.0, 3.0
GRID_U0, GRID_V0 = 0.0, 0.0             # 격자 시작 자리(창 넓이가 가장 커지게 고름)


def _inside(cs, X, Y):
    """단면 안에 있는 점(짝홀 규칙)."""
    inside = np.zeros(X.shape, bool)
    for poly in cs.to_polygons():
        poly = np.asarray(poly)
        x0, y0 = poly[:, 0], poly[:, 1]
        x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
        for a0, b0, a1, b1 in zip(x0, y0, x1, y1):
            if b0 == b1:
                continue
            hit = ((b0 > Y) != (b1 > Y)) & (X < (a1 - a0) * (Y - b0) / (b1 - b0) + a0)
            inside ^= hit
    return inside


def _max_rect(M):
    """참인 칸으로만 된 가장 큰 직사각형 (i0, i1, j0, j1) — 줄마다 막대 높이를 쌓아 찾음."""
    best, h = (0, None), np.zeros(M.shape[1], int)
    for i in range(M.shape[0]):
        h = np.where(M[i], h + 1, 0)
        for j0 in range(M.shape[1]):
            m = h[j0]
            for j1 in range(j0, M.shape[1]):
                m = min(m, h[j1])
                if m == 0:
                    break
                if m * (j1 - j0 + 1) > best[0]:
                    best = (m * (j1 - j0 + 1), (i - m + 1, i, j0, j1))
    return best[1]


def windows_cs(keeps, circles, soft=()):
    """판 창(단면): 18 mm 격자 살(폭 3) 사이 칸마다, 테두리 3.2 안쪽이면서 기능 요소(+3 mm, 타이 홈은 +1.5)를 피한
    가장 큰 직사각형 하나(남은 자리가 넉넉하면 하나 더)를 뚫는다. 모든 창이 격자에 맞춘 둥근 사각형(R1.5)."""
    field = outline().offset(-3.2, RND)
    kz = m3.CrossSection()
    for (a, b, c, d) in soft:
        kz += m3.CrossSection([[(a, c), (b, c), (b, d), (a, d)]]).offset(-1.5, m3.JoinType.Miter)   # 아래에서 +3 → 결국 +1.5
    for (a, b, c, d) in keeps:
        kz += m3.CrossSection([[(a, c), (b, c), (b, d), (a, d)]])
    for (x, y, r) in circles:
        kz += m3.CrossSection.circle(r, 48).translate([x, y])
    free = field - kz.offset(3.0, RND, 2.0, 32)
    out = m3.CrossSection()
    st = 0.5
    for u0 in np.arange(-130.0 + GRID_U0, 100, GRID_P):
        for v0 in np.arange(V0 - GRID_P + GRID_V0, V1, GRID_P):
            us = np.arange(u0 + st / 2, u0 + GRID_P - RIB, st)
            vs = np.arange(v0 + st / 2, v0 + GRID_P - RIB, st)
            X, Y = np.meshgrid(us, vs)
            M = _inside(free, X, Y)
            for _ in range(3):
                r = _max_rect(M)
                if r is None:
                    break
                i0, i1, j0, j1 = r
                a, b, c, d = us[j0] - st / 2 + 0.1, us[j1] + st / 2 - 0.1, vs[i0] - st / 2 + 0.1, vs[i1] + st / 2 - 0.1
                if min(b - a, d - c) < 3.5 or (b - a) * (d - c) < 20.0:
                    break
                out += rrect(a, b, c, d, 1.5)
                M[max(0, i0 - 6):i1 + 7, max(0, j0 - 6):j1 + 7] = False        # 다음 창과는 살 3 mm
    return out


def _thread_cut(side):
    t = F.female_cut(M5[0], THREAD_D, M5[1], M5[2])
    if side > 0:
        t = t.mirror([1, 0, 0])        # +y 쪽은 거울상으로 놓이므로 국소에서 왼나사
    return t.translate([BOSS_U, BOSS_V, BOSS_W1 - THREAD_D])


def _plate(side, boards, ties, usb_pocket=False):
    """boards: [(rect, [(edge, a, b, kwargs)])], ties: [(uc, vc, along)]"""
    keeps, circles = [], []
    feats = []                                                     # 돋은 것들: 하나씩 필렛(판 속으로 1.2 묻혀 밑동은 안 둥글어짐)
    for x in HOLES_X:                                              # M3 머리 자리 돋움
        feats.append(cyl_w(SEAT_R, T - 1.2, SEAT_W1, x, SEAT_V))
        circles.append((x, SEAT_V, SEAT_R))
    for rect, brs in boards:
        for (edge, a, b, kw) in brs:
            feats.append(bracket(rect, edge, a, b, **kw))
            keeps.append(bracket_keep(rect, edge, a, b, **kw))
    r = R_BOSS_D / 2                                               # 볼트 기둥 + 밑동(양옆 보드 멈춤 턱)
    feats.append(cyl_w(r, T - 1.2, BOSS_W1, BOSS_U, BOSS_V) + box(BOSS_U - r, BOSS_U + r, BOSS_V - r, BOSS_V + r, T - 1.2, W_PCB1 + 0.6))
    # 기둥 받침 살: 속 빈 Ø9.5 기둥(벽 2)이 28 mm 혼자 서 있어 뿌리에서 꺾일 수 있음 → 보드가 없는 ±v 쪽으로 세모 살 2 장(두께 2.4, 밑 8 → 기둥 높이 2/3 에서 0)
    GUS = 8.0
    for sg in (-1, 1):
        va, vb = sorted((BOSS_V + sg * (r - 0.5), BOSS_V + sg * (r + GUS)))
        ta, tb = sorted((BOSS_V + sg * (r - 0.5), BOSS_V + sg * (r - 0.3)))
        feats.append(m3.Manifold.hull(box(BOSS_U - 1.2, BOSS_U + 1.2, va, vb, T - 1.2, T + 0.4)
                                      + box(BOSS_U - 1.2, BOSS_U + 1.2, ta, tb, T + (BOSS_W1 - T) * 0.67, T + (BOSS_W1 - T) * 0.67 + 0.4)))
    keeps.append((BOSS_U - r, BOSS_U + r, BOSS_V - r - GUS, BOSS_V + r + GUS))
    slab = prism_uv(outline(), -3.0, T)
    for uc in HOOK_U:                                              # 갈고리 다리 돋움(판 가장자리와 한 면)
        slab += box(uc - 9, uc + 9, V0 - 5, HV, -3.0, BRIDGE_W1) ^ prism_uv(outline(), -3.0, 10)
        keeps.append((uc - 9, uc + 9, V0, HV + 2.4))
    cuts = m3.Manifold()
    soft = []
    for (uc, vc, along) in ties:
        c, kp = tie_cut(uc, vc, along)
        cuts += c; soft.append(kp)
    if usb_pocket:
        keeps.append((ESP_U[1] - 1, 120, USB_V[0], USB_V[1]))
    p = fillet(slab, 0.8) ^ box(-300, 300, -300, 300, 0, 100)      # 바깥 모서리 R0.8 (바닥에 닿는 뒷면 모서리는 그대로)
    p -= prism_uv(windows_cs(keeps, circles, soft), -4, T + 1)
    for f in feats:
        p += fillet(f, 0.8)
    # ---- 필렛 뒤에 뚫는 것 ----
    for x in HOLES_X:
        p -= cyl_horiz(1.7, -5, 20, x)                             # M3 관통, 수평
        p -= cyl_horiz(CB_R, SPOT_T0, SPOT_T0 + 80, x)             # 머리 자리 Ø7 (깊이 2.6 이상) + 드라이버 길
    p -= cyl_w(2.75, -1, BOSS_W1 - THREAD_D + 0.5, BOSS_U, BOSS_V)   # 기둥 속 비움(바닥까지 뚫림 → 서포트 없음)
    p -= _thread_cut(side)
    for uc in HOOK_U:
        p -= box(uc - HOOK_W / 2 - 0.3, uc + HOOK_W / 2 + 0.3, HV, HV + 2.4, -1, BRIDGE_W1 + 1)       # 홈 12.6 × 2.4
        p -= prism_vw([(HV + 0.01, -0.5), (HV + 0.01, BRIDGE_W0), (HV - BRIDGE_W0, 0.0), (HV - BRIDGE_W0 - 0.5, -0.5)],
                      uc - HOOK_W / 2 - 0.3, uc + HOOK_W / 2 + 0.3)                                    # 다리 밑 45° 도피(발가락 자리)
        p -= prism_vw([(HV + 2.39, -0.5), (HV + 2.39, T + 1), (HV + 3.6, T + 1), (HV + 2.4, 0.4)],
                      uc - HOOK_W / 2 - 0.3, uc + HOOK_W / 2 + 0.3)                                    # 안쪽 벽 비탈(기울여 넣는 길)
    p -= cuts
    if usb_pocket:                                                 # USB 플러그 몸통 밑 자리(깊이 1.4)
        p -= prism_uv(rrect(ESP_U[1] + 0.8, 140, USB_V[0], USB_V[1] - 1.0), T - 1.4, T + 1)
    return _clean(p)


R_TIES = [((ARM_U[0] + ARM_U[1]) / 2, V1 - WALL - 0.4 - 1.7, "u"), (-86.8, sum(PWR_V) / 2, "v"), (-11.0, -3.6, "v")]
L_TIES = [(-86.8, sum(PWR_V) / 2, "v")]


def _right_boards():
    b0, b1 = BRD_U
    e0, e1 = ESP_U
    perf = ((b0, b1, *BRD_V), [("v0", b0 + 4, b0 + 10, {}), ("v0", b1 - 18, b1 - 12, {}),
                               ("v1", b0 + 4, b0 + 10, {"stop": False}), ("v1", b1 - 18, b1 - 12, {"stop": False}),
                               ("u0", 5.0, 11.0, {})])
    esp = ((e0, e1, *ESP_V), [("v0", e0 + 1.0, e0 + 6.5, {"ledge": 1.5}), ("v1", e0 + 1.0, e0 + 6.5, {"ledge": 1.5}),
                              ("v0", e1 - 4.6, e1, {"ledge": 1.5}), ("v1", e1 - 4.6, e1 - 1.6, {"ledge": 1.5, "st": 1.6})])
    return [perf, esp]


def _left_boards():
    x0, x1 = XL_U
    l0, l1 = LM_U
    xl = ((x0, x1, *XL_V), [("v0", x0 + 6, x0 + 12, {}), ("v0", x1 - 18, x1 - 12, {}), ("v1", x0 + 6, x0 + 12, {}),
                            ("v1", x1 - 18, x1 - 12, {}), ("u0", 17.0, 23.0, {}), ("u1", 6.0, 12.0, {})])
    lm = ((l0, l1, *LM_V), [("v0", l0 + 3, l0 + 9, {}), ("v0", l1 - 20, l1 - 14, {}), ("v1", l0 + 3, l0 + 9, {}),
                            ("v1", l1 - 20, l1 - 14, {}), ("u0", 9.0, 15.0, {}), ("u1", 9.0, 15.0, {})])
    return [xl, lm]


def pod_right():
    """+y: 배선판(뒤, 19 × 11 칸 50 × 30) + ESP32-S3-DevKitC-1(앞, 핀이 덮개 쪽, USB 가 앞)."""
    return _plate(1, _right_boards(), R_TIES, usb_pocket=True)


def pod_left():
    """−y: XL4015(뒤, 52 × 26) + LM2596(앞, 43 × 21)."""
    return _plate(-1, _left_boards(), L_TIES)


# ---------------- 덮개 ----------------
def _post(tip, root, w_tip):
    """누름 기둥: 천장 뿌리(5 × 5 이상) → 보드 윗면 0.2 위 끝. tip/root = (u0, u1, v0, v1)."""
    a = box(tip[0], tip[1], tip[2], tip[3], w_tip, w_tip + 0.2)
    b = box(root[0], root[1], root[2], root[3], W_ROOF0 - 0.2, W_OUT - 0.2)
    return (a + b).hull()


def _lid(posts, fins, uslots, louv, intake_u):
    """posts: [(tip, root)], fins: 윗벽에서 나온 누름 턱 [(u0, u1, v_from)], uslots: 테두리로 열린 홈(빼낼 덩어리),
    louv: (u 중심, [v 중심...]), intake_u: 아랫벽 흡기 홈 u 중심들."""
    outer, inner = lid_outer(), lid_inner()
    L = (outer - inner) ^ box(-300, 300, -300, 300, T, 100)
    for c in uslots:
        L -= c
    extra = m3.Manifold()
    for (tip, root) in posts:
        extra += _post(tip, root, W_PCB1 + 0.2)
    for (u0, u1, vf) in fins:                      # 윗벽 누름 턱: 낮은 선반 + 45° 밑받침(기울여 열 때 높은 부품을 쓸지 않게 낮게)
        w0, vw = W_PCB1 + 0.2, V1 - WALL + 0.3
        extra += prism_vw([(vf, w0), (vw, w0), (vw, w0 + 2.7 + (vw - vf)), (vf, w0 + 2.7)], u0, u1)
    extra += cyl_w(6.5, BOSS_W1, W_OUT, BOSS_U, BOSS_V)                               # 볼트 자리 돋움
    for uc in HOOK_U:                                                                 # 갈고리 뿌리(45° 밑면)
        extra += prism_vw([(V0 + 0.5, T), (HV + 0.4 + HOOK_T, T), (HV + 0.4 + HOOK_T, T + 3.0), (V0 + 0.5, T + 3.0 + HV + HOOK_T - V0 - 0.1)],
                          uc - HOOK_W / 2, uc + HOOK_W / 2)
    L += extra ^ outer ^ box(-300, 300, -300, 300, T, 100)
    for uc in HOOK_U:                                                                 # 판 다리 돋움이 들어오는 테두리 홈
        L -= box(uc - 9.3, uc + 9.3, V0 - 1, HV + 0.2, T - 1, BRIDGE_W1 + 0.2)
    # 볼트: 머리 자리(F.counterbore) + 45° 원뿔 바닥(천장을 바닥에 놓고 출력해도 뜨지 않음) + 관통
    L -= F.counterbore(HEAD_D, HEAD_H, 0.5, 0.2, 5.0).rotate([180, 0, 0]).translate([BOSS_U, BOSS_V, BOLT_W + 0.5])
    L -= m3.Manifold.cylinder(2.3, 2.75, 5.05, SEG).translate([BOSS_U, BOSS_V, BOLT_W + 0.5 - 2.3 + 0.01])
    L -= cyl_w(2.8, BOSS_W1 - 1, W_OUT, BOSS_U, BOSS_V)
    uc, vs = louv                                                                     # 루버(천장)
    for v in vs:
        L -= prism_uv(louvre(24.0).translate([uc, v]), W_ROOF0 - 4, W_OUT + 1)
    for u in intake_u:                                                                # 흡기(아랫벽, 긴 축이 w)
        L -= prism_uw(louvre(12.0).rotate(90).translate([u, T + 15.0]), V0 - 1, V0 + WALL + 0.6)
    hooks = m3.Manifold()                                                             # 갈고리 혀 + 발가락(테두리 아래로)
    for uc in HOOK_U:
        tv = HV + 0.4
        hooks += prism_vw([(tv, 0.2), (tv + HOOK_T, 0.2), (tv + HOOK_T, T + 0.5), (tv, T + 0.5)], uc - HOOK_W / 2, uc + HOOK_W / 2)
        hooks += prism_vw([(tv + 0.01, BRIDGE_W0 - 0.283 + 0.4), (tv - TOE, BRIDGE_W0 - 0.283 + 0.4 - TOE), (tv - TOE, 0.2), (tv + 0.01, 0.2)],
                          uc - HOOK_W / 2, uc + HOOK_W / 2)
    return _clean(L + hooks)


def _uslot_uw(u0, u1, depth, v0, v1):
    return prism_uw(rrect(u0, u1, T - 5, T + depth), v0, v1)


def _uslot_vw(v0, v1, depth, u0, u1):
    return prism_vw(rrect(v0, v1, T - 5, T + depth), u0, u1)


PWR_SLOT = lambda: _uslot_vw(PWR_V[0], PWR_V[1], PWR_D, -130, UR(PWR_V[1]) + WALLU_R + 0.6)
LOUV_V5 = (1.5, 7.5, 13.5, 19.5, 25.5)
R_LOUV = (14.0, LOUV_V5[2:])               # ESP32 위 3 줄
L_LOUV = (sum(XL_U) / 2, LOUV_V5)          # XL4015 위 5 줄
R_INTAKE = (8.0, 14.0, 20.0)
L_INTAKE = tuple(sum(XL_U) / 2 + 6.0 * i for i in (-2, -1, 0, 1, 2))


def lid_right():
    posts = []
    for row in ESP_ROWS:                                           # ESP32: 핀 헤더 몸통 양 끝 바로 밖에서 기판을 누름(몸통이 기둥 사이에 끼어 앞뒤로 못 감)
        top = row > ESP_VC
        posts.append(((HDR_U[0] - 2.8, HDR_U[0] - 0.3, row - 1.5, row + 1.5),
                      (HDR_U[0] - 5.8, HDR_U[0] - 0.3, row - 2.5 + (0.5 if top else -0.5), row + 2.5 + (0.5 if top else -0.5))))
        posts.append(((HDR_U[1] + 0.3, HDR_U[1] + 2.8, row - 1.5, row + 1.5),
                      (HDR_U[1] + 0.3, (UF(row) + 1 if top else HDR_U[1] + 5.3), row - 2.5, row + 2.5)))
    for uc in (BRD_U[0] + 15.0, BRD_U[1] - 5.0):                   # 배선판 아랫변
        posts.append(((uc - 1.5, uc + 1.5, BRD_V[0], BRD_V[0] + 1.8), (uc - 2.5, uc + 2.5, BRD_V[0] - 3.2, BRD_V[0] + 1.8)))
    fins = [(BRD_U[0] + 12, BRD_U[0] + 18, BRD_V[1] - 1.5), (BRD_U[1] - 10, BRD_U[1] - 4, BRD_V[1] - 1.5)]      # 배선판 윗변
    us = [_uslot_uw(ARM_U[0], ARM_U[1], ARM_D, V1 - WALL - 0.6, V1 + 1),
          PWR_SLOT(),
          _uslot_uw(SON_U[0], SON_U[1], SON_D, V0 - 1, V0 + WALL + 0.6),
          _uslot_vw(USB_V[0], USB_V[1], USB_D, ESP_U[1] - 0.5, 140)]
    return _lid(posts, fins, us, R_LOUV, R_INTAKE)


def lid_left():
    posts = []                                                     # 모듈 양 끝(짧은 변) 가운데 여백을 누름: 기둥 윗쪽(−v)에 높은 부품이 없어 기울여 열 수 있음
    for (u0, u1), vc in ((XL_U, 15.0), (LM_U, sum(LM_V) / 2)):
        posts.append(((u0, u0 + 1.9, vc - 2.0, vc + 2.0), (u0 - 3.1, u0 + 1.9, vc - 2.5, vc + 2.5)))
        v2 = 8.0 if (u0, u1) == XL_U else vc                       # XL4015 앞 끝은 볼트 기둥을 피해 아래로
        posts.append(((u1 - 1.9, u1, v2 - 2.0, v2 + 2.0), (u1 - 1.9, u1 + 3.1, v2 - 2.5, v2 + 2.5)))
    return _lid(posts, [], [PWR_SLOT()], L_LOUV, L_INTAKE)


def lid_bolt(side, a=0.0):
    """덮개 볼트 1 개(국소). F.pan_bolt M5 × 11, 머리 Ø9 × 3 가 바깥(+w), 축 −w. 머리 밑면 w = BOLT_W.
    side +1 은 국소에서 왼나사(to_world 가 거울상이라 조립하면 오른나사). a = 축 둘레 위상(도)."""
    b = F.pan_bolt(M5[0], LID_BOLT_L, M5[1], M5[2], head_d=HEAD_D, head_h=HEAD_H).rotate([0, 0, a]).rotate([180, 0, 0])
    if side > 0:
        b = b.mirror([1, 0, 0])
    return b.translate([BOSS_U, BOSS_V, BOLT_W])


def lid_bolts(bosses, side=None, a=0.0):
    if side is None:
        side = -1 if bosses is L_BOSS else 1
    b = m3.Manifold()
    for _ in bosses:
        b += lid_bolt(side, a)
    return b


def tilt(mf, theta, dv=0.0, dw=0.0):
    """덮개를 아랫변 바깥 모서리(v V0, w T) 둘레로 theta 도 열고 (dv, dw) 옮김(국소)."""
    return mf.translate([0, -V0, -T]).rotate([theta, 0, 0]).translate([0, V0 + dv, T + dw])


def to_t(mf):
    me = mf.to_mesh()
    return trimesh.Trimesh(me.vert_properties[:, :3], me.tri_verts)


# ---------------- 보드 모형 ----------------
def dummies_local():
    e0, e1 = ESP_U
    vc = ESP_VC
    esp = box(e0, e1, *ESP_V, W_PCB0, W_PCB1)                                             # 기판(부품 면이 판 쪽)
    esp += box(e0, e0 + 19.0, vc - 9, vc + 9, W_PCB0 - 3.2, W_PCB0 + 0.01)                # 모듈(뒤 끝)
    esp += box(e0 + 19.0, e1 - 9.0, vc - 9, vc + 9, W_PCB0 - 2.0, W_PCB0 + 0.01)          # 부품
    for row in ESP_ROWS:
        esp += box(HDR_U[0], HDR_U[1], row - 0.5, row + 0.5, W_PCB0 - 1.0, W_PCB0 + 0.01)  # 납땜 꼬리
        # J1(아래 줄, 배선 14 가닥): 핀 헤더 몸통 + 핀 + 듀폰 하우징. J3(위 줄): 쓰지 않음 → 몸통 + 핀(6)만
        esp += box(HDR_U[0], HDR_U[1], row - 1.27, row + 1.27, W_PCB1 - 0.01, W_DUP if row < ESP_VC else W_PCB1 + HDR_H + 6.0)
    for s in (-1, 1):                                                                      # USB 단자 2 + 플러그 2
        c = vc + s * USB_OFF
        wc = W_PCB0 - 1.6
        esp += box(e1 - 7.3, e1 + 1.0, c - 4.47, c + 4.47, W_PCB0 - 3.2, W_PCB0 + 0.01)
        esp += box(e1 + 0.9, e1 + 1.5, c - 4.15, c + 4.15, wc - 1.3, wc + 1.3)
        esp += box(e1 + 1.4, e1 + 23.4, c - 6.2, c + 6.2, wc - 3.5, wc + 3.5)
        esp += m3.Manifold.cylinder(12.0, 2.25, 2.25, 24).rotate([0, 90, 0]).translate([e1 + 23.3, c, wc])
    b0, b1 = BRD_U
    bv0, bv1 = BRD_V
    brd = box(b0, b1, bv0, bv1, W_PCB0, W_PCB1)
    brd += box(b0 + 2.0, b1 - 2.0, bv0 + 2.0, bv1 - 2.0, W_PCB0 - 2.2, W_PCB0 + 0.01)     # 납땜 면
    brd += box(b0 + 3.41, b0 + 46.59, bv1 - 26.43, bv1 - 6.11, W_PCB1 - 0.01, W_DUP)       # 핀 헤더 + 듀폰(2~18 열, 3~10 줄)
    brd += cyl_w(5.0, W_PCB1 - 0.01, W_PCB1 + 17.0, b0 + 14.84, bv1 - 11.2, 32)            # C1 1000 µF (Ø10 × 17 이하)
    xl = box(*XL_U, *XL_V, W_PCB0, W_PCB1) + box(XL_U[0] + 2, XL_U[1] - 2, XL_V[0] + 2, XL_V[1] - 2, W_PCB0 - 2.0, W_PCB1 + 14.4)
    lm = box(*LM_U, *LM_V, W_PCB0, W_PCB1) + box(LM_U[0] + 2, LM_U[1] - 2, LM_V[0] + 2, LM_V[1] - 2, W_PCB0 - 2.0, W_PCB1 + 12.4)
    return brd, esp, xl, lm


def dummies():
    brd, esp, xl, lm = dummies_local()
    return to_world(esp + brd, 1), to_world(xl + lm, -1)


def cables_local():
    """선 묶음이 지나는 자리(검사용): 팔 8 × 5, 전원 8 × 7, 초음파 4 × 4."""
    arm = box(ARM_U[0] + 1, ARM_U[1] - 1, V1 - WALL - 3.6, V1 + 12, T + 0.2, T + 5.2)
    pwr = box(UR(sum(PWR_V) / 2) - 12, -84.5, PWR_V[0] + 2, PWR_V[1] - 2, T + 0.2, T + 7.2)
    son = box(SON_U[0] + 1, SON_U[1] - 1, V0 - 12, -7.0, T + 0.2, T + 4.2)
    return arm + pwr + son, pwr


def print_pose(mf, side, flip=False):
    """출력 자세. 판: 뒷면이 바닥. 덮개(flip): 천장이 바닥. side +1 은 u 로 뒤집음(조립 모양과 같은 손)."""
    if side > 0:
        mf = mf.mirror([1, 0, 0])
    if flip:
        mf = mf.rotate([180, 0, 0])
    return mf.translate([0, 0, -mf.bounding_box()[2]])      # x·y 는 국소 좌표 그대로(오른쪽은 u 가 뒤집힘) → 구멍 자리 검사에 그대로 씀


if __name__ == "__main__":
    import time
    t0 = time.time()
    R, Lp = pod_right(), pod_left()
    LR, LL = lid_right(), lid_left()
    de, dl = dummies()
    cr, cl = cables_local()
    for nm, plate_l, lid_l, side, dm in (("pod_right", R, LR, 1, de), ("pod_left", Lp, LL, -1, dl)):
        w = to_world(plate_l, side); lw = to_world(lid_l, side); bw = to_world(lid_bolt(side), side)
        to_t(w).export(os.path.join(OUT, f"asm_{nm}.stl")); to_t(lw).export(os.path.join(OUT, f"asm_{nm}_lid.stl"))
        to_t(bw).export(os.path.join(OUT, f"asm_{nm}_bolts.stl"))
        to_t(print_pose(plate_l, side)).export(os.path.join(OUT, f"print_{nm}.stl"))
        to_t(print_pose(lid_l, side, flip=True)).export(os.path.join(OUT, f"print_{nm}_lid.stl"))
        print(nm, "판 덩어리", len(plate_l.decompose()), "덮개 덩어리", len(lid_l.decompose()),
              "판 g", round(plate_l.volume() * 1.24e-3, 1), "덮개 g", round(lid_l.volume() * 1.24e-3, 1),
              "world", [round(x, 1) for x in lw.bounding_box()])
    to_t(de).export(os.path.join(OUT, "dum_pod_right.stl")); to_t(dl).export(os.path.join(OUT, "dum_pod_left.stl"))
    to_t(to_world(cr, 1)).export(os.path.join(OUT, "dum_pod_right_cables.stl"))
    to_t(to_world(cl, -1)).export(os.path.join(OUT, "dum_pod_left_cables.stl"))
    pb = F.pan_bolt(M5[0], LID_BOLT_L, M5[1], M5[2], head_d=HEAD_D, head_h=HEAD_H)
    to_t(pb.translate([0, 0, HEAD_H])).export(os.path.join(OUT, "print_pod_bolt.stl"))       # 머리가 바닥
    print("H_IN", H_IN, "W_OUT", W_OUT, "outline", [(round(a, 1), round(b, 1)) for a, b in OUTLINE_PTS], "초", round(time.time() - t0, 1))
