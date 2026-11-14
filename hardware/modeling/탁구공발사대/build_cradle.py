"""LIMO 지붕 받침대(크래들) + 각도 서보(MG996R) 자리.

좌표(지붕 좌표): 원점 = 지붕 M3 구멍 65x52 사각형 중심의 지붕 윗면, x 앞, y 왼, z 위. base_link 는 원점보다 14.9 mm 앞.
지붕 구멍: 65x52 네 개(±32.5, ±26) + 10 mm 간격 네 개(±5, ±36) — 공식 매뉴얼 도면, 10 mm 간격은 팀 메시에도 있음.
환기구(지붕 x 6.6~41.3, |y| ≤ 11) 위는 비움. 고정 나사 M3x5(LIMO 동봉 20개) — 바닥판 3 mm.

발사대 축: 지붕 좌표 (X_AX, 0, H_AX). 발사대 자기 좌표의 축 = build_launcher.AX.
각도 θ: 발사대 점 p → Ry(-θ)(p - AX) + (X_AX, 0, H_AX)  (θ 가 클수록 총구가 들림)."""
import os, math, json
import numpy as np, manifold3d as m3
import build_launcher as B
from build_launcher import box, cyl_y, cyl_z, tear_y, servo_solid, servo_holes, place, SERVO_K, HORN_TOP, HORN_T

D = os.path.dirname(os.path.abspath(__file__))
X_AX, H_AX = -25.0, 62.0
ANG = (5.0, 30.0)
BASE_T = 3.0
ROOF_HOLES = [(sx * 32.5, sy * 26.0) for sx in (-1, 1) for sy in (-1, 1)] + [(sx * 5.0, sy * 36.0) for sx in (-1, 1) for sy in (-1, 1)]
VENT = (5.0, 43.0, -13.0, 13.0)
# 각도 서보: 혼 윗면 = 몸통 -y 면(-KEEL_Y), 축 끝 = 그보다 1.6 바깥, 플랜지 윗면 = 축 끝 - 13.8
TILT_SPLINE_Y = -B.KEEL_Y - HORN_TOP
TILT_FLANGE_Y = TILT_SPLINE_Y - SERVO_K["spline_end"]
CHEEK_T = 4.0
CHEEK_N = (TILT_FLANGE_Y, TILT_FLANGE_Y + CHEEK_T)          # -y 볼(서보 플랜지가 바깥 면에)
CHEEK_P = (B.KEEL_Y + 1.0, B.KEEL_Y + 1.0 + CHEEK_T)        # +y 볼(축 핀)

def tilt_servo_place(m):
    # w = +y(혼이 몸통 쪽), u = -x(몸통이 축 뒤), v = w x u = +z
    return place(m, (X_AX, TILT_FLANGE_Y, H_AX), (-1, 0, 0), (0, 0, 1), (0, 1, 0))

def pose(m, theta):
    """발사대 자기 좌표 → 지붕 좌표(각도 theta)."""
    ax, az = B.AX
    return m.translate([-ax, 0, -az]).rotate([0, -theta, 0]).translate([X_AX, 0, H_AX])

def build_cradle():
    S = SERVO_K
    uc = S["body_l"] / 2 - S["shaft_off"]
    hx = [X_AX - (uc + d * S["hole_dl"] / 2) for d in (-1, 1)]       # 플랜지 구멍 x
    xa, xb = min(hx) - 6, max(hx) + 6
    ytop = H_AX + 12
    # 바닥판
    base = box(-42.0, 37.0, -44.0, 44.0, 0, BASE_T) + box(xa - 4, xb + 4, -54.0, -40.0, 0, BASE_T)
    base -= box(VENT[0], VENT[1] + 5, VENT[2], VENT[3], -1, BASE_T + 1)
    for (x, y) in ROOF_HOLES:
        base -= cyl_z(1.7, -1, BASE_T + 1, x, y, 24)
    # -y 볼 + 서보 주머니 끝벽 + +y 볼
    cn = box(xa, xb, CHEEK_N[0], CHEEK_N[1], 0, ytop - 6) + (cyl_y(10.0, CHEEK_N[0], CHEEK_N[1], X_AX, H_AX) + box(X_AX - 10, X_AX + 10, CHEEK_N[0], CHEEK_N[1], 0, H_AX))
    cn = cn ^ box(xa, xb, -200, 200, 0, ytop)
    ends = box(xa - 4, xa, -54.0, CHEEK_N[1], 0, H_AX + 10) + box(xb, xb + 4, -54.0, CHEEK_N[1], 0, H_AX + 10)
    cp = box(X_AX - 15, X_AX + 15, CHEEK_P[0], CHEEK_P[1], 0, H_AX) + cyl_y(15.0, CHEEK_P[0], CHEEK_P[1], X_AX, H_AX)
    cp = cp ^ box(-200, 200, -200, 200, 0, H_AX + 15)
    # +y 볼 바깥 45° 받침 2장
    gus = m3.Manifold()
    for gx in (X_AX - 15, X_AX + 11):
        gus += m3.Manifold.hull(box(gx, gx + 4, CHEEK_P[1], CHEEK_P[1] + 0.1, 0, H_AX - 15) + box(gx, gx + 4, CHEEK_P[1], min(CHEEK_P[1] + H_AX - 15, 46.0), 0, 0.1))
    # 바닥판 창: 나사 자리(r 6)·볼과 끝벽 밑 띠·둘레 띠(폭 6)·구멍 잇는 띠(폭 8)만 남긴다
    keep = box(-42.0, 37.0, -44.0, 44.0, -1, BASE_T + 1) - box(-36.0, 31.0, -38.0, 38.0, -2, BASE_T + 2)
    keep += box(xa - 4, xb + 4, -54.0, -40.0, -1, BASE_T + 1)
    for (x, y) in ROOF_HOLES:
        keep += cyl_z(6.0, -1, BASE_T + 1, x, y, 32)
    for y in (-26.0, 26.0):
        keep += box(-42.0, 37.0, y - 4, y + 4, -1, BASE_T + 1)
    keep += box(xa - 4, xb + 4, -54.0, 44.0, -1, BASE_T + 1) ^ box(-200, 200, CHEEK_N[0] - 4, CHEEK_N[1] + 4, -5, 5)
    keep += box(X_AX - 19, X_AX + 19, CHEEK_P[0] - 4, CHEEK_P[1] + 4, -1, BASE_T + 1)
    keep += box(X_AX - 19, X_AX + 19, CHEEK_P[1], 44.0, -1, BASE_T + 1)            # 45° 받침 밑
    keep += box(xa - 8, xa + 4, -54.0, CHEEK_N[1], -1, BASE_T + 1) + box(xb - 4, xb + 8, -54.0, CHEEK_N[1], -1, BASE_T + 1)   # 끝벽 밑
    base = base ^ keep
    c = base + cn + ends + cp + gus
    c -= tilt_servo_place(servo_solid(S, horn=False, clr=B.SERVO_CLR))
    c -= tilt_servo_place(servo_holes(S))
    c -= tear_y(B.AXLE_R + 0.2, CHEEK_P[0] - 1, CHEEK_P[1] + 1, X_AX, H_AX)
    # 바닥판 아래로 내려간 서보 자리(구멍) 정리: 바닥판보다 아래 없음
    return c

if __name__ == "__main__":
    c = build_cradle()
    tm = B.to_t(c)
    print("cradle pieces", len(tm.split()), "watertight", tm.is_watertight, "vol cm3", round(tm.volume / 1000, 1), "ext", np.round(tm.extents, 1))
    tm.export(os.path.join(D, "stl", "roof_cradle.stl"))
    tm2 = tm.copy(); tm2.apply_translation([0, 0, -tm2.bounds[0][2]]); tm2.export(os.path.join(D, "stl", "print_cradle.stl"))

def export_params():
    """시뮬(MuJoCo 환경엔 manifold 없음)이 쓰는 치수만 params.json 으로."""
    keys = ["BORE_R", "X_REAR", "X_MUZ", "HOP_X", "HOP_CX", "HOP_TOP", "PLUNGER_L", "PLUNGER_R", "PLUNGER_CHAMF", "RAMP", "DRAW",
            "PIN_REST", "PIN_COCK", "PIN_OVER", "LEVER_COCK_DEG", "LEVER_TRIP_DEG", "AX", "N_HOPPER",
            "SLOT_X0", "SLOT_X1", "SLOT_W", "PIN_R", "PIN_Y", "BAND_Y", "LEVER_PIV", "LEVER_Y", "LEVER_LEN", "LEVER_W",
            "LEVER_REST_DEG", "LUG_C", "LUG_X0", "LUG_R", "LUG_H", "FINGER_DEG", "FINGER_R", "FINGER_W",
            "SEAR_PIV", "SEAR_Y", "SEAR_YF", "SEAR_GAP", "SEAR_FACE_X", "SEAR_TOP", "SEAR_ARM_Z", "SEAR_TAIL", "SEAR_PEG", "SEAR_POST",
            "SEAR_STOP_X", "SEAR_NOSE", "SEAR_PIN_R"]
    p = {k: getattr(B, k) for k in keys}
    p.update(X_AX=X_AX, H_AX=H_AX, ANG=ANG)
    json.dump(p, open(os.path.join(D, "params.json"), "w", encoding="utf-8"), indent=1)

if __name__ == "__main__":
    export_params()
