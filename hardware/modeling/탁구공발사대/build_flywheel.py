"""비교안 B: DC 모터(130) 단일 플라이휠 + MG996R 밀대. 총신·공 통·15° 받침은 A(고무줄)와 같은 치수.

좌표(발사대 자기 좌표): x 발사 방향, y 옆(+y 밀대 서보, -y 모터), z 총신에 수직 위. 단위 mm.
동작
 1. 밀대가 뒤(면 x=-50)에 있으면 공 통에서 공이 떨어져 밀대 면에 기댄다(15° 기울기).
 2. MG996R 포크 팔이 가로대를 잡고 밀대를 28 mm 앞으로 → 공이 바퀴에 닿는다.
 3. 바퀴(Ø42, 고무줄 테)가 공을 천장(후드)에 눌러 굴리며 쏜다. 공은 바퀴 테 속도의 약 절반, 역회전.
 4. 밀대가 뒤로 → 다음 공이 떨어짐. 앞에 있는 동안은 밀대 몸이 다음 공을 막는다.
130 모터 치수: 몸통 25.1 x 20.1 x 15.1, 축 Ø2 x 8.2 (부싱 Ø6.15 x 1.6) — 마부치 FA-130RA 데이터시트.
"""
import os, json, math
import numpy as np, manifold3d as m3
import build_launcher as A
from build_launcher import box, cyl_x, cyl_y, cyl_z, to_t, horn_mount, SERVO_K, BORE_R, BARREL_R, WALL, HORN_T

D = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(D, "stl"); os.makedirs(OUT, exist_ok=True)
PITCH = A.PITCH
X_REAR, X_MUZ = -66.0, 58.0
HOP_CX, HOP_TOP = -25.0, 112.0              # A 와 같음(구멍 -46 ~ -4)
FACE_BACK, FACE_FWD = -50.0, -22.0          # 밀대 면
PUSH_L = 22.0                               # 밀대 길이(앞에 있을 때 공 통 구멍 뒤쪽을 막음)
PIN_OFF = -10.0                             # 면에서 가로대까지
PIN_R, SLOT_W = 3.0, 6.6
PIN_BACK, PIN_FWD = FACE_BACK + PIN_OFF, FACE_FWD + PIN_OFF
LEVER_Y = (33.5, 38.5)
PIV = ((PIN_BACK + PIN_FWD) / 2, -16.0)    # 서보 축 (x, z), 축 방향 y
FORK_T = 5.0
WHEEL_X, WHEEL_Z, WHEEL_R, WHEEL_W = 12.0, -39.0, 21.0, 14.0   # Ø42 = 출력 Ø40 + 고무줄 1 mm
HOOD_Z = 20.0                               # 바퀴 위 천장(공 2 mm 눌림: 바퀴 꼭대기 -18 ~ 천장 20 = 38)
MOTOR = dict(L=25.1, W=20.1, H=15.1, shaft_l=8.2, boss_d=6.15, boss_h=1.6)
MOTOR_FACE_Y = -10.5                        # 모터 앞면(부싱 끝) y
FOOT_Z = (-70.0, -65.0)

def servo_plate(fl_y, piv):
    """+y 쪽 MG996R 받침판. 축 방향 -y. 몸통은 바깥."""
    s = SERVO_K; cx = piv[0] - s["shaft_off"]; cz = piv[1]
    plate = box(cx - s["flange_l"] / 2 - 3, cx + s["flange_l"] / 2 + 3, fl_y - 4, fl_y, FOOT_Z[0], cz + s["body_w"] / 2 + 4)
    plate -= box(cx - s["body_l"] / 2, cx + s["body_l"] / 2, fl_y - 5, fl_y + 1, cz - s["body_w"] / 2, cz + s["body_w"] / 2)
    for dx in (-1, 1):
        for dz in (-1, 1):
            plate -= cyl_y(s["hole_d"] / 2, fl_y - 5, fl_y + 1, cx + dx * s["hole_dl"] / 2, cz + dz * s["hole_dw"] / 2, 24)
    return plate, cx

def build_body():
    hop = cyl_z(BARREL_R, 0, HOP_TOP, HOP_CX)
    tube = cyl_x(BARREL_R, X_REAR, X_MUZ)
    foot = box(X_REAR, X_MUZ, -BARREL_R - 22, 52, FOOT_Z[0], FOOT_Z[1])
    legs = box(X_REAR, X_REAR + 12, -BARREL_R, BARREL_R, FOOT_Z[1] - 0.1, 0) + box(40, 52, -BARREL_R, BARREL_R, FOOT_Z[1] - 0.1, 0)
    side = box(X_REAR, X_MUZ, BARREL_R - 3, BARREL_R, FOOT_Z[1] - 0.1, 0)          # +y 옆벽
    # 모터 집: 모터 축 y, 몸통 -y 쪽. 앞판에 부싱 구멍
    mx0, mx1 = WHEEL_X - MOTOR["H"] / 2, WHEEL_X + MOTOR["H"] / 2
    mz0, mz1 = WHEEL_Z - MOTOR["W"] / 2, WHEEL_Z + MOTOR["W"] / 2
    my1 = MOTOR_FACE_Y - MOTOR["boss_h"]; my0 = my1 - MOTOR["L"]
    house = box(mx0 - 2.4, mx1 + 2.4, my0 + 6, my1 + 2.4, FOOT_Z[1] - 0.1, mz1 + 2.4)
    fl_y = LEVER_Y[1] + HORN_T + SERVO_K["top_h"]
    plate, servo_cx = servo_plate(fl_y, PIV)
    hood = box(WHEEL_X - 12, WHEEL_X + 20, -14, 14, HOOD_Z, BARREL_R) ^ cyl_x(BORE_R + 0.1, X_REAR, X_MUZ)   # 공 통 구멍(~-4) 앞에서 시작: 겹치면 떨어지는 공이 턱에 걸림
    for (a, b) in ((-54.0, -8.0), (24.0, 40.0)):        # 발판 창(모터 집·다리·서보판 피함)
        for (c, d) in ((-38.0, -10.0), (10.0, 40.0)):
            foot -= box(a, b, c, d, FOOT_Z[0] - 1, FOOT_Z[1] + 1)
    body = tube + hop + foot + legs + side + house + plate
    body -= cyl_x(BORE_R, X_REAR - 1, X_MUZ + 1)
    body += hood
    body -= cyl_z(BORE_R, 0, HOP_TOP + 1, HOP_CX)
    for s in (-1, 1):
        y0, y1 = sorted((s * (BORE_R - 1), s * (BARREL_R + 1)))
        body -= box(PIN_BACK - SLOT_W / 2, PIN_FWD + SLOT_W / 2, y0, y1, -SLOT_W / 2, SLOT_W / 2)
    # 바퀴 자리(총신 바닥 틈 + 아래 공간), 모터 자리(-y 로 열린 주머니 + 부싱 구멍)
    body -= box(WHEEL_X - WHEEL_R - 2, WHEEL_X + WHEEL_R + 2, -WHEEL_W / 2 - 1.5, WHEEL_W / 2 + 1.5, WHEEL_Z - WHEEL_R - 2, -10)
    body -= box(mx0 - 0.15, mx1 + 0.15, my0 - 1, my1, mz0 - 0.15, mz1 + 0.15)
    body -= cyl_y(MOTOR["boss_d"] / 2 + 0.3, my1 - 1, MOTOR_FACE_Y + 3, WHEEL_X, WHEEL_Z, 32)
    return body, dict(flange_y=fl_y, servo_cx=servo_cx, motor=(mx0, mx1, my0, my1, mz0, mz1))

def build_pusher():
    p = cyl_x(BORE_R - 0.4, -PUSH_L, 0) - cyl_x(BORE_R - 1.8, -PUSH_L - 1, -3)
    p += cyl_y(5.5, -BORE_R + 0.9, BORE_R - 0.9, PIN_OFF, 0) ^ cyl_x(BORE_R - 0.6, -PUSH_L, 0)
    return p - cyl_y(PIN_R + 0.2, -30, 30, PIN_OFF, 0, 32)

def build_pin():
    return cyl_y(PIN_R, -LEVER_Y[1] - 1.5, LEVER_Y[1] + 1.5, 0, 0, 48)

def build_fork():
    """부품 좌표: 축 원점, 두께 z, 팔은 +x 방향(조립 0° = 앞으로 누움). 가로대가 지나는 홈 r 11~27."""
    hub = horn_mount(12.0, FORK_T)
    arm = box(0, 32, -6, 6, 0, FORK_T) + cyl_z(6, 0, FORK_T, 32, 0, 32)
    arm -= box(11, 27, -(PIN_R + 0.25), PIN_R + 0.25, -1, FORK_T + 1) + cyl_z(PIN_R + 0.25, -1, FORK_T + 1, 11, 0, 24) + cyl_z(PIN_R + 0.25, -1, FORK_T + 1, 27, 0, 24)
    return (arm - cyl_z(3.5, -1, FORK_T + 1)) + hub

def build_wheel():
    """부품 좌표: 축 z. 출력 Ø40 꽉 찬 원판(무거울수록 쏠 때 속도가 덜 떨어짐), 테 홈(고무줄) 폭 8, 축 구멍 Ø1.9 억지끼움(모터 축 Ø2), 허브는 -z 로 2.5 더 나옴."""
    r = WHEEL_R - 1.0
    w = cyl_z(r, 0, WHEEL_W, 0, 0, 128)
    w -= cyl_z(r + 1, WHEEL_W / 2 - 4, WHEEL_W / 2 + 4, 0, 0, 128) - cyl_z(r - 0.8, WHEEL_W / 2 - 4, WHEEL_W / 2 + 4, 0, 0, 128)
    w += cyl_z(4, -2.5, 0, 0, 0, 32)
    return w - cyl_z(0.95, -3, WHEEL_W - 3, 0, 0, 24)

def to_world(mf): return mf.rotate([0, -PITCH, 0])

def build_wedge(body_w):
    bb = body_w.bounding_box(); zmin = bb[2] - 5
    Y0, Y1 = -BARREL_R - 22, 52
    xs = (bb[0] - 5, bb[3] + 5)
    under = to_world(box(X_REAR, X_MUZ, Y0, Y1, -300, FOOT_Z[0]))
    region = box(*xs, Y0, Y1, zmin, 400) ^ under
    top = region - to_world(box(X_REAR - 50, X_MUZ + 50, Y0 - 1, Y1 + 1, -300, FOOT_Z[0] - 4))
    bottom = region ^ box(*xs, Y0, Y1, zmin, zmin + 4)
    ribs = m3.Manifold()
    for (a, b) in ((Y0, Y0 + 6), (-6, 6), (Y1 - 6, Y1)):
        ribs += region ^ box(*xs, a, b, zmin, 400)
    win = m3.Manifold()                       # 바닥판·윗판 창(필라멘트 절약)
    for (a, b) in ((-56.0, -8.0), (8.0, 46.0)):
        for (c, d) in ((-38.0, -10.0), (10.0, 44.0)):
            win += box(a, b, c, d, -10, 10)
    top -= to_world(win.translate([0, 0, FOOT_Z[0] - 5]))
    bottom -= win.translate([0, 0, zmin])
    return top + bottom + ribs, zmin

if __name__ == "__main__":
    body, info = build_body()
    pegs = [(-30.0, 0.0), (30.0, 0.0)]            # 받침 가운데 갈비 위
    for (px, py) in pegs:
        body -= cyl_z(4.2, FOOT_Z[0] - 1, FOOT_Z[1] + 1, px, py, 32)
    body_w = to_world(body)
    wedge, _ = build_wedge(body_w)
    for (px, py) in pegs:
        wedge += to_world(cyl_z(4.0, FOOT_Z[0] - 3, FOOT_Z[1] - 1, px, py, 32))
    zshift = -wedge.bounding_box()[2]
    parts = dict(body=body, pusher=build_pusher(), pin=build_pin(), fork=build_fork(), wheel=build_wheel())
    meta = {}
    asm = dict(body=body_w, wedge=wedge,
               pusher=to_world(parts["pusher"].translate([FACE_BACK, 0, 0])),
               pin=to_world(parts["pin"].translate([PIN_BACK, 0, 0])),
               wheel=to_world(parts["wheel"].rotate([90, 0, 0]).translate([WHEEL_X, WHEEL_W / 2, WHEEL_Z])))
    for n, mf in asm.items():
        tm = to_t(mf.translate([0, 0, zshift])); tm.export(os.path.join(OUT, f"B_asm_{n}.stl"))
    for n, mf in list(parts.items()) + [("wedge", wedge)]:
        tm = to_t(mf); tm.apply_translation([0, 0, -tm.bounds[0][2]]); tm.export(os.path.join(OUT, f"B_print_{n}.stl"))
        meta[n] = dict(pieces=len(tm.split()), watertight=bool(tm.is_watertight), vol_cm3=round(tm.volume / 1000, 2),
                       ext=[round(v, 1) for v in tm.extents])
    meta["zshift"] = zshift; meta.update(info)
    json.dump(meta, open(os.path.join(D, "build_meta_B.json"), "w"), indent=1, ensure_ascii=False, default=float)
    for k, v in meta.items(): print(k, v)
