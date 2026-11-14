"""B(플라이휠) 움직임 점검: 포크 팔 회전·가로대·밀대·바퀴·모터·서보 몸통이 본체와 부딪히는지,
포크가 가로대를 뒤(-60)~앞(-32) 전 구간에서 잡는지, 공이 공 통 → 바퀴 → 총구까지 걸리지 않는지."""
import json, os, math
import numpy as np, manifold3d as m3
import build_flywheel as F
from build_launcher import box, cyl_y, cyl_z, SERVO_K

body, info = F.build_body()
pin0, fork0, pusher0, wheel0 = F.build_pin(), F.build_fork(), F.build_pusher(), F.build_wheel()
vol = lambda a: a.volume()
hit = lambda a, b: vol(a ^ b) > 1e-3

def fork_at(th): return fork0.rotate([90, 0, 0]).rotate([0, th, 0]).translate([F.PIV[0], F.LEVER_Y[1], F.PIV[1]])
def pin_at(x): return pin0.translate([x, 0, 0])
def pusher_at(face): return pusher0.translate([face, 0, 0])
wheel = wheel0.rotate([90, 0, 0]).translate([F.WHEEL_X, F.WHEEL_W / 2, F.WHEEL_Z])
wheel_env = cyl_y(F.WHEEL_R + 0.0, -F.WHEEL_W / 2, F.WHEEL_W / 2, F.WHEEL_X, F.WHEEL_Z, 128)   # 고무줄 포함 Ø42, 돌아도 같은 모양
mx0, mx1, my0, my1, mz0, mz1 = info["motor"]
motor = box(mx0, mx1, my0, my1, mz0, mz1) + cyl_y(F.MOTOR["boss_d"] / 2, my1, F.MOTOR_FACE_Y, F.WHEEL_X, F.WHEEL_Z, 32) \
        + cyl_y(1.0, F.MOTOR_FACE_Y, F.MOTOR_FACE_Y + F.MOTOR["shaft_l"], F.WHEEL_X, F.WHEEL_Z, 16)
S = SERVO_K; fl = info["flange_y"]; cx = info["servo_cx"]; cz = F.PIV[1]
servo = box(cx - S["body_l"] / 2, cx + S["body_l"] / 2, fl - S["top_h"] + 4, fl, cz - S["body_w"] / 2, cz + S["body_w"] / 2) + \
        box(cx - S["body_l"] / 2, cx + S["body_l"] / 2, fl + S["flange_t"], fl + S["flange_t"] + S["below_h"], cz - S["body_w"] / 2, cz + S["body_w"] / 2)

r = {}
r["motor_vs_body_mm3"] = round(vol(motor ^ body), 2)
r["shaft_in_wheel_mm"] = round(vol(cyl_y(1.0, F.MOTOR_FACE_Y, F.MOTOR_FACE_Y + F.MOTOR["shaft_l"], F.WHEEL_X, F.WHEEL_Z, 16) ^ cyl_y(1.0, -F.WHEEL_W / 2 - 2.5, F.WHEEL_W / 2, F.WHEEL_X, F.WHEEL_Z, 16)) / (math.pi * 1.0 ** 2), 1)
r["wheel_vs_body_mm3"] = round(vol(wheel_env ^ body), 2)
r["wheel_vs_motor_mm3"] = round(vol(wheel ^ (motor - cyl_y(1.0, -20, 20, F.WHEEL_X, F.WHEEL_Z, 16))), 2)
r["servo_vs_body_mm3"] = round(vol(servo ^ body), 2)
r["wheel_top_z"] = F.WHEEL_Z + F.WHEEL_R
r["nip_gap_mm(hood-wheel_top)"] = round(F.HOOD_Z - (F.WHEEL_Z + F.WHEEL_R), 2)
r["pin_vs_body"] = [x for x in np.arange(F.PIN_BACK, F.PIN_FWD + 0.1, 2) if hit(pin_at(x), body)]
r["pusher_vs_body"] = [f for f in np.arange(F.FACE_BACK, F.FACE_FWD + 0.1, 2) if hit(pusher_at(f), body)]
# 포크: 가로대 위치 x 에서 포크 각도 = 가로대가 홈 안에 있게(홈 중심선이 가로대 지남)
track, fork_hits = [], []
for x in np.arange(F.PIN_BACK, F.PIN_FWD + 0.1, 2):
    dx, dz = x - F.PIV[0], 0 - F.PIV[1]
    th = -math.degrees(math.atan2(dz, dx))          # 팔(+x)을 위로 돌리는 각(음수)
    fk = fork_at(th)
    track.append((round(float(x), 1), round(th, 1), round(math.hypot(dx, dz), 1), hit(fk, pin_at(x))))
    if hit(fk, body): fork_hits.append(round(float(x), 1))
    if hit(fk, servo): fork_hits.append(f"servo@{x:.0f}")
r["fork_track(pin_x, angle, r, pin_hits_fork)"] = track
r["fork_hits"] = fork_hits
# 공 길: 바닥에 놓인 공(z=-0.8)이 공 통 아래 ~ 바퀴 앞까지, 바퀴 위(들림 +2.8), 총구까지
ball = lambda x, z: m3.Manifold.sphere(20.0, 64).translate([x, 0, z])
path = []
for x in np.arange(-30, F.X_MUZ + 25, 4):
    z = -0.8
    dxw = x - F.WHEEL_X
    if abs(dxw) < 41: z = max(z, F.WHEEL_Z + math.sqrt(max(0, 41 ** 2 - dxw ** 2)))     # 바퀴에 얹힘
    z_free = min(z, F.HOOD_Z - 20)                                                       # 천장 아래로 눌림
    squeeze = round(z - z_free, 2)
    path.append((round(float(x)), round(z, 1), squeeze, hit(ball(x, z_free), body)))
r["ball_path(x, z, squeeze_mm, hits_body)"] = path
# 공 통에서 떨어지기: 밀대 뒤일 때 공 통 아래로
r["hopper_drop_clear"] = not hit(ball(F.HOP_CX, 40.0), body) and not hit(ball(F.HOP_CX - 5, -0.8), body)
print(json.dumps(r, ensure_ascii=False))
json.dump(r, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_motion_B.json"), "w"), indent=1, ensure_ascii=False)
