"""B(플라이휠) 밀대 뒤·앞 두 자세 STL(서보·모터 몸통은 상자) — 그림 점검용."""
import os, json, math, shutil, manifold3d as m3
import build_flywheel as F, check_motion_B as C
D = os.path.dirname(os.path.abspath(__file__)); z = json.load(open(os.path.join(D, "build_meta_B.json"), encoding="utf-8"))["zshift"]
save = lambda mf, p: F.to_t(F.to_world(mf).translate([0, 0, z])).export(p)
for pose, (face, ballx) in {"back": (F.FACE_BACK, -30.0), "fwd": (F.FACE_FWD, -2.0)}.items():
    o = os.path.join(D, "stl", "B_pose_" + pose); os.makedirs(o, exist_ok=True)
    for f in os.listdir(o): os.remove(os.path.join(o, f))
    pin_x = face + F.PIN_OFF
    th = -math.degrees(math.atan2(0 - F.PIV[1], pin_x - F.PIV[0]))
    save(C.body, f"{o}/body.stl"); save(C.pusher_at(face), f"{o}/plunger.stl"); save(C.pin_at(pin_x), f"{o}/pin.stl")
    save(C.fork_at(th), f"{o}/lever.stl"); save(C.wheel, f"{o}/latch.stl"); save(C.servo + C.motor, f"{o}/servo.stl")
    save(m3.Manifold.sphere(20, 64).translate([ballx, 0, -0.8]), f"{o}/ball.stl")
    shutil.copy(os.path.join(D, "stl", "B_asm_wedge.stl"), f"{o}/wedge.stl")
print("ok")
