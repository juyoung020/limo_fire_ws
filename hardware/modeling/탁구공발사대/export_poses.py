"""v4 조립 자세 STL(지붕 좌표, mm): 각도 θ 별로 몸통·레버 서보·혼·레버·걸쇠·걸쇠 핀·핀·플런저·공·공 통·뚜껑·축 핀 + 받침대·각도 서보."""
import os, shutil
import numpy as np, manifold3d as m3
import build_launcher as B, build_cradle as C
D = os.path.dirname(os.path.abspath(__file__))
def save(mf, path): B.to_t(mf).export(path)
body = B.build_body()
SL, SK = B.SERVO_L, B.SERVO_K
servos = B.lever_servo_place(B.servo_solid(SL))
cap = B.build_hopper_cap().translate([B.HOP_CX, 0, B.HOP_TOP - 8.0]) + B.build_hopper_tube()
spin = B.sear_pin_at()
horn_tilt = B.cyl_y(SK["horn_r"], -B.KEEL_Y - B.HORN_T, -B.KEEL_Y, B.AX[0], B.AX[1], 64)
axle = B.build_axle_pin().translate([B.AX[0], B.KEEL_Y - 15 + 5, B.AX[1]])
def balls(n_hop, barrel_x):
    b = m3.Manifold()
    for i in range(n_hop): b += m3.Manifold.sphere(20.0, 48).translate([B.HOP_CX, 0, B.BORE_R - 0.4 + 20.3 + 40.4 * i])
    if barrel_x is not None: b += m3.Manifold.sphere(20.0, 48).translate([barrel_x, 0, -0.8])
    return b
poses = {"15_rest": (15, 0, 0, B.PIN_REST, balls(4, 20.5)), "15_cock": (15, B.LEVER_COCK_DEG, 0, B.PIN_COCK, balls(3, B.PIN_COCK - B.PIN_REST + 20.5)),
         "15_held": (15, 0, 0, B.SEAR_FACE_X - B.PIN_R, balls(3, B.SEAR_FACE_X - B.PIN_R - B.PIN_REST + 20.5)),
         "15_trip": (15, B.LEVER_TRIP_DEG, 21.3, B.PIN_REST, balls(4, 20.5)),
         "5_rest": (5, 0, 0, B.PIN_REST, balls(4, 20.5)), "30_rest": (30, 0, 0, B.PIN_REST, balls(4, 20.5))}
cradle = C.build_cradle(); tilt_sv = C.tilt_servo_place(B.servo_solid(SK, horn=False))
for name, (th, lev, lat, pin_x, bl) in poses.items():
    o = os.path.join(D, "stl", "pose_" + name); shutil.rmtree(o, ignore_errors=True); os.makedirs(o)
    P = lambda m: C.pose(m, th)
    save(P(body), f"{o}/body.stl"); save(P(servos + horn_tilt), f"{o}/servo.stl")
    save(P(B.lever_at(lev)), f"{o}/lever.stl"); save(P(B.sear_at(lat)), f"{o}/sear.stl"); save(P(spin), f"{o}/sear_pin.stl")
    save(P(B.build_pin().translate([pin_x, 0, 0])), f"{o}/pin.stl")
    save(P(B.build_plunger().translate([pin_x - B.PIN_REST, 0, 0])), f"{o}/plunger.stl")
    save(P(bl), f"{o}/ball.stl"); save(P(cap), f"{o}/cap.stl"); save(P(axle), f"{o}/axle.stl")
    save(cradle, f"{o}/cradle.stl"); save(tilt_sv, f"{o}/tiltservo.stl")
print("ok")
