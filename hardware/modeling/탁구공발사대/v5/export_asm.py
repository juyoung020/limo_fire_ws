"""v5 조립 그림용 STL(발사대 자기 좌표): 쉴 때 / 장전(걸쇠에 걸림) — stl/asm_<자세>_<부품>.stl"""
import os, math
import build_launcher as B
D = os.path.dirname(os.path.abspath(__file__)); O = os.path.join(D, "stl")
body = B.build_body(); pl = B.build_plunger(); pin = B.build_pin(); lev = B.build_lever(); sear = B.build_sear()
ro, ri, y0, y1 = B.ROLLER
roller = B.build_roller().rotate([-90, 0, 0]).translate([0, y0, 0])          # z(0~L) → y(y0~y1)
servo = B.lever_servo_place(B.servo_solid(B.SERVO_L, horn=True))
def put(x_pin, th):
    dx = x_pin - B.PIN_REST
    return dict(plunger=pl.translate([dx, 0, 0]), pin=pin.translate([x_pin, 0, 0]), roller=roller.translate([x_pin, 0, 0]),
                lever=B.lever_at(th, lev))
for name, (xp, th) in dict(rest=(B.PIN_REST, 0.0), cocked=(B.SEAR_HOLD_X, 0.0)).items():
    parts = dict(body=body, sear=sear, sear_pin=B.sear_pin_at(), servo=servo, **put(xp, th))
    for k, m in parts.items():
        B.to_t(m).export(os.path.join(O, f"asm_{name}_{k}.stl"))
print("ok")
