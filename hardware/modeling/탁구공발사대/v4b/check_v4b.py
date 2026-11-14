import numpy as np, manifold3d as m3, json
import build_launcher as B
body = B.build_body(); lev = B.build_lever(); pin = B.build_pin(); pl = B.build_plunger(); sear = B.build_sear()
ro, ri, y0, y1 = B.ROLLER
roller = B.build_roller().rotate([-90, 0, 0]).translate([0, y0, 0])
def hit(a, b, t=0.01): return (a ^ b).volume() > t
pin_at = lambda x: pin.translate([x, 0, 0]); rol_at = lambda x: roller.translate([x, 0, 0]); pl_at = lambda x: pl.translate([x - B.PIN_REST, 0, 0])
R = {}
xs = np.arange(B.PIN_COCK - B.PIN_OVER, B.PIN_REST + 0.01, 1.0)
R["roller_hits_body"] = [round(x, 1) for x in xs if hit(rol_at(x), body)]
R["roller_hits_sear_closed"] = [round(x, 1) for x in xs if hit(rol_at(x), sear)]
R["plunger_hits_body"] = [round(x, 1) for x in xs if hit(pl_at(x), body)]
x = B.PIN_REST; track = []
for th in np.arange(0, B.LEVER_COCK_DEG + 8.01, 0.5):
    L = B.lever_at(th, lev)
    while (hit(L, pin_at(x)) or hit(L, rol_at(x))) and x > -100: x -= 0.05
    track.append((float(th), round(x, 2)))
R["first_touch_deg"] = next((t for t, xx in track if xx < B.PIN_REST - 0.01), None)
R["deg_to_hold(-76.5)"] = next((t for t, xx in track if xx <= B.PIN_COCK + B.SEAR_GAP), None)
R["deg_to_-80"] = next((t for t, xx in track if xx <= B.PIN_COCK), None)
R["pin_at_66"] = dict(track).get(66.0); R["pin_at_69"] = dict(track).get(69.0); R["pin_at_74"] = dict(track).get(74.0)
R["rest_lever_hits_roller_path"] = [round(xx, 1) for xx in xs if hit(B.lever_at(0, lev), rol_at(xx))]
R["trip_lever_hits_roller_path"] = [round(xx, 1) for xx in xs if hit(B.lever_at(B.LEVER_TRIP_DEG, lev), rol_at(xx))]
print(json.dumps(R, ensure_ascii=False))
