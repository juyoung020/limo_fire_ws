"""v4 전체 조립 검사(실제 모양 서보·혼 포함). 결과: check_assembly.json
A. 발사대 자기 좌표: 레버(팔·손가락)·걸쇠·핀·플런저·고무줄 움직임 전 범위, 걸쇠가 걸리고 풀리는 각, 조립 길, 나사 길.
B. 지붕 좌표: 각도 5~30° 전 범위에서 발사대 ↔ 받침대·각도 서보·LIMO, 공 통 뚜껑·공 포함.
걸쇠 각 s: 0 = 닫힘(멈춤 턱에 닿음), + = 열림(이빨이 내려감). 레버 각: + = 장전 쪽, - = 발사 쪽."""
import os, json, math
import numpy as np, manifold3d as m3, trimesh
import build_launcher as B, build_cradle as C

D = os.path.dirname(os.path.abspath(__file__))
vol = lambda a: a.volume()
hit = lambda a, b, tol=1e-2: vol(a ^ b) > tol
R = {}

body = B.build_body()
lever0, sear0, pin0, plunger0, spin0 = B.build_lever(), B.build_sear(), B.build_pin(), B.build_plunger(), B.sear_pin_at()
sv_lever = B.lever_servo_place(B.servo_solid(B.SERVO_L, horn=False))
horn_lever = B.lever_servo_place(B.cyl_z(B.SERVO_L["horn_r"], B.SERVO_L["spline_end"] + B.HORN_TOP - B.HORN_T, B.SERVO_L["spline_end"] + B.HORN_TOP, 0, 0, 64))
pin_at = lambda x: pin0.translate([x, 0, 0])
plunger_at = lambda x: plunger0.translate([x - B.PIN_REST, 0, 0])
S_AT = lambda s: B.sear_at(s, sear0)

# ---------- A ----------
R["servo_lever_vs_body_mm3"] = round(vol(sv_lever ^ body), 2)
R["horn_lever_vs_body_mm3"] = round(vol(horn_lever ^ body), 2)
R["horn_lever_vs_lever_hub_mm3(혼 윗면에 앉음 = 0)"] = round(vol(horn_lever ^ B.lever_at(0, lever0)), 2)
R["sear_pin_vs_body_mm3(구멍과 같은 지름)"] = round(vol(spin0 ^ body), 2)
x_min_pin = B.PIN_COCK - B.PIN_OVER
R["pin_travel_hits_body"] = [round(x, 1) for x in np.arange(x_min_pin, B.PIN_REST + 0.01, 1.0) if hit(pin_at(x), body)]
R["pin_travel_hits_horn_servo"] = [round(x, 1) for x in np.arange(x_min_pin, B.PIN_REST + 0.01, 2.5) if hit(pin_at(x), horn_lever + sv_lever + spin0)]
R["plunger_travel_hits_body"] = [round(x, 1) for x in np.arange(x_min_pin, B.PIN_REST + 0.01, 2.5) if hit(plunger_at(x), body)]
R["plunger_insert_path_hits(dx)"] = [round(d, 1) for d in np.arange(0, B.PLUNGER_L + B.PIN_REST - B.X_REAR + 2, 4.0) if hit(plunger0.translate([-d, 0, 0]), body)]
R["plunger_rest_gap_to_slot_front(mm3 at +0.2)"] = round(vol(plunger0.translate([0.2, 0, 0]) ^ body), 2)
# 레버: 몸통·서보·걸쇠 핀과 안 부딪힘(발사 -4° ~ 장전 +8°), 핀을 미는 표
track, lev_hits, x = [], [], B.PIN_REST
for th in np.arange(B.LEVER_TRIP_DEG - 4.0, B.LEVER_COCK_DEG + 8.01, 0.5):
    L = B.lever_at(th, lever0)
    if hit(L, body): lev_hits.append(f"body{th:.1f}")
    if hit(L, sv_lever + spin0): lev_hits.append(f"servo/searpin{th:.1f}")
    if th >= 0:
        while hit(L, pin_at(x)) and x > -100: x -= 0.05
        track.append((float(th), round(x, 2)))
R["lever_hits"] = lev_hits
R["lever_track_every5"] = track[::10]
R["lever_angle_to_cock(-80)"] = next((th for th, xx in track if xx <= B.PIN_COCK), None)
R["lever_first_touch_deg"] = next((th for th, xx in track if xx < B.PIN_REST - 0.01), None)
R["pin_x_at_cmd_angle"] = dict(track).get(B.LEVER_COCK_DEG)
R["rest_lever_blocks_pin_path"] = [round(xx, 1) for xx in np.arange(x_min_pin, B.PIN_REST + 0.01, 2.5) if hit(B.lever_at(0, lever0), pin_at(xx))]
R["trip_lever_blocks_pin_path"] = [round(xx, 1) for xx in np.arange(x_min_pin, B.PIN_REST + 0.01, 2.5) if hit(B.lever_at(B.LEVER_TRIP_DEG, lever0), pin_at(xx))]

# 걸쇠
def s_needed(xp):
    """핀이 xp 에 있을 때 걸쇠가 안 닿으려면 열려야 하는 각."""
    for s in np.arange(0, 40, 0.1):
        if not hit(S_AT(s), pin_at(xp)): return round(float(s), 1)
    return None
cam = {f"{xx:.1f}": s_needed(xx) for xx in np.arange(x_min_pin, B.PIN_REST + 0.01, 1.0)}
R["sear_open_needed_vs_pin_x"] = {k: v for k, v in cam.items() if v}
s_clear = max(v for v in cam.values() if v is not None)
hold_x = B.SEAR_FACE_X - B.PIN_R
R["sear_open_to_release_deg(핀이 걸림면 앞으로 지나가는 데 필요한 최대)"] = max(s_needed(xx) for xx in np.arange(hold_x, B.SEAR_FACE_X + 8, 0.25))
R["sear_closed_holds_pin(pin +0.3 forward of hold hits)"] = hit(S_AT(0), pin_at(hold_x + 0.3))
R["sear_hold_pin_x"] = round(hold_x, 2)
R["sear_closed_free_at_cock_positions"] = {f"{xx}": not hit(S_AT(0), pin_at(xx)) for xx in (B.PIN_COCK, x_min_pin, hold_x - 0.05)}
R["sear_sweep_hits_body_deg"] = [s for s in np.arange(0, 35.1, 0.5) if hit(S_AT(s), body + spin0)]
R["sear_closed_gap_to_stop(mm3 at -0.3 deg)"] = round(vol(S_AT(-0.3) ^ body), 2)
# 장전 중: 레버 각마다 핀 위치 → 걸쇠가 그만큼 열려 있을 때 레버(팔·손가락)와 걸쇠가 안 닿음
cock_conf = []
for th, xx in track[::2]:
    s = s_needed(xx) or 0.0
    if hit(B.lever_at(th, lever0), S_AT(s)): cock_conf.append((th, xx, s))
R["lever_vs_sear_during_cock(th, pin_x, sear_s)"] = cock_conf
# 발사: 레버를 앞으로 돌릴 때 손가락이 꼬리를 밀어 걸쇠가 열리는 각
tt, s = [], 0.0
for th in np.arange(0, B.LEVER_TRIP_DEG - 4.01, -0.25):
    L = B.lever_at(th, lever0)
    while hit(L, S_AT(s)) and s < 45: s += 0.05
    tt.append((round(float(th), 2), round(s, 2)))
R["trip_track_every1deg(lever, sear)"] = tt[::4]
R["trip_first_touch_lever_deg"] = next(th for th, ss in tt if ss > 0)
R["trip_release_lever_deg"] = next(th for th, ss in tt if ss >= R["sear_open_to_release_deg(핀이 걸림면 앞으로 지나가는 데 필요한 최대)"])
s_trip = dict(tt)[B.LEVER_TRIP_DEG]; R["sear_s_at_trip_cmd"] = s_trip
R["sear_at_trip_blocks_pin_path"] = [round(xx, 1) for xx in np.arange(x_min_pin, B.PIN_REST + 0.01, 0.5) if hit(S_AT(s_trip), pin_at(xx))]
R["sear_at_trip_hits_body"] = hit(S_AT(s_trip), body + spin0)
R["lever_finger_vs_sear_at_rest_gap_ok"] = not hit(B.lever_at(0, lever0), S_AT(0))
# 고무줄(가닥 = 지름 1.4 원기둥 두 줄)
def band(xpin, hx, side, hz=0.0):
    out = m3.Manifold(); yb = side * sum(B.BAND_Y) / 2
    for zz in (-2.6, 2.6):                     # 핀 홈(z 0) → 고리(z hz) 두 가닥
        out += m3.Manifold.hull(B.cyl_y(0.7, yb - 0.1, yb + 0.1, xpin + 3.0, zz, 12) + B.cyl_y(0.7, yb - 0.1, yb + 0.1, hx - 3.0, hz + zz, 12))
    return out
bh = {}
for hx, hz in zip(B.HOOK_XS, B.HOOK_ZS):
    for xp in (B.PIN_REST, hold_x, B.PIN_COCK, x_min_pin):
        for side in (-1, 1):
            b = band(xp, hx, side, hz); h = []
            if hit(b, body - B.box(hx - 5, hx + 5, -45, 45, hz - 6, hz + 6), 0.05): h.append("body")
            for s in (0, s_trip):
                if hit(b, S_AT(s), 0.05): h.append(f"sear{s}")
            for th in (B.LEVER_TRIP_DEG, 0, 30, B.LEVER_COCK_DEG):
                if hit(b, B.lever_at(th, lever0), 0.05): h.append(f"lever{th}")
            bh[f"hook{hx}_pin{xp}_y{side}"] = h
R["band_hits"] = bh
# 걸쇠 고무줄(갈고리 ↔ 기둥, 두 가닥, y 22.3)
gx, gz = B.SEAR_PEG; px, pz = B.SEAR_POST
def sear_band(s):
    sx, sz = B.SEAR_PIV; a = math.radians(s)
    qx, qz = gx - sx, gz - sz
    g = (sx + qx * math.cos(a) + qz * math.sin(a), sz - qx * math.sin(a) + qz * math.cos(a))
    out = m3.Manifold()
    for dz in (-2.9, 2.9):
        out += m3.Manifold.hull(B.cyl_y(0.7, 21.6, 23.0, g[0] + 3.5, g[1] + dz, 12) + B.cyl_y(0.7, 21.6, 23.0, px - 3.5, pz + dz, 12))
    return out, g
sbh = {}
for s in (0, s_clear, s_trip):
    b, g = sear_band(s); h = []
    if hit(b, body, 0.05): h.append("body")
    if hit(b, S_AT(s), 0.05): h.append("sear")
    for th in (B.LEVER_TRIP_DEG, 0, 30, B.LEVER_COCK_DEG):
        if hit(b, B.lever_at(th, lever0), 0.05): h.append(f"lever{th}")
    for xx in (B.PIN_REST, hold_x, x_min_pin):
        if hit(b, plunger_at(xx) + pin_at(xx), 0.05): h.append(f"pin{xx}")
    sbh[f"s{s}"] = dict(hits=h, span_mm=round(math.hypot(px - g[0], pz - g[1]), 2),
                       arm_mm=round(abs((px - g[0]) * (B.SEAR_PIV[1] - g[1]) - (pz - g[1]) * (B.SEAR_PIV[0] - g[0])) / math.hypot(px - g[0], pz - g[1]), 2))
R["sear_band"] = sbh
# 조립 길
asm_lever = sv_lever + horn_lever + B.lever_at(B.LEVER_INSERT_DEG, lever0)
R["lever_servo_insert_path_hits(dz, 걸쇠 닫힘 있음)"] = [d for d in range(0, 81, 3) if hit(asm_lever.translate([0, 0, d]), body + S_AT(0))]
# 걸쇠 넣는 길(레버 끼우기 전, 축 핀 없이): 총신과 벽 사이로 아래에서 올려(앞으로 40 비킨 자리) → 뒤로 40 밀어 축 자리에
R["sear_insert_path_hits(1: 아래에서 올림 dz, 앞 40)"] = [d for d in range(0, 121, 3) if hit(S_AT(0).translate([40, 0, -d]), body)]
R["sear_insert_path_hits(2: 뒤로 밂 dx)"] = [d for d in range(0, 41, 2) if hit(S_AT(0).translate([d, 0, 0]), body)]
R["sear_pin_insert_path_hits(dy, 레버 서보 있음)"] = [d for d in range(2, 41, 2) if hit(spin0.translate([0, d, 0]), body + sv_lever + horn_lever + S_AT(0))]
R["sear_pin_vs_sear_hole_mm3"] = round(vol(spin0 ^ S_AT(0)), 2)
R["pin_insert_path_hits(-dy, 플런저 쉴 때)"] = [d for d in range(0, 91, 5) if hit(pin_at(B.PIN_REST).translate([0, -d, 0]), body + S_AT(0) + B.lever_at(0, lever0))]

# ---------- B ----------
cradle = C.build_cradle()
tilt_sv = C.tilt_servo_place(B.servo_solid(B.SERVO_K, horn=False))
tilt_horn_local = B.cyl_y(B.SERVO_K["horn_r"], -B.KEEL_Y - B.HORN_T, -B.KEEL_Y, B.AX[0], B.AX[1], 64)   # 몸통에 붙은 혼
axle_local = B.cyl_y(B.AXLE_R, B.KEEL_Y - 15 + 3, B.KEEL_Y + 5 + 3, B.AX[0], B.AX[1], 48)
cap_local = B.build_hopper_cap().translate([B.HOP_CX, 0, B.HOP_TOP - 8.0])
balls_local = m3.Manifold()
for i in range(B.N_HOPPER):
    balls_local += m3.Manifold.sphere(20.0, 48).translate([B.HOP_CX, 0, B.BORE_R + 0.5 + 20 + 40.4 * i])
balls_local += m3.Manifold.sphere(20.0, 48).translate([B.PIN_COCK - B.PIN_REST + 20 + 0.5, 0, -0.8])
tube_local = B.build_hopper_tube()
moving_static = body + sv_lever + horn_lever + spin0 + tube_local + cap_local + tilt_horn_local
R["hopper_tube_vs_body_mm3"] = round(vol(tube_local ^ body), 2)
R["hopper_tube_insert_path_hits(dz, 레버 서보 있음)"] = [d for d in range(1, 60, 3) if hit(tube_local.translate([0, 0, d]), body + sv_lever + horn_lever)]
R["hopper_tube_seat_gap(mm3 at -0.3 down; >0 = 짧은 통 위에 앉음)"] = round(vol(tube_local.translate([0, 0, -0.3]) ^ body), 2)
R["ball_path_hopper_balls_vs_parts_mm3"] = round(vol(balls_local ^ (body + tube_local + cap_local)), 2)
R["cradle_vs_tilt_servo_mm3"] = round(vol(cradle ^ tilt_sv), 2)
P = np.load(os.path.join(D, "ref", "limo", "limo_pts_roofframe_m.npy")).astype(float) * 1000
G = 2.0
from collections import defaultdict
lt = defaultdict(lambda: -1e9)
for (a, b_, z) in zip(np.floor(P[:, 0] / G).astype(int), np.floor(P[:, 1] / G).astype(int), P[:, 2]):
    if z > lt[(a, b_)]: lt[(a, b_)] = z
def min_gap(mf):
    tm = B.to_t(mf); s, _ = trimesh.sample.sample_surface_even(tm, 40000); p = np.r_[tm.vertices, s]
    best = (1e9, None)
    for (x, y, z) in p:
        k = (int(math.floor(x / G)), int(math.floor(y / G)))
        if k in lt and z - lt[k] < best[0]: best = (z - lt[k], (round(x), round(y)))
    return round(best[0], 1), best[1]
tilt = {}
for th in np.arange(C.ANG[0], C.ANG[1] + 0.01, 2.5):
    ms = C.pose(moving_static, th)
    lev = [C.pose(B.lever_at(a, lever0), th) for a in (0, B.LEVER_COCK_DEG)]
    lev.append(C.pose(B.lever_at(B.LEVER_TRIP_DEG, lever0), th))
    lat = [C.pose(S_AT(a), th) for a in (0, s_trip)]
    allm = ms + lev[0] + lev[1] + lev[2] + lat[0] + lat[1] + C.pose(balls_local, th)
    tilt[f"{th:.1f}"] = dict(
        launcher_vs_cradle_mm3=round(vol(allm ^ cradle), 1),
        launcher_vs_tilt_servo_mm3=round(vol((allm - C.pose(tilt_horn_local, th)) ^ tilt_sv), 1),
        min_gap_to_limo=min_gap(allm),
        front_x=round(B.to_t(allm).bounds[1][0], 1), rear_x=round(B.to_t(allm).bounds[0][0], 1), top_z=round(B.to_t(allm).bounds[1][2], 1))
R["tilt"] = tilt
R["cradle_min_gap_to_limo"] = min_gap(cradle)
R["axle_pin_vs_cradle_and_body_mm3(구멍 안이면 0)"] = [round(vol(C.pose(axle_local, th) ^ cradle), 1) for th in (5, 30)] + [round(vol(axle_local ^ body), 1)]
# ---------- 조립 길·드라이버 길 (15°, 지붕 좌표) ----------
TH = 15.0
everything = C.pose(body + sv_lever + horn_lever + spin0 + B.lever_at(0, lever0) + S_AT(0), TH) + cradle + tilt_sv
axle_real = B.build_axle_pin().translate([B.AX[0], B.KEEL_Y - 15 + 5, B.AX[1]])
R["axle_pin_insert_path_hits(dy, 15deg)"] = [d for d in range(3, 61, 3) if hit(C.pose(axle_real.translate([0, d, 0]), TH), everything)]
drv = B.cyl_y(2.0, -B.KEEL_Y + 2.0, 140.0, B.AX[0], B.AX[1], 24)          # 혼 가운데 나사 머리(-14)부터 +y 로 드라이버 Ø4
R["tilt_horn_screw_driver_hits_mm3"] = round(vol(C.pose(drv, TH) ^ everything), 2)
R["tilt_servo_insert_path_hits(dy)"] = [d for d in range(3, 31, 3) if hit(tilt_sv.translate([0, -d, 0]), C.pose(body + tilt_horn_local, TH) + cradle)]
def flange_driver(S, place_fn):
    hs = m3.Manifold(); uc = S["body_l"] / 2 - S["shaft_off"]
    for du in (-1, 1):
        for dv in (-1, 1):
            hs += B.cyl_z(2.5, -S["flange_t"] - 60, -S["flange_t"], uc + du * S["hole_dl"] / 2, dv * S["hole_dw"] / 2, 24)
    return place_fn(hs)
R["flange_screw_driver_hits_mm3"] = dict(
    lever=round(vol(C.pose(flange_driver(B.SERVO_L, B.lever_servo_place), TH) ^ everything), 2),
    tilt=round(vol(flange_driver(B.SERVO_K, C.tilt_servo_place) ^ everything), 2))
# 레버 혼 나사(허브 가운데·혼 나사 4개)는 서보를 끼우기 전 밖에서 조인다. 걸쇠 축 핀은 벽 밖에서 손으로 꽂는다.
print(json.dumps(R, ensure_ascii=False, indent=1, default=str))
json.dump(R, open(os.path.join(D, "check_assembly.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
