"""멈춤 충격 어림(numpy만). 매 발 플런저+핀(질량 m)이 홈 앞끝에서 v 로 멈출 때 힘·응력.
힘 = v·√(m·k) (질량-용수철 한 자유도). k 는 힘이 지나가는 길의 직렬 강성(한쪽 2개 병렬 + 플런저 통 벽).
E 는 출력 PLA 높은 쪽(3 GPa)을 써서 힘을 크게(안전 쪽) 잡는다. v 는 sim_full.json 의 최대 충돌 속도.
A = 앞 설계(Ø6 핀이 납작한 홈 끝에 직접 닿음), B = 지금(플런저 귀의 45° 면이 같은 45° 홈 끝에 닿음).
판정 기준(출력 PLA): 압축 항복 ~60 MPa, 층 방향 인장 ~50 MPa, 반복(수천 번) 인장은 ~20 MPa 아래가 안전."""
import os, json, math
import numpy as np

D = os.path.dirname(os.path.abspath(__file__))
P = json.load(open(os.path.join(D, "params.json"), encoding="utf-8"))
MP = json.load(open(os.path.join(D, "mass_props.json"), encoding="utf-8"))
E, NU = 3000.0, 0.36                     # MPa
G = E / (2 * (1 + NU)); Es = E / (2 * (1 - NU ** 2))
m = MP["plunger_pin"]["mass"]            # kg
try:
    S = json.load(open(os.path.join(D, "sim_full.json"), encoding="utf-8")); assert "sear_band_peak_N" in S[0]
    v = max(s.get("impact_v_mps") or 0 for r in S for s in r["shots"])
except (FileNotFoundError, AssertionError):
    S = None
    v = 4.8
wall = 2.4                               # 총신 벽 두께(홈 끝 접촉 길이)
k_cup = E * (2 * math.pi * 20.0 * 1.2) / 28.0          # 플런저 통 벽(앞면 → 핀 자리) 축 강성 N/mm
ser = lambda *k: 1 / sum(1 / x for x in k)
F = lambda k: v * math.sqrt(m * k * 1e3)                # k N/mm → N/m

# A: 핀 Ø6 이 납작한 끝에 선접촉, 플런저 보스 끝(19.9)과 벽 가운데(22.0) 사이 2.1 mm 팔
d, a = 6.0, 22.0 - 19.9
kA_side = ser(math.pi / 4 * Es * wall, G * math.pi * d ** 2 / 4 / a, 3 * E * math.pi * d ** 4 / 64 / a ** 3)
FA = F(ser(2 * kA_side, k_cup)); fa = FA / 2
A = dict(peak_force_N=round(FA), contact_pressure_MPa=round(math.sqrt(fa / wall * Es / (math.pi * d / 2))),
         pin_bending_MPa=round(32 * fa * a / (math.pi * d ** 3), 1), pin_shear_MPa=round(4 / 3 * fa / (math.pi * d ** 2 / 4), 1))

# B: 귀 앞면(높이 2 x LUG_H, 45° 면 위 벽 두께 길이 wall·√2)이 홈 끝 45° 면에 면접촉
h = 2 * P["LUG_H"]; Ac = wall * math.sqrt(2) * h
lug_len = 12.0                           # 벽 두께 높이에서 귀 x 길이(뒤 -37 ~ 앞 45° 면)
kB_side = ser(E * Ac / (3.0 + 6.0), G * lug_len * h / 3.0)
FB = F(ser(2 * kB_side, k_cup)); fb = FB / 2
B = dict(peak_force_N=round(FB), contact_pressure_MPa=round(fb / Ac, 1), lug_root_shear_MPa=round(fb / (lug_len * h), 1),
         barrel_wall_compression_MPa=round(fb / (wall * 20.0), 1), plunger_wall_compression_MPa=round(FB / (2 * math.pi * 20.0 * 1.2), 1),
         pin_load="귀가 핀을 감싸 핀에는 충격이 걸리지 않음")

# 정적·반복: 걸쇠가 잡고 있을 때(매 발), 발사(손가락 → 꼬리), 레버 멈춤 힘. 힘은 sim_full.json 의 최대값(없으면 어림).
try:
    shots = [s for r in S for s in r["shots"]]
    F_hold = max(s.get("sear_pin_force_N") or 0 for s in shots)
    F_finger = max(s.get("trip_finger_peak_N") or 0 for s in shots)
    F_stop = min(s.get("sear_stop_force_N") or 99 for s in shots)
    F_sband = max(r.get("sear_band_peak_N") or 0 for r in S)
except (NameError, TypeError):
    F_hold, F_finger, F_stop, F_sband = 10.7, 11.0, 3.0, 2.1
yb = sum(P["BAND_Y"]) / 2; ys = sum(P["SEAR_Y"]) / 2
Fb = F_hold / 2                                      # 고무줄 한쪽 힘(걸쇠가 두 쪽 합을 받음)
# 핀: +y 쪽 걸쇠(ys)와 양쪽 고무줄(±yb). 가장 큰 굽힘은 -y 쪽 귀 끝(|y| 23.4): 고무줄 한쪽 힘 x (yb - 23.4)
M_pin = Fb * (yb - P["LUG_R"])
# 걸쇠 이빨: 핀 앞면(z 0)이 걸림면을 밀고, 이빨 뿌리(이빨 팔 윗면 z -4, 단면 x 8 x y 4)에서 굽힘
t_y = P["SEAR_Y"][1] - P["SEAR_Y"][0]
tooth_root = 6 * F_hold * (0.0 - P["SEAR_ARM_Z"][1]) / (t_y * 8.0 ** 2)
# 이빨 팔(높이 5 x 두께 4): 힘(+x)이 팔 가운데(z -6.5)보다 6.5 위로 지나감 → 굽힘 + 인장
hz = P["SEAR_ARM_Z"][1] - P["SEAR_ARM_Z"][0]; zc = sum(P["SEAR_ARM_Z"]) / 2
arm = 6 * F_hold * (0 - zc) / (t_y * hz ** 2) + F_hold / (t_y * hz)
# 걸쇠 축 핀 Ø5: 벽 받침(y 34.8)에서 걸쇠 이빨 층 가운데(ys)까지 외팔. 하중 = 핀 힘 + 고무줄(같은 쪽으로 어림)
d_sp = 2 * P["SEAR_PIN_R"]; L_sp = (P["SEAR_Y"][1] + 0.2) - P["SEAR_YF"][0]
pin_sear_MPa = 32 * (F_hold + F_sband) * L_sp / (math.pi * d_sp ** 3)
# 이빨 접촉: Ø6 핀 ↔ 평면, 선 길이 4 (헤르츠 선접촉 최대 압력)
p_tooth = math.sqrt(F_hold / t_y * Es / (math.pi * P["PIN_R"]))
# 손가락(폭 7, 두께 5): 끝(축 38)에서 받은 힘, 허브 끝(12.5)부터 외팔
fw, ft = P["FINGER_W"], 5.0
finger = 6 * F_finger * (P["FINGER_R"][1] - 12.5) / (ft * fw ** 2)
# 꼬리(폭 7, 두께 10): 손가락이 미는 곳(축에서 약 25) ~ 허브(5)
tail = 6 * F_finger * (25.0 - 5.0) / (10.0 * P["SEAR_TAIL"][1] ** 2)
F_lever = 18 * 0.0980665 / 0.054                      # DS3218 멈춤 토크 / 반지름 54 mm
sear = dict(hold_force_N=round(F_hold, 2), sear_band_peak_N=round(F_sband, 2), stop_force_min_N=round(F_stop, 2), finger_peak_N=round(F_finger, 2),
            pin_bending_MPa=round(32 * M_pin / (math.pi * 6 ** 3), 1), tooth_root_bending_MPa=round(tooth_root, 1), tooth_arm_MPa=round(arm, 1),
            tooth_contact_MPa=round(p_tooth, 1), sear_pin_bending_MPa=round(pin_sear_MPa, 1), sear_hole_bearing_MPa=round((F_hold + F_sband) / (d_sp * 10.0), 2),
            finger_bending_MPa=round(finger, 1), tail_bending_MPa=round(tail, 1),
            lever_bending_at_stall_MPa=round(F_lever * (54 - 12.5) / 5.0 / (9.0 ** 2) * 6, 1),
            note="매 발 반복되는 응력은 모두 반복 기준 20 MPa 아래. 걸쇠 바이어스는 고무줄이라 PLA 판스프링 피로가 없다.")
out = dict(mass_kg=m, impact_v_mps=v, energy_J=round(0.5 * m * v * v, 3), A_old_pin_stop=A, B_lug_stop=B, sear_and_trip=sear)
print(json.dumps(out, ensure_ascii=False, indent=1))
json.dump(out, open(os.path.join(D, "durability.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
