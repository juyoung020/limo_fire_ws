"""v5 탐색: sim_full(v5 사본)의 치수를 실행 중에 바꿔 가며 돌린다."""
import math, json, sys, copy
import sim_full as S
S.B.DT = 1e-4
BASE = copy.deepcopy(vars(S.B))
def setup(strike=True, gap=-8.0, rib_h=8.0, muz=None, hop_dx=6.0, s0=None, piv=None, rest_deg=None, cock_deg=None, lev_len=None,
          sear_gap=None, finger_deg=None, finger_r=None, ball_g=5.3, e=0.7, plunger_scale=None, k=1.0):
    for kk, v in BASE.items(): setattr(S.B, kk, copy.deepcopy(v))
    B = S.B
    if strike:
        bx = 0.0 + 20.0 + gap                          # 플런저 쉴 때 앞면 x = 0
        B.RIB_H = rib_h; B.RIB_X = bx - math.sqrt(20.0 ** 2 - (20.0 - rib_h - 0.8) ** 2)
        hc = bx + hop_dx                                # 공 통을 앞으로 비켜: 위 공 무게가 아래 공을 턱 쪽(뒤)으로 민다
        B.HOP_X = [hc - 20.0, hc + 20.0]; B.HOP_CX = hc; B.X_MUZ = muz if muz is not None else hc + 20.5
        B.RAMP = [30.0, 0.01]
        S.E_BALL = e
    else:
        B.RIB_X = None; S.E_BALL = None
    if s0 is not None:                                 # 고무줄 쉴 때 늘림(유니아트 4cm: 늘림 0 인 걸이 거리 31.7)
        S.PRE = s0 * 1e-3; S.HOOK_X = B.PIN_REST + 31.7 + s0
    else:
        S.PRE = 0.0103; S.HOOK_X = 12.0
    if piv is not None: B.LEVER_PIV = list(piv)
    if rest_deg is not None: B.LEVER_REST_DEG = rest_deg
    if cock_deg is not None: B.LEVER_COCK_DEG = cock_deg
    if lev_len is not None: B.LEVER_LEN = lev_len
    if sear_gap is not None:
        B.SEAR_GAP = sear_gap; B.SEAR_FACE_X = B.PIN_COCK + B.PIN_R + sear_gap
    if finger_deg is not None: B.FINGER_DEG = finger_deg
    if finger_r is not None: B.FINGER_R = list(finger_r)
    S.M_BALL = ball_g * 1e-3
    return k
def shoot(theta=30, cock_theta=None, shots=3, k=1.0, **kw):
    k = setup(**kw)
    r = S.run(theta=theta, cock_theta=cock_theta, k_scale=k, shots=shots)
    sh = r["shots"]
    return dict(v=[x.get("exit_speed_mps") for x in sh], held=[x.get("held_by_sear") for x in sh],
                hold=max(x.get("lever_hold_torque_frac") or 0 for x in sh), peak=r.get("lever_peak_torque_frac"),
                cock=[x.get("lever_deg_at_cock") for x in sh], extra=[x.get("extra_balls") for x in sh],
                inb=[x.get("ball_in_barrel") for x in sh], trip=max(x.get("trip_lever_peak_torque_frac") or 0 for x in sh))
if __name__ == "__main__":
    print("v4 push", shoot(strike=False))
    print("strike only", shoot(strike=True))
