"""v5 밀기 방식 최적화 탐색. 레버 축(x,z)·고무줄 쉴 때 늘림·걸쇠 걸림 자리·움직이는 무게·레버-핀 마찰을 바꿔
sim_full(v5 사본)로 쏜다. 비교 기준: 장전 중 서보 회전력 최대(시작 가속 80 ms 제외)를 0.6 으로 맞춘 고무줄 세기에서의 공 속도."""
import math, json, sys, copy, re
import numpy as np
import sim_full as S
S.B.DT = 1e-4
BASE = copy.deepcopy(vars(S.B)); MP0 = copy.deepcopy(S.MP); BX0 = S.build_xml
CFG = {}
def build_xml_patched(theta, n_balls=5):
    xml = BX0(theta, n_balls)
    mu = CFG.get("mulev")
    if mu is not None:                       # 레버 층 핀(롤러) 마찰만 따로
        fr = f'friction="{mu} 0.005 0.0001"'
        xml = xml.replace('contype="64" conaffinity="16"', f'contype="64" conaffinity="16" {fr}')
        m_ = re.search(r'<geom name="pinP" type="cylinder" fromto="([-0-9.]+) 0.018 0 ([-0-9.]+) ([-0-9.]+) 0" size="([-0-9.]+)" ([^/]*)/>', xml)
        pr, _, y1, r, ct = m_.groups()
        xml = xml.replace(m_.group(0), f'<geom name="pinL" type="cylinder" fromto="{pr} 0.018 0 {pr} 0.0300 0" size="{r}" {ct} {fr}/>'
                                       f'<geom name="pinP" type="cylinder" fromto="{pr} 0.0300 0 {pr} {y1} 0" size="{r}" {ct}/>')
    return xml
S.build_xml = build_xml_patched

def geom(piv=(-57.0, 48.0), gap10=10.0, over=5.0, face_gap=None, s0=None, mscale=1.0, mulev=None, ball_g=2.7):
    for kk, v in BASE.items(): setattr(S.B, kk, copy.deepcopy(v))
    S.MP = copy.deepcopy(MP0); B = S.B
    px, pz = piv; B.LEVER_PIV = [px, pz]
    ang = lambda x: math.degrees(math.atan2(-pz, x - px))      # 축에서 핀(x, 0) 방향
    a_rest = ang(B.PIN_REST); a_cock = ang(B.PIN_COCK - over)
    B.LEVER_REST_DEG = a_rest + gap10
    B.LEVER_COCK_DEG = (a_rest - a_cock) + gap10                 # 이 각에서 핀이 PIN_COCK - over
    dmax = max(math.hypot(B.PIN_REST - px, pz), math.hypot(B.PIN_COCK - over - px, pz))
    B.LEVER_LEN = dmax + 2.0
    if face_gap is not None:                                     # 걸쇠 걸림면: 장전 끝(-80)에서 face_gap 앞
        B.SEAR_GAP = face_gap; B.SEAR_FACE_X = B.PIN_COCK + B.PIN_R + face_gap
    sx, sz = B.SEAR_PIV; ux, uz = px - sx, pz - sz; L = math.hypot(ux, uz)
    tl = B.SEAR_TAIL[0]; T = (sx + ux / L * tl, sz + uz / L * tl)
    ta = math.degrees(math.atan2(T[1] - pz, T[0] - px)); td = math.hypot(T[0] - px, T[1] - pz)
    B.FINGER_DEG = ta - 9.5; B.FINGER_R = [11.0, td + 5.0]
    if s0 is not None: S.PRE = s0 * 1e-3; S.HOOK_X = B.PIN_REST + 31.7 + s0
    else: S.PRE = 0.0103; S.HOOK_X = 12.0
    if mscale != 1.0:
        p = S.MP["plunger_pin"]; p["mass"] *= mscale; p["inertia"] = [[v * mscale for v in row] for row in p["inertia"]]
    CFG["mulev"] = mulev; S.M_BALL = ball_g * 1e-3
    return dict(rest=round(B.LEVER_REST_DEG, 1), cock=round(B.LEVER_COCK_DEG, 1), len=round(B.LEVER_LEN, 1), finger=round(B.FINGER_DEG, 1), fr=round(B.FINGER_R[1], 1))

def shoot(theta=30, cock_theta=None, shots=3, k=1.0, tau=1.0):
    st = dict(t0=None, peak=0.0, prev=0.0)
    tau_max = S.SERVOS["DS3218"]["tau"] * tau
    def cb(m, d, tgt):
        if tgt["lever"] > 0 and st["prev"] <= 0: st["t0"] = d.time
        st["prev"] = tgt["lever"]
        if tgt["lever"] > 0 and st["t0"] is not None and d.time - st["t0"] > 0.08 and d.qvel[m.jnt_dofadr[m.joint("lever").id]] > 0.05:
            st["peak"] = max(st["peak"], abs(d.ctrl[m.actuator("lever").id]) / tau_max)
    r = S.run(theta=theta, cock_theta=cock_theta, k_scale=k, tau_scale=tau, shots=shots, on_step=cb)
    sh = r["shots"]
    v = [x.get("exit_speed_mps") for x in sh]
    return dict(v=v, vmean=round(float(np.mean([x for x in v if x])), 3) if all(v) else None, held=all(x.get("held_by_sear") for x in sh),
                push_peak=round(st["peak"], 3), hold=round(max(x.get("lever_hold_torque_frac") or 0 for x in sh), 3),
                extra=[x.get("extra_balls") for x in sh], pin_cock=[x.get("pin_x_at_cock_mm") for x in sh], ok=r.get("all_ok"))

def at60(g, theta=30, shots=3, target=0.6, **kw):
    """k=1 로 한 번, 최대 회전력이 target 이 되게 k 를 맞춰 다시 쏜다."""
    r1 = shoot(theta=theta, shots=shots, k=1.0)
    if not r1["push_peak"]: return dict(g=g, r1=r1)
    k2 = target / r1["push_peak"]
    r2 = shoot(theta=theta, shots=shots, k=k2)
    return dict(g=g, r1=r1, k60=round(k2, 2), r60=r2)
if __name__ == "__main__":
    cases = json.loads(sys.argv[1])
    for c in cases:
        g = geom(**c)
        print(json.dumps(dict(case=c, **at60(g)), ensure_ascii=False, default=str), flush=True)
