"""A(고무줄+서보 2개) vs B(130 DC 모터 단일 플라이휠 + 서보 1개) MuJoCo 비교.

같은 공·공기 조건(항력 + 회전 양력). 회전 양력 계수는 탁구공 실측 양력계수 C_L≈0.3(회전비 1 근처)에 맞춰
MuJoCo 마그누스 계수 = 3/8 · C_L / S ≈ 0.1 로 둔다(기본 1.0 은 약 9배 과대).
각 설계의 '모르는 값'을 흔들어, 불꽃(높이 0.07~0.20 m) 맞는 로봇 거리 구간이 얼마나 움직이는지 본다.
A: 고무줄 세기 ±25 %, 초기 늘림 ±5 mm, 공-총신 마찰 0.2~0.5
B: 배터리 전압(=모터 속도) ±10 %, 바퀴 고무 마찰 0.6~1.0, 공 눌림 ±0.5 mm, 공 탄성 ±50 %, 회전 양력 0.05~0.2
"""
import json, math, os, sys
import numpy as np, mujoco
import sim_launcher as SA

D = os.path.dirname(os.path.abspath(__file__))
MB = json.load(open(os.path.join(D, "build_meta_B.json"), encoding="utf-8"))
import importlib.util
spec = importlib.util.spec_from_file_location("F", os.path.join(D, "build_flywheel.py"))
BASE_H, PITCH = SA.BASE_H, 15.0
BORE_R, M_BALL = 0.0208, 0.0027
FLUID = lambda magnus: f'fluidshape="ellipsoid" fluidcoef="0.5 0.25 1.5 1.0 {magnus}"'
TARGET = (0.07, 0.20)

def traj_stats(tr, x0):
    """tr: (t,x,y,z). 불꽃 높이 구간을 지나는 수평 거리(로봇 총구 기준)."""
    z = tr[:, 3]; ok = (z > TARGET[0]) & (z < TARGET[1])
    if not ok.any(): return None
    xs = tr[ok, 1] - x0
    return dict(win=(round(float(xs.min()), 3), round(float(xs.max()), 3)), mid=round(float(xs.mean()), 3), top=round(float(z.max()), 3))

# ---------------- A ----------------
def run_A(k=80.0, pre=0.015, mu=0.35, n=1, magnus=0.1):
    SA.K_BAND, SA.PRE = k, pre
    orig = SA.build_xml
    def bx(nside):
        x = orig(nside).replace('euler="0 -15.0 0"', f'euler="0 {-PITCH} 0"')
        x = x.replace('friction="0.35 0.005 0.0001"', f'friction="{mu} 0.005 0.0001"')
        return x.replace('fluidshape="ellipsoid" fluidcoef="0.5 0.25 1.5 1.0 1.0"', FLUID(magnus))
    SA.build_xml = bx
    try:
        rec, tr = SA.run(n)
    finally:
        SA.build_xml = orig
    if tr is None: return dict(fired=False)
    st = traj_stats(tr, tr[0, 1] - 0.02)
    return dict(v=rec["exit_speed_mps"], spin=rec["spin_rad_s"][1], **(st or {}))

# ---------------- B ----------------
FB = dict(X_REAR=-0.066, X_MUZ=0.058, HOP_CX=-0.025, HOP_TOP=0.112, FACE_BACK=-0.050, FACE_FWD=-0.022, PUSH_L=0.022,
          WX=0.012, WZ=-0.039, WR=0.021, WW=0.014, HOOD=0.020)
TS5, W05 = 0.0042, 1590.0      # FA-130RA 5 V 환산: 정지 토크 4.2 mN·m, 무부하 1590 rad/s(15,200 rpm)

def xml_B(mu_w=0.9, hood=FB["HOOD"], ball_k=6000.0, magnus=0.1, m_wheel=0.015):
    """ball_k: 탁구공 눌림 강성 N/m → 접촉 시간상수 sqrt(m/k) 로(MuJoCo 음수 solref 는 질량으로 나뉜 값이라 쓰지 않음)."""
    top = lambda a: math.sin(a) > 0.01
    bot = lambda a: math.sin(a) < -0.80           # 바퀴 틈(실물 폭 17 mm; 상자 근사라 넉넉히)
    w = [SA.ring("bA", (0, 0, 0), "x", BORE_R, FB["X_REAR"], -0.046),
         SA.ring("bB", (0, 0, 0), "x", BORE_R, -0.046, FB["WX"] - 0.023, skip=top),
         SA.ring("bC", (0, 0, 0), "x", BORE_R, FB["WX"] - 0.023, -0.004, skip=lambda a: top(a) or bot(a)),
         SA.ring("bD", (0, 0, 0), "x", BORE_R, -0.004, FB["WX"] + 0.023, skip=bot),
         SA.ring("bE", (0, 0, 0), "x", BORE_R, FB["WX"] + 0.023, FB["X_MUZ"]),
         SA.ring("hp", (FB["HOP_CX"], 0, 0), "z", BORE_R, BORE_R + 0.0005, FB["HOP_TOP"])]
    I = 0.5 * m_wheel * (FB["WR"] - 0.001) ** 2
    return f"""
<mujoco model="flywheel">
  <option timestep="0.00005" density="1.204" viscosity="1.8e-5" integrator="implicitfast"/>
  <default>
    <geom friction="0.35 0.005 0.0001" solref="0.002 1" solimp="0.95 0.99 0.001"/>
    <default class="wall"><geom rgba="0.8 0.8 0.8 0.25"/></default>
  </default>
  <worldbody>
    <geom type="plane" size="5 5 0.1"/>
    <body name="launcher" pos="0 0 {BASE_H + MB['zshift'] / 1000:.4f}" euler="0 {-PITCH} 0">
      {chr(10).join(w)}
      <geom name="hood" type="box" pos="{FB['WX'] + 0.004} 0 {hood + 0.001}" size="0.016 0.012 0.001" class="wall" contype="0" conaffinity="0"/>
      <body name="pusher"><joint name="push" type="slide" axis="1 0 0" range="{FB['FACE_BACK'] - 0.001} {FB['FACE_FWD'] + 0.001}" damping="0.05"/>
        <geom type="cylinder" fromto="{-FB['PUSH_L']} 0 0 0 0 0" size="{BORE_R - 0.0004}" mass="0.012"/></body>
      <body name="wheel" pos="{FB['WX']} 0 {FB['WZ']}">
        <joint name="spin" type="hinge" axis="0 1 0" armature="2e-7" damping="1e-7"/>
        <geom name="wheel" type="cylinder" size="{FB['WR']} {FB['WW'] / 2}" euler="90 0 0" mass="{m_wheel}" contype="0" conaffinity="0"/>
      </body>
    </body>
    <body name="ball1" pos="0 0 1"><freejoint/>
      <geom name="ball1" type="sphere" size="0.02" mass="{M_BALL}" {FLUID(magnus)}/></body>
    <body name="ball2" pos="0 0 1.2"><freejoint/>
      <geom name="ball2" type="sphere" size="0.02" mass="{M_BALL}" {FLUID(magnus)}/></body>
  </worldbody>
  <contact>
    <pair geom1="ball1" geom2="wheel" solref="{math.sqrt(M_BALL / ball_k):.6f} 0.7" friction="{mu_w} {mu_w} 0.005 0.0001 0.0001" condim="6"/>
    <pair geom1="ball2" geom2="wheel" solref="{math.sqrt(M_BALL / ball_k):.6f} 0.7" friction="{mu_w} {mu_w} 0.005 0.0001 0.0001" condim="6"/>
    <pair geom1="ball1" geom2="hood" solref="{math.sqrt(M_BALL / ball_k):.6f} 0.7" friction="0.35 0.35 0.005 0.0001 0.0001" condim="6"/>
    <pair geom1="ball2" geom2="hood" solref="{math.sqrt(M_BALL / ball_k):.6f} 0.7" friction="0.35 0.35 0.005 0.0001 0.0001" condim="6"/>
  </contact>
  <actuator>
    <position name="servo" joint="push" kp="3000" kv="30" forcerange="-60 60"/>
    <motor name="motor" joint="spin" gear="1" ctrllimited="false"/>
  </actuator>
</mujoco>"""

def run_B(duty=0.5, volt=1.0, shots=1, interval=2.0, first=2.5, **kw):
    """first: 첫 발 시각(s). 바퀴가 다 돌기까지 1 s 넘게 걸려 2.5 s 로."""
    m = mujoco.MjModel.from_xml_string(xml_B(**kw)); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    R = d.body("launcher").xmat.reshape(3, 3); O = d.body("launcher").xpos.copy(); W = lambda p: O + R @ np.array(p)
    balls = ["ball1", "ball2"]
    def place(name, slot):
        a = m.jnt_qposadr[m.body(name).jntadr[0]]
        d.qpos[a:a + 3] = W((FB["HOP_CX"], 0, BORE_R + 0.021 + 0.041 * slot)); d.qpos[a + 3:a + 7] = (1, 0, 0, 0)
        d.qvel[m.jnt_dofadr[m.body(name).jntadr[0]]:][:6] = 0
    d.qpos[m.jnt_qposadr[m.joint("push").id]] = FB["FACE_BACK"]
    place("ball1", 0); place("ball2", 1)
    sp = m.jnt_dofadr[m.joint("spin").id]; act_s, act_m = m.actuator("servo").id, m.actuator("motor").id
    muz = W((FB["X_MUZ"], 0, 0)); ax = R @ np.array([1, 0, 0])
    out, spin_up = [], None
    t_shot = [first + i * interval for i in range(shots)]
    cur, tr, done, vwheel_before = 0, [], False, []
    while d.time < t_shot[-1] + 1.4:
        t = d.time; om = d.qvel[sp]
        d.ctrl[act_m] = TS5 * (duty * volt - om / W05)          # 직류 모터: 토크 = 정지토크 x (전압비 - 속도/무부하속도)
        if spin_up is None and om >= 0.95 * duty * volt * W05 * 0.999: spin_up = round(t, 3)
        # 밀대: 각 발사 시각에 0.25 s 앞으로, 0.4 s 뒤 되돌림
        ph = [t - ts for ts in t_shot if t >= ts]
        if ph and ph[-1] < 0.25: d.ctrl[act_s] = FB["FACE_BACK"] + (FB["FACE_FWD"] - FB["FACE_BACK"]) * ph[-1] / 0.25
        elif ph and ph[-1] < 0.6: d.ctrl[act_s] = FB["FACE_FWD"]
        else: d.ctrl[act_s] = FB["FACE_BACK"]
        if cur < shots and abs(t - t_shot[cur]) < 2.5e-5: vwheel_before.append(round(float(om), 1))
        mujoco.mj_step(m, d)
        if cur < shots:
            b = balls[cur % 2]; p = d.body(b).xpos.copy()
            if np.dot(p - muz, ax) > 0.02 and not done:
                va = m.jnt_dofadr[m.body(b).jntadr[0]]; v = d.qvel[va:va + 3].copy()
                out.append(dict(v=round(float(np.linalg.norm(v)), 2), spin=round(float(d.qvel[va + 4]), 1),
                                elev=round(math.degrees(math.atan2(v[2], math.hypot(v[0], v[1]))), 1)))
                done = True; tr = []
            if done:
                tr.append((t, *p))
                if p[2] < 0.03 or t > t_shot[cur] + 1.3:
                    out[-1].update(traj_stats(np.array(tr), muz[0]) or {})
                    # 다음 발: 쏜 공을 공 통 맨 위로 되돌림
                    place(b, 1); cur += 1; done = False
    return dict(spin_up_s=spin_up, wheel_rad_s_before=vwheel_before, shots=out)

if __name__ == "__main__":
    res = {}
    def show(tag, r): print(tag, json.dumps(r, ensure_ascii=False), flush=True); return r
    # A 기본과 흔들기
    A = res["A"] = {}
    A["nominal"] = show("A 기본", run_A())
    for tag, kw in (("k-25%", dict(k=60.0)), ("k+25%", dict(k=100.0)), ("pre-5mm", dict(pre=0.010)), ("pre+5mm", dict(pre=0.020)),
                    ("mu0.2", dict(mu=0.2)), ("mu0.5", dict(mu=0.5)), ("magnus0.05", dict(magnus=0.05)), ("magnus0.2", dict(magnus=0.2))):
        A[tag] = show("A " + tag, run_A(**kw))
    # B: A 와 같은 거리(가운데) 나오는 듀티 찾기
    target = A["nominal"]["mid"]
    lo, hi, scan = 0.55, 1.0, {}
    for it in range(6):
        duty = round((lo + hi) / 2, 3); r = run_B(duty=duty)["shots"][0]; scan[duty] = r
        print("B duty", duty, r, flush=True)
        if r.get("mid", 0) < target: lo = duty
        else: hi = duty
    duty = min(scan, key=lambda k: abs(scan[k].get("mid", 0) - target))
    B = res["B"] = dict(duty=duty, duty_scan=scan)
    B["nominal"] = scan[duty]
    for tag, kw in (("volt-10%", dict(volt=0.9)), ("volt+10%", dict(volt=1.1)), ("mu_w0.6", dict(mu_w=0.6)), ("mu_w1.0", dict(mu_w=1.0)),
                    ("hood-0.5mm(더 눌림)", dict(hood=FB["HOOD"] - 0.0005)), ("hood+0.5mm", dict(hood=FB["HOOD"] + 0.0005)),
                    ("ball_k x0.5", dict(ball_k=3000.0)), ("ball_k x1.5", dict(ball_k=9000.0)),
                    ("magnus0.05", dict(magnus=0.05)), ("magnus0.2", dict(magnus=0.2))):
        r = run_B(duty=duty, **kw); B[tag] = show("B " + tag, r["shots"][0] if r["shots"] else dict(fired=False))
    r = run_B(duty=duty, shots=3, interval=2.0); B["3shots_2s"] = show("B 연속 3발(2초 간격)", r)
    r = run_B(duty=duty, shots=3, interval=0.8); B["3shots_0.8s"] = show("B 연속 3발(0.8초 간격)", r)
    json.dump(res, open(os.path.join(D, "sim_compare.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
