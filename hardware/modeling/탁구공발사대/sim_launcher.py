"""MuJoCo 발사 시험: 모델링 치수 그대로(총신·공 통·플런저 질량), 고무줄은 플런저 미끄럼 관절의 스프링.

순서: 쉼(플런저 앞) → 레버가 0.6 s 동안 50 mm 당김(위치 구동) → 공 통에서 공이 떨어지는지 →
걸쇠가 잡음(구속) → 레버 힘 끔 → 걸쇠 풂 → 발사. 공 속도·궤적·1.25 m 지점 높이를 잰다.
레버·걸쇠 서보는 동작만 흉내(플런저 위치 구동/고정 구속)하고 모양은 넣지 않았다.

사용: python sim_launcher.py [고무줄 한쪽 가닥 수 ...]"""
import sys, json, math, os
import numpy as np, mujoco

D = os.path.dirname(os.path.abspath(__file__))
PITCH_DEG = 15.0
PITCH = math.radians(PITCH_DEG)
BORE_R, X_REAR, X_MUZ, DRAW = 0.0208, -0.112, 0.022, 0.050
HOP_CX, HOP_TOP = -0.025, 0.112
PLA = 1240.0
BASE_H = 0.1405        # LIMO 지붕 윗면 높이(m, 팀 저장소 limo_base.stl 실측). 받침대 높이는 build_meta zshift 에 들어 있음
MUZ_LIFT = 0.0         # 발판 위 총신 축 높이 → 아래에서 계산

# 질량: 출력 STL 부피 x PLA 밀도 x 채움(얇은 벽이라 거의 꽉 참 0.9)
META = json.load(open(os.path.join(D, "build_meta.json"), encoding="utf-8"))
vol = lambda n: META[n]["vol_cm3"] * 1e-6
M_PLUNGER = (vol("plunger") + vol("pin")) * PLA * 0.9
M_BALL = 0.0027

# 고무줄(사무용 #16 정도 가정): 한 가닥 k, 쉴 때 이미 늘어난 길이
K_BAND = 80.0          # N/m  (실측 필요: 추 매달아 늘어난 길이)
PRE = 0.015            # m

def ring(name, cx, ax, r_in, x0, x1, n=28, skip=None, t=0.004):
    """축 ax('x' or 'z') 따라 x0~x1, 안 반지름 r_in 인 원통을 상자 n개로."""
    out = []
    for i in range(n):
        a = 2 * math.pi * i / n
        if skip and skip(a): continue
        rc = r_in + t / 2
        hw = (r_in + t) * math.pi / n * 1.05
        L = (x1 - x0) / 2
        if ax == "x":
            pos = (cx[0] + (x0 + x1) / 2, cx[1] + rc * math.cos(a), cx[2] + rc * math.sin(a))
            q = (math.cos(a / 2), math.sin(a / 2), 0, 0)
            size = (L, t / 2, hw)
        else:
            pos = (cx[0] + rc * math.cos(a), cx[1] + rc * math.sin(a), cx[2] + (x0 + x1) / 2)
            q = (math.cos(a / 2), 0, 0, math.sin(a / 2))
            size = (t / 2, hw, L)
        out.append(f'<geom name="{name}{i}" type="box" pos="{pos[0]:.5f} {pos[1]:.5f} {pos[2]:.5f}" '
                   f'quat="{q[0]:.5f} {q[1]:.5f} {q[2]:.5f} {q[3]:.5f}" size="{size[0]:.5f} {size[1]:.5f} {size[2]:.5f}" class="wall"/>')
    return "\n".join(out)

def build_xml(n_side):
    k = 2 * n_side * K_BAND
    # 총신: 위쪽 공 통 구멍 구간은 따로(위 4/9 둘레 빼고)
    top = lambda a: math.sin(a) > 0.01      # 실제 모델: 공 통 구멍이 총신 위 절반을 통째로 뚫음
    walls = [ring("bA", (0, 0, 0), "x", BORE_R, X_REAR, -0.046),
             ring("bB", (0, 0, 0), "x", BORE_R, -0.046, -0.004, skip=top),
             ring("bC", (0, 0, 0), "x", BORE_R, -0.004, X_MUZ),
             ring("hp", (HOP_CX, 0, 0), "z", BORE_R, BORE_R + 0.0005, HOP_TOP, skip=lambda a: False)]
    pl_len = 0.052
    return f"""
<mujoco model="pingpong_launcher">
  <option timestep="0.0001" density="1.204" viscosity="1.8e-5" integrator="implicitfast"/>
  <default>
    <geom friction="0.35 0.005 0.0001" solref="0.002 1" solimp="0.95 0.99 0.001"/>
    <default class="wall"><geom rgba="0.8 0.8 0.8 0.25" contype="1" conaffinity="1"/></default>
  </default>
  <worldbody>
    <light pos="0 0 3"/>
    <geom type="plane" size="5 5 0.1" rgba="0.3 0.3 0.3 1"/>
    <body name="launcher" pos="0 0 {BASE_H + META['zshift'] / 1000:.4f}" euler="0 {-PITCH_DEG} 0">
      {chr(10).join(walls)}
      <body name="plunger" pos="0 0 0">
        <joint name="slide" type="slide" axis="1 0 0" range="{-DRAW - 0.002} 0" limited="true"
               stiffness="{k}" springref="{PRE}" damping="0.02" solreflimit="0.001 1"/>
        <geom name="plunger" type="cylinder" fromto="{-pl_len} 0 0 0 0 0" size="{BORE_R - 0.0004}" mass="{M_PLUNGER:.5f}" rgba="1 0.6 0.1 1"/>
      </body>
    </body>
    <body name="ball1" pos="0 0 1"><freejoint/>
      <geom type="sphere" size="0.02" mass="{M_BALL}" rgba="1 1 1 1" fluidshape="ellipsoid" fluidcoef="0.5 0.25 1.5 1.0 1.0"/></body>
    <body name="ball2" pos="0 0 1.2"><freejoint/>
      <geom type="sphere" size="0.02" mass="{M_BALL}" rgba="1 1 0.6 1" fluidshape="ellipsoid" fluidcoef="0.5 0.25 1.5 1.0 1.0"/></body>
  </worldbody>
  <actuator>
    <position name="lever" joint="slide" kp="30000" kv="200" forcerange="-200 200"/>
  </actuator>
  <equality><joint name="latch" joint1="slide" active="false" polycoef="{-DRAW} 0 0 0 0"/></equality>
</mujoco>"""

def run(n_side, log_traj=False):
    m = mujoco.MjModel.from_xml_string(build_xml(n_side)); d = mujoco.MjData(m)
    L = m.body("launcher"); mujoco.mj_forward(m, d)
    R = d.body("launcher").xmat.reshape(3, 3); O = d.body("launcher").xpos.copy()
    W = lambda p: O + R @ np.array(p)
    # 공 2개: 공 통 안(플런저 위), 위아래로
    for i, name in enumerate(("ball1", "ball2")):
        j = m.body(name).jntadr[0]; a = m.jnt_qposadr[j]
        d.qpos[a:a + 3] = W((HOP_CX, 0, BORE_R + 0.021 + 0.041 * i)); d.qpos[a + 3:a + 7] = (1, 0, 0, 0)
    sl = m.jnt_qposadr[m.joint("slide").id]; act = m.actuator("lever").id
    eq = m.equality("latch").id
    t_cock, t_latch, t_free, t_fire, t_end = 0.6, 0.9, 1.0, 1.2, 2.2
    rec = dict(n_side=n_side, k_total=2 * n_side * K_BAND, m_plunger_g=round(M_PLUNGER * 1000, 1))
    exit_v, traj, fell_in, b2_stays = None, [], False, None
    ball = lambda n: d.body(n).xpos.copy()
    muz_w = W((X_MUZ, 0, 0)); axis_w = R @ np.array([1, 0, 0])
    peak_force = 0.0
    while d.time < t_end:
        t = d.time
        if t < t_cock: d.ctrl[act] = -DRAW * min(1, t / (t_cock - 0.1)) - 0.001
        elif t < t_latch: d.ctrl[act] = -DRAW - 0.001
        if t_latch <= t and not d.eq_active[eq]: d.eq_active[eq] = 1
        if t_free <= t < t_fire:   # 레버 힘 끔 (위치 이득 0)
            m.actuator_gainprm[act, 0] = 0; m.actuator_biasprm[act, 1] = 0; m.actuator_biasprm[act, 2] = 0
        if t >= t_fire and d.eq_active[eq]: d.eq_active[eq] = 0
        if t < t_fire: peak_force = max(peak_force, abs(d.actuator_force[act]))
        mujoco.mj_step(m, d)
        if abs(t - (t_latch - 0.01)) < 5e-5:
            p = ball("ball1"); rel = R.T @ (p - O)
            fell_in = bool(rel[2] < 0.005 and rel[0] < -0.005)
            rec["ball1_local_mm_before_fire"] = [round(v * 1000, 1) for v in rel]
        if t >= t_fire:
            p = ball("ball1")
            if exit_v is None and np.dot(p - muz_w, axis_w) > 0.02:
                v = d.body("ball1").cvel  # not used
                j = m.body("ball1").jntadr[0]; va = m.jnt_dofadr[j]
                exit_v = d.qvel[va:va + 3].copy(); rec["t_exit_ms"] = round((t - t_fire) * 1000, 1)
                rec["spin_rad_s"] = [round(x, 1) for x in d.qvel[va + 3:va + 6]]
            if exit_v is not None: traj.append((t, *p))
    rec["peak_lever_force_N"] = round(peak_force, 1)
    rec["ball_fell_into_barrel"] = fell_in
    if exit_v is None:
        rec["fired"] = False; return rec, None
    rec["fired"] = True
    rec["exit_speed_mps"] = round(float(np.linalg.norm(exit_v)), 2)
    rec["exit_elev_deg"] = round(math.degrees(math.atan2(exit_v[2], math.hypot(exit_v[0], exit_v[1]))), 1)
    tr = np.array(traj); x0 = muz_w[0]
    rec["muzzle_height_m"] = round(float(muz_w[2]), 3)
    i = np.searchsorted(tr[:, 1] - x0, 1.25)
    rec["height_at_1.25m_m"] = round(float(tr[i, 3]), 3) if i < len(tr) else None
    land = np.where(tr[:, 3] < 0.021)[0]
    rec["landing_dist_m"] = round(float(tr[land[0], 1] - x0), 2) if len(land) else None
    b2 = R.T @ (ball("ball2") - O); rec["ball2_local_mm_after"] = [round(v * 1000, 1) for v in b2]
    return rec, tr

if __name__ == "__main__":
    ns = [int(a) for a in sys.argv[1:]] or [1, 2, 3]
    out = []
    for n in ns:
        r, _ = run(n); out.append(r); print(json.dumps(r, ensure_ascii=False))
    json.dump(out, open(os.path.join(D, "sim_result.json"), "w"), indent=1, ensure_ascii=False)
