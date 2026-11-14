"""발사대 전체 동작 시뮬(MuJoCo). 레버·걸쇠·플런저+핀·공 5개·각도 축이 모두 실제 강체로 움직인다.

- 모양: build_launcher 치수(params.json), 질량·관성은 mass_props.json(출력 무게 추정, 서보 무게 포함).
- 플런저+핀: x 미끄럼 + 좌우·위아래 이동·기울기(총신 틈 0.4 안에서 자유, 벽에 닿아 막힘).
  핀은 홈 위아래 벽, 레버, 걸쇠 이빨에 실제로 닿고, 플런저 귀(45° 면)가 홈 앞끝(45° 면)에 닿아 멈춘다.
- 레버(DS3218)·각도(MG996R): 강체 + 서보 모델
    토크 한계(멈춤 토크, 속도가 오를수록 줄어드는 직선), 무부하 속도, 기어 관성(armature 2e-3 kg·m2), 출력축 각도 비례 제어(4° 에서 최대 토크).
    각도 서보는 기어 유격 0.8° 를 둔 탄성 연결로 발사대(실제 질량 분포)를 붙든다. 발사 반동·무게가 그대로 걸린다.
- 걸쇠: 서보 없는 강체(축 y). 고무줄 1 고리(갈고리 ↔ 몸통 기둥)가 닫는 쪽으로 당기고, 닫힘 멈춤 턱에 닿아 선다.
  이빨(걸림면·비탈)은 핀에, 꼬리는 레버 손가락에 실제로 닿는다. 핀이 비탈을 눌러 열고, 지나가면 고무줄이 닫는다.
- 고무줄: 핀 홈 ↔ 고리(x 12) 사이 직선 힘, 고리 k_loop(N/m) x 늘어난 길이. 반작용은 몸통에 걸린다.
- 공: 껍데기 관성(2/3 m r2), 공기 저항 Cd 0.45 + 회전 양력. 마찰 0.35(PLA-PLA), 견고성 확인은 0.5.
- 명령은 실제 코드처럼 시간표대로만 준다(센서 없음). 한 발:
    [15° 미만이면 15° 로 천천히 올림] → 장전(레버 66°) → 레버 천천히 0°(200°/s, 핀이 걸쇠에 걸림)
    → [목표 각도로 40°/s 로 내리고 0.15 s 멈춤] → 레버 -18°(발사)
사용: python sim_full.py            전체(각도 5·15·30 x 고무줄 ±25 % x 서보 힘 반 + 마찰 0.5 + 고무 히스테리시스)
      python sim_full.py quick      15° 기본 한 번
      python sim_full.py core       15° 고정 3 발(기본·고무줄 ±25 %·서보 힘 반) → sim_core.json"""
import os, sys, json, math
import numpy as np, mujoco
from types import SimpleNamespace as NS

D = os.path.dirname(os.path.abspath(__file__))
P = json.load(open(os.path.join(D, "params.json"), encoding="utf-8")); B = NS(**P)
MP = json.load(open(os.path.join(D, "mass_props.json"), encoding="utf-8"))
mm = 0.001
ROOF_Z = 0.1405
BASE_LINK_X = 14.9 * mm
TARGET = (0.07, 0.20)
M_BALL, R_BALL = 0.0027, 0.020
K_LOOP, PRE = 78.0, 0.0103            # 고무줄 한 고리(N/m), 쉴 때 늘림(m) — 실측 후 고침
HOOK_X = 12.0
KGCM = 0.0980665
SERVOS = dict(DS3218=dict(tau=18 * KGCM, w0=math.radians(60) / 0.16, J=2e-3),
              MG996R=dict(tau=9.4 * KGCM, w0=math.radians(60) / 0.17, J=2e-3))
MU = 0.35                             # PLA-PLA, PLA-공
COCK_MIN = 15.0                       # 장전은 이 각도 이상에서 하고 목표 각도로 천천히 내린다(5° 에서 바로 장전하면 공 두 개가 같이 나감)
FLUID = 'fluidshape="ellipsoid" fluidcoef="0.225 0.25 1.5 1.0 0.1"'
# 충돌 비트: 1 공, 2 벽, 4 플런저 앞면·윗면(공), 8 플런저 몸통(총신 벽), 16 핀, 32 홈 벽·끝, 64 레버 팔,
#           128 걸쇠(핀·멈춤 턱·손가락과 닿음), 256 멈춤 턱, 512 레버 손가락
CT = dict(ball='contype="1" conaffinity="7"', wall='contype="2" conaffinity="8"', face='contype="4" conaffinity="0"',
          bore='contype="8" conaffinity="0"', pin='contype="16" conaffinity="0"', slot='contype="32" conaffinity="16"',
          arm='contype="64" conaffinity="16"', sear='contype="128" conaffinity="784"', stop='contype="256" conaffinity="0"',
          finger='contype="512" conaffinity="0"')
K_SEAR, SEAR_FREE = 78.0, 0.0317      # 걸쇠 고무줄 한 고리(N/m), 늘림 0 인 갈고리-기둥 거리(m) — 주 고무줄과 같은 고리 어림

def f3(v): return " ".join(f"{x:.6f}" for x in v)

def ring(r_in, x0, x1, n=48, skip=None, t=0.004, ax="x", c=(0, 0, 0)):
    out = []
    for i in range(n):
        a = 2 * math.pi * i / n
        if skip and skip(a): continue
        rc = r_in + t / 2; hw = (r_in + t) * math.pi / n * 1.05; L = (x1 - x0) / 2
        if ax == "x":
            pos = (c[0] + (x0 + x1) / 2, c[1] + rc * math.cos(a), c[2] + rc * math.sin(a)); q = (math.cos(a / 2), math.sin(a / 2), 0, 0); size = (L, t / 2, hw)
        else:
            pos = (c[0] + rc * math.cos(a), c[1] + rc * math.sin(a), c[2] + (x0 + x1) / 2); q = (math.cos(a / 2), 0, 0, math.sin(a / 2)); size = (t / 2, hw, L)
        out.append(f'<geom type="box" pos="{f3(pos)}" quat="{f3(q)}" size="{f3(size)}" class="wall"/>')
    return "\n".join(out)

def launcher_pose(theta):
    ax, az = B.AX; th = math.radians(theta)
    x = B.X_AX + (-ax) * math.cos(th) - (-az) * math.sin(th)
    z = B.H_AX + (-ax) * math.sin(th) + (-az) * math.cos(th)
    return np.array([x * mm, 0.0, ROOF_Z + z * mm])

def inertial(key, origin=(0, 0, 0)):
    p = MP[key]; c = np.array(p["com"]) - np.array(origin) * mm; I = np.array(p["inertia"])
    return (f'<inertial pos="{f3(c)}" mass="{p["mass"]:.6f}" '
            f'fullinertia="{I[0,0]:.4e} {I[1,1]:.4e} {I[2,2]:.4e} {I[0,1]:.4e} {I[0,2]:.4e} {I[1,2]:.4e}"/>')

def build_xml(theta, n_balls=5):
    BR = B.BORE_R * mm
    sx0, sx1 = B.SLOT_X0, B.SLOT_X1
    hx0, hx1 = B.HOP_X
    xs = sorted({B.X_REAR, sx0, hx0, sx1, hx1, B.X_MUZ})
    walls = []
    for a, b in zip(xs[:-1], xs[1:]):
        mid = (a + b) / 2
        in_hop = hx0 <= mid <= hx1; in_slot = sx0 <= mid <= sx1
        def skip(t, in_hop=in_hop, in_slot=in_slot):
            return (in_hop and math.sin(t) > 0.01) or (in_slot and abs(math.sin(t)) < (B.SLOT_W / 2 + 0.4) / B.BORE_R)
        walls.append(ring(BR, a * mm, b * mm, skip=skip))
    walls.append(ring(BR, BR + 0.0005, B.HOP_TOP * mm, ax="z", c=(B.HOP_CX * mm, 0, 0)))
    walls.append(f'<geom type="cylinder" pos="{B.HOP_CX * mm:.5f} 0 {(B.HOP_TOP + 1) * mm:.5f}" size="{(B.BORE_R + 3) * mm:.5f} 0.001" class="wall"/>')
    # 홈: 위·아래 벽(핀·귀를 위아래로 안내), 앞끝 = 45° 면(x + |y| = LUG_C), 뒤끝은 트임
    yi, yo = B.BORE_R, B.BORE_R + 2.4; hw = B.SLOT_W / 2
    slot = []
    ym = (yi + yo) / 2; ht = (yo - yi) / 2 * math.sqrt(2)          # 45° 면 위 벽 두께 구간 반길이
    for s in (-1, 1):
        yc = s * ym * mm; yh = (yo - yi) / 2 * mm
        for zc in (hw + 1.5, -hw - 1.5):
            slot.append(f'<geom name="slotwall{s}{int(zc>0)}" type="box" pos="{(sx0 + sx1) / 2 * mm:.5f} {yc:.5f} {zc * mm:.5f}" size="{(sx1 - sx0) / 2 * mm:.5f} {yh:.5f} 0.0015" {CT["slot"]}/>')
        n = np.array([1, s, 0]) / math.sqrt(2)                       # 귀 앞면 바깥 법선
        fp = np.array([B.LUG_C - ym, s * ym, 0.0])                      # 면 위 한 점(벽 두께 가운데)
        c = (fp + n * 6.0) * mm
        slot.append(f'<geom name="stopF{s}" type="box" pos="{f3(c)}" euler="0 0 {45 * s}" size="0.006 {ht * mm:.5f} {(hw + 1) * mm:.5f}" solref="0.0005 1" {CT["slot"]}/>')
    lug_xml = ""
    for s_ in (-1, 1):
        n = np.array([1, s_, 0]) / math.sqrt(2); fp = np.array([B.LUG_C - ym, s_ * ym, 0.0])
        c = (fp - n * 5.0) * mm
        lug_xml += (f'<geom name="lug{s_}" type="box" pos="{f3(c)}" euler="0 0 {45 * s_}" size="0.005 {ht * mm:.5f} {B.LUG_H * mm:.5f}" solref="0.0005 1" {CT["pin"]}/>')
    p = launcher_pose(theta)
    L = B.PLUNGER_L * mm; PR = B.PLUNGER_R * mm; rl, rh = B.RAMP[0] * mm, B.RAMP[1] * mm
    pr = B.PIN_REST * mm; pin_r = B.PIN_R * mm
    balls = "".join(f'<body name="ball{i}" pos="0 0 {2 + i * 0.1}"><freejoint/><inertial pos="0 0 0" mass="{M_BALL}" '
                    f'diaginertia="{2/3*M_BALL*R_BALL**2:.3e} {2/3*M_BALL*R_BALL**2:.3e} {2/3*M_BALL*R_BALL**2:.3e}"/>'
                    f'<geom type="sphere" size="{R_BALL}" {CT["ball"]} {FLUID}/></body>' for i in range(n_balls))
    LP = (B.LEVER_PIV[0] * mm, 0, B.LEVER_PIV[1] * mm); ly0, ly1 = B.LEVER_Y[0] * mm, B.LEVER_Y[1] * mm
    ra = math.radians(B.LEVER_REST_DEG); dvec = np.array([math.cos(ra), 0, math.sin(ra)]); nvec = np.array([-math.sin(ra), 0, math.cos(ra)])
    lev_c = (dvec * B.LEVER_LEN / 2 + nvec * B.LEVER_W / 2) * mm + np.array([0, (ly0 + ly1) / 2, 0])     # 미는 면 = 축을 지나는 직선, 팔 몸은 그 위쪽
    lev_t = (dvec * B.LEVER_LEN + nvec * B.LEVER_W / 2) * mm
    SP = np.array([B.SEAR_PIV[0], 0.0, B.SEAR_PIV[1]])
    rel = lambda x, z: f3(((x - SP[0]) * mm, 0, (z - SP[2]) * mm))
    ty0, ty1 = B.SEAR_Y; yc_t = (ty0 + ty1) / 2 * mm; yh_t = (ty1 - ty0) / 2 * mm
    fy0, fy1 = B.SEAR_YF; yc_f = (fy0 + fy1) / 2 * mm; yh_f = (fy1 - fy0) / 2 * mm
    fx, top = B.SEAR_FACE_X, B.SEAR_TOP; a0, a1 = B.SEAR_ARM_Z
    def gbox(cx, cz, hx, hz, yc, yh, ang=0.0, name=""):
        nm = f'name="{name}" ' if name else ""
        return f'<geom {nm}type="box" pos="{(cx - SP[0]) * mm:.6f} {yc:.6f} {(cz - SP[2]) * mm:.6f}" euler="0 {ang:.3f} 0" size="{hx * mm:.6f} {yh:.6f} {hz * mm:.6f}" {CT["sear"]}/>'
    ux, uz = B.LEVER_PIV[0] - SP[0], B.LEVER_PIV[1] - SP[2]; Lu = math.hypot(ux, uz); ux, uz = ux / Lu, uz / Lu
    tl, tw = B.SEAR_TAIL
    sear_xml = chr(10).join([
        gbox(fx + 1.0, (a1 + top) / 2, 1.0, (top - a1) / 2, yc_t, yh_t, name="tooth"),
        gbox(fx + 5.0 - 0.53, -1.0 - 0.53, 3 * math.sqrt(2), 0.75, yc_t, yh_t, ang=45.0, name="ramp"),
        gbox((SP[0] - 1.0 + fx + 8.0) / 2, (a0 + a1) / 2, (fx + 8.0 - SP[0] + 1.0) / 2, (a1 - a0) / 2, yc_t, yh_t, name="searArm"),
        gbox(SP[0] + ux * tl / 2, SP[2] + uz * tl / 2, tl / 2, tw / 2, yc_f, yh_f, ang=-math.degrees(math.atan2(uz, ux)), name="tail"),
        gbox((B.SEAR_NOSE[0] + B.SEAR_NOSE[1]) / 2, B.SEAR_NOSE[2] + 1.0, (B.SEAR_NOSE[1] - B.SEAR_NOSE[0]) / 2, 1.0, yc_f, yh_f, name="nose")])
    stop_xml = (f'<geom name="searStop" type="box" pos="{(B.SEAR_STOP_X[0] + B.SEAR_STOP_X[1]) / 2 * mm:.6f} {(ty0 + 0.2 + ty1 + 3) / 2 * mm:.6f} {(B.SEAR_NOSE[2] - 1.25) * mm:.6f}" '
                f'size="{(B.SEAR_STOP_X[1] - B.SEAR_STOP_X[0]) / 2 * mm:.6f} {(ty1 + 3 - ty0 - 0.2) / 2 * mm:.6f} {1.25 * mm:.6f}" {CT["stop"]}/>')
    fa = math.radians(B.FINGER_DEG); fd = np.array([math.cos(fa), 0, math.sin(fa)]); fn = np.array([-math.sin(fa), 0, math.cos(fa)])
    fr0, fr1 = B.FINGER_R; fw = B.FINGER_W
    fc = (fd * (fr0 + fr1) / 2 - fn * fw / 2) * mm + np.array([0, (ly0 + ly1) / 2, 0]); ftp = (fd * fr1 - fn * fw / 2) * mm
    finger_xml = (f'<geom name="finger" type="box" pos="{f3(fc)}" euler="0 {-B.FINGER_DEG:.3f} 0" size="{(fr1 - fr0) / 2 * mm:.6f} {(ly1 - ly0) / 2:.6f} {fw / 2 * mm:.6f}" {CT["finger"]}/>'
                  f'<geom type="cylinder" fromto="{ftp[0]:.6f} {ly0:.6f} {ftp[2]:.6f} {ftp[0]:.6f} {ly1:.6f} {ftp[2]:.6f}" size="{fw / 2 * mm:.6f}" {CT["finger"]}/>')
    ax = (B.AX[0] * mm, 0, B.AX[1] * mm)
    xml = f"""
<mujoco model="launcher_full">
  <option timestep="{B.DT}" density="1.204" viscosity="1.8e-5" integrator="implicitfast" cone="elliptic">
    <flag filterparent="disable"/></option>
  <default><geom friction="{MU} 0.005 0.0001" solref="0.002 1" solimp="0.95 0.99 0.001" mass="0"/>
    <default class="wall"><geom {CT["wall"]} rgba="0.8 0.8 0.8 0.25"/></default></default>
  <worldbody>
    <geom type="plane" size="5 5 0.1"/>
    <body name="launcher" pos="{f3(p)}" euler="0 {-theta} 0">
      <joint name="tilt" type="hinge" axis="0 1 0" pos="{f3(ax)}" armature="0" damping="0.0005"/>
      {inertial("static")}
      {chr(10).join(walls)}
      {chr(10).join(slot)}
      <body name="plunger">
        <joint name="slide" type="slide" axis="1 0 0" range="{-(B.DRAW + B.PIN_OVER + 8) * mm:.5f} 0.004" limited="true" damping="0.002"/>
        <joint name="py" type="slide" axis="0 1 0" range="-0.002 0.002" limited="true"/>
        <joint name="pz" type="slide" axis="0 0 1" range="-0.002 0.002" limited="true"/>
        <joint name="yaw" type="hinge" axis="0 0 1" pos="-0.02 0 0" range="-3 3" limited="true"/>
        <joint name="pitch" type="hinge" axis="0 1 0" pos="-0.02 0 0" range="-3 3" limited="true"/>
        {inertial("plunger_pin")}
        <geom name="bore" type="cylinder" fromto="{-L:.5f} 0 0 -0.001 0 0" size="{PR:.5f}" {CT["bore"]}/>
        <geom type="cylinder" fromto="{-L:.5f} 0 0 {-rl:.5f} 0 0" size="{PR:.5f}" {CT["face"]}/>
        <geom type="cylinder" fromto="{-rl:.5f} 0 0 0 0 0" size="{PR - rh:.5f}" {CT["face"]}/>
        <geom type="box" pos="{-rl / 2:.5f} 0 {PR - rh / 2 - 0.0015:.5f}" euler="0 {math.degrees(math.atan2(rh, rl)):.3f} 0" size="{rl / 2:.5f} 0.012 0.0015" {CT["face"]}/>
        <geom name="pinN" type="cylinder" fromto="{pr:.5f} {B.PIN_Y[0] * mm:.5f} 0 {pr:.5f} -0.018 0" size="{pin_r:.5f}" {CT["pin"]}/>
        <geom name="pinP" type="cylinder" fromto="{pr:.5f} 0.018 0 {pr:.5f} {B.PIN_Y[1] * mm:.5f} 0" size="{pin_r:.5f}" {CT["pin"]}/>
        {lug_xml}
        <site name="grooveN" pos="{pr:.5f} {-sum(B.BAND_Y) / 2 * mm:.5f} 0"/><site name="grooveP" pos="{pr:.5f} {sum(B.BAND_Y) / 2 * mm:.5f} 0"/>
      </body>
      <body name="lever" pos="{f3(LP)}">
        <joint name="lever" type="hinge" axis="0 1 0" armature="{SERVOS['DS3218']['J']}" damping="0.001"/>
        {inertial("lever", (B.LEVER_PIV[0], 0, B.LEVER_PIV[1]))}
        <geom type="box" pos="{f3(lev_c)}" euler="0 {-B.LEVER_REST_DEG:.3f} 0" size="{B.LEVER_LEN / 2 * mm:.5f} {(ly1 - ly0) / 2:.5f} {B.LEVER_W / 2 * mm:.5f}" {CT["arm"]}/>
        <geom type="cylinder" fromto="{lev_t[0]:.5f} {ly0:.5f} {lev_t[2]:.5f} {lev_t[0]:.5f} {ly1:.5f} {lev_t[2]:.5f}" size="{B.LEVER_W / 2 * mm:.5f}" {CT["arm"]}/>
        <geom type="cylinder" fromto="0 {ly0:.5f} 0 0 {ly1:.5f} 0" size="0.0125" {CT["arm"]}/>
        {finger_xml}
      </body>
      <body name="sear" pos="{f3(SP * mm)}">
        <joint name="sear" type="hinge" axis="0 1 0" armature="1e-7" damping="2e-5"/>
        {inertial("sear", (B.SEAR_PIV[0], 0, B.SEAR_PIV[1]))}
        {sear_xml}
        <site name="searPeg" pos="{rel(*B.SEAR_PEG)}"/>
      </body>
      {stop_xml}
      <site name="searPost" pos="{f3((B.SEAR_POST[0] * mm, 0, B.SEAR_POST[1] * mm))}"/>
      <site name="hookN" pos="{HOOK_X * mm:.5f} {-sum(B.BAND_Y) / 2 * mm:.5f} 0"/><site name="hookP" pos="{HOOK_X * mm:.5f} {sum(B.BAND_Y) / 2 * mm:.5f} 0"/>
    </body>
    {balls}
  </worldbody>
  <actuator><motor name="lever" joint="lever"/><motor name="tilt" joint="tilt"/></actuator>
</mujoco>"""
    return xml

class Servo:
    """취미용 서보: 출력축 각도 비례 제어, 토크 = 멈춤 토크 x (1 - 속도/무부하 속도), 기어 관성은 관절 armature."""
    def __init__(self, spec, scale=1.0, kp_deg=4.0):
        self.tau = spec["tau"] * scale; self.w0 = spec["w0"]; J = spec["J"] + 2e-6
        self.kp = self.tau / math.radians(kp_deg); self.kd = 1.4 * math.sqrt(self.kp * J)
        self.peak = 0.0
    def torque(self, q, qd, target):
        t = self.kp * (target - q) - self.kd * qd
        lim = self.tau * max(0.0, 1 - abs(qd) / self.w0) if t * qd > 0 else self.tau
        t = max(-lim, min(lim, t)); self.peak = max(self.peak, abs(t) / self.tau)
        return t

class TiltServo(Servo):
    """각도 서보: 모터 쪽 회전자(관성 J)와 발사대 축을 기어 유격 ±bl/2 + 탄성 kg 로 잇는다."""
    def __init__(self, spec, scale=1.0, backlash_deg=0.8, kg=60.0):
        super().__init__(spec, scale); self.J = spec["J"]; self.kp = self.tau / math.radians(4.0)
        self.kd = 1.4 * math.sqrt(self.kp * self.J); self.bl = math.radians(backlash_deg) / 2; self.kg = kg; self.cg = 2 * 0.3 * math.sqrt(kg * self.J)
        self.phi = 0.0; self.phid = 0.0; self.max_coupling = 0.0
    def step(self, dt, q, qd, target):
        tm = self.torque(q, self.phid, target)           # 위치 센서(가변 저항)는 출력축
        dlt = self.phi - q
        dead = math.copysign(max(0.0, abs(dlt) - self.bl), dlt)
        tc = self.kg * dead + (self.cg * (self.phid - qd) if dead != 0 else 0.0)
        self.phid += dt * (tm - tc) / self.J; self.phi += dt * self.phid
        self.max_coupling = max(self.max_coupling, abs(tc))
        return tc

def run(theta=15.0, n_band=1, k_scale=1.0, tau_scale=1.0, shots=5, hyst=0.0, timing=None, record=False, cock_theta=None, tilt_rate=40.0, lever_ret_rate=200.0, on_step=None):
    """cock_theta: 장전(공 내려놓기)을 이 각도에서 하고, 걸쇠가 잡은 뒤 theta 로 내려 쏜다(낮은 각도용)."""
    cock_theta = theta if cock_theta is None else max(theta, cock_theta)
    # tilt_rate: 각도 명령을 초당 이 각도만큼 천천히 바꾼다(한 번에 바꾸면 서보가 확 돌아 장전된 공이 튀어 나감)
    tm_ = dict(cock=0.60, ret=0.45, fire=0.40)          # 장전 뒤 0.6 s(공이 떨어지고 레버가 힘을 덜 씀)
    if timing: tm_.update(timing)
    xml = build_xml(cock_theta, shots)
    m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    Rl = d.xmat[m.body("launcher").id].reshape(3, 3).copy(); Ol = d.xpos[m.body("launcher").id].copy()
    W = lambda p: Ol + Rl @ (np.asarray(p) * mm)
    nb = shots; bid = [m.body(f"ball{i}").id for i in range(nb)]
    badr = [m.jnt_qposadr[m.body_jntadr[b]] for b in bid]; bdof = [m.jnt_dofadr[m.body_jntadr[b]] for b in bid]
    for i in range(nb):
        d.qpos[badr[i]:badr[i] + 3] = W((B.HOP_CX, 0, B.BORE_R + 20.2 + 40.4 * i)); d.qpos[badr[i] + 3:badr[i] + 7] = (1, 0, 0, 0)
    j = {n: m.joint(n).id for n in ("tilt", "slide", "lever", "sear", "yaw", "pitch")}
    qa = {n: m.jnt_qposadr[i] for n, i in j.items()}; va = {n: m.jnt_dofadr[i] for n, i in j.items()}
    act = {n: m.actuator(n).id for n in ("lever", "tilt")}
    lid, pid = m.body("launcher").id, m.body("plunger").id
    site = {n: m.site(n).id for n in ("grooveN", "grooveP", "hookN", "hookP", "searPeg", "searPost")}
    sid = m.body("sear").id; stop_g = m.geom("searStop").id; tooth_g = {m.geom("tooth").id, m.geom("ramp").id}
    pinP = m.geom("pinP").id; fing_g = m.geom("finger").id; tail_g = m.geom("tail").id
    sear_arm_g = {m.geom("nose").id}
    def cforce(ga, gb_set):
        f6 = np.zeros(6); tot = 0.0
        for c_i in range(d.ncon):
            c = d.contact[c_i]
            if (c.geom1 == ga and c.geom2 in gb_set) or (c.geom2 == ga and c.geom1 in gb_set):
                mujoco.mj_contactForce(m, d, c_i, f6); tot += abs(f6[0])
        return tot
    stopF = {m.geom(f"stopF{s}").id for s in (-1, 1)}
    pinG = {m.geom("lug-1").id, m.geom("lug1").id}
    k_side = n_band * K_LOOP * k_scale; d_free = (HOOK_X - B.PIN_REST) * mm - PRE
    lev = Servo(SERVOS["DS3218"], tau_scale); tilt = TiltServo(SERVOS["MG996R"], tau_scale)
    k_sear = K_SEAR * k_scale; sear_peak = 0.0
    tilt.phi = 0.0
    q_trip = math.radians(B.LEVER_TRIP_DEG); q_cock = math.radians(B.LEVER_COCK_DEG)
    tgt = dict(lever=0.0, tilt=0.0)
    q_aim = math.radians(cock_theta - theta)                       # 경첩 + = 총구 내림
    tilt_cmd = 0.0; lever_cmd = 0.0
    # 시간표(실제 코드와 같은 열린 제어)
    # 한 발: [총구 올림(장전 각도)] → 장전 → 레버 천천히 복귀(걸쇠에 걸림) → [총구 내림 + 멈춤] → 레버 발사 각(손가락이 걸쇠를 엶)
    d_tilt = abs(cock_theta - theta) / tilt_rate + (0.15 if cock_theta != theta else 0.0)
    sched = []; t = 0.35
    for k in range(shots):
        if k > 0 and d_tilt: sched.append((t, "tilt", 0.0, f"up{k}")); t += d_tilt
        t1 = t + tm_["cock"]; t3 = t1 + tm_["ret"]; t4 = t3 + d_tilt
        sched += [(t, "lever", q_cock, f"cock{k}"), (t1, "lever", 0.0, f"ret{k}"),
                  (t3, "tilt", q_aim, f"aim{k}"), (t4, "lever", q_trip, f"fire{k}")]
        t = t4 + tm_["fire"]
    t_end = t + 0.6
    cycle = tm_["cock"] + tm_["ret"] + tm_["fire"] + 2 * d_tilt
    hold_x = B.SEAR_FACE_X - B.PIN_R; trip_win = None
    log = [dict() for _ in range(shots)]; si = 0
    fired = {}; tracks = {i: [] for i in range(nb)}
    muz = B.X_MUZ * mm
    prev_band = [0.0, 0.0]; band_peak = 0.0
    impact = dict(v=0.0); stop_force = []; cur_shot = -1; tilt_dev = []; contact_buf = np.zeros(6)
    rec = [] if record else None
    dt = m.opt.timestep
    while d.time < t_end:
        tnow = d.time
        while si < len(sched) and tnow >= sched[si][0]:
            _, who, val, tag = sched[si]; tgt[who] = val; si += 1
            kind = tag.rstrip("0123456789"); k = int(tag[len(kind):])
            xpin = B.PIN_REST + d.qpos[qa["slide"]] * 1000
            if kind == "ret":                                          # 장전 끝(레버 되돌리기 직전)
                log[k]["pin_x_at_cock_mm"] = round(xpin, 2)
                log[k]["sear_deg_at_cock"] = round(math.degrees(d.qpos[qa["sear"]]), 2)
                log[k]["sear_behind_pin"] = bool(xpin + B.PIN_R < B.SEAR_FACE_X - 0.2 and abs(math.degrees(d.qpos[qa["sear"]])) < 2.0)
                log[k]["lever_deg_at_cock"] = round(math.degrees(d.qpos[qa["lever"]]), 1)
                log[k]["lever_hold_torque_frac"] = round(abs(d.ctrl[act["lever"]]) / lev.tau, 3)
                log[k]["ball_in_barrel"] = in_barrel(m, d, bid, lid, Ol)
            elif kind == "fire":
                log[k]["pin_x_held_mm"] = round(xpin, 2)
                log[k]["sear_deg_held"] = round(math.degrees(d.qpos[qa["sear"]]), 2)
                log[k]["held_by_sear"] = bool(abs(xpin - hold_x) < 1.0 and abs(math.degrees(d.qpos[qa["sear"]])) < 2.0)
                log[k]["sear_pin_force_N"] = round(cforce(pinP, tooth_g), 2)
                log[k]["sear_stop_force_N"] = round(cforce(stop_g, sear_arm_g), 2)
                log[k]["lever_deg_at_fire"] = round(math.degrees(d.qpos[qa["lever"]]), 1)
                Rn_ = d.xmat[lid].reshape(3, 3); On_ = d.xpos[lid]; face = B.PIN_REST - B.PIN_R * 0 + (xpin - B.PIN_REST) + 30.0
                gaps = [round((Rn_.T @ (d.xpos[b_] - On_))[0] * 1000 - 20.0 - face, 2) for b_ in bid
                        if abs((Rn_.T @ (d.xpos[b_] - On_))[2] * 1000) < 6 and B.X_REAR < (Rn_.T @ (d.xpos[b_] - On_))[0] * 1000 < B.X_MUZ]
                log[k]["ball_gap_to_plunger_mm"] = min(gaps) if gaps else None
                trip_win = (tnow, k)
                log[k]["tilt_hold_torque_frac"] = round(abs(tilt.torque(d.qpos[qa["tilt"]], tilt.phid, tilt_cmd)) / tilt.tau, 3)
                cur_shot = k; impact["v"] = 0.0; log[k]["_t_fire"] = tnow
                tilt_dev = []
        # 서보
        if tgt["lever"] == 0.0 and lever_cmd > 0.0:                  # 장전 → 쉼 복귀는 천천히(플런저가 걸쇠에 살며시 닿게)
            lever_cmd = max(0.0, lever_cmd - math.radians(lever_ret_rate) * dt)
        else:
            lever_cmd = tgt["lever"]
        d.ctrl[act["lever"]] = lev.torque(d.qpos[qa["lever"]], d.qvel[va["lever"]], lever_cmd)
        if trip_win and tnow - trip_win[0] < 0.2:
            kk = trip_win[1]
            log[kk]["trip_lever_peak_torque_frac"] = round(max(log[kk].get("trip_lever_peak_torque_frac", 0), abs(d.ctrl[act["lever"]]) / lev.tau), 3)
            log[kk]["trip_finger_peak_N"] = round(max(log[kk].get("trip_finger_peak_N", 0), cforce(fing_g, {tail_g})), 2)
            log[kk]["sear_max_open_deg"] = round(max(log[kk].get("sear_max_open_deg", 0), math.degrees(d.qpos[qa["sear"]])), 1)
        step_max = math.radians(tilt_rate) * dt
        tilt_cmd += max(-step_max, min(step_max, tgt["tilt"] - tilt_cmd))
        d.ctrl[act["tilt"]] = tilt.step(dt, d.qpos[qa["tilt"]], d.qvel[va["tilt"]], tilt_cmd)
        # 고무줄
        d.qfrc_applied[:] = 0
        for n, (g, h) in enumerate((("grooveN", "hookN"), ("grooveP", "hookP"))):
            pg, ph = d.site_xpos[site[g]], d.site_xpos[site[h]]
            v = ph - pg; L = np.linalg.norm(v); ext = max(0.0, L - d_free)
            F = k_side * ext
            if hyst and L < prev_band[n]: F *= (1 - hyst)
            prev_band[n] = L
            f = F * v / L
            mujoco.mj_applyFT(m, d, f, np.zeros(3), pg, pid, d.qfrc_applied)
            mujoco.mj_applyFT(m, d, -f, np.zeros(3), ph, lid, d.qfrc_applied)
            band_peak = max(band_peak, F)
        pg, ph = d.site_xpos[site["searPeg"]], d.site_xpos[site["searPost"]]          # 걸쇠 고무줄
        v = ph - pg; L = np.linalg.norm(v); Fs = k_sear * max(0.0, L - SEAR_FREE); sear_peak = max(sear_peak, Fs)
        mujoco.mj_applyFT(m, d, Fs * v / L, np.zeros(3), pg, sid, d.qfrc_applied)
        mujoco.mj_applyFT(m, d, -Fs * v / L, np.zeros(3), ph, lid, d.qfrc_applied)
        mujoco.mj_step(m, d)
        if on_step: on_step(m, d, tgt)          # 녹화(sim_gif.py)
        vx = d.qvel[va["slide"]]
        if cur_shot >= 0:
            impact["v"] = max(impact["v"], vx)
            if tnow - log[cur_shot]["_t_fire"] < 0.4:
                log[cur_shot]["tilt_peak_torque_frac_shot"] = round(max(log[cur_shot].get("tilt_peak_torque_frac_shot", 0), abs(tilt.torque(d.qpos[qa["tilt"]], tilt.phid, tilt_cmd)) / tilt.tau), 3)
                log[cur_shot]["tilt_dev_max_deg"] = round(max(log[cur_shot].get("tilt_dev_max_deg", 0), abs(math.degrees(d.qpos[qa["tilt"]] - tilt_cmd))), 3)
            tilt_dev.append(d.qpos[qa["tilt"]] - tilt_cmd)
            for c_i in range(d.ncon):
                c = d.contact[c_i]
                if (c.geom1 in stopF and c.geom2 in pinG) or (c.geom2 in stopF and c.geom1 in pinG):
                    if "impact_v_mps" not in log[cur_shot]:
                        log[cur_shot]["impact_v_mps"] = round(impact["v"], 2)
                        log[cur_shot]["max_tilt_dev_deg_to_impact"] = round(math.degrees(max(map(abs, tilt_dev))), 3)
        # 공
        Rn = d.xmat[lid].reshape(3, 3); On = d.xpos[lid]
        for i in range(nb):
            pw = d.xpos[bid[i]]
            if i not in fired:
                rel = Rn.T @ (pw - On)
                if rel[0] > muz + 0.02 and abs(rel[2]) < 0.06:
                    vv = d.qvel[bdof[i]:bdof[i] + 3]; sp = float(np.linalg.norm(vv))
                    elev = math.degrees(math.atan2(vv[2], vv[0]))
                    k = cur_shot if cur_shot >= 0 else -1
                    fired[i] = k
                    if k >= 0 and "ball" not in log[k]:
                        log[k].update(ball=i, exit_speed_mps=round(sp, 3), exit_elev_deg=round(elev, 2),
                                      spin_rad_s=round(float(d.qvel[bdof[i] + 4]), 1),
                                      tilt_at_exit_deg=round(math.degrees(d.qpos[qa["tilt"]]), 3),
                                      t_exit_after_fire_ms=round((tnow - log[k]["_t_fire"]) * 1000, 1))
                    elif k >= 0:
                        log[k].setdefault("extra_balls", []).append(i)
            elif int(tnow / dt) % 10 == 0 and pw[2] > 0.03:
                tracks[i].append((tnow, pw[0], pw[2]))
        if record and int(tnow / dt) % 20 == 0:
            rec.append((tnow, d.qpos[qa["tilt"]], d.qpos[qa["slide"]], d.qpos[qa["lever"]], d.qpos[qa["sear"]], d.qvel[va["slide"]]))
    # 결과 정리
    popped = [i for i in range(nb) if i not in fired]
    for k, s in enumerate(log):
        s.pop("_t_fire", None)
        i = s.get("ball")
        if i is None: continue
        tr = np.array(tracks[i])
        if len(tr) > 3:
            z = tr[:, 2]; down = np.r_[False, np.diff(z) < 0]
            ok = (z > TARGET[0]) & (z < TARGET[1]) & down
            if ok.any():
                xs_ = tr[ok, 1] - BASE_LINK_X
                s["hit_window_m"] = [round(float(xs_.min()), 3), round(float(xs_.max()), 3)]; s["hit_mid_m"] = round(float(xs_.mean()), 3)
            s["apex_m"] = round(float(z.max()), 3)
    ok_all = all(s.get("sear_behind_pin") and s.get("held_by_sear") and s.get("ball_in_barrel") and (s.get("exit_speed_mps") or 0) > 2.0 and not s.get("extra_balls") for s in log)
    out = dict(theta=theta, cock_theta=cock_theta, mu=MU, n_band=n_band, k_scale=k_scale, tau_scale=tau_scale, hyst=hyst, timing=tm_, cycle_s=round(cycle, 2),
               all_ok=bool(ok_all), balls_not_fired=popped, shots=log,
               lever_peak_torque_frac=round(lev.peak, 3), sear_band_peak_N=round(sear_peak, 2),
               tilt_peak_torque_frac=round(tilt.peak, 3), tilt_peak_coupling_Nm=round(tilt.max_coupling, 3),
               band_peak_N_per_side=round(band_peak, 2))
    if record: out["rec"] = rec
    return out

def in_barrel(m, d, bid, lid, Ol):
    R = d.xmat[lid].reshape(3, 3); O = d.xpos[lid]
    for b in bid:
        rel = R.T @ (d.xpos[b] - O) * 1000
        if abs(rel[2]) < 5 and B.X_REAR < rel[0] < B.X_MUZ: return True
    return False

def summary(r):
    sp = [s.get("exit_speed_mps") for s in r["shots"]]; hm = [s.get("hit_mid_m") for s in r["shots"]]
    return (f"θ{r['theta']:>4} k x{r['k_scale']:.2f} τ x{r['tau_scale']:.2f}  ok={r['all_ok']}  v={sp}  hit={hm}  "
            f"held={[s.get('pin_x_held_mm') for s in r['shots']]} trip {max((s.get('trip_lever_peak_torque_frac') or 0) for s in r['shots'])} lever_hold {max((s.get('lever_hold_torque_frac') or 0) for s in r['shots'])} "
            f"tilt_shot {max((s.get('tilt_peak_torque_frac_shot') or 0) for s in r['shots'])} dev {max((s.get('tilt_dev_max_deg') or 0) for s in r['shots'])}")

if __name__ == "__main__":
    B.DT = 1e-4
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        r = run(15.0); print(json.dumps(r, indent=1, ensure_ascii=False)); sys.exit()
    if len(sys.argv) > 1 and sys.argv[1] == "core":            # 핵심 부품 실물 시험과 같은 조건: 15° 고정, 3 번 장전-발사
        out = []
        for c in (dict(), dict(k_scale=1.25), dict(k_scale=0.75), dict(tau_scale=0.5)):
            r = run(15.0, shots=3, **c); out.append(r); print(summary(r), flush=True)
        json.dump(out, open(os.path.join(D, "sim_core.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False); sys.exit()
    out = []
    cases = [dict(), dict(k_scale=1.25), dict(k_scale=0.75), dict(tau_scale=0.5), dict(k_scale=1.25, tau_scale=0.5)]
    for th in (5.0, 15.0, 30.0):
        for c in cases:
            r = run(th, cock_theta=COCK_MIN, **c); out.append(r); print(summary(r), flush=True)
    # 각도-거리 표용 나머지 각도(기본 경우)
    for th in (10.0, 20.0, 25.0):
        r = run(th, cock_theta=COCK_MIN); out.append(r); print(summary(r), flush=True)
    # 견고성: PLA 마찰 0.5, 고무 히스테리시스 15 %
    MU = 0.5; r = run(15.0); out.append(r); print("mu 0.5", summary(r), flush=True)
    MU = 0.35; r = run(15.0, hyst=0.15); out.append(r); print("hyst 0.15", summary(r), flush=True)
    json.dump(out, open(os.path.join(D, "sim_full.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
