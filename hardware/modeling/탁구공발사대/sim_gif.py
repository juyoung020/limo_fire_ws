"""sim_full 의 실제 시뮬을 그대로 돌리며 화면을 찍어 GIF 로 만든다.
장면 1: 15° 에서 장전 → 30° 로 들어 발사.  장면 2: 15° 에서 장전 → 5° 로 내려 발사(2발, 재장전 포함).
발사 순간 앞뒤 80 ms 는 느리게(0.002 s 마다 한 장).
사용: python sim_gif.py  → render/sim_launch.gif"""
import os, math
import numpy as np, mujoco
from PIL import Image, ImageDraw, ImageFont
import sim_full as S
S.B.DT = 1e-4                      # sim_full 본 실행과 같은 시간 간격

D = os.path.dirname(os.path.abspath(__file__))
W, H = 960, 600
FONT = ImageFont.truetype(r"C:\Windows\Fonts\malgun.ttf", 20)
SMALL = ImageFont.truetype(r"C:\Windows\Fonts\malgun.ttf", 15)
COL = dict(plunger=(0.15, 0.75, 0.25, 1), lever=(1.0, 0.55, 0.1, 1), sear=(0.9, 0.15, 0.15, 1))

def scene(theta, cock_theta, shots, title, dist=0.48, look=(0.07, 0.0, 0.02), step_fast=0.04, elev=-6, t_end=None, az=90):
    frames = []
    st = dict(r=None, cam=None, last=-1.0, fire_t=[], init=False, prev_lever=0.0)

    def on_step(m, d, tgt):
        if not st["init"]:
            for g in range(m.ngeom):
                b = m.body(m.geom_bodyid[g]).name
                if b in COL: m.geom_rgba[g] = COL[b]
                elif b.startswith("ball"): m.geom_rgba[g] = (1.0, 0.6, 0.1, 1)
                elif b == "launcher" and m.geom_contype[g] in (32, 256): m.geom_rgba[g] = (0.5, 0.5, 0.55, 0.6)
                elif b == "world": m.geom_rgba[g] = (0.35, 0.35, 0.38, 1)
            m.vis.global_.offwidth, m.vis.global_.offheight = W, H
            st["r"] = mujoco.Renderer(m, H, W)
            cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            lid = m.body("launcher").id
            ax_w = d.xpos[lid] + d.xmat[lid].reshape(3, 3) @ (np.array([S.B.AX[0], 0, S.B.AX[1]]) * S.mm)
            st["ax_w"] = ax_w.copy()
            cam.lookat[:] = (ax_w[0] + look[0], look[1], ax_w[2] + look[2]); cam.distance = dist; cam.azimuth = az; cam.elevation = elev
            st["cam"] = cam; st["init"] = True
        t = d.time
        if tgt["lever"] < 0 and st["prev_lever"] >= 0: st["fire_t"].append(t)
        st["prev_lever"] = tgt["lever"]
        slow = any(-0.005 < t - ft < 0.08 for ft in st["fire_t"])
        step = 0.002 if slow else step_fast
        if t - st["last"] < step or (t_end and t > t_end): return
        st["last"] = t
        r = st["r"]; r.update_scene(d, camera=st["cam"])
        def add(typ, size, pos, rgba, mat=np.eye(3)):
            if r.scene.ngeom >= r.scene.maxgeom: return None
            g = r.scene.geoms[r.scene.ngeom]
            mujoco.mjv_initGeom(g, typ, np.asarray(size, float), np.asarray(pos, float), np.asarray(mat, float).flatten(), np.array(rgba, dtype=np.float32))
            r.scene.ngeom += 1; return g
        # 고무줄(노랑 선): 핀 홈 ↔ 고리, 걸쇠 갈고리 ↔ 기둥
        for a, b in (("grooveN", "hookN"), ("grooveP", "hookP"), ("searPeg", "searPost")):
            g = add(mujoco.mjtGeom.mjGEOM_CAPSULE, (0, 0, 0), (0, 0, 0), (1, 0.9, 0, 1))
            if g is not None: mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0015, d.site(a).xpos, d.site(b).xpos)
        # LIMO 지붕·받침 기둥·각도 서보(MG996R) — 그림만(시뮬엔 각도 축 관절과 서보 모델로 들어 있음)
        ax_w = st["ax_w"]; roof = S.ROOF_Z
        add(mujoco.mjtGeom.mjGEOM_BOX, (0.10, 0.06, 0.004), (ax_w[0] + 0.02, 0, roof - 0.004), (0.2, 0.2, 0.22, 1))
        add(mujoco.mjtGeom.mjGEOM_BOX, (0.015, 0.003, (ax_w[2] - roof) / 2), (ax_w[0], 0.05, (ax_w[2] + roof) / 2), (0.3, 0.3, 0.33, 1))
        add(mujoco.mjtGeom.mjGEOM_BOX, (0.020, 0.010, 0.0275), (ax_w[0] + 0.01, 0.06, ax_w[2] - 0.005), (0.15, 0.3, 0.85, 1))
        tq = d.joint("tilt").qpos[0]; c, s_ = math.cos(-tq), math.sin(-tq)
        # 각도 서보 혼(축과 같이 도는 막대)
        add(mujoco.mjtGeom.mjGEOM_BOX, (0.016, 0.004, 0.002), ax_w + np.array([0.0, 0.072, 0.0]), (1, 1, 1, 1),
            np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]]))
        add(mujoco.mjtGeom.mjGEOM_CYLINDER, (0.004, 0.035, 0), ax_w + np.array([0, 0.03, 0]), (0.9, 0.8, 0.1, 1),
            np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]))
        img = Image.fromarray(r.render())
        dr = ImageDraw.Draw(img)
        tilt = theta_cmd = math.degrees(d.joint("tilt").qpos[0])
        ang = cock_theta - tilt
        lev = math.degrees(d.joint("lever").qpos[0])
        tv = abs(math.degrees(d.joint("tilt").qvel[0]))
        since_fire = min((t - ft for ft in st["fire_t"] if t >= ft), default=9)
        if tgt["lever"] > 0: ph = "① 장전: 레버가 핀(플런저)을 뒤로 민다 → 걸쇠가 핀을 잡음"
        elif tgt["lever"] < 0 and since_fire < 0.25: ph = "③ 발사: 레버 손가락이 걸쇠를 눌러 열고 고무줄이 플런저를 당김"
        elif tv > 3: ph = "② 각도 조절: 각도 서보(파랑)가 축을 돌려 총신을 기울임"
        elif tgt["lever"] == 0 and lev > 1: ph = "레버 복귀(핀은 걸쇠에 걸린 채, 공이 플런저 앞으로 내려옴)"
        else: ph = "대기"
        dr.rectangle((0, 0, W, 64), fill=(20, 20, 24))
        dr.text((10, 4), f"{title}", font=FONT, fill=(255, 255, 255))
        dr.text((10, 34), f"t {t:5.2f}s   발사대 각도 {ang:5.1f}°   레버 {lev:5.1f}°   {ph}" + ("   [느린 화면]" if slow else ""), font=SMALL, fill=(255, 220, 120))
        frames.append((img, 20 if slow else 40))

    S.run(theta=theta, cock_theta=cock_theta, shots=shots, on_step=on_step)
    return frames

if __name__ == "__main__":
    fr = scene(30.0, 30.0, 1, "장면 1: 30° 발사 (장전 → 걸쇠 잡음 → 발사)")
    fr += scene(5.0, 15.0, 2, "장면 2: 15°에서 장전 → 5°로 내려 발사, 2발 연속")
    imgs = [f[0].convert("P", palette=Image.ADAPTIVE, colors=128) for f in fr]
    out = os.path.join(D, "render", "sim_launch.gif")
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=[f[1] for f in fr], loop=0, optimize=True)
    print(out, len(imgs), "frames", round(os.path.getsize(out) / 1e6, 1), "MB")
