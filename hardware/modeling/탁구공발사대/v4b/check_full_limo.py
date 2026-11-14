"""LIMO 위 완성품 전체 겹침 검사(지붕 좌표): 발사대 전 부품(5~30°)+공 통 고리·잠금 뚜껑·손나사
↔ 열화상·레이더 팔(뚜껑·볼트) ↔ 초음파 받침 ↔ 옆 전장 판 ↔ LIMO 몸체 메시. 공 날아가는 길(지름 40)과 거리."""
import os, sys, math
import numpy as np, manifold3d as m3, trimesh
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D); sys.path.insert(0, os.path.join(D, "..", "..", "lib"))
import build_launcher as B, build_cradle as C
src = open(os.path.join(D, "build_hopper_lock.py"), encoding="utf-8").read().replace('if __name__ == "__main__":', 'if False:')
H = {"__file__": os.path.join(D, "build_hopper_lock.py")}; exec(compile(src, "H", "exec"), H)
L = os.path.join(D, "..", "..", "LIMO센서", "stl")
ld = lambda p: (lambda t: m3.Manifold(m3.Mesh(t.vertices.astype(np.float32), t.faces.astype(np.uint32))))(trimesh.load(p))
vol = lambda a: round(a.volume(), 2)

# LIMO 메시 → 지붕 좌표: x = -y + 14.5, y = x, z = z - 145.5
lt = trimesh.load(os.path.join(D, "ref", "limo", "limo_base.stl"))
v = lt.vertices; lt = trimesh.Trimesh(np.c_[-v[:, 1] + 14.5, v[:, 0], v[:, 2] - 145.5], lt.faces)
limo = m3.Manifold(m3.Mesh(lt.vertices.astype(np.float32), lt.faces.astype(np.uint32)))

mounts = {n: ld(os.path.join(L, f)) for n, f in (("arm", "asm_arm.stl"), ("arm_holder", "asm_arm_holder.stl"), ("arm_caps", "asm_arm_caps.stl"), ("arm_bolts", "asm_arm_bolts.stl"),
          ("sonar", "asm_sonar.stl"),
          ("pod_r", "asm_pod_right.stl"), ("pod_r_lid", "asm_pod_right_lid.stl"), ("pod_r_bolts", "asm_pod_right_bolts.stl"),
          ("pod_l", "asm_pod_left.stl"), ("pod_l_lid", "asm_pod_left_lid.stl"), ("pod_l_bolts", "asm_pod_left_bolts.stl"))}
# LIMO 메시는 닫힌 몸체가 아니라 겹침 부피를 못 잼 → 촘촘한 표면 점(300만)과의 최소 거리. 지붕·옆판·밑면 나사로 붙는 면은 0 근처가 정상
from scipy.spatial import cKDTree
LP, _ = trimesh.sample.sample_surface(lt, 3000000); KT = cKDTree(LP)
for n, m in mounts.items():
    me = m.to_mesh(); tm = trimesh.Trimesh(me.vert_properties[:, :3], me.tri_verts)
    p, _ = trimesh.sample.sample_surface(tm, 150000); p = np.r_[tm.vertices, p]
    d, _ = KT.query(p); k = d.argmin()
    print(f"{n:11s} LIMO 최소 거리 {d[k]:.2f} at {p[k].round(1)}")
names = list(mounts)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        x = vol(mounts[names[i]] ^ mounts[names[j]])
        if x > 0.01: print("  겹침", names[i], names[j], x)

body = B.build_body(); lever0, sear0, spin0 = B.build_lever(), B.build_sear(), B.sear_pin_at()
sv_lever = B.lever_servo_place(B.servo_solid(B.SERVO_L, horn=False))
tube = B.build_hopper_tube()
hp, hn = H["halves"](); asm_b, *_ = H["bolts_nuts"]()
capl, reach = H["cap_locked"](); ri = B.BORE_R + B.HOP_WALL + 0.3
thumb = H["thumb_for_cap"](reach).rotate([0, 0, 270]).rotate([0, -90, 0]).translate([ri + reach, 0, 4.0])
capw = (capl + thumb).translate([B.HOP_CX, 0, B.HOP_TOP - 8.0])
balls = m3.Manifold()
for i in range(B.N_HOPPER):
    balls += m3.Manifold.sphere(20.0, 48).translate([B.HOP_CX, 0, B.BORE_R + 0.5 + 20 + 40.4 * i])
bx = B.PIN_COCK - B.PIN_REST + 20 + 0.5
launcher = body + sv_lever + spin0 + tube + hp + hn + asm_b + capw + balls + B.lever_at(0, lever0) + B.lever_at(B.LEVER_COCK_DEG, lever0) + B.sear_at(0, sear0)
path = m3.Manifold.cylinder(800, 20, 20, 48).rotate([0, 90, 0]).translate([bx + 30, 0, -0.8])   # 총구 밖 공 길
cradle = C.build_cradle(); tilt_sv = C.tilt_servo_place(B.servo_solid(B.SERVO_K, horn=False))
allm = m3.Manifold()
for m in mounts.values(): allm += m
print("cradle∩mounts", vol(cradle ^ allm), " tilt servo∩mounts", vol(tilt_sv ^ allm))
for th in (5.0, 10.0, 15.0, 20.0, 25.0, 30.0):
    lw = C.pose(launcher, th); pw = C.pose(path, th)
    d = trimesh.proximity.closest_point if False else None
    pts = np.asarray(allm.to_mesh().vert_properties[:, :3])
    # 공 길 축과 받침 점들의 최소 거리 - 20
    o = np.array(C.pose(m3.Manifold.sphere(0.01, 8).translate([bx + 30, 0, -0.8]), th).bounding_box()[:3]) + 0.005
    e = np.array(C.pose(m3.Manifold.sphere(0.01, 8).translate([bx + 830, 0, -0.8]), th).bounding_box()[:3]) + 0.005
    u = (e - o) / np.linalg.norm(e - o); w = pts - o; t = np.clip(w @ u, 0, None)
    gap = np.min(np.linalg.norm(w - np.outer(t, u), axis=1)) - 20
    print(f"{th:4.1f}° 발사대∩받침 {vol(lw ^ allm)}  공 길∩받침 {vol(pw ^ allm)}  공 길 최소 틈 {gap:.1f}")

# 그림용 내보내기(15°)
OUTR = os.path.join(D, "..", "..", "LIMO센서", "render", "asm"); os.makedirs(OUTR, exist_ok=True)
th = 15.0
def ex(m, n):
    me = m.to_mesh(); trimesh.Trimesh(me.vert_properties[:, :3], me.tri_verts).export(os.path.join(OUTR, n + ".stl"))
ex(C.pose(body + spin0 + B.lever_at(0, lever0) + B.sear_at(0, sear0) + tube, th), "launcher")
ex(C.pose(sv_lever, th) + tilt_sv, "servos")
ex(C.pose(hp + hn + asm_b + capw, th), "locks")
ex(C.pose(balls, th), "balls")
ex(cradle, "cradle")
lt.export(os.path.join(OUTR, "limo.stl"))
