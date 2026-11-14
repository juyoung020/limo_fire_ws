"""바퀴(공식 URDF 메시)와 출력 부품의 틈: 앞바퀴는 조향 −30~+30° 를 5° 간격으로 돌려 가며 잰다."""
import os, numpy as np, trimesh, manifold3d as m3
from scipy.spatial import cKDTree
D = os.path.dirname(os.path.abspath(__file__))
P = np.load(os.path.join(D, "ref", "limo_wheel", "limo_wheel_pts.npy")) * 1000.0
# 회전체 윤곽(축 y): y 칸마다 최대 반지름 → 회전체
ys = np.arange(P[:, 1].min(), P[:, 1].max() + 0.5, 0.5); r = np.hypot(P[:, 0], P[:, 2])
prof = [(y, r[np.abs(P[:, 1] - y) <= 2.0].max()) for y in ys if (np.abs(P[:, 1] - y) <= 2.0).any()]   # 꼭짓점이 듬성해 ±2 mm 안의 최대(조금 크게 잡힘)
print("바퀴 윤곽: y(조향 축 기준) → 반지름", [(round(y, 0), round(rr, 1)) for y, rr in prof[::10]])
wheel = m3.Manifold()
for (y0, r0), (y1, r1) in zip(prof[:-1], prof[1:]):
    wheel += m3.Manifold.cylinder(y1 - y0 + 0.01, r0, r1, 72).rotate([-90, 0, 0]).translate([0, y0, 0])
ZC = -96.5
def at(front, left, deg):
    w = wheel if left else wheel.mirror([0, 1, 0])
    return w.rotate([0, 0, deg]).translate([114.5 if front else -85.5, 70 if left else -70, ZC])
def ld(f):
    t = trimesh.load(os.path.join(D, "stl", f)); t.merge_vertices()
    return m3.Manifold(m3.Mesh(np.asarray(t.vertices, np.float32), np.asarray(t.faces, np.uint32)))
def pts(mf, n=60000):
    me = mf.to_mesh(); t = trimesh.Trimesh(me.vert_properties[:, :3], me.tri_verts); return trimesh.sample.sample_surface(t, n)[0]
groups = {"초음파 받침(받침·누름쇠·볼트·센서)": ["asm_sonar.stl", "dum_hcsr04.stl"],
          "오른 전장 상자(+y)": ["asm_pod_right.stl", "asm_pod_right_lid.stl", "dum_pod_right_cables.stl"],
          "왼 전장 상자(−y)": ["asm_pod_left.stl", "asm_pod_left_lid.stl", "dum_pod_left_cables.stl"]}
for gn, fs in groups.items():
    g = m3.Manifold()
    for f in fs: g += ld(f)
    kt = cKDTree(pts(g, 200000))
    print("==", gn)
    for front in (True, False):
        for left in (True, False):
            worst = (1e9, 0, 0)
            for deg in (range(-30, 31, 5) if front else (0,)):
                w = at(front, left, deg); ov = (w ^ g).volume(); d = kt.query(pts(w, 40000))[0].min()
                if ov > 0.01: d = -ov
                if d < worst[0]: worst = (d, deg, ov)
            print(f"  {'앞' if front else '뒤'}{'왼(+y)' if left else '오른(−y)'} 바퀴: 최소 틈 {worst[0]:.1f} mm (조향 {worst[1]}°, 겹침 {worst[2]:.1f} mm³)" if worst[2] < 0.01 else
                  f"  {'앞' if front else '뒤'}{'왼(+y)' if left else '오른(−y)'} 바퀴: 겹침 {worst[2]:.1f} mm³ (조향 {worst[1]}°)")
