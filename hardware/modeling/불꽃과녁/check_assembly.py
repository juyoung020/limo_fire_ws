"""불꽃 과녁 전체 조립 검사: 받침(object_2_8) + 과녁(원본 object_17_3 / 불꽃) + 무게추 상자(object_12_6).
- 무게추: 과녁 다리 끝(y -76)이 상자 홈 바닥까지 들어감. 상자 윗면 = 과녁 y -52, 홈은 상자 z 방향, 상자 x = 과녁 x, 상자 y = 과녁 두께(z).
- 넘어가는 범위 전체에서 (과녁 + 상자) 와 받침 겹침, 다리와 상자 겹침(원본과 같아야 함).
- 다시 서는지: 실제 출력 무게로 축 기준 무게중심 높이."""
import os, json, numpy as np, trimesh, manifold3d as m3
d = os.path.dirname(os.path.abspath(__file__))
to_m = lambda tm: m3.Manifold(m3.Mesh(np.asarray(tm.vertices, np.float32), np.asarray(tm.faces, np.uint32)))
def to_t(mf): me = mf.to_mesh(); return trimesh.Trimesh(me.vert_properties[:, :3], me.tri_verts)
L = lambda n: to_m(trimesh.load(os.path.join(d, n)))
base = L("ref/object_2_8.stl")
box_raw = L("ref/object_12_6.stl")                       # z -14..14, 홈은 +z 쪽에서 들어감
BOX_TOP_Y = -52.0
box_t = box_raw.rotate([90, 0, 0]).translate([0, BOX_TOP_Y - 14, 0])   # 상자 +z → 과녁 +y(축 쪽), 상자 y → 과녁 -z
box_t = box_t.translate([0, 0, -1.0 - (-1.0)])          # 다리 두께 중심(z -2)과 홈 중심(0) 맞춤 ↓
box_t = box_t.translate([0, 0, -2.0])
AX_Y, AX_Z, HOLE_Z = -32.0, -0.5, 26.0
def pose(m, a): return m.translate([0, -AX_Y, -AX_Z]).rotate([90, 0, 0]).rotate([a, 0, 0]).translate([0, 0, HOLE_Z])
res = {}
targets = {"원본": L("ref/object_17_3.stl"), "불꽃": L("target_flame_merged.stl")}
for name, t in targets.items():
    res[name + "_다리↔상자_겹침mm3"] = round((t ^ box_t).volume(), 1)
    sweep = []
    for a in range(-110, 61, 10):
        sweep.append((a, round((pose(t, a) ^ base).volume(), 1), round((pose(box_t, a) ^ base).volume(), 1)))
    res[name + "_각도별(과녁↔받침, 무게추↔받침)"] = sweep
# 다시 서는지: 실제 g (무게추는 3번 판 39.19 g 출력값, 과녁은 2 mm 판이라 거의 꽉 참 → 부피 x 1.24)
M_BOX = 39.19
for name, t in targets.items():
    tm = to_t(t); mt = tm.volume / 1000 * 1.24
    cg_t = tm.center_mass[1] - AX_Y
    cg_b = to_t(box_t).center_mass[1] - AX_Y
    cg = (mt * cg_t + M_BOX * cg_b) / (mt + M_BOX)
    res[name + "_서는힘"] = dict(과녁_g=round(mt, 1), 과녁_무게중심_축위_mm=round(cg_t, 1), 무게추_무게중심_mm=round(cg_b, 1),
                               전체_무게중심_mm=round(cg, 1), 다시섬=bool(cg < 0))
os.makedirs(os.path.join(d, "asm_check"), exist_ok=True)
for a in (0, -90):
    for n, m in (("box", box_t), ("red", L("target_flame_red.stl")), ("white", L("target_flame_white.stl"))):
        to_t(pose(m, a)).export(os.path.join(d, "asm_check", f"{n}_{abs(a)}.stl"))
to_t(base).export(os.path.join(d, "asm_check", "base.stl"))
print(json.dumps(res, ensure_ascii=False, indent=1))
