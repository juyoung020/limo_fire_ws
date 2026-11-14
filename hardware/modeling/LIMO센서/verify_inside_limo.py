"""blender -b --python verify_inside_limo.py : 고정대 점이 LIMO 메시 안에 들어갔는지(광선 14 방향 짝홀 다수결).
LIMO 메시는 완전히 닫힌 몸체가 아니라 방향마다 다를 수 있어 다수결(10/14 이상 '안')만 센다."""
import bpy, bmesh, os, random
from mathutils import Vector
from mathutils.bvhtree import BVHTree
D = os.path.dirname(os.path.abspath(__file__))
def load(p):
    bpy.ops.wm.stl_import(filepath=p); o = bpy.context.selected_objects[0]
    bm = bmesh.new(); bm.from_mesh(o.data); return bm
limo = load(os.path.join(D, "render", "asm", "limo.stl")); tree = BVHTree.FromBMesh(limo)
dirs = [Vector(v).normalized() for v in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1),(1,1,1),(-1,1,1),(1,-1,1),(1,1,-1),(-1,-1,1),(-1,1,-1),(1,-1,-1),(-1,-1,-1)]]
def inside(p):
    n = 0
    for d in dirs:
        o = p.copy(); c = 0
        for _ in range(200):
            h = tree.ray_cast(o, d, 2000.0)
            if h[0] is None: break
            c += 1; o = h[0] + d * 1e-4
        n += c % 2
    return n
random.seed(1)
for f in ("asm_sonar", "asm_sonar_tabs", "asm_sonar_bolts", "asm_arm", "asm_arm_caps", "asm_arm_bolts", "asm_pod_right", "asm_pod_right_lid",
          "asm_pod_right_bolts", "asm_pod_left", "asm_pod_left_lid", "asm_pod_left_bolts"):
    bm = load(os.path.join(D, "stl", f + ".stl"))
    pts = [v.co.copy() for v in bm.verts] + [fc.calc_center_median() for fc in bm.faces]
    if len(pts) > 12000: pts = random.sample(pts, 12000)
    ins = [p for p in pts if inside(p) >= 10]
    print("   z 범위", min((p.z for p in ins), default=0), max((p.z for p in ins), default=0))
    ex = [tuple(round(c, 1) for c in p) for p in ins[:5]]
    print(f"INS {f:20s} 점 {len(pts)}  LIMO 안 {len(ins)}  {ex}", flush=True)
