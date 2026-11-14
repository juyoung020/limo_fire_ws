"""blender -b --python thickness.py -- a.stl b.stl ...
면마다 안쪽(법선 반대)으로 광선을 쏴 반대편 벽까지 거리 = 두께. 면적 가중 하위 1 %·최솟값과 그 위치 출력.
구멍 둘레 같은 정상적인 얇은 곳도 잡히므로 위치를 보고 판단한다."""
import bpy, sys, bmesh
from mathutils.bvhtree import BVHTree

for p in sys.argv[sys.argv.index("--") + 1:]:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.stl_import(filepath=p)
    o = bpy.context.selected_objects[0]
    bm = bmesh.new(); bm.from_mesh(o.data); bm.normal_update()
    tree = BVHTree.FromBMesh(bm)
    res = []
    for f in bm.faces:
        if f.calc_area() < 1e-4:
            continue
        c = f.calc_center_median(); n = f.normal
        hit = tree.ray_cast(c - n * 1e-3, -n, 50.0)
        if hit[0] is not None:
            res.append((hit[3], f.calc_area(), tuple(round(v, 1) for v in c)))
    res.sort()
    tot = sum(a for _, a, _ in res); acc = 0; p1 = None
    for d, a, c in res:
        acc += a
        if acc > 0.01 * tot:
            p1 = d; break
    thin = [(round(d, 2), c) for d, a, c in res if d < 1.2][:6]
    print(f"THK {p.split('/')[-1]}: 최소 {res[0][0]:.2f} mm at {res[0][2]}, 하위1% {p1:.2f} mm, 1.2 mm 미만 면 {sum(1 for d,_,_ in res if d<1.2)}개 예 {thin}")
