"""blender -b --python render_asm.py -- out.png "stl1|r,g,b" "stl2|r,g,b" ... [views=dx,dy,dz;dx,dy,dz]
여러 STL 을 색 입혀 한 장면에 놓고 방향별로 그려 한 장으로 붙인다."""
import bpy, sys, math, os
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
out = argv[0]
items = [a for a in argv[1:] if "|" in a]
vopt = [a for a in argv[1:] if a.startswith("views=")]
views = [tuple(float(v) for v in s.split(",")) for s in vopt[0][6:].split(";")] if vopt else [(1, 1, 0.6), (-1, -1, 0.5), (1, -0.3, 0.2), (0, 0, 1)]

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
objs = []
for it in items:
    path, col = it.split("|")
    bpy.ops.wm.stl_import(filepath=path)
    o = bpy.context.selected_objects[0]
    m = bpy.data.materials.new(os.path.basename(path)); m.use_nodes = True
    bsdf = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    rgb = [float(c) for c in col.split(",")]
    bsdf.inputs["Base Color"].default_value = (*rgb, 1)
    o.data.materials.append(m); objs.append(o)
mn = Vector((1e9,) * 3); mx = Vector((-1e9,) * 3)
for o in objs:
    for c in o.bound_box:
        w = o.matrix_world @ Vector(c); mn = Vector(map(min, mn, w)); mx = Vector(map(max, mx, w))
ctr = (mn + mx) / 2; size = (mx - mn).length
cam_d = bpy.data.cameras.new("c"); cam_d.type = "ORTHO"; cam_d.ortho_scale = size * 1.05
cam = bpy.data.objects.new("c", cam_d); sc.collection.objects.link(cam); sc.camera = cam
for e in ((1, 1, 2), (-1, -0.5, 1), (0, -1, -0.3)):
    ld = bpy.data.lights.new("l", "SUN"); ld.energy = 3.0
    lo = bpy.data.objects.new("l", ld); sc.collection.objects.link(lo)
    lo.rotation_euler = Vector(e).to_track_quat("Z", "Y").to_euler()
sc.world = bpy.data.worlds.new("w"); sc.world.color = (0.22, 0.22, 0.24)
try:
    sc.render.engine = "BLENDER_EEVEE_NEXT"
except TypeError:
    sc.render.engine = "BLENDER_EEVEE"
sc.render.resolution_x = 900; sc.render.resolution_y = 900
tiles = []
for i, v in enumerate(views):
    d = Vector(v).normalized()
    cam.location = ctr + d * size * 3
    cam.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
    cam_d.clip_end = size * 10
    p = out.replace(".png", f"_{i}.png"); sc.render.filepath = p
    bpy.ops.render.render(write_still=True); tiles.append(p)
# 붙이기
imgs = [bpy.data.images.load(p) for p in tiles]
W = 900; n = len(imgs); cols = 2 if n > 1 else 1; rows = (n + cols - 1) // cols
big = bpy.data.images.new("sheet", W * cols, W * rows)
px = [0.0] * (W * cols * W * rows * 4)
for k, im in enumerate(imgs):
    src = list(im.pixels); r0 = rows - 1 - k // cols; c0 = k % cols
    for y in range(W):
        row = src[y * W * 4:(y + 1) * W * 4]
        o = ((r0 * W + y) * W * cols + c0 * W) * 4
        px[o:o + W * 4] = row
big.pixels = px; big.filepath_raw = out; big.file_format = "PNG"; big.save()
for p in tiles:
    os.remove(p)
print("saved", out)
