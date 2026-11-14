import bpy, sys, os, math
from mathutils import Vector
src, dst, tag = sys.argv[sys.argv.index("--") + 1:][:3]
bpy.ops.wm.read_factory_settings(use_empty=True)
s = bpy.context.scene; s.render.engine = "BLENDER_WORKBENCH"
s.render.resolution_x, s.render.resolution_y = 900, 700
sh = s.display.shading; sh.color_type = "OBJECT"; sh.show_cavity = True; sh.light = "STUDIO"
COL = {"body": (0.85,0.85,0.85,1), "wedge": (0.5,0.5,0.55,1), "plunger": (0.95,0.6,0.1,1), "pin": (0.9,0.1,0.1,1),
       "lever": (0.1,0.4,0.95,1), "sear": (0.1,0.75,0.2,1), "ball": (1,1,1,1), "servo": (0.15,0.15,0.4,1)}
objs = []
for f in sorted(os.listdir(src)):
    if not f.lower().endswith(".stl"): continue
    bpy.ops.wm.stl_import(filepath=os.path.join(src, f)); o = bpy.context.selected_objects[0]
    key = next((k for k in COL if k in f), "body"); o.color = COL[key]; objs.append(o)
mn = Vector((1e9,)*3); mx = Vector((-1e9,)*3)
for o in objs:
    for b in o.bound_box:
        w = o.matrix_world @ Vector(b); mn = Vector(map(min, mn, w)); mx = Vector(map(max, mx, w))
c = (mn + mx) / 2; r = (mx - mn).length
cam = bpy.data.objects.new("c", bpy.data.cameras.new("c")); s.collection.objects.link(cam); s.camera = cam
for i, v in enumerate([(1,-1.2,0.8),(-1,-1,0.6),(0.2,1.3,0.7),(0,-1,0.05),(0.001,0.001,1),(1,1,-0.4)]):
    v = Vector(v).normalized(); cam.location = c + v * r * 1.5
    cam.rotation_euler = (-v).to_track_quat("-Z", "Y").to_euler()
    s.render.filepath = os.path.join(dst, f"{tag}_{i}.png"); bpy.ops.render.render(write_still=True)
