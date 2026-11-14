"""축 핀 잠금 — 받침대 +y 볼(벽)을 앞·뒤 모서리에서 감싸는 고리 2 조각 + 긴 출력 M5 볼트(x 방향).
뒤 조각은 핀 머리(Ø14, y 21~24)를 바깥에서 누르는 판까지 한 몸. 받침대·몸통은 이미 뽑은 것 그대로.
지붕 좌표. 볼: x −40 ~ −10, y 17 ~ 21, 45° 받침 x −40~−36·−14~−10 (y 21 → 46, z 47 → 0).
빈 곳(발사대 5~30° 휩쓸린 부피 기준): 볼 안쪽(y 12~17)은 z 37 아래, 바깥 앞 모서리(x ≥ −8)는 z 52 아래."""
import os, sys, math
import numpy as np, manifold3d as m3, trimesh
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D); sys.path.insert(0, os.path.join(D, "..", "..", "lib"))
import build_launcher as B, build_cradle as C
import fasteners as F
from shape import fillet
box = C.box
M5 = (5.0, 1.25, 0.4)
Z0 = 6.0                      # 바닥판(z 3) 위 3
HOOK_Z1 = 28.0
YH0, YH1 = 13.5, 16.8         # 볼 안쪽 고리(볼 안면 y 17)
Y_OUT = 32.0                  # 옆벽 바깥 끝
BZ, BY = 40.0, 29.0           # 볼트 축 (x 방향)
X_R, X_F = -40.0, -10.0       # 볼 뒤·앞 끝


def gusset_clear():
    g = m3.Manifold()
    for gx in (X_AX := C.X_AX - 15, C.X_AX + 11):
        g += m3.Manifold.hull(box(gx - 0.3, gx + 4.3, 20.0, 21.4, -1, 47.6) + box(gx - 0.3, gx + 4.3, 20.0, 46.6, -1, 0.2))
    return g


def rear():
    w = box(X_R - 4.0, X_R - 0.3, YH0, Y_OUT, Z0, HOOK_Z1) + box(X_R - 4.0, X_R - 0.3, 17.0, Y_OUT, Z0, 72.0)
    hook = box(X_R - 0.31, X_R + 3.5, YH0, YH1, Z0, HOOK_Z1)
    bridge = box(X_R - 4.0, -16.0, 24.0, 28.5, 50.0, 72.0)   # 안면 y 24 = 핀 머리 바깥면(옆으로 밀어 넣을 때 머리 위를 미끄러짐)
    r = w + hook + bridge
    r -= gusset_clear()
    r -= B.cyl_x(M5[0] / 2 + 0.3, X_R - 6, X_R + 1, 0, BZ, 32).translate([0, BY, 0])
    return fillet(r, 0.8)


def front():
    w = box(X_F + 0.3, X_F + 4.0, YH0, Y_OUT, Z0, HOOK_Z1) + box(X_F + 0.3, X_F + 4.0, 17.0, Y_OUT, Z0, 48.0)
    hook = box(X_F - 3.5, X_F + 0.31, YH0, YH1, Z0, HOOK_Z1)
    f = w + hook
    f -= gusset_clear()
    f -= B.cyl_x(M5[0] / 2 + 0.3, X_F - 1, X_F + 6, 0, BZ, 32).translate([0, BY, 0])
    return fillet(f, 0.8)


BOLT_L = (X_F + 4.0) - (X_R - 4.0) + 4.5 + 1.0       # 두 벽 사이 + 너트 4.5


def bolt_nut():
    b = F.bolt(M5[0], BOLT_L, M5[1], M5[2], head_d=8.0, head_h=3.0).rotate([0, 90, 0]).translate([X_R - 4.0, BY, BZ])
    n = F.nut(M5[0], 4.5, M5[1], M5[2]).rotate([0, 90, 0]).translate([X_F + 4.0, BY, BZ])
    return b, n


def to_t(m):
    me = m.to_mesh(); return trimesh.Trimesh(me.vert_properties[:, :3], me.tri_verts)


if __name__ == "__main__":
    vol = lambda m: round(m.volume(), 2)
    R, Fr = rear(), front(); bl, nt = bolt_nut()
    cr = C.build_cradle()
    pin = C.pose(B.build_axle_pin().translate([B.AX[0], B.KEEL_Y - 15 + 5, B.AX[1]]), 15.0)
    body = B.build_body(); lever0 = B.build_lever()
    L = body + B.lever_servo_place(B.servo_solid(B.SERVO_L, horn=False)) + B.lever_at(0, lever0) + B.lever_at(B.LEVER_COCK_DEG, lever0) + B.sear_at(0, B.build_sear())
    sw = m3.Manifold()
    for th in np.arange(5, 30.1, 1.0): sw += C.pose(L, th)
    allk = R + Fr + bl + nt
    print("조각 덩어리", len(to_t(R).split(only_watertight=False)), len(to_t(Fr).split(only_watertight=False)))
    print("고리∩받침대", vol(allk ^ cr), " ∩핀", vol(allk ^ pin), " ∩발사대(5~30° 휩쓸림)", vol(allk ^ sw), " 뒤∩앞", vol(R ^ Fr), " 볼트∩조각", vol(bl ^ (R + Fr)))
    print("핀이 0.3 바깥(+y)으로 나오면 뒤 조각과 겹침", vol(pin.translate([0, 0.3, 0]) ^ R))
    print("고리가 볼 안면에 닿음: 조각 0.3 바깥(+y)으로 ∩ 받침대", vol((R + Fr).translate([0, 0.3, 0]) ^ cr))
    print("볼트 길이", round(BOLT_L, 1))
    best = min(vol(F.nut(M5[0], 4.5, M5[1], M5[2]).rotate([0, 0, a]).rotate([0, 90, 0]).translate([X_F + 4.0, BY, BZ]) ^ bl) for a in range(0, 360, 20))
    print("볼트·너트 물림(최적 위상)", best)
    os.makedirs(os.path.join(D, "stl"), exist_ok=True)
    for nm, m in (("asm_axle_keeper", R + Fr), ("asm_axle_keeper_bolt", bl + nt)):
        to_t(m).export(os.path.join(D, "stl", nm + ".stl"))
    # 출력 자세: 옆벽 바깥면(x 끝)이 바닥 → 볼트 구멍이 세로, 판은 세워짐
    for nm, m in (("axle_keeper_r", R.rotate([0, -90, 0])), ("axle_keeper_f", Fr.rotate([0, 90, 0]))):
        t = to_t(m); t.apply_translation(-t.bounds[0]); t.export(os.path.join(D, "stl", f"print_{nm}.stl"))
    tb = to_t(F.bolt(M5[0], BOLT_L, M5[1], M5[2], head_d=8.0, head_h=3.0)); tb.apply_translation([0, 0, -tb.bounds[0][2]])
    tb.export(os.path.join(D, "stl", "print_axle_keeper_bolt.stl"))
    to_t(F.nut(M5[0], 4.5, M5[1], M5[2])).export(os.path.join(D, "stl", "print_axle_keeper_nut.stl"))
