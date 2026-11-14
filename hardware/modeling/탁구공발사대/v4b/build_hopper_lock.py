"""공 통 조임 고리 — 긴 공 통 소켓과 몸통 짧은 통을 함께 감싸 출력 M5 볼트 2 개로 조임.

이미 출력한 몸통·긴 공 통은 그대로. 고리 반쪽 2 개(+y, −y)를 이음매에 대고 앞·뒤 귀를 볼트·너트로 조인다.
- 아래 띠(z 57~62): 몸통 짧은 통(바깥 r 22.4)을 꽉 물음
- 가운데(z 62~72.2): 공 통 소켓(바깥 r 24.2)을 꽉 물음
- 위 턱(z 72.2~74.0 경사 + 74~76): 소켓 위 45° 어깨를 눌러 통이 위로 빠지지 않음
- 반쪽 사이 틈 1.2 mm 가 조이는 여유(조이면 닫히며 둘레가 줄어 꽉 잡음)
좌표: 발사대 자기 좌표(build_launcher.py). 공 통 축 (HOP_CX, 0).
"""
import os, sys, math
import numpy as np
import manifold3d as m3
import trimesh

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
sys.path.insert(0, os.path.join(D, "..", "..", "lib"))
import build_launcher as B
import fasteners as F
from shape import fillet

SEG = 128
CX = B.HOP_CX
R_STUB = B.BORE_R + B.HOP_WALL               # 22.4
R_SOCK = R_STUB + 0.2 + 1.6                  # 24.2 (build_hopper_tube 의 rso)
Z_SPLIT = B.HOP_SPLIT_Z                      # 72
Z_SOCK0 = Z_SPLIT - B.HOP_SOCK               # 62
Z0, Z1 = 57.0, 76.0
FIT = 0.05                                   # 반지름 틈(조이면 닫힘)
R_OUT = 27.6
GAP = 1.2                                    # 반쪽 사이
BOLT_D, BOLT_P, BOLT_E = 5.0, 1.25, 0.4
EAR_X = R_OUT + 4.5                          # 귀 가운데(축에서)
EAR_W, EAR_T = 14.0, 7.0                     # 귀 x 폭, 반쪽마다 y 두께
Z_BOLT = (Z0 + 1 + Z1) / 2


def rev(profile_rz):
    """(r, z) 다각형을 z 축으로 돌림."""
    cs = m3.CrossSection([[(r, z) for r, z in profile_rz]])
    return m3.Manifold.revolve(cs, SEG)   # revolve 는 x=r, y=z 평면을 z(=y) 축으로 → 아래에서 축 바꿈


def ring_solid():
    # 바깥 원통 - 안쪽 계단 구멍
    outer = m3.Manifold.cylinder(Z1 - Z0, R_OUT, R_OUT, SEG).translate([0, 0, Z0])
    zc0, zc1 = Z_SPLIT + 0.2, Z_SPLIT + 0.2 + (R_SOCK - R_STUB)   # 소켓 위 45° 어깨
    inner = m3.Manifold.cylinder(Z_SOCK0 - Z0 + 0.01, R_STUB + FIT, R_STUB + FIT, SEG).translate([0, 0, Z0 - 0.01])
    inner += m3.Manifold.cylinder(zc0 - Z_SOCK0, R_SOCK + FIT, R_SOCK + FIT, SEG).translate([0, 0, Z_SOCK0])
    inner += m3.Manifold.cylinder(zc1 - zc0, R_SOCK + FIT, R_STUB + FIT, SEG).translate([0, 0, zc0])
    inner += m3.Manifold.cylinder(Z1 - zc1 + 0.01, R_STUB + FIT, R_STUB + FIT, SEG).translate([0, 0, zc1])
    r = outer - inner
    # 귀(앞·뒤)
    for s in (-1, 1):
        r += m3.Manifold.cube([EAR_W, 2 * EAR_T + GAP, Z1 - Z0 - 1.0]).translate(
            [s * EAR_X - EAR_W / 2, -(EAR_T + GAP / 2), Z0 + 0.5])
    return r


def halves():
    r = ring_solid()
    hp = r ^ m3.Manifold.cube([200, 100, 100]).translate([-100, GAP / 2, 0])
    hn = r ^ m3.Manifold.cube([200, 100, 100]).translate([-100, -100 - GAP / 2, 0])
    for s in (-1, 1):
        x = s * EAR_X
        # 출력은 세워서(z 그대로) → 가로 구멍은 위가 뾰족한 눈물방울, 너트 자리는 꼭짓점이 위
        hole = B.tear_y(BOLT_D / 2 + 0.3, -30, 30, x, Z_BOLT)
        hp -= hole; hn -= hole
        hp -= B.tear_y(5.2, GAP / 2 + EAR_T - 2.6, GAP / 2 + EAR_T + 10, x, Z_BOLT)            # +y: 볼트 머리 자리
        hn -= F.hex_pocket(BOLT_D).rotate([0, 0, 30]).rotate([90, 0, 0]).translate([x, -(GAP / 2 + EAR_T) + 2.6, Z_BOLT])   # −y: 너트 자리
    hp, hn = hp.translate([CX, 0, 0]), hn.translate([CX, 0, 0])
    # 몸통 짧은 통 옆 경사벽(x −16~−8, y 20~27, z ≤ 60) 자리 비움(0.5 띄움)
    hp -= m3.Manifold.cube([10.0, 11.5, 61.0 - (Z0 - 1)]).translate([-16.5, 17.0, Z0 - 1])   # 안쪽 구멍까지 엶(얇은 막 없음)
    return fillet(hp, 0.8), fillet(hn, 0.8)


def bolts_nuts():
    """조립 자세의 볼트·너트(확인용)와 출력용."""
    L = 2 * EAR_T + GAP + 1.0
    asm = m3.Manifold()
    for s in (-1, 1):
        x = CX + s * EAR_X
        b = F.bolt(BOLT_D, L, BOLT_P, BOLT_E, head_d=8.0, head_h=3.0)       # 머리 z<0, 나사 z 0→L
        asm += b.rotate([90, 0, 0]).translate([x, GAP / 2 + EAR_T - 2.5, Z_BOLT])
        n = F.nut(BOLT_D, 4.5, BOLT_P, BOLT_E)
        asm += n.rotate([0, 0, 30]).rotate([90, 0, 0]).translate([x, -(GAP / 2 + EAR_T) + 2.5, Z_BOLT])   # 너트 자리와 같은 방향
    pb = F.bolt(BOLT_D, L, BOLT_P, BOLT_E, head_d=8.0, head_h=3.0)
    pn = F.nut(BOLT_D, 4.5, BOLT_P, BOLT_E)
    return asm, pb, pn, L


def to_t(mf):
    me = mf.to_mesh()
    return trimesh.Trimesh(me.vert_properties[:, :3], me.tri_verts)


if __name__ == "__main__":
    hp, hn = halves()
    asm_b, pb, pn, L = bolts_nuts()
    os.makedirs(os.path.join(D, "stl"), exist_ok=True)
    to_t(hp).export(os.path.join(D, "stl", "asm_hoplock_p.stl")); to_t(hn).export(os.path.join(D, "stl", "asm_hoplock_n.stl"))
    to_t(asm_b).export(os.path.join(D, "stl", "asm_hoplock_bolts.stl"))
    # 출력 자세: 반쪽은 세워서(z 그대로, 아래 띠가 판), 볼트는 머리가 판, 너트는 눕힘
    for nm, m in (("hoplock_p", hp), ("hoplock_n", hn)):
        t = to_t(m); t.apply_translation([0, 0, -t.bounds[0][2]]); t.export(os.path.join(D, "stl", f"print_{nm}.stl"))
    tb = to_t(pb); tb.apply_translation([0, 0, -tb.bounds[0][2]]); tb.export(os.path.join(D, "stl", "print_hoplock_bolt.stl"))
    tn = to_t(pn); tn.export(os.path.join(D, "stl", "print_hoplock_nut.stl"))
    print("bolt length", L)
    for nm, m in (("half+", hp), ("half-", hn)):
        t = to_t(m); print(nm, "bodies", len(t.split(only_watertight=False)), "watertight", t.is_watertight, "bounds", t.bounds.round(1).tolist())


# ---------------- 공 통 뚜껑 잠금: 뚜껑 옆 보스 + 손나사(M5)가 공 통 윗띠를 누름 ----------------
CAP_BOSS = (BOLT_D, BOLT_P, BOLT_E)


def cap_locked():
    """뚜껑(원래와 같은 끼움) + 앞(+x) 보스에 M5 암나사. 뚜껑 좌표: 아래 열림 z 0, 윗판 z 8~10."""
    ro = B.BORE_R + B.HOP_WALL
    ri = ro + 0.3
    cap = B.build_hopper_cap()
    r_boss = ri + 1.6 + 6.0
    boss = m3.Manifold.cube([r_boss - ri, 11.0, 10.0]).translate([ri, -5.5, 0.0])     # 윗판(z 10)까지: 뒤집어 출력할 때 판에 닿음
    cap += boss
    cap = fillet(cap, 0.6)                 # 뚜껑 벽 1.6 이 남도록 R0.6, 나사산은 뒤에
    th = F.female_cut(BOLT_D, r_boss - ri + 0.6, BOLT_P, BOLT_E).rotate([0, 90, 0]).translate([ri - 0.3, 0, 4.0])
    cap -= th
    return cap, r_boss - ri


def thumb_for_cap(reach):
    return F.thumb_screw(BOLT_D, reach + 0.4, BOLT_P, BOLT_E, knob_d=14.0, knob_h=5.0)


if __name__ == "__main__":
    cap, reach = cap_locked()
    to_t(cap).export(os.path.join(D, "stl", "asm_cap_locked_local.stl"))
    t = to_t(cap.rotate([180, 0, 0])); t.apply_translation([0, 0, -t.bounds[0][2]]); t.export(os.path.join(D, "stl", "print_cap_locked.stl"))
    ts = to_t(thumb_for_cap(reach)); ts.apply_translation([0, 0, -ts.bounds[0][2]]); ts.export(os.path.join(D, "stl", "print_cap_thumb.stl"))
    print("cap bodies", len(t.split(only_watertight=False)), "thumb reach", round(reach + 0.4, 1))
