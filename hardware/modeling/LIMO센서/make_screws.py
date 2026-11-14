"""LIMO 기존 M3 나사 구멍(표준 나사산)에 넣는 출력 M3 나사. 눕혀 출력(층이 길이 방향), 바닥 0.3 mm 평평.
- M3 × 8: 초음파 받침 → 차체 밑면 4 개(+예비 2)
- M3 × 5: 팔 → 지붕 2 개, 전장 판 → 옆판 8 개(+예비 2)
머리 Ø6 × 2.4 일자 홈: 팔 머리 자리 Ø6.2, 전장 판 머리 자리 Ø6.4 에 들어감."""
import os, sys
import trimesh
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(D, "..", "lib"))
import fasteners as F

def to_t(m):
    me = m.to_mesh(); return trimesh.Trimesh(me.vert_properties[:, :3], me.tri_verts)

for L in (8.0, 5.0):
    s = F.iso_screw_print(3.0, L, 0.5, head_d=6.0, head_h=2.4)
    t = to_t(s); t.apply_translation([-t.bounds[0][0], -t.bounds[0][1], 0])
    t.export(os.path.join(D, "print", f"print_m3x{int(L)}.stl"))
    print(f"M3x{int(L)}", t.bounds.round(2).tolist(), "bodies", len(t.split(only_watertight=False)))
