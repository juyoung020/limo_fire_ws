"""참고 모델(Rexi Studios Auto-Reset Target)의 과녁판(원판 Ø100, 2 mm)을 불꽃으로 교체.
축·다리·끼움 걸쇠는 원본 그대로 두고, 원판만 잘라내 불꽃(빨강 + 흰 속불꽃 상감)을 붙인다.
불꽃 크기는 '면적 x 축까지 거리'(=넘어지는/돌아오는 회전력)가 원판과 같도록 정한다 → 자동 복귀 유지.

v2 (2026-11-10): v1 은 불꽃 밑이 축 가까이서 넓어(±39 mm) 받침 기둥 꼭대기(안쪽 면 x ±27, 축 위 9 mm)에 닿아 조립이 안 됐다.
과녁이 넘어갔다 서는 범위(-100° ~ +60°) 동안 기둥이 지나가는 자리를 1 mm 여유로 깎는다(축·다리는 원본 그대로).
흰 속불꽃은 앞·뒤 두 면에 0.6 mm 씩 넣어 양쪽에서 보이게 한다."""
import os, numpy as np, trimesh, manifold3d as m3
from build_flame import cr, OUTER, INNER

d = os.path.dirname(os.path.abspath(__file__))
orig = trimesh.load(os.path.join(d, "ref", "object_17_3.stl"))   # 원본 과녁판(판 z -3..-1, 축 y≈-30)
Z0, Z1, INLAY = -3.0, -1.0, 0.6
AXLE_Y, CUT_Y, FLAME_Y0 = -30.0, -22.0, -26.0                     # 축, 원판 잘라내는 선, 불꽃 밑변

def to_m(tm): return m3.Manifold(m3.Mesh(np.asarray(tm.vertices, np.float32), np.asarray(tm.faces, np.uint32)))
def to_t(mf): me = mf.to_mesh(); return trimesh.Trimesh(me.vert_properties[:, :3], me.tri_verts)
def poly_area_centroid(p):
    x, y = p[:, 0], p[:, 1]; x1, y1 = np.roll(x, -1), np.roll(y, -1); c = x * y1 - x1 * y
    A = c.sum() / 2; return abs(A), (y + y1) @ c / (6 * A)

AX_Y, AX_Z, HOLE_Z = -32.0, -0.5, 26.0      # 원본 축 중심(과녁 좌표, 메시 실측), 받침 구멍 중심 높이
out0, in0 = cr(OUTER), cr(INNER)

def ext(poly, z0, z1):
    return m3.Manifold.extrude(m3.CrossSection([poly.tolist()], m3.FillRule.EvenOdd), z1 - z0).translate([0, 0, z0])

# 받침 기둥이 지나가는 자리(과녁 좌표로 옮긴 받침을 각도별로 합침). 조립: 과녁 축(y -32, z -0.5) = 받침 (0, 0, 26),
# 과녁 y → 받침 z(위), 과녁 z → 받침 -y. 여유 1 mm 는 받침을 y·z 로 ±1 mm 옮긴 것까지 합쳐서 준다.
base_m = to_m(trimesh.load(os.path.join(d, "ref", "object_2_8.stl")))
fat = base_m
for dy, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
    fat += base_m.translate([0, dy, dz])
def base_in_target(a):
    return fat.translate([0, 0, -HOLE_Z]).rotate([-a, 0, 0]).rotate([-90, 0, 0]).translate([0, AX_Y, AX_Z])
SWEEP = (-100, 60)
sweep = m3.Manifold.batch_boolean([base_in_target(a) for a in range(SWEEP[0], SWEEP[1] + 1, 2)], m3.OpType.Add)
keep = to_m(orig) ^ m3.Manifold.cube([300, 300, 20]).translate([-150, CUT_Y - 300, -10])   # 축·다리만 남김

def make(sc):
    place = lambda p: p * sc + np.array([0, FLAME_Y0])
    outer = ext(place(out0), Z0, Z1)
    inner_cut = ext(place(in0), Z1 - INLAY, Z1 + 1) + ext(place(in0), Z0 - 1, Z0 + INLAY)
    white = ext(place(in0), Z1 - INLAY, Z1) + ext(place(in0), Z0, Z0 + INLAY)
    return keep + ((outer - inner_cut) - sweep), white - sweep

def moment(mf):
    """축 기준 무게 회전력(부피 x 무게중심 높이). 원본 과녁판과 같아야 같은 무게추로 다시 선다."""
    tm = to_t(mf); return tm.volume * (tm.center_mass[1] - AX_Y)

# 원본 과녁판(구멍 뚫린 원판)의 실제 회전력에 맞춰 불꽃 배율을 이분법으로 찾는다(v1 은 구멍 없는 원판 넓이로 계산해 38 % 무거웠음)
target_m = moment(to_m(orig))
lo, hi = 0.3, 1.0
for _ in range(14):
    mid = (lo + hi) / 2
    r, w = make(mid)
    if moment(r + w) > target_m: hi = mid
    else: lo = mid
s = (lo + hi) / 2
red, white = make(s)
print("불꽃 배율 %.3f → 크기 %.0f x %.0f mm, 회전력 불꽃/원본 %.3f" % (s, np.ptp(out0[:, 0]) * s, np.ptp(out0[:, 1]) * s,
      moment(red + white) / target_m))

for name, mf in (("target_flame_red", red), ("target_flame_white", white), ("target_flame_merged", red + white)):
    tm = to_t(mf); tm.export(os.path.join(d, name + ".stl"))
    print(name, "덩어리", len(tm.split()), "닫힘", tm.is_watertight, "부피cm3", round(tm.volume / 1000, 2))
print("원본 과녁판 부피cm3", round(orig.volume / 1000, 2))
