"""출력용 공(탁구공 대신): 속 빈 매끈한 공을 반구 2개로 나눠 나사로 잠근다.
- 바깥 지름 39.8 (총신 Ø41.6 → 틈 0.9), 벽 0.8.
- 위 반구(A): 아래로 수나사 입술. 아래 반구(B): 안쪽 암나사. 끝까지 돌리면 45° 원뿔 자리에 물려 조여진다.
- 나사: 편심 원을 비틀어 뽑은 둥근 나사(깊이 0.8, 피치 2.0, 약 2바퀴), 반지름 틈 0.15.
- 출력: 두 반구 모두 접합면을 판에 대고(둥근 쪽 위), 서포트 없음.
  반구 꼭대기 안쪽은 지름 16 평평한 천장(다리 놓기)으로 막아 처지는 돔 끝을 없앴다.
사용: cad\\.venv python build_ball.py → stl/ball_A.stl, stl/ball_B.stl, stl/ball_asm.stl(조립 확인용)"""
import os, math
import numpy as np, trimesh
from manifold3d import Manifold, CrossSection

D = os.path.dirname(os.path.abspath(__file__))
R = 19.9
T = float(os.environ.get('BALL_T', 0.8))   # 벽
RI = R - T
SEG = 160
LAND = 0.6                  # 접합면 바깥 평평한 띠 폭
CLR = 0.15                  # 나사 반지름 틈
E, P = 0.3, 2.0             # 편심(나사 깊이 2E), 피치
ZB = float(os.environ.get('BALL_ZB', -4.5))   # 암나사 바닥 높이
CAP_Z = RI * math.cos(math.radians(25))   # 안쪽 천장 높이(지름 약 16)

def sphere(r): return Manifold.sphere(r, SEG)
def cyl(r, z0, z1, r1=None):
    return Manifold.cylinder(z1 - z0, r, r if r1 is None else r1, SEG).translate((0, 0, z0))
def box_z(z0, z1, s=60): return Manifold.cube((2 * s, 2 * s, z1 - z0)).translate((-s, -s, z0))
def thread(rc, z0, z1):     # 편심 원을 피치 P 로 비틀어 올린 나사 몸통
    h = z1 - z0
    cs = CrossSection.circle(rc, SEG).translate((E, 0))
    return Manifold.extrude(cs, h, int(h / P * 48), 360.0 * h / P).translate((0, 0, z0))

r_seat_out = R - LAND                       # 원뿔 자리 바깥(=19.3), z=0
rs_ZB = math.sqrt(R * R - ZB * ZB)
rf = rs_ZB - T - E                          # 암나사 편심원 반지름(최대 반지름 = 바깥-벽)
z_cone = -(r_seat_out - (rf + E))           # 원뿔이 암나사 최대 반지름에 닿는 높이
rm = rf - CLR                               # 수나사
LIP = float(os.environ.get('BALL_LIP', 1.0))
r_lip_in = rm - E - LIP                     # 입술 안 반지름(얇은 곳 LIP)

cavity_cut = sphere(RI) ^ box_z(-CAP_Z, CAP_Z)       # 빈 속(위아래 꼭대기는 평평한 천장 위로 꽉 참 — 처지는 돔 끝 없음)

# 접합 원뿔: 바깥 평평띠(r 19.3~R, z=0) → 45° 로 안쪽·아래로 → 암나사 최대 반지름
cone = cyl(rf + E, z_cone, 0.0, r_seat_out)          # 아래 r=rf+E, 위 r=19.3

# --- B (아래 반구): z ZB~0 은 꽉 찬 띠, 그 안을 원뿔 자리 + 암나사로 판다
B = (sphere(R) ^ box_z(-30, 0)) - (cavity_cut ^ box_z(-30, ZB))
B = B - cone - thread(rf, ZB - 0.01, z_cone + 0.01) - cyl(rf - E, ZB - 0.01, 0.1)

# --- A (위 반구): z 0~1 턱 + 아래로 원뿔 + 수나사 입술
LEDGE = 0.6
A = (sphere(R) ^ box_z(0, 30)) - (cavity_cut ^ box_z(LEDGE, 30))
cone_A = cyl(rf + E - CLR, z_cone, 0.3, r_seat_out - CLR + 0.3) ^ sphere(R)   # 원뿔 자리보다 CLR 안쪽, 턱 속으로 0.3 겹침
A = A + (thread(rm, ZB - 0.01, z_cone + 0.01) ^ box_z(ZB + 0.3, 1)) + cone_A   # 암나사와 같은 높이에서 시작(나사 위상 일치)
A = A - cyl(r_lip_in, ZB - 0.1, LEDGE + 0.5)

def tm(m):
    o = m.to_mesh(); t = trimesh.Trimesh(np.array(o.vert_properties)[:, :3], np.array(o.tri_verts), process=True)
    parts = [q for q in t.split(only_watertight=False) if abs(q.volume) > 1.0]   # 부피 0 조각 제거
    return trimesh.util.concatenate(parts)

out = os.path.join(D, "stl"); os.makedirs(out, exist_ok=True)
a, b = tm(A), tm(B)
tm(A + B).export(os.path.join(out, "ball_asm.stl"))
# 출력 방향: 접합면(z=0 쪽 가장 아래)을 판에
a_p = a.copy(); a_p.apply_translation((0, 0, -a.bounds[0][2]))
b_p = b.copy(); b_p.apply_transform(trimesh.transformations.rotation_matrix(math.pi, (1, 0, 0))); b_p.apply_translation((0, 0, -b_p.bounds[0][2]))
a_p.export(os.path.join(out, "ball_A.stl")); b_p.export(os.path.join(out, "ball_B.stl"))
print(dict(rf=round(rf, 3), rm=round(rm, 3), z_cone=round(z_cone, 3), lip_in=round(r_lip_in, 3), cap_z=round(CAP_Z, 2),
           A_cm3=round(a.volume / 1000, 3), B_cm3=round(b.volume / 1000, 3), g=round((a.volume + b.volume) / 1000 * 1.24, 2),
           A_bodies=len(a.split(only_watertight=False)), B_bodies=len(b.split(only_watertight=False)),
           A_h=round(a_p.extents[2], 2), B_h=round(b_p.extents[2], 2), overlap_cm3=round((A ^ B).volume() / 1000, 4)))
