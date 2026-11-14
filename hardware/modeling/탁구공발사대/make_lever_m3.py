"""레버 M3 판: 혼이 M3 구멍(알루미늄 원판 혼)일 때. 나사 홈 폭 2.2 → 3.4, 머리 자리 Ø5.2 → Ø6.2. 나머지는 print_lever.stl 과 같다.
혼 구멍을 재 보고 M3(지름 약 3)면 이 파일(stl/print_lever_M3.stl)을 뽑는다. 핵심 세트 파일은 바꾸지 않는다."""
import os, math
import manifold3d as m3
import build_launcher as B
D = os.path.dirname(os.path.abspath(__file__))
box, cyl_z = B.box, B.cyl_z
def horn_mount_m3(r_hub, t):
    hub = cyl_z(r_hub, 0, t, 0, 0, 64)
    hub -= cyl_z(3.5, -1, t + 1, 0, 0, 32)
    for a in range(4):
        sl = box(6, 10, -1.7, 1.7, -1, t + 1) + cyl_z(1.7, -1, t + 1, 6, 0, 16) + cyl_z(1.7, -1, t + 1, 10, 0, 16)
        hub -= sl.rotate([0, 0, 45 + 90 * a])
    return hub
B.horn_mount = horn_mount_m3
orig_cyl = B.cyl_z
def cyl_z_head(r, z0, z1, x=0, y=0, n=96):
    return orig_cyl(3.1 if abs(r - 2.6) < 1e-9 else r, z0, z1, x, y, n)
B.cyl_z = cyl_z_head
lev = B.build_lever()
B.cyl_z = orig_cyl
tm = B.to_t(lev.rotate([180, 0, 0])); tm.apply_translation([0, 0, -tm.bounds[0][2]])
print("pieces", len(tm.split()), "watertight", tm.is_watertight, "vol cm3", round(tm.volume / 1000, 2))
tm.export(os.path.join(D, "stl", "print_lever_M3.stl"))
