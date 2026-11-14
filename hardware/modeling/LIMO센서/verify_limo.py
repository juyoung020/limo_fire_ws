"""LIMO 센서 고정대 전체 조립 검증(지붕 좌표). stl/asm_*.stl · dum_*.stl 을 그대로 읽는다(먼저 build_*.py 실행).
1 부품 겹침  2 볼트 물림(위상 돌려 최소)  3 잡힘(0.6 mm 밀어 6 방향 모두 막히는지)
4 끼우는 길(0.5 mm 간격 쓸기)  5 M3 드라이버 길  6 LIMO 표면 거리"""
import os, sys, math
import numpy as np, manifold3d as m3, trimesh
from scipy.spatial import cKDTree
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D); sys.path.insert(0, os.path.join(D, "..", "lib"))
import fasteners as F
import build_sonar as S, build_pod as P, build_arm as A
ST = os.path.join(D, "stl")
ld = lambda n: (lambda t: m3.Manifold(m3.Mesh(t.vertices.astype(np.float32), t.faces.astype(np.uint32))))(trimesh.load(os.path.join(ST, n)))
vol = lambda a: round(a.volume(), 2)
cen = lambda m: (np.array(m.bounding_box()[:3]) + np.array(m.bounding_box()[3:])) / 2
U = lambda ms: sum(ms, m3.Manifold())

# ---------- 부품 ----------
parts = {"초음파 받침": ld("asm_sonar.stl"), "팔": ld("asm_arm.stl"),
         "오른 판": ld("asm_pod_right.stl"), "왼 판": ld("asm_pod_left.stl"),
         "오른 덮개": ld("asm_pod_right_lid.stl"), "왼 덮개": ld("asm_pod_left_lid.stl")}
tabs = sorted(ld("asm_sonar_tabs.stl").decompose(), key=lambda m: cen(m)[1])
assert len(tabs) == 3, len(tabs)
parts["누름쇠 오른(−y)"], parts["누름쇠 가운데"], parts["누름쇠 왼(+y)"] = tabs
parts["센서 뚜껑"] = ld("asm_arm_caps.stl")
boards = {}
boards["HC-SR04 오른"], boards["HC-SR04 왼"] = S.pose(S.dummy_sensor_local(), -1), S.pose(S.dummy_sensor_local(), 1)   # 안쪽 끝이 맞닿음
boards["열화상 보드"] = ld("dum_thermal.stl"); boards["레이더 보드"] = ld("dum_radar.stl")
pr = sorted(ld("dum_pod_right.stl").decompose(), key=lambda m: cen(m)[0]); pl = sorted(ld("dum_pod_left.stl").decompose(), key=lambda m: cen(m)[0])
assert len(pr) == 2 and len(pl) == 2, (len(pr), len(pl))
boards["배선판"], boards["ESP32"] = pr
boards["XL4015"], boards["LM2596"] = pl

# ---------- 볼트(위상 a) ----------
bolt = lambda L, a: F.bolt(5.0, L, 1.25, 0.4, head_d=8.0, head_h=3.0).rotate([0, 0, a])
def pod_bolt(u, vr, side):
    return lambda a: P.to_world(bolt(P.LID_BOLT_L, a).rotate([180, 0, 0]).translate([u, (vr[0] + vr[1]) / 2, P.W_ROOF0 + P.ROOF]), side)
zt = A.TOP_Z0 + A.T
bolts = {}   # 이름 → (만드는 함수, 암나사 부품, 누르는 부품)
for i, (u, vr) in enumerate(P.R_BOSS): bolts[f"오른 덮개 볼트{i+1}"] = (pod_bolt(u, vr, 1), "오른 판", "오른 덮개")
for i, (u, vr) in enumerate(P.L_BOSS): bolts[f"왼 덮개 볼트{i+1}"] = (pod_bolt(u, vr, -1), "왼 판", "왼 덮개")
bolts["열화상 뚜껑 볼트"] = (lambda a: bolt(A.BOLT_L, a).rotate([180, 0, 0]).translate([A.TB_X, 0, zt + A.TH_H + A.TH_Z0 + A.CAP_T]), "팔", "센서 뚜껑")
bolts["레이더 뚜껑 볼트"] = (lambda a: bolt(A.BOLT_L, a).rotate([180, 0, 0]).translate([A.RB_X, A.RD_Y, A.RD_Z0 + A.RD_L + A.CAP_T]), "팔", "센서 뚜껑")
for side, nm in ((1, "누름쇠 왼(+y)"), (-1, "누름쇠 오른(−y)")):
    bolts[nm + " 볼트"] = ((lambda side: lambda a: S.pose(bolt(S.TAB_L, a).rotate([0, 90, 0]).translate([S.TAB_X0, side * S.TAB_Y, 0]), side))(side), "초음파 받침", nm)
bolts["누름쇠 가운데 볼트"] = (lambda a: (S.slot_bolt(S.CB_L).rotate([0, 0, a])).rotate([180, 0, 0]).translate([S.CB_X, 0, S.CP_Z1 + S.CT_T]), "초음파 받침", "누름쇠 가운데")

print("== 1. 부품·보드 서로 겹침 (0 이어야 함)")
allp = {**parts, **boards}; names = list(allp); bad = 0
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        x = vol(allp[names[i]] ^ allp[names[j]])
        if x > 0.01: print("  겹침", names[i], "↔", names[j], x); bad += 1
print("  겹치는 쌍", bad, "/", len(names) * (len(names) - 1) // 2)

print("== 2. 볼트: 암나사에 들어감(위상 돌려 최소 겹침 0), 누르는 부품·다른 것과 겹침 0, 물린 나사 길이")
bolt_m = {}
for nm, (fn, host, clamp) in bolts.items():
    best = min(((vol(fn(a) ^ parts[host]), a) for a in range(0, 360, 15)))
    b = fn(best[1]); bolt_m[nm] = b
    others = U([m for k, m in allp.items() if k != host])
    # 물림 길이: 볼트를 1.25 mm(한 피치) 빼는 쪽으로 못 움직이는지 = 산이 걸림. 반 피치 축 방향으로 밀어 겹치면 물림
    ax = cen(fn(0)) - cen(fn(0))  # 자리표시
    print(f"  {nm:14s} 암나사 겹침 {best[0]:5.2f}(위상 {best[1]:3d}°)  반 바퀴 어긋나면 {vol(fn(best[1] + 180) ^ parts[host]):6.2f}  다른 것과 {vol(b ^ others):.2f}")

print("== 3. 잡힘: 0.6 mm 밀었을 때 막히는 방향(±x ±y ±z 6 개 모두 막혀야 고정)")
dirs = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0), "+z": (0, 0, 1), "-z": (0, 0, -1)}
fixed = ("초음파 받침", "팔", "오른 판", "왼 판")          # LIMO 에 M3 로 조이는 것
world = {**allp, **bolt_m}
for nm, m in world.items():
    if nm in fixed: continue
    rest = U([o for k, o in world.items() if k != nm])
    free = [d for d, v in dirs.items() if vol(m.translate([0.6 * c for c in v]) ^ rest) < 0.02]
    print(f"  {nm:16s} {'고정' if not free else '풀린 방향: ' + ' '.join(free)}")

print("== 4. 끼우는 길 (0.5 mm 간격, 겹침 최대값·처음 걸리는 거리)")
lt = trimesh.load(os.path.join(D, "..", "탁구공발사대", "v4b", "ref", "limo", "limo_base.stl"))
v = lt.vertices; lt = trimesh.Trimesh(np.c_[-v[:, 1] + 14.5, v[:, 0], v[:, 2] - 145.5], lt.faces)
LP, _ = trimesh.sample.sample_surface(lt, 3000000); KT = cKDTree(LP)
def limo_d(m):
    me = m.to_mesh(); tm = trimesh.Trimesh(me.vert_properties[:, :3], me.tri_verts)
    p, _ = trimesh.sample.sample_surface(tm, 20000); d, _ = KT.query(np.r_[tm.vertices, p]); return d.min()
def sweep(nm, mover, obst, dist, limo=True):
    """mover(d) = 끼운 자리에서 d mm 뺀 모양. obst = 그때 이미 있는 것."""
    worst, first, dmin = 0.0, None, 99.0
    for d in np.arange(0.5, dist + 0.01, 0.5):
        m = mover(d); x = vol(m ^ obst)
        if x > 0.02 and first is None: first = d
        worst = max(worst, x)
        if limo and int(d * 2) % 4 == 0: dmin = min(dmin, limo_d(m))
    print(f"  {nm:34s} {'걸림 없음' if first is None else f'걸림: {first} mm 에서, 최대 {worst}'}" + (f"  LIMO 최소 {dmin:.1f}" if limo else ""))
c, s = math.cos(P.A), math.sin(P.A)
W = lambda side: np.array([0, c * side, s])
mv = lambda m, vec: (lambda d: m.translate(list(np.array(vec) * d)))
# 전장: 보드를 판에(바깥에서), 덮개를 그 위에
for side, pn, ln, bs in ((1, "오른 판", "오른 덮개", ("배선판", "ESP32")), (-1, "왼 판", "왼 덮개", ("XL4015", "LM2596"))):
    for b in bs: sweep(f"{b} → {pn} (바깥에서 곧게)", mv(boards[b], W(side)), parts[pn], 35)
    sweep(f"{ln} (바깥에서 곧게)", mv(parts[ln], W(side)), parts[pn] + U([boards[b] for b in bs]), 35)
# 팔: 보드 위에서, 뚜껑 위에서
sweep("열화상 보드 → 팔 (위에서)", mv(boards["열화상 보드"], (0, 0, 1)), parts["팔"], 25)
sweep("레이더 보드 → 팔 (위에서)", mv(boards["레이더 보드"], (0, 0, 1)), parts["팔"], 50)
sweep("센서 뚜껑 (위에서)", mv(parts["센서 뚜껑"], (0, 0, 1)), parts["팔"] + boards["열화상 보드"] + boards["레이더 보드"], 15)
# 초음파: 센서를 앞판 뒤에서 앞으로(국소 −x 로 뺌). 받침이 LIMO 에 달린 채 되는지 = LIMO 거리
sb = parts["초음파 받침"]
for side, bn in ((1, "HC-SR04 왼"), (-1, "HC-SR04 오른")):
    other = boards["HC-SR04 오른" if side > 0 else "HC-SR04 왼"]
    sweep(f"{bn} (뒤에서 곧게 16 mm)", (lambda side: lambda d: S.pose(S.dummy_sensor_local().translate([-d, 0, 0]), side))(side), sb, 16)
    sweep(f"{bn} (둘째로 넣을 때, 첫째 있음)", (lambda side: lambda d: S.pose(S.dummy_sensor_local().translate([-d, 0, 0]), side))(side), sb + other, 16, limo=False)
sens = boards["HC-SR04 왼"] + boards["HC-SR04 오른"]
for side, tn in ((1, "누름쇠 왼(+y)"), (-1, "누름쇠 오른(−y)")):
    sweep(f"{tn} (뒤로 곧게)", (lambda side: lambda d: S.pose(S.tab_local(side).translate([-d, 0, 0]), side))(side), sb + sens, 12)
ct = parts["누름쇠 가운데"]
sweep("누름쇠 가운데 (뒤로 곧게 12, LIMO 에 달기 전)", mv(ct, (-1, 0, 0)), sb + sens, 12, limo=False)
whole = sb + sens + U(tabs)
print("  초음파 받침 조립품을 LIMO 밑에서 올려 붙임(아래로 d 내렸을 때 LIMO 최소 거리):", [(d, round(limo_d(whole.translate([0, 0, -d])), 1)) for d in (1, 3, 6, 10, 20)])

print("== 5. M3 드라이버 길 (Ø6.5 × 80 원통이 자기 부품·이미 달린 것과 겹침 0)")
drv = m3.Manifold.cylinder(80, 3.25, 3.25, 32)
for (x, y) in A.HOLES:
    print("  팔 지붕 M3", (x, y), vol(drv.translate([x, y, A.ROOF_Z + A.T]) ^ parts["팔"]))
for side, pn in ((1, "오른 판"), (-1, "왼 판")):
    for x in P.HOLES_X:
        cyl = P.to_world(P.cyl_horiz(3.1, P.SPOT_T0 + 0.05, P.SPOT_T0 + 80, x), side)
        print(f"  {pn} M3 x {x}", vol(cyl ^ parts[pn]))
for (x, y) in S.HOLES_F + S.HOLES_R:
    print("  초음파 밑면 M3", (x, y), vol(drv.rotate([180, 0, 0]).translate([x, y, S.Z_BOT - S.T_BASE]) ^ sb))
# 초음파 바깥 누름쇠 볼트 드라이버(뒤에서): LIMO 와 거리
for side in (1, -1):
    path = S.pose(m3.Manifold.cylinder(60, 4.5, 4.5, 32).rotate([0, -90, 0]).translate([S.TAB_X0 - 3.0, side * S.TAB_Y, 0]), side)
    print(f"  바깥 누름쇠 볼트 뒤쪽 길(Ø9 × 60) side {side}: 부품과 겹침 {vol(path ^ U(list(allp.values())))}  LIMO 최소 {limo_d(path):.1f}")

cb = m3.Manifold.cylinder(60, 3.25, 3.25, 32).translate([S.CB_X, 0, S.CP_Z1 + S.CT_T + 3.0])
print(f"  가운데 볼트 드라이버 길(위, Ø6.5 × 60): 부품·보드와 겹침 {vol(cb ^ U(list(allp.values())))}  LIMO 최소 {limo_d(cb):.1f}")
print("== 6. LIMO 표면까지 최소 거리")
for nm, m in {**parts, **bolt_m}.items():
    print(f"  {nm:16s} {limo_d(m):.2f}")
