"""초음파 받침 v4 단독 검사 (python check_sonar.py). build_sonar 의 모양을 직접 만들어 검사한다.
(a) 덩어리 수  (b) 겹침·볼트 나사 위상  (c) 0.6 mm 밀어 6 방향 잡힘  (d) 끼우는 길 0.5 mm 간격
(e) 드라이버 길  (f) 원통 4 개의 ±15° 빔 원뿔  (g) LIMO 거리·무게·출력 자세 오버행
LIMO 메시(render/asm/limo.stl)는 닫힌 덩어리가 아니라(겉면 조각 1665 개) 부피 겹침을 못 잰다 →
앞부분 겉면을 촘촘히 찍은 점(간격 ≈ 0.1 mm)까지의 최소 거리로 본다. 거리가 0.15 넘으면 겉면을 안 뚫은 것."""
import os, sys, math, subprocess, faulthandler
faulthandler.enable()
import numpy as np, manifold3d as m3, trimesh
from scipy.spatial import cKDTree
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D); sys.path.insert(0, os.path.join(D, "..", "lib"))
import build_sonar as S

vol = lambda a: round(a.volume(), 2)
U = lambda ms: sum(ms, m3.Manifold())
FAIL = []
def ok(cond, msg):
    if not cond: FAIL.append(msg)
    return "OK " if cond else "FAIL"

# ---------- 부품 ----------
mount = S.build()
tabs = {"누름쇠 왼": S.pose(S.tab_local(1), 1), "누름쇠 오른": S.pose(S.tab_local(-1), -1), "누름쇠 가운데": S.center_tab()}
boards = {"HC-SR04 왼": S.pose(S.dummy_sensor_local(), 1), "HC-SR04 오른": S.pose(S.dummy_sensor_local(), -1)}
wires = S.pose(S.wire_bend_local(), 1) + S.pose(S.wire_bend_local(), -1)
bolt_fn = {"볼트 왼": (lambda a: S.outer_bolt(1, a), "누름쇠 왼"), "볼트 오른": (lambda a: S.outer_bolt(-1, a), "누름쇠 오른"),
           "볼트 가운데": (lambda a: S.center_bolt(a), "누름쇠 가운데")}

# ---------- LIMO 앞부분 겉면 점 ----------
lt = trimesh.load(os.path.join(D, "render", "asm", "limo.stl"))
fc = lt.triangles_center; keep = (fc[:, 0] > 105) & (fc[:, 2] < -20)
lf = trimesh.Trimesh(lt.vertices, lt.faces[keep], process=False)
np.random.seed(1)
LP, _ = trimesh.sample.sample_surface(lf, 3000000); LP = np.r_[LP, lf.vertices[np.unique(lf.faces)]]
KT = cKDTree(LP)
def pts(m, n=40000):
    me = m.to_mesh(); tm = trimesh.Trimesh(me.vert_properties[:, :3], me.tri_verts, process=False)
    p, _ = trimesh.sample.sample_surface(tm, n); return np.r_[tm.vertices, p]
def limo_d(P): return KT.query(P, workers=-1)[0].min()

print("== (a) 출력 부품 덩어리 수 (1 이어야 함)")
pp = S.print_parts()
for nm, m in {"받침": mount, "바깥 누름쇠 왼": S.tab_local(1), "바깥 누름쇠 오른": S.tab_local(-1), "가운데 누름쇠": S.center_tab(), "볼트": S.pan(S.TAB_L)}.items():
    n = len(m.decompose()); print(f"  {ok(n == 1, nm + ' 덩어리')} {nm:10s} {n}")

print("== (b) 겹침 (0 이어야 함)")
bolts = {}
for nm, (fn, tab) in bolt_fn.items():
    best = min(((vol(fn(a) ^ mount), a) for a in range(0, 360, 15)))
    off = vol(fn(best[1] + 180) ^ mount); bolts[nm] = fn(best[1])
    print(f"  {ok(best[0] < 0.01 and off > 0.5, nm + ' 나사 위상')} {nm:8s} 암나사 겹침 최소 {best[0]:.2f} (위상 {best[1]}°), 반 바퀴 어긋나면 {off:.2f}")
allp = {"받침": mount, **tabs, **bolts, **boards, "선 꺾임": wires}
names = list(allp); bad = 0
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        x = vol(allp[names[i]] ^ allp[names[j]])
        if x > 0.01: print("  겹침", names[i], "↔", names[j], x); bad += 1
print(f"  {ok(bad == 0, '부품 겹침')} 겹치는 쌍 {bad} / {len(names) * (len(names) - 1) // 2}")
P_all = {k: pts(m) for k, m in allp.items()}
mp = P_all["받침"]; above = mp[mp[:, 2] > S.Z_BOT + 0.5]
for nm in names:
    d = limo_d(above if nm == "받침" else P_all[nm])
    print(f"  {ok(d > 0.5, nm + ' LIMO 거리')} {nm:12s} LIMO 겉면까지 최소 {d:.2f}" + ("  (밑판 뺀 나머지)" if nm == "받침" else ""))
top = mp[(mp[:, 2] > S.Z_BOT - 0.02)]; under = top[top[:, 0] < 160.0]
print(f"  받침 밑판 윗면 z 최대 {mp[mp[:, 0] < 160][:, 2].max():.3f} (차체 아랫면 −118.000 에 닿음, 그 위로 안 올라감: {ok(mp[mp[:, 0] < 160][:, 2].max() <= S.Z_BOT + 1e-3, '밑판이 차체 안으로')})")

print("== (c) 잡힘: 0.6 mm 밀었을 때 막히는 방향 (6 방향 모두 막혀야 함)")
dirs = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0), "+z": (0, 0, 1), "-z": (0, 0, -1)}
world = {k: v for k, v in allp.items() if k != "선 꺾임"}
for nm, m in world.items():
    if nm == "받침": continue
    rest = U([o for k, o in world.items() if k != nm])
    free = [d for d, v in dirs.items() if vol(m.translate([0.6 * c for c in v]) ^ rest) < 0.02]
    print(f"  {ok(not free, nm + ' 잡힘')} {nm:12s} {'6 방향 막힘' if not free else '풀린 방향: ' + ' '.join(free)}")

print("== (d) 끼우는 길 (0.5 mm 간격으로 빼 보며 겹침)")
def sweep(nm, mover, obst, dist):
    worst, first = 0.0, None
    for d in np.arange(0.5, dist + 0.01, 0.5):
        x = vol(mover(d) ^ obst)
        if x > 0.02 and first is None: first = d
        worst = max(worst, x)
    print(f"  {ok(first is None, nm)} {nm:44s} {'걸림 없음' if first is None else f'걸림: {first} mm 에서, 최대 {worst}'}")
dl = S.dummy_sensor_local() + S.wire_bend_local()
for side, bn, on in ((1, "HC-SR04 왼", "HC-SR04 오른"), (-1, "HC-SR04 오른", "HC-SR04 왼")):
    mv = (lambda side: lambda d: S.pose(dl.translate([-d, 0, 0]), side))(side)
    sweep(f"{bn} → 빈 받침 (뒤에서 곧게 30)", mv, mount, 30)
    sweep(f"{bn} 둘째로 (다른 보드 있음)", mv, mount + boards[on], 30)
sens = U(list(boards.values())) + wires
for side, tn in ((1, "누름쇠 왼"), (-1, "누름쇠 오른")):
    sweep(f"{tn} (센서 넣은 뒤, 뒤로 곧게 20)", (lambda side: lambda d: S.pose(S.tab_local(side).translate([-d, 0, 0]), side))(side),
          mount + sens + tabs["누름쇠 가운데"], 20)
sweep("누름쇠 가운데 (센서·바깥 누름쇠 있음, 뒤로 20)", lambda d: tabs["누름쇠 가운데"].translate([-d, 0, 0]), mount + sens + tabs["누름쇠 왼"] + tabs["누름쇠 오른"], 20)
for nm, (fn, tab) in bolt_fn.items():      # 볼트를 축 방향으로 뺄 때 누름쇠·보드에 안 걸리는지(받침 암나사는 돌려 넣으므로 뺌)
    ax = np.array(S.pose(S.box(-1, 0, -.01, .01, -.01, .01), 1 if "왼" in nm else -1).bounding_box()) if "가운데" not in nm else None
    u = np.array([-1.0, 0, 0]) if ax is None else None
    if u is None:
        side = 1 if "왼" in nm else -1
        c0 = np.array(S.pose(S.box(-.01, .01, -.01, .01, -.01, .01), side).bounding_box()).reshape(2, 3).mean(0)
        c1 = np.array(S.pose(S.box(-1.01, -0.99, -.01, .01, -.01, .01), side).bounding_box()).reshape(2, 3).mean(0); u = c1 - c0
    sweep(f"{nm} (뒤로 곧게 20, 누름쇠·센서와)", (lambda b, u: lambda d: b.translate(list(u * d)))(bolts[nm], u), U(list(tabs.values())) + sens, 20)
# 조립품을 LIMO 에: 앞·아래 비스듬히(수평에서 45° 아래)로 와서 마지막 2 mm 는 곧게 위로
whole_p = np.r_[tuple(P_all.values())]; whole_p = whole_p[np.random.permutation(len(whole_p))[:60000]]    # 길 검사는 6 만 점으로
def path(d):
    if d <= 2.0: return np.array([0, 0, -d])
    return np.array([(d - 2.0) * math.cos(math.radians(45)), 0, -2.0 - (d - 2.0) * math.sin(math.radians(45))])
res = [(d, limo_d(whole_p + path(d))) for d in np.arange(0.5, 90.01, 0.5)]
dmin = min(res, key=lambda r: r[1])
mono = all(res[i + 1][1] >= res[i][1] - 0.05 for i in range(len(res) - 1))
print(f"  {ok(dmin[1] >= 0.5 - 0.02, '조립품 접근 길')} 조립품(받침+센서+누름쇠+볼트+선) → LIMO: 자리에서 d 뺀 곳의 LIMO 최소 거리, 0.5 mm 간격 0.5 … 90")
print("     ", ", ".join(f"d {d:g}: {x:.2f}" for d, x in res if d in (0.5, 1, 2, 3, 5, 10, 20, 40, 90)), f" | 가장 가까운 때 d {dmin[0]:g} → {dmin[1]:.2f}, 멀어지기만 함: {mono}")
print("      (자리에 앉으면 밑판 윗면이 차체 아랫면에 닿음 = 0)")

print("== (e) 드라이버 길")
everything = U(list(allp.values()))
drv = m3.Manifold.cylinder(80, 3.25, 3.25, 32)
for (x, y) in S.HOLES_F + S.HOLES_R:
    c = drv.rotate([180, 0, 0]).translate([x, y, S.Z_BOT - S.T_BASE])
    v = vol(c ^ everything)
    print(f"  {ok(v < 0.01, 'M3 드라이버 길')} 밑면 M3 {(x, y)} Ø6.5 × 80 아래로: 부품과 겹침 {v}, LIMO 최소 {limo_d(pts(c, 5000)):.1f}")
bd = m3.Manifold.cylinder(60, 3.25, 3.25, 32).rotate([0, -90, 0])      # 머리 뒤(−x)로 60
for side, nm in ((1, "볼트 왼"), (-1, "볼트 오른")):
    c = S.pose(bd.translate([S.TAB_X0 - S.HEAD_H - 0.3, side * S.TAB_Y, S.TAB_Z]), side)
    print(f"  {ok(vol(c ^ everything) < 0.01, nm + ' 드라이버 길')} {nm} Ø6.5 × 60 뒤로(LIMO 에 달기 전): 부품과 겹침 {vol(c ^ everything)}")
c = bd.translate([S.CB_X - S.HEAD_H - 0.3, 0, S.CP_Z1])
print(f"  {ok(vol(c ^ everything) < 0.01, '볼트 가운데 드라이버 길')} 볼트 가운데 Ø6.5 × 60 뒤로(LIMO 에 달기 전): 부품과 겹침 {vol(c ^ everything)}")

print("== (f) 빔 원뿔 ±15° (원통 앞면 Ø16 에서 300 mm) 안에 받침·누름쇠·볼트·다른 센서·LIMO 가 없어야 함")
Lc = 300.0; r1 = 8.0 + Lc * math.tan(math.radians(15))
cone_l = m3.Manifold.cylinder(Lc, 8.0, r1, 64).rotate([0, 90, 0])
for side in (1, -1):
    for s in (-1, 1):
        x0 = -S.BEZEL_T + S.CAN_H + 0.05
        cone = S.pose(cone_l.translate([x0, s * S.CAN_GAP / 2, 0]), side)
        others = U([m for k, m in allp.items() if k != ("HC-SR04 왼" if side > 0 else "HC-SR04 오른")])
        o = np.array(S.pose(S.box(x0 - .01, x0 + .01, s * 13 - .01, s * 13 + .01, -.01, .01), side).bounding_box()).reshape(2, 3).mean(0)
        o2 = np.array(S.pose(S.box(x0 + 99.99, x0 + 100.01, s * 13 - .01, s * 13 + .01, -.01, .01), side).bounding_box()).reshape(2, 3).mean(0)
        u = (o2 - o) / 100.0; w = LP - o; sx = w @ u; rad = np.linalg.norm(w - np.outer(sx, u), axis=1)
        nin = int(((sx > 0) & (sx < Lc) & (rad < 8.0 + sx * math.tan(math.radians(15)))).sum())
        v = vol(cone ^ others)
        print(f"  {ok(v < 0.01 and nin == 0, '빔 원뿔')} {'왼' if side > 0 else '오른'} 센서 원통 {'안쪽' if s * side < 0 else '바깥'}: 부품과 겹침 {v}, 원뿔 안 LIMO 점 {nin}")
# 두 빔 중심선이 만나는 곳·덮는 폭
yin = S.SENSOR_Y - (13 + 8) * math.cos(math.radians(S.YAW))      # 안쪽 원통의 안쪽 가장자리 |y|
print(f"  두 센서 축: 좌우 ±{S.YAW}°, 위 {S.PITCH}°, 원통 중심 z {S.SENSOR_Z}. 안쪽 원통 가장자리 |y| {yin:.1f}, ±15° 빔 안쪽 가장자리는 가운데로 {15 - S.YAW:.0f}° → "
      f"원통 앞 {yin / math.tan(math.radians(15 - S.YAW)):.0f} mm 에서 두 빔이 만나고 그 뒤로는 가운데 빈 띠 없음")
# 위에서 볼 때 뒤쪽(보드·누름쇠·볼트)이 처마에 가려지는지
backs = U(list(tabs.values())) + U(list(bolts.values())) + U([b ^ S.pose(S.box(-60, -S.BEZEL_T, -40, 40, -11, 40), sd) for b, sd in ((boards["HC-SR04 왼"], 1), (boards["HC-SR04 오른"], -1))])
pa = backs.project(); seen = pa - mount.project()
print(f"  {ok(seen.area() < 0.02 * pa.area(), '처마가 뒤를 가림')} 바로 위에서 볼 때 보드(아랫변 위)·누름쇠·볼트 넓이 {pa.area():.0f} mm² 중 처마 밖으로 보이는 넓이 {seen.area():.1f} mm²")
print("== (g) 크기·무게·출력 자세 오버행")
bb = np.array(mount.bounding_box()).reshape(2, 3)
print(f"  받침 크기 x {bb[0][0]:.1f} … {bb[1][0]:.1f}, y {bb[0][1]:.1f} … {bb[1][1]:.1f}, z {bb[0][2]:.1f} … {bb[1][2]:.1f}  ({ok(bb[1][1] <= S.Y_MAX + 0.01 and bb[0][2] >= -121.1, '받침 범위')} |y| ≤ {S.Y_MAX}, z ≥ −121.1)")
ab = np.array(everything.bounding_box()).reshape(2, 3)
print(f"  조립품(센서·선 포함) x 최대 {ab[1][0]:.1f} (범퍼 앞 끝 173.0 에서 {ab[1][0] - 173.0:.1f}), |y| 최대 {max(-ab[0][1], ab[1][1]):.1f}")
g = lambda m: m.volume() * 1.24e-3
mt = g(mount); tt = sum(g(m) for m in tabs.values()); bt = sum(g(m) for m in bolts.values())
print(f"  {ok(mt + tt + bt <= 50.0, '무게')} 무게(속 찬 PLA 1.24): 받침 {mt:.1f} g, 누름쇠 3 개 {tt:.1f} g, 볼트 3 개 {bt:.1f} g, 합 {mt + tt + bt:.1f} g")
tool = os.path.expanduser(r"~\.claude\skills\bambu-studio-gui\scripts\overhang_regions.py")
if os.path.exists(tool):
    for k in ("print_sonar.stl", "print_sonar_tab.stl", "print_sonar_tab_r.stl", "print_sonar_ctab.stl", "print_sonar_bolt.stl"):
        pth = os.path.join(S.OUT, k); pp[k].export(pth)
        r = subprocess.run([sys.executable, tool, pth, "--top", "8"], capture_output=True, text=True, encoding="utf-8", errors="replace")
        print(f"  --- {k}"); print("   " + (r.stdout + r.stderr).strip().replace("\n", "\n   "))
print("\n결과:", "모두 통과" if not FAIL else f"실패 {len(FAIL)}: " + "; ".join(FAIL))
