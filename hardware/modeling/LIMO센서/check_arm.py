"""열화상·레이더 팔 v4 검증(지붕 좌표, 혼자 돌아감). build_arm 에서 모양을 바로 만들고 LIMO 는 render/asm/limo.stl.
(a) 덩어리  (b) 겹침·볼트 물림  (c) 6 방향 잡힘  (d) 끼우는 길  (e) M3 드라이버 길  (f) 시야  (g) LIMO 거리·높이·무게·서포트
(h) 발사대 5~30° 전체 검사는 탁구공발사대/v4b/check_full_limo.py 를 따로 돌린다."""
import math, os, subprocess, sys
import numpy as np, manifold3d as m3, trimesh
from scipy.spatial import cKDTree

D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
sys.stdout.reconfigure(encoding="utf-8")
import build_arm as A

vol = lambda a: round(a.volume(), 3)
U = lambda ms: sum(ms, m3.Manifold())
tm = A.to_t
OK = lambda c: "통과" if c else "실패"
fails = []
def judge(name, cond):
    if not cond: fails.append(name)
    return OK(cond)

arm, holder = A.build(), A.radar_holder()
cap, _ = A.caps()
th, rd = A.dummies()
limo = trimesh.load(os.path.join(D, "render", "asm", "limo.stl"))
LP, _ = trimesh.sample.sample_surface(limo, 4000000, seed=1); KT = cKDTree(LP)

def pts_of(m, n=60000):
    t = tm(m); p, _ = trimesh.sample.sample_surface(t, n, seed=2); return np.r_[t.vertices, p]
def limo_d(m, n=60000):
    p = pts_of(m, n); d, _ = KT.query(p); k = d.argmin(); return d[k], p[k]

print("== (a) 덩어리 수·닫힘 (출력 파일)")
for nm, m, fn in (("팔", arm, "print_arm.stl"), ("레이더 받침", holder, "print_arm_holder.stl"), ("뚜껑", cap, "print_cap.stl"), ("볼트", A.cap_bolt(), "print_arm_bolt.stl")):
    t = trimesh.load(os.path.join(D, "print", fn))
    mm = m3.Manifold(m3.Mesh(t.vertices.astype(np.float32), t.faces.astype(np.uint32)))      # 출력 파일을 다시 읽어 manifold 로
    nb = len(mm.decompose()) if mm.status() == m3.Error.NoError else -1
    e4 = int((np.unique(t.edges_sorted, axis=0, return_counts=True)[1] != 2).sum())
    print(f"  {nm:8s} 출력 STL: manifold {mm.status().name}, 덩어리 {nb}, 부피 {mm.volume() / 1000:.2f} cm3, 면 2 개가 아닌 모서리 {e4} (trimesh 닫힘 {t.is_watertight}), 크기 {np.round(t.extents, 1).tolist()}  {judge('a ' + nm, nb == 1)}")
print("== (b) 겹침 (mm3, 0 이어야 함)")
phase = [min(((A._bolt_at(x, y, a) ^ arm).volume(), a) for a in range(0, 360, 15)) for (x, y) in A.BOLT_POS]
bolts = [A._bolt_at(x, y, ph[1]) for (x, y), ph in zip(A.BOLT_POS, phase)]
for nm, (x, y), ph in zip(("열화상 쪽 볼트", "레이더 쪽 볼트"), A.BOLT_POS, phase):
    off = vol(A._bolt_at(x, y, ph[1] + 180) ^ arm)
    print(f"  {nm}: 암나사 겹침 {ph[0]:.3f} (위상 {ph[1]}°), 반 바퀴 어긋나면 {off}  {judge('b 나사 ' + nm, ph[0] < 0.01 and off > 0.5)}")
P = {"팔": arm, "받침": holder, "뚜껑": cap, "볼트1": bolts[0], "볼트2": bolts[1], "열화상 보드": th, "레이더 보드": rd}
ks = list(P); bad = 0
for i in range(len(ks)):
    for j in range(i + 1, len(ks)):
        x = vol(P[ks[i]] ^ P[ks[j]])
        if x > 0.01: print("  겹침", ks[i], "↔", ks[j], x); bad += 1
print(f"  겹치는 쌍 {bad} / {len(ks) * (len(ks) - 1) // 2}  {judge('b 겹침', bad == 0)}")
def inside(t, q, chunk=400):
    """점 q 가 닫힌 메시 t 안인지: +z 광선이 면을 홀수 번 지나는지(xy 투영 무게중심 좌표)."""
    tri = t.triangles; a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    v0, v1 = (b - a)[:, :2], (c - a)[:, :2]; den = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]; ok = np.abs(den) > 1e-12
    out = np.zeros(len(q), bool)
    for i in range(0, len(q), chunk):
        p = q[i:i + chunk]; w = p[:, None, :2] - a[None, :, :2]
        s_ = (w[..., 0] * v1[None, :, 1] - w[..., 1] * v1[None, :, 0]) / np.where(ok, den, 1)
        t_ = (v0[None, :, 0] * w[..., 1] - v0[None, :, 1] * w[..., 0]) / np.where(ok, den, 1)
        hit = ok[None] & (s_ >= 0) & (t_ >= 0) & (s_ + t_ <= 1)
        z = a[None, :, 2] + s_ * (b - a)[None, :, 2] + t_ * (c - a)[None, :, 2]
        out[i:i + chunk] = ((hit & (z > p[:, None, 2])).sum(1) % 2) == 1
    return out
ins = 0
for nm, m in P.items():      # LIMO 표면 점이 부품 속에 있는지(겉면에서 0.1 mm 넘게 안쪽)
    t = tm(m); bb = t.bounds; q = LP[np.all((LP > bb[0]) & (LP < bb[1]), axis=1)]; n0 = len(q)
    if n0 > 6000: q = q[np.random.default_rng(0).choice(n0, 6000, False)]
    k = 0
    if len(q):
        hit = q[inside(t, q)]
        if len(hit): k = int((cKDTree(pts_of(m, 300000)).query(hit)[0] > 0.1).sum())
    ins += k
    print(f"  LIMO 표면 점 {n0:6d} 개가 {nm} 상자 안, 그중 {len(q)} 개를 검사 → 부품 속에 있는 점 {k}")
print(f"  LIMO 표면 점이 부품 속에 들어간 수 {ins}  {judge('b LIMO', ins == 0)}")
print("== (c) 0.6 mm 밀었을 때 6 방향 모두 막히는지")
dirs = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0), "+z": (0, 0, 1), "-z": (0, 0, -1)}
for nm, m in P.items():
    if nm == "팔": continue
    rest = U([o for k, o in P.items() if k != nm])
    free = [d for d, v in dirs.items() if vol(m.translate([0.6 * c for c in v]) ^ rest) < 0.02]
    extra = ""
    if nm == "레이더 보드":
        for dn, v in (("+긴 축", A.RD_U), ("-긴 축", -A.RD_U), ("+법선", A.RD_N), ("-법선", -A.RD_N)):
            if vol(m.translate([float(0.6 * c) for c in v]) ^ rest) < 0.02: free.append(dn)
        extra = " (기운 축 4 방향 포함)"
    print(f"  {nm:8s} {'고정' if not free else '풀린 방향: ' + ' '.join(free)}{extra}  {judge('c ' + nm, not free)}")
print("  닿는 곳: 열화상 보드 = 아래 판·앞뒤 바닥 턱 / 뚜껑 앞 치마·옆 볼·뒤 턱 / 위 뚜껑 밑면.  레이더 보드 = 받침 홈(앞 턱·뒤 받침·옆벽)·아래 모서리 빗면 2 / 위는 뚜껑 밑면이 보드 앞 윗모서리 선(15 mm)을 누름")
for nm, a, b in (("뚜껑 밑면 ↔ 열화상 보드 윗변", cap, th), ("뚜껑 밑면 ↔ 레이더 보드 윗모서리", cap, rd)):
    g = [d for d in np.arange(0.05, 0.61, 0.05) if vol(b.translate([0, 0, float(d)]) ^ a) > 0.001]
    print(f"  {nm}: 위로 {g[0] if g else '>0.6'} mm 에서 닿음")

print("== (d) 끼우는 길 (0.5 mm 간격, 팔은 LIMO 에 달린 채)")
def sweep(nm, mover_vec, m, obst, dist):
    worst, first, dmin = 0.0, None, 99.0
    for d in np.arange(0.5, dist + 0.01, 0.5):
        mm = m.translate([float(c * d) for c in mover_vec]); x = vol(mm ^ obst)
        if x > 0.02 and first is None: first = d
        worst = max(worst, x)
        if int(d * 2) % 4 == 0: dmin = min(dmin, limo_d(mm, 8000)[0])
    print(f"  {nm:40s} {'걸림 없음' if first is None else f'걸림: {first} mm 에서, 최대 {worst}'}  LIMO 최소 {dmin:.1f}  {judge('d ' + nm, first is None and dmin > 0.5)}")
sweep("열화상 보드 ← 위에서 25", (0, 0, 1), th, arm + holder + rd, 25)
sweep("레이더 받침 ← +y 에서 혀로 25", (0, 1, 0), holder, arm + th, 25)
sweep("레이더 받침(보드 꽂은 채) ← +y 에서 25", (0, 1, 0), holder + rd, arm + th, 25)
sweep("레이더 보드 ← 기운 축 따라 위에서 50", A.RD_U, rd, arm + holder + th, 50)
sweep("뚜껑 ← 위에서 25", (0, 0, 1), cap, arm + holder + th + rd, 25)

print("== (e) M3 드라이버 길 (Ø6.5 × 80, 밑판 윗면 z 6.25 에서 위로)")
drv = m3.Manifold.cylinder(80, 3.25, 3.25, 48)
everything = U(list(P.values()))
for (x, y) in A.HOLES:
    c = drv.translate([x, y, A.ROOF_Z + A.T]); o = vol(c ^ everything); dl = limo_d(c, 20000)[0]
    print(f"  구멍 {(x, y)}: 부품·보드와 겹침 {o}, LIMO 최소 {dl:.2f}  {judge('e 드라이버', o < 0.01 and dl > 0.3)}")

print("== (f) 시야 (장애물 표면 점이 시야 뿔 안에 있는지, 300 mm 안)")
obst = {"팔": pts_of(arm, 400000), "받침": pts_of(holder, 400000), "뚜껑": pts_of(cap, 400000), "볼트": pts_of(bolts[0] + bolts[1], 50000), "열화상 보드": pts_of(th, 50000), "레이더 보드": pts_of(rd, 50000), "LIMO": LP}
def fov(name, origins, n, u, y, haz, hel, skip=(), rng=300.0):
    worst = {}
    tot = 0
    for k, p in obst.items():
        if k in skip: continue
        best = (999, None, None)
        cnt = 0
        for o in origins:
            d = p - o; dn = d @ n; dy = d @ y; du = d @ u; r = np.linalg.norm(d, axis=1)
            m = (dn > 0.02) & (r < rng)
            if not m.any(): continue
            az = np.degrees(np.arctan2(np.abs(dy[m]), dn[m])); el = np.degrees(np.arctan2(np.abs(du[m]), dn[m]))
            ex = np.maximum(az - haz, el - hel)           # < 0 이면 시야 안
            cnt += int((ex < 0).sum()); i = ex.argmin()
            if ex[i] < best[0]: best = (ex[i], p[m][i], r[m][i])
        tot += cnt
        if best[1] is not None:
            worst[k] = best
            print(f"    {k:8s} 시야 안 점 {cnt:6d}   가장 가까이 온 점: 시야 끝에서 {best[0]:+.1f}° ({'안' if best[0] < 0 else '밖'}), 거리 {best[2]:.0f} mm, 위치 {np.round(best[1], 1).tolist()}")
    print(f"  → {name}: {judge('f ' + name, tot == 0)}")
ey = np.array([0.0, 1.0, 0.0])
org = [A.RD_PB + 0.02 * A.RD_N + b * ey + c * A.RD_U for b in (-6, -3, 0, 3, 6) for c in (2, 7, 12, 17, 22, 27, 32, 37, 42)]
print("  레이더 ±60° × ±35° (보드 법선 기준, 가운데 12 × 40 의 45 점)")
fov("레이더 시야", org, A.RD_N, A.RD_U, ey, 60.0, 35.0, skip=("레이더 보드",))
ex, ez = np.array([1.0, 0, 0]), np.array([0, 0, 1.0])
for haz, hel in ((27.5, 17.5), (55.0, 37.5)):
    print(f"  열화상 {2 * haz:.0f}° × {2 * hel:.0f}° (렌즈 통 앞 끝 가운데, 렌즈 위치 후보 y {A.TH_LENS_Y})")
    fov(f"열화상 시야 {2 * haz:.0f}×{2 * hel:.0f}", [np.array([A.TH_X + A.TH_T + 6.0, yl, A.TH_LENS_Z]) for yl in A.TH_LENS_Y], ex, ez, ey, haz, hel, skip=("열화상 보드",))

print("== (g) LIMO 까지 최소 거리(밑판 빼고 ≥ 1), 가장 높은 곳, 무게, 서포트")
foot_cut = arm - A.box(30, 53.5, -20, 20, -5, A.ROOF_Z + A.T + 0.01)
for nm, m in (("팔(밑판 뺌)", foot_cut), ("받침", holder), ("뚜껑", cap), ("볼트", bolts[0] + bolts[1])):
    d, p = limo_d(m, 300000)
    print(f"  {nm:10s} {d:.2f} mm at {np.round(p, 1).tolist()}  {judge('g 거리 ' + nm, d >= 1.0)}")
d, p = limo_d(arm, 300000); print(f"  팔(밑판 포함) {d:.2f} mm at {np.round(p, 1).tolist()} (지붕에 닿는 면)")
zmax = max(tm(m).bounds[1][2] for m in P.values())
print(f"  가장 높은 곳 z {zmax:.2f}  {judge('g 높이', zmax <= 76.0)}")
g = {nm: m.volume() * 1.24 / 1000 for nm, m in (("팔", arm), ("받침", holder), ("뚜껑", cap))}
print("  무게(속 찬 기준 g):", {k: round(v, 1) for k, v in g.items()}, "합", round(sum(g.values()), 1), "+ 볼트 2 개", round(2 * A.cap_bolt().volume() * 1.24 / 1000, 1), judge("g 무게", sum(g.values()) <= 42.0))
tool = os.path.expanduser(r"~\.claude\skills\bambu-studio-gui\scripts\overhang_regions.py")
for fn in ("print_arm.stl", "print_arm_holder.stl", "print_cap.stl", "print_arm_bolt.stl"):
    r = subprocess.run([sys.executable, tool, os.path.join(D, "print", fn), "--top", "8"], capture_output=True, text=True, encoding="utf-8")
    print("  --", fn); print("\n".join("     " + l for l in (r.stdout + r.stderr).strip().split("\n")))
print("== 실패:", fails if fails else "없음")

if "--full" in sys.argv:
    # (h) 발사대 5~30° 전체 검사: 탁구공발사대/v4b/check_full_limo.py 를 고치지 않고 그대로 읽어, 받침(asm_arm_holder.stl)만 목록에 더해 돌린다
    FD = os.path.join(D, "..", "탁구공발사대", "v4b"); fp = os.path.join(FD, "check_full_limo.py")
    src = open(fp, encoding="utf-8").read()
    key = '("arm_bolts", "asm_arm_bolts.stl"),'
    assert key in src
    src = src.replace(key, key + ' ("arm_holder", "asm_arm_holder.stl"),')
    print("== (h) check_full_limo.py (+ asm_arm_holder.stl)")
    sys.path.insert(0, FD)
    exec(compile(src, fp, "exec"), {"__file__": fp, "__name__": "__main__"})

