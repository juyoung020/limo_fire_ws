"""초음파 띠 v5(테이프로 붙임) 단독 검사 (python check_sonar.py). build_sonar 의 모양을 직접 만들어 검사한다.
(a) 덩어리  (b) 겹침  (c) LIMO 거리  (d) 자리 잡힘(0.6 mm 밀기)  (e) 보드·듀폰 끼우는 길 0.5 mm 간격
(f) 빔 원뿔 ±15°  (g) 라이다 여유  (h) 모서리 램프 가림  (i) 바퀴 틈(조향 ±30°, 5° 간격)  (j) 크기·무게·오버행  (k) 테이프 하중 손계산
LIMO 메시(render/asm/limo.stl)는 닫힌 덩어리가 아니라 부피 겹침을 못 잰다 → 앞부분 겉면에 촘촘히 찍은 점까지의 최소 거리로 본다.
rtree 가 없어 광선은 점을 걸어가며 cKDTree 로 잰다(check_lamp.py 와 같은 방식).
이전(나사) 설계 검사는 check_sonar_v4.py (build_sonar_v4.py 를 build_sonar 로 되돌려야 돈다)."""
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
    return "OK  " if cond else "FAIL"

# ---------- 부품 ----------
bar = S.build()
boards = {"HC-SR04 왼": S.pose(S.dummy_sensor_local(), 1), "HC-SR04 오른": S.pose(S.dummy_sensor_local(), -1)}
wires = S.pose(S.wire_bend_local(), 1) + S.pose(S.wire_bend_local(), -1)
tA, tC, tB = S.tape_dummy()
tapes = {"테이프 위 패드": tA, "테이프 아래 패드": tC, "테이프 보드": tB}

# ---------- LIMO 앞부분 겉면 점 ----------
lt = trimesh.load(os.path.join(D, "render", "asm", "limo.stl"))
fc = lt.triangles_center; keep = (fc[:, 0] > 105) & (fc[:, 2] < -5)
lf = trimesh.Trimesh(lt.vertices, lt.faces[keep], process=False)
np.random.seed(1)
LP, LFI = trimesh.sample.sample_surface(lf, 4000000); LN = lf.face_normals[LFI]
KT = cKDTree(LP)
def pts(m, n=40000):
    me = m.to_mesh(); tm = trimesh.Trimesh(me.vert_properties[:, :3], me.tri_verts, process=False)
    p, _ = trimesh.sample.sample_surface(tm, n); return np.r_[tm.vertices, p]
def limo_d(P): return KT.query(P, workers=-1)[0].min()
BP = pts(bar, 400000); BD = KT.query(BP, workers=-1)[0]

print("== (a) 덩어리 수")
n = len(bar.decompose()); print(f"  {ok(n == 1, '띠 덩어리')} 띠 {n} 덩어리")

print("== (b) 겹침 (0 이어야 함; 테이프는 면에 딱 닿음)")
allp = {"띠": bar, **boards, "선 꺾임": wires, **tapes}
names = list(allp); bad = 0
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        x = vol(allp[names[i]] ^ allp[names[j]])
        if x > 0.01: print("  겹침", names[i], "↔", names[j], x); bad += 1
print(f"  {ok(bad == 0, '부품 겹침')} 겹치는 쌍 {bad} / {len(names) * (len(names) - 1) // 2}")
for nm, t, o in (("테이프 위 패드", tA, bar), ("테이프 아래 패드", tC, bar), ("테이프 보드", tB, bar), ("테이프 보드", tB, U(list(boards.values())))):
    g = max(vol(t.translate([dx, 0, 0]) ^ o) for dx in (0.1, -0.1))
    print(f"  {ok(g > 1.0, nm + ' 닿음')} {nm} 를 x 로 ±0.1 밀면 {'띠' if o is bar else '보드'}와 겹침 {g} mm³ (닿아 있음)")

print("== (c) LIMO 거리")
k = 1.0 / math.cos(math.atan(-S.UP_B)); k2 = 1.0 / math.cos(math.atan(S.LOW_S))
dA = BP[:, 0] - (S.xf_up(BP[:, 2]) + S.TAPE * k); dC = BP[:, 0] - (S.xf_low(BP[:, 2]) + S.TAPE * k2)
onA = (np.abs(dA) < 0.02) & (np.abs(BP[:, 1]) <= S.PADA_HW) & (BP[:, 2] >= S.PADA_Z[0] - 0.01)
onC = (np.abs(dC) < 0.02) & (np.abs(BP[:, 1]) <= S.PADC_HW) & (BP[:, 2] <= -76.0)
ear = (np.abs(BP[:, 1]) > S.NOSE_HW) & (np.abs(BP[:, 1]) < S.EAR_Y0 + 0.6) & (BP[:, 0] < 166.0)      # 귀·다리 안쪽 면
foot = (BP[:, 2] < S.SHO_Z + 0.5) & (BP[:, 0] < 156.0)
rest = ~(onA | onC | ear | foot)
print(f"  {ok(abs(BD[onA].min() - 1.0) < 0.06, '위 패드 틈')} 위 패드 면 ↔ LIMO 위 앞면: {BD[onA].min():.2f} … {np.percentile(BD[onA], 99):.2f} (테이프 1.0)")
print(f"  {ok(abs(BD[onC].min() - 1.0) < 0.06, '아래 패드 틈')} 아래 패드 면 ↔ LIMO 아래 앞면: {BD[onC].min():.2f} … {np.percentile(BD[onC], 99):.2f} (테이프 1.0)")
print(f"  {ok(0.25 <= BD[ear].min() <= 0.4, '귀 틈')} 귀·다리 안쪽 면 ↔ 코 상자 옆벽: 최소 {BD[ear].min():.2f} (자리 잡는 틈 0.3)")
print(f"  {ok(0.25 <= BD[foot].min() <= 0.35, '다리 틈')} 다리 밑 ↔ 어깨 윗면: 최소 {BD[foot].min():.2f} (자리 잡는 틈 0.3)")
i = np.argmin(np.where(rest, BD, 9)); print(f"  {ok(BD[rest].min() >= 0.25, '그 밖 LIMO 거리')} 그 밖의 띠 겉면 ↔ LIMO: 최소 {BD[rest].min():.2f} (곳 {BP[i].round(1).tolist()})")
for nm, m in {**boards, "선 꺾임": wires}.items():
    d = limo_d(pts(m)); print(f"  {ok(d >= 1.0, nm + ' LIMO 거리')} {nm:12s} ↔ LIMO 최소 {d:.2f} (뒷면 부품을 4.0 으로 본 부피; ≥ 1.0)")
m = (np.abs(LP[:, 1]) < 28) & (LP[:, 2] < -39.0) & (LP[:, 2] > -71.5) & (LP[:, 0] > 160)
res = LP[m, 0] - S.xf_up(LP[m, 2]); m2 = res > -1.5; res = res[m2]; rec = LP[m][m2][res < -0.3]      # −2.3 은 껍질 안쪽 면(겉면 아님)
print(f"  위 앞면 실측: 평면에서 벗어남 rms {res[np.abs(res) < 0.15].std():.3f}, 튀어나온 점 {(res > 0.15).sum()} 개. LIMO 글자는 {(-res[res < -0.3]).mean() * math.cos(math.atan(-S.UP_B)):.2f} mm 파임, "
      f"|y| ≤ {np.abs(rec[:, 1]).max():.1f}, z {rec[:, 2].min():.1f} … {rec[:, 2].max():.1f}")
print(f"  {ok(S.PADA_Z[0] > rec[:, 2].max(), '위 패드가 글자 밖')} 위 패드 z {S.PADA_Z[0]} … {S.PADA_Z[1]} 는 글자 위 끝({rec[:, 2].max():.1f})보다 위. 아래 패드는 아래 앞면(글자 없음)")
aA = (S.PADA_Z[1] - S.PADA_Z[0]) * k * 2 * S.PADA_HW; aC = (S.PADC_TAPE_Z[1] - S.PADC_TAPE_Z[0]) * k2 * 2 * S.PADC_HW
rA = vol(tA.translate([0.05, 0, 0]) ^ bar) / 0.05 * k; rC = vol(tC.translate([0.05, 0, 0]) ^ bar) / 0.05 * k2      # 띠 면이 실제로 테이프를 받는 넓이
print(f"  {ok(rA + rC >= 900, '패드 넓이')} 테이프 넓이: 위 {aA:.0f} + 아래 {aC:.0f} = {aA + aC:.0f} mm²; 띠 면이 실제로 받는 넓이 위 {rA:.0f} + 아래 {rC:.0f} = {rA + rC:.0f} (≥ 900). "
      f"위 앞면만으로는 {rA:.0f} mm² — 아래 패드는 아래 앞면에 있음(요구와 다름)")

print("== (d) 자리 잡힘: 띠를 0.05 씩 0.6 mm 까지 밀며 LIMO 겉면까지 최소 거리(0.12 아래면 닿음). 테이프는 따로")
dirs = {"+y": (0, 1, 0), "-y": (0, -1, 0), "-z": (0, 0, -1), "-x": (-1, 0, 0), "+z": (0, 0, 1), "+x": (1, 0, 0)}
tape_l = tA + tC
for dn, v in dirs.items():
    v = np.array(v, float); ds = [(s, limo_d(BP + v * s)) for s in np.arange(0.05, 0.601, 0.05)]
    hit = next((s for s, d in ds if d < 0.12), None); tv = vol(bar.translate(list(v * 0.6)) ^ tape_l)
    if dn in ("+y", "-y", "-z"):
        print(f"  {ok(hit is not None, dn + ' 잡힘')} {dn}: " + (f"LIMO 에 {hit:.2f} mm 에서 닿음" if hit else f"LIMO 에 안 닿음(0.6 에서 최소 {ds[-1][1]:.2f})"))
    elif dn == "-x":
        print(f"  {ok(tv > 1.0, '-x 잡힘')} -x: 테이프를 누름(0.6 밀면 테이프와 겹침 {tv} mm³). LIMO 까지는 0.6 에서 {ds[-1][1]:.2f} — 테이프 1.0 이 x 를 정함")
    else:
        print(f"  ---- {dn}: 막는 모양 없음(0.6 에서 LIMO 최소 {ds[-1][1]:.2f}, 테이프 겹침 {tv}) → 테이프 접착만으로 잡음")

print("== (e) 끼우는 길 (0.5 mm 간격으로 겹침)")
def sweep(nm, mover, obst, steps, limo=False):
    worst, first, dl = 0.0, None, 9.0
    for d in steps:
        mm = mover(d); x = vol(mm ^ obst)
        if x > 0.02 and first is None: first = d
        worst = max(worst, x)
        if limo: dl = min(dl, limo_d(pts(mm, 4000)))
    good = first is None and (not limo or dl >= 0.5)
    print(f"  {ok(good, nm)} {nm:58s} {'걸림 없음' if first is None else f'걸림: {first} mm 에서, 최대 {worst}'}" + (f", LIMO 최소 {dl:.2f}" if limo else ""))
bo = S.board_only_local()
RET, SLIDE = 15.0, 60.0
for side, bn, on in ((1, "HC-SR04 왼", "HC-SR04 오른"), (-1, "HC-SR04 오른", "HC-SR04 왼")):
    def path(d, side=side):      # 자리 → 뒤로 RET → 바깥 끝 쪽으로 SLIDE (끼울 때는 거꾸로)
        return S.pose(bo.translate([-d, 0, 0]) if d <= RET else bo.translate([-RET, side * (d - RET), 0]), side)
    st = np.arange(0.5, RET + SLIDE + 0.01, 0.5)
    sweep(f"{bn} → 빈 띠 (바깥에서 옆으로 {SLIDE:g}, 앞으로 {RET:g})", path, bar, st)
    sweep(f"{bn} 둘째로 (다른 보드·듀폰·테이프 있음)", path, bar + boards[on] + wires + S.pose(S.board_tape_local(), -side), st)
    pl = S.plug_local()
    mv = (lambda side: lambda d: S.pose(pl.translate([0, 0, -d]), side))(side)
    sweep(f"{bn} 듀폰 꽂기 (아래에서 위로 14, 띠가 LIMO 에 붙은 채)", mv, bar + S.pose(S.box(S.XPCB, -S.BEZEL_T, -22.5, 22.5, -10, 10), side), np.arange(0.5, 14.01, 0.5), limo=True)
    lim = next((d for d in np.arange(0.25, RET + 0.01, 0.25) if limo_d(pts(S.pose(bo.translate([-d, 0, 0]), side), 6000)) < 0.15), None)
    print(f"  ---- {bn}: 띠가 LIMO 에 붙은 채로는 뒤로 {lim} mm 에서 LIMO 에 닿음(원통을 빼려면 11.1 필요) → 보드는 붙이기 전에 넣어야 함")

print("== (f) 빔 원뿔 ±15° (원통 앞면 Ø16 에서 300 mm) 안에 띠·다른 센서·LIMO 가 없어야 함")
Lc = 300.0; r1 = 8.0 + Lc * math.tan(math.radians(15))
cone_l = m3.Manifold.cylinder(Lc, 8.0, r1, 64).rotate([0, 90, 0])
for side in (1, -1):
    for s in (-1, 1):
        x0 = -S.BEZEL_T + S.CAN_H + 0.05
        cone = S.pose(cone_l.translate([x0, s * S.CAN_GAP / 2, 0]), side)
        others = U([mm for kk, mm in allp.items() if kk != ("HC-SR04 왼" if side > 0 else "HC-SR04 오른")])
        o = np.array(S.pose(S.box(x0 - .01, x0 + .01, s * 13 - .01, s * 13 + .01, -.01, .01), side).bounding_box()).reshape(2, 3).mean(0)
        o2 = np.array(S.pose(S.box(x0 + 99.99, x0 + 100.01, s * 13 - .01, s * 13 + .01, -.01, .01), side).bounding_box()).reshape(2, 3).mean(0)
        u = (o2 - o) / 100.0; w = LP - o; sx = w @ u; rad = np.linalg.norm(w - np.outer(sx, u), axis=1)
        nin = int(((sx > 0) & (sx < Lc) & (rad < 8.0 + sx * math.tan(math.radians(15)))).sum())
        v = vol(cone ^ others)
        print(f"  {ok(v < 0.01 and nin == 0, '빔 원뿔')} {'왼' if side > 0 else '오른'} 센서 원통 {'안쪽' if s * side < 0 else '바깥'}: 부품과 겹침 {v}, 원뿔 안 LIMO 점 {nin}")
yin = S.SENSOR_Y - (13 + 8) * math.cos(math.radians(S.YAW))
print(f"  두 센서 축: 좌우 ±{S.YAW}°, 위 {S.PITCH}°, 원통 중심 z {S.SENSOR_Z} (땅 위 {S.SENSOR_Z + 140.5:.1f}). 두 빔은 원통 앞 {yin / math.tan(math.radians(15 - S.YAW)):.0f} mm 에서 만남")
backs = U([b ^ S.pose(S.box(-60, -S.BEZEL_T, -40, 40, -11, 40), sd) for b, sd in ((boards["HC-SR04 왼"], 1), (boards["HC-SR04 오른"], -1))])
pa = backs.project(); seen = pa - bar.project()
print(f"  {ok(seen.area() < 0.02 * pa.area(), '지붕이 보드를 가림')} 바로 위에서 볼 때 보드(아랫변 위) 넓이 {pa.area():.0f} mm² 중 지붕 밖으로 보이는 넓이 {seen.area():.1f} mm²")

print("== (g) 라이다")
bb = np.array(bar.bounding_box()).reshape(2, 3)
top = LP[(LP[:, 0] > 150) & (LP[:, 0] < 164) & (np.abs(LP[:, 1]) < 28) & (LN[:, 2] > 0.9) & (LP[:, 2] > -45) & (LP[:, 2] < -30)]
dn_ = LP[(LP[:, 0] > 125) & (np.abs(LP[:, 1]) < 45) & (LN[:, 2] < -0.9) & (LP[:, 2] > -35) & (LP[:, 2] < 0)]
print(f"  {ok(bb[1][2] <= -41.0, '라이다 한계')} 띠 맨 위 z {bb[1][2]:.2f} (한계 −41). 라이다 틈 바닥(코 상자 윗면) z {np.median(top[:, 2]):.1f} → 여유 {np.median(top[:, 2]) - bb[1][2]:.1f} mm. "
      f"틈 천장(아래를 보는 면) z {np.unique(dn_[:, 2].round(0)).tolist()}, 스캔 면 z ≈ −30 (설계.md) 까지 {-30 - bb[1][2]:.1f} mm")

print("== (h) 앞 모서리 램프 가림 (check_lamp.py 와 같은 방법: 램프 면의 점에서 광선, 부품 겉면 0.6 안을 지나면 가림). 부품 = 띠 + 센서 + 선 꺾임")
nn = np.load(os.path.join(D, "render", "asm", "lamp_n.npy")); P0 = np.load(os.path.join(D, "render", "asm", "lamp_pts.npy"))[::6]
partm = trimesh.util.concatenate([S.to_t(bar), S.to_t(U(list(boards.values()))), S.to_t(wires)])
kt = cKDTree(trimesh.sample.sample_surface(partm, 600000)[0])
up = np.array([0, 0, 1.0]); sd = np.cross(up, nn); sd /= np.linalg.norm(sd); steps = np.arange(1.0, 90.0, 0.6)
for lab, mir in (("램프(lamp_pts.npy)", 1), ("반대쪽 램프(거울상)", -1)):
    nn_ = nn * [1, mir, 1]; sd_ = sd * [1, mir, 1]; P = P0 * [1, mir, 1]
    def blocked(a, e):
        dv = nn_ * np.cos(np.radians(a)) * np.cos(np.radians(e)) + sd_ * np.sin(np.radians(a)) * np.cos(np.radians(e)) + up * np.sin(np.radians(e))
        dv = dv / np.linalg.norm(dv); hit = np.zeros(len(P), bool)
        for s in steps: hit |= kt.query(P + nn_ * 0.5 + dv * s)[0] < 0.6
        return hit.mean() * 100
    r = [(name, blocked(a, e)) for name, a, e in (("면에 수직", 0, 0), ("바깥 30°", 30, 0), ("안쪽 30°", -30, 0), ("안쪽 60°(거의 정면)", -60, 0), ("위 20°", 0, 20), ("아래 20°", 0, -20))]
    print(f"  {ok(max(v for _, v in r) < 1.0, '램프 가림')} {lab} y {P[:, 1].min():.0f}…{P[:, 1].max():.0f}: " + ", ".join(f"{n_} {v:.1f} %" for n_, v in r))

print("== (i) 바퀴 틈 (check_wheels.py 와 같은 방법: 공식 URDF 바퀴 윤곽 회전체, 앞바퀴 조향 −30…+30° 5° 간격). 부품 = 띠 + 센서 + 선 꺾임")
WP = np.load(os.path.join(D, "ref", "limo_wheel", "limo_wheel_pts.npy")) * 1000.0
ys = np.arange(WP[:, 1].min(), WP[:, 1].max() + 0.5, 0.5); rr = np.hypot(WP[:, 0], WP[:, 2])
prof = [(y, rr[np.abs(WP[:, 1] - y) <= 2.0].max()) for y in ys if (np.abs(WP[:, 1] - y) <= 2.0).any()]
wheel = m3.Manifold()
for (y0, r0), (y1, r1_) in zip(prof[:-1], prof[1:]):
    wheel += m3.Manifold.cylinder(y1 - y0 + 0.01, r0, r1_, 72).rotate([-90, 0, 0]).translate([0, y0, 0])
g = bar + U(list(boards.values())) + wires; ktg = cKDTree(pts(g, 200000))
for left in (True, False):
    worst = (1e9, 0, 0)
    for deg in range(-30, 31, 5):
        w = (wheel if left else wheel.mirror([0, 1, 0])).rotate([0, 0, deg]).translate([114.5, 70 if left else -70, -96.5])
        ov = (w ^ g).volume(); d = ktg.query(pts(w, 40000))[0].min()
        if ov > 0.01: d = -ov
        if d < worst[0]: worst = (d, deg, ov)
    print(f"  {ok(worst[0] >= 5.0, '바퀴 틈')} 앞{'왼(+y)' if left else '오른(−y)'} 바퀴: 최소 틈 {worst[0]:.1f} mm (조향 {worst[1]}°, 겹침 {worst[2]:.1f} mm³)")

print("== (j) 크기·무게·출력 자세 오버행")
print(f"  띠 크기 x {bb[0][0]:.1f} … {bb[1][0]:.1f}, y {bb[0][1]:.1f} … {bb[1][1]:.1f}, z {bb[0][2]:.1f} … {bb[1][2]:.1f}  ({ok(bb[1][1] <= S.Y_MAX + 0.01 and bb[0][1] >= -S.Y_MAX - 0.01, '띠 폭')} |y| ≤ {S.Y_MAX}); 땅(−140.5)까지 {bb[0][2] + 140.5:.1f}")
ab = np.array((bar + U(list(boards.values())) + wires).bounding_box()).reshape(2, 3)
print(f"  조립품(센서·선 포함) x 최대 {ab[1][0]:.1f} (범퍼 꼭대기 173.0 에서 {ab[1][0] - 173.0:.1f}), z 최소 {ab[0][2]:.1f}")
mb = bar.volume() * 1.24e-3
print(f"  {ok(mb <= 22.0, '무게')} 띠 무게(속 찬 PLA 1.24): {mb:.1f} g (목표 ≤ 22)")
tool = os.path.expanduser(r"~\.claude\skills\bambu-studio-gui\scripts\overhang_regions.py")
pth = os.path.join(S.OUT, "print_sonar.stl"); S.print_parts()["print_sonar.stl"].export(pth)
if os.path.exists(tool):
    r = subprocess.run([sys.executable, tool, pth, "--top", "8"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    print("  --- print_sonar.stl (지붕이 판에 닿게 뒤집음)"); print("   " + (r.stdout + r.stderr).strip().replace("\n", "\n   "))
else:
    print("  오버행 도구 없음 — 재지 못함"); FAIL.append("오버행 도구 없음")

print("== (k) 테이프 하중 손계산 (가정: 1 mm 아크릴 폼 테이프의 오래 걸리는 안전 응력 0.10 MPa, 짧은 하중 0.5 MPa)")
gN = 9.81e-3; m_s, m_c = 8.5, 6.0
cb = S.to_t(bar).center_mass
cs = np.array([np.array(b.bounding_box()).reshape(2, 3).mean(0) for b in boards.values()]).mean(0)
mt = mb + 2 * m_s + m_c; com = (mb * cb + 2 * m_s * cs + m_c * np.array([178.0, 0, -85.0])) / mt
zA, zC = sum(S.PADA_Z) / 2, sum(S.PADC_TAPE_Z) / 2; pA = np.array([S.xf_up(zA), zA]); pC = np.array([S.xf_low(zC), zC])
nA = np.array([math.cos(math.atan(-S.UP_B)), math.sin(math.atan(-S.UP_B))]); nC = np.array([math.cos(math.atan(S.LOW_S)), -math.sin(math.atan(S.LOW_S))])
W = mt * gN
print(f"  무게: 띠 {mb:.1f} + 센서 2 × {m_s} + 선·듀폰 {m_c} (가정) = {mt:.1f} g → {W:.2f} N. 무게중심 x {com[0]:.1f}, z {com[2]:.1f} → 위 앞면에서 앞으로 {com[0] - S.xf_up(com[2]):.1f} mm")
def solve(F, at):
    """xz 평면 힘 F 가 at 에 걸릴 때 두 패드 가운데가 받는 힘(수직·면 방향 4 미지수, 최소 노름 해)."""
    tA_, tC_ = np.array([-nA[1], nA[0]]), np.array([-nC[1], nC[0]])
    cols = [nA, tA_, nC, tC_]; pp = [pA, pA, pC, pC]
    A = np.array([[c[0] for c in cols], [c[1] for c in cols], [p[0] * c[1] - p[1] * c[0] for p, c in zip(pp, cols)]])
    b = np.array([-F[0], -F[1], -(at[0] * F[1] - at[1] * F[0])])
    return np.linalg.lstsq(A, b, rcond=None)[0]
def show(lab, F, at, lim):
    NA, SA, NC, SC = solve(np.array(F, float), np.array(at, float))      # 패드가 띠에 주는 힘: 수직 − = 띠를 LIMO 쪽으로 당김(테이프 당김)
    sA = math.hypot(NA, SA) / rA; sC = math.hypot(NC, SC) / rC; w = max(sA, sC)
    print(f"  {ok(w <= lim, lab)} {lab}: 위 패드 수직 {NA:+.2f} N·면 방향 {SA:+.2f} N → {sA * 1000:.1f} kPa; 아래 패드 수직 {NC:+.2f}·면 방향 {SC:+.2f} → {sC * 1000:.1f} kPa (− 수직 = 테이프 당김; 한계 {lim * 1000:.0f} kPa, 여유 {lim / w:.0f} 배)")
show("자중(오래)", (0, -W), (com[0], com[2]), 0.10)
arm = np.linalg.norm(pA - pC); Mp = W * (com[0] - (pA[0] + pC[0]) / 2)
print(f"       떼어 내는 모멘트(두 패드 가운데 기준) {Mp:.1f} N·mm, 패드 사이 팔 {arm:.1f} mm → 짝힘 {Mp / arm:.2f} N")
show("정면 충돌 30 N (원통 끝 z −57.5 에 −x)", (-30, 0), (ab[1][0], S.SENSOR_Z), 0.5)
show("걸림 10 N 앞으로 (+x, 원통 끝)", (10, 0), (ab[1][0], S.SENSOR_Z), 0.5)
show("걸림 10 N 위로 (+z, 원통 끝)", (0, 10), (ab[1][0], S.SENSOR_Z), 0.5)
ba = vol(S.board_tape_local()) / S.TAPE
print(f"       보드 테이프: 보드 1 장 {m_s} g → {m_s * gN:.3f} N, 테이프 {ba:.0f} mm² → {m_s * gN / ba * 1000:.2f} kPa. 듀폰을 꽂고 빼는 힘(위아래)은 테두리(위 덩이·아래 턱받이)가 받음")

print("\n결과:", "모두 통과" if not FAIL else f"실패 {len(FAIL)}: " + "; ".join(FAIL))
