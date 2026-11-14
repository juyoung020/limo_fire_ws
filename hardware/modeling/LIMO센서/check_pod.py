"""옆판 전장 상자 검사(단독 실행). 먼저 build_pod.py 를 돌려 stl/ 를 만든다.
(a) 출력 부품 덩어리 수  (b) 겹침(판·덮개·볼트·보드·선·LIMO)  (c) 0.6 mm 밀어 6 방향 잡힘
(d) 끼우는 길: 보드(바깥에서 곧게), 덮개(갈고리 걸고 기울여 닫기, 2° 간격)  (e) M3 드라이버 길 8
(f) LIMO 거리·바깥 |y|·무게·서포트
쓰기: python check_pod.py [--fast]   (--fast 는 서포트 측정 생략)"""
import os, sys, math, subprocess
import numpy as np, manifold3d as m3, trimesh
from scipy.spatial import cKDTree
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D); sys.path.insert(0, os.path.join(D, "..", "lib"))
import build_pod as P
ST = os.path.join(D, "stl")
FAIL = []


def ld(n):
    t = trimesh.load(os.path.join(ST, n))
    return m3.Manifold(m3.Mesh(t.vertices.astype(np.float32), t.faces.astype(np.uint32)))


vol = lambda a: round(a.volume(), 3)
cen = lambda m: (np.array(m.bounding_box()[:3]) + np.array(m.bounding_box()[3:])) / 2
U = lambda ms: sum(ms, m3.Manifold())


def check(name, ok, detail=""):
    print(f"  [{'통과' if ok else '실패'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


def axes(side):
    """국소 u v w 축의 지붕 좌표 방향."""
    return {"u": np.array([1.0, 0, 0]), "v": np.array([0, -P.SA * side, P.CA]), "w": np.array([0, P.CA * side, P.SA])}


def from_world(mf, side):
    R = np.array([[1, 0, 0], [0, -P.SA * side, P.CA * side], [0, P.CA, P.SA]])
    t = np.array([0, P.Y_HOLE * side, P.Z_HOLE])
    Ri = R.T
    return mf.transform(np.c_[Ri, -Ri @ t].tolist())


# ---------- LIMO ----------
# limo.stl 은 닫힌 메시가 아니라 불 연산을 못 함 → 겉면 점 거리로 잰다. 상자 둘레(옆판·화면·머리)만 촘촘히(약 0.1 mm 간격) 뿌림.
limo = trimesh.load(os.path.join(D, "render", "asm", "limo.stl"))
_tc = limo.triangles
_lo, _hi = _tc.min(axis=1), _tc.max(axis=1)
_near = (_hi[:, 0] > -160) & (_lo[:, 0] < 130) & (np.abs(_tc[:, :, 1]).max(axis=1) > 35) & (_hi[:, 2] > -75) & (_lo[:, 2] < 60)
limo_near = trimesh.Trimesh(limo.vertices, limo.faces[_near], process=False)
LP, _ = trimesh.sample.sample_surface(limo_near, 6000000, seed=1)
LP = np.r_[LP, limo_near.vertices[np.unique(limo_near.faces)]]
KT = cKDTree(LP)
print(f"LIMO 겉면 점 {len(LP)} 개, 넓이 {limo_near.area:.0f} mm² → 점 간격 약 {math.sqrt(limo_near.area / len(LP)):.2f} mm")


def pts_of(m, n=30000):
    me = m.to_mesh(); tm = trimesh.Trimesh(me.vert_properties[:, :3], me.tri_verts, process=False)
    p, _ = trimesh.sample.sample_surface(tm, n, seed=2)
    return np.r_[tm.vertices, p]


def limo_d(m, sel=None, n=30000):
    """부품 겉면 점 → LIMO 겉면 점 최소 거리(점 간격만큼 크게 나올 수 있음). sel = LIMO 쪽 점 고르기."""
    p = pts_of(m, n)
    if sel is None:
        d, _ = KT.query(p)
    else:
        d, _ = cKDTree(LP[sel(LP)]).query(p)
    return float(d.min())


print("== (a) 출력 부품 덩어리 수 (1 이어야 함), 무게(부피 × 1.24)")
mass = {}
for n in ("print_pod_right", "print_pod_left", "print_pod_right_lid", "print_pod_left_lid", "print_pod_bolt"):
    m = ld(n + ".stl"); k = len(m.decompose()); mass[n] = m.volume() * 1.24e-3
    bb = m.bounding_box()
    check(n, k == 1, f"덩어리 {k}  {mass[n]:.1f} g  크기 {bb[3]-bb[0]:.1f} × {bb[4]-bb[1]:.1f} × {bb[5]-bb[2]:.1f}  바닥 z {bb[2]:.2f}")
tot = sum(mass[n] for n in mass if n != "print_pod_bolt")
check("판 2 + 덮개 2 무게 ≤ 115 g", tot <= 115.0, f"{tot:.1f} g (+ 볼트 2 개 {2 * mass['print_pod_bolt']:.1f} g)")

world = {}
for side, nm in ((1, "right"), (-1, "left")):
    print(f"\n######## {nm} (side {side:+d}) ########")
    ax = axes(side)
    plate, lid = ld(f"asm_pod_{nm}.stl"), ld(f"asm_pod_{nm}_lid.stl")
    brds = sorted(ld(f"dum_pod_{nm}.stl").decompose(), key=lambda m: cen(m)[0])
    assert len(brds) == 2, len(brds)
    bn = ("배선판", "ESP32") if side > 0 else ("XL4015", "LM2596")
    cab = ld(f"dum_pod_{nm}_cables.stl")
    bolt_fn = lambda a: P.to_world(P.lid_bolt(side, a), side)

    print("== (b) 겹침")
    best = min((vol(bolt_fn(a) ^ plate), a) for a in range(0, 360, 15))
    bolt = bolt_fn(best[1])
    check("볼트 ↔ 판 암나사(가장 맞는 위상)", best[0] < 0.01, f"{best[0]} mm³ (위상 {best[1]}°), 반 바퀴 어긋나면 {vol(bolt_fn(best[1] + 180) ^ plate)}")
    check("볼트 반 바퀴 어긋나면 겹침(산이 물림)", vol(bolt_fn(best[1] + 180) ^ plate) > 1.0)
    items = {"판": plate, "덮개": lid, "볼트": bolt, bn[0]: brds[0], bn[1]: brds[1], "선 묶음": cab}
    names = list(items); bad = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            x = vol(items[names[i]] ^ items[names[j]])
            if x > 0.01:
                bad.append((names[i], names[j], x))
    check("판·덮개·볼트·보드·선 서로 겹침 0", not bad, str(bad) if bad else f"({len(names) * (len(names) - 1) // 2} 쌍)")
    eng = vol(P.to_world(P.cyl_w(2.5, P.BOLT_W - P.LID_BOLT_L, P.BOSS_W1, P.BOSS_U, P.BOSS_V), side) ^ plate)
    print(f"     나사 물림 길이 {P.BOSS_W1 - (P.BOLT_W - P.LID_BOLT_L):.1f} mm (끝 모따기 0.8 포함), 볼트 머리 윗면은 덮개 겉면보다 {P.W_OUT - P.BOLT_W - P.HEAD_H:.1f} 안")

    print("== (c) 잡힘: 0.6 mm 밀었을 때 막히는 방향(국소 ±u ±v ±w 모두 막혀야 함)")
    for nm2 in ("덮개", "볼트", bn[0], bn[1]):
        m = items[nm2]
        rest = U([o for k, o in items.items() if k not in (nm2, "선 묶음")])
        free = []
        for an, av in ax.items():
            for sg in (1, -1):
                if vol(m.translate(list(av * 0.6 * sg)) ^ rest) < 0.02:
                    free.append(("+" if sg > 0 else "-") + an)
        check(f"{nm2} 6 방향 막힘", not free, ("풀린 방향 " + " ".join(free)) if free else "")
    # 보드가 덮개 쪽으로 뜨는 틈
    for b, nmb in zip(brds, bn):
        gap = next((d for d in np.arange(0.05, 1.01, 0.05) if vol(b.translate(list(ax["w"] * d)) ^ lid) > 0.02), None)
        check(f"{nmb} 덮개 쪽 틈 ≤ 0.3", gap is not None and gap <= 0.3 + 1e-6, f"{gap} mm 에서 닿음")

    print("== (d) 끼우는 길")
    for b, nmb in zip(brds, bn):
        worst = max(vol(b.translate(list(ax["w"] * d)) ^ plate) for d in np.arange(0.5, 40.01, 0.5))
        check(f"{nmb} → 판 (바깥에서 곧게 40 mm, 0.5 간격)", worst < 0.02, f"최대 겹침 {worst}")
    # 덮개: 갈고리 걸고 기울여 닫기. 닫힌 자리에서 2° 씩 열며, 각도마다 (dv, dw) 를 0.25 간격으로 찾아 겹침 0 인 자리를 이어 감
    lid_l = from_world(lid, side)
    obst = from_world(plate + U(brds) + cab, side)
    path, cur, ok = [], (0.0, 0.0), True
    dmin = 99.0
    for th in range(2, 61, 2):
        found = None
        cand = sorted(((a, b) for a in np.arange(-1.5, 1.51, 0.25) for b in np.arange(-1.5, 1.51, 0.25)), key=lambda c: abs(c[0]) + abs(c[1]))
        for (a, b) in cand:
            dv, dw = cur[0] + a, cur[1] + b
            if vol(P.tilt(lid_l, th, dv, dw) ^ obst) < 0.02:
                found = (dv, dw); break
        if found is None:
            ok = False
            print(f"     {th}° 에서 길 없음 (직전 자리 {cur})"); break
        cur = found; path.append((th, round(cur[0], 2), round(cur[1], 2)))
        dmin = min(dmin, limo_d(P.to_world(P.tilt(lid_l, th, *cur) ^ P.tilt(P.box(-300, 300, -300, 300, P.T + 0.01, 100), th, *cur), side), n=8000))
        # 갈고리가 다 빠졌으면(바깥으로 곧게 30 mm 가 비면) 끝
        if th >= 14 and all(vol(P.tilt(lid_l, th, cur[0], cur[1] + d) ^ obst) < 0.02 for d in np.arange(1.0, 40.01, 1.0)):
            path.append(("곧게 빼기", th)); break
    else:
        ok = False
    check("덮개 갈고리·기울임 길(보드·선 있는 채, 2° 간격)", ok, f"{path}")
    check("  그 길에서 LIMO 까지 ≥ 1.5", dmin >= 1.5, f"최소 {dmin:.2f}")

    print("== (e) M3 드라이버 길 Ø6.5 × 80 (수평 나사 축)")
    for x in P.HOLES_X:
        cyl = P.to_world(P.cyl_horiz(3.25, P.SPOT_T0 + 0.02, P.SPOT_T0 + 80, x), side)
        a, b = vol(cyl ^ plate), vol(cyl ^ U(brds))
        check(f"x {x:6.1f}", a < 0.01 and b < 0.01, f"판과 {a}  보드와 {b}  LIMO 까지 {limo_d(cyl, n=5000):.1f}")
    # 머리 자리 깊이: 머리(Ø6 × 2.4) 가 돋움 안에 다 들어가는지
    head = P.to_world(P.cyl_horiz(3.0, P.SPOT_T0, P.SPOT_T0 + 2.4, P.HOLES_X[0]), side)
    hull = P.to_world(P.cyl_w(P.SEAT_R, 0, P.SEAT_W1, P.HOLES_X[0], P.SEAT_V), side)
    check("M3 머리(Ø6 × 2.4)가 돋움 높이 안", vol(head - hull) < 0.01, f"밖으로 나온 부피 {vol(head - hull)}")

    print("== (f) LIMO 거리")
    # LIMO 점을 국소 좌표로: w > 0.2 = 판 뒷면보다 앞(바깥)에 있는 것, 그 밖은 판 뒤(옆판·윗 테두리)
    dyl, dzl = LP[:, 1] * side - P.Y_HOLE, LP[:, 2] - P.Z_HOLE
    lw = P.CA * dyl + P.SA * dzl
    lv = -P.SA * dyl + P.CA * dzl
    front_of = lambda q, _m=(lw > 0.2): _m
    near = (np.abs(lv - 8) < 18) & (LP[:, 0] > -57) & (LP[:, 0] < 50) & (lw > -3) & (lw < 0.2)
    print(f"     판 뒷면(w 0) ↔ LIMO 옆판: 가장 가까운 면까지 {-lw[near].max():.2f} (윗 테두리 턱), 옆판 넓은 면까지 {-np.percentile(lw[near & (lv < 18)], 99.9):.2f}")
    d_plate = limo_d(plate, sel=front_of)
    check("판: 뒷면만 LIMO 에 닿음(판 뒷면보다 앞에 있는 LIMO 까지 ≥ 1.5)", d_plate >= 1.5, f"{d_plate:.2f}")
    lid_shell = lid ^ P.to_world(P.box(-300, 300, -300, 300, P.T + 0.01, 100), side)      # 갈고리 혀(판 두께 안)는 따로
    d_shell, d_lid, d_bolt = limo_d(lid_shell), limo_d(lid), limo_d(bolt)
    check("덮개(판 앞면 위쪽) ≥ 1.5", d_shell >= 1.5, f"{d_shell:.2f}   갈고리 혀 끝(판 뒷면 0.2 앞)까지 넣으면 {d_lid:.2f} (옆판까지)")
    print(f"     볼트 {d_bolt:.2f}")
    both = lid_shell + plate
    rear = limo_d(both, sel=lambda q: (q[:, 0] < -60) & (lw > 0.2))
    front = limo_d(both, sel=lambda q: (q[:, 0] > 40) & (lw > 0.2))
    print(f"     뒤 화면 받침까지 {rear:.2f}   앞(머리 옆벽)까지 {front:.2f}  (점 간격 0.25 안팎)")
    lv2 = pts_of(lid, 1000)
    i = np.argmax(lv2[:, 1] * side)
    print(f"     바깥 |y| 최대 {abs(lv2[i, 1]):.2f} (z {lv2[i, 2]:.1f}), z 범위 {lv2[:, 2].min():.1f} ~ {lv2[:, 2].max():.1f}, x 범위 {lv2[:, 0].min():.1f} ~ {lv2[:, 0].max():.1f}")
    world[nm] = (plate, lid, bolt)

if "--fast" not in sys.argv:
    print("\n== 서포트(출력 자세 그대로, 큰 것부터 8)")
    tool = os.path.expanduser(r"~\.claude\skills\bambu-studio-gui\scripts\overhang_regions.py")
    for n in ("print_pod_right", "print_pod_left", "print_pod_right_lid", "print_pod_left_lid", "print_pod_bolt"):
        r = subprocess.run([sys.executable, tool, os.path.join(ST, n + ".stl"), "--top", "8"], capture_output=True, text=True, encoding="utf-8", errors="replace")
        print(f"--- {n}\n" + (r.stdout or r.stderr).strip())

print("\n실패:", FAIL if FAIL else "없음")
