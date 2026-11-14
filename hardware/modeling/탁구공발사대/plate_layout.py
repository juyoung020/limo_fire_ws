"""한 판 배치 확인(X2D, 오른쪽 노즐 영역 x 20.5~256, y 0~256, 높이 ≤ 250, 부품 사이 5 mm 이상).
출력 방향 STL(stl/print_*.stl)을 위에서 본 볼록 껍질로 놓고 서로 거리·판 경계를 잰다. 결과: plate_layout.json, render/plate_layout.png
서포트는 몸통에만 있고(빌드 플레이트에서만) 모두 몸통 바닥 윤곽 안쪽이라 몸통 껍질에 들어간다."""
import os, json, math
import numpy as np, trimesh
from scipy.spatial import ConvexHull
from PIL import Image, ImageDraw

D = os.path.dirname(os.path.abspath(__file__))
PLATE = (20.5, 256.0, 0.0, 256.0); H_MAX = 250.0; GAP = 5.0
# 부품: (파일, z 축 회전 각, 놓을 자리 = 회전 뒤 바닥 윤곽 최소 x·y)
LAYOUT = {
    "body":        ("print_body.stl", 0, (25.0, 5.0)),
    "cradle":      ("print_cradle.stl", 0, (25.0, 103.0)),
    "hopper_tube": ("print_hopper_tube.stl", 0, (165.0, 5.0)),
    "hopper_cap":  ("print_hopper_cap.stl", 0, (165.0, 59.0)),
    "lever":       ("print_lever.stl", 0, (137.0, 113.0)),
    "sear":        ("print_sear.stl", 0, (137.0, 176.0)),
    "plunger":     ("print_plunger.stl", 0, (191.0, 176.0)),
    "pin":         ("print_pin.stl", 90, (25.0, 209.0)),
    "sear_pin":    ("print_sear_pin.stl", 90, (25.0, 222.0)),
    "axle_pin":    ("print_axle_pin.stl", 0, (58.0, 222.0)),
}

def poly_dist(a, b):
    """두 볼록 다각형(꼭짓점 배열) 사이 최소 거리(겹치면 0)."""
    def seg_pt(p, q, r):
        d = q - p; t = np.clip(np.dot(r - p, d) / max(np.dot(d, d), 1e-12), 0, 1); return np.linalg.norm(p + t * d - r)
    def inside(poly, r):
        s = None
        for i in range(len(poly)):
            p, q = poly[i], poly[(i + 1) % len(poly)]
            c = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
            if s is None: s = c > 0
            elif (c > 0) != s and abs(c) > 1e-9: return False
        return True
    if any(inside(a, r) for r in b) or any(inside(b, r) for r in a): return 0.0
    best = 1e9
    for P, Q in ((a, b), (b, a)):
        for i in range(len(P)):
            for r in Q: best = min(best, seg_pt(P[i], P[(i + 1) % len(P)], r))
    return best

out, polys, vol = {}, {}, 0.0
for n, (f, rz, (x0, y0)) in LAYOUT.items():
    tm = trimesh.load(os.path.join(D, "stl", f))
    tm.apply_transform(trimesh.transformations.rotation_matrix(math.radians(rz), [0, 0, 1]))
    tm.apply_translation([x0 - tm.bounds[0][0], y0 - tm.bounds[0][1], -tm.bounds[0][2]])
    v = tm.vertices[:, :2]; hull = v[ConvexHull(v).vertices]
    polys[n] = hull; vol += tm.volume
    b = tm.bounds
    out[n] = dict(file=f, rot_z_deg=rz, x=[round(float(b[0][0]), 1), round(float(b[1][0]), 1)], y=[round(float(b[0][1]), 1), round(float(b[1][1]), 1)], height=round(float(b[1][2]), 1),
                  inside_plate=bool(b[0][0] >= PLATE[0] and b[1][0] <= PLATE[1] and b[0][1] >= PLATE[2] and b[1][1] <= PLATE[3]),
                  height_ok=bool(b[1][2] <= H_MAX), vol_cm3=round(float(tm.volume) / 1000, 2))
names = list(polys)
pairs = {}
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        dd = poly_dist(polys[names[i]], polys[names[j]])
        if dd < 15: pairs[f"{names[i]}-{names[j]}"] = round(float(dd), 1)
res = dict(parts=out, close_pairs_mm=pairs, min_gap_mm=min(pairs.values()) if pairs else None,
           all_ok=bool(all(p["inside_plate"] and p["height_ok"] for p in out.values()) and (min(pairs.values()) if pairs else 99) >= GAP),
           total_vol_cm3=round(float(vol) / 1000, 1))
json.dump(res, open(os.path.join(D, "plate_layout.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False))
# 그림(위에서 본 판, 1 mm = 3 px)
S = 3; img = Image.new("RGB", (int(256 * S) + 20, int(256 * S) + 20), "white"); dr = ImageDraw.Draw(img)
T = lambda x, y: (10 + x * S, 10 + (256 - y) * S)
dr.rectangle([T(0, 256), T(256, 0)], outline="black", width=2)
dr.rectangle([T(0, 256), T(PLATE[0], 0)], fill=(230, 230, 230))
cols = [(66, 133, 244), (219, 68, 55), (244, 180, 0), (15, 157, 88), (171, 71, 188), (0, 172, 193), (255, 112, 67), (158, 157, 36), (92, 107, 192), (240, 98, 146)]
for k, n in enumerate(names):
    dr.polygon([T(*p) for p in polys[n]], outline=cols[k % len(cols)], fill=tuple(int(c * 0.35 + 255 * 0.65) for c in cols[k % len(cols)]), width=2)
    cx, cy = polys[n].mean(axis=0); dr.text(T(cx - 8, cy + 2), n, fill="black")
dr.text(T(2, 250), "left 20.5 mm: left nozzle only", fill="black")
os.makedirs(os.path.join(D, "render"), exist_ok=True)
img.save(os.path.join(D, "render", "plate_layout.png"))
