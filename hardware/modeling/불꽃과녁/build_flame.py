"""불꽃판 STL 생성: 빨강(속불꽃 자리 홈) + 흰 속불꽃(홈을 정확히 채움). 단위 mm."""
import numpy as np, manifold3d as m3, trimesh, os

def cr(pts, n=10):
    P = np.array(pts, float); N = len(P); out = []
    for i in range(N):
        p0, p1, p2, p3 = P[(i - 1) % N], P[i], P[(i + 1) % N], P[(i + 2) % N]
        for t in np.arange(n) / n:
            out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    a = np.array(out); a[:, 1] = np.maximum(a[:, 1], 0); return a

OUTER = [(-45,0),(45,0),(62,14),(70,38),(70,64),(62,88),(52,104),(56,124),(68,152),(48,134),(36,128),(38,152),(32,178),(20,152),(10,140),(6,168),(-4,206),(-14,170),(-22,142),(-30,152),(-48,174),(-42,142),(-48,122),(-64,138),(-60,106),(-68,82),(-72,56),(-68,30),(-58,12)]
INNER = [(-24,10),(24,10),(36,28),(40,54),(34,78),(24,96),(28,120),(14,106),(6,102),(2,130),(-4,154),(-12,122),(-18,102),(-30,114),(-32,92),(-38,66),(-38,40),(-32,20)]
T, INLAY = 1.8, 0.6          # 판 전체 두께, 흰색 두께
ROD_R, ROD_L = 2.5, 150      # 회전 막대

def ext(poly, h):
    cs = m3.CrossSection([poly.tolist()], m3.FillRule.EvenOdd)
    return m3.Manifold.extrude(cs, h)

outer = ext(cr(OUTER), T)
inner = ext(cr(INNER), INLAY + 1).translate([0, 0, T - INLAY])
rod = m3.Manifold.cylinder(ROD_L, ROD_R, ROD_R, 64, True).rotate([0, 90, 0]).translate([0, -2, ROD_R])
red = (outer - inner) + rod
white = ext(cr(INNER), INLAY).translate([0, 0, T - INLAY])

# 받침: 막대 중심 높이 H, 판이 앞으로 5° 기울어 앞 기둥(막대 위 25 mm 지점)에 기댐
import math
H, TH = 10.0, math.radians(5)
STOP_Y = -(T - ROD_R) * math.cos(TH) - 25 * math.sin(TH)   # 판 앞면이 닿는 y (앞 = -y)
def box(x0, x1, y0, y1, z0, z1):
    return m3.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])
base = box(-85, 85, -16, 12, 0, 3) - box(-36, 36, -10, 6, -1, 5)
for s in (-1, 1):
    x0, x1 = sorted((s * 62, s * 72)); base += box(x0, x1, -5, 5, 2, H + 4)          # 고리
    x0, x1 = sorted((s * 42, s * 50)); base += box(x0, x1, -16, STOP_Y, 2, H + 25 * math.cos(TH))  # 앞 기둥
    # 딸깍 클립: 구멍 Ø5.6(막대 Ø5가 돌 만큼), 위 입구 4.4 → 눌러 끼우면 맞아도 안 빠짐
    hole = m3.Manifold.cylinder(20, 2.8, 2.8, 64, True).rotate([0, 90, 0]).translate([s * 67, 0, H])
    x0, x1 = sorted((s * 57, s * 77))
    base -= hole + box(x0, x1, -2.2, 2.2, H, H + 10)

d = os.path.dirname(os.path.abspath(__file__))
for name, mf in (("flame_red", red), ("flame_white", white), ("flame_merged", outer + rod), ("base", base)):
    me = mf.to_mesh()
    tm = trimesh.Trimesh(me.vert_properties[:, :3], me.tri_verts)
    tm.export(os.path.join(d, name + ".stl"))
    print(name, "덩어리", len(tm.split()), "watertight", tm.is_watertight, "부피cm3", round(tm.volume / 1000, 1))
