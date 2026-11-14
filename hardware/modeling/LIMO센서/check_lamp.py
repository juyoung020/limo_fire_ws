"""앞 모서리 램프(LIMO 메시에서 찾은 면: 법선 방위 62°·아래 11°, x 147~164, y ±39~50, z −102~−82, 344 mm²)를
초음파 받침이 얼마나 가리는지: 램프 면의 점에서 여러 방향으로 광선을 쏴 부품 표면 0.6 mm 안을 지나면 가림."""
import os, numpy as np, trimesh
from scipy.spatial import cKDTree
D = os.path.dirname(os.path.abspath(__file__))
nn = np.load(os.path.join(D, "render", "asm", "lamp_n.npy")); P0 = np.load(os.path.join(D, "render", "asm", "lamp_pts.npy"))[::6]
parts = trimesh.util.concatenate([trimesh.load(os.path.join(D, "stl", f)) for f in ("asm_sonar.stl", "dum_hcsr04.stl")])
kt = cKDTree(trimesh.sample.sample_surface(parts, 600000)[0])
up = np.array([0, 0, 1.0]); side = np.cross(up, nn); side /= np.linalg.norm(side)
steps = np.arange(1.0, 90.0, 0.6)
def blocked(dv):
    dv = dv / np.linalg.norm(dv); hit = np.zeros(len(P0), bool)
    for s in steps: hit |= kt.query(P0 + nn * 0.5 + dv * s)[0] < 0.6
    return hit.mean() * 100
rot = lambda a, e: nn * np.cos(np.radians(a)) * np.cos(np.radians(e)) + side * np.sin(np.radians(a)) * np.cos(np.radians(e)) + up * np.sin(np.radians(e))
for name, dv in (("램프 면에 수직", rot(0, 0)), ("바깥으로 30°", rot(30, 0)), ("안쪽(정면 쪽)으로 30°", rot(-30, 0)), ("안쪽으로 60°(거의 정면)", rot(-60, 0)), ("위로 20°", rot(0, 20)), ("아래로 20°", rot(0, -20))):
    print(f"{name:20s} 가려진 비율 {blocked(dv):5.1f} %")
