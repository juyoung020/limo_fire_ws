"""필렛(열림 연산)이 얇은 기능 요소를 지워 버리지 않았는지: 필렛 없이 만든 것 − 필렛 한 것 을 0.4 mm 깎아
남는 덩어리(= 두께 0.8 mm 넘는 재료가 사라진 곳)를 찍는다. 모서리 둥글림 껍질은 0.73 mm 아래라 남지 않는다."""
import sys, os
import manifold3d as m3
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D); sys.path.insert(0, os.path.join(D, "..", "lib"))
import build_sonar as S, build_pod as P, build_arm as A
jobs = [("초음파 받침", S, S.build), ("가운데 누름쇠", S, S.center_tab), ("팔", A, A.build), ("오른 판", P, P.pod_right), ("왼 판", P, P.pod_left),
        ("오른 덮개", P, P.lid_right), ("왼 덮개", P, P.lid_left)]
for nm, mod, fn in jobs:
    real = mod.fillet; done = fn()
    mod.fillet = lambda A_, r: A_; raw = fn(); mod.fillet = real
    lost = (raw - done).minkowski_difference(m3.Manifold.sphere(0.4, 8))
    ps = [p for p in lost.decompose() if p.volume() > 0.01]
    print(nm, "필렛으로 줄어든 부피", round(raw.volume() - done.volume(), 1), "mm3, 사라진 두꺼운 곳", len(ps),
          [[round(v, 1) for v in p.bounding_box()] for p in ps[:6]], flush=True)
