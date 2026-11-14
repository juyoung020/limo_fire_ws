"""v5 설계용 1차원 물리 모델(빠른 탐색). MuJoCo(sim_full) 로 최종 검증.
고무줄 F(x) = k (s0 + x), x = 장전 이동 0..D. 레버(반지름 방향 미는 면, 축 높이 h, 쉴 때 핀이 축보다 p 앞):
  핀 이동/레버 회전 g(x) = h + d^2/h, d = p - x, 미는 힘 기울기(압력각) atan(d/h).
서보 회전력 τ(x) = F(x) g(x) / (1 - μ tan|φ|) (핀이 홈에 눌려 생기는 마찰 어림).
공 속도: 플런저(+핀) M 과 공 m 이 같이 가속 → 플런저가 앞 멈춤에 부딪혀 멈추고 공만 나감.
  v = sqrt(2 E / (M + m + m_band/3)), 공 에너지 비율 = m / (M + m + m_band/3)."""
import math, numpy as np
D = 50.0
def evals(s0, p, h, mu=0.35, n=400):
    x = np.linspace(0, D, n); d = p - x; phi = np.arctan(np.abs(d) / h)
    g = h + d * d / h
    fr = 1 - mu * np.tan(phi)
    if fr.min() <= 0.05: return None
    tau = (s0 + x) * g / fr                     # k=1 기준
    E = s0 * D + D * D / 2
    return dict(E=E, tau=tau.max(), phi_max=math.degrees(phi.max()), theta=math.degrees(math.atan(p / h) - math.atan((p - D) / h)))
if __name__ == "__main__":
    base = evals(10.3, 27, 48)
    tau_b = 1.0
    def score(r): return r["E"] / r["tau"]       # 같은 서보 회전력에서 저장 에너지(상대)
    sb = score(base); print("현재 v4: E/τ =", round(sb, 3), base)
    best = []
    for s0 in (10.3, 20, 40, 60, 100, 150):
        for h in np.arange(25, 61, 2.5):
            for p in np.arange(-10, 61, 2.5):
                r = evals(s0, p, h)
                if r is None or r["phi_max"] > 45: continue
                best.append((score(r) / sb, s0, h, p, r))
    best.sort(key=lambda t: -t[0])
    for b in best[:8]: print(round(b[0], 2), "s0", b[1], "h", b[2], "p", b[3], {k: round(v, 1) for k, v in b[4].items()})
    for s0 in (10.3, 40, 100):
        bs = [b for b in best if b[1] == s0]
        print("s0", s0, "최고", round(bs[0][0], 2), "h", bs[0][2], "p", bs[0][3], "θ범위", round(bs[0][4]["theta"], 1))
