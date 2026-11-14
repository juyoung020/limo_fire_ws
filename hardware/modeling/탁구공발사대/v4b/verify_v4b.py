"""v5 최종 검증(CAD 치수 그대로 = params.json, 질량 = mass_props.json). 결과 → verify_v5.json
경우: 각도 30/15/5(장전 15), 5발, 공 2.7·5.3 g, 고무줄 세기 k(1 = 유니아트 4cm 한쪽 1개), 견고성(마찰·고무줄·서보 힘)."""
import sys, json, math
import numpy as np
import sim_full as S
S.B.DT = 1e-4
TAU = S.SERVOS["DS3218"]["tau"]
def shoot(theta, k=1.0, cock_theta=None, shots=5, ball=2.7, mu=None, tau=1.0, cock_t=0.6):
    S.M_BALL = ball * 1e-3
    if mu is not None: S.MU = mu
    st = dict(t0=None, peak=0.0, prev=0.0)
    def cb(m, d, tgt):
        if tgt["lever"] > 0 and st["prev"] <= 0: st["t0"] = d.time
        st["prev"] = tgt["lever"]
        if tgt["lever"] > 0 and st["t0"] is not None and d.time - st["t0"] > 0.08 and d.qvel[m.jnt_dofadr[m.joint("lever").id]] > 0.05:
            st["peak"] = max(st["peak"], abs(d.ctrl[m.actuator("lever").id]) / (TAU * tau))
    r = S.run(theta=theta, cock_theta=cock_theta, k_scale=k, tau_scale=tau, shots=shots, timing=dict(cock=cock_t), on_step=cb)
    S.MU = 0.35
    sh = r["shots"]; v = [x.get("exit_speed_mps") for x in sh]
    good = [x for x in v if x]
    return dict(theta=theta, k=k, ball=ball, mu=mu, tau=tau, cock_t=cock_t, v=v, v_mean=round(float(np.mean(good)), 3) if good else None,
                v_sd=round(float(np.std(good)), 3) if len(good) > 1 else None, all_ok=r.get("all_ok"),
                held=all(x.get("held_by_sear") for x in sh), extra=[x.get("extra_balls") for x in sh],
                push_peak=round(st["peak"], 3), hold=round(max(x.get("lever_hold_torque_frac") or 0 for x in sh), 3),
                sear_open=max(x.get("sear_max_open_deg") or 0 for x in sh), impact_v=max(x.get("impact_v_mps") or 0 for x in sh),
                pin_cock=[x.get("pin_x_at_cock_mm") for x in sh][:1])
if __name__ == "__main__":
    cases = json.loads(sys.argv[1]); out = sys.argv[2]
    with open(out, "a", encoding="utf-8") as f:
        for c in cases:
            r = shoot(**c); f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush(); print(r, flush=True)
