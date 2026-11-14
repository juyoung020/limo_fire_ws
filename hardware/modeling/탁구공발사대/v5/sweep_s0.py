import sys, json, math
import numpy as np
import verify_v5 as V
S = V.S
def run(s0, k, cock_t=0.6, theta=30, shots=3, tau=1.0):
    S.PRE = s0 * 1e-3; S.HOOK_X = S.B.PIN_REST + 31.7 + s0; S.M_BALL = 0.0027
    st = dict(t0=None, peak=0.0, prev=0.0)
    def cb(m, d, tgt):
        if tgt["lever"] > 0 and st["prev"] <= 0: st["t0"] = d.time
        st["prev"] = tgt["lever"]
        if tgt["lever"] > 0 and st["t0"] is not None:
            st["peak"] = max(st["peak"], abs(d.ctrl[m.actuator("lever").id]) / V.TAU if d.time - st["t0"] > 0.08 else 0)
    r = S.run(theta=theta, k_scale=k, tau_scale=tau, shots=shots, timing=dict(cock=cock_t), on_step=cb)
    sh = r["shots"]; v = [x.get("exit_speed_mps") for x in sh]
    return dict(s0=s0, k=k, cock_t=cock_t, tau=tau, v=v, held=[x.get("held_by_sear") for x in sh], peak=round(st["peak"], 3))
for c in json.loads(sys.argv[1]):
    print(json.dumps(run(**c)), flush=True)
