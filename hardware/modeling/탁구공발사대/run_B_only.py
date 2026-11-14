"""B(플라이휠)만 다시: 바퀴가 다 돈 뒤(2.5 s) 첫 발. A 결과는 sim_compare.json 의 A 를 그대로 씀."""
import json, os, sim_compare as C
D = os.path.dirname(os.path.abspath(__file__)); P = os.path.join(D, "sim_compare.json")
res = json.load(open(P, encoding="utf-8")); target = res["A"]["nominal"]["mid"]
show = lambda tag, r: (print(tag, json.dumps(r, ensure_ascii=False), flush=True), r)[1]
lo, hi, scan = 0.4, 1.0, {}
for it in range(6):
    duty = round((lo + hi) / 2, 3); r = C.run_B(duty=duty)["shots"][0]; scan[duty] = show(f"B duty {duty}", r)
    if r.get("mid", 0) < target: lo = duty
    else: hi = duty
duty = min(scan, key=lambda k: abs(scan[k].get("mid", 0) - target))
B = res["B"] = dict(duty=duty, duty_scan=scan, nominal=scan[duty])
for tag, kw in (("volt-10%", dict(volt=0.9)), ("volt+10%", dict(volt=1.1)), ("mu_w0.6", dict(mu_w=0.6)), ("mu_w1.0", dict(mu_w=1.0)),
                ("hood-0.5mm(더 눌림)", dict(hood=C.FB["HOOD"] - 0.0005)), ("hood+0.5mm", dict(hood=C.FB["HOOD"] + 0.0005)),
                ("ball_k x0.5", dict(ball_k=3000.0)), ("ball_k x1.5", dict(ball_k=9000.0)),
                ("magnus0.05", dict(magnus=0.05)), ("magnus0.2", dict(magnus=0.2))):
    r = C.run_B(duty=duty, **kw); B[tag] = show("B " + tag, r["shots"][0] if r["shots"] else dict(fired=False))
B["3shots_2s"] = show("B 연속 3발(2초 간격)", C.run_B(duty=duty, shots=3, interval=2.0))
B["3shots_0.8s"] = show("B 연속 3발(0.8초 간격)", C.run_B(duty=duty, shots=3, interval=0.8))
B["spinup_1.0s"] = show("B 켜고 1.0초 만에 첫 발", C.run_B(duty=duty, first=1.0)["shots"])
json.dump(res, open(P, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
