"""시뮬용 질량·무게중심·관성(발사대 자기 좌표, kg·m) → mass_props.json. CAD 환경에서 실행(build_launcher 모양 그대로).

출력 무게 추정: 벽 0.8 mm(2줄) + 채움 15 %. 핀·걸쇠 축 핀은 100 % 채움(힘 받는 부품, 설계_v4.md 출력 설정).
서보: DS3218 60 g, MG996R 55 g(판매처 무게), 혼: DS3218 알루미늄 원판 2.5 g, MG996R 플라스틱 1 g. 고무줄 1 고리 0.1 g.
모양별 무게중심·관성은 메시를 고른 밀도로 계산한 뒤 위 무게로 맞춘다."""
import os, json
import numpy as np, trimesh
import build_launcher as B

D = os.path.dirname(os.path.abspath(__file__))
PLA = 1.24e-3            # g/mm3
SHELL, FILL = 0.8, 0.15

def printed_mass(tm, solid=False):
    if solid: return tm.volume * PLA
    v_shell = min(tm.volume, tm.area * SHELL)
    return PLA * (v_shell + FILL * (tm.volume - v_shell))

def props(tm, mass_g):
    """(질량 kg, 무게중심 m, 무게중심 기준 관성 kg·m2) — 발사대 좌표."""
    tm = tm.copy(); tm.density = 1.0
    com = tm.center_mass
    I = tm.moment_inertia / tm.volume * mass_g          # g·mm2, 무게중심 기준
    return mass_g * 1e-3, com * 1e-3, I * 1e-9

def combine(items):
    m = sum(i[0] for i in items)
    c = sum(i[0] * i[1] for i in items) / m
    I = np.zeros((3, 3))
    for mi, ci, Ii in items:
        r = ci - c
        I += Ii + mi * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    return m, c, I

def pack(m, c, I, about=None):
    """about 이 주어지면 그 점 기준 관성도 넣는다."""
    out = dict(mass=round(float(m), 6), com=[round(float(v), 6) for v in c],
               inertia=[[float(f"{v:.4e}") for v in row] for row in I])
    return out

if __name__ == "__main__":
    T = B.to_t
    body = T(B.build_body())
    sv_l = T(B.lever_servo_place(B.servo_solid(B.SERVO_L, horn=False)))
    cap = T(B.build_hopper_cap().translate([B.HOP_CX, 0, B.HOP_TOP - 8.0]))
    tube = T(B.build_hopper_tube()); spin = T(B.sear_pin_at())
    horn_l = T(B.lever_servo_place(B.cyl_z(B.SERVO_L["horn_r"], B.SERVO_L["spline_end"] + B.HORN_TOP - B.HORN_T, B.SERVO_L["spline_end"] + B.HORN_TOP, 0, 0, 64)))
    horn_tilt = T(B.cyl_y(B.SERVO_K["horn_r"], -B.KEEL_Y - B.HORN_T, -B.KEEL_Y, B.AX[0], B.AX[1], 64))
    m_body = printed_mass(body); m_cap = printed_mass(cap)
    m_tube = printed_mass(tube); m_spin = printed_mass(spin, solid=True)
    static = combine([props(body, m_body), props(sv_l, 60.0), props(tube, m_tube), props(cap, m_cap), props(horn_tilt, 1.0), props(spin, m_spin)])

    plunger = T(B.build_plunger()); pin = T(B.build_pin())       # 쉴 때 자리(핀 x = PIN_REST)
    pin_t = pin.copy(); pin_t.apply_translation([B.PIN_REST, 0, 0])
    m_pl, m_pin = printed_mass(plunger), printed_mass(pin_t, solid=True)
    band = (0.2 / 3 * 1e-3, np.array([B.PIN_REST, 0, 0]) * 1e-3, np.zeros((3, 3)))    # 고무줄 2 고리 0.2 g 의 1/3(움직이는 몫)
    moving = combine([props(plunger, m_pl), props(pin_t, m_pin), band])

    lever = T(B.lever_at(0)); sear = T(B.build_sear())
    lev = combine([props(lever, printed_mass(lever)), props(horn_l, 2.5)])
    sband = (0.1 / 2 * 1e-3, np.array([B.SEAR_PEG[0], 0, B.SEAR_PEG[1]]) * 1e-3, np.zeros((3, 3)))     # 걸쇠 고무줄 절반
    lat = combine([props(sear, printed_mass(sear)), sband])

    out = dict(static=pack(*static), plunger_pin=pack(*moving), lever=pack(*lev), sear=pack(*lat),
               grams=dict(body=round(m_body, 1), hopper_cap=round(m_cap, 1), plunger=round(m_pl, 1), pin=round(m_pin, 1),
                          lever=round(printed_mass(lever), 1), sear=round(printed_mass(sear), 1), sear_pin=round(m_spin, 1),
                          hopper_tube=round(m_tube, 1), axle_pin=round(printed_mass(T(B.build_axle_pin()), solid=True), 1),
                          cradle=round(printed_mass(T(__import__("build_cradle").build_cradle())), 1)))
    json.dump(out, open(os.path.join(D, "mass_props.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(out, indent=1))
