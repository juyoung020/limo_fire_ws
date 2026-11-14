"""출력용 나사·너트 (manifold3d).

PLA 출력 나사는 가는 ISO 나사산이 잘 안 나오므로, 원을 축에서 살짝 비켜 놓고 꼬아 올린 '둥근 나사산'을 쓴다.
같은 함수로 수나사·암나사를 만들어 서로 맞고, 암나사 쪽에 틈(clr)을 더한다.
- 수나사 바깥 지름 d, 나사산 깊이 = 2·e, 간격 pitch (기본 M6: e 0.45, pitch 1.5).
- 출력은 0.12 mm 층, 볼트는 세워서(머리가 판), 너트는 눕혀서.
"""
import math
import manifold3d as m3

SEG = 48


def _thread(d, length, pitch, e, clr=0.0):
    r = d / 2 - e + clr
    cs = m3.CrossSection.circle(r, SEG).translate([e, 0])
    turns = length / pitch
    return cs.extrude(length, n_divisions=max(8, int(turns * 24)), twist_degrees=-360.0 * turns)


def male(d=6.0, length=12.0, pitch=1.5, e=0.45):
    """수나사 몸(z 0 → length), 끝 0.8 mm 모따기."""
    t = _thread(d, length, pitch, e)
    ch = m3.Manifold.cylinder(length, d / 2, d / 2, SEG) - \
        (m3.Manifold.cylinder(1.0, d / 2 + 1, d / 2 + 1, SEG).translate([0, 0, length - 0.8]) -
         m3.Manifold.cylinder(1.0, d / 2 - 0.2, d / 2 + 0.8, SEG).translate([0, 0, length - 0.8]))
    return t ^ ch


def female_cut(d=6.0, length=12.0, pitch=1.5, e=0.45, clr=0.25):
    """암나사로 빼낼 부피(z 0 → length). 양 끝 0.6 mm 모따기 포함."""
    # 1 mm 아래서 시작하면 나사산 위상이 360/pitch 도 어긋나므로 되돌림
    t = _thread(d, length + 2, pitch, e, clr).translate([0, 0, -1]).rotate([0, 0, 360.0 / pitch])
    t += m3.Manifold.cylinder(0.8, d / 2 + clr + 0.6, d / 2 + clr - 0.2, SEG).translate([0, 0, -0.01])
    t += m3.Manifold.cylinder(0.8, d / 2 + clr - 0.2, d / 2 + clr + 0.6, SEG).translate([0, 0, length - 0.79])
    return t


def bolt(d=6.0, length=16.0, pitch=1.5, e=0.45, head_d=None, head_h=None):
    """육각 머리 볼트: 머리(z −head_h → 0) + 나사(z 0 → length)."""
    head_d = head_d or d * 1.8
    head_h = head_h or d * 0.7
    hexa = m3.Manifold.cylinder(head_h, head_d / 2 / math.cos(math.pi / 6), head_d / 2 / math.cos(math.pi / 6), 6) \
        .translate([0, 0, -head_h])
    return hexa + male(d, length, pitch, e)


def nut(d=6.0, h=None, pitch=1.5, e=0.45, af=None):
    """육각 너트(맞변 af), 두께 h, 가운데 암나사."""
    af = af or d * 1.8
    h = h or d * 0.9
    hexa = m3.Manifold.cylinder(h, af / 2 / math.cos(math.pi / 6), af / 2 / math.cos(math.pi / 6), 6)
    return hexa - female_cut(d, h, pitch, e)


def thumb_screw(d=5.0, length=8.0, pitch=1.25, e=0.4, knob_d=14.0, knob_h=5.0):
    """손으로 돌리는 나사: 톱니 손잡이 + 나사."""
    knob = m3.Manifold.cylinder(knob_h, knob_d / 2, knob_d / 2, 24).translate([0, 0, -knob_h])
    for i in range(12):
        a = 2 * math.pi * i / 12
        knob -= m3.Manifold.cylinder(knob_h + 1, 1.0, 1.0, 12).translate(
            [knob_d / 2 * math.cos(a), knob_d / 2 * math.sin(a), -knob_h - 0.5])
    return knob + male(d, length, pitch, e)


def hex_pocket(d=6.0, h=None, af=None, clr=0.25):
    """너트를 넣는 육각 자리(빼낼 부피)."""
    af = (af or d * 1.8) + 2 * clr
    h = (h or d * 0.9) + 0.3
    return m3.Manifold.cylinder(h, af / 2 / math.cos(math.pi / 6), af / 2 / math.cos(math.pi / 6), 6)


if __name__ == "__main__":
    # 자체 검사: 볼트와 너트가 겹치지 않고(틈), 나사산이 맞물림(조금 돌리면 겹침)
    b = male(6, 10)
    n_cut = female_cut(6, 10)
    blk = m3.Manifold.cube([14, 14, 10], center=True).translate([0, 0, 5]) - n_cut
    print("bolt∩nut(제자리) mm3", round((b ^ blk).volume(), 3))
    print("bolt∩nut(반 바퀴 어긋남) mm3", round((b.rotate([0, 0, 180]) ^ blk).volume(), 3))
    print("bolt∩nut(0.75 mm 올림) mm3", round((b.translate([0, 0, 0.75]) ^ blk).volume(), 3))


# ---------------- 표준(ISO) 미터 수나사: 기존 쇠 암나사 구멍(LIMO M3 등)에 들어가는 출력 나사 ----------------
def _iso_profile_r(t, d, pitch):
    """한 피치 안 위치 t(0~1)에서 수나사 반지름. ISO 기본 산형(H = 0.866p), 산 끝 p/8·골 p/4 평평."""
    H = 0.866025 * pitch
    r_maj = d / 2 - 0.08                       # 출력 여유(바깥 지름 0.16 작게)
    r_min = r_maj - 5 * H / 8
    # 산 끝 평평 p/8, 옆면 각 3p/8 구간 경사, 골 평평 p/4 ... 대칭
    x = (t % 1.0) * pitch
    a, b, c = pitch / 16, pitch / 16 + 5 * pitch / 16 * 1.0, pitch
    flat_top = pitch / 8
    flank = 5 * pitch / 16
    if x < flat_top / 2 or x > pitch - flat_top / 2:
        return r_maj
    x2 = x - flat_top / 2
    if x2 < flank:
        return r_maj - (r_maj - r_min) * x2 / flank
    x2 -= flank
    if x2 < pitch / 4:
        return r_min
    x2 -= pitch / 4
    return r_min + (r_maj - r_min) * x2 / flank


def iso_male(d=3.0, length=6.0, pitch=0.5, n=96):
    """ISO 수나사 몸(z 0 → length): 단면 r(θ) 를 피치당 한 바퀴 꼬아 올림(나선 산), 끝 모따기."""
    pts = []
    for i in range(n):
        th = 2 * math.pi * i / n
        r = _iso_profile_r(i / n, d, pitch)
        pts.append((r * math.cos(th), r * math.sin(th)))
    cs = m3.CrossSection([pts])
    turns = length / pitch
    t = cs.extrude(length, n_divisions=max(16, int(turns * 32)), twist_degrees=-360.0 * turns)
    ch = m3.Manifold.cylinder(length, d / 2, d / 2, SEG) - \
        (m3.Manifold.cylinder(1.0, d / 2 + 1, d / 2 + 1, SEG).translate([0, 0, length - 0.4]) -
         m3.Manifold.cylinder(1.0, d / 2 - 0.4, d / 2 + 0.6, SEG).translate([0, 0, length - 0.4]))
    return t ^ ch


def iso_screw(d=3.0, length=6.0, pitch=0.5, head_d=6.0, head_h=2.4, hex_key=None, slot=True):
    """출력 ISO 나사: 둥근 머리(z −head_h → 0) + 일자 홈 + 나사(z 0 → length)."""
    head = m3.Manifold.cylinder(head_h, head_d / 2, head_d / 2, SEG).translate([0, 0, -head_h])
    if slot:
        head -= m3.Manifold.cube([0.9, head_d + 2, head_h]).translate([-0.45, -head_d / 2 - 1, -head_h - 0.01])
    return head + iso_male(d, length, pitch)


def iso_screw_print(d=3.0, length=6.0, pitch=0.5, head_d=6.0, head_h=2.4):
    """눕혀 출력하는 자세: 축이 x, 바닥(z 0)에 0.3 mm 평평하게 깎음(층이 나사 길이 방향 → 안 부러짐)."""
    s = iso_screw(d, length, pitch, head_d, head_h).rotate([0, 90, 0])
    cut = d / 2 - 0.3
    s = s ^ m3.Manifold.cube([200, 200, 200]).translate([-100, -100, -cut])
    return s.translate([0, 0, cut])


def pan_bolt(d=5.0, length=10.0, pitch=1.25, e=0.4, head_d=9.0, head_h=3.0, slot_w=1.4, slot_d=1.6):
    """둥근 머리 + 일자 홈 볼트: 머리(z −head_h → 0) + 나사(z 0 → length).
    자리파기(Ø head_d + 1, 깊이 head_h + 0.2)에 파묻어 머리가 겉면 밖으로 안 나오게 쓰는 볼트.
    육각 머리는 파묻으면 돌릴 수 없으므로 일자 드라이버 홈을 낸다. 머리를 판에 대고 세워 출력."""
    head = m3.Manifold.cylinder(head_h, head_d / 2, head_d / 2, 48).translate([0, 0, -head_h])
    head -= m3.Manifold.cube([head_d + 2, slot_w, slot_d + 0.1]).translate([-(head_d + 2) / 2, -slot_w / 2, -head_h - 0.1])
    return head + male(d, length, pitch, e)


def counterbore(head_d=9.0, head_h=3.0, clr=0.5, extra=0.2, up=20.0):
    """pan_bolt 머리 자리: 머리 밑면 z 0 기준, 아래로 head_h + extra 까지가 머리, 위(−z 쪽 바깥)로 up 만큼 열림.
    반환 원통은 z −(head_h + extra + up) → 0, 반지름 head_d/2 + clr."""
    r = head_d / 2 + clr
    return m3.Manifold.cylinder(head_h + extra + up, r, r, 48).translate([0, 0, -(head_h + extra + up)])
