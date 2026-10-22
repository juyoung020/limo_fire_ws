//! 덕트호스 시연장 절차 생성기.
//! 기말 시연장(강의실 바닥에 덕트호스로 통로를 만들고 라바콘·박스를 놓은 미로)을 흉내 낸다.
//! 덕트호스 ≈ 지름 0.3 m 의 꺾인 띠, 박스 ≈ 0.4 m 정사각형, 라바콘 ≈ 지름 0.3 m 원.
//! 구역(방) = 2 × 2 사분면, 비상구 = 외곽 벽에 낸 1 m 출입구 1–2 곳.

use crate::{FMap, Rng, Room, RoomType, FREE, WALL};

const HOSE: f32 = 0.30;
const BOX: f32 = 0.40;
const CONE_R: f32 = 0.15;
const OUTER: f32 = 0.10;

pub fn generate(seed: u64) -> FMap {
    let mut r = Rng(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1);
    let (aw, ah) = (r.range(5.0, 8.0), r.range(6.0, 10.0));
    let mut m = FMap::new(0.0, 0.0, aw, ah, 0.5);
    m.fill_polygon(&[(0.0, 0.0), (aw, 0.0), (aw, ah), (0.0, ah)], FREE, 0);
    let corners = [(0.0, 0.0), (aw, 0.0), (aw, ah), (0.0, ah)];
    for k in 0..4 { m.draw_segment(corners[k], corners[(k + 1) % 4], OUTER, WALL); }

    // 덕트호스: 벽이나 안쪽 한 점에서 시작해 축 방향으로 2–3 번 꺾이는 띠
    for _ in 0..(3 + r.below(4)) {
        let mut p = if r.unit() < 0.6 {
            match r.below(4) { 0 => (r.range(0.8, aw - 0.8), 0.0), 1 => (aw, r.range(0.8, ah - 0.8)), 2 => (r.range(0.8, aw - 0.8), ah), _ => (0.0, r.range(0.8, ah - 0.8)) }
        } else { (r.range(1.0, aw - 1.0), r.range(1.0, ah - 1.0)) };
        let mut horizontal = r.unit() < 0.5;
        for _ in 0..(2 + r.below(2)) {
            let len = r.range(1.2, 3.5) * if r.unit() < 0.5 { -1.0 } else { 1.0 };
            let q = if horizontal { ((p.0 + len).clamp(0.0, aw), p.1) } else { (p.0, (p.1 + len).clamp(0.0, ah)) };
            m.draw_segment(p, q, HOSE, WALL);
            p = q;
            horizontal = !horizontal;
        }
    }
    // 박스·라바콘
    for _ in 0..(3 + r.below(6)) {
        let (cx, cy) = (r.range(0.6, aw - 0.6), r.range(0.6, ah - 0.6));
        let h = BOX * 0.5;
        m.fill_polygon(&[(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)], WALL, 0);
    }
    for _ in 0..(2 + r.below(4)) { m.fill_disc((r.range(0.5, aw - 0.5), r.range(0.5, ah - 0.5)), CONE_R, WALL); }

    // 출입구(비상구): 외곽 벽 1–2 곳을 1 m 뚫는다. 바깥은 OUTSIDE 라 새어 나가지 않는다.
    for _ in 0..(1 + r.below(2)) {
        let (a, b, mid) = match r.below(4) {
            0 => { let x = r.range(1.0, aw - 1.0); ((x - 0.5, 0.0), (x + 0.5, 0.0), (x, 0.2)) }
            1 => { let y = r.range(1.0, ah - 1.0); ((aw, y - 0.5), (aw, y + 0.5), (aw - 0.2, y)) }
            2 => { let x = r.range(1.0, aw - 1.0); ((x - 0.5, ah), (x + 0.5, ah), (x, ah - 0.2)) }
            _ => { let y = r.range(1.0, ah - 1.0); ((0.0, y - 0.5), (0.0, y + 0.5), (0.2, y)) }
        };
        m.draw_segment(a, b, OUTER * 1.5, FREE);
        m.exits.push(mid);
    }
    // 구역: 사분면
    for _ in 0..4 { m.rooms.push(Room { kind: RoomType::Zone, area: 0.0 }); }
    for j in 0..m.h { for i in 0..m.w {
        let k = m.idx(i, j);
        if m.cell[k] != FREE { continue; }
        let (x, y) = m.center(i, j);
        m.room[k] = 1 + (x >= aw * 0.5) as u16 + 2 * (y >= ah * 0.5) as u16;
    } }
    m
}
