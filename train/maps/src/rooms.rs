//! 절차 생성 방: 직사각 바깥 테두리를 BSP 로 나눠 방·복도를 만들고, 나눈 벽마다 문 하나를 낸다.
//! 탐사 RL 학습 맵 묶음에서 작은 방(< 30 m²)·작은 집 쪽 넓이를 채운다(ProcTHOR 집은 대부분 그보다 크다).
//! - 바깥 벽은 막힘(문 없음) — 학습 장면은 집이 닫혀 있어야 한다(sim build_world).
//! - 나눌 때 얇은 쪽(1.0~1.6 m)을 떼면 복도가 된다.
//! - 문 폭 0.8~1.1 m (실내 문 0.8~0.9 m 에 여유). LIMO 몸통 폭 약 0.22 m 라 지나간다.
//! - 큰 방(> 6 m²)에는 가구 크기 상자(0.4~1.0 m)를 0~2 개. 문 앞을 막아도 keep_reachable 이 가장 큰 영역만 남긴다.

use crate::{FMap, Rng, Room, RoomType, FREE, WALL};

const WALL_THICK: f32 = 0.10;

/// 넓이 등급 grade(0~3)에 맞는 크기로 하나 만든다. 같은 seed → 같은 맵.
pub fn generate(seed: u64, grade: u32) -> FMap {
    let mut r = Rng(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1);
    // 바깥 크기(m) — 등급별 목표 넓이 대략: 0 → 6~28, 1 → 30~80, 2 → 80~150, 3 → 150~400
    let (lo, hi) = match grade { 0 => (2.5, 6.0), 1 => (5.0, 10.0), 2 => (8.0, 14.0), _ => (11.0, 22.0) };
    let aw = r.range(lo, hi);
    let ah = r.range(lo, hi).min(aw * 2.5).max(aw / 2.5);
    let mut m = FMap::new(0.0, 0.0, aw, ah, 0.3);
    m.fill_polygon(&[(0.0, 0.0), (aw, 0.0), (aw, ah), (0.0, ah)], FREE, 0);
    let corners = [(0.0, 0.0), (aw, 0.0), (aw, ah), (0.0, ah)];
    for k in 0..4 { m.draw_segment(corners[k], corners[(k + 1) % 4], WALL_THICK, WALL); }
    // BSP: 방 하나가 목표 넓이보다 크면 나눈다
    let target = r.range(8.0, 25.0);
    let mut stack = vec![(0.0f32, 0.0f32, aw, ah)];
    let mut leaves: Vec<(f32, f32, f32, f32)> = Vec::new();
    while let Some((x0, y0, x1, y1)) = stack.pop() {
        let (w, h) = (x1 - x0, y1 - y0);
        if w * h < target * 1.3 || (w < 3.0 && h < 3.0) { leaves.push((x0, y0, x1, y1)); continue; }
        let vertical = if w > h * 1.2 { true } else if h > w * 1.2 { false } else { r.unit() < 0.5 };
        let span = if vertical { w } else { h };
        // 복도: 긴 쪽이 6 m 넘으면 가끔 얇은 띠를 뗀다
        let cut = if span > 6.0 && r.unit() < 0.3 {
            let c = r.range(1.0, 1.6);
            if r.unit() < 0.5 { c } else { span - c }
        } else {
            span * r.range(0.35, 0.65)
        };
        let (a, b) = if vertical { ((x0 + cut, y0), (x0 + cut, y1)) } else { ((x0, y0 + cut), (x1, y0 + cut)) };
        m.draw_segment(a, b, WALL_THICK, WALL);
        // 문: 벽 끝에서 0.3 m 이상 떨어진 곳
        let len = if vertical { h } else { w };
        let dw = r.range(0.8, 1.1).min(len - 0.6);
        if dw > 0.5 {
            let t = r.range(0.3, len - 0.3 - dw);
            let (p, q) = if vertical { ((a.0, y0 + t), (a.0, y0 + t + dw)) } else { ((x0 + t, a.1), (x0 + t + dw, a.1)) };
            m.draw_segment(p, q, WALL_THICK * 1.6, FREE);
        }
        if vertical { stack.push((x0, y0, x0 + cut, y1)); stack.push((x0 + cut, y0, x1, y1)); }
        else { stack.push((x0, y0, x1, y0 + cut)); stack.push((x0, y0 + cut, x1, y1)); }
    }
    // 방 번호·가구 상자
    for &(x0, y0, x1, y1) in &leaves {
        let narrow = (x1 - x0).min(y1 - y0) < 1.7;
        m.rooms.push(Room { kind: if narrow { RoomType::Corridor } else { RoomType::Other }, area: 0.0 });
        let rid = m.rooms.len() as u16;
        for j in 0..m.h { for i in 0..m.w {
            let k = m.idx(i, j);
            let (x, y) = m.center(i, j);
            if m.cell[k] == FREE && x > x0 && x < x1 && y > y0 && y < y1 { m.room[k] = rid; }
        } }
        if narrow || (x1 - x0) * (y1 - y0) < 6.0 { continue; }
        for _ in 0..r.below(3) {
            let (bw, bh) = (r.range(0.4, 1.0), r.range(0.4, 1.0));
            if x1 - x0 < bw + 1.2 || y1 - y0 < bh + 1.2 { continue; }
            let (cx, cy) = (r.range(x0 + 0.6 + bw * 0.5, x1 - 0.6 - bw * 0.5), r.range(y0 + 0.6 + bh * 0.5, y1 - 0.6 - bh * 0.5));
            let (hx, hy) = (bw * 0.5, bh * 0.5);
            m.fill_polygon(&[(cx - hx, cy - hy), (cx + hx, cy - hy), (cx + hx, cy + hy), (cx - hx, cy + hy)], WALL, 0);
        }
    }
    m
}
