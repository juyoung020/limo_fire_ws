//! 절차 가구: 학습 격자(빈 칸 / 막힘)에 가구·장애물을 놓는다. 시험 맵(AWS 작은 집·서점)은 가구가 많은데 ProcTHOR 집은 벽·문뿐이라서.
//!
//! 종류 (모두 축에 맞춘 직사각, 칸 = 5 cm)
//! - 벽에 붙은 상자: 소파·책장·침대·장 크기 0.4~2.0 m × 0.4~1.0 m, 높이 0.4~2.0 m. 벽과 한 칸(5 cm) 띄운다(아래 "덩어리").
//! - 탁자 + 의자: 탁자 0.7~1.6 × 0.6~1.0 m, 의자 0.4~0.5 m. **다리만** 막는다(다리 하나 = 1~2 칸 기둥) — 라이다 평면(0.111 m)에서는
//!   실물도 다리만 보인다. 판·상판은 만들지 않는다(깊이 카메라에는 다리 기둥만 보임 — 한계).
//! - 작은 물체: 0.1~0.3 m 상자, 높이 0.15~0.4 m (라이다 평면보다 높게 — 낮은 물체는 일부러 만들지 않음).
//!
//! 덩어리: sim 의 map2scene 은 4 이웃으로 붙은 막힌 칸 덩어리를 물체 하나로 만들고, 라이다 재질(lidar_real.h material_of)은 물체마다 판마다 뽑는다.
//! 가구 하나하나가 따로 재질을 받게 놓을 칸과 그 8 이웃이 모두 빈 칸일 때만 놓는다(벽·다른 가구와 한 칸 이상 떨어짐).
//! 탁자·의자 다리도 각각 따로 덩어리라 재질도 다리마다 뽑힌다(실물은 한 가구의 다리가 같은 재질 — 한계).
//!
//! 막지 않는 곳
//! - 출발점 0.6 m 안.
//! - 좁은 곳(문 틈·좁은 복도): 가로나 세로로 빈 칸이 이어진 길이가 1.3 m 미만인 칸 — 문 폭은 0.8~1.1 m 라 문 칸은 모두 여기 든다.
//!   그 칸에서 0.5 m 안에는 놓지 않는다.
//! - 고립: 가구를 하나 놓을 때마다 "로봇이 설 수 있는 칸"(막힘까지 0.15 m 이상, fmap ROBOT_CLEARANCE)을 출발점에서 다시 번지고,
//!   가구 전에 설 수 있었고 지금도 설 수 있는 칸 중 하나라도 못 닿으면 그 가구를 뺀다(방·구석이 갇힘).
//!
//! 밀도: 맵마다 씨앗으로 배율 0.5~1.5 를 뽑고, 방(방 번호가 같은 빈 칸)마다 넓이에 비례해 개수를 정한다.

use crate::Rng;
use std::collections::VecDeque;

pub const RES: f32 = crate::RES;
const STAND_CLEAR: f32 = crate::ROBOT_CLEARANCE;
const START_KEEP: f32 = 0.6;
const NARROW_RUN: f32 = 1.3;
const NARROW_KEEP: f32 = 0.5;

/// 놓은 가구 수 (종류별)
#[derive(Default, Debug, Clone, Copy)]
pub struct Count { pub wall_box: u32, pub table: u32, pub chair: u32, pub small: u32, pub rejected: u32 }

pub struct Grid<'a> {
    pub w: usize,
    pub h: usize,
    /// true = 막힘. 바뀐다(가구를 놓으면 막힘이 됨)
    pub occ: &'a mut Vec<bool>,
    /// 칸 높이(cm), 0 = 벽(기본 높이). 가구 칸만 채운다
    pub height: &'a mut Vec<u8>,
    pub room: &'a [u16],
    pub start: (usize, usize),
}

fn dist_map(w: usize, h: usize, occ: &[bool]) -> Vec<f32> {
    // 가장 가까운 막힘 칸 가운데까지 거리(m, 8 이웃 BFS 근사) — fmap::FMap::wall_distance 와 같은 방식
    let n = w * h;
    let mut d = vec![f32::INFINITY; n];
    let mut q = VecDeque::new();
    for k in 0..n { if occ[k] { d[k] = 0.0; q.push_back(k); } }
    const D: f32 = RES * std::f32::consts::SQRT_2;
    while let Some(k) = q.pop_front() {
        let (i, j) = ((k % w) as i64, (k / w) as i64);
        for (di, dj, s) in [(1, 0, RES), (-1, 0, RES), (0, 1, RES), (0, -1, RES), (1, 1, D), (1, -1, D), (-1, 1, D), (-1, -1, D)] {
            let (u, v) = (i + di, j + dj);
            if u < 0 || v < 0 || u as usize >= w || v as usize >= h { continue; }
            let m = v as usize * w + u as usize;
            if d[k] + s + 1e-6 < d[m] { d[m] = d[k] + s; q.push_back(m); }
        }
    }
    d
}

/// 출발점에서 설 수 있는 칸(dist ≥ STAND_CLEAR)을 4 이웃으로 번진 결과
fn reach(w: usize, h: usize, dist: &[f32], start: usize) -> Vec<bool> {
    let mut seen = vec![false; w * h];
    if dist[start] < STAND_CLEAR { return seen; }
    let mut q = VecDeque::from([start]);
    seen[start] = true;
    while let Some(k) = q.pop_front() {
        let (i, j) = (k % w, k / w);
        let nb = [(i + 1, j), (i.wrapping_sub(1), j), (i, j + 1), (i, j.wrapping_sub(1))];
        for (u, v) in nb {
            if u >= w || v >= h { continue; }
            let m = v * w + u;
            if !seen[m] && dist[m] >= STAND_CLEAR { seen[m] = true; q.push_back(m); }
        }
    }
    seen
}

/// 맵 하나에 가구를 놓는다. 같은 seed → 같은 결과.
pub fn furnish(g: &mut Grid, seed: u64) -> Count {
    let (w, h) = (g.w, g.h);
    let n = w * h;
    let mut r = Rng(seed.wrapping_mul(0xA24B_AED4_963E_E407) | 1);
    let mut cnt = Count::default();
    let start = g.start.1 * w + g.start.0;
    // 막지 않을 칸
    let mut keep = vec![false; n];
    {
        let rk = (START_KEEP / RES).ceil() as i64;
        for dj in -rk..=rk { for di in -rk..=rk {
            if ((di * di + dj * dj) as f32) * RES * RES > START_KEEP * START_KEEP { continue; }
            let (u, v) = (g.start.0 as i64 + di, g.start.1 as i64 + dj);
            if u >= 0 && v >= 0 && (u as usize) < w && (v as usize) < h { keep[v as usize * w + u as usize] = true; }
        } }
        // 좁은 칸: 가로·세로 빈 칸 줄 길이
        let mut narrow = vec![false; n];
        let lim = (NARROW_RUN / RES).round() as usize;
        for j in 0..h {
            let mut i = 0;
            while i < w {
                if g.occ[j * w + i] { i += 1; continue; }
                let s = i;
                while i < w && !g.occ[j * w + i] { i += 1; }
                if i - s < lim { for t in s..i { narrow[j * w + t] = true; } }
            }
        }
        for i in 0..w {
            let mut j = 0;
            while j < h {
                if g.occ[j * w + i] { j += 1; continue; }
                let s = j;
                while j < h && !g.occ[j * w + i] { j += 1; }
                if j - s < lim { for t in s..j { narrow[t * w + i] = true; } }
            }
        }
        // 좁은 칸에서 NARROW_KEEP 안 (빈 칸 위로 번짐)
        let steps = (NARROW_KEEP / RES).ceil() as usize;
        let mut d = vec![usize::MAX; n];
        let mut q = VecDeque::new();
        for k in 0..n { if narrow[k] { d[k] = 0; q.push_back(k); } }
        while let Some(k) = q.pop_front() {
            keep[k] = true;
            if d[k] >= steps { continue; }
            let (i, j) = (k % w, k / w);
            for (u, v) in [(i + 1, j), (i.wrapping_sub(1), j), (i, j + 1), (i, j.wrapping_sub(1))] {
                if u >= w || v >= h { continue; }
                let m = v * w + u;
                if !g.occ[m] && d[m] == usize::MAX { d[m] = d[k] + 1; q.push_back(m); }
            }
        }
    }
    let base_dist = dist_map(w, h, g.occ);
    let base_reach = reach(w, h, &base_dist, start);
    if !base_reach[start] { return cnt; }

    // 칸 덩어리 하나(직사각 i0..i1 × j0..j1, 끝 포함)를 놓을 수 있나: 칸과 8 이웃이 모두 빈 칸, 지킬 칸 아님
    let can = |occ: &Vec<bool>, i0: i64, j0: i64, i1: i64, j1: i64| -> bool {
        if i0 < 1 || j0 < 1 || i1 as usize + 1 >= w || j1 as usize + 1 >= h || i1 < i0 || j1 < j0 { return false; }
        for j in j0 - 1..=j1 + 1 { for i in i0 - 1..=i1 + 1 {
            let k = j as usize * w + i as usize;
            if occ[k] { return false; }
            if i >= i0 && i <= i1 && j >= j0 && j <= j1 && keep[k] { return false; }
        } }
        true
    };
    // 여러 직사각(한 가구의 다리들 등)을 한 번에 놓고 고립 검사, 실패면 되돌림
    let place = |g: &mut Grid, rects: &[(i64, i64, i64, i64)], hcm: u8, cnt: &mut Count| -> bool {
        for &(i0, j0, i1, j1) in rects { if !can(g.occ, i0, j0, i1, j1) { return false; } }
        // 서로 붙은 직사각이 없게(다리끼리도 따로 덩어리): 앞에서 놓은 것과 can 이 이미 검사 — 차례로 놓으며 다시 확인
        let mut placed: Vec<usize> = Vec::new();
        for &(i0, j0, i1, j1) in rects {
            if !can(g.occ, i0, j0, i1, j1) { for &k in &placed { g.occ[k] = false; } return false; }
            for j in j0..=j1 { for i in i0..=i1 { let k = j as usize * w + i as usize; g.occ[k] = true; placed.push(k); } }
        }
        let d = dist_map(w, h, g.occ);
        let now = reach(w, h, &d, start);
        let isolated = (0..n).any(|k| base_reach[k] && d[k] >= STAND_CLEAR && !now[k]);
        if isolated {
            for &k in &placed { g.occ[k] = false; }
            cnt.rejected += 1;
            return false;
        }
        for &k in &placed { g.height[k] = hcm; }
        true
    };

    // 방 목록: 방 번호 → 빈 칸들. 방 번호가 없으면(0) 전체를 한 방으로
    let max_room = g.room.iter().copied().max().unwrap_or(0) as usize;
    let mut rooms: Vec<Vec<usize>> = vec![Vec::new(); max_room + 1];
    for k in 0..n { if !g.occ[k] { rooms[g.room[k] as usize].push(k); } }
    let density = r.range(0.5, 1.5);
    let cells = |m: f32| ((m / RES).round() as i64).max(1);
    for cellsv in rooms.iter() {
        let area = cellsv.len() as f32 * RES * RES;
        if area < 2.0 { continue; }
        let n_wall = (area / 7.0 * density).round() as usize;
        let n_table = if area >= 8.0 { (area / 25.0 * density + r.unit()).floor() as usize } else { 0 };
        let n_small = (area / 6.0 * density).round() as usize;
        // 벽에 붙은 상자: 벽에서 두 칸째 빈 칸(사이 한 칸)에서 벽 반대쪽으로
        //   후보(칸, 벽 쪽 방향)를 먼저 모은다 — 무작위 칸·방향으로 찾으면 거의 안 맞음
        let dirs = [(1i64, 0i64), (-1, 0), (0, 1), (0, -1)];
        let anchors: Vec<(usize, (i64, i64))> = {
            let at = |u: i64, v: i64| -> bool { u < 0 || v < 0 || u as usize >= w || v as usize >= h || g.occ[v as usize * w + u as usize] };
            cellsv.iter().flat_map(|&k| dirs.iter().map(move |&d| (k, d)))
                .filter(|&(k, d)| { let (i, j) = ((k % w) as i64, (k / w) as i64); at(i + 2 * d.0, j + 2 * d.1) && !at(i + d.0, j + d.1) }).collect()
        };
        let mut placed = 0;
        for _ in 0..n_wall * 12 {
            if placed >= n_wall || anchors.is_empty() { break; }
            let (k, d) = anchors[r.below(anchors.len())];
            let (i, j) = ((k % w) as i64, (k / w) as i64);
            let (len, dep) = (cells(r.range(0.4, 2.0)), cells(r.range(0.4, 1.0)));
            // d 가 벽 쪽. 깊이는 -d 로, 길이는 d 에 수직으로 가운데 맞춤
            let (a0, a1) = (-(len / 2), len - len / 2 - 1);
            let rect = if d.0 != 0 {
                let (x0, x1) = if d.0 > 0 { (i - dep + 1, i) } else { (i, i + dep - 1) };
                (x0, j + a0, x1, j + a1)
            } else {
                let (y0, y1) = if d.1 > 0 { (j - dep + 1, j) } else { (j, j + dep - 1) };
                (i + a0, y0, i + a1, y1)
            };
            let hcm = (r.range(0.4, 2.0) * 100.0) as u8;
            if place(g, &[rect], hcm, &mut cnt) { placed += 1; cnt.wall_box += 1; }
        }
        // 탁자 + 의자: 벽에서 1 m 넘게 떨어진 곳
        let mut placed = 0;
        for _ in 0..n_table * 12 {
            if placed >= n_table { break; }
            let k = cellsv[r.below(cellsv.len())];
            if base_dist[k] < 1.0 { continue; }
            let (ci, cj) = ((k % w) as i64, (k / w) as i64);
            let (tw, th) = (cells(r.range(0.7, 1.6)), cells(r.range(0.6, 1.0)));
            let (tw, th) = if r.unit() < 0.5 { (tw, th) } else { (th, tw) };
            let leg = 1 + r.below(2) as i64;   // 다리 굵기 5~10 cm
            let (x0, y0) = (ci - tw / 2, cj - th / 2);
            let (x1, y1) = (x0 + tw - 1, y0 + th - 1);
            let legs = |x0: i64, y0: i64, x1: i64, y1: i64, l: i64| {
                [(x0, y0, x0 + l - 1, y0 + l - 1), (x1 - l + 1, y0, x1, y0 + l - 1), (x0, y1 - l + 1, x0 + l - 1, y1), (x1 - l + 1, y1 - l + 1, x1, y1)]
            };
            if !place(g, &legs(x0, y0, x1, y1, leg), 75, &mut cnt) { continue; }
            placed += 1;
            cnt.table += 1;
            // 의자: 탁자 네 변 바깥에 0~4 개 (탁자와 0.05~0.15 m 띄움)
            for side in 0..4 {
                if r.unit() < 0.35 { continue; }
                let s = cells(r.range(0.4, 0.5));
                let gap = 1 + r.below(3) as i64;
                let (cx0, cy0) = match side {
                    0 => (x0 + (tw - s) / 2, y0 - gap - s),
                    1 => (x0 + (tw - s) / 2, y1 + gap + 1),
                    2 => (x0 - gap - s, y0 + (th - s) / 2),
                    _ => (x1 + gap + 1, y0 + (th - s) / 2),
                };
                if place(g, &legs(cx0, cy0, cx0 + s - 1, cy0 + s - 1, 1), 45, &mut cnt) { cnt.chair += 1; }
            }
        }
        // 작은 물체
        let mut placed = 0;
        for _ in 0..n_small * 12 {
            if placed >= n_small { break; }
            let k = cellsv[r.below(cellsv.len())];
            if base_dist[k] < 0.3 { continue; }
            let (i, j) = ((k % w) as i64, (k / w) as i64);
            let (sw, sh) = (cells(r.range(0.1, 0.3)), cells(r.range(0.1, 0.3)));
            let hcm = (r.range(0.15, 0.4) * 100.0) as u8;
            if place(g, &[(i, j, i + sw - 1, j + sh - 1)], hcm, &mut cnt) { placed += 1; cnt.small += 1; }
        }
    }
    cnt
}
