//! 공통 맵 형식(.fmap)과 소방 시설 배치.
//!
//! 모든 데이터셋을 같은 점유 격자로 바꾼 뒤, 로봇이 갈 수 있는 영역(도달 영역)을 남기고
//! 소화기·비상구·화재 감지기를 놓는다. 불은 여기서 놓지 않는다 — GPU 환경이 판마다
//! 도달 영역 안 무작위 칸에 만든다.
//!
//! .fmap (리틀 엔디언)
//! ```text
//! "FMAP" u32 version(=1)
//! u32 w, u32 h, f32 res(m), f32 origin_x, f32 origin_y      // 칸 (i, j) 중심 = origin + (i + 0.5, j + 0.5) * res
//! u8  cell[w*h]          0 바깥·도달 불가, 1 빈 곳(도달), 2 벽·장애물 — 광선은 0 과 2 에서 모두 막힌다
//! u16 room[w*h]          0 없음, 1.. 방 번호
//! u32 n_rooms, { u8 type, f32 area_m2 } * n_rooms
//! u32 n, { f32 x, f32 y } * n   소화기
//! u32 n, { f32 x, f32 y } * n   비상구
//! u32 n, { f32 x, f32 y } * n   화재 감지기
//! ```

use std::collections::VecDeque;
use std::io::{self, Write};

pub mod duct;
pub mod houseexpo;
pub mod procthor;

pub const OUTSIDE: u8 = 0;
pub const FREE: u8 = 1;
pub const WALL: u8 = 2;

/// 격자 해상도(m). LIMO 라이다·지도와 같은 5 cm.
pub const RES: f32 = 0.05;
/// 로봇 중심이 벽에서 떨어져야 하는 거리(m). LIMO 몸통 반폭 약 0.11 m + 여유.
pub const ROBOT_CLEARANCE: f32 = 0.15;
/// 소화기 보행거리(m). 화재안전기준 소형소화기 20 m.
pub const EXT_WALK_M: f32 = 20.0;
/// 이 넓이 이상인 방에는 소화기를 하나 더 둔다(m²).
pub const EXT_ROOM_AREA_M2: f32 = 33.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum RoomType {
    None = 0,
    Bedroom = 1,
    Living = 2,
    Kitchen = 3,
    Bathroom = 4,
    Corridor = 5,
    Storage = 6,
    Office = 7,
    Zone = 8,
    Other = 9,
}

impl RoomType {
    pub fn from_name(s: &str) -> RoomType {
        let s = s.to_ascii_lowercase();
        if s.contains("bed") { RoomType::Bedroom }
        else if s.contains("living") || s.contains("lounge") || s.contains("dining") { RoomType::Living }
        else if s.contains("kitchen") { RoomType::Kitchen }
        else if s.contains("bath") || s.contains("toilet") { RoomType::Bathroom }
        else if s.contains("hall") || s.contains("corridor") || s.contains("lobby") || s.contains("entr") { RoomType::Corridor }
        else if s.contains("storage") || s.contains("garage") { RoomType::Storage }
        else if s.contains("office") || s.contains("study") { RoomType::Office }
        else { RoomType::Other }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Room {
    pub kind: RoomType,
    pub area: f32,
}

pub struct FMap {
    pub w: usize,
    pub h: usize,
    pub origin: (f32, f32),
    pub cell: Vec<u8>,
    pub room: Vec<u16>,
    pub rooms: Vec<Room>,
    pub extinguishers: Vec<(f32, f32)>,
    pub exits: Vec<(f32, f32)>,
    pub detectors: Vec<(f32, f32)>,
}

impl FMap {
    /// (x0, y0)–(x1, y1) 를 덮는 빈 격자(전부 바깥). 둘레에 여백 margin(m).
    pub fn new(x0: f32, y0: f32, x1: f32, y1: f32, margin: f32) -> FMap {
        let ox = x0 - margin;
        let oy = y0 - margin;
        let w = ((x1 - x0 + 2.0 * margin) / RES).ceil() as usize + 1;
        let h = ((y1 - y0 + 2.0 * margin) / RES).ceil() as usize + 1;
        FMap {
            w, h, origin: (ox, oy),
            cell: vec![OUTSIDE; w * h], room: vec![0; w * h], rooms: Vec::new(),
            extinguishers: Vec::new(), exits: Vec::new(), detectors: Vec::new(),
        }
    }

    #[inline] pub fn idx(&self, i: usize, j: usize) -> usize { j * self.w + i }
    #[inline] pub fn center(&self, i: usize, j: usize) -> (f32, f32) {
        (self.origin.0 + (i as f32 + 0.5) * RES, self.origin.1 + (j as f32 + 0.5) * RES)
    }
    pub fn to_cell(&self, x: f32, y: f32) -> Option<(usize, usize)> {
        let i = ((x - self.origin.0) / RES).floor();
        let j = ((y - self.origin.1) / RES).floor();
        if i < 0.0 || j < 0.0 || i as usize >= self.w || j as usize >= self.h { None } else { Some((i as usize, j as usize)) }
    }

    /// 다각형 안(짝수-홀수 규칙, 칸 중심 기준)을 value 로 칠한다. room 이 0 이 아니면 방 번호도.
    pub fn fill_polygon(&mut self, poly: &[(f32, f32)], value: u8, room: u16) {
        if poly.len() < 3 { return; }
        for j in 0..self.h {
            let y = self.origin.1 + (j as f32 + 0.5) * RES;
            let mut xs: Vec<f32> = Vec::new();
            for k in 0..poly.len() {
                let (ax, ay) = poly[k];
                let (bx, by) = poly[(k + 1) % poly.len()];
                if (ay <= y && by > y) || (by <= y && ay > y) {
                    xs.push(ax + (y - ay) / (by - ay) * (bx - ax));
                }
            }
            xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            for p in xs.chunks(2) {
                if p.len() < 2 { break; }
                let i0 = (((p[0] - self.origin.0) / RES) - 0.5).ceil().max(0.0) as usize;
                let i1 = (((p[1] - self.origin.0) / RES) - 0.5).floor();
                if i1 < 0.0 { continue; }
                let i1 = (i1 as usize).min(self.w - 1);
                for i in i0..=i1 {
                    let k = self.idx(i, j);
                    self.cell[k] = value;
                    if room != 0 { self.room[k] = room; }
                }
            }
        }
    }

    /// 선분을 두께 thick(m) 로 칠한다.
    pub fn draw_segment(&mut self, a: (f32, f32), b: (f32, f32), thick: f32, value: u8) {
        let r = thick * 0.5;
        let (minx, maxx) = (a.0.min(b.0) - r, a.0.max(b.0) + r);
        let (miny, maxy) = (a.1.min(b.1) - r, a.1.max(b.1) + r);
        let (Some((i0, j0)), Some((i1, j1))) = (self.to_cell(minx.max(self.origin.0), miny.max(self.origin.1)),
            self.to_cell(maxx.min(self.origin.0 + self.w as f32 * RES - 1e-4), maxy.min(self.origin.1 + self.h as f32 * RES - 1e-4))) else { return };
        let (dx, dy) = (b.0 - a.0, b.1 - a.1);
        let len2 = (dx * dx + dy * dy).max(1e-12);
        let r2 = (r.max(RES * 0.5)) * (r.max(RES * 0.5));
        for j in j0..=j1 {
            for i in i0..=i1 {
                let (x, y) = self.center(i, j);
                let t = (((x - a.0) * dx + (y - a.1) * dy) / len2).clamp(0.0, 1.0);
                let (px, py) = (a.0 + t * dx - x, a.1 + t * dy - y);
                if px * px + py * py <= r2 {
                    let k = self.idx(i, j);
                    self.cell[k] = value;
                    if value != FREE { self.room[k] = 0; }
                }
            }
        }
    }

    pub fn fill_disc(&mut self, c: (f32, f32), r: f32, value: u8) { self.draw_segment(c, c, 2.0 * r, value); }

    /// 로봇이 갈 수 있는 곳만 FREE 로 남긴다: 벽에서 ROBOT_CLEARANCE 이상 떨어진 빈 칸의
    /// 가장 큰 연결 영역 + 그 영역에서 벽 쪽으로 붙은 빈 칸(센서로는 보이는 곳). 나머지 빈 칸은 OUTSIDE.
    /// 반환: 도달 영역 넓이(m²).
    pub fn keep_reachable(&mut self) -> f32 {
        let n = self.w * self.h;
        let clear = self.wall_distance();
        let ok: Vec<bool> = (0..n).map(|k| self.cell[k] == FREE && clear[k] >= ROBOT_CLEARANCE).collect();
        // 가장 큰 연결 영역(4-이웃)
        let mut comp = vec![u32::MAX; n];
        let (mut best, mut best_size, mut cid) = (u32::MAX, 0usize, 0u32);
        for s in 0..n {
            if !ok[s] || comp[s] != u32::MAX { continue; }
            let mut q = VecDeque::from([s]);
            comp[s] = cid;
            let mut size = 0;
            while let Some(k) = q.pop_front() {
                size += 1;
                for nb in self.nb4(k) { if ok[nb] && comp[nb] == u32::MAX { comp[nb] = cid; q.push_back(nb); } }
            }
            if size > best_size { best_size = size; best = cid; }
            cid += 1;
        }
        // 도달 영역에서 빈 칸으로 번지기(벽 앞 좁은 띠 포함)
        let mut keep = vec![false; n];
        let mut q: VecDeque<usize> = VecDeque::new();
        for k in 0..n { if comp[k] == best && best != u32::MAX { keep[k] = true; q.push_back(k); } }
        while let Some(k) = q.pop_front() {
            for nb in self.nb4(k) { if self.cell[nb] == FREE && !keep[nb] { keep[nb] = true; q.push_back(nb); } }
        }
        for k in 0..n { if self.cell[k] == FREE && !keep[k] { self.cell[k] = OUTSIDE; self.room[k] = 0; } }
        // 도달 영역과 닿지 않는 벽은 바깥으로(외곽 정리)
        let mut touch = vec![false; n];
        for k in 0..n { if self.cell[k] == FREE { for nb in self.nb8(k) { touch[nb] = true; } } }
        for k in 0..n { if self.cell[k] == WALL && !touch[k] { self.cell[k] = OUTSIDE; } }
        best_size as f32 * RES * RES
    }

    /// 각 칸에서 가장 가까운 벽·바깥 칸까지 거리(m, 8-이웃 BFS 근사).
    pub fn wall_distance(&self) -> Vec<f32> {
        let n = self.w * self.h;
        let mut d = vec![f32::INFINITY; n];
        let mut q = VecDeque::new();
        for k in 0..n { if self.cell[k] != FREE { d[k] = 0.0; q.push_back(k); } }
        while let Some(k) = q.pop_front() {
            for (nb, step) in self.nb8w(k) {
                let nd = d[k] + step;
                if nd + 1e-6 < d[nb] { d[nb] = nd; q.push_back(nb); }
            }
        }
        d
    }

    /// 빈 칸 위 보행거리(m) — sources 에서 시작하는 다중 출발 BFS(8-이웃).
    pub fn walk_distance(&self, sources: &[usize]) -> Vec<f32> {
        let n = self.w * self.h;
        let mut d = vec![f32::INFINITY; n];
        let mut q = VecDeque::new();
        for &s in sources { d[s] = 0.0; q.push_back(s); }
        while let Some(k) = q.pop_front() {
            for (nb, step) in self.nb8w(k) {
                if self.cell[nb] != FREE { continue; }
                let nd = d[k] + step;
                if nd + 1e-6 < d[nb] { d[nb] = nd; q.push_back(nb); }
            }
        }
        d
    }

    fn nb4(&self, k: usize) -> impl Iterator<Item = usize> + '_ {
        let (i, j) = ((k % self.w) as i64, (k / self.w) as i64);
        [(1, 0), (-1, 0), (0, 1), (0, -1)].into_iter().filter_map(move |(di, dj)| self.at(i + di, j + dj))
    }
    fn nb8(&self, k: usize) -> impl Iterator<Item = usize> + '_ { self.nb8w(k).map(|(n, _)| n) }
    fn nb8w(&self, k: usize) -> impl Iterator<Item = (usize, f32)> + '_ {
        let (i, j) = ((k % self.w) as i64, (k / self.w) as i64);
        const D: f32 = RES * std::f32::consts::SQRT_2;
        [(1, 0, RES), (-1, 0, RES), (0, 1, RES), (0, -1, RES), (1, 1, D), (1, -1, D), (-1, 1, D), (-1, -1, D)]
            .into_iter().filter_map(move |(di, dj, s)| self.at(i + di, j + dj).map(|n| (n, s)))
    }
    fn at(&self, i: i64, j: i64) -> Option<usize> {
        if i < 0 || j < 0 || i as usize >= self.w || j as usize >= self.h { None } else { Some(j as usize * self.w + i as usize) }
    }

    /// 방 넓이 다시 세기(도달 영역 정리 뒤).
    pub fn recount_rooms(&mut self) {
        for r in self.rooms.iter_mut() { r.area = 0.0; }
        for k in 0..self.w * self.h {
            let r = self.room[k] as usize;
            if r > 0 && self.cell[k] == FREE && r <= self.rooms.len() { self.rooms[r - 1].area += RES * RES; }
        }
    }

    /// 소화기·비상구·감지기 배치. 비상구는 데이터셋이 준 것이 없을 때만 외벽에서 고른다.
    pub fn place_facilities(&mut self, seed: u64) {
        let n = self.w * self.h;
        let free: Vec<usize> = (0..n).filter(|&k| self.cell[k] == FREE).collect();
        if free.is_empty() { return; }
        let wd = self.wall_distance();
        // 벽에 붙은 빈 칸 = 소화기 후보
        let cand: Vec<bool> = (0..n).map(|k| self.cell[k] == FREE && wd[k] <= 1.5 * RES).collect();

        // 1) 보행거리 EXT_WALK_M 덮기 — 덮이지 않은 칸 중 가장 먼 칸 근처 벽에 하나씩
        let mut ext: Vec<usize> = Vec::new();
        let mut rng = Rng(seed | 1);
        loop {
            let d = if ext.is_empty() { vec![f32::INFINITY; n] } else { self.walk_distance(&ext) };
            let far = free.iter().copied().filter(|&k| d[k] > EXT_WALK_M)
                .max_by(|&a, &b| d[a].partial_cmp(&d[b]).unwrap().then(b.cmp(&a)));
            let Some(target) = (if ext.is_empty() { Some(free[rng.below(free.len())]) } else { far }) else { break };
            match self.nearest_where(target, &cand) { Some(c) => ext.push(c), None => break }
            if ext.len() >= 64 { break; }
        }
        // 2) 넓은 방에는 방마다 하나(그 방 안 벽 칸 중 방 중심에서 가까운 곳)
        for (ri, room) in self.rooms.iter().enumerate() {
            if room.area < EXT_ROOM_AREA_M2 { continue; }
            let rid = (ri + 1) as u16;
            if ext.iter().any(|&k| self.room[k] == rid) { continue; }
            let cells: Vec<usize> = free.iter().copied().filter(|&k| self.room[k] == rid).collect();
            if cells.is_empty() { continue; }
            let c = centroid(self, &cells);
            if let Some(&k) = cells.iter().filter(|&&k| cand[k]).min_by(|&&a, &&b| dist2(self.center_k(a), c).partial_cmp(&dist2(self.center_k(b), c)).unwrap()) {
                ext.push(k);
            }
        }
        self.extinguishers = ext.iter().map(|&k| self.center_k(k)).collect();

        // 3) 비상구: 데이터셋에 없으면 외벽 앞 칸 중 서로 가장 먼 1–2 곳.
        //    외벽 = 벽을 뚫고 곧게 나가도 0.6 m 안에 빈 곳이 다시 나오지 않는 쪽(안쪽 벽은 얇아 금방 빈 곳이 나온다).
        if self.exits.is_empty() {
            let outer: Vec<usize> = free.iter().copied().filter(|&k| cand[k] && self.faces_exterior(k)).collect();
            if !outer.is_empty() {
                let a = outer[rng.below(outer.len())];
                let da = self.walk_distance(&[a]);
                let b = *outer.iter().max_by(|&&x, &&y| da[x].partial_cmp(&da[y]).unwrap()).unwrap();
                let area = free.len() as f32 * RES * RES;
                self.exits.push(self.center_k(b));
                if area >= 100.0 {
                    let db = self.walk_distance(&[b]);
                    let c = *outer.iter().max_by(|&&x, &&y| db[x].partial_cmp(&db[y]).unwrap()).unwrap();
                    self.exits.push(self.center_k(c));
                }
            }
        } else {
            // 데이터셋 비상구를 가장 가까운 도달 칸으로 맞춘다
            let snapped: Vec<(f32, f32)> = self.exits.iter().filter_map(|&(x, y)| {
                let (i, j) = self.to_cell(x, y)?;
                self.nearest_free_cell(self.idx(i, j)).map(|k| self.center_k(k))
            }).collect();
            self.exits = snapped;
        }

        // 4) 화재 감지기: 방마다 방 중심(천장). 방 정보가 없으면 6 m 격자.
        let mut det = Vec::new();
        for ri in 0..self.rooms.len() {
            let rid = (ri + 1) as u16;
            let cells: Vec<usize> = free.iter().copied().filter(|&k| self.room[k] == rid).collect();
            if cells.is_empty() { continue; }
            let c = centroid(self, &cells);
            let k = *cells.iter().min_by(|&&a, &&b| dist2(self.center_k(a), c).partial_cmp(&dist2(self.center_k(b), c)).unwrap()).unwrap();
            det.push(self.center_k(k));
        }
        if det.is_empty() {
            let step = (6.0 / RES) as usize;
            for j in (step / 2..self.h).step_by(step) {
                for i in (step / 2..self.w).step_by(step) {
                    if let Some(k) = self.nearest_free_cell(self.idx(i, j)) {
                        let p = self.center_k(k);
                        if dist2(p, self.center(i, j)) < 9.0 { det.push(p); }
                    }
                }
            }
        }
        self.detectors = det;
    }

    pub fn center_k(&self, k: usize) -> (f32, f32) { self.center(k % self.w, k / self.w) }
    /// 축 방향 넷 중 하나로 벽을 뚫고 나가면 0.6 m 이상 빈 곳이 없거나 격자 끝에 닿는가.
    fn faces_exterior(&self, k: usize) -> bool {
        let need = (0.6 / RES) as i64;
        let (i, j) = ((k % self.w) as i64, (k / self.w) as i64);
        'dir: for (di, dj) in [(1i64, 0i64), (-1, 0), (0, 1), (0, -1)] {
            let mut solid = 0;
            let mut started = false;
            for step in 1..(need + 8) {
                let Some(m) = self.at(i + di * step, j + dj * step) else { if started { return true; } continue 'dir; };
                if self.cell[m] == FREE {
                    if started { continue 'dir; }  // 벽에 닿기 전 빈 칸 → 이 방향 아님
                    else { continue; }
                }
                started = true;
                solid += 1;
                if solid >= need { return true; }
            }
        }
        false
    }
    /// 빈 칸 위 BFS 로 start 에서 가장 가까운 want 칸.
    fn nearest_where(&self, start: usize, want: &[bool]) -> Option<usize> {
        let mut seen = vec![false; self.w * self.h];
        let mut q = VecDeque::from([start]);
        seen[start] = true;
        while let Some(k) = q.pop_front() {
            if want[k] { return Some(k); }
            for nb in self.nb8(k) { if !seen[nb] && self.cell[nb] == FREE { seen[nb] = true; q.push_back(nb); } }
        }
        None
    }
    /// 아무 칸에서든(벽·바깥 포함) 가장 가까운 FREE 칸.
    fn nearest_free_cell(&self, start: usize) -> Option<usize> {
        let mut seen = vec![false; self.w * self.h];
        let mut q = VecDeque::from([start]);
        seen[start] = true;
        while let Some(k) = q.pop_front() {
            if self.cell[k] == FREE { return Some(k); }
            for nb in self.nb8(k) { if !seen[nb] { seen[nb] = true; q.push_back(nb); } }
        }
        None
    }

    pub fn write(&self, out: &mut impl Write) -> io::Result<()> {
        out.write_all(b"FMAP")?;
        out.write_all(&1u32.to_le_bytes())?;
        out.write_all(&(self.w as u32).to_le_bytes())?;
        out.write_all(&(self.h as u32).to_le_bytes())?;
        out.write_all(&RES.to_le_bytes())?;
        out.write_all(&self.origin.0.to_le_bytes())?;
        out.write_all(&self.origin.1.to_le_bytes())?;
        out.write_all(&self.cell)?;
        for r in &self.room { out.write_all(&r.to_le_bytes())?; }
        out.write_all(&(self.rooms.len() as u32).to_le_bytes())?;
        for r in &self.rooms { out.write_all(&[r.kind as u8])?; out.write_all(&r.area.to_le_bytes())?; }
        for pts in [&self.extinguishers, &self.exits, &self.detectors] {
            out.write_all(&(pts.len() as u32).to_le_bytes())?;
            for &(x, y) in pts.iter() { out.write_all(&x.to_le_bytes())?; out.write_all(&y.to_le_bytes())?; }
        }
        Ok(())
    }

    /// 확인용 그림: 빈 곳 흰색, 벽 검정, 바깥 회색, 소화기 빨강, 비상구 초록, 감지기 파랑. scale 칸당 픽셀.
    pub fn write_png(&self, path: &std::path::Path, scale: usize) -> io::Result<()> {
        let (pw, ph) = (self.w * scale, self.h * scale);
        let mut img = vec![0u8; pw * ph * 3];
        let put = |x: usize, y: usize, c: [u8; 3], img: &mut Vec<u8>| {
            if x < pw && y < ph { let o = ((ph - 1 - y) * pw + x) * 3; img[o..o + 3].copy_from_slice(&c); }
        };
        for j in 0..self.h { for i in 0..self.w {
            let c = match self.cell[self.idx(i, j)] { FREE => [255, 255, 255], WALL => [30, 30, 30], _ => [190, 190, 190] };
            for dy in 0..scale { for dx in 0..scale { put(i * scale + dx, j * scale + dy, c, &mut img); } }
        } }
        let mark = |pts: &[(f32, f32)], c: [u8; 3], r: i64, img: &mut Vec<u8>| {
            for &(x, y) in pts {
                let px = ((x - self.origin.0) / RES * scale as f32) as i64;
                let py = ((y - self.origin.1) / RES * scale as f32) as i64;
                for dy in -r..=r { for dx in -r..=r {
                    if dx * dx + dy * dy <= r * r && px + dx >= 0 && py + dy >= 0 { put((px + dx) as usize, (py + dy) as usize, c, img); }
                } }
            }
        };
        let r = (3 * scale) as i64;
        mark(&self.detectors, [40, 90, 230], r, &mut img);
        mark(&self.exits, [20, 170, 60], r + 1, &mut img);
        mark(&self.extinguishers, [220, 30, 30], r, &mut img);
        let f = std::fs::File::create(path)?;
        let mut enc = png::Encoder::new(io::BufWriter::new(f), pw as u32, ph as u32);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        enc.write_header().and_then(|mut w| w.write_image_data(&img)).map_err(io::Error::other)
    }
}

fn centroid(m: &FMap, cells: &[usize]) -> (f32, f32) {
    let (mut sx, mut sy) = (0.0f64, 0.0f64);
    for &k in cells { let (x, y) = m.center_k(k); sx += x as f64; sy += y as f64; }
    ((sx / cells.len() as f64) as f32, (sy / cells.len() as f64) as f32)
}
fn dist2(a: (f32, f32), b: (f32, f32)) -> f32 { (a.0 - b.0).powi(2) + (a.1 - b.1).powi(2) }

/// 의존성 없는 작은 난수(xorshift64*). 같은 seed → 같은 맵.
pub struct Rng(pub u64);
impl Rng {
    pub fn next_u64(&mut self) -> u64 { self.0 ^= self.0 >> 12; self.0 ^= self.0 << 25; self.0 ^= self.0 >> 27; self.0.wrapping_mul(0x2545F4914F6CDD1D) }
    pub fn unit(&mut self) -> f32 { (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32 }
    pub fn range(&mut self, a: f32, b: f32) -> f32 { a + (b - a) * self.unit() }
    pub fn below(&mut self, n: usize) -> usize { (self.next_u64() % n.max(1) as u64) as usize }
}

/// 맵 하나 마무리: 도달 영역 정리 → 방 넓이 → 시설 배치. 도달 넓이가 min_area 미만이면 None.
pub fn finish(mut m: FMap, seed: u64, min_area: f32) -> Option<FMap> {
    let area = m.keep_reachable();
    if area < min_area { return None; }
    m.recount_rooms();
    m.place_facilities(seed);
    Some(m)
}
