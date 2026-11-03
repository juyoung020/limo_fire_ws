//! 탐사 RL 학습 맵 묶음의 1 단계: 데이터셋 집·절차 생성 방 → 점유 격자(Nav2 map_server 형식 PGM + YAML) + 출발점.
//! 2 단계는 sim 의 map2scene(격자 → .scene, 벽 높이 2.4 m), 3 단계는 scene_list(build_world 닫힘 검사·넓이·벽 칸 수·등급).
//! 명령은 train/explore/scenes/README.md.
//!
//! scene_grids --out <폴더> [--procthor <jsonl.gz> --procthor-limit <N>] [--rooms <등급>:<개수>]... [--seed <S>] [--max-cells 510]
//! 출력: <out>/<이름>.pgm·.yaml, <out>/manifest.tsv (이름, 출처, 칸 w·h, 빈 칸 넓이 m², 출발점 x·y), <out>/skipped.tsv (버린 것·까닭)
//!
//! - PGM: 도달 영역(FMap FREE) = 254, 그 밖(벽·장애물·바깥·닿지 않는 빈 곳) = 0. map2scene 은 < 128 을 막힘으로 읽고
//!   그 경계(칸 가장자리)에 벽면을 세운다 — 바깥이 모두 막힘이라 데이터 벽에 틈이 있어도 격자에서는 새지 않는다
//!   (새는 것은 build_world 가 다시 잡는다).
//! - 격자 = 막힘 아닌 칸의 바운딩 상자 + 둘레 4 칸. 한 변이 --max-cells 를 넘으면 버린다(sim 지도 512 칸 = 25.6 m,
//!   build_world 는 칸 수 + 1 을 쓰므로 여유 둠).
//! - 출발점: 도달 영역 중 벽·바깥까지 0.4 m 이상인 칸에서 무작위(씨앗 = --seed 와 맵 순번). 칸 가운데.
//! - ProcTHOR 줄은 믿지 않는 입력: 줄 길이 상한, JSON 파싱 실패·좌표 이상은 버림(procthor::load).

use fmap::{procthor, rooms, FMap, Rng, FREE, RES};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

// 칸 가운데끼리 거리(wall_distance)라 벽면(칸 가장자리)까지는 반 칸 짧다 — 벽면에서 0.4 m 이상이 되게 반 칸 더함
const START_CLEAR: f32 = 0.4 + 0.5 * RES;
const MARGIN: usize = 4;
const MAX_LINE: usize = 8 << 20;

enum Src { Procthor(String), Rooms(u32) }
struct Job { name: String, src: Src, seed: u64 }
struct Grid { name: String, source: &'static str, w: usize, h: usize, free: f32, start: (f32, f32), origin: (f32, f32), px: Vec<u8>, room: Vec<u16>, furnished: bool, heights: Vec<u8> }

/// 크기 등급 (scene_list 와 같은 문턱, 여기서는 FMap 빈 칸 넓이로 어림 — 최종 등급은 scene_list 가 build_world 바닥 넓이로 다시 매김)
fn grade(free: f32) -> usize { if free < 30.0 { 0 } else if free < 80.0 { 1 } else if free < 150.0 { 2 } else { 3 } }

fn crop(m: &FMap) -> Option<(usize, usize, usize, usize)> {
    let (mut i0, mut j0, mut i1, mut j1) = (usize::MAX, usize::MAX, 0, 0);
    for j in 0..m.h { for i in 0..m.w {
        if m.cell[m.idx(i, j)] == FREE { i0 = i0.min(i); j0 = j0.min(j); i1 = i1.max(i); j1 = j1.max(j); }
    } }
    if i0 == usize::MAX { return None; }
    Some((i0.saturating_sub(MARGIN), j0.saturating_sub(MARGIN), i1 + MARGIN, j1 + MARGIN))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let all = |k: &str| args.iter().enumerate().filter(|(_, a)| *a == k).filter_map(|(i, _)| args.get(i + 1).cloned()).collect::<Vec<_>>();
    let out = PathBuf::from(get("--out").expect("--out <폴더> 필요"));
    let seed: u64 = get("--seed").and_then(|s| s.parse().ok()).unwrap_or(1);
    let max_cells: usize = get("--max-cells").and_then(|s| s.parse().ok()).unwrap_or(510);
    let pt_limit: usize = get("--procthor-limit").and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
    std::fs::create_dir_all(&out).expect("출력 폴더");

    let mut jobs: Vec<Job> = Vec::new();
    if let Some(path) = get("--procthor") {
        let gz = flate2::read::GzDecoder::new(std::fs::File::open(&path).expect("procthor jsonl.gz"));
        let mut rd = BufReader::new(gz.take(4 << 30));
        let mut n = 0usize;
        let mut line = String::new();
        while n < pt_limit {
            line.clear();
            match (&mut rd).take(MAX_LINE as u64).read_line(&mut line) { Ok(0) | Err(_) => break, Ok(_) => {} }
            if !line.ends_with('\n') && line.len() >= MAX_LINE { eprintln!("[scene_grids] procthor 줄 {n} 이 너무 김 — 멈춤"); break; }
            jobs.push(Job { name: format!("procthor_{n:05}"), src: Src::Procthor(std::mem::take(&mut line)), seed: 0 });
            n += 1;
        }
    }
    for spec in all("--rooms") {
        let (g, c) = spec.split_once(':').expect("--rooms <등급>:<개수>");
        let (g, c): (u32, usize) = (g.parse().expect("등급"), c.parse().expect("개수"));
        for k in 0..c { jobs.push(Job { name: format!("rooms_g{g}_{k:04}"), src: Src::Rooms(g), seed: (g as u64) << 32 | k as u64 }); }
    }
    eprintln!("[scene_grids] 작업 {} 개", jobs.len());

    let next = AtomicUsize::new(0);
    let rows: Mutex<Vec<Grid>> = Mutex::new(Vec::new());
    let per_grade: usize = get("--per-grade").and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
    let skipped: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let k = next.fetch_add(1, Ordering::Relaxed);
                let Some(job) = jobs.get(k) else { break };
                let skip = |why: &str| skipped.lock().unwrap().push(format!("{}\t{}", job.name, why));
                let m = match &job.src {
                    Src::Procthor(line) => serde_json::from_str::<serde_json::Value>(line).ok().and_then(|v| procthor::load(&v)),
                    Src::Rooms(g) => Some(rooms::generate(seed ^ job.seed.wrapping_mul(0x9E37_79B9), *g)),
                };
                let Some(mut m) = m else { skip("파싱 실패·좌표 이상"); continue };
                let area = m.keep_reachable();
                if area < 3.0 { skip("도달 영역 < 3 m²"); continue; }
                let Some((i0, j0, i1, j1)) = crop(&m) else { skip("빈 칸 없음"); continue };
                let (w, h) = (i1 - i0 + 1, j1 - j0 + 1);
                if w > max_cells || h > max_cells { skip(&format!("격자 {w}×{h} 칸 > {max_cells} (25.6 m 상한)")); continue; }
                let at = |i: usize, j: usize| -> u8 {
                    if i < m.w && j < m.h && m.cell[m.idx(i, j)] == FREE { 254 } else { 0 }
                };
                // 출발점 후보
                let wd = m.wall_distance();
                let cand: Vec<usize> = (0..m.w * m.h).filter(|&q| m.cell[q] == FREE && wd[q] >= START_CLEAR).collect();
                if cand.is_empty() { skip("벽에서 0.4 m 떨어진 칸 없음"); continue; }
                let mut rng = Rng((seed ^ (k as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03)) | 1);
                let st = cand[rng.below(cand.len())];
                let (sx, sy) = m.center_k(st);
                let (ox, oy) = (m.origin.0 + i0 as f32 * RES, m.origin.1 + j0 as f32 * RES);
                // PGM: 위 행 = y 가 큰 쪽 (map_server 규약)
                let mut px = Vec::with_capacity(w * h);
                for r in 0..h { let j = j1 - r; for i in i0..=i1 { px.push(at(i, j)); } }
                // 방 번호 (y 위 차례, 가구 밀도용)
                let mut room = Vec::with_capacity(w * h);
                for j in j0..=j1 { for i in i0..=i1 { room.push(if i < m.w && j < m.h { m.room[m.idx(i, j)] } else { 0 }); } }
                let free = px.iter().filter(|&&v| v == 254).count() as f32 * RES * RES;
                rows.lock().unwrap().push(Grid { name: job.name.clone(), source: match job.src { Src::Procthor(_) => "procthor", Src::Rooms(_) => "rooms" },
                    w, h, free, start: (sx, sy), origin: (ox, oy), px, room, furnished: false, heights: Vec::new() });
            });
        }
    });
    let mut rows = rows.into_inner().unwrap();
    rows.sort_by(|a, b| a.name.cmp(&b.name));
    let pool = rows.len();
    // 등급마다 per_grade 개: 절차 생성 방(따로 개수를 주문한 것)을 먼저 다 넣고, 나머지를 ProcTHOR 에서 씨앗 섞은 차례로 채운다
    let key = |g: &Grid| { let mut r = Rng(seed ^ g.name.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3)) | 1); r.next_u64() };
    let mut chosen: Vec<Grid> = Vec::new();
    let mut buckets: Vec<Vec<Grid>> = (0..4).map(|_| Vec::new()).collect();
    for g in rows.drain(..) { let gr = grade(g.free); buckets[gr].push(g); }
    for mut b in buckets {
        b.sort_by_key(|g| (g.source != "rooms", key(g), g.name.clone()));
        chosen.extend(b.into_iter().take(per_grade));
    }
    let mut rows = chosen;
    // 가구 판: 고른 맵마다 같은 출발점에 가구를 놓은 "<이름>_f" 를 더한다 (fmap::furnish, 씨앗 = --seed 와 이름)
    let mut furn_total = fmap::furnish::Count::default();
    if args.iter().any(|a| a == "--furnish") {
        let made: Mutex<Vec<(Grid, fmap::furnish::Count)>> = Mutex::new(Vec::new());
        let next = AtomicUsize::new(0);
        std::thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(|| loop {
                    let k = next.fetch_add(1, Ordering::Relaxed);
                    let Some(b) = rows.get(k) else { break };
                    let (w, h) = (b.w, b.h);
                    let mut occ: Vec<bool> = (0..w * h).map(|q| b.px[(h - 1 - q / w) * w + q % w] == 0).collect();
                    let mut hgt = vec![0u8; w * h];
                    let st = (((b.start.0 - b.origin.0) / RES).floor() as usize, ((b.start.1 - b.origin.1) / RES).floor() as usize);
                    let c = fmap::furnish::furnish(&mut fmap::furnish::Grid { w, h, occ: &mut occ, height: &mut hgt, room: &b.room, start: st }, seed ^ key(b));
                    let mut px = vec![0u8; w * h];
                    let mut heights = vec![0u8; w * h];
                    for q in 0..w * h { let t = (h - 1 - q / w) * w + q % w; px[t] = if occ[q] { 0 } else { 254 }; heights[t] = hgt[q]; }
                    let free = px.iter().filter(|&&v| v == 254).count() as f32 * RES * RES;
                    made.lock().unwrap().push((Grid { name: format!("{}_f", b.name), source: b.source, w, h, free, start: b.start, origin: b.origin,
                        px, room: Vec::new(), furnished: true, heights }, c));
                });
            }
        });
        for (g, c) in made.into_inner().unwrap() {
            furn_total.wall_box += c.wall_box; furn_total.table += c.table; furn_total.chair += c.chair; furn_total.small += c.small; furn_total.rejected += c.rejected;
            rows.push(g);
        }
        eprintln!("[scene_grids] 가구: 벽 상자 {}, 탁자 {}, 의자 {}, 작은 물체 {}, 고립시켜 뺀 것 {}", furn_total.wall_box, furn_total.table, furn_total.chair, furn_total.small, furn_total.rejected);
    }
    rows.sort_by(|a, b| a.name.cmp(&b.name));
    for g in &rows {
        let base = out.join(&g.name);
        let mut f = std::fs::File::create(base.with_extension("pgm")).unwrap();
        write!(f, "P5\n{} {}\n255\n", g.w, g.h).unwrap();
        f.write_all(&g.px).unwrap();
        if g.furnished {
            let mut f = std::fs::File::create(out.join(format!("{}.h.pgm", g.name))).unwrap();
            write!(f, "P5\n{} {}\n255\n", g.w, g.h).unwrap();
            f.write_all(&g.heights).unwrap();
        }
        std::fs::write(base.with_extension("yaml"), format!(
            "image: {}.pgm\nresolution: {RES}\norigin: [{:.4}, {:.4}, 0.0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n", g.name, g.origin.0, g.origin.1)).unwrap();
    }
    let mut skipped = skipped.into_inner().unwrap();
    skipped.sort();
    let mut f = std::fs::File::create(out.join("manifest.tsv")).unwrap();
    writeln!(f, "name\tsource\tw\th\tfree_m2\tstart_x\tstart_y\tfurnished\theights").unwrap();
    for g in &rows { writeln!(f, "{}\t{}\t{}\t{}\t{:.2}\t{:.4}\t{:.4}\t{}\t{}", g.name, g.source, g.w, g.h, g.free, g.start.0, g.start.1,
        g.furnished as u8, if g.furnished { format!("{}.h.pgm", g.name) } else { "-".into() }).unwrap(); }
    let mut f = std::fs::File::create(out.join("skipped.tsv")).unwrap();
    writeln!(f, "name\treason").unwrap();
    for r in &skipped { writeln!(f, "{r}").unwrap(); }
    eprintln!("[scene_grids] 격자 {pool} 개 중 {} 개 고름 (등급당 최대 {per_grade}), 버림 {} 개 → {}", rows.len(), skipped.len(), out.display());
}
