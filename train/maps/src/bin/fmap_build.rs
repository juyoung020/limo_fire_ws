//! 데이터셋 → .fmap 일괄 변환.
//!
//! fmap_build --out <폴더> [--houseexpo <json 폴더>] [--procthor <jsonl.gz>]... [--duct <개수>]
//!            [--limit <소스당 최대>] [--png-every <N>] [--seed <S>]
//! 출력: <out>/<소스>/<id>.fmap, N 개마다 <id>.png, <out>/index.tsv

use fmap::{duct, finish, houseexpo, procthor, FMap};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

struct Job { source: &'static str, id: String, make: Box<dyn Fn() -> Option<FMap> + Send + Sync> }

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let all = |k: &str| args.iter().enumerate().filter(|(_, a)| *a == k).filter_map(|(i, _)| args.get(i + 1).cloned()).collect::<Vec<_>>();
    let out = PathBuf::from(get("--out").expect("--out <폴더> 필요"));
    let limit: usize = get("--limit").and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
    let png_every: usize = get("--png-every").and_then(|s| s.parse().ok()).unwrap_or(500);
    let seed: u64 = get("--seed").and_then(|s| s.parse().ok()).unwrap_or(1);

    let mut jobs: Vec<Job> = Vec::new();
    if let Some(dir) = get("--houseexpo") {
        let mut files: Vec<PathBuf> = std::fs::read_dir(&dir).expect("houseexpo 폴더").filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().map_or(false, |e| e == "json")).collect();
        files.sort();
        for f in files.into_iter().take(limit) {
            let id = f.file_stem().unwrap().to_string_lossy().into_owned();
            jobs.push(Job { source: "houseexpo", id, make: Box::new(move || {
                let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&f).ok()?).ok()?;
                houseexpo::load(&v)
            }) });
        }
    }
    for path in all("--procthor") {
        let split = Path::new(&path).file_name().unwrap().to_string_lossy().split('.').next().unwrap().to_string();
        let rd = BufReader::new(flate2::read::GzDecoder::new(std::fs::File::open(&path).expect("procthor jsonl.gz")));
        for (n, line) in rd.lines().map_while(Result::ok).take(limit).enumerate() {
            jobs.push(Job { source: "procthor", id: format!("{split}_{n:05}"), make: Box::new(move || {
                let v: serde_json::Value = serde_json::from_str(&line).ok()?;
                procthor::load(&v)
            }) });
        }
    }
    if let Some(n) = get("--duct").and_then(|s| s.parse::<usize>().ok()) {
        for k in 0..n.min(limit) {
            jobs.push(Job { source: "duct", id: format!("duct_{k:05}"), make: Box::new(move || Some(duct::generate(k as u64 + 1))) });
        }
    }
    eprintln!("[fmap] 작업 {} 개", jobs.len());
    for s in ["houseexpo", "procthor", "duct"] { std::fs::create_dir_all(out.join(s)).unwrap(); }

    let next = AtomicUsize::new(0);
    let rows: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let skipped = AtomicUsize::new(0);
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let k = next.fetch_add(1, Ordering::Relaxed);
                let Some(job) = jobs.get(k) else { break };
                let min_area = if job.source == "duct" { 10.0 } else { 6.0 };
                let Some(m) = (job.make)().and_then(|m| finish(m, seed ^ (k as u64 * 0x9E37), min_area)) else {
                    skipped.fetch_add(1, Ordering::Relaxed); continue
                };
                let base = out.join(job.source).join(&job.id);
                let mut f = BufWriter::new(std::fs::File::create(base.with_extension("fmap")).unwrap());
                m.write(&mut f).unwrap();
                if k % png_every == 0 { let _ = m.write_png(&base.with_extension("png"), 2); }
                let free = m.cell.iter().filter(|&&c| c == fmap::FREE).count() as f32 * fmap::RES * fmap::RES;
                rows.lock().unwrap().push(format!("{}\t{}\t{}\t{}\t{:.1}\t{}\t{}\t{}\t{}", job.source, job.id, m.w, m.h, free,
                    m.rooms.len(), m.extinguishers.len(), m.exits.len(), m.detectors.len()));
                if k % 2000 == 0 { eprintln!("[fmap] {k}/{}", jobs.len()); }
            });
        }
    });
    let mut rows = rows.into_inner().unwrap();
    rows.sort();
    let mut idx = BufWriter::new(std::fs::File::create(out.join("index.tsv")).unwrap());
    writeln!(idx, "source\tid\tw\th\tfree_m2\trooms\textinguishers\texits\tdetectors").unwrap();
    for r in &rows { writeln!(idx, "{r}").unwrap(); }
    eprintln!("[fmap] 완료 {} 개, 건너뜀 {} 개 → {}", rows.len(), skipped.load(Ordering::Relaxed), out.display());
}
