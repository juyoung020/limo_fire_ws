// trainview — 탐사 RL 학습 뷰어 서버. 읽기 전용: 실행 폴더의 파일만 읽고 아무것도 쓰지 않는다.
// std TcpListener + 연결마다 스레드, 요청 줄을 직접 읽음, GET 만, Connection: close, SSE 는 /api/live.
//
//   trainview --root train/explore/runs [--root DIR …] [--port 8095] [--bind 127.0.0.1]
mod http;
mod live;
mod replay;
mod runs;
mod table;
mod tail;

use http::Req;
use runs::{Root, RunRef};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

include!(concat!(env!("OUT_DIR"), "/assets.rs"));

pub struct App {
    pub roots: Vec<Root>,
    runs: Mutex<(Instant, Arc<Vec<RunRef>>)>,
    prog: Mutex<HashMap<String, Arc<Mutex<tail::Progress>>>>,
    eps: Mutex<HashMap<String, Arc<Mutex<table::Episodes>>>>,
    pub heads: Mutex<replay::HeadCache>,
    lines: Mutex<tail::LineCount>,
    disk: Mutex<runs::DiskCache>,
}

impl App {
    /// 실행 목록(2 초 캐시)
    pub fn runs(&self) -> Arc<Vec<RunRef>> {
        let mut g = self.runs.lock().unwrap();
        if g.0.elapsed() > Duration::from_secs(2) || g.1.is_empty() {
            *g = (Instant::now(), Arc::new(runs::discover(&self.roots)));
        }
        g.1.clone()
    }
    pub fn find(&self, id: &str) -> Option<RunRef> {
        if let Some(r) = self.runs().iter().find(|r| r.id == id) {
            return Some(r.clone());
        }
        // 막 생긴 실행: 캐시를 버리고 한 번 더
        self.runs.lock().unwrap().0 = Instant::now() - Duration::from_secs(60);
        self.runs().iter().find(|r| r.id == id).cloned()
    }
    pub fn progress(&self, r: &RunRef) -> Arc<Mutex<tail::Progress>> {
        let a = self.prog.lock().unwrap().entry(r.id.clone()).or_insert_with(|| Arc::new(Mutex::new(tail::Progress::new(r.dir.join(trainfmt::PROGRESS))))).clone();
        a.lock().unwrap().update();
        a
    }
    pub fn episodes(&self, r: &RunRef, stream: &str) -> Arc<Mutex<table::Episodes>> {
        let key = format!("{}|{}", r.id, stream);
        let a = self
            .eps
            .lock()
            .unwrap()
            .entry(key)
            .or_insert_with(|| {
                let edges: Vec<f64> = runs::read_meta(&r.dir)
                    .and_then(|m| m.get("area_bins").and_then(|x| x.as_array()).map(|a| a.iter().filter_map(|v| v.as_f64()).collect()))
                    .unwrap_or_default();
                Arc::new(Mutex::new(table::Episodes::new(trainfmt::episodes_base(&r.dir, stream), edges)))
            })
            .clone();
        a.lock().unwrap().update();
        a
    }
    pub fn rewound_of(&self, id: &str) -> Option<usize> {
        self.prog.lock().unwrap().get(id).map(|p| p.lock().unwrap().rewound)
    }
    pub fn runs_json(&self) -> String {
        let rs = self.runs();
        let mut lines = self.lines.lock().unwrap();
        let mut disk = self.disk.lock().unwrap();
        let rows: Vec<Value> = rs.iter().map(|r| runs::summary(r, &self.roots, &mut lines, &mut disk, self.rewound_of(&r.id))).collect();
        json!({
            "runs": rows,
            "latest": runs::latest(&self.roots, &rs),
            "roots": self.roots.iter().map(|r| json!({"label": r.label, "path": r.path.display().to_string()})).collect::<Vec<_>>(),
            "now": runs::now(),
            "stall_s": runs::STALL_S,
        })
        .to_string()
    }
    /// 10 분 안 본 실행의 저장소를 내린다
    fn evict(&self) {
        let old = |t: Instant| t.elapsed() > Duration::from_secs(600);
        self.prog.lock().unwrap().retain(|_, p| p.try_lock().map(|g| !old(g.last_access)).unwrap_or(true));
        self.eps.lock().unwrap().retain(|_, p| p.try_lock().map(|g| !old(g.last_access)).unwrap_or(true));
    }
}

fn asset(name: &str) -> Option<&'static [u8]> {
    FILES.iter().find(|(n, _)| *n == name).map(|(_, b)| *b)
}

fn run_of(app: &App, s: &mut TcpStream, req: &Req) -> Option<RunRef> {
    let id = req.get("run");
    match app.find(id) {
        Some(r) => Some(r),
        None => {
            http::not_found(s, &format!("no run '{}' under the roots", id));
            None
        }
    }
}

fn stream_of(s: &mut TcpStream, req: &Req) -> Option<String> {
    let st = req.get("stream");
    let st = if st.is_empty() { "main" } else { st };
    if !trainfmt::safe_name(st) {
        http::bad(s, "stream name must match [A-Za-z0-9_.=-]+");
        return None;
    }
    Some(st.to_string())
}

fn handle(mut s: TcpStream, app: Arc<App>) {
    let Some(req) = http::read_request(&s) else { return };
    let gz = req.gzip;
    let p = req.path.as_str();
    match p {
        "/" | "/index.html" => http::respond(&mut s, 200, "text/html; charset=utf-8", asset("index.html").unwrap_or(b"no index"), gz),
        "/api/runs" => http::json(&mut s, &app.runs_json(), gz),
        "/api/meta" => {
            let Some(r) = run_of(&app, &mut s, &req) else { return };
            match std::fs::read_to_string(r.dir.join(trainfmt::RUN_JSON)) {
                Ok(t) => http::json(&mut s, &t, gz),
                Err(_) => http::json(&mut s, &json!({"why": format!("no run.json in {}", r.dir.display())}).to_string(), gz),
            }
        }
        "/api/progress" => {
            let Some(r) = run_of(&app, &mut s, &req) else { return };
            let pa = app.progress(&r);
            let pg = pa.lock().unwrap();
            if pg.total() == 0 {
                let why = if r.dir.join(trainfmt::PROGRESS).exists() { "progress.jsonl has no complete lines yet" } else { "no progress.jsonl in this run" };
                http::json(&mut s, &json!({"start": 0, "total": 0, "sig": pg.sig, "x": {}, "cols": {}, "keys": [], "why": why}).to_string(), gz);
                return;
            }
            let keys = req.get("keys");
            if keys.is_empty() {
                http::json(&mut s, &pg.keys_json(), gz);
                return;
            }
            let ks: Vec<String> = if keys == "*" { pg.keys.clone() } else { keys.split(',').filter(|k| !k.is_empty()).map(|k| k.to_string()).collect() };
            let mut from = req.num("from").unwrap_or(0.0) as usize;
            if let Some(sig) = req.num("sig") {
                if sig as u64 != pg.sig {
                    from = 0; // 파일이 바뀌었다: 처음부터
                }
            }
            let mp = req.num("max_points").unwrap_or(0.0) as usize;
            http::json(&mut s, &pg.rows_json(Some(&ks), from, mp), gz);
        }
        "/api/episodes" => {
            let Some(r) = run_of(&app, &mut s, &req) else { return };
            let Some(st) = stream_of(&mut s, &req) else { return };
            let ea = app.episodes(&r, &st);
            let e = ea.lock().unwrap();
            if e.total() == 0 {
                let why = format!("no episode lines in {}.jsonl", trainfmt::episodes_base(&r.dir, &st).display());
                http::json(&mut s, &json!({"start": 0, "total": 0, "rows": [], "why": why}).to_string(), gz);
                return;
            }
            http::json(&mut s, &e.rows_json(req.num("from").unwrap_or(0.0) as usize), gz);
        }
        "/api/table" => {
            let Some(r) = run_of(&app, &mut s, &req) else { return };
            let Some(st) = stream_of(&mut s, &req) else { return };
            let ea = app.episodes(&r, &st);
            let e = ea.lock().unwrap();
            if e.total() == 0 {
                http::json(&mut s, &json!({"rows": [], "cols": [], "cells": [], "why": "no episodes.jsonl (sample) in this split"}).to_string(), gz);
                return;
            }
            let rows = if req.get("rows").is_empty() { "map" } else { req.get("rows") };
            let cols = if req.get("cols").is_empty() { "size" } else { req.get("cols") };
            let filt = [("map", req.get("map")), ("stage", req.get("stage"))];
            let upto = req.num("upto").unwrap_or(f64::NAN);
            http::json(&mut s, &e.table_json(rows, cols, req.num("last").unwrap_or(0.0) as usize, &filt, upto), gz);
        }
        "/api/replays" => {
            let Some(r) = run_of(&app, &mut s, &req) else { return };
            let Some(st) = stream_of(&mut s, &req) else { return };
            let d = trainfmt::stream_dir(&r.dir, &st).join(trainfmt::REPLAYS_DIR);
            let v = app.heads.lock().unwrap().list(&d);
            http::json(&mut s, &v.to_string(), gz);
        }
        "/api/replay" => {
            let Some(r) = run_of(&app, &mut s, &req) else { return };
            let Some(st) = stream_of(&mut s, &req) else { return };
            let id = req.get("id");
            if !trainfmt::safe_name(id) || !id.ends_with(trainfmt::REPLAY_EXT) {
                return http::bad(&mut s, "id must be a .xrp file name");
            }
            match std::fs::read(trainfmt::stream_dir(&r.dir, &st).join(trainfmt::REPLAYS_DIR).join(id)) {
                Ok(b) => http::respond(&mut s, 200, "application/octet-stream", &b, gz),
                Err(_) => http::not_found(&mut s, &format!("no replay {}", id)),
            }
        }
        "/api/evals" => {
            let Some(r) = run_of(&app, &mut s, &req) else { return };
            let mut rows: Vec<(String, Value)> = vec![];
            if let Ok(rd) = std::fs::read_dir(r.dir.join(trainfmt::EVALS_DIR)) {
                for e in rd.flatten() {
                    let n = e.file_name().to_string_lossy().to_string();
                    if !n.ends_with(".json") {
                        continue;
                    }
                    if let Some(v) = std::fs::read_to_string(e.path()).ok().and_then(|t| serde_json::from_str::<Value>(&t).ok()) {
                        rows.push((n, v));
                    }
                }
            }
            rows.sort_by(|a, b| a.0.cmp(&b.0));
            let why = if rows.is_empty() { json!("no evals/*.json in this run") } else { Value::Null };
            http::json(&mut s, &json!({"evals": rows.into_iter().map(|(n, v)| json!({"file": n, "eval": v})).collect::<Vec<_>>(), "why": why}).to_string(), gz);
        }
        "/api/events" => {
            // 러너의 사람이 읽는 사건 기록(events.txt): 커리큘럼 단계·체크포인트·평가 — 끝 200 줄
            let Some(r) = run_of(&app, &mut s, &req) else { return };
            let t = std::fs::read_to_string(r.dir.join(trainfmt::EVENTS)).unwrap_or_default();
            let ls: Vec<&str> = t.lines().collect();
            let k = ls.len().saturating_sub(200);
            http::json(&mut s, &json!({"lines": &ls[k..], "total": ls.len()}).to_string(), gz);
        }
        "/api/live" => live::serve(s, app, &req),
        _ => {
            let name = p.trim_start_matches('/');
            match asset(name) {
                Some(b) => http::respond(&mut s, 200, http::content_type(name), b, gz),
                None => http::not_found(&mut s, "no such path"),
            }
        }
    }
}

const USAGE: &str = "usage: trainview --root DIR [--root DIR …] [--port 8095] [--bind 127.0.0.1]";

fn main() {
    let mut roots: Vec<PathBuf> = vec![];
    let (mut port, mut bind) = (8095u16, "127.0.0.1".to_string());
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--root" => {
                if let Some(r) = it.next() {
                    roots.push(PathBuf::from(r));
                }
            }
            "--port" => port = it.next().and_then(|v| v.parse().ok()).unwrap_or(port),
            "--bind" => bind = it.next().unwrap_or(bind),
            "-h" | "--help" => {
                eprintln!("{}", USAGE);
                return;
            }
            _ => roots.push(PathBuf::from(a)),
        }
    }
    if roots.is_empty() {
        // 저장소 기본 자리(train/explore/runs) — 빌드한 곳 기준
        let p = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../runs"));
        if p.is_dir() {
            roots.push(p);
        }
    }
    if roots.is_empty() {
        eprintln!("{}\n(no --root and train/explore/runs does not exist)", USAGE);
        std::process::exit(2);
    }
    // 뿌리 이름표 = 폴더 이름(겹치면 ~2, ~3)
    let mut rs: Vec<Root> = vec![];
    for p in roots {
        if !p.is_dir() {
            eprintln!("trainview: --root {} is not a directory", p.display());
            std::process::exit(2);
        }
        let p = std::fs::canonicalize(&p).unwrap_or(p);
        let base = p.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or("root".into());
        let mut label = base.clone();
        let mut k = 2;
        while rs.iter().any(|r| r.label == label) {
            label = format!("{}~{}", base, k);
            k += 1;
        }
        rs.push(Root { label, path: p });
    }
    let app = Arc::new(App {
        roots: rs,
        runs: Mutex::new((Instant::now(), Arc::new(vec![]))),
        prog: Mutex::new(HashMap::new()),
        eps: Mutex::new(HashMap::new()),
        heads: Mutex::new(replay::HeadCache::default()),
        lines: Mutex::new(tail::LineCount::default()),
        disk: Mutex::new(runs::DiskCache(HashMap::new())),
    });
    let l = TcpListener::bind((bind.as_str(), port)).unwrap_or_else(|e| {
        eprintln!("trainview: cannot listen on {}:{}: {} (port in use? try --port)", bind, port, e);
        std::process::exit(1);
    });
    let n = app.runs().len();
    eprintln!("trainview: http://{}:{}  ({} runs under {})", if bind == "0.0.0.0" { "localhost" } else { &bind }, port, n, app.roots.iter().map(|r| r.path.display().to_string()).collect::<Vec<_>>().join(", "));
    {
        let a = app.clone();
        std::thread::spawn(move || loop {
            std::thread::sleep(Duration::from_secs(60));
            a.evict();
        });
    }
    for c in l.incoming().flatten() {
        let _ = c.set_nodelay(true);
        let app = app.clone();
        std::thread::spawn(move || handle(c, app));
    }
}
