// 실행 찾기, 줄기, 학습 상태 불(pid 확인), latest, /api/runs 의 한 줄
use crate::tail;
use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[derive(Clone)]
pub struct Root {
    pub label: String,
    pub path: PathBuf,
}

#[derive(Clone, Debug)]
pub struct RunRef {
    pub id: String,
    pub root: usize,
    pub dir: PathBuf,
}

const SKIP: &[&str] = &["replays", "evals", "target", "build", ".git"];

fn is_run(d: &Path) -> bool {
    d.join(trainfmt::RUN_JSON).is_file() || d.join(trainfmt::PROGRESS).is_file()
}

/// 뿌리 밑 깊이 4 까지에서 run.json 또는 progress.jsonl 이 있는 폴더. 실행 폴더 안으로는 내려가지 않는다(줄기·재생·체크포인트 폴더).
pub fn discover(roots: &[Root]) -> Vec<RunRef> {
    let mut out = vec![];
    for (ri, r) in roots.iter().enumerate() {
        let mut stack = vec![(r.path.clone(), 0usize)];
        if is_run(&r.path) {
            // 뿌리가 곧 실행 폴더
            let name = r.path.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
            out.push(RunRef { id: format!("{}/{}", r.label, name), root: ri, dir: r.path.clone() });
            continue;
        }
        while let Some((d, depth)) = stack.pop() {
            if depth > 0 && is_run(&d) {
                let rel = d.strip_prefix(&r.path).unwrap_or(&d).to_string_lossy().to_string();
                out.push(RunRef { id: format!("{}/{}", r.label, rel), root: ri, dir: d.clone() });
                continue;
            }
            if depth >= 4 {
                continue;
            }
            let Ok(rd) = fs::read_dir(&d) else { continue };
            for e in rd.flatten() {
                let n = e.file_name().to_string_lossy().to_string();
                if n.starts_with('.') || n.starts_with("s_") || SKIP.contains(&n.as_str()) {
                    continue;
                }
                if e.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                    stack.push((e.path(), depth + 1));
                }
            }
        }
    }
    out.sort_by(|a, b| a.id.cmp(&b.id));
    out
}

/// 줄기: "main" + 하위 폴더 s_<이름>
pub fn streams(dir: &Path) -> Vec<String> {
    let mut v = vec!["main".to_string()];
    if let Ok(rd) = fs::read_dir(dir) {
        let mut s: Vec<String> = rd
            .flatten()
            .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
            .filter_map(|e| e.file_name().to_string_lossy().strip_prefix("s_").map(|x| x.to_string()))
            .filter(|x| trainfmt::safe_name(x))
            .collect();
        s.sort();
        v.extend(s);
    }
    v
}

pub fn read_meta(dir: &Path) -> Option<Value> {
    serde_json::from_str(&fs::read_to_string(dir.join(trainfmt::RUN_JSON)).ok()?).ok()
}

pub fn now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

pub fn mtime(p: &Path) -> Option<f64> {
    fs::metadata(p).ok()?.modified().ok()?.duration_since(UNIX_EPOCH).ok().map(|d| d.as_secs_f64())
}

/// pid 가 살아 있고 `/proc/<pid>/stat` 시작 시각이 pid_start 와 같을 때만 true(pid 재사용을 거름). pid 가 없으면 None.
pub fn pid_alive(meta: &Value) -> Option<bool> {
    let pid = meta.get("pid")?.as_u64()? as u32;
    let st = trainfmt::pid_start(pid);
    Some(match (st, meta.get("pid_start").and_then(|x| x.as_u64())) {
        (None, _) => false,
        (Some(a), Some(b)) => a == b,
        (Some(_), None) => true,
    })
}

/// 학습 상태 불: training = 프로세스 살아 있음(pid·시작 시각 맞음) + STALL_S 안에 기록, stalled = 살아 있는데 기록 없음,
/// finished = run.json 에 ended(정상 끝), crashed = 프로세스가 없는데 ended 없음, unknown = pid 를 안 적은 실행.
pub const STALL_S: f64 = 60.0;
pub fn status(r: &RunRef) -> Value {
    let m = read_meta(&r.dir).unwrap_or(json!({}));
    let prog = r.dir.join(trainfmt::PROGRESS);
    let t_last = [mtime(&prog), mtime(&r.dir.join("episodes.jsonl")), mtime(&r.dir.join(trainfmt::RUN_JSON))].into_iter().flatten().fold(f64::NAN, f64::max);
    let age = now() - t_last;
    let ended = m.get("ended").is_some();
    let state = if ended {
        "finished"
    } else {
        match pid_alive(&m) {
            Some(true) => {
                if age.is_finite() && age <= STALL_S { "training" } else { "stalled" }
            }
            Some(false) => "crashed",
            None => "unknown",
        }
    };
    let last = tail::last_line(&prog);
    let it = last.as_ref().and_then(|l| l.get("time/iterations")).cloned().unwrap_or(Value::Null);
    json!({"id": r.id, "state": state, "age": if age.is_finite() { json!(age) } else { Value::Null },
           "last_write": if t_last.is_finite() { json!(t_last) } else { Value::Null }, "iter": it, "pid": m.get("pid")})
}

/// 폴더 크기(MB) — 1 분에 한 번만 다시 잰다
pub fn disk_mb(dir: &Path) -> f64 {
    fn walk(d: &Path, depth: usize) -> u64 {
        let Ok(rd) = fs::read_dir(d) else { return 0 };
        let mut s = 0;
        for e in rd.flatten() {
            let Ok(md) = e.metadata() else { continue };
            if md.is_dir() {
                if depth < 3 {
                    s += walk(&e.path(), depth + 1);
                }
            } else {
                s += md.len();
            }
        }
        s
    }
    walk(dir, 0) as f64 / 1e6
}

pub fn count_replays(d: &Path) -> usize {
    fs::read_dir(d).map(|rd| rd.flatten().filter(|e| e.file_name().to_string_lossy().ends_with(trainfmt::REPLAY_EXT)).count()).unwrap_or(0)
}

pub struct DiskCache(pub std::collections::HashMap<PathBuf, (Instant, f64)>);

/// 실행 목록 카드에 쓰는 최근 값
const HEALTH_KEYS: &[&str] = &[
    "rollout/success_rate",
    "rollout/collision_rate",
    "rollout/coverage_mean",
    "rollout/t95_mean",
    "rollout/ep_len_mean",
    "eval/success_rate",
    "eval/coverage_mean",
    "eval/t95_mean",
    "curriculum/stage",
    "time/total_timesteps",
];

pub fn summary(r: &RunRef, roots: &[Root], lines: &mut tail::LineCount, disk: &mut DiskCache, rewound: Option<usize>) -> Value {
    let meta = read_meta(&r.dir);
    let m = meta.clone().unwrap_or(json!({}));
    let prog = r.dir.join(trainfmt::PROGRESS);
    let last = tail::last_line(&prog);
    let base = trainfmt::episodes_base(&r.dir, "main");
    let mut eps = 0;
    for k in 0.. {
        let p = trainfmt::episodes_file(&base, k);
        if !p.exists() {
            break;
        }
        eps += lines.count(&p);
    }
    let t_last = [mtime(&prog), mtime(&r.dir.join("episodes.jsonl")), mtime(&r.dir.join(trainfmt::RUN_JSON))].into_iter().flatten().fold(f64::NAN, f64::max);
    let age = now() - t_last;
    let g = |k: &str| last.as_ref().and_then(|x| x.get(k)).cloned().unwrap_or(Value::Null);
    let first_ts = m.get("segments").and_then(|s| s.as_array()).and_then(|a| a.first()).and_then(|x| x.get("started")).cloned().unwrap_or(Value::Null);
    let dm = {
        let e = disk.0.entry(r.dir.clone()).or_insert((Instant::now() - std::time::Duration::from_secs(3600), 0.0));
        if e.0.elapsed().as_secs() >= 60 {
            *e = (Instant::now(), disk_mb(&r.dir));
        }
        e.1
    };
    json!({
        "id": r.id,
        "name": m.get("name").cloned().unwrap_or(json!(r.dir.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default())),
        "root": roots[r.root].path.display().to_string(),
        "dir": r.dir.display().to_string(),
        "kind": m.get("kind").cloned().unwrap_or(json!("?")),
        "group": m.get("group"),
        "seed": m.get("seed"),
        "synthetic": m.get("synthetic").and_then(|x| x.as_bool()).unwrap_or(false),
        "streams": streams(&r.dir),
        "has_meta": meta.is_some(),
        "age": if age.is_finite() { json!(age) } else { Value::Null },
        "iters": g("time/iterations"),
        "env_steps": g("time/total_timesteps"),
        "wall": g("time/time_elapsed"),
        "eps": eps,
        "replays": count_replays(&r.dir.join(trainfmt::REPLAYS_DIR)),
        "eval_replays": count_replays(&r.dir.join("s_eval").join(trainfmt::REPLAYS_DIR)),
        "evals": fs::read_dir(r.dir.join(trainfmt::EVALS_DIR)).map(|rd| rd.count()).unwrap_or(0),
        "segments": m.get("segments").and_then(|s| s.as_array()).map(|a| a.len()).unwrap_or(0),
        "first_ts": first_ts,
        "rewound": rewound,
        "status": status(r),
        "health": tail::recent_values(&prog, HEALTH_KEYS),
        "disk_mb": (dm * 10.0).round() / 10.0,
    })
}

/// 기본으로 여는 실행: 뿌리마다 latest.txt — 가장 최근에 바뀐 것. 가짜 시험 자료(synthetic)는 진짜가 하나라도 있으면 후보가 아니다.
pub fn latest(roots: &[Root], runs: &[RunRef]) -> Option<String> {
    let syn = |r: &RunRef| read_meta(&r.dir).and_then(|m| m.get("synthetic").and_then(|x| x.as_bool())).unwrap_or(false);
    let real: Vec<RunRef> = runs.iter().filter(|r| !syn(r)).cloned().collect();
    let pool: &[RunRef] = if real.is_empty() { runs } else { &real };
    let mut best: Option<(f64, String)> = None;
    for (ri, r) in roots.iter().enumerate() {
        let p = r.path.join(trainfmt::LATEST);
        let Ok(t) = fs::read_to_string(&p) else { continue };
        let name = t.trim();
        let Some(rr) = pool.iter().find(|x| x.root == ri && x.dir.strip_prefix(&r.path).map(|s| s.to_string_lossy() == name).unwrap_or(false)) else { continue };
        let mt = mtime(&p).unwrap_or(0.0);
        if best.as_ref().map(|b| mt > b.0).unwrap_or(true) {
            best = Some((mt, rr.id.clone()));
        }
    }
    if best.is_none() {
        for rr in pool {
            let mt = mtime(&rr.dir.join(trainfmt::PROGRESS)).or(mtime(&rr.dir.join(trainfmt::RUN_JSON))).unwrap_or(0.0);
            if best.as_ref().map(|b| mt > b.0).unwrap_or(true) {
                best = Some((mt, rr.id.clone()));
            }
        }
    }
    best.map(|b| b.1)
}
