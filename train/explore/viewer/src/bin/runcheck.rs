// runcheck — 실행 폴더가 trainfmt 규약대로인지 확인한다(쓰는 쪽 runfolder.h 와 읽는 쪽 trainview 가 어긋나지 않게).
//
//   runcheck <실행 폴더> [<실행 폴더> …]
//
// 오류(규약 위반)가 하나라도 있으면 끝 코드 1. 경고(모르는 키 등)는 알리기만 한다.
use serde_json::Value;
use std::fs;
use std::path::Path;
use trainfmt::{keys, xrp, OUTCOMES};

struct Report {
    errors: Vec<String>,
    warns: Vec<String>,
}
impl Report {
    fn err(&mut self, s: String) {
        if self.errors.len() < 50 {
            self.errors.push(s);
        }
    }
    fn warn(&mut self, s: String) {
        if !self.warns.contains(&s) && self.warns.len() < 50 {
            self.warns.push(s);
        }
    }
}

fn read_lines(p: &Path) -> Vec<String> {
    fs::read_to_string(p).map(|t| t.lines().map(|l| l.to_string()).collect()).unwrap_or_default()
}

fn check_run(dir: &Path, rep: &mut Report) -> String {
    let mut summary = vec![];
    // run.json
    match fs::read_to_string(dir.join(trainfmt::RUN_JSON)).ok().and_then(|t| serde_json::from_str::<Value>(&t).ok()) {
        None => rep.err("run.json missing or not JSON".into()),
        Some(m) => {
            if m["schema"].as_i64() != Some(trainfmt::SCHEMA) {
                rep.err(format!("run.json schema {} (want {})", m["schema"], trainfmt::SCHEMA));
            }
            for k in ["kind", "name", "pid", "started", "segments"] {
                if m.get(k).is_none() {
                    rep.err(format!("run.json: no '{}'", k));
                }
            }
            if m.get("pid_start").is_none() {
                rep.warn("run.json: no pid_start (viewer cannot rule out pid reuse)".into());
            }
            let segs = m["segments"].as_array().map(|a| a.len()).unwrap_or(0);
            summary.push(format!("segments {}", segs));
            summary.push(format!("{}", if m.get("ended").is_some() { "ended" } else { "not ended" }));
        }
    }
    // progress.jsonl
    let lines = read_lines(&dir.join(trainfmt::PROGRESS));
    let (mut last_it, mut last_st) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    let mut nkeys = std::collections::BTreeSet::new();
    for (i, l) in lines.iter().enumerate() {
        let Ok(Value::Object(o)) = serde_json::from_str::<Value>(l) else {
            rep.err(format!("progress.jsonl line {}: not a JSON object", i + 1));
            continue;
        };
        for r in keys::REQUIRED {
            if !o.contains_key(*r) {
                rep.err(format!("progress.jsonl line {}: required key {} missing", i + 1, r));
            }
        }
        for (k, v) in &o {
            if !v.is_number() {
                rep.err(format!("progress.jsonl line {}: {} = {} (only numbers; leave unmeasured keys out)", i + 1, k, v));
            }
            if !keys::known(k) {
                rep.warn(format!("progress key '{}' not in trainfmt::keys (typo?)", k));
            }
            nkeys.insert(k.clone());
        }
        let it = o.get("time/iterations").and_then(|x| x.as_f64()).unwrap_or(f64::NAN);
        let st = o.get("time/total_timesteps").and_then(|x| x.as_f64()).unwrap_or(f64::NAN);
        if it <= last_it || st < last_st {
            rep.err(format!("progress.jsonl line {}: iterations/total_timesteps not increasing ({} after {}, {} after {})", i + 1, it, last_it, st, last_st));
        }
        last_it = it;
        last_st = st;
        for (k, v) in &o {
            let x = v.as_f64().unwrap_or(0.0);
            if (k.ends_with("_rate") || k.starts_with("rollout/coverage")) && !(0.0..=1.0).contains(&x) {
                rep.err(format!("progress.jsonl line {}: {} = {} outside [0, 1]", i + 1, k, x));
            }
        }
    }
    summary.push(format!("progress {} rows, {} keys", lines.len(), nkeys.len()));
    // 줄기마다 episodes + replays
    let mut streams = vec!["main".to_string()];
    if let Ok(rd) = fs::read_dir(dir) {
        let mut v: Vec<String> = rd.flatten().filter_map(|e| e.file_name().to_string_lossy().strip_prefix("s_").map(|s| s.to_string())).collect();
        v.sort();
        streams.extend(v);
    }
    for st in &streams {
        let base = trainfmt::episodes_base(dir, st);
        let mut n = 0;
        let mut eps = std::collections::HashSet::new();
        for fi in 0.. {
            let p = trainfmt::episodes_file(&base, fi);
            if !p.exists() {
                break;
            }
            for (i, l) in read_lines(&p).iter().enumerate() {
                n += 1;
                let Ok(v) = serde_json::from_str::<Value>(l) else {
                    rep.err(format!("{}: line {} not JSON", p.display(), i + 1));
                    continue;
                };
                let oc = v["outcome"].as_str().unwrap_or("");
                if !OUTCOMES.iter().any(|(o, _)| *o == oc) {
                    rep.err(format!("{} line {}: outcome '{}' not in trainfmt::OUTCOMES", p.display(), i + 1, oc));
                }
                if let (Some(ret), Some(r)) = (v["ret"].as_f64(), v["r"].as_object()) {
                    let s: f64 = r.values().filter_map(|x| x.as_f64()).sum();
                    if (s - ret).abs() > 1e-3 * ret.abs().max(1.0) {
                        rep.err(format!("{} line {}: ret {} ≠ Σ r {}", p.display(), i + 1, ret, s));
                    }
                }
                if let Some(e) = v["ep"].as_i64() {
                    if !eps.insert(e) {
                        rep.err(format!("{} line {}: episode number {} repeats", p.display(), i + 1, e));
                    }
                }
                for k in ["iter", "env_steps", "ep", "t", "steps", "coverage"] {
                    if !v[k].is_number() {
                        rep.err(format!("{} line {}: '{}' missing", p.display(), i + 1, k));
                    }
                }
            }
        }
        let rd = trainfmt::stream_dir(dir, st).join(trainfmt::REPLAYS_DIR);
        let mut nr = 0;
        let mut bytes = 0u64;
        if let Ok(rdd) = fs::read_dir(&rd) {
            for e in rdd.flatten() {
                let name = e.file_name().to_string_lossy().to_string();
                if name.ends_with(".tmp") {
                    rep.warn(format!("{}: leftover .tmp (writer died mid-write?)", name));
                    continue;
                }
                if !name.ends_with(trainfmt::REPLAY_EXT) {
                    continue;
                }
                nr += 1;
                let b = fs::read(e.path()).unwrap_or_default();
                bytes += b.len() as u64;
                match xrp::Xrp::parse(&b) {
                    Err(why) => rep.err(format!("{}: {}", e.path().display(), why)),
                    Ok(x) => {
                        if x.head["meta"]["replay"].as_str() != Some(name.as_str()) {
                            rep.err(format!("{}: head meta.replay = {} (should be its own file name)", name, x.head["meta"]["replay"]));
                        }
                        if x.n_frames == 0 {
                            rep.err(format!("{}: no frames", name));
                        }
                    }
                }
            }
        }
        summary.push(format!("[{}] episodes {}, replays {} ({:.1} MB)", st, n, nr, bytes as f64 / 1e6));
    }
    // evals
    let mut ne = 0;
    if let Ok(rd) = fs::read_dir(dir.join(trainfmt::EVALS_DIR)) {
        for e in rd.flatten() {
            if !e.file_name().to_string_lossy().ends_with(".json") {
                continue;
            }
            ne += 1;
            match fs::read_to_string(e.path()).ok().and_then(|t| serde_json::from_str::<Value>(&t).ok()) {
                Some(v) if v["rows"].is_array() => {}
                _ => rep.err(format!("{}: not JSON with rows[]", e.path().display())),
            }
        }
    }
    summary.push(format!("evals {}", ne));
    summary.join(" · ")
}

fn main() {
    let dirs: Vec<String> = std::env::args().skip(1).collect();
    if dirs.is_empty() {
        eprintln!("usage: runcheck <run dir> [<run dir> …]");
        std::process::exit(2);
    }
    let mut bad = false;
    for d in dirs {
        let mut rep = Report { errors: vec![], warns: vec![] };
        let s = check_run(Path::new(&d), &mut rep);
        println!("{}: {}", d, s);
        for w in &rep.warns {
            println!("  warn: {}", w);
        }
        for e in &rep.errors {
            println!("  ERROR: {}", e);
        }
        println!("  {}", if rep.errors.is_empty() { "OK" } else { "FAILED" });
        bad |= !rep.errors.is_empty();
    }
    std::process::exit(if bad { 1 } else { 0 });
}
