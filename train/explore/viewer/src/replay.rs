// .xrp 머리 읽기(목록). 파일 본문은 /api/replay 가 그대로 보내고 화면(replay.js)이 읽는다.
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

#[derive(Default)]
pub struct HeadCache {
    m: HashMap<PathBuf, (u64, u64, Value)>, // mtime ns, len, 머리
}

impl HeadCache {
    pub fn head(&mut self, p: &Path) -> Option<Value> {
        let md = fs::metadata(p).ok()?;
        let mt = md.modified().ok()?.duration_since(std::time::UNIX_EPOCH).ok()?.as_nanos() as u64;
        if let Some((a, b, v)) = self.m.get(p) {
            if *a == mt && *b == md.len() {
                return Some(v.clone());
            }
        }
        let mut f = fs::File::open(p).ok()?;
        let mut b8 = [0u8; 8];
        f.read_exact(&mut b8).ok()?;
        let n = trainfmt::xrp::head_len(&b8)?;
        if n > 16 << 20 {
            return None;
        }
        let mut buf = vec![0u8; 8 + n];
        buf[..8].copy_from_slice(&b8);
        f.read_exact(&mut buf[8..]).ok()?;
        let h = trainfmt::xrp::read_head(&buf)?;
        self.m.insert(p.to_path_buf(), (mt, md.len(), h.clone()));
        Some(h)
    }

    /// 줄기의 replays/ 목록: 머리의 meta(그 판의 episodes 줄) + 파일 크기·시각. 새것 먼저.
    pub fn list(&mut self, dir: &Path) -> Value {
        let mut rows: Vec<(f64, Value)> = vec![];
        let mut bad = vec![];
        if let Ok(rd) = fs::read_dir(dir) {
            for e in rd.flatten() {
                let n = e.file_name().to_string_lossy().to_string();
                if !n.ends_with(trainfmt::REPLAY_EXT) || !trainfmt::safe_name(&n) {
                    continue;
                }
                let Ok(md) = e.metadata() else { continue };
                let mt = md.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_secs_f64()).unwrap_or(0.0);
                match self.head(&e.path()) {
                    Some(h) => rows.push((mt, json!({"file": n, "size": md.len(), "mtime": mt, "meta": h.get("meta"), "n_frames": h.get("n_frames"), "dt": h.get("dt")}))),
                    None => bad.push(n),
                }
            }
        }
        rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let why = if rows.is_empty() { json!(format!("no {} files in {}", trainfmt::REPLAY_EXT, dir.display())) } else { Value::Null };
        json!({"rows": rows.into_iter().map(|r| r.1).collect::<Vec<_>>(), "bad": bad, "why": why})
    }
}
