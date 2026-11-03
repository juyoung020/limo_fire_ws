// episodes.jsonl 저장소: 줄 위치(원문은 파일에서 다시 읽음) + 판당 작은 칸(표용). 탐사 표: 맵 × 단계 × 맵 크기 칸
use crate::http::{jstr, num64};
use crate::tail::{FileTail, Got};
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::time::Instant;

/// /api/episodes 한 번에 보내는 줄 상한(잘랐으면 잘랐다고 말한다)
pub const MAX_ROWS: usize = 20_000;

#[derive(Clone, Copy)]
pub struct Ep {
    pub map: u16,
    pub stage: u16,
    pub outcome: u16,
    pub size: u8,
    pub env_steps: f64,
    /// 값 칸: METRICS 차례, NaN = 그 판에 없음
    pub v: [f32; N_MET],
}

/// 칸 값 이름(줄 키) — 비율은 0/1 로 넣어 평균이 곧 비율
pub const METRICS: &[&str] = &["success", "collision", "truncation", "declare_short", "coverage", "coverage_at_declare", "t95", "t95_reached", "t", "ret", "path_len"];
pub const N_MET: usize = 11;

#[derive(Default)]
pub struct Dict {
    pub names: Vec<String>,
    idx: HashMap<String, u16>,
}
impl Dict {
    fn id(&mut self, s: &str) -> u16 {
        if let Some(&i) = self.idx.get(s) {
            return i;
        }
        self.names.push(s.to_string());
        let i = (self.names.len() - 1) as u16;
        self.idx.insert(s.to_string(), i);
        i
    }
}

pub struct Episodes {
    /// episodes.jsonl, episodes.1.jsonl, … (1 GiB 마다 다음 번호)
    pub files: Vec<FileTail>,
    pub lines: Vec<(u16, u64, u32)>,
    pub eps: Vec<Ep>,
    pub maps: Dict,
    pub stages: Dict,
    pub outcomes: Dict,
    pub size_edges: Vec<f64>,
    pub size_names: Vec<String>,
    pub bad: usize,
    /// ret ≠ Σ r 인 줄 수
    pub ret_mismatch: usize,
    pub sig: u64,
    pub last_access: Instant,
    base: PathBuf,
}

fn size_names(edges: &[f64]) -> Vec<String> {
    let mut n = vec![format!("< {} m²", edges[0])];
    for w in edges.windows(2) {
        n.push(format!("{}–{} m²", w[0], w[1]));
    }
    n.push(format!("≥ {} m²", edges.last().unwrap()));
    n
}

fn size_of(edges: &[f64], v: f64) -> u8 {
    if !v.is_finite() {
        return 255;
    }
    edges.iter().position(|&e| v < e).unwrap_or(edges.len()) as u8
}

impl Episodes {
    /// `base` = 줄기의 episodes 파일 이름 앞부분. `edges` = run.json area_bins(맵 넓이 칸 경계, m²)
    pub fn new(base: PathBuf, edges: Vec<f64>) -> Episodes {
        let edges = if edges.is_empty() { vec![50.0, 150.0] } else { edges };
        Episodes {
            files: vec![],
            lines: vec![],
            eps: vec![],
            maps: Dict::default(),
            stages: Dict::default(),
            outcomes: Dict::default(),
            size_names: size_names(&edges),
            size_edges: edges,
            bad: 0,
            ret_mismatch: 0,
            sig: 1,
            last_access: Instant::now(),
            base,
        }
    }
    pub fn total(&self) -> usize {
        self.eps.len()
    }
    pub fn update(&mut self) -> bool {
        self.last_access = Instant::now();
        if self.files.is_empty() {
            self.files.push(FileTail::new(trainfmt::episodes_file(&self.base, 0)));
        }
        while trainfmt::episodes_file(&self.base, self.files.len()).exists() {
            let p = trainfmt::episodes_file(&self.base, self.files.len());
            self.files.push(FileTail::new(p));
        }
        let mut changed = false;
        for fi in 0..self.files.len() {
            loop {
                match self.files[fi].poll() {
                    Got::Same => break,
                    Got::Lines(reset, lines) => {
                        changed = true;
                        if reset {
                            // 파일이 새로 쓰였다(재개 되감기 등): 전부 다시
                            let sig = self.sig + 1;
                            *self = Episodes::new(self.base.clone(), self.size_edges.clone());
                            self.sig = sig;
                            return self.update() || true;
                        }
                        let more = !lines.is_empty();
                        for (off, l) in lines {
                            self.push(fi as u16, off, &l);
                        }
                        if !more {
                            break;
                        }
                    }
                }
            }
        }
        changed
    }
    fn push(&mut self, fi: u16, off: u64, l: &str) {
        let v: Value = match serde_json::from_str(l) {
            Ok(v @ Value::Object(_)) => v,
            _ => {
                self.bad += 1;
                return;
            }
        };
        let s = |k: &str| v.get(k).and_then(|x| x.as_str()).unwrap_or("?").to_string();
        let f = |k: &str| v.get(k).and_then(|x| x.as_f64()).unwrap_or(f64::NAN);
        if let (Some(ret), Some(r)) = (v.get("ret").and_then(|x| x.as_f64()), v.get("r").and_then(|x| x.as_object())) {
            let sum: f64 = r.values().filter_map(|x| x.as_f64()).sum();
            if (sum - ret).abs() > 1e-3 * ret.abs().max(1.0) {
                self.ret_mismatch += 1;
            }
        }
        let oc = s("outcome");
        let is = |o: &str| if oc == "?" { f64::NAN } else if oc == o { 1.0 } else { 0.0 };
        let t95 = f("t95");
        let mut m = [f32::NAN; N_MET];
        let vals = [
            is("success"),
            is("collision"),
            is("truncated"),
            is("declared_short"),
            f("coverage"),
            f("coverage_at_declare"),
            t95,
            if oc == "?" { f64::NAN } else if t95.is_finite() { 1.0 } else { 0.0 },
            f("t"),
            f("ret"),
            f("path_len"),
        ];
        for (k, x) in vals.iter().enumerate() {
            m[k] = *x as f32;
        }
        let ep = Ep {
            map: self.maps.id(&s("map")),
            stage: self.stages.id(&s("stage")),
            outcome: self.outcomes.id(&oc),
            size: size_of(&self.size_edges, f("area_m2")),
            env_steps: f("env_steps"),
            v: m,
        };
        self.eps.push(ep);
        self.lines.push((fi, off, l.len() as u32));
    }

    /// 줄 그대로 `from` 부터 MAX_ROWS 개
    pub fn rows_json(&self, from: usize) -> String {
        let total = self.total();
        let from = from.min(total);
        let end = (from + MAX_ROWS).min(total);
        let mut s = format!("{{\"start\":{},\"total\":{},\"cap\":{},\"truncated\":{},\"sig\":{},\"bad\":{},\"ret_mismatch\":{},\"rows\":[", from, total, MAX_ROWS, end < total, self.sig, self.bad, self.ret_mismatch);
        let mut cur: Option<(u16, fs::File)> = None;
        let mut first = true;
        for i in from..end {
            let (fi, off, len) = self.lines[i];
            if cur.as_ref().map(|c| c.0) != Some(fi) {
                cur = fs::File::open(trainfmt::episodes_file(&self.base, fi as usize)).ok().map(|f| (fi, f));
            }
            let Some((_, f)) = cur.as_mut() else { continue };
            let mut buf = vec![0u8; len as usize];
            if f.seek(SeekFrom::Start(off)).is_err() || f.read_exact(&mut buf).is_err() {
                continue;
            }
            if !first {
                s.push(',');
            }
            first = false;
            s.push_str(&String::from_utf8_lossy(&buf));
        }
        s.push_str("]}");
        s
    }

    fn dim(&self, name: &str, e: &Ep) -> Option<u16> {
        Some(match name {
            "map" => e.map,
            "stage" => e.stage,
            "outcome" => e.outcome,
            "size" => {
                if e.size == 255 {
                    return None;
                }
                e.size as u16
            }
            _ => 0,
        })
    }
    fn labels(&self, name: &str) -> Vec<String> {
        match name {
            "map" => self.maps.names.clone(),
            "stage" => self.stages.names.clone(),
            "outcome" => self.outcomes.names.clone(),
            "size" => self.size_names.clone(),
            _ => vec!["all".into()],
        }
    }

    /// 칸마다 최근 `last` 판(0 = 전체). 거르기: map / stage 이름. `upto` = 이 env_steps 까지(시점 커서)
    pub fn table_json(&self, rows: &str, cols: &str, last: usize, filt: &[(&str, &str)], upto: f64) -> String {
        let rl = self.labels(rows);
        let cl = self.labels(cols);
        let nr = rl.len().max(1);
        let nc = cl.len().max(1);
        let mut fl: Vec<(&str, u16)> = vec![];
        for (k, v) in filt {
            if v.is_empty() {
                continue;
            }
            match self.labels(k).iter().position(|n| n == v) {
                Some(i) => fl.push((k, i as u16)),
                None => return format!("{{\"rows\":[],\"cols\":[],\"cells\":[],\"why\":{}}}", serde_json::to_string(&format!("no episodes with {}={}", k, v)).unwrap()),
            }
        }
        #[derive(Clone, Copy)]
        struct C {
            n: usize,
            k: [usize; N_MET],
            s: [f64; N_MET],
        }
        let mut cells = vec![C { n: 0, k: [0; N_MET], s: [0.0; N_MET] }; nr * nc];
        let cap = if last == 0 { usize::MAX } else { last };
        let mut full = 0usize;
        for e in self.eps.iter().rev() {
            if full == nr * nc {
                break;
            }
            if upto.is_finite() && e.env_steps.is_finite() && e.env_steps > upto {
                continue;
            }
            if fl.iter().any(|(k, id)| self.dim(k, e) != Some(*id)) {
                continue;
            }
            let (Some(r), Some(c)) = (self.dim(rows, e), self.dim(cols, e)) else { continue };
            let cell = &mut cells[r as usize * nc + c as usize];
            if cell.n >= cap {
                continue;
            }
            cell.n += 1;
            if cell.n == cap {
                full += 1;
            }
            for m in 0..N_MET {
                if e.v[m].is_finite() {
                    cell.k[m] += 1;
                    cell.s[m] += e.v[m] as f64;
                }
            }
        }
        let mut s = String::from("{\"rows\":");
        s.push_str(&serde_json::to_string(&rl).unwrap());
        s.push_str(",\"cols\":");
        s.push_str(&serde_json::to_string(&cl).unwrap());
        s.push_str(&format!(",\"total\":{},\"last\":{},\"cells\":[", self.total(), last));
        for r in 0..nr {
            if r > 0 {
                s.push(',');
            }
            s.push('[');
            for c in 0..nc {
                if c > 0 {
                    s.push(',');
                }
                let x = &cells[r * nc + c];
                s.push_str(&format!("{{\"n\":{}", x.n));
                for (m, name) in METRICS.iter().enumerate() {
                    s.push(',');
                    jstr(&mut s, name);
                    s.push(':');
                    num64(&mut s, if x.k[m] > 0 { x.s[m] / x.k[m] as f64 } else { f64::NAN });
                }
                s.push('}');
            }
            s.push(']');
        }
        s.push_str("],\"dims\":{");
        for (i, d) in ["map", "stage", "outcome", "size"].iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&format!("\"{}\":", d));
            s.push_str(&serde_json::to_string(&self.labels(d)).unwrap());
        }
        s.push_str("}}");
        s
    }
}
