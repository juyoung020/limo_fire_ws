//! SAM 3 자동 라벨(teacher/sam3_label 의 JSONL) → YOLO 탐지 데이터셋.
//!
//! 규칙(config.json):
//! 1. 클래스별 점수 문턱(score_thr)을 넘는 검출만.
//! 2. 같은 클래스는 프롬프트가 달라도 한데 모아 NMS(nms_iou) — 동의어 프롬프트의 겹침 제거.
//! 3. 사람 상자 넓이의 person_in_exit_sign 이상이 비상구 표지 상자 안이면 버린다(표지 속 달리는 사람 그림).
//! 4. 변이 min_box_px 보다 짧은 상자는 버린다.
//!
//! 출력: <out>/images/<split>/<이름> (원본으로 심볼릭 링크), <out>/labels/<split>/<이름>.txt ("클래스 cx cy w h", 0..1),
//!       <out>/data.yaml, <out>/stats.json. 이름 = 원본 경로 sha1 앞 16 자 + 확장자(출처가 달라도 겹치지 않게).
//!       라벨에 마스크("m")가 있으면 <out>/masks/<split>/<이름>.msk 도 쓴다: 사진 전체(가로세로 따로 늘인)를 덮는
//!       288×288 번호 지도 — 0 배경, j+1 = 라벨 파일 j 번째 줄. 겹치면 작은 마스크가 위(Ultralytics overlap_mask 와
//!       같은 규칙). 형식: "YMSK" | u16 크기 | u16 물체 수 | u32 n | u32 런[n] (길이 << 8 | 번호), 행 우선.
//!
//! 사용: yolods <labels.jsonl> <lists 폴더(train/val/test.txt)> <out> [config.json]
use anyhow::{bail, Context, Result};
use serde::Deserialize;
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

#[derive(Deserialize)]
struct Config {
    classes: Vec<String>,
    score_thr: Vec<f32>,
    nms_iou: f32,
    person_in_exit_sign: f32,
    min_box_px: f32,
}

#[derive(Clone, Debug)]
struct Det {
    c: usize,
    s: f32,
    b: [f32; 4],
    m: Option<Vec<u32>>, // 288×288 이진 마스크 RLE(0 부터 길이 교대)
}

/// 번호 지도 한 칸에 담을 수 있는 물체 수(u8)
const MAX_INST: usize = 255;

/// 라벨 차례대로의 마스크 RLE → 번호 지도(작은 마스크가 위) → 런 길이 부호
fn index_map(dets: &[Det], size: usize) -> Vec<u32> {
    let n = size * size;
    let mut bins: Vec<(usize, Vec<bool>)> = Vec::new();
    for (j, d) in dets.iter().enumerate().take(MAX_INST) {
        let Some(r) = &d.m else { continue };
        let mut m = vec![false; n];
        let (mut p, mut on) = (0usize, false);
        for &len in r {
            let e = (p + len as usize).min(n);
            if on {
                m[p..e].iter_mut().for_each(|x| *x = true);
            }
            p = e;
            on = !on;
        }
        bins.push((j, m));
    }
    // 넓이 큰 것부터 칠해 작은 것이 덮는다(같으면 라벨 차례)
    let area = |m: &Vec<bool>| m.iter().filter(|&&x| x).count();
    bins.sort_by_key(|(j, m)| (std::cmp::Reverse(area(m)), *j));
    let mut idx = vec![0u8; n];
    for (j, m) in &bins {
        for (o, &v) in idx.iter_mut().zip(m.iter()) {
            if v {
                *o = (*j + 1) as u8;
            }
        }
    }
    let mut runs = Vec::new();
    let mut i = 0;
    while i < n {
        let v = idx[i];
        let mut k = i + 1;
        while k < n && idx[k] == v {
            k += 1;
        }
        runs.push(((k - i) as u32) << 8 | v as u32);
        i = k;
    }
    runs
}

fn area(b: &[f32; 4]) -> f32 {
    (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0)
}
fn inter(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let w = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let h = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    w * h
}
fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let i = inter(a, b);
    let u = area(a) + area(b) - i;
    if u > 0.0 { i / u } else { 0.0 }
}

/// 규칙을 적용해 남길 검출과 버린 이유별 개수
fn select(dets: Vec<Det>, cfg: &Config, w: f32, h: f32, drop: &mut BTreeMap<&'static str, usize>) -> Vec<Det> {
    let exit_id = cfg.classes.iter().position(|c| c == "exit_sign");
    let person_id = cfg.classes.iter().position(|c| c == "person");
    // 1, 4: 문턱·크기, 그림 밖은 자른다
    let mut v: Vec<Det> = Vec::new();
    for mut d in dets {
        if d.s <= cfg.score_thr[d.c] {
            *drop.entry("문턱").or_default() += 1;
            continue;
        }
        d.b = [d.b[0].clamp(0.0, w), d.b[1].clamp(0.0, h), d.b[2].clamp(0.0, w), d.b[3].clamp(0.0, h)];
        if d.b[2] - d.b[0] < cfg.min_box_px || d.b[3] - d.b[1] < cfg.min_box_px {
            *drop.entry("작음").or_default() += 1;
            continue;
        }
        v.push(d);
    }
    // 2: 클래스별 NMS
    v.sort_by(|a, b| b.s.partial_cmp(&a.s).unwrap());
    let mut keep: Vec<Det> = Vec::new();
    for d in v {
        if keep.iter().any(|k| k.c == d.c && iou(&k.b, &d.b) >= cfg.nms_iou) {
            *drop.entry("겹침").or_default() += 1;
            continue;
        }
        keep.push(d);
    }
    // 3: 비상구 표지 안의 사람 그림
    if let (Some(e), Some(p)) = (exit_id, person_id) {
        let signs: Vec<[f32; 4]> = keep.iter().filter(|d| d.c == e).map(|d| d.b).collect();
        keep.retain(|d| {
            if d.c != p {
                return true;
            }
            let a = area(&d.b);
            let inside = signs.iter().any(|s| a > 0.0 && inter(&d.b, s) / a >= cfg.person_in_exit_sign);
            if inside {
                *drop.entry("표지 속 사람").or_default() += 1;
            }
            !inside
        });
    }
    keep
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        bail!("사용: yolods <labels.jsonl> <lists 폴더> <out> [config.json]");
    }
    let cfg_path = a.get(4).cloned().unwrap_or_else(|| {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("config.json").to_string_lossy().into_owned()
    });
    let cfg: Config = serde_json::from_str(&fs::read_to_string(&cfg_path).context("config")?)?;
    if cfg.score_thr.len() != cfg.classes.len() {
        bail!("score_thr 개수가 클래스 수와 다름");
    }
    // 메타: 라벨러 클래스 번호와 이름이 맞는지
    let meta: Value = serde_json::from_str(&fs::read_to_string(format!("{}.meta.json", a[1])).context("meta")?)?;
    for c in meta["classes"].as_array().context("meta classes")? {
        let id = c["id"].as_u64().unwrap() as usize;
        if cfg.classes.get(id).map(|s| s.as_str()) != c["name"].as_str() {
            bail!("클래스 번호가 라벨러와 다름: {} {:?}", id, c["name"]);
        }
    }
    // 나눔
    let mut split_of: HashMap<String, &'static str> = HashMap::new();
    for sp in ["train", "val", "test"] {
        for l in fs::read_to_string(format!("{}/{}.txt", a[2], sp))?.lines() {
            if !l.is_empty() {
                split_of.insert(l.to_string(), sp);
            }
        }
    }
    let out = Path::new(&a[3]);
    let mask_size = meta["mask_size"].as_u64().unwrap_or(0) as usize;
    for sp in ["train", "val", "test"] {
        fs::create_dir_all(out.join("images").join(sp))?;
        fs::create_dir_all(out.join("labels").join(sp))?;
        if mask_size > 0 {
            fs::create_dir_all(out.join("masks").join(sp))?;
        }
    }
    let mut n_mask = 0usize;
    let mut n_mask_cut = 0usize;
    let nc = cfg.classes.len();
    let mut n_img: BTreeMap<&str, usize> = BTreeMap::new();
    let mut n_box: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    let mut n_err = 0usize;
    let mut n_unsplit = 0usize;
    let mut drop: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut names = fs::File::create(out.join("names.tsv"))?;
    let f = BufReader::new(fs::File::open(&a[1])?);
    for line in f.lines() {
        let line = line?;
        let r: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue, // 쓰다 만 마지막 줄
        };
        let img = r["img"].as_str().unwrap_or("").to_string();
        if r.get("err").is_some() {
            n_err += 1;
            continue;
        }
        let Some(&sp) = split_of.get(&img) else {
            n_unsplit += 1;
            continue;
        };
        let (w, h) = (r["w"].as_f64().unwrap() as f32, r["h"].as_f64().unwrap() as f32);
        let mut dets = Vec::new();
        for d in r["det"].as_array().unwrap() {
            let b: Vec<f32> = d["b"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
            let m = d.get("m").and_then(|v| v.as_array()).map(|a| a.iter().map(|x| x.as_u64().unwrap() as u32).collect());
            dets.push(Det { c: d["c"].as_u64().unwrap() as usize, s: d["s"].as_f64().unwrap() as f32, b: [b[0], b[1], b[2], b[3]], m });
        }
        let keep = select(dets, &cfg, w, h, &mut drop);
        let ext = Path::new(&img).extension().and_then(|e| e.to_str()).unwrap_or("jpg").to_lowercase();
        let stem = &sha1_smol::Sha1::from(img.as_bytes()).digest().to_string()[..16];
        let name = format!("{stem}.{ext}");
        let link = out.join("images").join(sp).join(&name);
        if !link.exists() {
            std::os::unix::fs::symlink(&img, &link).with_context(|| format!("링크 {}", link.display()))?;
        }
        let mut txt = String::new();
        let cnt = n_box.entry(sp).or_insert_with(|| vec![0; nc]);
        for d in &keep {
            let (cx, cy) = ((d.b[0] + d.b[2]) / 2.0 / w, (d.b[1] + d.b[3]) / 2.0 / h);
            let (bw, bh) = ((d.b[2] - d.b[0]) / w, (d.b[3] - d.b[1]) / h);
            txt += &format!("{} {:.6} {:.6} {:.6} {:.6}\n", d.c, cx, cy, bw, bh);
            cnt[d.c] += 1;
        }
        fs::write(out.join("labels").join(sp).join(format!("{stem}.txt")), txt)?;
        if mask_size > 0 {
            let runs = index_map(&keep, mask_size);
            let mut buf = Vec::with_capacity(12 + 4 * runs.len());
            buf.extend_from_slice(b"YMSK");
            buf.extend_from_slice(&(mask_size as u16).to_le_bytes());
            buf.extend_from_slice(&(keep.len().min(MAX_INST) as u16).to_le_bytes());
            buf.extend_from_slice(&(runs.len() as u32).to_le_bytes());
            for r in &runs {
                buf.extend_from_slice(&r.to_le_bytes());
            }
            fs::write(out.join("masks").join(sp).join(format!("{stem}.msk")), buf)?;
            n_mask += keep.iter().take(MAX_INST).filter(|d| d.m.is_some()).count();
            n_mask_cut += keep.len().saturating_sub(MAX_INST);
        }
        writeln!(names, "{sp}\t{name}\t{img}")?;
        *n_img.entry(sp).or_default() += 1;
    }
    let yaml = format!(
        "path: {}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{}",
        fs::canonicalize(out)?.display(),
        cfg.classes.iter().enumerate().map(|(i, c)| format!("  {i}: {c}\n")).collect::<String>()
    );
    fs::write(out.join("data.yaml"), yaml)?;
    let stats = serde_json::json!({
        "images": n_img, "boxes": n_box, "classes": cfg.classes, "dropped": drop,
        "read_errors": n_err, "not_in_split": n_unsplit, "masks": n_mask, "masks_over_255": n_mask_cut, "config": serde_json::from_str::<Value>(&fs::read_to_string(&cfg_path)?)?,
    });
    fs::write(out.join("stats.json"), serde_json::to_string_pretty(&stats)?)?;
    println!("{}", serde_json::to_string_pretty(&stats)?);
    Ok(())
}
