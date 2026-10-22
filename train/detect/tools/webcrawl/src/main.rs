//! 라이선스가 붙은 공개 이미지 모으기.
//!
//! webcrawl <queries.json> <출력 폴더> [클래스 …]
//!
//! - Wikimedia Commons: 분류(하위 분류까지 깊이 제한) + 파일 검색. Commons 는 자유 라이선스만 받으므로 모두 쓴다.
//! - Openverse: CC 이미지 검색. 변경 금지(ND)·라이선스 모름은 뺀다.
//! - 이미지는 긴 변 1024 px 이하로 받는다(Commons 는 썸네일, Openverse 는 원본).
//! - 같은 이미지(바이트 SHA-1)는 한 번만. 이미 받은 것은 건너뛴다(다시 돌려도 됨).
//! - <출력>/<클래스>/<출처>_<해시12>.<jpg|png>, <출력>/<클래스>/manifest.jsonl 에 줄마다
//!   {file, source, page, license, author, title, width, height}.
//!
//! 예의: User-Agent 에 저장소 주소, Commons API 1 초에 한 번, Openverse 쪽은 3 초에 한 번, 429 면 기다렸다 다시.

use serde_json::{json, Value};
use std::collections::{HashSet, VecDeque};
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::thread::sleep;
use std::time::Duration;

const UA: &str = "limo-fire-dataset/0.1 (https://github.com/juyoung020/limo_fire_ws)";
const COMMONS: &str = "https://commons.wikimedia.org/w/api.php";
const OPENVERSE: &str = "https://api.openverse.org/v1/images/";
const OV_LICENSES: &str = "cc0,pdm,by,by-sa,by-nc,by-nc-sa";
const MIN_SIDE: u64 = 300;

struct Candidate {
    source: &'static str,
    url: String,
    page: String,
    license: String,
    author: String,
    title: String,
    width: u64,
    height: u64,
}

fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new().user_agent(UA).timeout(Duration::from_secs(60)).build()
}

/// GET → JSON. 429·5xx 는 기다렸다 최대 5 번 다시.
fn get_json(a: &ureq::Agent, url: &str, query: &[(&str, &str)]) -> Option<Value> {
    for attempt in 0..5u64 {
        let mut r = a.get(url);
        for (k, v) in query { r = r.query(k, v); }
        match r.call() {
            Ok(resp) => return resp.into_json().ok(),
            Err(ureq::Error::Status(code, _)) if code == 429 || code >= 500 => {
                eprintln!("  [{code}] 기다림 {}s", 10 * (attempt + 1));
                sleep(Duration::from_secs(10 * (attempt + 1)));
            }
            Err(e) => { eprintln!("  요청 실패: {e}"); return None; }
        }
    }
    None
}

fn strip_html(s: &str) -> String {
    let mut out = String::new();
    let mut tag = false;
    for c in s.chars() {
        match c { '<' => tag = true, '>' => tag = false, _ if !tag => out.push(c), _ => {} }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Commons 분류 안 파일 제목(하위 분류는 depth 만큼 내려감)
fn commons_category(a: &ureq::Agent, cat: &str, depth: u32, limit: usize) -> Vec<String> {
    let mut files = Vec::new();
    let mut seen = HashSet::new();
    let mut q = VecDeque::from([(format!("Category:{cat}"), 0u32)]);
    while let Some((c, d)) = q.pop_front() {
        if !seen.insert(c.clone()) || files.len() >= limit { continue; }
        let mut cont = String::new();
        loop {
            let mut params = vec![("action", "query"), ("list", "categorymembers"), ("cmtitle", c.as_str()),
                ("cmtype", "file|subcat"), ("cmlimit", "500"), ("format", "json")];
            if !cont.is_empty() { params.push(("cmcontinue", cont.as_str())); }
            let Some(v) = get_json(a, COMMONS, &params) else { break };
            sleep(Duration::from_secs(1));
            for m in v["query"]["categorymembers"].as_array().into_iter().flatten() {
                let t = m["title"].as_str().unwrap_or("").to_string();
                if t.starts_with("File:") { files.push(t); }
                else if t.starts_with("Category:") && d < depth { q.push_back((t, d + 1)); }
            }
            match v["continue"]["cmcontinue"].as_str() { Some(n) if files.len() < limit => cont = n.to_string(), _ => break }
        }
    }
    files.truncate(limit);
    files
}

fn commons_search(a: &ureq::Agent, q: &str, limit: usize) -> Vec<String> {
    let mut files = Vec::new();
    let query = format!("{q} filetype:bitmap");
    let mut offset = 0usize;
    while files.len() < limit {
        let off = offset.to_string();
        let Some(v) = get_json(a, COMMONS, &[("action", "query"), ("list", "search"), ("srnamespace", "6"),
            ("srsearch", query.as_str()), ("srlimit", "100"), ("sroffset", off.as_str()), ("format", "json")]) else { break };
        sleep(Duration::from_secs(1));
        let hits = v["query"]["search"].as_array().cloned().unwrap_or_default();
        if hits.is_empty() { break; }
        for h in &hits { if let Some(t) = h["title"].as_str() { files.push(t.to_string()); } }
        match v["continue"]["sroffset"].as_u64() { Some(n) => offset = n as usize, None => break }
    }
    files.truncate(limit);
    files
}

/// 파일 제목 → 내려받을 후보(썸네일 1024, 라이선스·작성자)
fn commons_info(a: &ureq::Agent, titles: &[String]) -> Vec<Candidate> {
    let mut out = Vec::new();
    for chunk in titles.chunks(50) {
        let joined = chunk.join("|");
        let Some(v) = get_json(a, COMMONS, &[("action", "query"), ("titles", joined.as_str()), ("prop", "imageinfo"),
            ("iiprop", "url|mime|size|extmetadata"), ("iiurlwidth", "1024"), ("format", "json")]) else { continue };
        sleep(Duration::from_secs(1));
        for p in v["query"]["pages"].as_object().into_iter().flatten().map(|(_, p)| p) {
            let Some(ii) = p["imageinfo"].get(0) else { continue };
            let mime = ii["mime"].as_str().unwrap_or("");
            if mime != "image/jpeg" && mime != "image/png" { continue; }
            let (w, h) = (ii["width"].as_u64().unwrap_or(0), ii["height"].as_u64().unwrap_or(0));
            if w.min(h) < MIN_SIDE { continue; }
            let em = &ii["extmetadata"];
            let url = ii["thumburl"].as_str().or(ii["url"].as_str()).unwrap_or("").to_string();
            if url.is_empty() { continue; }
            out.push(Candidate {
                source: "commons",
                url,
                page: ii["descriptionurl"].as_str().unwrap_or("").to_string(),
                license: em["LicenseShortName"]["value"].as_str().unwrap_or("").to_string(),
                author: strip_html(em["Artist"]["value"].as_str().unwrap_or("")),
                title: p["title"].as_str().unwrap_or("").to_string(),
                width: w,
                height: h,
            });
        }
    }
    out
}

fn openverse(a: &ureq::Agent, q: &str, limit: usize) -> Vec<Candidate> {
    let mut out = Vec::new();
    for page in 1..=12u32 {
        if out.len() >= limit { break; }
        let pg = page.to_string();
        let Some(v) = get_json(a, OPENVERSE, &[("q", q), ("page_size", "20"), ("page", pg.as_str()), ("license", OV_LICENSES)]) else { break };
        sleep(Duration::from_secs(3));
        let rs = v["results"].as_array().cloned().unwrap_or_default();
        if rs.is_empty() { break; }
        for r in rs {
            let (w, h) = (r["width"].as_u64().unwrap_or(0), r["height"].as_u64().unwrap_or(0));
            if w != 0 && w.min(h) < MIN_SIDE { continue; }
            let lic = r["license"].as_str().unwrap_or("");
            if lic.contains("nd") || lic.is_empty() { continue; }
            out.push(Candidate {
                source: "openverse",
                url: r["url"].as_str().unwrap_or("").to_string(),
                page: r["foreign_landing_url"].as_str().unwrap_or("").to_string(),
                license: format!("CC {} {}", lic.to_uppercase(), r["license_version"].as_str().unwrap_or("")).trim().to_string(),
                author: r["creator"].as_str().unwrap_or("").to_string(),
                title: r["title"].as_str().unwrap_or("").to_string(),
                width: w,
                height: h,
            });
        }
        if v["page_count"].as_u64().map_or(true, |n| page as u64 >= n) { break; }
    }
    out
}

fn download(a: &ureq::Agent, url: &str) -> Option<Vec<u8>> {
    let resp = a.get(url).call().ok()?;
    let mut buf = Vec::new();
    resp.into_reader().take(25 << 20).read_to_end(&mut buf).ok()?;
    let ok = buf.starts_with(&[0xFF, 0xD8, 0xFF]) || buf.starts_with(&[0x89, b'P', b'N', b'G']);
    if ok { Some(buf) } else { None }
}

fn run_class(a: &ureq::Agent, name: &str, cfg: &Value, out_dir: &Path) {
    let dir = out_dir.join(name);
    fs::create_dir_all(&dir).unwrap();
    let max = cfg["max"].as_u64().unwrap_or(500) as usize;
    let manifest = dir.join("manifest.jsonl");
    let mut have: HashSet<String> = HashSet::new();   // 이미 받은 해시·주소
    if let Ok(s) = fs::read_to_string(&manifest) {
        for l in s.lines() {
            if let Ok(v) = serde_json::from_str::<Value>(l) {
                for k in ["file", "url"] { if let Some(x) = v[k].as_str() { have.insert(x.to_string()); } }
            }
        }
    }
    let mut count = have.len() / 2;
    eprintln!("[{name}] 이미 {count} 장, 목표 {max}");

    let mut titles: Vec<String> = Vec::new();
    for c in cfg["commons_categories"].as_array().into_iter().flatten() {
        let (cat, depth) = (c[0].as_str().unwrap_or(""), c[1].as_u64().unwrap_or(1) as u32);
        let t = commons_category(a, cat, depth, max * 2);
        eprintln!("  Commons 분류 {cat} (깊이 {depth}): {} 파일", t.len());
        titles.extend(t);
    }
    for q in cfg["commons_search"].as_array().into_iter().flatten().filter_map(|q| q.as_str()) {
        let t = commons_search(a, q, max);
        eprintln!("  Commons 검색 \"{q}\": {}", t.len());
        titles.extend(t);
    }
    let mut seen_t = HashSet::new();
    titles.retain(|t| seen_t.insert(t.clone()));
    let mut cands = commons_info(a, &titles);
    for q in cfg["openverse"].as_array().into_iter().flatten().filter_map(|q| q.as_str()) {
        let c = openverse(a, q, max);
        eprintln!("  Openverse \"{q}\": {}", c.len());
        cands.extend(c);
    }
    eprintln!("  후보 {} 개", cands.len());

    let mut mf = OpenOptions::new().create(true).append(true).open(&manifest).unwrap();
    for c in cands {
        if count >= max { break; }
        if c.url.is_empty() || have.contains(&c.url) { continue; }
        let Some(bytes) = download(a, &c.url) else { continue };
        if c.source == "commons" { sleep(Duration::from_millis(200)); }
        let hash = sha1_smol::Sha1::from(&bytes).digest().to_string();
        let ext = if bytes.starts_with(&[0x89, b'P']) { "png" } else { "jpg" };
        let file = format!("{}_{}.{}", c.source, &hash[..12], ext);
        if have.contains(&file) { continue; }
        fs::write(dir.join(&file), &bytes).unwrap();
        have.insert(file.clone());
        have.insert(c.url.clone());
        let line = json!({"file": file, "url": c.url, "source": c.source, "page": c.page, "license": c.license,
            "author": c.author, "title": c.title, "width": c.width, "height": c.height});
        writeln!(mf, "{line}").unwrap();
        count += 1;
        if count % 50 == 0 { eprintln!("  {count}/{max}"); }
    }
    eprintln!("[{name}] 끝: {count} 장");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 { eprintln!("webcrawl <queries.json> <출력 폴더> [클래스 …]"); std::process::exit(2); }
    let cfg: Value = serde_json::from_str(&fs::read_to_string(&args[1]).expect("queries.json")).expect("JSON");
    let out = PathBuf::from(&args[2]);
    let only: Vec<&String> = args[3..].iter().collect();
    let a = agent();
    for (name, c) in cfg.as_object().unwrap() {
        if name.starts_with('_') || (!only.is_empty() && !only.iter().any(|o| *o == name)) { continue; }
        run_class(&a, name, c, &out);
    }
}
