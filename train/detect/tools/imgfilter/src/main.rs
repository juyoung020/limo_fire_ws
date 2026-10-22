//! 크롤링한 학습 이미지 거르기.
//!
//! imgfilter <prompts.json> <SigLIP 폴더> <data/raw> <data/clean> [클래스 …]
//!
//! 클래스마다 <raw>/<출처>/<클래스>/ 아래 모든 이미지를 보고 차례로 거른다.
//! 1. 규칙: 못 여는 파일, 짧은 변 < min_side, 가로세로 비 > max_aspect.
//! 2. 내용: SigLIP(so400m-patch14-384) 제로샷. keep·reject 설명 전부에 softmax 를 하고
//!    keep 쪽 확률 합이 keep_min 미만이면 버린다(그림·아이콘·관계없는 사진).
//! 3. 중복: 남은 것끼리 이미지 임베딩 코사인 ≥ dup_cos 면 큰 사진 하나만 남긴다.
//! 남긴 사진은 <clean>/<클래스>/<출처>__<검색어 경로>__<파일> 로 하드링크(원본은 그대로),
//! 판정은 모두 <clean>/<클래스>.tsv 에 (경로, 판정, 이유, keep 확률, 가장 높은 설명).
//! <clean>/<클래스>/ 는 매번 새로 만든다.

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::VarBuilder;
use candle_transformers::models::siglip;
use rayon::prelude::*;
use serde_json::Value;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

const BATCH: usize = 32;

struct Cfg {
    sources: Vec<String>,
    keep_min: f32,
    dup_cos: f32,
    min_side: u32,
    max_aspect: f32,
}

struct Item {
    path: PathBuf,
    name: String,
    pixels: u64,
    verdict: &'static str,
    reason: String,
    keep_p: f32,
    top: String,
}

fn strs(v: &Value) -> Vec<String> {
    v.as_array().map(|a| a.iter().filter_map(|s| s.as_str().map(String::from)).collect()).unwrap_or_default()
}

fn is_image(p: &Path) -> bool {
    matches!(
        p.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref(),
        Some("jpg" | "jpeg" | "png" | "webp" | "gif" | "bmp")
    )
}

/// 짧은 쪽을 맞추지 않고 정사각형으로 늘려 [-1, 1] 로 (SigLIP 전처리와 같음).
fn load(path: &Path, size: u32) -> Result<(u32, u32, Vec<f32>)> {
    let img = image::ImageReader::open(path)?.with_guessed_format()?.decode()?;
    let (w, h) = (img.width(), img.height());
    let rgb = img.resize_exact(size, size, image::imageops::FilterType::Triangle).to_rgb8();
    let mut chw = vec![0f32; 3 * (size * size) as usize];
    let plane = (size * size) as usize;
    for (i, p) in rgb.pixels().enumerate() {
        for c in 0..3 {
            chw[c * plane + i] = p[c] as f32 / 127.5 - 1.0;
        }
    }
    Ok((w, h, chw))
}

fn l2norm(t: &Tensor) -> Result<Tensor> {
    Ok(t.broadcast_div(&t.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?)?)
}

fn text_features(model: &siglip::Model, tok: &Tokenizer, cfg: &siglip::Config, texts: &[String], dev: &Device) -> Result<Tensor> {
    let max_len = cfg.text_config.max_position_embeddings;
    let pad = cfg.text_config.pad_token_id;
    let mut ids = Vec::new();
    for t in texts {
        let mut v = tok.encode(t.as_str(), true).map_err(anyhow::Error::msg)?.get_ids().to_vec();
        v.truncate(max_len);
        v.resize(max_len, pad);
        ids.push(Tensor::new(v, dev)?);
    }
    l2norm(&model.get_text_features(&Tensor::stack(&ids, 0)?)?)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        bail!("imgfilter <prompts.json> <SigLIP 폴더> <data/raw> <data/clean> [클래스 …]");
    }
    let prompts: Value = serde_json::from_str(&fs::read_to_string(&args[1])?)?;
    let (model_dir, raw, clean) = (Path::new(&args[2]), Path::new(&args[3]), Path::new(&args[4]));
    let cfg = Cfg {
        sources: strs(&prompts["sources"]),
        keep_min: prompts["keep_min"].as_f64().unwrap_or(0.5) as f32,
        dup_cos: prompts["dup_cos"].as_f64().unwrap_or(0.95) as f32,
        min_side: prompts["min_side"].as_u64().unwrap_or(200) as u32,
        max_aspect: prompts["max_aspect"].as_f64().unwrap_or(3.0) as f32,
    };
    let common_reject = strs(&prompts["common_reject"]);
    let classes: Vec<String> = if args.len() > 5 {
        args[5..].to_vec()
    } else {
        prompts["classes"].as_object().context("classes 없음")?.keys().cloned().collect()
    };

    let dev = Device::cuda_if_available(0)?;
    let dtype = if dev.is_cuda() { DType::BF16 } else { DType::F32 };
    eprintln!("장치 {dev:?}, {dtype:?}");
    let scfg: siglip::Config = serde_json::from_str(&fs::read_to_string(model_dir.join("config.json"))?)?;
    let tok = Tokenizer::from_file(model_dir.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_dir.join("model.safetensors")], dtype, &dev)? };
    let model = siglip::Model::new(&scfg, vb)?;
    let size = scfg.vision_config.image_size as u32;
    let scale = Tensor::new(&[100f32], &dev)?; // logit_scale.exp() 근삿값 — softmax 날카로움만 정한다

    for class in &classes {
        let c = &prompts["classes"][class];
        let keep = strs(&c["keep"]);
        if keep.is_empty() {
            eprintln!("{class}: prompts.json 에 없음, 건너뜀");
            continue;
        }
        let mut texts = keep.clone();
        texts.extend(strs(&c["reject"]));
        texts.extend(common_reject.iter().cloned());
        let tfeat = text_features(&model, &tok, &scfg, &texts, &dev)?.to_dtype(DType::F32)?;

        // 파일 모으기
        let mut items = Vec::new();
        for src in &cfg.sources {
            let root = raw.join(src).join(class);
            if !root.is_dir() { continue; }
            for e in walkdir::WalkDir::new(&root).sort_by_file_name().into_iter().filter_map(|e| e.ok()) {
                let p = e.path();
                if !e.file_type().is_file() || !is_image(p) { continue; }
                let rel = p.strip_prefix(&root)?.to_string_lossy().replace('/', "__");
                items.push(Item {
                    path: p.to_path_buf(),
                    name: format!("{src}__{rel}"),
                    pixels: 0,
                    verdict: "keep",
                    reason: String::new(),
                    keep_p: 0.0,
                    top: String::new(),
                });
            }
        }
        eprintln!("== {class}: {} 장", items.len());

        // 규칙 + SigLIP, 배치마다 디코딩은 병렬
        let mut feats: Vec<Option<Tensor>> = vec![None; items.len()];
        for start in (0..items.len()).step_by(BATCH) {
            let end = (start + BATCH).min(items.len());
            let loaded: Vec<_> = items[start..end].par_iter().map(|it| load(&it.path, size)).collect();
            let mut idx = Vec::new();
            let mut pix = Vec::new();
            for (k, r) in loaded.into_iter().enumerate() {
                let it = &mut items[start + k];
                match r {
                    Err(e) => { it.verdict = "reject"; it.reason = format!("열기 실패: {e}"); }
                    Ok((w, h, chw)) => {
                        it.pixels = w as u64 * h as u64;
                        let aspect = w.max(h) as f32 / w.min(h).max(1) as f32;
                        if w.min(h) < cfg.min_side {
                            it.verdict = "reject"; it.reason = format!("작음 {w}x{h}");
                        } else if aspect > cfg.max_aspect {
                            it.verdict = "reject"; it.reason = format!("가로세로 {aspect:.1}");
                        } else {
                            idx.push(start + k);
                            pix.push(Tensor::from_vec(chw, (3, size as usize, size as usize), &dev)?);
                        }
                    }
                }
            }
            if idx.is_empty() { continue; }
            let ifeat = l2norm(&model.get_image_features(&Tensor::stack(&pix, 0)?.to_dtype(dtype)?)?.to_dtype(DType::F32)?)?;
            let probs = candle_nn::ops::softmax(&ifeat.matmul(&tfeat.t()?)?.broadcast_mul(&scale)?, D::Minus1)?
                .to_vec2::<f32>()?;
            for (j, &i) in idx.iter().enumerate() {
                let p = &probs[j];
                let it = &mut items[i];
                it.keep_p = p[..keep.len()].iter().sum();
                let top = (0..p.len()).max_by(|&a, &b| p[a].total_cmp(&p[b])).unwrap();
                it.top = texts[top].clone();
                if it.keep_p < cfg.keep_min {
                    it.verdict = "reject";
                    it.reason = "내용".into();
                } else {
                    feats[i] = Some(ifeat.get(j)?);
                }
            }
            if (start / BATCH) % 20 == 0 { eprintln!("  {end}/{}", items.len()); }
        }

        // 중복: 큰 사진부터 남기고, 이미 남긴 것과 너무 비슷하면 버림
        let mut order: Vec<usize> = (0..items.len()).filter(|&i| feats[i].is_some()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(items[i].pixels));
        if !order.is_empty() {
            let m = Tensor::stack(&order.iter().map(|&i| feats[i].clone().unwrap()).collect::<Vec<_>>(), 0)?;
            let sim = m.matmul(&m.t()?)?.to_vec2::<f32>()?;
            let mut kept: Vec<usize> = Vec::new();
            for a in 0..order.len() {
                if let Some(&b) = kept.iter().find(|&&b| sim[a][b] >= cfg.dup_cos) {
                    let reason = format!("중복 {:.3} {}", sim[a][b], items[order[b]].name);
                    let it = &mut items[order[a]];
                    it.verdict = "reject";
                    it.reason = reason;
                } else {
                    kept.push(a);
                }
            }
        }

        // 결과
        let out = clean.join(class);
        if out.exists() { fs::remove_dir_all(&out)?; }
        fs::create_dir_all(&out)?;
        let mut tsv = fs::File::create(clean.join(format!("{class}.tsv")))?;
        writeln!(tsv, "path\tverdict\treason\tkeep_p\ttop")?;
        let (mut n_keep, mut n_rule, mut n_content, mut n_dup) = (0, 0, 0, 0);
        for it in &items {
            writeln!(tsv, "{}\t{}\t{}\t{:.3}\t{}", it.path.display(), it.verdict, it.reason, it.keep_p, it.top)?;
            if it.verdict == "keep" {
                let dst = out.join(&it.name);
                fs::hard_link(&it.path, &dst).or_else(|_| fs::copy(&it.path, &dst).map(|_| ()))?;
                n_keep += 1;
            } else if it.reason == "내용" {
                n_content += 1;
            } else if it.reason.starts_with("중복") {
                n_dup += 1;
            } else {
                n_rule += 1;
            }
        }
        eprintln!("== {class}: 남김 {n_keep} / {} (깨짐·크기 {n_rule}, 내용 {n_content}, 중복 {n_dup})", items.len());
    }
    Ok(())
}
