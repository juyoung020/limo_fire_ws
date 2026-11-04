//! `.xrp` — 탐사 판 하나의 2D 재생, 바이너리. 모두 little-endian. 쓰는 쪽은 `include/runfolder.h` `rf::ReplayWriter`.
//!
//! ```text
//! "XRP1"  u32 head_len  head(JSON, UTF-8)  [0 채움 → 8 정렬]  = 데이터 시작 D
//! 섹션은 D + off 에 있다(off 는 8 의 배수, head.sections 에 이름·off·len).
//!
//! frames : n_frames × n_cols × f32         (행 우선, 열 이름 = head.cols)
//! gt     : grid.w × grid.h × u8            정답 맵(처음 한 번): 0 빈 곳, 1 벽·장애물, 2 맵 밖
//! obs    : 기록 이어짐 [u32 frame][i32 x0][i32 y0][i32 x1][i32 y1][i8 칸 × (x1−x0+1)(y1−y0+1)]
//!          관측 지도(정책이 쌓은 확률 지도): −1 모름, 0–100 점유 %. 첫 기록 = frame 0 의 전체 격자, 그 뒤는 바뀐 칸을 감싼 사각형만
//!          (한 프레임에 기록 여러 개 가능 — runfolder.h 는 16×16 타일마다 하나, 쓴 차례대로 덮어 쓴다).
//!          칸 차례는 행 우선, 행 0 = y0(작은 y). 사각형 꼭짓점 포함.
//! cov    : 기록 이어짐 [u32 frame][u32 n][u32 칸 번호 × n]  그 걸음에 새로 덮인 칸(번호 = j·w + i). 덮음은 늘기만 한다.
//! scan   : (선택) n_frames × lidar.n × u16  라이다 거리 mm, 0xFFFF = 반사 없음. 빔 k 각도 = lidar.a0 + k·lidar.da (로봇 기준, 라디안)
//! ```
//!
//! 머리 JSON: `format` "XRP1", `meta`(그 판의 episodes.jsonl 줄), `dt`(s), `n_frames`, `cols`, `grid` {w, h, res, ox, oy}
//! (칸 (i, j) 가운데 = (ox + (i+½)·res, oy + (j+½)·res), map 좌표 REP-103), `lidar` {n, a0, da, rmax} (선택),
//! `limits` {v, vy, w} 행동 한도, `reward_terms` [항 이름], `sections` [{name, off, len}].
//!
//! 프레임 열(이름으로 찾는다 — 쓰는 쪽은 가진 것만 싣는다). 필수: `t`, `x`, `y`, `yaw`(참 자세).
//! 선택: `sx`, `sy`, `syaw`(지도 자세 = 참값 + 오차), `v`, `vy`, `w`(잰 속도), `a_v`, `a_vy`, `a_w`(정책 행동, 물리 단위),
//! `p_end`(끝 선언 확률), `a_end`(뽑은 선언 0/1), `value`, `r_<항>`(이번 걸음 보상 항), `ret`(누적 리턴),
//! `coverage`(정답 기준 덮은 비율), `known_m2`(정책이 아는 넓이), `ev`(사건 비트, 아래),
//! `h<k>_x`, `h<k>_y`, `h<k>_r`(움직이는 사람 k 의 원기둥 가운데·반지름, 세계 좌표 m — 그 걸음에 없는 사람은 NaN).

use serde_json::Value;

pub const MAGIC: &[u8; 4] = b"XRP1";
pub const REQUIRED_COLS: &[&str] = &["t", "x", "y", "yaw"];
/// `ev` 비트
pub const EV_COLLISION: u32 = 1;
pub const EV_DECLARE: u32 = 2;
pub const EV_TRUNCATED: u32 = 4;
pub const EV_REACHED95: u32 = 8;
/// gt 칸 값
pub const GT_FREE: u8 = 0;
pub const GT_WALL: u8 = 1;
pub const GT_OUTSIDE: u8 = 2;
pub const SCAN_NONE: u16 = 0xFFFF;

pub fn pad8(n: usize) -> usize {
    (n + 7) & !7
}

/// 머리 길이(파일 앞 8 바이트에서). 데이터 시작 = pad8(8 + 이 값)
pub fn head_len(b: &[u8]) -> Option<usize> {
    if b.len() < 8 || &b[0..4] != MAGIC {
        return None;
    }
    Some(u32::from_le_bytes([b[4], b[5], b[6], b[7]]) as usize)
}

/// 머리만 읽기(목록용). 파일 앞부분 바이트(8 + head_len 이상)를 받는다.
pub fn read_head(b: &[u8]) -> Option<Value> {
    let n = head_len(b)?;
    if b.len() < 8 + n {
        return None;
    }
    serde_json::from_slice(&b[8..8 + n]).ok()
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Grid {
    pub w: usize,
    pub h: usize,
    pub res: f64,
    pub ox: f64,
    pub oy: f64,
}

/// 다 읽은 판 하나(검사용 — 뷰어 화면은 같은 규칙을 JS 로 읽는다)
#[derive(Debug)]
pub struct Xrp {
    pub head: Value,
    pub cols: Vec<String>,
    pub n_frames: usize,
    pub frames: Vec<f32>,
    pub grid: Grid,
    pub gt: Vec<u8>,
    /// (frame, x0, y0, x1, y1, 칸)
    pub obs: Vec<(u32, i32, i32, i32, i32, Vec<i8>)>,
    /// (frame, 새로 덮인 칸 번호)
    pub cov: Vec<(u32, Vec<u32>)>,
    pub scan: Option<(usize, Vec<u16>)>,
}

fn rd_u32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
fn rd_i32(b: &[u8], o: usize) -> i32 {
    rd_u32(b, o) as i32
}

impl Xrp {
    pub fn col(&self, name: &str) -> Option<usize> {
        self.cols.iter().position(|c| c == name)
    }
    pub fn at(&self, frame: usize, col: usize) -> f32 {
        self.frames[frame * self.cols.len() + col]
    }

    /// 파일 전체를 읽고 규칙을 확인한다. 어긋나면 무엇이 어긋났는지 글로.
    pub fn parse(b: &[u8]) -> Result<Xrp, String> {
        let hl = head_len(b).ok_or("not an XRP1 file (magic)")?;
        if b.len() < 8 + hl {
            return Err("file shorter than its head".into());
        }
        let head: Value = serde_json::from_slice(&b[8..8 + hl]).map_err(|e| format!("head JSON: {}", e))?;
        let d0 = pad8(8 + hl);
        let cols: Vec<String> = head["cols"].as_array().ok_or("head.cols missing")?.iter().map(|c| c.as_str().unwrap_or("").to_string()).collect();
        for r in REQUIRED_COLS {
            if !cols.iter().any(|c| c == r) {
                return Err(format!("required frame column '{}' missing", r));
            }
        }
        let n_frames = head["n_frames"].as_u64().ok_or("head.n_frames missing")? as usize;
        if !(head["dt"].as_f64().map(|d| d > 0.0).unwrap_or(false)) {
            return Err("head.dt missing or ≤ 0".into());
        }
        let g = &head["grid"];
        let grid = Grid {
            w: g["w"].as_u64().ok_or("grid.w")? as usize,
            h: g["h"].as_u64().ok_or("grid.h")? as usize,
            res: g["res"].as_f64().ok_or("grid.res")?,
            ox: g["ox"].as_f64().ok_or("grid.ox")?,
            oy: g["oy"].as_f64().ok_or("grid.oy")?,
        };
        if grid.w == 0 || grid.h == 0 || grid.res <= 0.0 {
            return Err("grid must be non-empty with res > 0".into());
        }
        let sec = |name: &str| -> Result<Option<&[u8]>, String> {
            let Some(arr) = head["sections"].as_array() else { return Err("head.sections missing".into()) };
            let Some(s) = arr.iter().find(|s| s["name"] == name) else { return Ok(None) };
            let off = s["off"].as_u64().ok_or("section.off")? as usize;
            let len = s["len"].as_u64().ok_or("section.len")? as usize;
            if off % 8 != 0 {
                return Err(format!("section {} off {} not 8-aligned", name, off));
            }
            let a = d0 + off;
            if a + len > b.len() {
                return Err(format!("section {} [{}, {}) past end of file ({} B)", name, a, a + len, b.len()));
            }
            Ok(Some(&b[a..a + len]))
        };
        let fr = sec("frames")?.ok_or("frames section missing")?;
        if fr.len() != n_frames * cols.len() * 4 {
            return Err(format!("frames: {} B, expected n_frames {} × cols {} × 4", fr.len(), n_frames, cols.len()));
        }
        let frames: Vec<f32> = fr.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let gt = sec("gt")?.ok_or("gt section missing")?.to_vec();
        if gt.len() != grid.w * grid.h {
            return Err(format!("gt: {} B, expected w×h {}", gt.len(), grid.w * grid.h));
        }
        if let Some(v) = gt.iter().find(|&&v| v > GT_OUTSIDE) {
            return Err(format!("gt cell value {} (allowed 0 free, 1 wall, 2 outside)", v));
        }
        let mut obs = vec![];
        if let Some(o) = sec("obs")? {
            let mut p = 0;
            let mut last = 0u32;
            while p < o.len() {
                if p + 20 > o.len() {
                    return Err("obs: truncated record head".into());
                }
                let (f, x0, y0, x1, y1) = (rd_u32(o, p), rd_i32(o, p + 4), rd_i32(o, p + 8), rd_i32(o, p + 12), rd_i32(o, p + 16));
                if x0 < 0 || y0 < 0 || x1 < x0 || y1 < y0 || x1 as usize >= grid.w || y1 as usize >= grid.h {
                    return Err(format!("obs: rect ({},{})–({},{}) outside grid {}×{}", x0, y0, x1, y1, grid.w, grid.h));
                }
                if f as usize >= n_frames.max(1) || f < last {
                    return Err(format!("obs: frame {} out of order or ≥ n_frames", f));
                }
                last = f;
                let n = ((x1 - x0 + 1) * (y1 - y0 + 1)) as usize;
                if p + 20 + n > o.len() {
                    return Err("obs: truncated cells".into());
                }
                let cells: Vec<i8> = o[p + 20..p + 20 + n].iter().map(|&c| c as i8).collect();
                if let Some(c) = cells.iter().find(|&&c| c < -1 || c > 100) {
                    return Err(format!("obs: cell value {} (allowed −1, 0..100)", c));
                }
                obs.push((f, x0, y0, x1, y1, cells));
                p += 20 + n;
            }
            if let Some(first) = obs.first() {
                if first.0 != 0 || first.1 != 0 || first.2 != 0 || first.3 as usize != grid.w - 1 || first.4 as usize != grid.h - 1 {
                    return Err("obs: first record must be the full grid at frame 0".into());
                }
            }
        }
        let mut cov = vec![];
        if let Some(c) = sec("cov")? {
            let mut p = 0;
            let mut seen = vec![false; grid.w * grid.h];
            while p < c.len() {
                if p + 8 > c.len() {
                    return Err("cov: truncated record head".into());
                }
                let (f, n) = (rd_u32(c, p), rd_u32(c, p + 4) as usize);
                if p + 8 + 4 * n > c.len() {
                    return Err("cov: truncated indices".into());
                }
                if f as usize >= n_frames.max(1) {
                    return Err(format!("cov: frame {} ≥ n_frames", f));
                }
                let idx: Vec<u32> = (0..n).map(|k| rd_u32(c, p + 8 + 4 * k)).collect();
                for &i in &idx {
                    let i = i as usize;
                    if i >= seen.len() {
                        return Err(format!("cov: cell {} outside grid", i));
                    }
                    if seen[i] {
                        return Err(format!("cov: cell {} covered twice (coverage only grows)", i));
                    }
                    seen[i] = true;
                }
                cov.push((f, idx));
                p += 8 + 4 * n;
            }
        }
        let scan = match sec("scan")? {
            None => None,
            Some(s) => {
                let n = head["lidar"]["n"].as_u64().ok_or("scan section without head.lidar.n")? as usize;
                if s.len() != n_frames * n * 2 {
                    return Err(format!("scan: {} B, expected n_frames × lidar.n × 2", s.len()));
                }
                Some((n, s.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()))
            }
        };
        let x = Xrp { head, cols, n_frames, frames, grid, gt, obs, cov, scan };
        let ti = x.col("t").unwrap();
        for f in 1..x.n_frames {
            if !(x.at(f, ti) >= x.at(f - 1, ti)) {
                return Err(format!("frame {}: t goes backwards", f));
            }
        }
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// 시험용 최소 쓰개(진짜 쓰개는 runfolder.h)
    fn build(head: Value, secs: &[(&str, Vec<u8>)]) -> Vec<u8> {
        let mut h = head.clone();
        let mut off = 0usize;
        let mut sj = vec![];
        for (n, b) in secs {
            sj.push(json!({"name": n, "off": off, "len": b.len()}));
            off = pad8(off + b.len());
        }
        h["sections"] = Value::Array(sj);
        let hs = h.to_string();
        let mut out = MAGIC.to_vec();
        out.extend_from_slice(&(hs.len() as u32).to_le_bytes());
        out.extend_from_slice(hs.as_bytes());
        out.resize(pad8(out.len()), 0);
        let d0 = out.len();
        for (_, b) in secs {
            out.resize(d0 + pad8(out.len() - d0), 0);
            out.extend_from_slice(b);
        }
        out
    }

    #[test]
    fn roundtrip_and_checks() {
        let head = json!({"format": "XRP1", "dt": 0.1, "n_frames": 2, "cols": ["t", "x", "y", "yaw"], "grid": {"w": 2, "h": 2, "res": 0.1, "ox": 0.0, "oy": 0.0}});
        let fr: Vec<u8> = [0.0f32, 1.0, 2.0, 0.0, 0.1, 1.5, 2.0, 0.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let gt = vec![0u8, 1, 0, 2];
        let mut obs = vec![];
        for v in [0u32, 0, 0, 1, 1] {
            obs.extend_from_slice(&v.to_le_bytes());
        }
        obs.extend_from_slice(&[255u8, 0, 100, 255]);
        let mut cov = vec![];
        for v in [1u32, 2, 0, 2] {
            cov.extend_from_slice(&v.to_le_bytes());
        }
        let b = build(head.clone(), &[("frames", fr.clone()), ("gt", gt.clone()), ("obs", obs.clone()), ("cov", cov.clone())]);
        let x = Xrp::parse(&b).unwrap();
        assert_eq!(x.at(1, 1), 1.5);
        assert_eq!(x.obs[0].5, vec![-1, 0, 100, -1]);
        assert_eq!(x.cov[0].1, vec![0, 2]);
        // 같은 칸 두 번 덮음 → 오류
        let mut cov2 = cov.clone();
        for v in [1u32, 1, 2] {
            cov2.extend_from_slice(&v.to_le_bytes());
        }
        let b2 = build(head, &[("frames", fr), ("gt", gt), ("obs", obs), ("cov", cov2)]);
        assert!(Xrp::parse(&b2).unwrap_err().contains("twice"));
    }
}
