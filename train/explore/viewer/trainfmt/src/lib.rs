//! 탐사 RL 실행 폴더 규약 — 형식은 이 크레이트 한 곳에서 정한다. 쓰는 쪽은 `include/runfolder.h`(C++ 러너·fake_run),
//! 읽는 쪽은 trainview 서버와 `runcheck`. 둘이 어긋나면 `runcheck <실행 폴더>` 가 잡는다.
//!
//! ```text
//! <root>/latest.txt                      기본으로 열 실행 이름 (러너가 시작할 때 씀)
//! <root>/<이름>/run.json                 설정·상수 (schema 1): kind, group, seed, pid, pid_start, started, segments, git, refs …
//!               progress.jsonl           업데이트마다(≤ 1 줄/s 로 합침) 한 줄 — 모집단(그 사이 끝난 모든 판)
//!               episodes.jsonl[.N]       판마다 한 줄 — 표본(log_envs 개 환경에서 끝난 판 전부)
//!               replays/ep_<판>_<결과>.xrp 판 하나 2D 재생(바이너리, `xrp` 모듈)
//!               evals/<이름>.json         평가 표
//!               events.txt               사람이 읽는 사건(단계 바뀜·체크포인트)
//!               s_<줄기>/episodes_<줄기>.jsonl, s_<줄기>/replays/   집계에 섞으면 안 되는 판(평가 등)
//! ```
//!
//! 쓰는 쪽은 **덧붙이기**(progress, episodes, events)와 **이름 바꾸기**(run.json, evals, replays: `.tmp` 에 쓰고 rename)만 한다.
//! 재개(resume)할 때만 progress·episodes 에서 체크포인트 이터 이상 줄을 지운다(`.tmp` 후 rename).

pub mod keys;
pub mod xrp;

use std::fs;
use std::path::{Path, PathBuf};

pub const SCHEMA: i64 = 1;
/// 실행 종류 — 지금은 탐사 PPO 하나
pub const KIND_EXPLORE_PPO: &str = "explore_ppo";

pub const RUN_JSON: &str = "run.json";
pub const PROGRESS: &str = "progress.jsonl";
pub const EVENTS: &str = "events.txt";
pub const LATEST: &str = "latest.txt";
pub const EVALS_DIR: &str = "evals";
pub const REPLAYS_DIR: &str = "replays";
/// 재생 파일 확장자
pub const REPLAY_EXT: &str = ".xrp";
/// episodes 파일 하나의 상한 — 넘으면 episodes.1.jsonl, episodes.2.jsonl … (runfolder.h 와 같은 값)
pub const EPISODES_ROTATE_BYTES: u64 = 1 << 30;

/// 판 끝 사유(episodes.jsonl `outcome`, 재생 머리 meta.outcome)
pub const OUTCOMES: &[(&str, &str)] = &[
    ("success", "선언 — 기준 덮음 이상(성공)"),
    ("declared_short", "선언 — 기준 미달"),
    ("collision", "충돌(종료)"),
    ("truncated", "안전 시간(끊김, truncation)"),
];

/// `/proc/<pid>/stat` 의 22 번째 칸(부팅 뒤 시작 시각, 클럭 틱). pid 재사용을 거르는 데 쓴다.
pub fn pid_start(pid: u32) -> Option<u64> {
    let s = fs::read_to_string(format!("/proc/{}/stat", pid)).ok()?;
    // comm 은 괄호 안에 공백을 품을 수 있다: 마지막 ')' 뒤부터 센다 (3 번째 칸 = state)
    let rest = &s[s.rfind(')')? + 2..];
    rest.split_whitespace().nth(19)?.parse().ok()
}

/// 실행·줄기·파일 이름으로 받는 글자: `[A-Za-z0-9_.=-]+`
pub fn safe_name(s: &str) -> bool {
    !s.is_empty() && s != "." && s != ".." && s.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '=' | '-'))
}

/// 줄기 폴더: "main" = 실행 폴더, 그 밖 = `s_<줄기>`
pub fn stream_dir(run: &Path, stream: &str) -> PathBuf {
    if stream.is_empty() || stream == "main" { run.to_path_buf() } else { run.join(format!("s_{}", stream)) }
}

/// 줄기의 episodes 파일 이름 앞부분(…/episodes 또는 …/s_eval/episodes_eval). 뒤에 `.jsonl` 또는 `.<N>.jsonl`
pub fn episodes_base(run: &Path, stream: &str) -> PathBuf {
    if stream.is_empty() || stream == "main" { run.join("episodes") } else { stream_dir(run, stream).join(format!("episodes_{}", stream)) }
}

/// episodes 파일 `i` 번(0 = `….jsonl`, 1 = `….1.jsonl` …)
pub fn episodes_file(base: &Path, i: usize) -> PathBuf {
    let mut s = base.as_os_str().to_owned();
    if i > 0 {
        s.push(format!(".{}", i));
    }
    s.push(".jsonl");
    PathBuf::from(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn names() {
        assert!(safe_name("ep_000012_success.xrp"));
        assert!(!safe_name("../x"));
        assert!(!safe_name(".."));
        assert_eq!(episodes_file(&episodes_base(Path::new("/r"), "eval"), 2), PathBuf::from("/r/s_eval/episodes_eval.2.jsonl"));
        assert_eq!(episodes_file(&episodes_base(Path::new("/r"), "main"), 0), PathBuf::from("/r/episodes.jsonl"));
    }
    #[test]
    fn self_pid() {
        assert!(pid_start(std::process::id()).is_some());
    }
}
