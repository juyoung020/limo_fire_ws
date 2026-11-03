//! 저장소 규칙 검사 (CLAUDE.md "Git" 절을 기계로 지키게).
//!
//!   repo_check changelog [CHANGELOG.md]   형식 검사
//!   repo_check push                        푸시 전 검사: CHANGELOG + 아직 안 올린 커밋 묶음 + 기여자 표기
//!   repo_check hook                        Claude Code PreToolUse(Bash) 훅: stdin JSON 의 명령이
//!                                          git commit / git push 면 위 검사를 돌리고, 어기면 종료 코드 2 로 막는다
//!
//! 규칙 근거: https://keepachangelog.com/ko/1.1.0/ — 사람을 위한 기록, 같은 종류끼리 묶음,
//! 최신이 위, 버전마다 ISO 8601 날짜, 버전 링크, 커밋 로그를 그대로 옮기지 않음.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{exit, Command};

/// 항목 한 줄의 최대 글자 수. 커밋 로그처럼 길게 늘어지지 않게.
const MAX_ENTRY_CHARS: usize = 100;
/// 항목 하나에 넣을 수 있는 코드 표기(`…`) 수. 경로·이름 나열을 막는다.
const MAX_CODE_SPANS: usize = 1;
const TYPES: [&str; 6] = ["Added", "Changed", "Deprecated", "Removed", "Fixed", "Security"];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(String::as_str).unwrap_or("");
    let code = match cmd {
        "changelog" => {
            let p = args.get(2).cloned().unwrap_or_else(|| "CHANGELOG.md".into());
            report(check_changelog_file(Path::new(&p)))
        }
        "push" => match repo_root(Path::new(".")) {
            Some(root) => report(check_push(&root, false)),
            None => fail_msg("git 저장소가 아님"),
        },
        "hook" => hook(),
        _ => {
            eprintln!("쓰임: repo_check changelog [파일] | push | hook");
            2
        }
    };
    exit(code);
}

fn report(errs: Vec<String>) -> i32 {
    if errs.is_empty() {
        println!("repo_check: 통과");
        0
    } else {
        for e in &errs {
            eprintln!("repo_check: {e}");
        }
        2
    }
}

fn fail_msg(m: &str) -> i32 {
    eprintln!("repo_check: {m}");
    2
}

// ---------------------------------------------------------------- CHANGELOG

fn check_changelog_file(p: &Path) -> Vec<String> {
    match std::fs::read_to_string(p) {
        Ok(s) => check_changelog(&s),
        Err(e) => vec![format!("{} 를 못 읽음: {e}", p.display())],
    }
}

#[derive(Debug, PartialEq, PartialOrd, Clone, Copy)]
struct Ver(u32, u32, u32);

fn parse_ver(s: &str) -> Option<Ver> {
    let mut it = s.split('.').map(|x| x.parse::<u32>().ok());
    let v = Ver(it.next()??, it.next()??, it.next()??);
    if it.next().is_some() {
        return None;
    }
    Some(v)
}

fn valid_date(d: &str) -> bool {
    let b = d.as_bytes();
    if b.len() != 10 || b[4] != b'-' || b[7] != b'-' {
        return false;
    }
    let (y, m, dd) = (d[0..4].parse::<u32>(), d[5..7].parse::<u32>(), d[8..10].parse::<u32>());
    matches!((y, m, dd), (Ok(_), Ok(m), Ok(dd)) if (1..=12).contains(&m) && (1..=31).contains(&dd))
}

pub fn check_changelog(text: &str) -> Vec<String> {
    let mut errs = Vec::new();
    let mut lines = text.lines().enumerate().peekable();
    match lines.peek() {
        Some((_, l)) if l.trim() == "# Changelog" => {}
        _ => errs.push("1 줄: 첫 줄은 '# Changelog'".into()),
    }
    let mut versions: Vec<String> = Vec::new(); // "Unreleased" 또는 "x.y.z"
    let mut last_ver: Option<Ver> = None;
    let mut last_date: Option<String> = None;
    let mut cur_types: Vec<usize> = Vec::new(); // 이 버전에서 본 종류 순서
    let mut cur_type: Option<usize> = None;
    let mut cur_type_entries = 0usize;
    let mut cur_type_line = 0usize;
    let mut links: BTreeMap<String, ()> = BTreeMap::new();
    let mut in_links = false;

    let close_type = |errs: &mut Vec<String>, t: Option<usize>, n: usize, line: usize| {
        if let Some(t) = t {
            if n == 0 {
                errs.push(format!("{line} 줄: '### {}' 아래 항목이 없음 (빈 칸은 지운다)", TYPES[t]));
            }
        }
    };

    for (i, raw) in text.lines().enumerate() {
        let n = i + 1;
        let l = raw.trim_end();
        if let Some(rest) = l.strip_prefix("## ") {
            close_type(&mut errs, cur_type, cur_type_entries, cur_type_line);
            cur_type = None;
            cur_types.clear();
            if in_links {
                errs.push(format!("{n} 줄: 버전 링크 정의 뒤에 버전 제목이 옴 (링크는 맨 끝)"));
            }
            if rest == "[Unreleased]" {
                if !versions.is_empty() {
                    errs.push(format!("{n} 줄: [Unreleased] 는 맨 위 한 번만"));
                }
                versions.push("Unreleased".into());
                continue;
            }
            // ## [x.y.z] - YYYY-MM-DD
            let ok = rest
                .strip_prefix('[')
                .and_then(|r| r.split_once("] - "))
                .and_then(|(v, d)| parse_ver(v).map(|pv| (v.to_string(), pv, d.to_string())));
            match ok {
                Some((vs, pv, d)) => {
                    if !valid_date(&d) {
                        errs.push(format!("{n} 줄: 날짜는 ISO 8601 (YYYY-MM-DD): '{d}'"));
                    }
                    if let Some(lv) = last_ver {
                        if pv >= lv {
                            errs.push(format!("{n} 줄: 버전은 최신이 위 — {vs} 가 앞 버전보다 크거나 같음"));
                        }
                    }
                    if let Some(ld) = &last_date {
                        if d.as_str() > ld.as_str() {
                            errs.push(format!("{n} 줄: {vs} 날짜가 위 버전보다 늦음"));
                        }
                    }
                    last_ver = Some(pv);
                    last_date = Some(d);
                    versions.push(vs);
                }
                None => errs.push(format!("{n} 줄: 버전 제목은 '## [Unreleased]' 또는 '## [x.y.z] - YYYY-MM-DD'")),
            }
            continue;
        }
        if let Some(t) = l.strip_prefix("### ") {
            close_type(&mut errs, cur_type, cur_type_entries, cur_type_line);
            if versions.is_empty() {
                errs.push(format!("{n} 줄: 종류 제목이 버전 밖에 있음"));
            }
            match TYPES.iter().position(|x| *x == t) {
                Some(k) => {
                    if cur_types.contains(&k) {
                        errs.push(format!("{n} 줄: 한 버전에 '### {t}' 가 두 번"));
                    } else if cur_types.last().is_some_and(|&p| p > k) {
                        errs.push(format!(
                            "{n} 줄: 종류 순서는 Added · Changed · Deprecated · Removed · Fixed · Security ('{t}' 를 더 위로)"
                        ));
                    }
                    cur_types.push(k);
                    cur_type = Some(k);
                }
                None => {
                    errs.push(format!("{n} 줄: 종류는 {} 중 하나: '{t}'", TYPES.join(" · ")));
                    cur_type = None;
                }
            }
            cur_type_entries = 0;
            cur_type_line = n;
            continue;
        }
        if l.starts_with('[') && l.contains("]: ") {
            in_links = true;
            let name = l[1..l.find("]: ").unwrap()].to_ascii_lowercase();
            links.insert(name, ());
            continue;
        }
        if l.trim().is_empty() || versions.is_empty() {
            continue; // 머리말 문단
        }
        if let Some(entry) = l.strip_prefix("- ") {
            if cur_type.is_none() {
                errs.push(format!("{n} 줄: 항목이 '### 종류' 아래에 있지 않음"));
            }
            cur_type_entries += 1;
            let chars = entry.chars().count();
            if chars > MAX_ENTRY_CHARS {
                errs.push(format!(
                    "{n} 줄: 항목이 {chars} 자 (최대 {MAX_ENTRY_CHARS}) — 사용자가 알 변화만 한 줄로, 경로·수치·원인은 커밋 메시지와 docs 로"
                ));
            }
            let spans = entry.matches('`').count() / 2;
            if spans > MAX_CODE_SPANS {
                errs.push(format!("{n} 줄: 코드 표기 `…` 가 {spans} 개 (최대 {MAX_CODE_SPANS}) — 이름 나열 말고 무엇이 되는지"));
            }
            continue;
        }
        errs.push(format!("{n} 줄: 버전 안에는 '### 종류' 와 '- 항목' 한 줄씩만 (이어 쓴 줄·문단 금지): '{}'", short(l)));
    }
    close_type(&mut errs, cur_type, cur_type_entries, cur_type_line);

    if versions.first().map(String::as_str) != Some("Unreleased") {
        errs.push("맨 위 버전은 '## [Unreleased]'".into());
    }
    for v in &versions {
        if !links.contains_key(&v.to_ascii_lowercase()) {
            errs.push(format!("버전 링크 '[{}]: …' 가 맨 끝에 없음", v.to_ascii_lowercase()));
        }
    }
    errs
}

fn short(s: &str) -> String {
    let c: String = s.chars().take(40).collect();
    if s.chars().count() > 40 { format!("{c}…") } else { c }
}

// ---------------------------------------------------------------- 푸시 전

fn git(root: &Path, args: &[&str]) -> Option<String> {
    let o = Command::new("git").arg("-C").arg(root).args(args).output().ok()?;
    o.status.success().then(|| String::from_utf8_lossy(&o.stdout).into_owned())
}

fn repo_root(dir: &Path) -> Option<PathBuf> {
    git(dir, &["rev-parse", "--show-toplevel"]).map(|s| PathBuf::from(s.trim()))
}

/// 커밋 묶음을 가르는 "영역". 문서·변경 기록처럼 모든 작업이 건드리는 곳은 뺀다.
pub fn area(path: &str) -> Option<String> {
    const SHARED: [&str; 6] = ["CHANGELOG.md", "CLAUDE.md", "README.md", ".gitignore", "archive/README.md", "src/external/README.md"];
    if SHARED.contains(&path) || path.starts_with("docs/") || path.starts_with(".claude/") {
        return None;
    }
    let p: Vec<&str> = path.split('/').collect();
    let depth = match p[0] {
        "src" => 3,                                       // src/limo_fire/<패키지>, src/external/<것>
        "train" | "sim" | "archive" | "tools" | "firmware" => 2, // train/explore, sim/webots …
        _ => 1,
    };
    if p.len() <= depth {
        // 그 깊이의 파일 자체 (예: train/README.md) 는 상위 폴더 단위
        return Some(p[..p.len().saturating_sub(1).max(1)].join("/"));
    }
    Some(p[..depth].join("/"))
}

const FORBIDDEN_TRAILERS: [&str; 3] = ["co-authored-by: claude", "generated with [claude code]", "generated with claude code"];

fn check_push(root: &Path, allow_same_area: bool) -> Vec<String> {
    let mut errs = check_changelog_file(&root.join("CHANGELOG.md"));
    let upstream = git(root, &["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "origin/main".into());
    let Some(list) = git(root, &["rev-list", "--reverse", &format!("{upstream}..HEAD")]) else {
        return errs;
    };
    let commits: Vec<&str> = list.lines().collect();
    let me = git(root, &["config", "user.email"]).unwrap_or_default().trim().to_string();
    let mut by_area: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for c in &commits {
        let msg = git(root, &["log", "-1", "--format=%B", c]).unwrap_or_default();
        let author = git(root, &["log", "-1", "--format=%ae", c]).unwrap_or_default().trim().to_string();
        let subject = msg.lines().next().unwrap_or("").to_string();
        let lower = msg.to_lowercase();
        // 다른 사람 커밋의 메시지는 고치지 않는다 (히스토리를 다시 쓸 때 섞여 들어옴)
        if author == me && FORBIDDEN_TRAILERS.iter().any(|t| lower.contains(t)) {
            errs.push(format!("{}: 커밋 메시지에 Claude 기여자 표기 — 작성자는 사람만 (CLAUDE.md)", &c[..7]));
        }
        let files = git(root, &["diff-tree", "--no-commit-id", "--name-only", "-r", c]).unwrap_or_default();
        let mut areas: Vec<String> = files.lines().filter_map(area).collect();
        areas.sort();
        areas.dedup();
        for a in areas {
            by_area.entry(a).or_default().push(format!("{} {}", &c[..7], short(&subject)));
        }
    }
    if !allow_same_area {
        for (a, cs) in &by_area {
            if cs.len() > 1 {
                errs.push(format!(
                    "아직 안 올린 커밋 {} 개가 같은 영역 '{a}' 를 건드림 — 사용자가 맡긴 작업 하나면 한 커밋으로 합친다 \
                     (git reset --soft {upstream} 뒤 다시 커밋). 정말 다른 작업이면 사용자에게 확인받고 \
                     REPO_CHECK_SEPARATE=1 git push …\n      {}",
                    cs.len(),
                    cs.join("\n      ")
                ));
            }
        }
    }
    errs
}

// ---------------------------------------------------------------- Claude Code 훅

/// JSON 문자열 값 하나를 꺼낸다 (의존성 없이: "key":"…" 의 이스케이프만 푼다).
pub fn json_str(json: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\"");
    let mut start = json.find(&pat)? + pat.len();
    let b = json.as_bytes();
    while start < b.len() && (b[start] == b' ' || b[start] == b':') {
        start += 1;
    }
    if b.get(start) != Some(&b'"') {
        return None;
    }
    let mut out = String::new();
    let mut chars = json[start + 1..].chars();
    while let Some(c) = chars.next() {
        match c {
            '"' => return Some(out),
            '\\' => match chars.next()? {
                'n' => out.push('\n'),
                't' => out.push('\t'),
                'r' => out.push('\r'),
                'u' => {
                    let h: String = chars.by_ref().take(4).collect();
                    out.push(char::from_u32(u32::from_str_radix(&h, 16).ok()?).unwrap_or('?'));
                }
                o => out.push(o),
            },
            c => out.push(c),
        }
    }
    None
}

fn is_git(cmd: &str, sub: &str) -> bool {
    // "git push", "git -C x push", "cd a && git commit …" 등
    cmd.split(|c| c == ';' || c == '&' || c == '|' || c == '\n').any(|part| {
        let w: Vec<&str> = part.split_whitespace().filter(|x| !x.contains('=')).collect();
        let Some(gi) = w.iter().position(|x| *x == "git") else { return false };
        let mut k = gi + 1;
        while k < w.len() && w[k].starts_with('-') {
            k += if w[k] == "-C" || w[k] == "-c" { 2 } else { 1 };
        }
        w.get(k) == Some(&sub)
    })
}

fn hook() -> i32 {
    let mut input = String::new();
    if std::io::stdin().read_to_string(&mut input).is_err() {
        return 0;
    }
    let Some(cmd) = json_str(&input, "command") else { return 0 };
    let (commit, push) = (is_git(&cmd, "commit"), is_git(&cmd, "push"));
    if !commit && !push {
        return 0;
    }
    // 이 저장소(이 도구가 든 저장소)에서 도는 명령만 본다
    let cwd = json_str(&input, "cwd").unwrap_or_else(|| ".".into());
    let dir = cmd
        .split_whitespace()
        .collect::<Vec<_>>()
        .windows(2)
        .find(|w| w[0] == "-C" || w[0] == "cd")
        .map(|w| {
            let p = PathBuf::from(w[1].trim_matches(|c| c == '"' || c == '\''));
            if p.is_absolute() { p } else { Path::new(&cwd).join(p) }
        })
        .unwrap_or_else(|| PathBuf::from(&cwd));
    let Some(root) = repo_root(&dir) else { return 0 };
    if !root.join("tools/repo_check/Cargo.toml").exists() {
        return 0;
    }
    let mut errs = Vec::new();
    let lower = cmd.to_lowercase();
    if commit && FORBIDDEN_TRAILERS.iter().any(|t| lower.contains(t)) {
        errs.push("커밋 메시지에 Claude 기여자 표기 — 작성자는 사람만 (CLAUDE.md)".to_string());
    }
    if commit {
        errs.extend(check_changelog_file(&root.join("CHANGELOG.md")));
    }
    if push {
        errs.extend(check_push(&root, cmd.contains("REPO_CHECK_SEPARATE=1")));
    }
    errs.sort();
    errs.dedup();
    if errs.is_empty() {
        return 0;
    }
    eprintln!("repo_check 가 막음 ({}) — 고치고 다시:", if push { "git push" } else { "git commit" });
    for e in &errs {
        eprintln!("  - {e}");
    }
    2
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOOD: &str = "# Changelog\n\n머리말.\n\n## [Unreleased]\n\n### Added\n\n- 새 기능 `a`.\n\n### Fixed\n\n- 고침.\n\n\
                        ## [0.2.0] - 2026-10-23\n\n### Changed\n\n- 바뀜.\n\n## [0.1.0] - 2026-10-22\n\n### Added\n\n- 처음.\n\n\
                        [unreleased]: u\n[0.2.0]: u\n[0.1.0]: u\n";

    #[test]
    fn good_passes() {
        assert_eq!(check_changelog(GOOD), Vec::<String>::new());
    }

    #[test]
    fn catches_problems() {
        let bad = GOOD
            .replace("### Fixed", "### Bugfix")
            .replace("- 바뀜.", &format!("- {}", "가".repeat(120)))
            .replace("- 처음.", "- `a` `b` 둘")
            .replace("2026-10-22", "2026/10/22")
            .replace("[0.1.0]: u\n", "");
        let e = check_changelog(&bad).join("\n");
        for want in ["종류는", "120 자", "코드 표기", "ISO 8601", "버전 링크 '[0.1.0]"] {
            assert!(e.contains(want), "{want} 못 잡음:\n{e}");
        }
    }

    #[test]
    fn order_and_continuation() {
        let bad = GOOD.replace("### Added\n\n- 새 기능 `a`.\n\n### Fixed", "### Fixed\n\n- x\n\n### Added")
            .replace("- 고침.", "- 고침.\n  이어 쓴 줄");
        let e = check_changelog(&bad).join("\n");
        assert!(e.contains("종류 순서"), "{e}");
        assert!(e.contains("이어 쓴 줄"), "{e}");
    }

    #[test]
    fn areas() {
        assert_eq!(area("train/explore/sim/a.cu").as_deref(), Some("train/explore"));
        assert_eq!(area("src/limo_fire/limo_fire_base/x.cpp").as_deref(), Some("src/limo_fire/limo_fire_base"));
        assert_eq!(area("docs/a.md"), None);
        assert_eq!(area("CHANGELOG.md"), None);
        assert_eq!(area("env.sh").as_deref(), Some("env.sh"));
        assert_eq!(area("train/README.md").as_deref(), Some("train"));
    }

    #[test]
    fn git_detection_and_json() {
        assert!(is_git("cd a && git -C b push origin main", "push"));
        assert!(is_git("REPO_CHECK_SEPARATE=1 git push", "push"));
        assert!(is_git("git commit -m \"x\"", "commit"));
        assert!(!is_git("git log --grep push", "push"));
        assert!(!is_git("echo git push", "commit"));
        let j = r#"{"tool_input":{"command":"git commit -m \"a\\nb\""},"cwd":"/x"}"#;
        assert_eq!(json_str(j, "command").as_deref(), Some("git commit -m \"a\\nb\""));
        assert_eq!(json_str(j, "cwd").as_deref(), Some("/x"));
    }
}
