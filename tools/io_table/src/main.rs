//! io_table — 실행 중인 우리 노드의 입출력을 `ros2 node info` 로 읽어 docs/runtime_io.md 의 생성 칸과 견준다.
//!
//!   io_table check [--doc docs/runtime_io.md] [--all]   생성 칸마다: 그 노드가 떠 있으면 지금 입출력과 비교, 다르면 끝 코드 1.
//!                                                     안 떠 있는 노드는 건너뛰고 알림 (--all 이면 그것도 실패)
//!   io_table write [--doc docs/runtime_io.md]           떠 있는 노드의 생성 칸을 지금 입출력으로 새로 씀 (안 떠 있는 칸은 그대로)
//!
//! 생성 칸 꼴 (문서 안):
//!   <!-- io:/explore/explorer -->
//!   | 방향 | 이름 | 타입 |
//!   …
//!   <!-- /io -->
//! ROS 2 가 모든 노드에 붙이는 것(/parameter_events · /rosout · 파라미터 서비스 · 타입 설명 서비스)은 뺀다.
//! 먼저 ROS 2 환경을 source 하고(`source env.sh`) 스택을 띄운다 (docs/runtime_io.md "I/O 표 맞추기").
use std::collections::BTreeSet;
use std::process::{exit, Command};

const BEGIN: &str = "<!-- io:";
const END: &str = "<!-- /io -->";
const HEAD: &str = "| 방향 | 이름 | 타입 |\n|---|---|---|\n";

/// `ros2 node info <node>` 출력 → 표 줄 (방향, 이름, 타입) — 차례는 방향·이름 순
pub fn parse_info(text: &str, node: &str) -> Vec<(String, String, String)> {
    const PARAM_SRV: [&str; 7] = [
        "describe_parameters", "get_parameter_types", "get_parameters", "list_parameters",
        "set_parameters", "set_parameters_atomically", "get_type_description",
    ];
    let mut kind = "";
    let mut rows = BTreeSet::new();
    for line in text.lines() {
        let t = line.trim();
        let k = match t {
            "Subscribers:" => Some("1 받음"),
            "Publishers:" => Some("2 냄"),
            "Service Servers:" => Some("3 서비스"),
            "Service Clients:" => Some("4 서비스 부름"),
            "Action Servers:" => Some("5 액션"),
            "Action Clients:" => Some("6 액션 부름"),
            _ => None,
        };
        if let Some(k) = k {
            kind = k;
            continue;
        }
        if kind.is_empty() || !line.starts_with("    ") {
            continue;
        }
        let Some((name, ty)) = t.split_once(": ") else { continue };
        if name == "/parameter_events" || name == "/rosout" {
            continue;
        }
        if let Some(rest) = name.strip_prefix(node).and_then(|r| r.strip_prefix('/')) {
            if PARAM_SRV.contains(&rest) {
                continue;
            }
        }
        rows.insert((kind.to_string(), name.to_string(), ty.to_string()));
    }
    rows.into_iter().map(|(k, n, t)| (k[2..].to_string(), n, t)).collect()
}

pub fn render(rows: &[(String, String, String)]) -> String {
    let mut s = String::from(HEAD);
    for (k, n, t) in rows {
        s.push_str(&format!("| {k} | `{n}` | `{t}` |\n"));
    }
    s
}

/// 문서의 생성 칸들: (노드 이름, 몸 시작 바이트, 몸 끝 바이트)
pub fn blocks(doc: &str) -> Vec<(String, usize, usize)> {
    let mut out = Vec::new();
    let mut at = 0;
    while let Some(b) = doc[at..].find(BEGIN) {
        let b = at + b;
        let Some(close) = doc[b..].find("-->") else { break };
        let node = doc[b + BEGIN.len()..b + close].trim().to_string();
        let body = b + close + 3;
        let body = if doc[body..].starts_with('\n') { body + 1 } else { body };
        let Some(e) = doc[body..].find(END) else { break };
        out.push((node, body, body + e));
        at = body + e + END.len();
    }
    out
}

fn running_nodes() -> BTreeSet<String> {
    let o = Command::new("ros2").args(["node", "list"]).output();
    match o {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).lines().map(|l| l.trim().to_string()).collect(),
        _ => {
            eprintln!("io_table: `ros2 node list` 실패 — ROS 2 환경을 source 했나 (source env.sh)");
            exit(2);
        }
    }
}

fn node_info(node: &str) -> Option<String> {
    let o = Command::new("ros2").args(["node", "info", node]).output().ok()?;
    o.status.success().then(|| String::from_utf8_lossy(&o.stdout).into_owned())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(String::as_str).unwrap_or("");
    let mut doc_path = "docs/runtime_io.md".to_string();
    let mut all = false;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--doc" if i + 1 < args.len() => { doc_path = args[i + 1].clone(); i += 1; }
            "--all" => all = true,
            a => { eprintln!("io_table: 모르는 인자 {a}"); exit(2); }
        }
        i += 1;
    }
    if cmd != "check" && cmd != "write" {
        eprintln!("io_table check|write [--doc docs/runtime_io.md] [--all]");
        exit(2);
    }
    let doc = std::fs::read_to_string(&doc_path).unwrap_or_else(|e| { eprintln!("io_table: {doc_path}: {e}"); exit(2) });
    let live = running_nodes();
    let mut out = String::new();
    let mut last = 0;
    let (mut n_same, mut n_diff, mut n_skip) = (0, 0, 0);
    for (node, b, e) in blocks(&doc) {
        out.push_str(&doc[last..b]);
        last = e;
        let info = if live.contains(&node) { node_info(&node) } else { None };
        let Some(info) = info else {
            println!("건너뜀 {node}: 안 떠 있음");
            n_skip += 1;
            out.push_str(&doc[b..e]);
            continue;
        };
        let now = render(&parse_info(&info, &node));
        if now == doc[b..e] {
            println!("같음 {node}");
            n_same += 1;
        } else {
            println!("다름 {node}{}", if cmd == "write" { " — 새로 씀" } else { "" });
            if cmd == "check" {
                for l in now.lines().filter(|l| !doc[b..e].lines().any(|d| d == *l)) { println!("  + {l}"); }
                for l in doc[b..e].lines().filter(|d| !now.lines().any(|l| l == *d)) { println!("  - {l}"); }
            }
            n_diff += 1;
        }
        out.push_str(if cmd == "write" { &now } else { &doc[b..e] });
    }
    out.push_str(&doc[last..]);
    println!("io_table {cmd}: 같음 {n_same} · 다름 {n_diff} · 건너뜀 {n_skip}");
    if cmd == "write" {
        if out != doc { std::fs::write(&doc_path, out).unwrap_or_else(|e| { eprintln!("io_table: {doc_path}: {e}"); exit(2) }); }
        return;
    }
    if n_diff > 0 || (all && n_skip > 0) { exit(1); }
}

#[cfg(test)]
mod tests {
    use super::*;
    const INFO: &str = "/explore/explorer\n  Subscribers:\n    /parameter_events: rcl_interfaces/msg/ParameterEvent\n    /scan: sensor_msgs/msg/LaserScan\n  Publishers:\n    /explore/status: std_msgs/msg/String\n    /rosout: rcl_interfaces/msg/Log\n  Service Servers:\n    /explore/enable: std_srvs/srv/SetBool\n    /explore/explorer/get_parameters: rcl_interfaces/srv/GetParameters\n  Service Clients:\n\n  Action Servers:\n\n  Action Clients:\n\n";
    #[test]
    fn parse_filters_ros_builtin() {
        let r = parse_info(INFO, "/explore/explorer");
        assert_eq!(r.len(), 3);
        assert_eq!(r[0], ("받음".into(), "/scan".into(), "sensor_msgs/msg/LaserScan".into()));
        assert_eq!(r[2].0, "서비스");
    }
    #[test]
    fn blocks_found_and_replaced() {
        let doc = "a\n<!-- io:/explore/explorer -->\nold\n<!-- /io -->\nb\n";
        let bl = blocks(doc);
        assert_eq!(bl.len(), 1);
        assert_eq!(bl[0].0, "/explore/explorer");
        assert_eq!(&doc[bl[0].1..bl[0].2], "old\n");
    }
}
