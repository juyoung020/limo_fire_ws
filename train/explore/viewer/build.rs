// 화면 파일(assets/*)을 바이너리에 넣는다 — 서버 하나만 옮기면 된다.
use std::path::Path;

fn main() {
    let dir = Path::new("assets");
    let mut names: Vec<_> = std::fs::read_dir(dir)
        .expect("assets/ missing")
        .flatten()
        .filter(|e| e.path().is_file())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();
    names.sort();
    let mut code = String::from("pub const FILES: &[(&str, &[u8])] = &[\n");
    for n in names {
        let abs = std::fs::canonicalize(dir.join(&n)).unwrap();
        code.push_str(&format!("    ({:?}, include_bytes!({:?})),\n", n, abs.to_string_lossy()));
        println!("cargo:rerun-if-changed={}", dir.join(&n).display());
    }
    println!("cargo:rerun-if-changed=assets");
    code.push_str("];\n");
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("assets.rs");
    std::fs::write(out, code).unwrap();
}
