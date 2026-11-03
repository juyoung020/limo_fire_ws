//! 8 비트 P5 PGM → 회색 PNG (눈으로 확인용 — scene_list --png 가 낸 격자 그림을 이미지 보기에서 열려고).
//! pgm2png <입력.pgm> [<입력.pgm> …]   → 같은 이름 .png. 머리의 # 주석은 다루지 않는다(우리 도구가 쓴 PGM 만).

use std::io::BufWriter;

fn main() {
    for path in std::env::args().skip(1) {
        let data = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        // 머리: "P5" w h maxval, 공백으로 나뉨, 그 뒤 공백 하나 다음부터 픽셀
        let mut fields = Vec::new();
        let mut i = 0;
        while fields.len() < 4 && i < data.len() {
            while i < data.len() && data[i].is_ascii_whitespace() { i += 1; }
            let s = i;
            while i < data.len() && !data[i].is_ascii_whitespace() { i += 1; }
            fields.push(String::from_utf8_lossy(&data[s..i]).into_owned());
        }
        i += 1;
        let (w, h): (usize, usize) = (fields[1].parse().expect("폭"), fields[2].parse().expect("높이"));
        assert!(fields[0] == "P5" && fields[3] == "255" && w > 0 && h > 0 && w * h <= 1 << 26, "{path}: 8 비트 P5 PGM 아님");
        let px = data.get(i..i + w * h).unwrap_or_else(|| panic!("{path}: 잘림"));
        let out = std::path::Path::new(&path).with_extension("png");
        let f = std::fs::File::create(&out).expect("png 쓰기");
        let mut enc = png::Encoder::new(BufWriter::new(f), w as u32, h as u32);
        enc.set_color(png::ColorType::Grayscale);
        enc.set_depth(png::BitDepth::Eight);
        enc.write_header().and_then(|mut wr| wr.write_image_data(px)).expect("png 쓰기");
        println!("{}", out.display());
    }
}
