//! HouseExpo(SUNCG 기반 2D 평면도) JSON → FMap.
//! `verts` = 집 안 빈 곳의 외곽선(안쪽 벽은 외곽선이 파고든 홈으로 들어 있다), 미터.
//! `room_category` = 방 종류 → 방 경계 상자 목록 [x0, y0, x1, y1].

use crate::{FMap, Room, RoomType, FREE, WALL};
use serde_json::Value;

pub fn load(v: &Value) -> Option<FMap> {
    let verts: Vec<(f32, f32)> = v["verts"].as_array()?.iter()
        .filter_map(|p| Some((p[0].as_f64()? as f32, p[1].as_f64()? as f32))).collect();
    if verts.len() < 3 { return None; }
    let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for &(x, y) in &verts { x0 = x0.min(x); y0 = y0.min(y); x1 = x1.max(x); y1 = y1.max(y); }
    let mut m = FMap::new(x0, y0, x1, y1, 0.5);
    m.fill_polygon(&verts, FREE, 0);
    for k in 0..verts.len() { m.draw_segment(verts[k], verts[(k + 1) % verts.len()], 2.0 * crate::RES, WALL); }

    // 방: 작은 상자부터 칠해서 큰 상자가 작은 방을 덮지 않게
    let mut boxes: Vec<(RoomType, [f32; 4])> = Vec::new();
    if let Some(obj) = v["room_category"].as_object() {
        for (name, list) in obj {
            for b in list.as_array().into_iter().flatten() {
                let b: Vec<f32> = b.as_array()?.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect();
                if b.len() == 4 { boxes.push((RoomType::from_name(name), [b[0], b[1], b[2], b[3]])); }
            }
        }
    }
    boxes.sort_by(|a, b| area(&a.1).partial_cmp(&area(&b.1)).unwrap());
    for (kind, b) in boxes {
        m.rooms.push(Room { kind, area: 0.0 });
        let rid = m.rooms.len() as u16;
        let (Some((i0, j0)), Some((i1, j1))) = (m.to_cell(b[0].max(m.origin.0), b[1].max(m.origin.1)), m.to_cell(b[2], b[3])) else { continue };
        for j in j0..=j1 { for i in i0..=i1 {
            let k = m.idx(i, j);
            if m.cell[k] == FREE && m.room[k] == 0 { m.room[k] = rid; }
        } }
    }
    Some(m)
}

fn area(b: &[f32; 4]) -> f32 { (b[2] - b[0]).abs() * (b[3] - b[1]).abs() }
