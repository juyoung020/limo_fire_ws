//! ProcTHOR-10K 집(jsonl 한 줄) → FMap.
//! 방 바닥 다각형(x, z) 을 칠하고, 벽 id `wall|<방>|x0|z0|x1|z1` 의 선분을 벽으로 그린 뒤
//! 문 구멍(`holePolygon` 의 x 범위, wall0 시작점 기준)을 뚫는다. 바깥 벽(`wall|exterior|…`)에 난 문 = 비상구.
//! 가구(`objects`)는 크기 정보가 없어 아직 넣지 않는다.

use crate::{FMap, Room, RoomType, FREE, WALL};
use serde_json::Value;

const WALL_THICK: f32 = 0.10;
const DOOR_THICK: f32 = 0.16;

fn seg_of(id: &str) -> Option<((f32, f32), (f32, f32))> {
    let p: Vec<&str> = id.split('|').collect();
    if p.len() < 6 { return None; }
    let f = |s: &str| s.parse::<f32>().ok();
    Some(((f(p[2])?, f(p[3])?), (f(p[4])?, f(p[5])?)))
}

pub fn load(v: &Value) -> Option<FMap> {
    let rooms = v["rooms"].as_array()?;
    let mut polys: Vec<(RoomType, Vec<(f32, f32)>)> = Vec::new();
    let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for r in rooms {
        let poly: Vec<(f32, f32)> = r["floorPolygon"].as_array()?.iter()
            .filter_map(|p| Some((p["x"].as_f64()? as f32, p["z"].as_f64()? as f32))).collect();
        for &(x, y) in &poly { x0 = x0.min(x); y0 = y0.min(y); x1 = x1.max(x); y1 = y1.max(y); }
        polys.push((RoomType::from_name(r["roomType"].as_str().unwrap_or("")), poly));
    }
    if polys.is_empty() { return None; }
    let mut m = FMap::new(x0, y0, x1, y1, 0.5);
    for (kind, poly) in &polys {
        m.rooms.push(Room { kind: *kind, area: 0.0 });
        let rid = m.rooms.len() as u16;
        m.fill_polygon(poly, FREE, rid);
    }
    for w in v["walls"].as_array().into_iter().flatten() {
        if let Some((a, b)) = w["id"].as_str().and_then(seg_of) { m.draw_segment(a, b, WALL_THICK, WALL); }
    }
    for d in v["doors"].as_array().into_iter().flatten() {
        let Some((a, b)) = d["wall0"].as_str().and_then(seg_of) else { continue };
        let hole = match d["holePolygon"].as_array() { Some(h) if h.len() >= 2 => h, _ => continue };
        let (h0, h1) = (hole[0]["x"].as_f64().unwrap_or(0.0) as f32, hole[1]["x"].as_f64().unwrap_or(0.0) as f32);
        let len = ((b.0 - a.0).powi(2) + (b.1 - a.1).powi(2)).sqrt().max(1e-6);
        let dir = ((b.0 - a.0) / len, (b.1 - a.1) / len);
        let p = (a.0 + dir.0 * h0.min(h1), a.1 + dir.1 * h0.min(h1));
        let q = (a.0 + dir.0 * h0.max(h1), a.1 + dir.1 * h0.max(h1));
        let exterior = d["wall1"].as_str().map_or(false, |s| s.contains("exterior")) || d["wall0"].as_str().map_or(false, |s| s.contains("exterior"));
        if exterior {
            m.exits.push(((p.0 + q.0) * 0.5, (p.1 + q.1) * 0.5));
        } else {
            m.draw_segment(p, q, DOOR_THICK, FREE);
        }
    }
    Some(m)
}
