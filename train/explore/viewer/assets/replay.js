// Replay 탭 — 탐사 판 하나를 2D 로(.xrp, 형식은 trainfmt/src/xrp.rs). 의존 없음, 캔버스 직접.
// 정답 맵 · 시간에 따라 쌓이는 관측 지도 · 덮인 칸 · 참/지도 자세 궤적 · 라이다 빔 · 걸음마다 행동·선언 확률·보상 항 · 끝 사유 · 시간 막대.
// 실행이 바뀌면 화면부터 비운다(앞 실행의 판이 남아 보이지 않게).
import { fmt, esc, color, cssv } from "./charts.js";

const $ = id => document.getElementById(id);
const OUT_COLOR = { success: "--good", declared_short: "--warning", collision: "--critical", truncated: "--muted" };
const OUT_WORD = { success: "success (declared ≥ threshold)", declared_short: "declared short", collision: "collision", truncated: "truncated (safety time)" };
const EV = { COLLISION: 1, DECLARE: 2, TRUNCATED: 4, REACHED95: 8 };
const SNAP = 50;   // 관측 지도 사본 간격(프레임)

// ------------------------------------------------------------------ .xrp 읽기
export function parseXrp(buf) {
  const dv = new DataView(buf), u8 = new Uint8Array(buf);
  if (String.fromCharCode(u8[0], u8[1], u8[2], u8[3]) !== "XRP1") throw new Error("not an XRP1 file");
  const hl = dv.getUint32(4, true);
  const head = JSON.parse(new TextDecoder().decode(u8.subarray(8, 8 + hl)));
  const d0 = (8 + hl + 7) & ~7;
  const sec = n => { const s = (head.sections || []).find(x => x.name === n); return s ? { off: d0 + s.off, len: s.len } : null; };
  const cols = head.cols, nc = cols.length, n = head.n_frames, g = head.grid, W = g.w, H = g.h;
  const fr = sec("frames"); if (!fr || fr.len !== n * nc * 4) throw new Error("frames section size mismatch");
  const frames = new Float32Array(buf, fr.off, n * nc);
  const gs = sec("gt"); if (!gs || gs.len !== W * H) throw new Error("gt section size mismatch");
  const gt = new Uint8Array(buf, gs.off, W * H);
  // 관측 지도 기록 → 사본(SNAP 프레임마다) + 기록 목록
  const recs = [];
  const os = sec("obs");
  if (os) {
    let p = os.off; const end = os.off + os.len;
    while (p + 20 <= end) {
      const f = dv.getUint32(p, true), x0 = dv.getInt32(p + 4, true), y0 = dv.getInt32(p + 8, true), x1 = dv.getInt32(p + 12, true), y1 = dv.getInt32(p + 16, true);
      const cnt = (x1 - x0 + 1) * (y1 - y0 + 1);
      recs.push({ f, x0, y0, x1, y1, off: p + 20 });
      p += 20 + cnt;
    }
  }
  const i8 = new Int8Array(buf);
  const apply = (grid, r) => { let k = r.off; for (let j = r.y0; j <= r.y1; j++) { const row = j * W; for (let i = r.x0; i <= r.x1; i++) grid[row + i] = i8[k++]; } };
  const snaps = [], snapRec = [];   // snaps[k] = 프레임 k·SNAP 까지 적용한 지도, snapRec[k] = 다음에 적용할 기록 번호
  { const cur = new Int8Array(W * H).fill(-1); let ri = 0;
    for (let k = 0; k * SNAP < Math.max(1, n); k++) {
      while (ri < recs.length && recs[ri].f <= k * SNAP) apply(cur, recs[ri++]);
      snaps.push(cur.slice()); snapRec.push(ri);
    } }
  // 덮음: 칸마다 처음 덮인 프레임(−1 = 끝까지 안 덮임)
  const covAt = new Int32Array(W * H).fill(-1);
  const cs = sec("cov");
  if (cs) { let p = cs.off; const end = cs.off + cs.len; while (p + 8 <= end) { const f = dv.getUint32(p, true), m = dv.getUint32(p + 4, true); for (let k = 0; k < m; k++) { const c = dv.getUint32(p + 8 + 4 * k, true); if (c < covAt.length && covAt[c] < 0) covAt[c] = f; } p += 8 + 4 * m; } }
  const ss = sec("scan"), L = head.lidar;
  const scan = ss && L ? new Uint16Array(buf, ss.off, n * L.n) : null;
  const ci = {}; cols.forEach((c, i) => ci[c] = i);
  const x = {
    head, cols, ci, n, nc, frames, gt, W, H, g, recs, snaps, snapRec, covAt, scan, lidar: L, size: buf.byteLength,
    v: (f, c) => ci[c] == null ? null : frames[f * nc + ci[c]],
    has: c => ci[c] != null,
    obsAt(f) {   // 프레임 f 의 관측 지도(새 배열)
      const k = Math.min(snaps.length - 1, Math.floor(f / SNAP)), g2 = snaps[k].slice();
      for (let ri = snapRec[k]; ri < recs.length && recs[ri].f <= f; ri++) apply(g2, recs[ri]);
      return g2;
    },
  };
  x.terms = head.reward_terms || cols.filter(c => c.startsWith("r_")).map(c => c.slice(2));
  return x;
}

// ------------------------------------------------------------------ 탭
export class Replay {
  constructor({ api, getCursorIter }) {
    this.api = api; this.getCursorIter = getCursorIter;
    this.run = null; this.meta = null; this.rows = []; this.cur = null; this.X = null; this.f = 0;
    this.playing = false; this.visible = false; this.view = null; this.drag = null; this.mapCache = { f: -1 };
    this.cv = $("rp_canvas"); this.strip = $("rp_strip");
    this.off = document.createElement("canvas");
    $("rp_stream").onchange = () => this.refreshList();
    $("rp_outcome").onchange = () => this.renderList();
    $("rp_map").onchange = () => this.renderList();
    $("rp_near").onchange = () => this.renderList();
    $("rp_listbtn").onclick = () => { $("tab_replay").classList.toggle("nolist"); this.resize(); };
    $("rp_eprev").onclick = () => this.step_ep(-1);
    $("rp_enext").onclick = () => this.step_ep(1);
    $("rp_layerbtn").onclick = e => { e.stopPropagation(); $("rp_layers").hidden = !$("rp_layers").hidden; };
    document.addEventListener("click", e => { if (!$("rp_layers").hidden && !$("rp_layers").contains(e.target) && e.target !== $("rp_layerbtn")) $("rp_layers").hidden = true; });
    $("rp_layers").querySelectorAll("input").forEach(i => i.onchange = () => { this.mapCache = {}; if (i.id === "ly_follow" && !i.checked) this.view = null; this.draw(); });
    $("rp_play").onclick = () => this.toggle();
    $("rp_prev").onclick = () => this.seek(this.f - 1);
    $("rp_next").onclick = () => this.seek(this.f + 1);
    $("rp_time").oninput = e => this.seek(+e.target.value);
    $("rp_list").onclick = e => { const it = e.target.closest(".item"); if (it) this.open(it.dataset.file); };
    this.strip.onclick = e => { if (!this.X) return; const r = this.strip.getBoundingClientRect(); this.seek(Math.round((e.clientX - r.left - 4) / (r.width - 8) * (this.X.n - 1))); };
    // 끌어서 옮기기, 휠로 확대
    this.cv.addEventListener("wheel", e => { if (!this.X) return; e.preventDefault(); this.ensureView(); this.userView = true; const r = this.cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top, k = Math.exp(-e.deltaY * 0.0015); const v = this.view; v.ox = mx - (mx - v.ox) * k; v.oy = my - (my - v.oy) * k; v.s *= k; this.draw(); }, { passive: false });
    this.cv.addEventListener("mousedown", e => { this.ensureView(); this.userView = true; this.drag = { x: e.clientX, y: e.clientY, ox: this.view.ox, oy: this.view.oy }; });
    addEventListener("mouseup", () => this.drag = null);
    addEventListener("mousemove", e => { if (!this.drag) return; this.view.ox = this.drag.ox + e.clientX - this.drag.x; this.view.oy = this.drag.oy + e.clientY - this.drag.y; $("ly_follow").checked = false; this.draw(); });
    this.cv.addEventListener("dblclick", () => { this.view = null; this.userView = false; this.draw(); });
    addEventListener("keydown", e => {
      if (!this.visible || /INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName) && document.activeElement.type !== "range") return;
      const big = e.shiftKey && this.X ? Math.round(1 / (this.X.head.dt || 0.1)) : 1;
      if (e.key === " ") { this.toggle(); e.preventDefault(); }
      else if (e.key === "ArrowLeft") { this.seek(this.f - big); e.preventDefault(); }
      else if (e.key === "ArrowRight") { this.seek(this.f + big); e.preventDefault(); }
      else if (e.key === "ArrowUp") { this.step_ep(-1); e.preventDefault(); }
      else if (e.key === "ArrowDown") { this.step_ep(1); e.preventDefault(); }
    });
    // 지도 칸 크기가 바뀌면(창 크기, HUD 가 나타남, 목록 숨김) 다시 재고, 사람이 옮기지 않았으면 다시 맞춘다
    new ResizeObserver(() => { if (!this.userView) this.view = null; this.resize(); }).observe($("rp_view"));
    new ResizeObserver(() => this.resize()).observe(this.strip);
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { this.mapCache = {}; this.draw(); });
  }

  // ---------------------------------------------------------------- 실행 · 목록
  setRun(id, meta, streams) {
    this.run = id; this.meta = meta; this.rows = []; this.close();
    $("rp_stream").innerHTML = streams.map(s => `<option>${esc(s)}</option>`).join("");
    this.refreshList();
  }
  close() {
    this.pause(); this.cur = null; this.X = null; this.f = 0; this.view = null; this.mapCache = { f: -1 };
    $("rp_empty").hidden = false; $("rp_title").textContent = ""; $("rp_obadge").hidden = true; $("rp_hud").innerHTML = ""; $("rp_marks").innerHTML = ""; $("rp_legend").innerHTML = "";
    $("rp_time").max = 0; $("rp_tlabel").textContent = "";
    const g = this.cv.getContext("2d"); g.clearRect(0, 0, this.cv.width, this.cv.height);
    const s = this.strip.getContext("2d"); s.clearRect(0, 0, this.strip.width, this.strip.height);
  }
  async refreshList() {
    if (!this.run) return;
    const id = this.run, st = $("rp_stream").value || "main";
    const j = await this.api("/api/replays", { run: id, stream: st });
    if (this.run !== id) return;
    this.rows = j.rows || [];
    this.why = j.why; this.bad = j.bad || [];
    for (const [sel, key] of [["rp_outcome", "outcome"], ["rp_map", "map"]]) {
      const cur = $(sel).value, vals = [...new Set(this.rows.map(r => r.meta && r.meta[key]).filter(Boolean))].sort();
      $(sel).innerHTML = ['<option value="">all</option>'].concat(vals.map(v => `<option${v === cur ? " selected" : ""}>${esc(v)}</option>`)).join("");
    }
    this.renderList();
  }
  shown() {
    const oc = $("rp_outcome").value, mp = $("rp_map").value, near = $("rp_near").checked, ci = this.getCursorIter();
    const maxIt = Math.max(1, ...this.rows.map(r => (r.meta && r.meta.iter) || 0));
    return this.rows.filter(r => { const m = r.meta || {}; return (!oc || m.outcome === oc) && (!mp || m.map === mp) && (!near || ci == null || Math.abs((m.iter ?? 0) - ci) <= Math.max(1, 0.05 * maxIt)); });
  }
  renderList() {
    const rows = this.shown(), ci = this.getCursorIter();
    $("rp_note").textContent = this.rows.length ? `${rows.length} of ${this.rows.length} episodes${$("rp_near").checked ? (ci == null ? " (cursor at latest — near-cursor filter off)" : ` near iteration ${fmt(ci)}`) : ""}${this.bad.length ? ` · ${this.bad.length} unreadable files` : ""}` : (this.why || "no replays");
    $("rp_list").innerHTML = rows.map(r => {
      const m = r.meta || {}, oc = m.outcome || "?";
      return `<div class="item${r.file === this.cur ? " on" : ""}" data-file="${esc(r.file)}" title="${esc(r.file)}\nmap ${esc(m.map || "?")} · stage ${esc(m.stage ?? "?")}\n${(r.size / 1e3).toFixed(0)} kB"><span class="mono">${esc(String(m.ep ?? "?"))}</span><span class="mono">${fmt(m.iter)}</span><span class="oc" style="color:var(${OUT_COLOR[oc] || "--ink2"})">${esc(oc.replace("_", " "))}</span><span class="mono">${m.coverage != null ? (m.coverage * 100).toFixed(0) + "%" : "—"}</span><span class="mono">${m.t != null ? fmt(m.t) + " s" : "—"}</span></div>`;
    }).join("");
  }
  step_ep(d) {
    const rows = this.shown(); if (!rows.length) return;
    const i = rows.findIndex(r => r.file === this.cur);
    const n = rows[Math.max(0, Math.min(rows.length - 1, i < 0 ? 0 : i + d))];
    if (n) this.open(n.file);
  }

  // ---------------------------------------------------------------- 판 열기
  async open(file) {
    const id = this.run, st = $("rp_stream").value || "main";
    this.pause();
    this.cur = file; this.renderList();
    let X;
    try {
      const r = await fetch(`/api/replay?run=${encodeURIComponent(id)}&stream=${encodeURIComponent(st)}&id=${encodeURIComponent(file)}`);
      if (!r.ok) throw new Error((await r.json()).why || r.status);
      X = parseXrp(await r.arrayBuffer());
    } catch (e) {
      if (this.run !== id) return;
      this.close(); this.cur = file; $("rp_empty").hidden = false; $("rp_empty").textContent = `cannot read ${file}: ${e.message}`; return;
    }
    if (this.run !== id || this.cur !== file) return;
    this.X = X; this.view = null; this.userView = false; this.mapCache = { f: -1 };
    $("rp_empty").hidden = true;
    const m = X.head.meta || {}, oc = m.outcome || "?";
    $("rp_title").textContent = `ep ${m.ep ?? "?"} · ${m.map || "?"} · iter ${fmt(m.iter)}`;
    const ob = $("rp_obadge"); ob.hidden = false; ob.className = "obadge " + oc; ob.textContent = OUT_WORD[oc] || oc;
    $("rp_time").max = X.n - 1;
    // 사건 눈금
    const evc = X.ci.ev, marks = [];
    if (evc != null) for (let f = 0; f < X.n; f++) {
      const e = X.frames[f * X.nc + evc];
      if (e & EV.COLLISION) marks.push([f, "--critical", "collision"]);
      if (e & EV.DECLARE) marks.push([f, "--accent", "declare"]);
      if (e & EV.TRUNCATED) marks.push([f, "--muted", "truncated"]);
      if (e & EV.REACHED95) marks.push([f, "--good", "95 % coverage"]);
    }
    $("rp_marks").innerHTML = marks.map(([f, c, t]) => `<b style="left:${(f / Math.max(1, X.n - 1) * 100).toFixed(2)}%;border-bottom-color:var(${c})" title="${t} @ ${fmt(X.v(f, "t"))} s" data-f="${f}"></b>`).join("");
    $("rp_marks").querySelectorAll("b").forEach(b => b.onclick = () => this.seek(+b.dataset.f));
    $("rp_legend").innerHTML = X.terms.map((t, i) => `<span><i style="background:${color(i)}"></i>r_${esc(t)}</span>`).join("") +
      `<span class="sep">|</span><span><i style="background:var(--ink)"></i>return</span><span><i style="background:var(--s7)"></i>value</span><span class="sep">|</span><span><i style="background:var(--s3)"></i>coverage</span><span><i style="background:var(--s2)"></i>p(declare)</span>`;
    this.resize();
    this.seek(0);
  }

  // ---------------------------------------------------------------- 재생
  show(on) { this.visible = on; if (on) { this.refreshList(); this.resize(); } else this.pause(); }
  toggle() { this.playing ? this.pause() : this.play(); }
  play() {
    if (!this.X) return;
    if (this.f >= this.X.n - 1) this.seek(0);
    this.playing = true; $("rp_play").textContent = "❚❚";
    let last = performance.now(), acc = 0;
    const tick = now => {
      if (!this.playing) return;
      acc += (now - last) / 1000 * (+$("rp_speed").value); last = now;
      const dt = this.X.head.dt || 0.1, k = Math.floor(acc / dt);
      if (k > 0) {
        acc -= k * dt;
        let f = this.f + k;
        if (f >= this.X.n - 1) { if ($("rp_loop").checked) f = 0; else { this.seek(this.X.n - 1); this.pause(); return; } }
        this.seek(f);
      }
      this.raf = requestAnimationFrame(tick);
    };
    this.raf = requestAnimationFrame(tick);
  }
  pause() { this.playing = false; cancelAnimationFrame(this.raf); $("rp_play").textContent = "▶"; }
  seek(f) {
    if (!this.X) return;
    this.f = Math.max(0, Math.min(this.X.n - 1, f | 0));
    $("rp_time").value = this.f;
    $("rp_tlabel").textContent = `${fmt(this.X.v(this.f, "t"))} / ${fmt(this.X.v(this.X.n - 1, "t"))} s · step ${this.f + 1}/${this.X.n}`;
    this.draw(); this.hud(); this.drawStrip();
  }

  // ---------------------------------------------------------------- 그리기
  resize() {
    if (!this.visible) return;
    const dpr = Math.min(devicePixelRatio || 1, 2);
    for (const c of [this.cv, this.strip]) { const w = c.clientWidth, h = c.clientHeight; if (w && h) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); } }
    if (this.X) { this.draw(); this.drawStrip(); }
  }
  ensureView() {
    if (this.view || !this.X) return;
    const W = this.cv.clientWidth, H = this.cv.clientHeight, g = this.X.g;
    const s = Math.min((W - 20) / g.w, (H - 20) / g.h);   // 화면 px / 칸
    this.view = { s, ox: (W - g.w * s) / 2, oy: (H - g.h * s) / 2 };
  }
  // 세계(m) → 화면(px). 칸 행 0 = 작은 y 이므로 위아래를 뒤집는다(y 위로).
  toScreen(x, y) { const v = this.view, g = this.X.g; return [v.ox + (x - g.ox) / g.res * v.s, v.oy + (g.h - (y - g.oy) / g.res) * v.s]; }
  mapImage() {
    const X = this.X, f = this.f, key = f + "|" + ["ly_gt", "ly_obs", "ly_cov"].map(i => $(i).checked ? 1 : 0).join("");
    if (this.mapCache.key === key) return this.off;
    const W = X.W, H = X.H, off = this.off; off.width = W; off.height = H;
    const g = off.getContext("2d"), img = g.createImageData(W, H), d = img.data;
    const rgb = n => { const c = cssv(n), m = c.match(/^#([0-9a-f]{6})$/i); return m ? [parseInt(m[1].slice(0, 2), 16), parseInt(m[1].slice(2, 4), 16), parseInt(m[1].slice(4, 6), 16)] : [128, 128, 128]; };
    const dark = matchMedia("(prefers-color-scheme: dark)").matches;
    const bg = rgb("--bg"), wall = dark ? [150, 150, 145] : [70, 70, 68], free = dark ? [48, 48, 46] : [236, 236, 232], cov = rgb("--s3");
    const showGt = $("ly_gt").checked, showObs = $("ly_obs").checked, showCov = $("ly_cov").checked;
    const obs = showObs ? X.obsAt(f) : null;
    for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
      const c = j * W + i, p = ((H - 1 - j) * W + i) * 4;   // 행 0(작은 y)을 그림 아래로
      let col = bg, a = 255;
      const gv = X.gt[c];
      if (showGt) col = gv === 1 ? wall : gv === 0 ? free : bg;
      if (obs) {
        const o = obs[c];
        if (o < 0) { if (showGt && gv !== 2) col = col.map((v, k) => v * 0.45 + bg[k] * 0.55); }    // 모름: 정답 맵을 흐리게
        else { const t = o / 100, fo = dark ? [70, 74, 80] : [255, 255, 255], oc = dark ? [235, 235, 230] : [20, 20, 20]; col = fo.map((v, k) => v * (1 - t) + oc[k] * t); }
      }
      if (showCov && X.covAt[c] >= 0 && X.covAt[c] <= f) col = col.map((v, k) => v * 0.55 + cov[k] * 0.45);
      d[p] = col[0]; d[p + 1] = col[1]; d[p + 2] = col[2]; d[p + 3] = a;
    }
    g.putImageData(img, 0, 0);
    this.mapCache.key = key;
    return off;
  }
  draw() {
    const X = this.X; if (!X || !this.visible) return;
    const cv = this.cv, dpr = cv.width / Math.max(1, cv.clientWidth), g = cv.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = cssv("--surface"); g.fillRect(0, 0, cv.clientWidth, cv.clientHeight);
    this.ensureView();
    const f = this.f, x = X.v(f, "x"), y = X.v(f, "y"), yaw = X.v(f, "yaw");
    if ($("ly_follow").checked) { const [sx, sy] = this.toScreen(x, y); this.view.ox += cv.clientWidth / 2 - sx; this.view.oy += cv.clientHeight / 2 - sy; }
    const v = this.view;
    g.imageSmoothingEnabled = false;
    g.drawImage(this.mapImage(), v.ox, v.oy, X.W * v.s, X.H * v.s);
    // 라이다 빔
    if ($("ly_scan").checked && X.scan) {
      const L = X.lidar, base = f * L.n;
      g.lineWidth = 1;
      const [rx, ry] = this.toScreen(x, y);
      g.strokeStyle = "rgba(235,104,52,0.28)"; g.beginPath();
      const hits = [];
      for (let k = 0; k < L.n; k++) {
        const mm = X.scan[base + k], a = yaw + L.a0 + k * L.da, r = mm === 0xFFFF ? L.rmax : mm / 1000;
        const [hx, hy] = this.toScreen(x + r * Math.cos(a), y + r * Math.sin(a));
        g.moveTo(rx, ry); g.lineTo(hx, hy);
        if (mm !== 0xFFFF) hits.push([hx, hy]);
      }
      g.stroke();
      g.fillStyle = cssv("--s2"); for (const [hx, hy] of hits) g.fillRect(hx - 1.2, hy - 1.2, 2.4, 2.4);
    }
    // 궤적
    const path = (cx, cy, css, dash) => {
      if (!X.has(cx)) return;
      g.strokeStyle = cssv(css); g.lineWidth = 2; g.setLineDash(dash ? [5, 4] : []); g.beginPath();
      for (let k = 0; k <= f; k++) { const [px, py] = this.toScreen(X.v(k, cx), X.v(k, cy)); k ? g.lineTo(px, py) : g.moveTo(px, py); }
      g.stroke(); g.setLineDash([]);
    };
    if ($("ly_path").checked) path("x", "y", "--s1", false);
    if ($("ly_spath").checked) path("sx", "sy", "--s4", true);
    // 시작점
    const [s0x, s0y] = this.toScreen(X.v(0, "x"), X.v(0, "y"));
    g.strokeStyle = cssv("--ink2"); g.lineWidth = 1.5; g.beginPath(); g.arc(s0x, s0y, 5, 0, 7); g.stroke();
    // 로봇(반지름 0.2 m 원 + 방향)
    const [rx, ry] = this.toScreen(x, y), rr = Math.max(4, 0.2 / X.g.res * v.s);
    const ev = X.v(f, "ev") || 0;
    g.fillStyle = ev & EV.COLLISION ? cssv("--critical") : cssv("--accent"); g.globalAlpha = 0.85;
    g.beginPath(); g.arc(rx, ry, rr, 0, 7); g.fill(); g.globalAlpha = 1;
    g.strokeStyle = cssv("--surface"); g.lineWidth = 2; g.beginPath(); g.moveTo(rx, ry); g.lineTo(rx + Math.cos(yaw) * rr * 1.6, ry - Math.sin(yaw) * rr * 1.6); g.stroke();
    // 행동 화살표(지령 v, vy — 로봇 기준 → 세계)
    if (X.has("a_v")) {
      const av = X.v(f, "a_v"), avy = X.v(f, "a_vy") || 0, k = 1.0 / X.g.res * v.s;
      const wx = av * Math.cos(yaw) - avy * Math.sin(yaw), wy = av * Math.sin(yaw) + avy * Math.cos(yaw);
      g.strokeStyle = cssv("--s5"); g.lineWidth = 2.5; g.beginPath(); g.moveTo(rx, ry); g.lineTo(rx + wx * k, ry - wy * k); g.stroke();
    }
    // 축척 막대
    const m1 = 1 / X.g.res * v.s, bx = 14, by = cv.clientHeight - 14;
    g.strokeStyle = cssv("--ink"); g.lineWidth = 2; g.beginPath(); g.moveTo(bx, by); g.lineTo(bx + m1, by); g.stroke();
    g.fillStyle = cssv("--ink"); g.font = "11px system-ui, sans-serif"; g.fillText("1 m", bx, by - 5);
  }

  hud() {
    const X = this.X, f = this.f, m = X.head.meta || {}, lim = X.head.limits || (this.meta && this.meta.action_limits) || { v: 1, vy: 1, w: 3.14 };
    const val = c => X.v(f, c);
    const bar = (name, v, max, unit, ko) => {
      if (v == null) return "";
      const p = Math.max(-1, Math.min(1, v / max));
      return `<div class="abar" title="${esc(ko)}"><span>${name}</span><span class="ab"><i style="left:${p < 0 ? 50 + p * 50 : 50}%;width:${Math.abs(p) * 50}%"></i></span><span class="mono">${fmt(v)} ${unit}</span></div>`;
    };
    const pbar = (name, v, ko, css) => v == null ? "" : `<div class="abar" title="${esc(ko)}"><span>${name}</span><span class="ab p"><i style="left:0;width:${(Math.max(0, Math.min(1, v)) * 100).toFixed(1)}%;background:var(${css})"></i></span><span class="mono">${fmt(v, 1)}</span></div>`;
    let terms = "", rs = 0;
    for (const [i, t] of X.terms.entries()) { const r = val("r_" + t); if (r == null) continue; rs += r; terms += `<tr><td><i class="sw" style="background:${color(i)}"></i>${esc(t)}</td><td class="mono">${fmt(r)}</td></tr>`; }
    const ev = val("ev") || 0, evs = [ev & EV.COLLISION && "collision", ev & EV.DECLARE && "declare", ev & EV.TRUNCATED && "truncated", ev & EV.REACHED95 && "reached 95 %"].filter(Boolean);
    const thr = this.meta && this.meta.success_coverage;
    $("rp_hud").innerHTML = `
      <div class="hh"><b>t ${fmt(val("t"))} s</b> <span class="muted">step ${f + 1}/${X.n}</span></div>
      <div class="muted small">map ${esc(m.map || "?")} · stage ${esc(m.stage ?? "?")} · iter ${fmt(m.iter)} · area ${fmt(m.area_m2)} m²</div>
      <div class="muted small">ends: <b style="color:var(${OUT_COLOR[m.outcome] || "--ink"})">${esc(OUT_WORD[m.outcome] || m.outcome || "?")}</b> at ${fmt(m.t)} s · final coverage ${fmt(m.coverage, 1)}${m.t95 != null ? ` · t95 ${fmt(m.t95)} s` : ""}</div>
      <h4>Map</h4>
      ${pbar("coverage", val("coverage"), "정답 맵 기준 덮은 비율(가치에만 주는 참값)" + (thr != null ? ` — success threshold ${fmt(thr, 1)}` : ""), "--s3")}
      ${val("known_m2") != null ? `<div class="kv"><span>known area</span><span class="mono">${fmt(val("known_m2"))} m²</span></div>` : ""}
      <h4>Action</h4>
      ${bar("v", val("a_v"), lim.v, "m/s", "지령 전진 속도")}
      ${bar("vy", val("a_vy"), lim.vy, "m/s", "지령 옆 속도(메카넘)")}
      ${bar("ω", val("a_w"), lim.w, "rad/s", "지령 회전 속도")}
      ${pbar("p(declare)", val("p_end"), "끝 선언 확률(베르누이)", "--s2")}
      ${val("a_end") ? '<div class="evline">declared this step</div>' : ""}
      ${X.has("v") ? `<div class="kv muted small"><span>measured v · vy · ω</span><span class="mono">${fmt(val("v"))} · ${fmt(val("vy"))} · ${fmt(val("w"))}</span></div>` : ""}
      <h4>Reward <span class="muted small">this step</span></h4>
      <table class="rt">${terms}<tr class="sum"><td>Σ step</td><td class="mono">${fmt(rs)}</td></tr><tr><td>return so far</td><td class="mono">${fmt(val("ret"))}</td></tr>${X.has("value") ? `<tr><td>value V(s)</td><td class="mono">${fmt(val("value"))}</td></tr>` : ""}</table>
      ${evs.length ? `<div class="evline">${evs.join(" · ")}</div>` : ""}
      <div class="muted small" style="margin-top:6px">${(X.size / 1e3).toFixed(0)} kB · ${X.recs.length} map records · ${X.scan ? X.lidar.n + " beams" : "no lidar"}</div>`;
  }

  drawStrip() {
    const X = this.X, cv = this.strip; if (!X || !this.visible) return;
    const dpr = cv.width / Math.max(1, cv.clientWidth), W = cv.clientWidth, H = cv.clientHeight, g = cv.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
    const L = 4, R = W - 4, n = X.n, xf = k => L + (n > 1 ? k / (n - 1) : 0) * (R - L);
    const panes = [[2, 62], [66, 106], [110, H - 2]];   // 보상 항 막대 · 리턴/가치 · 덮음/선언 확률
    g.font = "10px system-ui, sans-serif";
    // 1) 걸음마다 보상 항(쌓은 막대). 막대가 너무 많으면 칸마다 합.
    {
      const [t, b] = panes[0], mid = (t + b) / 2, bins = Math.min(n, Math.max(1, Math.floor(R - L)));
      const pos = new Float32Array(bins * X.terms.length), neg = new Float32Array(bins * X.terms.length);
      for (let k = 0; k < n; k++) { const bi = Math.min(bins - 1, Math.floor(k / n * bins)); X.terms.forEach((tn, i) => { const r = X.v(k, "r_" + tn) || 0; if (r > 0) pos[bi * X.terms.length + i] += r; else neg[bi * X.terms.length + i] += r; }); }
      // 눈금 = 칸 합 크기의 98 % 분위수 — 끝 선언·충돌처럼 한 번 큰 값에 나머지 항이 눌리지 않게. 넘는 막대는 잘라 끝에 검은 표시
      const mags = [], peak = [0, 0];
      for (let bi = 0; bi < bins; bi++) { let p = 0, q = 0; for (let i = 0; i < X.terms.length; i++) { p += pos[bi * X.terms.length + i]; q -= neg[bi * X.terms.length + i]; } mags.push(p, q); peak[0] = Math.max(peak[0], p); peak[1] = Math.max(peak[1], q); }
      mags.sort((u, w) => u - w);
      const mx = Math.max(1e-9, mags[Math.floor(mags.length * 0.98)] || 0, Math.max(peak[0], peak[1]) * 0.02);
      const half = (b - t) / 2, sc = half / mx, bw = (R - L) / bins;
      for (let bi = 0; bi < bins; bi++) {
        let up = mid, dn = mid;
        X.terms.forEach((tn, i) => {
          g.fillStyle = color(i);
          const p = Math.min(pos[bi * X.terms.length + i] * sc, up - (mid - half)), q = Math.min(-neg[bi * X.terms.length + i] * sc, mid + half - dn);
          if (p > 0) { g.fillRect(L + bi * bw, up - p, Math.max(1, bw - 0.3), p); up -= p; }
          if (q > 0) { g.fillRect(L + bi * bw, dn, Math.max(1, bw - 0.3), q); dn += q; }
        });
        g.fillStyle = cssv("--ink");
        if (up <= mid - half + 0.01) g.fillRect(L + bi * bw - 1, t, Math.max(3, bw), 2);
        if (dn >= mid + half - 0.01) g.fillRect(L + bi * bw - 1, b - 2, Math.max(3, bw), 2);
      }
      g.strokeStyle = cssv("--grid"); g.beginPath(); g.moveTo(L, mid + 0.5); g.lineTo(R, mid + 0.5); g.stroke();
      g.fillStyle = cssv("--muted"); g.fillText(`reward terms per step (scale ±${fmt(mx)}; peaks +${fmt(peak[0])} / −${fmt(peak[1])}, clipped bars marked)`, L + 2, t + 9);
    }
    const line = (pane, cols, lo, hi, label) => {
      const [t, b] = pane;
      let a = lo, z = hi;
      if (a == null) { a = Infinity; z = -Infinity; for (const [c] of cols) for (let k = 0; k < n; k++) { const v = X.v(k, c); if (v != null && isFinite(v)) { a = Math.min(a, v); z = Math.max(z, v); } } }
      if (!isFinite(a)) return;
      if (z - a < 1e-9) z = a + 1;
      const yf = v => b - (v - a) / (z - a) * (b - t);
      for (const [c, css] of cols) {
        if (!X.has(c)) continue;
        g.strokeStyle = cssv(css); g.lineWidth = 1.5; g.beginPath();
        for (let k = 0; k < n; k++) { const v = X.v(k, c); k ? g.lineTo(xf(k), yf(v)) : g.moveTo(xf(k), yf(v)); }
        g.stroke();
      }
      g.strokeStyle = cssv("--grid"); g.strokeRect(L + 0.5, t + 0.5, R - L - 1, b - t - 1);
      g.fillStyle = cssv("--muted"); g.fillText(`${label} [${fmt(a)}, ${fmt(z)}]`, L + 2, t + 9);
    };
    line(panes[1], [["ret", "--ink"], ["value", "--s7"]], null, null, "return · value");
    line(panes[2], [["coverage", "--s3"], ["p_end", "--s2"]], 0, 1, "coverage · p(declare)");
    // 지금 걸음
    const cx = Math.round(xf(this.f)) + 0.5;
    g.strokeStyle = cssv("--accent"); g.lineWidth = 1; g.beginPath(); g.moveTo(cx, 0); g.lineTo(cx, H); g.stroke();
  }
}
