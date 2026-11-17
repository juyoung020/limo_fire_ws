// trainview 화면 — Run·Replay·Compare 탭. 순수 ES 모듈, 빌드 단계 없음.
// DOM 은 한 번 짓고 바뀐 글자만 바꾼다. 캔버스는 내용 서명이 바뀔 때만 다시 그린다. 보이지 않는 탭은 그리지 않는다.
import { Plot, ema, fmt, color, esc, hist, cssv } from "./charts.js";
import { Replay } from "./replay.js";

const $ = id => document.getElementById(id);
const enc = encodeURIComponent;
async function api(path, q = {}) {
  const u = path + "?" + Object.entries(q).filter(([, v]) => v !== undefined && v !== null).map(([k, v]) => k + "=" + enc(v)).join("&");
  const r = await fetch(u);
  return r.json();
}
function setText(el, t) { if (typeof el === "string") el = $(el); if (el && el.textContent !== t) el.textContent = t; }
function setHTML(el, h) { if (typeof el === "string") el = $(el); if (el && el._h !== h) { el.innerHTML = h; el._h = h; } }

const S = {
  runs: [], byId: {}, latest: null, sel: null, meta: null, d: null, tab: "run",
  es: null, poll: null, retry: null,
  eps: 0, rep: 0, cmpSel: new Set(), cmpData: {}, cmpKeys: {}, status: {},
};
const STOPS = [10, 20, 50, 100, 200, 500, 0];

// ---------------------------------------------------------------- 실행 상태 불 (서버 runs::status, SSE "status" 로 2 초마다)
const ST_KO = { training: "실행 중(pid 살아 있고 60 s 안에 기록)", stalled: "멈춤?(프로세스는 살아 있는데 60 s 넘게 기록 없음)", finished: "끝남(run.json ended)", crashed: "죽음(프로세스 없음, ended 없음)", unknown: "모름(run.json 에 pid 없음)" };
const ST_WORD = { training: "Running", stalled: "Stalled", finished: "Finished", crashed: "Crashed", unknown: "Status unknown" };
function stTitle(st) {
  if (!st) return "";
  const lw = st.last_write ? new Date(st.last_write * 1000).toLocaleTimeString() : "—", age = st.age != null ? (st.age < 120 ? st.age.toFixed(0) + " s" : (st.age / 60).toFixed(0) + " min") + " ago" : "";
  return `${st.state} — ${ST_KO[st.state] || ""}\npid ${st.pid ?? "—"}\nlast write ${lw} (${age})\niteration ${st.iter ?? "—"}`;
}
function stDot(id) { const st = S.status[id] || {}; return `<span class="stl ${st.state || "unknown"}" data-st="${esc(id)}" title="${esc(stTitle(st))}"></span>`; }
function applyStatus(list) {
  for (const st of list) S.status[st.id] = st;
  document.querySelectorAll("[data-st]").forEach(el => { const st = S.status[el.dataset.st]; if (!st) return; el.className = "stl " + st.state; el.title = stTitle(st); });
  document.querySelectorAll("[data-stl]").forEach(el => { const st = S.status[el.dataset.stl]; if (st) { el.textContent = ST_WORD[st.state] || st.state; el.title = stTitle(st); } });
}
const replay = new Replay({ api });

// ---------------------------------------------------------------- 실행 목록
function shortName(r) { return r.id.split("/").slice(1).join("/") || r.id; }
const pc0 = v => v == null ? "—" : (v * 100).toFixed(0) + "%";
const ageTxt = a => a == null ? "—" : a < 120 ? a.toFixed(0) + " s ago" : a < 7200 ? (a / 60).toFixed(0) + " min ago" : a < 172800 ? (a / 3600).toFixed(1) + " h ago" : (a / 86400).toFixed(0) + " d ago";
async function loadRuns() {
  let j;
  try { j = await api("/api/runs"); } catch (e) { setText("conn", "server offline"); return; }
  S.runs = j.runs; S.byId = Object.fromEntries(j.runs.map(r => [r.id, r])); S.latest = j.latest;
  for (const r of j.runs) if (r.status) S.status[r.id] = r.status;
  S.roots = (j.roots || []).map(x => x.path);
  renderRunMenu();
  renderEmpty(S.runs.length === 0);
  if (!S.sel) {
    const want = new URLSearchParams(location.hash.slice(1)).get("run");
    selectRun(want && S.byId[want] ? want : j.latest || (S.runs[0] || {}).id);
  } else updateStatus();
  buildCompareList();
}
function runRow(r) {
  const st = S.status[r.id] || {}, h = r.health || {};
  const res = h["rollout/success_rate"] != null ? `success ${pc0(h["rollout/success_rate"])}` : "";
  const cov = h["rollout/coverage_mean"] != null ? ` · cov ${pc0(h["rollout/coverage_mean"])}` : "";
  return `<div class="rm_row${r.id === S.sel ? " on" : ""}" data-id="${esc(r.id)}" title="${esc(r.id)}\n${esc(r.dir)}">${stDot(r.id)}<span class="rm_n mono">${esc(shortName(r))}</span><span class="rm_k">${esc(r.group || "")}${r.seed != null ? " · seed " + r.seed : ""}</span><span class="rm_s muted">${res}${cov}${res ? " · " : ""}${esc(ST_WORD[st.state] || "")} · ${ageTxt(r.age)}</span></div>`;
}
function renderRunMenu() {
  const q = ($("runsearch").value || "").toLowerCase().trim(), match = r => !q || (r.id + " " + (r.group || "")).toLowerCase().includes(q);
  const list = S.runs.filter(match), byAge = (a, b) => (a.age ?? 1e12) - (b.age ?? 1e12);
  const real = list.filter(r => !r.synthetic).sort(byAge), syn = list.filter(r => r.synthetic).sort(byAge);
  let h = real.length ? `<div class="rm_h">Runs <span class="muted">· newest first</span></div>${real.map(runRow).join("")}` : "";
  if (syn.length) h += `<details class="rm_d"${q || !real.length ? " open" : ""}><summary title="fake_run 이 만든 가짜 시험 자료 — 실제 학습 결과 아님">Test data (synthetic) <span class="muted">${syn.length}</span></summary>${syn.map(runRow).join("")}</details>`;
  setHTML("runlist", h || '<div class="muted small" style="padding:6px">no run matches</div>');
}
$("runbtn").onclick = e => { e.stopPropagation(); const m = $("runmenu"); m.hidden = !m.hidden; if (!m.hidden) { $("runsearch").value = ""; renderRunMenu(); $("runsearch").focus(); } };
$("runsearch").oninput = renderRunMenu;
document.addEventListener("click", e => {
  const c = e.target.closest(".rm_row"); if (c && c.dataset.id) { $("runmenu").hidden = true; selectRun(c.dataset.id); e.preventDefault(); return; }
  if (!$("runmenu").hidden && !$("runmenu").contains(e.target)) $("runmenu").hidden = true;
});
function renderEmpty(empty) {
  $("empty_state").hidden = !empty; $("run_body").hidden = empty; $("rundd").hidden = empty;
  if (!empty) return;
  const roots = (S.roots || []).map(p => `<code>${esc(p)}</code>`).join(" ") || "<code>(no --root)</code>";
  $("empty_state").innerHTML = `<h2>No runs yet</h2>
    <p>trainview watches the run folders under its <b>--root</b>. A run appears as soon as the runner writes <code>run.json</code> + <code>progress.jsonl</code> into a folder there (C++: <code>train/explore/viewer/include/runfolder.h</code>).</p>
    <p><b>Watching:</b> ${roots}</p>
    <p><b>Docs:</b> <code>train/explore/viewer/README.md</code> (run folder format, keys, C++ API)</p>
    <p class="muted small">Fake data to look at the screens: <code>train/explore/viewer/build/fake_run --out /tmp/tv/runs/fake_s1</code>, then <code>trainview --root /tmp/tv/runs</code>.</p>`;
}

function spark(vals, w = 70, h = 18) {
  const v = vals.filter(x => x != null && isFinite(x)); if (v.length < 2) return "";
  const lo = Math.min(...v), hi = Math.max(...v), d = hi - lo || 1, n = v.length;
  const pts = v.map((x, i) => `${(i / (n - 1) * (w - 2) + 1).toFixed(1)},${(h - 2 - (x - lo) / d * (h - 4)).toFixed(1)}`).join(" ");
  return `<svg class="spark" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}"><polyline points="${pts}" fill="none" stroke="var(--accent)" stroke-width="1.5"/></svg>`;
}
function dur(sec) { return sec == null ? "—" : sec >= 86400 ? (sec / 86400).toFixed(1) + " d" : sec >= 3600 ? (sec / 3600).toFixed(1) + " h" : (sec / 60).toFixed(0) + " min"; }
function updateStatus() {
  const r = S.byId[S.sel]; if (!r) return;
  setText("runname", shortName(r));
  const st = S.status[r.id] || {}, d = S.d;
  setHTML("rh_name", esc(r.id) + (r.group ? ` <span class="muted">· ${esc(r.group)}${r.seed != null ? " seed " + r.seed : ""}</span>` : ""));
  setHTML("rh_kind", S.meta && S.meta.kind ? `<span class="muted" title="run.json kind · mode">${esc(S.meta.kind + (S.meta.mode ? " · " + S.meta.mode : ""))}</span>` : "");
  setHTML("rh_state", `${stDot(r.id)}<b data-stl="${esc(r.id)}" title="${esc(stTitle(st))}">${esc(ST_WORD[st.state] || st.state || "")}</b>`);
  const wall = d && d.total ? lastAt("time/time_elapsed", d.total - 1) : r.wall;
  setText("rh_time", `${wall != null ? "ran " + dur(wall) + " · " : ""}last write ${ageTxt(r.age)}`);
  const keyK = d && d.cols && d.cols["rollout/success_rate"] ? "rollout/success_rate" : null;
  const cur = keyK ? lastAt(keyK, d.total - 1) : null;
  setHTML("rh_res", keyK ? `<span title="${esc(keyK)}">Success <b>${pc0(cur)}</b></span> ${spark((d.cols[keyK] || []).filter(x => x != null).slice(-120))}` : '<span class="muted">no result yet</span>');
  let b = "";
  if (r.synthetic) b += '<span class="badge syn" title="fake_run 이 만든 가짜 자료">synthetic</span>';
  if (r.segments > 1) b += `<span class="badge" title="재개(resume) 횟수 — run.json segments">resumed ×${r.segments - 1}</span>`;
  setHTML("badges", b);
  $("syn_banner").hidden = !r.synthetic;
  const last = d && d.total ? lastAt("time/iterations", d.total - 1) : null;
  setText("status", `iteration ${fmt(last)} · ${d ? d.total : 0} rows${d && d.rewound ? ` (${d.rewound} rewound rows dropped)` : ""}${d && d.bad ? ` · ${d.bad} unparsable rows` : ""} · ${r.eps} logged episodes · ${r.replays + r.eval_replays} replays · ${r.disk_mb} MB`);
}

async function selectRun(id) {
  if (!id) return;
  S.sel = id; S.d = null; S.meta = null;
  const h = new URLSearchParams(location.hash.slice(1)); h.set("run", id); history.replaceState(null, "", "#" + h.toString());
  renderRunMenu();
  stopLive();
  setHTML("table", ""); setHTML("evals", ""); setHTML("sample_charts", "");
  const meta = await api("/api/meta", { run: id });
  if (S.sel !== id) return;
  S.meta = meta.why ? null : meta;
  setText("metajson", JSON.stringify(meta, null, 1));
  const r = S.byId[id] || {};
  S.eps = r.eps || 0; S.rep = (r.replays || 0) + (r.eval_replays || 0);
  replay.setRun(id, S.meta, r.streams || ["main"]);
  setHTML("t_stream", (r.streams || ["main"]).map(s => `<option>${esc(s)}</option>`).join(""));
  await reloadProgress();
  loadEvals(); loadEvents(); loadTable(); loadSample();
  startLive();
}

// ---------------------------------------------------------------- progress (머리줄에 쓰는 열만, 이어 받기)
const HEAD_KEYS = "rollout/success_rate,time/time_elapsed,time/iterations";
async function reloadProgress() {
  const id = S.sel;
  const d = await api("/api/progress", { run: id, keys: HEAD_KEYS, max_points: 4000 });
  if (S.sel !== id) return;
  S.d = d.total ? d : { total: 0, sig: d.sig || 0, x: {}, cols: {}, why: d.why };
  render();
}
function merge(d) {
  const D = S.d;
  if (!D || d.sig !== D.sig || d.start > D.total || D.agg) { reloadProgress(); return; }
  const skip = D.total - d.start; if (d.total <= D.total) return;
  for (const k of Object.keys(d.x)) D.x[k] = (D.x[k] || []).concat(d.x[k].slice(skip));
  for (const k of Object.keys(d.cols)) {
    if (!D.cols[k]) D.cols[k] = new Array(D.total).fill(null);
    D.cols[k] = D.cols[k].concat(d.cols[k].slice(skip));
  }
  const n = d.total;
  for (const k of Object.keys(D.cols)) if (D.cols[k].length < n) D.cols[k] = D.cols[k].concat(new Array(n - D.cols[k].length).fill(null));
  D.total = n; D.rewound = d.rewound; D.bad = d.bad;
  render();
}

// ---------------------------------------------------------------- 실시간: SSE, 끊기면 3 초 폴링
function stopLive() {
  if (S.es) { S.es.close(); S.es = null; }
  clearInterval(S.poll); S.poll = null; clearTimeout(S.retry);
}
function startLive() {
  stopLive();
  const id = S.sel, D = S.d || { total: 0, sig: 0 };
  const es = new EventSource(`/api/live?runs=${enc(id)}&from=${D.total}&sig=${D.sig}&eps=${S.eps}&rep=${S.rep}`);
  S.es = es;
  es.onopen = () => { setText("conn", "live (SSE)"); $("conn").classList.add("ok"); clearInterval(S.poll); S.poll = null; };
  es.addEventListener("progress", e => { const m = JSON.parse(e.data); if (m.run === S.sel) merge(m.data); });
  es.addEventListener("reset", e => { const m = JSON.parse(e.data); if (m.run === S.sel) reloadProgress(); });
  es.addEventListener("episodes", e => { const m = JSON.parse(e.data); if (m.run === S.sel && m.total !== S.eps) { S.eps = m.total; debounce("eps", () => { loadTable(); loadSample(); }, 1500); } });
  es.addEventListener("replay", e => { const m = JSON.parse(e.data); if (m.run === S.sel) { S.rep = m.n; replay.refreshList(); debounce("ev", () => { loadEvals(); loadEvents(); }, 500); } });
  es.addEventListener("runs", () => loadRuns());
  es.addEventListener("status", e => { applyStatus(JSON.parse(e.data)); updateStatus(); });
  es.onerror = () => {
    es.close(); if (S.es === es) S.es = null;
    setText("conn", "polling 3 s"); $("conn").classList.remove("ok");
    if (!S.poll) S.poll = setInterval(pollOnce, 3000);
    S.retry = setTimeout(() => { if (S.sel === id) startLive(); }, 8000);
  };
}
async function pollOnce() {
  const id = S.sel, D = S.d; if (!D) return;
  const d = await api("/api/progress", { run: id, keys: HEAD_KEYS, from: D.total, sig: D.sig });
  if (S.sel !== id) return;
  if (d.sig !== D.sig) { reloadProgress(); return; }
  if (d.total > D.total) merge(d);
}
const timers = {};
function debounce(k, f, ms) { clearTimeout(timers[k]); timers[k] = setTimeout(f, ms); }
setInterval(loadRuns, 10000);

// ---------------------------------------------------------------- 값 꺼내기
function col(k) { return S.d && S.d.cols ? S.d.cols[k] : null; }
function lastAt(k, i) { const c = col(k); if (!c) return null; for (let j = Math.min(i, c.length - 1); j >= 0; j--) if (c[j] != null && isFinite(c[j])) return c[j]; return null; }

function render() {
  if (!S.d) return;
  updateStatus();
  const m = S.meta || {};
  const git = m.git && m.git.commit ? `git ${m.git.commit.slice(0, 7)}${m.git.dirty ? "+" : ""}` : "";
  setHTML("metaline", [m.baseline && esc(m.baseline), m.trainer && esc(m.trainer), m.n_envs ? `n_envs ${m.n_envs}` : "", m.ray_path ? `ray path ${esc(m.ray_path)}` : "", git].filter(Boolean).join(" · ") || esc(S.d.why || ""));
}

// ---------------------------------------------------------------- 표 (표본)
async function loadTable() {
  const id = S.sel; if (!id) return;
  const last = STOPS[+$("t_last").value];
  setText("t_last_v", last ? String(last) : "all");
  const q = { run: id, stream: $("t_stream").value || "main", rows: $("t_rows").value, cols: $("t_cols").value, last, map: $("t_map").value, stage: $("t_stage").value };
  const t = await api("/api/table", q);
  if (S.sel !== id) return;
  if (t.why || !t.cells || !t.cells.length) { setHTML("table", `<span class="muted">${esc(t.why || "no table")}</span>`); setText("t_note", ""); return; }
  for (const [sel, dim] of [["t_map", "map"], ["t_stage", "stage"]]) {
    const cur = $(sel).value;
    setHTML(sel, ['<option value="">all</option>'].concat((t.dims[dim] || []).map(n => `<option${n === cur ? " selected" : ""}>${esc(n)}</option>`)).join(""));
  }
  const met = $("t_metric").value, pct = !["t95", "t", "ret", "path_len"].includes(met);
  const hue = ["collision", "truncation", "declare_short"].includes(met) ? "--s2" : "--s1";
  const m = S.meta || {};
  setText("t_note", `${t.total} episodes in this split${m.log_envs != null ? ` (⚠ sample: every episode of ${m.log_envs} logged envs)` : ""}. Last ${last || "all"} episodes per cell. n < 20 is faded.`);
  let h = `<table class="t"><tr><th></th>${t.cols.map(c => `<th>${esc(c)}</th>`).join("")}</tr>`;
  t.rows.forEach((rn, ri) => {
    h += `<tr><th>${esc(rn)}</th>`;
    t.cols.forEach((cn, ci) => {
      const c = t.cells[ri][ci];
      if (!c.n) { h += `<td class="cell none">—</td>`; return; }
      const v = c[met];
      const a = pct && v != null ? Math.round(8 + 55 * Math.min(1, Math.max(0, v))) : 0;
      h += `<td class="cell${c.n < 20 ? " low" : ""}" style="${a ? `background:color-mix(in srgb, var(${hue}) ${a}%, transparent)` : ""}" title="n ${c.n}${c.n < 20 ? " — small sample" : ""}\nsuccess ${fmt(c.success, 1)} · coverage ${fmt(c.coverage, 1)} · t95 ${fmt(c.t95)} s (reached ${fmt(c.t95_reached, 1)})\ncollision ${fmt(c.collision, 1)} · declared short ${fmt(c.declare_short, 1)} · truncated ${fmt(c.truncation, 1)}\nlength ${fmt(c.t)} s · path ${fmt(c.path_len)} m · return ${fmt(c.ret)}"><b>${fmt(v, pct)}</b><span class="n">n ${c.n} · t ${fmt(c.t)} s</span></td>`;
    });
    h += "</tr>";
  });
  setHTML("table", h + "</table>");
}
["t_stream", "t_rows", "t_cols", "t_map", "t_stage", "t_metric"].forEach(id => $(id).addEventListener("change", loadTable));
$("t_last").addEventListener("input", loadTable);

async function loadSample() {
  const id = S.sel; if (!id) return;
  const host = $("sample_charts"); if (!host) return;
  if (!$("sample_panel").open) return;   // 접혀 있으면 안 받음
  if (!S.eps) { setText("sample_note", "no episodes.jsonl"); host.innerHTML = ""; return; }
  const e = await api("/api/episodes", { run: id, from: Math.max(0, S.eps - 20000) });
  if (S.sel !== id || !e.rows) return;
  setText("sample_note", `last ${e.rows.length} episodes (of ${e.total}${e.truncated ? ", truncated at cap " + e.cap : ""}${e.ret_mismatch ? `, ${e.ret_mismatch} rows with ret ≠ Σr` : ""}${e.bad ? `, ${e.bad} unparsable rows` : ""})`);
  const specs = [["Episode length t (s)", r => r.t, "--s1"], ["Final coverage", r => r.coverage, "--s3"], ["Time to 95 % (s, reached only)", r => r.t95, "--s7"], ["Path length (m)", r => r.path_len, "--s4"]];
  if (!host.children.length) host.innerHTML = specs.map(s => `<div class="chart"><div class="ct">${esc(s[0])}</div><canvas></canvas></div>`).join("");
  specs.forEach((s, i) => hist(host.children[i].querySelector("canvas"), e.rows.map(s[1]), 30, "episodes", cssv(s[2])));
}

// ---------------------------------------------------------------- 평가 표 · 사건
async function loadEvals() {
  const id = S.sel, j = await api("/api/evals", { run: id }); if (S.sel !== id) return;
  if (!j.evals.length) { setHTML("evals", `<span class="muted">${esc(j.why)}</span>`); return; }
  setHTML("evals", j.evals.slice(-12).reverse().map(({ file, eval: e }) => {
    const rows = e.rows || [];
    return `<div class="small" style="margin-top:8px"><b class="mono">${esc(file)}</b> ${e.iter != null ? "iter " + fmt(e.iter) : ""} ${e.ckpt ? '· <span class="mono muted">' + esc(String(e.ckpt).split("/").pop()) + "</span>" : ""}</div>
      <div class="tablewrap"><table class="t"><tr><th>eval</th><th>split</th><th>n</th><th>success</th><th>coverage</th><th title="95 % 도달 시간 · 프론티어 기준선">t95 (s) · frontier</th><th>collision</th><th>declared short</th><th>truncated</th></tr>${rows.map(r => `<tr><td class="l">${esc(r.eval ?? "")}</td><td class="l">${esc(r.split ?? "")}</td><td>${fmt(r.n)}</td><td class="${r.pass === true ? "pass" : r.pass === false ? "fail" : ""}">${fmt(r.success, 1)}${r.pass === true ? " ✓" : r.pass === false ? " ✗" : ""}</td><td>${fmt(r.coverage, 1)}</td><td class="${r.t95 != null && r.ref_t95 != null ? (r.t95 <= r.ref_t95 ? "pass" : "fail") : ""}">${fmt(r.t95)} · ${fmt(r.ref_t95)}</td><td>${fmt(r.collision, 1)}</td><td>${fmt(r.declare_short, 1)}</td><td>${fmt(r.truncation, 1)}</td></tr>`).join("")}</table></div>`;
  }).join(""));
}
async function loadEvents() {
  const id = S.sel, j = await api("/api/events", { run: id }); if (S.sel !== id) return;
  setText("events", j.lines.length ? j.lines.join("\n") : "no events.txt");
}

// ---------------------------------------------------------------- 탭 · 조작
function setTab(t) {
  S.tab = t;
  document.querySelectorAll("#tabs button").forEach(b => b.classList.toggle("on", b.dataset.tab === t));
  document.querySelectorAll(".tab").forEach(s => s.classList.toggle("on", s.id === "tab_" + t));
  const h = new URLSearchParams(location.hash.slice(1)); h.set("tab", t); history.replaceState(null, "", "#" + h.toString());
  replay.show(t === "replay");
  if (t === "run") { render(); loadSample(); }
  if (t === "compare") { buildCompareList(); drawCompare(); }
}
document.querySelectorAll("#tabs button").forEach(b => b.onclick = () => setTab(b.dataset.tab));
$("sample_panel").addEventListener("toggle", loadSample);
addEventListener("resize", () => debounce("rs", drawCompare, 150));
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { drawCompare(); loadSample(); });

// ---------------------------------------------------------------- Compare
const cmpCharts = [];
function buildCompareList() {
  const host = $("cmp_runs");
  const groups = {};
  for (const r of S.runs) (groups[r.group || r.id] ||= []).push(r);
  const sig = S.runs.map(r => r.id).join(",") + "|" + [...S.cmpSel].join(",");
  if (host._sig === sig) return; host._sig = sig;
  if (!S.cmpSel.size && S.sel) S.cmpSel.add(S.sel);
  const order = [...S.cmpSel];
  host.innerHTML = Object.entries(groups).map(([g, rs]) => `<div class="grp">${esc(g)} <span class="muted">(${rs.length})</span></div>` + rs.map(r => {
    const on = S.cmpSel.has(r.id), ci = order.indexOf(r.id);
    return `<label class="run"><input type="checkbox" data-id="${esc(r.id)}"${on ? " checked" : ""}><span class="sw" style="background:${on ? color(ci) : "transparent"}"></span>${stDot(r.id)}${esc(shortName(r))}<span class="muted small">${r.seed != null ? "seed " + r.seed : ""}${r.synthetic ? " · synthetic" : ""}</span></label>`;
  }).join("")).join("");
  host.querySelectorAll("input").forEach(i => i.onchange = () => { if (i.checked) S.cmpSel.add(i.dataset.id); else S.cmpSel.delete(i.dataset.id); host._sig = ""; buildCompareList(); });
  loadCompare();
}
async function loadCompare() {
  const ids = [...S.cmpSel];
  await Promise.all(ids.filter(id => !S.cmpKeys[id]).map(async id => { const j = await api("/api/progress", { run: id }); S.cmpKeys[id] = (j.keys || []).map(k => k.k); }));
  const union = [...new Set(ids.flatMap(id => S.cmpKeys[id] || []))].sort();
  for (const [sel, pref] of [["cmp_a", ["rollout/coverage_mean", "rollout/success_rate"]], ["cmp_b", ["rollout/t95_mean", "rollout/collision_rate"]]]) {
    const cur = $(sel).value || pref.find(p => union.includes(p)) || union[0] || "";
    setHTML(sel, union.map(k => `<option${k === cur ? " selected" : ""}>${esc(k)}</option>`).join(""));
  }
  const keys = [$("cmp_a").value, $("cmp_b").value].filter(Boolean);
  const tableKeys = ["rollout/success_rate", "rollout/coverage_mean", "rollout/t95_mean", "rollout/collision_rate", "rollout/declare_short_rate", "rollout/t95_reached_rate"].filter(k => union.includes(k));
  const need = [...new Set([...keys, ...tableKeys, "time/time_elapsed"])];
  await Promise.all(ids.map(async id => { S.cmpData[id] = await api("/api/progress", { run: id, keys: need.join(","), max_points: 1500 }); }));
  S.cmpTableKeys = tableKeys;
  drawCompare();
}
["cmp_a", "cmp_b"].forEach(id => $(id).onchange = loadCompare);
["cmp_x", "cmp_group"].forEach(id => $(id).onchange = drawCompare);
$("cmp_ema").oninput = drawCompare;

function cmpX(d, mode) {
  if (mode === "ts") return d.x.ts;
  if (mode === "wall") return d.x.time_elapsed;
  return d.x[{ iter: "iterations", env_steps: "total_timesteps" }[mode] || mode];
}
function drawCompare() {
  if (S.tab !== "compare") return;
  const ids = [...S.cmpSel].filter(id => S.cmpData[id] && S.cmpData[id].total);
  const keys = [$("cmp_a").value, $("cmp_b").value].filter(Boolean);
  const host = $("cmp_charts");
  while (cmpCharts.length < keys.length) {
    const el = document.createElement("div"); el.className = "chart";
    el.innerHTML = `<div class="ct"><span></span></div><canvas></canvas><div class="lg"></div><div class="cn"></div>`;
    host.appendChild(el); cmpCharts.push({ el, p: new Plot(el.querySelector("canvas")) });
  }
  const mode = $("cmp_x").value, a = +$("cmp_ema").value, grp = $("cmp_group").checked;
  const order = [...S.cmpSel];
  keys.forEach((k, ki) => {
    const C = cmpCharts[ki]; C.el.hidden = false;
    setText(C.el.querySelector(".ct span"), k);
    const pct = /success|_rate|coverage|fraction/.test(k);
    const series = [], missing = [], byGroup = {};
    for (const id of ids) {
      const y = S.cmpData[id].cols[k];
      if (!y || !y.some(v => v != null)) { missing.push(id); continue; }
      (byGroup[(S.byId[id] && S.byId[id].group) || id] ||= []).push(id);
    }
    const gi = Object.keys(byGroup);
    for (const [g, members] of Object.entries(byGroup)) {
      const dash = gi.indexOf(g) % 2 === 1;
      if (grp && members.length > 1) {
        // 평균 ± 표준편차 띠: 겹치는 x 구간을 200 칸으로 나눠 선형 보간
        const curves = members.map(id => { const d = S.cmpData[id]; const x = cmpX(d, mode), y = ema(d.cols[k], a); const pts = []; for (let i = 0; i < x.length; i++) if (y[i] != null && x[i] != null) pts.push([x[i], y[i]]); return pts; }).filter(p => p.length > 1);
        const lo = Math.max(...curves.map(p => p[0][0])), hi = Math.min(...curves.map(p => p[p.length - 1][0]));
        if (curves.length > 1 && hi > lo) {
          const X = [], M = [], L = [], H = [];
          for (let j = 0; j <= 200; j++) {
            const xv = lo + (hi - lo) * j / 200;
            const vs = curves.map(p => { let i = p.findIndex(q => q[0] >= xv); if (i <= 0) return p[Math.max(0, i)][1]; const [x0, y0] = p[i - 1], [x1, y1] = p[i]; return y0 + (y1 - y0) * (xv - x0) / (x1 - x0 || 1); });
            const mm = vs.reduce((s, v) => s + v, 0) / vs.length, sd = Math.sqrt(vs.reduce((s, v) => s + (v - mm) ** 2, 0) / Math.max(1, vs.length - 1));
            X.push(xv); M.push(mm); L.push(pct ? Math.max(0, mm - sd) : mm - sd); H.push(pct ? Math.min(1, mm + sd) : mm + sd);
          }
          series.push({ name: `${g} mean ± std (${curves.length} seeds)`, x: X, y: M, lo: L, hi: H, color: color(order.indexOf(members[0])), dash });
          continue;
        }
      }
      for (const id of members) {
        const d = S.cmpData[id];
        series.push({ name: shortName(S.byId[id] || { id }), x: cmpX(d, mode), y: ema(d.cols[k], a), color: color(order.indexOf(id)), dash });
      }
    }
    C.p.draw({ series, pct, xlabel: mode, xfmt: mode === "ts" ? (v => new Date(v * 1000).toLocaleTimeString()) : undefined, empty: "selected runs do not log this key" });
    setHTML(C.el.querySelector(".lg"), series.map(s => `<span class="${s.dash ? "dash" : ""}" style="--c:${s.color}">${esc(s.name)}</span>`).join(""));
    setText(C.el.querySelector(".cn"), missing.length ? `not logged in: ${missing.map(id => shortName(S.byId[id] || { id })).join(", ")}` : "");
  });
  for (let i = keys.length; i < cmpCharts.length; i++) cmpCharts[i].el.hidden = true;
  const tk = S.cmpTableKeys || [];
  let h = `<table class="t"><tr><th>run</th><th>group</th><th>total_timesteps</th>${tk.map(k => `<th title="${esc(k)}">${esc(k)}</th>`).join("")}</tr>`;
  for (const id of [...S.cmpSel]) {
    const d = S.cmpData[id]; if (!d) continue;
    const n = d.total || 0, a0 = Math.floor(n * 0.9);
    h += `<tr><td class="l">${esc(shortName(S.byId[id] || { id }))}</td><td class="l">${esc((S.byId[id] || {}).group || "")}</td><td>${fmt(d.x && d.x.total_timesteps ? d.x.total_timesteps[d.x.total_timesteps.length - 1] : null)}</td>`;
    for (const k of tk) {
      const c = d.cols && d.cols[k];
      if (!c) { h += `<td class="none">not logged</td>`; continue; }
      let s = 0, m = 0; for (let i = Math.min(a0, c.length - 1); i < c.length; i++) if (c[i] != null) { s += c[i]; m++; }
      h += `<td>${m ? fmt(s / m, /rate|coverage/.test(k)) : "—"}</td>`;
    }
    h += "</tr>";
  }
  setHTML("cmp_table", h + "</table>");
}

// ---------------------------------------------------------------- 시작
const h0 = new URLSearchParams(location.hash.slice(1));
loadRuns().then(() => { const t = h0.get("tab"); if (t && $("tab_" + t)) setTab(t); });   // 옛 링크(#tab=train)는 기본 Run 탭으로
window.__trainview = S;
