// trainview 화면 — Training·Replay·Compare 탭. 순수 ES 모듈, 빌드 단계 없음.
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
  runs: [], byId: {}, latest: null, sel: null, meta: null, d: null, keys: [], tab: "train",
  xmode: "env_steps", ema: 0.6, cursorPos: 1000, es: null, poll: null, retry: null,
  eps: 0, rep: 0, plots: [], built: "", cmpSel: new Set(), cmpData: {}, cmpKeys: {}, status: {},
};
const STOPS = [10, 20, 50, 100, 200, 500, 0];

// ---------------------------------------------------------------- 학습 상태 불 (서버 runs::status, SSE "status" 로 2 초마다)
const ST_KO = { training: "학습 중(pid 살아 있고 60 s 안에 기록)", stalled: "멈춤?(프로세스는 살아 있는데 60 s 넘게 기록 없음)", finished: "끝남(run.json ended)", crashed: "죽음(프로세스 없음, ended 없음)", unknown: "모름(run.json 에 pid 없음)" };
const ST_WORD = { training: "Training", stalled: "Stalled", finished: "Finished", crashed: "Crashed", unknown: "Status unknown" };
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
$("healthchip").onclick = () => { $("checks_box").hidden = !$("checks_box").hidden; };
const replay = new Replay({ api, getCursorIter: () => cursorIter() });

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
  const res = h["eval/success_rate"] != null ? `eval ${pc0(h["eval/success_rate"])}` : h["rollout/success_rate"] != null ? `success ${pc0(h["rollout/success_rate"])}` : "";
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
  $("empty_state").hidden = !empty; $("train_body").hidden = empty; $("rundd").hidden = empty;
  if (!empty) return;
  const roots = (S.roots || []).map(p => `<code>${esc(p)}</code>`).join(" ") || "<code>(no --root)</code>";
  $("empty_state").innerHTML = `<h2>No training runs yet</h2>
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
function stageName(si) { const st = S.meta && S.meta.curriculum && S.meta.curriculum.stages; return si == null ? null : st && st[si] ? (st[si].name || String(st[si])) : String(si); }
function updateStatus() {
  const r = S.byId[S.sel]; if (!r) return;
  setText("runname", shortName(r));
  const st = S.status[r.id] || {}, d = S.d;
  setHTML("rh_name", esc(r.id) + (r.group ? ` <span class="muted">· ${esc(r.group)}${r.seed != null ? " seed " + r.seed : ""}</span>` : ""));
  const si = d && d.total ? lastAt("curriculum/stage", d.total - 1) : null;
  setHTML("rh_stage", si != null ? `<span title="커리큘럼 단계 (curriculum/stage, 이름은 run.json curriculum.stages)">Stage ${si} · ${esc(stageName(si))}</span>` : `<span class="muted">${S.meta && S.meta.kind && S.meta.kind !== "explore_ppo" ? esc(S.meta.kind + (S.meta.mode ? " · " + S.meta.mode : "")) : "explore PPO"}</span>`);
  setHTML("rh_state", `${stDot(r.id)}<b data-stl="${esc(r.id)}" title="${esc(stTitle(st))}">${esc(ST_WORD[st.state] || st.state || "")}</b>`);
  const wall = d && d.total ? lastAt("time/time_elapsed", d.total - 1) : r.wall;
  setText("rh_time", `${wall != null ? "ran " + dur(wall) + " · " : ""}last write ${ageTxt(r.age)}`);
  const keyK = ["eval/success_rate", "rollout/success_rate"].find(k => d && d.cols && d.cols[k]);
  const cur = keyK ? lastAt(keyK, d.total - 1) : null;
  setHTML("rh_res", keyK ? `<span title="${esc(keyK)}">${keyK.startsWith("eval") ? "Eval success" : "Success"} <b>${pc0(cur)}</b></span> ${spark((d.cols[keyK] || []).filter(x => x != null).slice(-120))}` : '<span class="muted">no result yet</span>');
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
  S.sel = id; S.d = null; S.meta = null; S.built = "";
  const h = new URLSearchParams(location.hash.slice(1)); h.set("run", id); history.replaceState(null, "", "#" + h.toString());
  renderRunMenu();
  stopLive();
  setHTML("groups", ""); setHTML("cards", ""); setHTML("checks", ""); setHTML("table", ""); setHTML("evals", "");
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

// ---------------------------------------------------------------- progress (열, 이어 받기)
async function reloadProgress() {
  const id = S.sel;
  const d = await api("/api/progress", { run: id, keys: "*", max_points: 4000 });
  if (S.sel !== id) return;
  S.d = d.total ? d : { total: 0, sig: d.sig || 0, x: {}, cols: {}, why: d.why };
  S.keys = Object.keys(S.d.cols || {}).sort();
  S.built = "";
  render();
}
function merge(d) {
  const D = S.d;
  if (!D || d.sig !== D.sig || d.start > D.total || D.agg) { reloadProgress(); return; }
  const skip = D.total - d.start; if (d.total <= D.total) return;
  for (const k of Object.keys(d.x)) D.x[k] = (D.x[k] || []).concat(d.x[k].slice(skip));
  let newKey = false;
  for (const k of Object.keys(d.cols)) {
    if (!D.cols[k]) { D.cols[k] = new Array(D.total).fill(null); newKey = true; }
    D.cols[k] = D.cols[k].concat(d.cols[k].slice(skip));
  }
  const n = d.total;
  for (const k of Object.keys(D.cols)) if (D.cols[k].length < n) D.cols[k] = D.cols[k].concat(new Array(n - D.cols[k].length).fill(null));
  D.total = n; D.rewound = d.rewound; D.bad = d.bad;
  if (newKey) { S.keys = Object.keys(D.cols).sort(); S.built = ""; }
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
  const d = await api("/api/progress", { run: id, keys: "*", from: D.total, sig: D.sig });
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
function firstVal(k) { const c = col(k); if (!c) return null; for (const v of c) if (v != null && isFinite(v)) return v; return null; }
function cursorRow() { const n = S.d ? S.d.total : 0; if (!n) return -1; return S.cursorPos >= 1000 ? n - 1 : Math.round(S.cursorPos / 1000 * (n - 1)); }
function cursorIter() { const i = cursorRow(); return i < 0 || S.cursorPos >= 1000 ? null : S.d.x.iterations[i]; }
function xs() { if (!S.d || !S.d.x) return []; return S.xmode === "wall" ? S.d.x.time_elapsed : S.xmode === "iter" ? S.d.x.iterations : S.d.x.total_timesteps; }
function expand(pats) {
  const out = [];
  for (const p of pats) {
    if (!p.includes("*")) { if (col(p)) out.push(p); continue; }
    const re = new RegExp("^" + p.split("*").map(s => s.replace(/[.+?^${}()|[\]\\/]/g, "\\$&")).join("([^]*)") + "$");
    for (const k of S.keys) if (re.test(k) && !out.includes(k)) out.push(k);
  }
  return out;
}
function ref(key, label) { const v = S.meta && S.meta.refs ? S.meta.refs[key] : null; return v != null ? [{ y: v, label: label || key + " " + fmt(v) }] : []; }
function metaRef(path, label) { let v = S.meta; for (const p of path.split(".")) { if (v == null) return []; v = v[p]; } return typeof v === "number" ? [{ y: v, label: label + " " + fmt(v) }] : []; }

// ---------------------------------------------------------------- 곡선 묶음 — 이 한 곳이 화면을 정한다. 키 = trainfmt/src/keys.rs
function chartsSpec() {
  const fr = (() => { const c = col("eval/frontier_t95_mean"); return c ? [{ y: lastAt("eval/frontier_t95_mean", S.d.total - 1), label: "frontier baseline" }] : []; })();
  return [
    { g: "Episode outcome (all episodes)", ko: "판 끝 사유 — 학습 판 전체(모집단)", c: [
      { t: "Success rate (declared ≥ threshold)", ko: "선언 + 기준 덮음 이상", k: ["rollout/success_rate"], pct: 1, refs: () => ref("rollout/success_rate", "target") },
      { t: "Eval success rate", ko: "평가 판 성공률", k: ["eval/success_rate"], pct: 1 },
      { t: "Collision · declared short · truncated · no progress", ko: "충돌(종료) · 기준 미달 선언 · 안전 시간 끊김 · 진행 없음(종료)", k: ["rollout/collision_rate", "rollout/declare_short_rate", "rollout/truncation_rate", "rollout/no_progress_rate"], pct: 1 },
      { t: "Person collision rate", ko: "사람과 부딪혀 끝난 판 비율(충돌의 일부)", k: ["rollout/person_collision_rate"], pct: 1 },
      { t: "Declare rate", ko: "끝 선언으로 끝난 판 비율(성공 + 미달)", k: ["rollout/declare_rate"], pct: 1 },
      { t: "Episodes per row", ko: "줄마다 끝난 판 수", k: ["rollout/n_episodes"] },
    ]},
    { g: "Exploration", ko: "탐사 지표", c: [
      { t: "Coverage at episode end", ko: "판 끝 덮은 비율(정답 맵 기준)", k: ["rollout/coverage_mean", "eval/coverage_mean"], pct: 1, refs: () => metaRef("success_coverage", "success threshold") },
      { t: "Time to 95 % coverage (s)", ko: "95 % 덮음 도달 시간 — 도달한 판만. 기준선 = 같은 맵 프론티어 탐사", k: ["rollout/t95_mean", "eval/t95_mean"], refs: () => fr },
      { t: "Reached 95 % coverage", ko: "95 % 덮음에 도달한 판 비율", k: ["rollout/t95_reached_rate", "eval/t95_reached_rate"], pct: 1 },
      { t: "Coverage when declaring", ko: "선언한 판의 선언 때 덮음 — 일찍 선언하나", k: ["rollout/coverage_at_declare_mean", "eval/coverage_at_declare_mean"], pct: 1, refs: () => metaRef("success_coverage", "success threshold") },
      { t: "Path length (m)", ko: "이동 거리", k: ["rollout/path_len_mean", "eval/path_len_mean"] },
      { t: "Known area (m²)", ko: "판 끝 정책 지도에서 안 넓이", k: ["rollout/known_area_mean"] },
      { t: "Episode length (steps)", ko: "판 길이", k: ["rollout/ep_len_mean"], refs: () => (S.meta && S.meta.t_safe_s && S.meta.ctrl_hz) ? [{ y: S.meta.t_safe_s * S.meta.ctrl_hz, label: "safety time" }] : [] },
    ]},
    { g: "Reward", ko: "보상", c: [
      { t: "Episode return — rollout/ep_rew_mean", ko: "판 보상 합(리턴) 평균", k: ["rollout/ep_rew_mean", "eval/mean_reward"] },
      { t: "Reward terms (per-episode sum)", ko: "보상 항별 판 합 평균 — 항 이름은 mdp/rewards.h", k: ["reward/*"] },
      { t: "Reward per step", ko: "걸음당 보상", k: ["rollout/step_reward_mean"] },
      { t: "Eval reward terms", ko: "평가 판 보상 항", k: ["eval_reward/*"] },
    ]},
    { g: "Policy optimization (train/)", ko: "PPO 건강", c: [
      { t: "Approx KL — train/approx_kl", ko: "근사 KL", k: ["train/approx_kl"], refs: () => ref("train/approx_kl", "target_kl") },
      { t: "Clip fraction", ko: "잘린 비율", k: ["train/clip_fraction"], pct: 1 },
      { t: "Explained variance", ko: "설명된 분산", k: ["train/explained_variance"] },
      { t: "Entropy loss — train/entropy_loss", ko: "엔트로피 손실(= −엔트로피, SB3)", k: ["train/entropy_loss"] },
      { t: "Entropy per action component", ko: "성분별 엔트로피(v, vy, ω, 선언)", k: ["train/entropy/*"] },
      { t: "Policy std per action — train/std", ko: "행동별 σ", k: ["train/std", "train/std/*"] },
      { t: "Declare head — logit · probability", ko: "끝 선언 로짓 평균(−6~−8 로 시작) · 걸음당 선언 확률", k: ["train/declare_logit_mean"] },
      { t: "Declare probability per step", ko: "걸음당 선언 확률", k: ["train/declare_prob_mean"], pct: 1 },
      { t: "Value loss · policy loss", ko: "가치 손실 · 정책 손실", k: ["train/value_loss", "train/policy_gradient_loss", "train/loss"] },
      { t: "Gradient norm", ko: "기울기 노름", k: ["train/grad_norm"], refs: () => ref("train/grad_norm", "max_grad_norm") },
      { t: "Learning rate", ko: "학습률", k: ["train/learning_rate"] },
      { t: "Advantage std · value mean", ko: "이득 표준편차 · 가치 평균", k: ["train/advantage_std", "train/advantage_mean", "train/value_mean"] },
    ]},
    { g: "Curriculum", ko: "커리큘럼", c: [
      { t: "Curriculum stage", ko: "단계 번호(이름은 run.json curriculum.stages)", k: ["curriculum/stage"] },
      { t: "Success by stage", ko: "단계별 성공률", k: ["curriculum/success_rate/*"], pct: 1 },
      { t: "Other curriculum values", ko: "그 밖 curriculum/*", k: ["curriculum/*"] },
    ]},
    { g: "Throughput · resources (time/)", ko: "속도 · 자원", c: [
      { t: "FPS (env steps/s) — time/fps", ko: "초당 환경 걸음", k: ["time/fps"] },
      { t: "Iteration time (ms)", ko: "업데이트 시간", k: ["time/iter_ms", "time/rollout_ms", "time/update_ms"] },
      { t: "GPU memory (MB)", ko: "GPU 메모리", k: ["gpu/*"] },
      { t: "Dropped log records (should be 0)", ko: "로그 버림", k: ["log/*"] },
      { t: "Iterations merged per row", ko: "한 줄에 합친 업데이트 수", k: ["time/iterations_merged"] },
    ]},
  ];
}
const WANT = { "Episode outcome (all episodes)": ["rollout/success_rate", "rollout/collision_rate", "rollout/n_episodes"], "Exploration": ["rollout/coverage_mean", "rollout/t95_mean"], "Reward": ["rollout/ep_rew_mean", "reward/<term>"], "Policy optimization (train/)": ["train/approx_kl", "train/explained_variance"] };
const KEY_CHARTS = ["Success rate (declared ≥ threshold)", "Coverage at episode end", "Time to 95 % coverage (s)", "Collision · declared short · truncated", "Episode return — rollout/ep_rew_mean"];

function buildCharts() {
  const sig = S.sel + "|" + S.keys.join(",");
  if (S.built === sig) return;
  S.built = sig; S.plots = [];
  const host = $("groups"); host.innerHTML = "";
  const mk = (c, keys, box) => {
    const el = document.createElement("div"); el.className = "chart";
    el.innerHTML = `<div class="ct"><span title="${esc(c.ko || "")}">${esc(c.t)}</span><span class="u mono"></span></div><canvas></canvas><div class="lg"></div><div class="cn"></div>`;
    box.appendChild(el);
    S.plots.push({ c, keys, el, p: new Plot(el.querySelector("canvas")), sig: "" });
  };
  const spec = chartsSpec();
  // "Other curriculum values" 는 앞 두 장에 없는 curriculum/* 만
  const shown = new Set();
  const keysOf = c => { let ks = expand(c.k); if (c.t === "Other curriculum values") ks = ks.filter(k => k !== "curriculum/stage" && !k.startsWith("curriculum/success_rate/")); return ks; };
  const all = spec.flatMap(g => g.c.map(c => ({ c, keys: keysOf(c) }))).filter(x => x.keys.length);
  const key = KEY_CHARTS.map(t => all.find(x => x.c.t === t)).filter(Boolean).slice(0, 4), keySet = new Set(key.map(x => x.c));
  const top = document.createElement("div"); top.className = "group";
  top.innerHTML = `<h3 title="이 실행에서 가장 중요한 곡선 넷">Key charts <span class="nl">${key.length ? "" : "nothing logged yet"}</span></h3>`;
  const tb = document.createElement("div"); tb.className = "charts"; top.appendChild(tb); host.appendChild(top);
  for (const x of key) { mk(x.c, x.keys, tb); shown.add(x.c); }
  const more = document.createElement("details"); more.className = "morecharts"; more.id = "more_charts";
  more.innerHTML = `<summary>More charts <span class="muted">${all.length - key.length} more, grouped</span></summary>`;
  host.appendChild(more);
  for (const grp of spec) {
    const have = grp.c.map(c => ({ c, keys: keysOf(c) })).filter(x => x.keys.length && !keySet.has(x.c));
    const missing = grp.c.filter(c => !keysOf(c).length).flatMap(c => c.k);
    const div = document.createElement("div"); div.className = "group";
    const nl = have.length ? `${have.length} charts` : `not logged: ${(WANT[grp.g] || missing.slice(0, 4)).map(k => "<code>" + esc(k) + "</code>").join(" ")}`;
    div.innerHTML = `<h3 title="${esc(grp.ko)}">${esc(grp.g)} <span class="nl">${nl}</span></h3>`;
    if (have.length) { const box = document.createElement("div"); box.className = "charts"; div.appendChild(box); for (const x of have) mk(x.c, x.keys, box); }
    more.appendChild(div);
  }
  const div = document.createElement("div"); div.className = "group"; div.id = "sample_group";
  div.innerHTML = `<h3 title="판 기록(표본)의 분포">Logged episodes <span class="warn-s">⚠ sample</span> <span class="nl" id="sample_note"></span></h3><div class="charts" id="sample_charts"></div>`;
  more.appendChild(div);
  more.addEventListener("toggle", () => { if (more.open) { for (const P of S.plots) P.sig = ""; drawCharts(); loadSample(); } });
}

function label(keys, k) {
  if (keys.length === 1) return k;
  const parts = keys.map(x => x.split("/"));
  let i = 0; while (parts.every(p => p.length > i + 1 && p[i] === parts[0][i])) i++;
  return k.split("/").slice(i).join("/");
}

function drawCharts() {
  if (!S.d || S.tab !== "train") return;
  const x = xs(), n = S.d.total, crow = cursorRow();
  const cx = S.cursorPos < 1000 && crow >= 0 ? x[crow] : null;
  const xl = { iter: "time/iterations", env_steps: "time/total_timesteps", wall: "time/time_elapsed (s)" }[S.xmode];
  const theme = matchMedia("(prefers-color-scheme: dark)").matches;
  for (const P of S.plots) {
    const w = P.el.clientWidth; if (!w) continue;
    const sig = [n, S.xmode, S.ema, cx, w, theme].join("|");
    if (P.sig === sig) continue;
    P.sig = sig;
    const series = P.keys.slice(0, 8).map((k, i) => {
      const y = S.d.cols[k];
      return { name: label(P.keys, k), x, y: ema(y, S.ema), lo: S.d.lo && S.ema === 0 ? S.d.lo[k] : null, hi: S.d.hi && S.ema === 0 ? S.d.hi[k] : null, color: color(i) };
    });
    const refs = P.c.refs ? P.c.refs() : [];
    P.p.draw({ series, refs, pct: P.c.pct, cursor: cx, xlabel: xl, onPick: xv => setCursorX(xv) });
    const last = series.map(s => { for (let i = s.y.length - 1; i >= 0; i--) if (s.y[i] != null) return s.y[i]; return null; });
    setText(P.el.querySelector(".u"), series.length === 1 ? fmt(last[0], P.c.pct) : "");
    setHTML(P.el.querySelector(".lg"), series.length > 1 ? series.map(s => `<span style="--c:${s.color}">${esc(s.name)}</span>`).join("") : "");
    const notes = [];
    if (P.keys.length > 8) notes.push(`${P.keys.length - 8} more keys (8 series max)`);
    if (P.c.refs && !refs.length) notes.push("no reference value in run.json — line not drawn");
    if (S.d.agg) notes.push("downsampled: min–max band per bucket (smoothing 0) + last value");
    setText(P.el.querySelector(".cn"), notes.join(" · "));
  }
}
function setCursorX(xv) {
  const x = xs(); if (!x.length) return;
  let bi = 0, bd = Infinity; for (let i = 0; i < x.length; i++) { const d = Math.abs(x[i] - xv); if (d < bd) { bd = d; bi = i; } }
  S.cursorPos = x.length > 1 ? Math.round(bi / (x.length - 1) * 999) : 1000; $("cursor").value = S.cursorPos; render(); loadTable(); replay.refreshList();
}

// ---------------------------------------------------------------- 카드
function card(t, v, s, bad, ko) { return `<div class="card${bad ? " bad" : ""}"${ko ? ` title="${esc(ko)}"` : ""}><div class="t">${t}</div><div class="v">${v}</div><div class="s">${s || ""}</div></div>`; }
function renderCards() {
  if (!S.d) return;
  const i = cursorRow(), m = S.meta || {};
  if (i < 0) { setHTML("cards", card("progress", "—", esc(S.d.why || "no rows"))); setHTML("cards_more", ""); return; }
  const g = k => lastAt(k, i);
  const wall = g("time/time_elapsed");
  const big = (t, v, sub, ko, key, bad) => `<div class="card big${bad ? " bad" : ""}" title="${esc(ko)}"><div class="t">${t}</div><div class="v">${v}</div><div class="s">${sub}</div>${key ? spark((col(key) || []).slice(0, i + 1).filter(x => x != null).slice(-100), 120, 22) : ""}</div>`;
  const es = g("eval/success_rate"), sr = g("rollout/success_rate");
  const cov = g("rollout/coverage_mean"), t95 = g("rollout/t95_mean"), fr = g("eval/frontier_t95_mean");
  const coll = g("rollout/collision_rate");
  setHTML("cards", [
    big("Success", pc0(es ?? sr), es != null ? `eval · training rollouts ${pc0(sr)}` : "training rollouts (declared, coverage ≥ threshold)", "선언했고 덮음이 기준 이상인 판 비율", es != null ? "eval/success_rate" : "rollout/success_rate"),
    big("Coverage", pc0(cov), `at episode end · when declaring ${pc0(g("rollout/coverage_at_declare_mean"))}`, "판 끝 덮은 비율(정답 맵 기준) · 선언 때 덮음", "rollout/coverage_mean"),
    big("Time to 95 %", t95 != null ? fmt(t95) + " s" : "—", `reached in ${pc0(g("rollout/t95_reached_rate"))} of episodes${fr != null ? ` · frontier ${fmt(fr)} s` : ""}`, "95 % 덮음 도달 시간(도달한 판만). 프론티어 기준선보다 빨라야 한다", "rollout/t95_mean", t95 != null && fr != null && t95 > fr),
    big("Collisions", pc0(coll), `declared short ${pc0(g("rollout/declare_short_rate"))} · truncated ${pc0(g("rollout/truncation_rate"))} · no progress ${pc0(g("rollout/no_progress_rate"))}`, "충돌로 끝난 판 · 기준 미달 선언 · 안전 시간 끊김 · 진행 없음", "rollout/collision_rate"),
  ].join(""));
  const out = [];
  out.push(card("Iterations · timesteps", fmt(g("time/iterations")), `time/total_timesteps ${fmt(g("time/total_timesteps"))}`, false, "업데이트 수 · 누적 환경 걸음"));
  out.push(card("FPS · time elapsed", fmt(g("time/fps")), wall != null ? (wall >= 3600 ? (wall / 3600).toFixed(2) + " h" : (wall / 60).toFixed(1) + " min") : "—", false, "초당 환경 걸음 · 경과"));
  const si = g("curriculum/stage");
  out.push(card("Curriculum stage", si != null ? esc(`${si} · ${stageName(si)}`) : "—", si == null ? "not logged: <code>curriculum/stage</code>" : "", false, "커리큘럼 단계"));
  out.push(card("Episode return · length", fmt(g("rollout/ep_rew_mean")), `rollout/ep_len_mean ${fmt(g("rollout/ep_len_mean"))} steps`, false, "판 보상 합 평균 · 판 길이"));
  out.push(card("Path length", g("rollout/path_len_mean") != null ? fmt(g("rollout/path_len_mean")) + " m" : "—", `known area ${fmt(g("rollout/known_area_mean"))} m²`, false, "이동 거리 · 정책이 안 넓이"));
  const ne = col("rollout/n_episodes");
  let pop = null; if (ne) { pop = 0; for (let j = 0; j <= i; j++) if (ne[j] != null) pop += ne[j]; }
  out.push(card("Episodes", fmt(pop), `all (rollout) / logged sample ${fmt(S.eps)}`, false, "끝난 판 수: 모집단 / 기록된 표본"));
  const ls = g("train/log_std"), ls0 = firstVal("train/log_std");
  out.push(card("Policy log std", fmt(ls), ls != null && ls0 != null ? `std ${fmt(Math.exp(ls))} · ${fmt(Math.exp(ls - ls0), 1)} of initial` : "not logged: <code>train/log_std</code>", false, "정책 log σ (탐색)"));
  const kl = g("train/approx_kl"), tk = m.target_kl ?? (m.refs && m.refs["train/approx_kl"]);
  out.push(card("Approx KL · explained var.", fmt(kl), `${tk != null ? "target_kl " + fmt(tk) : "no target_kl"} · EV ${fmt(g("train/explained_variance"))}`, kl != null && tk != null && kl > 1.5 * tk, "근사 KL · 설명된 분산"));
  out.push(card("Declare head", fmt(g("train/declare_logit_mean")), `p(declare) per step ${fmt(g("train/declare_prob_mean"))}`, false, "끝 선언 로짓 평균 · 걸음당 확률"));
  const mem = g("gpu/mem_used_mb");
  out.push(card("GPU memory", mem != null ? fmt(mem / 1024) + " GB" : "—", mem != null ? "" : "not logged: <code>gpu/mem_used_mb</code>", false, "GPU 메모리"));
  const dk = expand(["log/*"]);
  let drop = null; for (const k of dk) { const c = col(k); for (let j = 0; j <= i; j++) if (c[j] > 0) drop = (drop || 0) + c[j]; }
  out.push(card("Dropped log records", dk.length ? fmt(drop || 0) : "—", dk.length ? dk.map(k => k.split("/").pop()).join(", ") : "not logged: <code>log/*_dropped</code>", drop > 0, "로그 버림"));
  if (S.sr) out.push(card("Success · coverage (logged sample)", fmt(S.sr.success, 1), `coverage ${fmt(S.sr.coverage, 1)} · collision ${fmt(S.sr.collision, 1)} · n ${S.sr.n}`, false, "표본으로 센 값 — 모집단(위)과 비교"));
  setHTML("cards_more", `<details class="morecards"><summary>More stats <span class="muted">${out.length} cards: stage, return, KL, declare head, GPU …</span></summary><div class="cards">${out.join("")}</div></details>`);
}

// ---------------------------------------------------------------- 학습 건강 검사. 못 잰 검사는 이유와 함께 따로(못 잰 것을 "이상 없음" 으로 세지 않음)
function slope(k, a, b) {
  const c = col(k), x = S.d.x.total_timesteps; if (!c) return null;
  let n = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (let i = a; i <= b; i++) { const v = c[i]; if (v == null || !isFinite(v)) continue; const xx = x[i]; n++; sx += xx; sy += v; sxx += xx * xx; sxy += xx * v; }
  if (n < 5) return null;
  const d = n * sxx - sx * sx; return d ? (n * sxy - sx * sy) / d : null;
}
function meanOf(k, a, b) { const c = col(k); if (!c) return null; let s = 0, n = 0; for (let i = a; i <= b; i++) if (c[i] != null && isFinite(c[i])) { s += c[i]; n++; } return n ? s / n : null; }
function renderChecks() {
  const chip = $("healthchip");
  if (!S.d || !S.d.total) { setHTML("checks", '<span class="muted">no progress rows — nothing to check</span>'); setText("checks_sum", ""); chip.className = "hchip na"; chip.textContent = "Health: —"; return; }
  const b = cursorRow(), a = Math.max(0, b - Math.max(20, Math.floor((b + 1) * 0.2))), m = S.meta || {}, th = m.checks || {};
  const R = [];
  const add = (st, name, msg, ko) => R.push({ st, name, msg, ko });
  // 1. 리턴은 오르는데 덮음·성공은 그대로 → 보상 꼼수
  const sr = slope("rollout/ep_rew_mean", a, b), sc = slope("rollout/coverage_mean", a, b), ss = slope("rollout/success_rate", a, b);
  if (sr == null || (sc == null && ss == null)) add("na", "Reward hacking", "not measurable: needs rollout/ep_rew_mean and rollout/coverage_mean or success_rate (≥ 5 values)", "보상 꼼수 의심");
  else { const bad = sr > 0 && (sc == null || sc <= 0) && (ss == null || ss <= 0); add(bad ? "warn" : "ok", "Reward hacking", bad ? "return rises while coverage and success are flat/falling" : "return and coverage/success trend together", "리턴은 오르는데 덮음·성공은 그대로"); }
  // 2. 일찍 선언: 선언 때 덮음이 기준보다 한참 낮고 미달 선언이 많음
  const cad = meanOf("rollout/coverage_at_declare_mean", a, b), dsr = meanOf("rollout/declare_short_rate", a, b), thr = m.success_coverage;
  if (dsr == null) add("na", "Premature declare", "not measurable: no rollout/declare_short_rate", "일찍 선언");
  else { const lim = th.declare_short ?? 0.3; add(dsr > lim ? "warn" : "ok", "Premature declare", `declared short ${fmt(dsr, 1)} (threshold ${fmt(lim, 1)}, assumed)${cad != null ? ` · coverage at declare ${fmt(cad, 1)}${thr != null ? " vs " + fmt(thr, 1) : ""}` : ""}`, "기준 미달 선언이 많음 — 선언 로짓 편향·보상 확인"); }
  // 3. 선언을 안 함: 거의 전부 안전 시간으로 끊김
  const tr = meanOf("rollout/truncation_rate", a, b);
  if (tr == null) add("na", "Never declares", "not measurable: no rollout/truncation_rate", "선언 안 함");
  else add(tr >= (th.truncation ?? 0.9) ? "warn" : "ok", "Never declares", `truncation rate ${fmt(tr, 1)}`, "거의 모든 판이 안전 시간으로 끊김");
  // 4. 안 움직임
  const pl = meanOf("rollout/path_len_mean", a, b);
  if (pl == null) add("na", "Agent not moving", "not measurable: no rollout/path_len_mean", "안 움직임");
  else add(pl < (th.min_path_m ?? 1.0) ? "warn" : "ok", "Agent not moving", `mean path ${fmt(pl)} m`, "이동 거리가 너무 짧음");
  // 5. 충돌 늘어남
  const cs = slope("rollout/collision_rate", a, b), cm = meanOf("rollout/collision_rate", a, b);
  if (cs == null) add("na", "Collisions rising", "not measurable: no rollout/collision_rate (≥ 5 values)", "충돌 늘어남");
  else add(cs > 0 && cm > 0.05 ? "warn" : "ok", "Collisions rising", `slope ${cs > 0 ? "+" : ""}${fmt(cs * 1e9)} per 1e9 steps · mean ${fmt(cm, 1)}`, "충돌 비율이 오름");
  // 6. 탐색 죽음
  const ls = lastAt("train/log_std", b), ls0 = firstVal("train/log_std");
  if (ls == null) add("na", "Entropy collapse", "not measurable: no train/log_std", "탐색 죽음");
  else { const lim = th.explore_drop ?? 2.3; add(ls0 - ls > lim ? "warn" : "ok", "Entropy collapse", `log std ${fmt(ls0)} → ${fmt(ls)} (threshold: drop > ${lim}, assumed)`, "탐색 죽음"); }
  // 7. KL
  const tk = m.target_kl ?? (m.refs && m.refs["train/approx_kl"]), kl = meanOf("train/approx_kl", a, b);
  if (kl == null || tk == null) add("na", "KL too high", `not measurable: no ${kl == null ? "train/approx_kl" : "target_kl in run.json"}`, "KL 넘침");
  else add(kl > tk * 1.5 ? "warn" : "ok", "KL too high", `window mean ${fmt(kl)} vs target_kl ${fmt(tk)} × 1.5`, "KL 넘침");
  // 8. 가치
  const ev = meanOf("train/explained_variance", a, b);
  if (ev == null) add("na", "Value not learning", "not measurable: no train/explained_variance", "가치 머리");
  else add(ev < (th.min_ev ?? 0) ? "warn" : "ok", "Value not learning", `explained variance ${fmt(ev)}`, "설명된 분산이 0 아래");
  // 9. 로그 버림
  const dk = expand(["log/*"]);
  if (!dk.length) add("na", "Dropped logs", "not measurable: log/*_dropped not logged", "로그 버림");
  else { let s = 0; for (const k of dk) { const c = col(k); for (let i = 0; i <= b; i++) if (c[i] > 0) s += c[i]; } add(s > 0 ? "warn" : "ok", "Dropped logs", `total ${fmt(s)}`, "로그 버림"); }
  const ic = { ok: "✓", warn: "!", na: "?" };
  setHTML("checks", R.map(r => `<div class="chk ${r.st}" title="${esc(r.ko)}"><span class="ic">${ic[r.st]}</span><b>${r.name}</b> <span>${esc(r.msg)}</span></div>`).join(""));
  const ok = R.filter(r => r.st !== "na").length, warn = R.filter(r => r.st === "warn").length;
  setText("checks_sum", `checked ${ok} (flagged ${warn}) · not measurable ${R.length - ok} · window = last ${b - a + 1} rows before cursor`);
  chip.className = "hchip " + (warn ? "warn" : ok ? "ok" : "na");
  chip.textContent = warn ? `Health: ${warn} warning${warn > 1 ? "s" : ""}` : ok ? "Health: OK" : "Health: not measurable";
}

function renderFirstNote() {
  const n = $("note_first"), m = S.meta || {};
  const ne = col("rollout/n_episodes");
  if (!S.d || !S.d.total || !ne || ne.some(v => v > 0)) { n.hidden = true; return; }
  n.hidden = false;
  n.textContent = m.t_safe_s && m.ctrl_hz && m.rollout_T ? `No episode finished yet — by the safety time every episode ends by iteration ~${Math.ceil(m.t_safe_s * m.ctrl_hz / m.rollout_T)} (t_safe_s × ctrl_hz / rollout_T). Empty chart ≠ broken.` : "No episode finished yet (run.json has no t_safe_s to estimate when)";
}

function render() {
  if (!S.d) return;
  updateStatus();
  const i = cursorRow();
  setText("cursor_v", S.cursorPos >= 1000 ? "latest (live)" : `iteration ${fmt(S.d.x.iterations[i])}`);
  const m = S.meta || {};
  const git = m.git && m.git.commit ? `git ${m.git.commit.slice(0, 7)}${m.git.dirty ? "+" : ""}` : "";
  setHTML("metaline", [m.trainer && esc(m.trainer), m.n_envs ? (m.rollout_T ? `N ${m.n_envs} × T ${m.rollout_T}` : `N ${m.n_envs}`) : "", m.gamma != null ? `γ ${m.gamma}` : "", m.reward_terms ? `reward terms: ${m.reward_terms.map(esc).join(", ")}` : "", git].filter(Boolean).join(" · ") || esc(S.d.why || ""));
  if (S.tab === "train") { buildCharts(); drawCharts(); renderCards(); renderChecks(); renderFirstNote(); }
}

// ---------------------------------------------------------------- 표 (표본)
async function loadTable() {
  const id = S.sel; if (!id) return;
  const last = STOPS[+$("t_last").value];
  setText("t_last_v", last ? String(last) : "all");
  const q = { run: id, stream: $("t_stream").value || "main", rows: $("t_rows").value, cols: $("t_cols").value, last, map: $("t_map").value, stage: $("t_stage").value };
  if (cursorIter() != null) q.upto = S.d.x.total_timesteps[cursorRow()];
  const [t, all] = await Promise.all([api("/api/table", q), api("/api/table", { run: id, stream: "main", rows: "all", cols: "all", last: 0, upto: q.upto })]);
  if (S.sel !== id) return;
  S.sr = all.cells && all.cells[0] && all.cells[0][0].n ? { ...all.cells[0][0] } : null;
  renderCards();
  if (t.why || !t.cells || !t.cells.length) { setHTML("table", `<span class="muted">${esc(t.why || "no table")}</span>`); setText("t_note", ""); return; }
  for (const [sel, dim] of [["t_map", "map"], ["t_stage", "stage"]]) {
    const cur = $(sel).value;
    setHTML(sel, ['<option value="">all</option>'].concat((t.dims[dim] || []).map(n => `<option${n === cur ? " selected" : ""}>${esc(n)}</option>`)).join(""));
  }
  const met = $("t_metric").value, pct = !["t95", "t", "ret", "path_len"].includes(met);
  const hue = ["collision", "truncation", "declare_short"].includes(met) ? "--s2" : "--s1";
  const m = S.meta || {};
  setText("t_note", `⚠ sample: every episode of ${m.log_envs ?? "?"} logged envs (${t.total} episodes in this split). Last ${last || "all"} episodes per cell${cursorIter() != null ? `, up to iteration ${fmt(cursorIter())}` : ""}. n < 20 is faded. Compare with the rollout/* curves (all episodes).`);
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
  if (!$("more_charts") || !$("more_charts").open) return;   // 접혀 있으면 안 그림
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
  if (t === "train") { S.plots.forEach(p => p.sig = ""); render(); }
  if (t === "compare") { buildCompareList(); drawCompare(); }
}
document.querySelectorAll("#tabs button").forEach(b => b.onclick = () => setTab(b.dataset.tab));
$("xmode").onchange = e => { S.xmode = e.target.value; render(); };
$("ema").oninput = e => { S.ema = +e.target.value; setText("ema_v", S.ema.toFixed(2)); render(); };
setText("ema_v", S.ema.toFixed(2));
$("cursor").oninput = e => { S.cursorPos = +e.target.value; render(); debounce("tbl", loadTable, 250); replay.refreshList(); };
$("cursor_off").onclick = () => { S.cursorPos = 1000; $("cursor").value = 1000; render(); loadTable(); replay.refreshList(); };
addEventListener("resize", () => debounce("rs", () => { S.plots.forEach(p => p.sig = ""); render(); drawCompare(); }, 150));
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { S.plots.forEach(p => p.sig = ""); render(); drawCompare(); });

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
  const tableKeys = ["rollout/success_rate", "rollout/coverage_mean", "rollout/t95_mean", "rollout/collision_rate", "rollout/declare_short_rate", "eval/success_rate", "eval/t95_mean"].filter(k => union.includes(k));
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
loadRuns().then(() => { if (h0.get("tab")) setTab(h0.get("tab")); });
window.__trainview = S;
