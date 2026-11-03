// runfolder.h — 탐사 RL 러너(C++)가 학습 뷰어(trainview)의 실행 폴더 규약대로 쓰는 헤더 하나. 표준 라이브러리 + POSIX 만 쓴다.
//
// 형식의 원본은 viewer/trainfmt(Rust)다. 이 헤더는 그 규약을 그대로 쓰는 쪽이고, 어긋나면 `runcheck <실행 폴더>` 가 잡는다.
// fake_run(viewer/tools/fake_run.cpp)도 이 헤더로 쓴다 — 진짜 러너와 같은 길.
//
//   <root>/latest.txt                기본으로 열 실행 이름
//   <root>/<이름>/run.json           설정·상수(schema 1) — 이름 바꾸기로 씀
//                progress.jsonl      업데이트마다(≤ 1 줄/s 로 합침) 한 줄 — 덧붙이기
//                episodes.jsonl      판마다 한 줄(표본) — 덧붙이기, 1 GiB 마다 episodes.1.jsonl …
//                events.txt          사람이 읽는 사건 — 덧붙이기
//                evals/<이름>.json    평가 표 — 이름 바꾸기
//                replays/ep_<판>_<결과>.xrp   판 하나 2D 재생 — 이름 바꾸기
//                s_<줄기>/episodes_<줄기>.jsonl, s_<줄기>/replays/   평가 등 집계에 섞지 않는 판
//
// 쓰임(러너의 로그 스레드 하나에서만 부른다 — 스레드 안전하지 않다):
//
//   rf::Json meta = rf::Json::object();
//   meta.set("group", "explore_base").set("seed", 1).set("n_envs", 4096).set("ctrl_hz", 10);
//   rf::RunFolder::Options o;                 // 이어 하기면 o.resume = true; o.resume_iter = <체크포인트 이터>;
//   rf::RunFolder run("train/explore/runs/explore_base_s1", meta, o);
//   // 업데이트마다:
//   auto& r = run.row();
//   r.last(rf::key::ITER, it); r.last(rf::key::STEPS, steps);
//   r.sum(rf::key::N_EPISODES, n_done);
//   r.mean_w(rf::key::SUCCESS_RATE, sr, n_done);     // n_done == 0 이면 키가 빠진다(잰 것만 쓴다)
//   r.mean_w(rf::reward_key("coverage"), cov_sum_mean, n_done);
//   r.mean(rf::key::APPROX_KL, kl);
//   run.end_update();                                 // 1 초 지났으면 한 줄 씀
//   // 판이 끝나면(표본 환경만):
//   rf::Json ep = rf::Json::object(); ep.set("iter", it).set("env_steps", steps).set("ep", run.new_episode_id()) …;
//   run.episode(ep);                                  // 평가 판은 run.episode(ep, "eval")
//   // 끝:
//   run.finish();
#pragma once

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace rf {

// =============================================================================================
// 작은 JSON (run.json·줄 하나를 만들고, 재개 때 옛 run.json·줄을 읽는 데만 쓴다)

class Json {
 public:
  enum Type { Null, Bool, Num, Str, Arr, Obj };
  Json() = default;
  Json(std::nullptr_t) {}
  Json(bool b) : t_(Bool), b_(b) {}
  Json(int v) : t_(Num), n_(v) {}
  Json(long v) : t_(Num), n_(double(v)) {}
  Json(long long v) : t_(Num), n_(double(v)) {}
  Json(unsigned v) : t_(Num), n_(v) {}
  Json(unsigned long v) : t_(Num), n_(double(v)) {}
  Json(unsigned long long v) : t_(Num), n_(double(v)) {}
  Json(double v) : t_(Num), n_(v) {}
  Json(float v) : t_(Num), n_(v) {}
  Json(const char* s) : t_(Str), s_(s) {}
  Json(std::string s) : t_(Str), s_(std::move(s)) {}
  static Json object() { Json j; j.t_ = Obj; return j; }
  static Json array() { Json j; j.t_ = Arr; return j; }
  template <class T>
  static Json array(const std::vector<T>& v) { Json j = array(); for (auto& x : v) j.push(Json(x)); return j; }

  Type type() const { return t_; }
  bool is_null() const { return t_ == Null; }
  bool is_num() const { return t_ == Num; }
  bool is_obj() const { return t_ == Obj; }
  bool is_arr() const { return t_ == Arr; }
  double num(double d = NAN) const { return t_ == Num ? n_ : d; }
  const std::string& str() const { return s_; }
  bool boolean() const { return t_ == Bool && b_; }
  const std::vector<Json>& items() const { return a_; }
  const std::vector<std::pair<std::string, Json>>& fields() const { return o_; }

  // 객체: 있으면 바꾸고 없으면 뒤에 더함(차례 유지)
  Json& set(const std::string& k, Json v) {
    if (t_ == Null) t_ = Obj;
    for (auto& kv : o_) if (kv.first == k) { kv.second = std::move(v); return *this; }
    o_.emplace_back(k, std::move(v));
    return *this;
  }
  bool has(const std::string& k) const { return find(k) != nullptr; }
  const Json* find(const std::string& k) const {
    for (auto& kv : o_) if (kv.first == k) return &kv.second;
    return nullptr;
  }
  Json* find(const std::string& k) {
    for (auto& kv : o_) if (kv.first == k) return &kv.second;
    return nullptr;
  }
  const Json& operator[](const std::string& k) const { static const Json nul; auto p = find(k); return p ? *p : nul; }
  void erase(const std::string& k) {
    o_.erase(std::remove_if(o_.begin(), o_.end(), [&](auto& kv) { return kv.first == k; }), o_.end());
  }
  Json& push(Json v) { if (t_ == Null) t_ = Arr; a_.push_back(std::move(v)); return *this; }

  std::string dump(int indent = -1) const { std::string o; dump_to(o, indent, 0); return o; }

  static void num_to(std::string& o, double v) {
    if (!std::isfinite(v)) { o += "null"; return; }
    char b[40];
    if (v == std::floor(v) && std::fabs(v) < 9e15) std::snprintf(b, sizeof b, "%.0f", v);
    else if (std::fabs(v) >= 1e5) std::snprintf(b, sizeof b, "%.15g", v);  // 시각(ts)·큰 값은 다 씀
    else std::snprintf(b, sizeof b, "%.7g", v);  // 유효 숫자 7 자리(f32 정밀도) — 줄 크기를 줄임
    o += b;
  }
  static void str_to(std::string& o, const std::string& s) {
    o += '"';
    for (unsigned char c : s) {
      switch (c) {
        case '"': o += "\\\""; break;
        case '\\': o += "\\\\"; break;
        case '\n': o += "\\n"; break;
        case '\r': o += "\\r"; break;
        case '\t': o += "\\t"; break;
        default:
          if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
          else o += char(c);
      }
    }
    o += '"';
  }

  // 해석. 실패하면 Null 과 ok = false.
  static Json parse(const std::string& s, bool* ok = nullptr) {
    size_t i = 0;
    Json j;
    bool good = parse_v(s, i, j);
    skip_ws(s, i);
    if (i != s.size()) good = false;
    if (ok) *ok = good;
    return good ? j : Json();
  }

 private:
  Type t_ = Null;
  bool b_ = false;
  double n_ = 0;
  std::string s_;
  std::vector<Json> a_;
  std::vector<std::pair<std::string, Json>> o_;

  void dump_to(std::string& o, int ind, int depth) const {
    auto nl = [&](int d) { if (ind >= 0) { o += '\n'; o.append(size_t(ind * d), ' '); } };
    switch (t_) {
      case Null: o += "null"; break;
      case Bool: o += b_ ? "true" : "false"; break;
      case Num: num_to(o, n_); break;
      case Str: str_to(o, s_); break;
      case Arr:
        o += '[';
        for (size_t k = 0; k < a_.size(); k++) { if (k) o += ','; nl(depth + 1); a_[k].dump_to(o, ind, depth + 1); }
        if (!a_.empty()) nl(depth);
        o += ']';
        break;
      case Obj:
        o += '{';
        for (size_t k = 0; k < o_.size(); k++) {
          if (k) o += ',';
          nl(depth + 1);
          str_to(o, o_[k].first);
          o += ind >= 0 ? ": " : ":";
          o_[k].second.dump_to(o, ind, depth + 1);
        }
        if (!o_.empty()) nl(depth);
        o += '}';
        break;
    }
  }
  static void skip_ws(const std::string& s, size_t& i) { while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) i++; }
  static void utf8(std::string& o, uint32_t c) {
    if (c < 0x80) o += char(c);
    else if (c < 0x800) { o += char(0xC0 | (c >> 6)); o += char(0x80 | (c & 0x3F)); }
    else if (c < 0x10000) { o += char(0xE0 | (c >> 12)); o += char(0x80 | ((c >> 6) & 0x3F)); o += char(0x80 | (c & 0x3F)); }
    else { o += char(0xF0 | (c >> 18)); o += char(0x80 | ((c >> 12) & 0x3F)); o += char(0x80 | ((c >> 6) & 0x3F)); o += char(0x80 | (c & 0x3F)); }
  }
  static bool parse_str(const std::string& s, size_t& i, std::string& out) {
    if (i >= s.size() || s[i] != '"') return false;
    i++;
    while (i < s.size()) {
      char c = s[i++];
      if (c == '"') return true;
      if (c != '\\') { out += c; continue; }
      if (i >= s.size()) return false;
      char e = s[i++];
      switch (e) {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/': out += '/'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u': {
          if (i + 4 > s.size()) return false;
          uint32_t c1 = uint32_t(std::stoul(s.substr(i, 4), nullptr, 16));
          i += 4;
          if (c1 >= 0xD800 && c1 < 0xDC00 && i + 6 <= s.size() && s[i] == '\\' && s[i + 1] == 'u') {
            uint32_t c2 = uint32_t(std::stoul(s.substr(i + 2, 4), nullptr, 16));
            i += 6;
            c1 = 0x10000 + ((c1 - 0xD800) << 10) + (c2 - 0xDC00);
          }
          utf8(out, c1);
          break;
        }
        default: return false;
      }
    }
    return false;
  }
  static bool parse_v(const std::string& s, size_t& i, Json& j) {
    skip_ws(s, i);
    if (i >= s.size()) return false;
    char c = s[i];
    if (c == '{') {
      j = object();
      i++;
      skip_ws(s, i);
      if (i < s.size() && s[i] == '}') { i++; return true; }
      while (true) {
        skip_ws(s, i);
        std::string k;
        if (!parse_str(s, i, k)) return false;
        skip_ws(s, i);
        if (i >= s.size() || s[i] != ':') return false;
        i++;
        Json v;
        if (!parse_v(s, i, v)) return false;
        j.o_.emplace_back(std::move(k), std::move(v));
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',') { i++; continue; }
        if (i < s.size() && s[i] == '}') { i++; return true; }
        return false;
      }
    }
    if (c == '[') {
      j = array();
      i++;
      skip_ws(s, i);
      if (i < s.size() && s[i] == ']') { i++; return true; }
      while (true) {
        Json v;
        if (!parse_v(s, i, v)) return false;
        j.a_.push_back(std::move(v));
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',') { i++; continue; }
        if (i < s.size() && s[i] == ']') { i++; return true; }
        return false;
      }
    }
    if (c == '"') { j.t_ = Str; return parse_str(s, i, j.s_); }
    if (s.compare(i, 4, "true") == 0) { i += 4; j = Json(true); return true; }
    if (s.compare(i, 5, "false") == 0) { i += 5; j = Json(false); return true; }
    if (s.compare(i, 4, "null") == 0) { i += 4; j = Json(); return true; }
    const char* b = s.c_str() + i;
    char* e = nullptr;
    double v = std::strtod(b, &e);
    if (e == b) return false;
    i += size_t(e - b);
    j = Json(v);
    return true;
  }
};

// =============================================================================================
// progress 키 이름 — trainfmt/src/keys.rs 와 같은 표(표준 SB3 꼴 + 탐사 지표)

namespace key {
// time/
inline constexpr const char* ITER = "time/iterations";            // 필수
inline constexpr const char* STEPS = "time/total_timesteps";      // 필수, 단조 증가
inline constexpr const char* ELAPSED = "time/time_elapsed";       // 안 넣으면 RunFolder 가 넣음(재개해도 이어 셈)
inline constexpr const char* FPS = "time/fps";
inline constexpr const char* ITER_MS = "time/iter_ms";
inline constexpr const char* ROLLOUT_MS = "time/rollout_ms";
inline constexpr const char* UPDATE_MS = "time/update_ms";
// rollout/ — 그 사이 끝난 모든 판. 비율은 mean_w(값, 판 수) — 판이 0 이면 키가 빠진다
inline constexpr const char* N_EPISODES = "rollout/n_episodes";   // sum
inline constexpr const char* EP_REW_MEAN = "rollout/ep_rew_mean";
inline constexpr const char* EP_LEN_MEAN = "rollout/ep_len_mean";
inline constexpr const char* SUCCESS_RATE = "rollout/success_rate";
inline constexpr const char* COLLISION_RATE = "rollout/collision_rate";
inline constexpr const char* TRUNCATION_RATE = "rollout/truncation_rate";
inline constexpr const char* DECLARE_RATE = "rollout/declare_rate";
inline constexpr const char* DECLARE_SHORT_RATE = "rollout/declare_short_rate";
inline constexpr const char* COVERAGE_MEAN = "rollout/coverage_mean";
inline constexpr const char* COVERAGE_AT_DECLARE = "rollout/coverage_at_declare_mean";  // 가중치 = 선언한 판 수
inline constexpr const char* T95_MEAN = "rollout/t95_mean";                 // 가중치 = 95 % 도달 판 수
inline constexpr const char* T95_REACHED_RATE = "rollout/t95_reached_rate";
inline constexpr const char* PATH_LEN_MEAN = "rollout/path_len_mean";
inline constexpr const char* KNOWN_AREA_MEAN = "rollout/known_area_mean";
inline constexpr const char* STEP_REWARD_MEAN = "rollout/step_reward_mean";
// train/
inline constexpr const char* PG_LOSS = "train/policy_gradient_loss";
inline constexpr const char* VALUE_LOSS = "train/value_loss";
inline constexpr const char* ENTROPY_LOSS = "train/entropy_loss";  // = −엔트로피
inline constexpr const char* APPROX_KL = "train/approx_kl";
inline constexpr const char* CLIP_FRACTION = "train/clip_fraction";
inline constexpr const char* EXPLAINED_VAR = "train/explained_variance";
inline constexpr const char* GRAD_NORM = "train/grad_norm";
inline constexpr const char* LEARNING_RATE = "train/learning_rate";
inline constexpr const char* LOSS = "train/loss";
inline constexpr const char* STD = "train/std";
inline constexpr const char* LOG_STD = "train/log_std";
inline constexpr const char* ADV_STD = "train/advantage_std";
inline constexpr const char* ADV_MEAN = "train/advantage_mean";
inline constexpr const char* VALUE_MEAN = "train/value_mean";
inline constexpr const char* DECLARE_LOGIT = "train/declare_logit_mean";
inline constexpr const char* DECLARE_PROB = "train/declare_prob_mean";
// eval/
inline constexpr const char* EVAL_N = "eval/n_episodes";
inline constexpr const char* EVAL_REWARD = "eval/mean_reward";
inline constexpr const char* EVAL_LEN = "eval/mean_ep_length";
inline constexpr const char* EVAL_SUCCESS = "eval/success_rate";
inline constexpr const char* EVAL_COLLISION = "eval/collision_rate";
inline constexpr const char* EVAL_TRUNCATION = "eval/truncation_rate";
inline constexpr const char* EVAL_DECLARE_SHORT = "eval/declare_short_rate";
inline constexpr const char* EVAL_COVERAGE = "eval/coverage_mean";
inline constexpr const char* EVAL_COVERAGE_AT_DECLARE = "eval/coverage_at_declare_mean";
inline constexpr const char* EVAL_T95 = "eval/t95_mean";
inline constexpr const char* EVAL_T95_REACHED = "eval/t95_reached_rate";
inline constexpr const char* EVAL_PATH_LEN = "eval/path_len_mean";
inline constexpr const char* EVAL_FRONTIER_T95 = "eval/frontier_t95_mean";
// 그 밖
inline constexpr const char* CURRICULUM_STAGE = "curriculum/stage";
inline constexpr const char* GPU_MEM_MB = "gpu/mem_used_mb";
}  // namespace key
inline std::string reward_key(const std::string& term) { return "reward/" + term; }        // 판당 항 합 평균
inline std::string eval_reward_key(const std::string& term) { return "eval_reward/" + term; }
inline std::string std_key(const std::string& comp) { return "train/std/" + comp; }        // v | vy | w
inline std::string entropy_key(const std::string& comp) { return "train/entropy/" + comp; }  // v | vy | w | declare
inline std::string dropped_key(const std::string& what) { return "log/" + what + "_dropped"; }

// 판 끝 사유(episodes.jsonl outcome)
namespace outcome {
inline constexpr const char* SUCCESS = "success";                // 선언, 기준 덮음 이상
inline constexpr const char* DECLARED_SHORT = "declared_short";  // 선언, 기준 미달
inline constexpr const char* COLLISION = "collision";            // 충돌(종료)
inline constexpr const char* TRUNCATED = "truncated";            // 안전 시간(끊김)
}  // namespace outcome

// =============================================================================================
// 파일 도우미

inline double now_ts() {
  timespec ts{};
  clock_gettime(CLOCK_REALTIME, &ts);
  return double(ts.tv_sec) + ts.tv_nsec * 1e-9;
}

// /proc/<pid>/stat 22 번째 칸(시작 시각, 클럭 틱). 뷰어가 pid 재사용을 거르는 데 쓴다. 못 읽으면 0.
inline unsigned long long pid_start(long pid) {
  std::ifstream f("/proc/" + std::to_string(pid) + "/stat");
  std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  auto p = s.rfind(')');
  if (p == std::string::npos) return 0;
  std::istringstream is(s.substr(p + 2));
  std::string tok;
  for (int k = 0; k < 20 && (is >> tok); k++) {}  // 3 번째 칸(state)부터 20 개 → 22 번째
  try { return std::stoull(tok); } catch (...) { return 0; }
}

inline bool exists(const std::string& p) { struct stat st; return ::stat(p.c_str(), &st) == 0; }
inline long long file_size(const std::string& p) { struct stat st; return ::stat(p.c_str(), &st) == 0 ? (long long)st.st_size : -1; }

inline void mkdirs(const std::string& p) {
  std::string cur;
  for (size_t i = 0; i <= p.size(); i++) {
    if (i == p.size() || p[i] == '/') {
      if (!cur.empty() && !exists(cur) && ::mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST)
        throw std::runtime_error("runfolder: cannot create " + cur + ": " + std::strerror(errno));
    }
    if (i < p.size()) cur += p[i];
  }
}

inline std::string read_file(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// .tmp 에 다 쓴 뒤 이름 바꾸기 — 읽는 쪽에 반쪽 파일이 안 보인다
inline void write_atomic(const std::string& path, const std::string& bytes) {
  std::string tmp = path + ".tmp";
  {
    std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("runfolder: cannot write " + tmp);
    f.write(bytes.data(), std::streamsize(bytes.size()));
    if (!f) throw std::runtime_error("runfolder: write failed " + tmp);
  }
  if (std::rename(tmp.c_str(), path.c_str()) != 0) throw std::runtime_error("runfolder: rename failed " + path + ": " + std::strerror(errno));
}

inline std::string basename_of(const std::string& p) {
  std::string s = p;
  while (s.size() > 1 && s.back() == '/') s.pop_back();
  auto k = s.rfind('/');
  return k == std::string::npos ? s : s.substr(k + 1);
}
inline std::string parent_of(const std::string& p) {
  std::string s = p;
  while (s.size() > 1 && s.back() == '/') s.pop_back();
  auto k = s.rfind('/');
  return k == std::string::npos ? "." : (k == 0 ? "/" : s.substr(0, k));
}

inline std::vector<std::string> list_dir(const std::string& d) {
  std::vector<std::string> out;
  if (DIR* dp = ::opendir(d.c_str())) {
    while (dirent* e = ::readdir(dp)) {
      std::string n = e->d_name;
      if (n != "." && n != "..") out.push_back(n);
    }
    ::closedir(dp);
  }
  std::sort(out.begin(), out.end());
  return out;
}

// 명령 출력(git 용). 실패하면 빈 글자.
inline std::string run_cmd(const std::string& cmd) {
  std::string out;
  if (FILE* p = ::popen(cmd.c_str(), "r")) {
    char b[512];
    size_t n;
    while ((n = std::fread(b, 1, sizeof b, p)) > 0) out.append(b, n);
    if (::pclose(p) != 0) return "";
  }
  while (!out.empty() && (out.back() == '\n' || out.back() == ' ')) out.pop_back();
  return out;
}

// 실행 폴더가 든 저장소의 커밋·더러움. 못 얻으면 빈 객체.
inline Json git_info(const std::string& dir) {
  Json g = Json::object();
  std::string q = "'" + dir + "'";
  std::string c = run_cmd("git -C " + q + " rev-parse HEAD 2>/dev/null");
  if (!c.empty()) g.set("commit", c);
  if (!c.empty()) g.set("dirty", !run_cmd("git -C " + q + " status --porcelain --untracked-files=no 2>/dev/null").empty());
  return g;
}

// =============================================================================================
// progress 한 줄 모으기: 키마다 합치는 법이 다르다(비율 = 판 수 가중 평균, 손실 = 평균, 이터·스텝 = 마지막, 판 수·버림 = 합)

class Row {
 public:
  void mean_w(const std::string& k, double v, double w) { put(k, MEAN, v, w); }  // w ≤ 0(판 없음)이면 넣지 않음
  void mean(const std::string& k, double v) { put(k, MEAN, v, 1); }
  void last(const std::string& k, double v) { put(k, LAST, v, 1); }
  void sum(const std::string& k, double v) { put(k, SUM, v, 1); }
  bool has(const std::string& k) const { return m_.count(k) > 0; }
  double get(const std::string& k, double d = NAN) const {
    auto it = m_.find(k);
    if (it == m_.end()) return d;
    return it->second.how == MEAN ? it->second.s / it->second.w : it->second.s;
  }
  bool empty() const { return m_.empty(); }  // 키가 하나도 없으면 줄을 쓰지 않는다
  void tick() { n_++; }
  // 한 줄로 만들고 비운다
  Json take() {
    Json o = Json::object();
    for (auto& kv : m_) o.set(kv.first, kv.second.how == MEAN ? kv.second.s / kv.second.w : kv.second.s);
    if (n_ > 1) o.set("time/iterations_merged", n_);
    m_.clear();
    n_ = 0;
    return o;
  }

 private:
  enum How { MEAN, LAST, SUM };
  struct E { How how; double s = 0, w = 0; };
  std::map<std::string, E> m_;
  unsigned n_ = 0;
  void put(const std::string& k, How how, double v, double w) {
    if (!std::isfinite(v) || !std::isfinite(w) || w <= 0) return;  // 안 잰 값은 키를 뺀다(0·null 을 쓰지 않음)
    auto& e = m_[k];
    e.how = how;
    if (how == MEAN) { e.s += v * w; e.w += w; }
    else if (how == LAST) { e.s = v; e.w = 1; }
    else { e.s += v; e.w = 1; }
  }
};

// =============================================================================================
// 실행 폴더

struct RunOptions {
  bool resume = false;         // 같은 폴더에 이어 쓴다(체크포인트에서 이어 학습)
  double resume_iter = -1;     // resume 이면 필수: 이 이터 이상인 progress·episodes 줄을 지운다(되감기)
  double resume_steps = NAN;   // 그 체크포인트의 time/total_timesteps(segments 에 적음)
  bool set_latest = true;      // 부모 폴더 latest.txt 를 이 실행 이름으로
  double every_s = 1.0;        // progress 줄 최소 간격(초). 0 이면 업데이트마다
};

class RunFolder {
 public:
  static constexpr long long EPISODES_ROTATE_BYTES = 1LL << 30;  // trainfmt::EPISODES_ROTATE_BYTES 와 같음
  using Options = RunOptions;

  // meta = run.json 내용(group, seed, n_envs, ctrl_hz, reward_terms, curriculum, refs, …).
  // schema·kind·name·pid·pid_start·started·segments·git 은 여기서 채운다. 실패하면 std::runtime_error.
  RunFolder(const std::string& dir, Json meta, const Options& opt = Options()) : dir_(dir), meta_(std::move(meta)), opt_(opt) {
    if (!meta_.is_obj()) throw std::runtime_error("runfolder: meta must be a JSON object");
    if (opt_.resume && !(opt_.resume_iter >= 0)) throw std::runtime_error("runfolder: resume needs resume_iter (checkpoint iteration)");
    mkdirs(dir_);
    started_ = now_ts();
    const long pid = long(::getpid());
    Json segs = Json::array();
    Json seg = Json::object();
    seg.set("started", started_).set("pid", pid);
    if (opt_.resume) {
      bool ok = false;
      Json old = Json::parse(read_file(dir_ + "/run.json"), &ok);
      if (ok && old["segments"].is_arr()) segs = old["segments"];
      // 판 번호는 지운 줄까지 포함해 가장 큰 번호 다음부터(재생 파일 이름이 겹치지 않게)
      next_ep_ = std::max(next_ep_, max_episode_id() + 1);
      long long cut_p = trim_jsonl(dir_ + "/progress.jsonl", "time/iterations", opt_.resume_iter, &elapsed_base_);
      long long cut_e = 0;
      for (auto& base : episode_bases()) {
        for (int i = 0;; i++) {
          std::string p = episodes_file(base, i);
          if (!exists(p)) break;
          cut_e += trim_jsonl(p, "iter", opt_.resume_iter, nullptr);
        }
      }
      // 되감긴 구간의 재생 파일·평가 표는 지우지 않고 비켜 둔다(뷰어 목록에 옛 미래가 섞이지 않게)
      long long moved = set_aside_rewound(opt_.resume_iter);
      seg.set("from_iter", opt_.resume_iter).set("trimmed_progress_lines", cut_p).set("trimmed_episode_lines", cut_e).set("set_aside_files", moved);
      if (std::isfinite(opt_.resume_steps)) seg.set("from_steps", opt_.resume_steps);
    }
    segs.push(seg);
    meta_.set("schema", 1);
    if (!meta_.has("kind")) meta_.set("kind", "explore_ppo");
    if (!meta_.has("name")) meta_.set("name", basename_of(dir_));
    meta_.set("pid", pid);
    if (auto s = pid_start(pid)) meta_.set("pid_start", s);
    meta_.set("started", started_);
    meta_.set("segments", segs);
    if (!meta_.has("git")) meta_.set("git", git_info(dir_));
    meta_.erase("ended");
    if (!opt_.resume) {
      // 새 실행: 옛 줄을 비운다(같은 이름을 다시 쓰는 경우)
      write_atomic(dir_ + "/progress.jsonl", "");
      for (auto& base : episode_bases())
        for (int i = 0;; i++) { std::string p = episodes_file(base, i); if (!exists(p)) break; std::remove(p.c_str()); }
    }
    write_meta();
    progress_.open(dir_ + "/progress.jsonl", std::ios::app);
    if (!progress_) throw std::runtime_error("runfolder: cannot open progress.jsonl in " + dir_);
    if (opt_.set_latest) write_atomic(parent_of(dir_) + "/latest.txt", basename_of(dir_) + "\n");
    event(opt_.resume ? "resume from iteration " + fmt_num(opt_.resume_iter) : "start");
  }
  ~RunFolder() { if (!finished_) flush_row(); }
  RunFolder(const RunFolder&) = delete;
  RunFolder& operator=(const RunFolder&) = delete;

  const std::string& dir() const { return dir_; }
  Json& meta() { return meta_; }
  // 학습 시작부터 초(재개하면 앞 구간의 time_elapsed 에 이어 셈)
  double elapsed() const { return elapsed_base_ + (now_ts() - started_); }

  // ---- progress
  Row& row() { return row_; }
  // 업데이트 하나를 다 모은 뒤 부른다. 간격(every_s)이 지났으면 한 줄 쓴다.
  void end_update() {
    row_.tick();
    if (now_ts() - last_emit_ >= opt_.every_s) flush_row();
  }
  // 모은 것이 있으면 지금 한 줄(단계 바뀜·체크포인트·끝 직전)
  void flush_row() {
    if (row_.empty()) { row_.take(); return; }
    Json o = row_.take();
    if (!o.has("time/time_elapsed")) o.set("time/time_elapsed", elapsed());
    o.set("ts", now_ts());
    progress_ << o.dump() << '\n';
    progress_.flush();  // 줄 단위로 내보냄: 뷰어는 줄끝까지 온 줄만 읽는다
    last_emit_ = now_ts();
    rows_++;
  }
  long long rows_written() const { return rows_; }

  // ---- 판
  // 판 번호(재개해도 이어 셈). 판 줄의 "ep" 와 재생 파일 이름에 같은 번호를 쓴다.
  long long new_episode_id() { return next_ep_++; }
  // 판 한 줄. stream "" = 본 줄기(표본), "eval" = s_eval/episodes_eval.jsonl. "ts" 가 없으면 넣는다.
  void episode(Json line, const std::string& stream = "") {
    if (!line.has("ts")) line.set("ts", now_ts());
    auto& f = ep_file(stream);
    std::string s = line.dump();
    f.out << s << '\n';
    f.out.flush();
    f.bytes += (long long)s.size() + 1;
  }

  // ---- 평가 표: evals/<name>.json  {"iter", "ckpt", "rows": [{"eval", "split", "n", "success", "collision", "truncation", "coverage", "t95", "ref_t95", "pass"}]}
  void eval(const std::string& name, const Json& v) {
    mkdirs(dir_ + "/evals");
    write_atomic(dir_ + "/evals/" + name + ".json", v.dump(1));
  }

  // ---- 사건(사람이 읽음): events.txt 에 "<시각> <글>" 한 줄
  void event(const std::string& text) {
    std::ofstream f(dir_ + "/events.txt", std::ios::app);
    time_t t = time_t(now_ts());
    char b[32];
    std::strftime(b, sizeof b, "%Y-%m-%d %H:%M:%S", std::localtime(&t));
    f << b << ' ' << text << '\n';
  }

  // ---- 재생 파일 자리: <줄기>/replays/ep_<판 6 자리>_<결과>.xrp (폴더를 만든다)
  std::string replay_path(long long ep, const std::string& outcome, const std::string& stream = "") {
    std::string d = stream_dir(stream) + "/replays";
    mkdirs(d);
    char b[64];
    std::snprintf(b, sizeof b, "ep_%06lld_", ep);
    return d + "/" + b + outcome + ".xrp";
  }

  // run.json 칸 하나 바꾸고 다시 씀(드묾: 커리큘럼 단계 바뀜 등)
  void set_meta(const std::string& k, Json v) { meta_.set(k, std::move(v)); write_meta(); }

  // 끝: 남은 줄을 쓰고 run.json 에 ended 를 적는다(뷰어 불 = finished). extra 의 칸도 run.json 에 더함.
  void finish(const Json& extra = Json::object()) {
    flush_row();
    for (auto& kv : extra.fields()) meta_.set(kv.first, kv.second);
    meta_.set("ended", now_ts());
    write_meta();
    event("finish");
    finished_ = true;
  }

 private:
  struct EpFile { std::ofstream out; long long bytes = 0; int index = 0; std::string base; };
  std::string dir_;
  Json meta_;
  Options opt_;
  Row row_;
  std::ofstream progress_;
  std::map<std::string, std::unique_ptr<EpFile>> eps_;
  double started_ = 0, last_emit_ = -1e300, elapsed_base_ = 0;
  long long rows_ = 0, next_ep_ = 0;
  bool finished_ = false;

  static std::string fmt_num(double v) { std::string s; Json::num_to(s, v); return s; }
  void write_meta() { write_atomic(dir_ + "/run.json", meta_.dump(1)); }
  std::string stream_dir(const std::string& stream) const { return stream.empty() || stream == "main" ? dir_ : dir_ + "/s_" + stream; }
  std::string episodes_base(const std::string& stream) const {
    return stream.empty() || stream == "main" ? dir_ + "/episodes" : dir_ + "/s_" + stream + "/episodes_" + stream;
  }
  static std::string episodes_file(const std::string& base, int i) { return i == 0 ? base + ".jsonl" : base + "." + std::to_string(i) + ".jsonl"; }
  // 본 줄기 + 있는 s_<줄기> 들
  std::vector<std::string> episode_bases() const {
    std::vector<std::string> v{episodes_base("")};
    for (auto& n : list_dir(dir_))
      if (n.rfind("s_", 0) == 0) v.push_back(episodes_base(n.substr(2)));
    return v;
  }
  EpFile& ep_file(const std::string& stream) {
    std::string key = stream == "main" ? "" : stream;
    auto it = eps_.find(key);
    if (it == eps_.end()) {
      auto f = std::make_unique<EpFile>();
      f->base = episodes_base(key);
      mkdirs(parent_of(f->base));
      while (exists(episodes_file(f->base, f->index + 1))) f->index++;  // 이어 쓰기: 가장 뒤 파일
      it = eps_.emplace(key, std::move(f)).first;
      open_ep(*it->second);
    }
    auto& f = *it->second;
    if (f.bytes >= EPISODES_ROTATE_BYTES) { f.out.close(); f.index++; open_ep(f); }
    return f;
  }
  static void open_ep(EpFile& f) {
    std::string p = episodes_file(f.base, f.index);
    f.bytes = std::max(0LL, file_size(p));
    f.out.open(p, std::ios::app);
    if (!f.out) throw std::runtime_error("runfolder: cannot open " + p);
  }
  // 가장 큰 판 번호: 모든 episodes 줄의 "ep" 와 replays/ 파일 이름(ep_<n>_…)
  long long max_episode_id() const {
    long long mx = -1;
    for (auto& base : episode_bases()) {
      for (int i = 0;; i++) {
        std::string p = episodes_file(base, i);
        if (!exists(p)) break;
        std::ifstream f(p);
        std::string l;
        while (std::getline(f, l)) {
          bool ok = false;
          Json j = Json::parse(l, &ok);
          if (ok) mx = std::max(mx, (long long)j["ep"].num(-1));
        }
      }
      for (const char* sub : {"/replays", "/replays_rewound"})
        for (auto& n : list_dir(parent_of(base) + sub))
          if (n.rfind("ep_", 0) == 0) { try { mx = std::max(mx, std::stoll(n.substr(3))); } catch (...) {} }
    }
    return mx;
  }
  // 재개 되감기: 머리 meta.iter(재생) 또는 iter(평가 표)가 iter 이상인 파일을 <줄기>/replays_rewound/, evals_rewound/ 로 옮긴다. 옮긴 수.
  long long set_aside_rewound(double iter) const {
    long long moved = 0;
    auto move = [&](const std::string& from_dir, const std::string& name, const std::string& to_dir) {
      mkdirs(to_dir);
      if (std::rename((from_dir + "/" + name).c_str(), (to_dir + "/" + name).c_str()) == 0) moved++;
    };
    for (auto& base : episode_bases()) {
      std::string sd = parent_of(base), rd = sd + "/replays";
      for (auto& n : list_dir(rd)) {
        if (n.size() < 4 || n.compare(n.size() - 4, 4, ".xrp") != 0) continue;
        std::ifstream f(rd + "/" + n, std::ios::binary);
        char h8[8];
        if (!f.read(h8, 8) || std::memcmp(h8, "XRP1", 4) != 0) continue;
        uint32_t hl = uint32_t(uint8_t(h8[4])) | uint32_t(uint8_t(h8[5])) << 8 | uint32_t(uint8_t(h8[6])) << 16 | uint32_t(uint8_t(h8[7])) << 24;
        if (hl > (16u << 20)) continue;
        std::string hs(hl, '\0');
        if (!f.read(&hs[0], hl)) continue;
        f.close();
        bool ok = false;
        Json h = Json::parse(hs, &ok);
        if (ok && h["meta"]["iter"].num(-1) >= iter) move(rd, n, sd + "/replays_rewound");
      }
    }
    for (auto& n : list_dir(dir_ + "/evals")) {
      bool ok = false;
      Json e = Json::parse(read_file(dir_ + "/evals/" + n), &ok);
      if (ok && e["iter"].num(-1) >= iter) move(dir_ + "/evals", n, dir_ + "/evals_rewound");
    }
    return moved;
  }
  // 재개 되감기: key 값이 iter 이상인 줄과 해석 못 하는 줄(반쪽 줄)을 지운다. 지운 줄 수.
  // last_elapsed 가 있으면 남은 줄의 마지막 time/time_elapsed 를 돌려줌.
  static long long trim_jsonl(const std::string& path, const char* k, double iter, double* last_elapsed) {
    if (!exists(path)) return 0;
    std::ifstream f(path);
    std::string l, kept;
    long long cut = 0;
    while (std::getline(f, l)) {
      bool ok = false;
      Json j = Json::parse(l, &ok);
      double v = ok ? j[k].num() : NAN;
      if (ok && std::isfinite(v) && v < iter) {
        kept += l;
        kept += '\n';
        if (last_elapsed) { double e = j["time/time_elapsed"].num(); if (std::isfinite(e)) *last_elapsed = e; }
      } else {
        cut++;
      }
    }
    f.close();
    write_atomic(path, kept);
    return cut;
  }
};

// =============================================================================================
// 재생 파일(.xrp) 쓰개 — 형식은 trainfmt/src/xrp.rs. 판 하나를 모아 두었다가 finish 에서 한 번에 쓴다.

struct Grid {
  int w = 0, h = 0;
  double res = 0.05, ox = 0, oy = 0;  // 칸 (i, j) 가운데 = (ox + (i+½)res, oy + (j+½)res), 행 0 = 작은 y
};
namespace gt {
inline constexpr uint8_t FREE = 0, WALL = 1, OUTSIDE = 2;
}
namespace ev {
inline constexpr uint32_t COLLISION = 1, DECLARE = 2, TRUNCATED = 4, REACHED95 = 8;
}
inline constexpr uint16_t SCAN_NONE = 0xFFFF;

class ReplayWriter {
 public:
  // cols: 프레임 열 이름(필수 t, x, y, yaw — 나머지는 xrp.rs 표). gt_cells: w×h, gt::FREE/WALL/OUTSIDE
  ReplayWriter(std::vector<std::string> cols, double dt, const Grid& g, const uint8_t* gt_cells)
      : cols_(std::move(cols)), dt_(dt), g_(g), gt_(gt_cells, gt_cells + size_t(g.w) * g.h), obs_prev_(size_t(g.w) * g.h, -1),
        cov_prev_(size_t(g.w) * g.h, 0) {
    for (const char* r : {"t", "x", "y", "yaw"})
      if (col(r) < 0) throw std::runtime_error(std::string("ReplayWriter: required column '") + r + "' missing");
    if (g.w <= 0 || g.h <= 0 || !(g.res > 0)) throw std::runtime_error("ReplayWriter: empty grid");
  }
  int col(const std::string& name) const {
    for (size_t i = 0; i < cols_.size(); i++) if (cols_[i] == name) return int(i);
    return -1;
  }
  size_t n_frames() const { return n_; }

  // 라이다(선택): 빔 n 개, 각도 a0 + k·da (로봇 기준 라디안), 최대 거리 rmax(m)
  void set_lidar(int n, float a0, float da, float rmax) { lidar_n_ = n; a0_ = a0; da_ = da; rmax_ = rmax; }
  // 행동 한도(화면 막대 눈금): v, vy (m/s), w (rad/s)
  void set_limits(float v, float vy, float w) { lim_[0] = v; lim_[1] = vy; lim_[2] = w; }
  void set_reward_terms(const std::vector<std::string>& t) { terms_ = t; }

  // 프레임 하나(cols 차례). 그다음 이 프레임의 obs·covered·scan 을 부른다.
  void frame(const float* row) {
    frames_.insert(frames_.end(), row, row + cols_.size());
    n_++;
    if (lidar_n_ > 0) scan_.resize(n_ * size_t(lidar_n_), SCAN_NONE);
  }
  // 지금 관측 지도 전체(w×h, −1 모름 / 0–100 %). 쓰개가 앞 것과 비교해 16×16 타일마다 바뀐 칸을 감싼 사각형만 남긴다(처음은 전체).
  void obs(const int8_t* grid) {
    need_frame();
    const uint32_t f = uint32_t(n_ - 1);
    if (!obs_any_) {
      rect(f, 0, 0, g_.w - 1, g_.h - 1, grid);
      obs_any_ = true;
    } else {
      // TILE × TILE 타일마다 바뀐 칸을 감싼 사각형 하나 — 라이다가 넓게 훑어도 바뀐 곳만 남는다
      for (int ty = 0; ty < g_.h; ty += TILE)
        for (int tx = 0; tx < g_.w; tx += TILE) {
          int x0 = g_.w, y0 = g_.h, x1 = -1, y1 = -1;
          for (int j = ty; j < std::min(g_.h, ty + TILE); j++)
            for (int i = tx; i < std::min(g_.w, tx + TILE); i++)
              if (grid[size_t(j) * g_.w + i] != obs_prev_[size_t(j) * g_.w + i]) { x0 = std::min(x0, i); x1 = std::max(x1, i); y0 = std::min(y0, j); y1 = std::max(y1, j); }
          if (x1 >= 0) rect(f, x0, y0, x1, y1, grid);
        }
    }
    std::copy(grid, grid + size_t(g_.w) * g_.h, obs_prev_.begin());
  }
  // 지금 덮인 칸 표시(w×h, 0/1). 새로 덮인 칸만 남긴다(덮음은 늘기만 함 — 0 으로 돌아간 칸은 무시).
  void covered(const uint8_t* mask) {
    need_frame();
    std::vector<uint32_t> nw;
    for (size_t k = 0; k < cov_prev_.size(); k++)
      if (mask[k] && !cov_prev_[k]) { nw.push_back(uint32_t(k)); cov_prev_[k] = 1; }
    if (nw.empty()) return;
    put_u32(cov_, uint32_t(n_ - 1));
    put_u32(cov_, uint32_t(nw.size()));
    for (auto k : nw) put_u32(cov_, k);
  }
  // 이번 프레임 라이다 거리(m, lidar_n 개). 반사 없음 = 무한·NaN·≥ rmax
  void scan(const float* ranges_m) {
    need_frame();
    if (lidar_n_ <= 0) throw std::runtime_error("ReplayWriter: scan() without set_lidar()");
    uint16_t* d = &scan_[(n_ - 1) * size_t(lidar_n_)];
    for (int k = 0; k < lidar_n_; k++) {
      float r = ranges_m[k];
      d[k] = (std::isfinite(r) && r >= 0 && r < rmax_) ? uint16_t(std::min(65534.f, std::round(r * 1000.f))) : SCAN_NONE;
    }
  }

  // 바이트로. meta = 그 판의 episodes.jsonl 줄.
  std::string bytes(const Json& meta) const {
    std::vector<std::pair<std::string, std::string>> secs;
    secs.emplace_back("frames", std::string(reinterpret_cast<const char*>(frames_.data()), frames_.size() * 4));
    secs.emplace_back("gt", std::string(reinterpret_cast<const char*>(gt_.data()), gt_.size()));
    if (!obs_.empty()) secs.emplace_back("obs", obs_);
    if (!cov_.empty()) secs.emplace_back("cov", cov_);
    if (lidar_n_ > 0) secs.emplace_back("scan", std::string(reinterpret_cast<const char*>(scan_.data()), scan_.size() * 2));
    Json h = Json::object();
    h.set("format", "XRP1").set("meta", meta).set("dt", dt_).set("n_frames", (unsigned long long)n_).set("cols", Json::array(cols_));
    Json g = Json::object();
    g.set("w", g_.w).set("h", g_.h).set("res", g_.res).set("ox", g_.ox).set("oy", g_.oy);
    h.set("grid", g);
    if (lidar_n_ > 0) { Json l = Json::object(); l.set("n", lidar_n_).set("a0", a0_).set("da", da_).set("rmax", rmax_); h.set("lidar", l); }
    if (lim_[0] > 0) { Json l = Json::object(); l.set("v", lim_[0]).set("vy", lim_[1]).set("w", lim_[2]); h.set("limits", l); }
    if (!terms_.empty()) h.set("reward_terms", Json::array(terms_));
    Json sj = Json::array();
    size_t off = 0;
    for (auto& s : secs) {
      Json e = Json::object();
      e.set("name", s.first).set("off", (unsigned long long)off).set("len", (unsigned long long)s.second.size());
      sj.push(e);
      off = pad8(off + s.second.size());
    }
    h.set("sections", sj);
    std::string hs = h.dump();
    std::string out = "XRP1";
    put_u32(out, uint32_t(hs.size()));
    out += hs;
    out.resize(pad8(out.size()), '\0');
    const size_t d0 = out.size();  // 섹션 off 는 여기부터(머리 길이가 off 에 안 걸린다)
    for (auto& s : secs) {
      out.resize(d0 + pad8(out.size() - d0), '\0');
      out += s.second;
    }
    return out;
  }
  // .tmp 에 쓰고 이름 바꾸기. 쓴 바이트 수.
  size_t finish(const std::string& path, const Json& meta) const {
    if (n_ == 0) throw std::runtime_error("ReplayWriter: no frames");
    std::string b = bytes(meta);
    write_atomic(path, b);
    return b.size();
  }

 private:
  std::vector<std::string> cols_;
  double dt_;
  Grid g_;
  std::vector<uint8_t> gt_;
  std::vector<float> frames_;
  std::string obs_, cov_;
  std::vector<int8_t> obs_prev_;
  std::vector<uint8_t> cov_prev_;
  std::vector<uint16_t> scan_;
  std::vector<std::string> terms_;
  float lim_[3] = {0, 0, 0};
  int lidar_n_ = 0;
  float a0_ = 0, da_ = 0, rmax_ = 0;
  size_t n_ = 0;
  bool obs_any_ = false;

  static constexpr int TILE = 16;
  static size_t pad8(size_t n) { return (n + 7) & ~size_t(7); }
  static void put_u32(std::string& o, uint32_t v) { for (int k = 0; k < 4; k++) o += char((v >> (8 * k)) & 0xFF); }  // little-endian
  void need_frame() const { if (n_ == 0) throw std::runtime_error("ReplayWriter: call frame() first"); }
  void rect(uint32_t f, int x0, int y0, int x1, int y1, const int8_t* grid) {
    put_u32(obs_, f);
    for (int v : {x0, y0, x1, y1}) put_u32(obs_, uint32_t(v));
    for (int j = y0; j <= y1; j++) obs_.append(reinterpret_cast<const char*>(grid + size_t(j) * g_.w + x0), size_t(x1 - x0 + 1));
  }
};

}  // namespace rf
