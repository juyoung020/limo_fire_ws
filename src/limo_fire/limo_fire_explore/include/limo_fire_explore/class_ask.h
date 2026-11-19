// 지시 글 → 탐지기 클래스: 판단 모델(llama-server /v1/systemone)에 choice 질문 한 번 ("이 지시가 가리키는 클래스는? 목록 중 하나 / none").
// 실물 노드(/explore/explorer · 저장 지도 /explore/object_goto)와 시뮬(low_explore_check)이 같은 코드로 묻고 받는다. ROS 없음.
//   선택지 = 탐지기가 낼 수 있는 클래스 전체(class_names — 지금 물체 지도에 있는 것만이 아님, 밑줄은 띄어 씀) + none.
//   답이 none · 꼴 틀림 · 시간 끝이면 클래스 없음 (판단 모델 위층은 물체 칸 없이 지시 글만 보고 탐사 자리를 고름).
#pragma once
#include <algorithm>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>

#include "limo_fire_explore/decider_io.h"

namespace limo_fire_explore {

// 인식 설정 파일(limo_fire_perception config/yolo_*.yaml)의 class_names — 노드 칸 아무 데나 있는 첫 목록
inline std::vector<std::string> load_class_names(const std::string& file) {
  for (const auto& kv : YAML::LoadFile(file)) {
    const YAML::Node p = kv.second["ros__parameters"];
    if (p && p["class_names"]) return p["class_names"].as<std::vector<std::string>>();
  }
  return {};
}
// 여러 설정의 class_names 를 차례대로 합침 (겹치면 처음 것만)
inline std::vector<std::string> merge_class_names(const std::vector<std::vector<std::string>>& lists) {
  std::vector<std::string> out;
  for (const auto& l : lists)
    for (const auto& c : l)
      if (std::find(out.begin(), out.end(), c) == out.end()) out.push_back(c);
  return out;
}

// 설정 파일 이름들 → 합친 class_names. '/' 가 없는 이름은 base_dir 안 (노드: limo_fire_perception 설치 config/)
inline std::vector<std::string> load_class_files(const std::vector<std::string>& files, const std::string& base_dir) {
  std::vector<std::vector<std::string>> ls;
  for (const auto& f : files) ls.push_back(load_class_names(f.find('/') == std::string::npos ? base_dir + "/" + f : f));
  return merge_class_names(ls);
}

class ClassAsk {
 public:
  static constexpr const char* kNone = "none";
  explicit ClassAsk(std::vector<std::string> classes = {}) : classes_(std::move(classes)) {}
  const std::vector<std::string>& classes() const { return classes_; }
  // 선택지 이름 (밑줄 → 띄어쓰기: "potted plant")
  static std::string option_of(const std::string& cls) {
    std::string o = cls;
    for (auto& c : o) if (c == '_') c = ' ';
    return o;
  }
  // 요청 본문
  std::string body(const std::string& instruction) const {
    nlohmann::ordered_json crit = nlohmann::ordered_json::object();
    crit[kNone] = nullptr;
    for (const auto& c : classes_) crit[option_of(c)] = nullptr;
    nlohmann::ordered_json b;
    b["state"] = {{"task", "A user gave the robot an instruction. The robot finds objects with a camera object detector that only knows the classes listed as options."},
                  {"instruction", instruction}};
    b["questions"]["cls"] = {{"type", "choice"},
                             {"instructions", "Which detector class is the object the user's instruction asks the robot to find or go to? Answer none if the "
                                              "instruction is to explore or the object is not one of these classes."},
                             {"criteria", crit}};
    return b.dump();
  }
  // 답 → 클래스 이름 ("" = none · 실패 · 목록에 없음)
  std::string class_of(const lxd::Answer& a) const {
    if (!a.ok) return "";
    for (const auto& c : classes_) if (option_of(c) == a.choice) return c;
    return "";
  }
  // 묻기 (지시가 바뀔 때마다 — 앞 물음의 답은 번호가 달라 버림). 클래스 목록이 비면 묻지 않고 바로 "없음"
  void ask(lxd::Transport& tr, const std::string& instruction, int step) {
    instr_ = instruction;
    step_ = step;
    done_ = false;
    out_.clear();
    err_.clear();
    if (classes_.empty() || instruction.empty()) { pending_ = false; done_ = true; err_ = classes_.empty() ? "no class list" : ""; return; }
    lxd::Req r;
    r.id = ++id_;
    r.step = step;
    r.body = body(instruction);
    r.why = "instruction class";
    r.qid = "cls";
    pending_ = true;
    tr.send(r);
  }
  // 걸음마다: 답이 왔거나 timeout_steps 가 지났으면 true (그 지시의 클래스는 cls() — 빈 글 = 없음). 끝난 뒤에는 false
  bool poll(lxd::Transport& tr, int step, int timeout_steps) {
    if (done_) { done_ = false; return true; }   // ask 에서 바로 끝난 것 (목록 없음 · 빈 지시)
    lxd::Answer a;
    bool got = false;
    while (tr.poll(&a)) {
      if (!pending_ || a.id != id_) continue;   // 옛 지시의 답
      pending_ = false;
      got = true;
      out_ = class_of(a);
      err_ = a.ok ? "" : a.err;
      ms_ = a.ms;
    }
    if (!got && pending_ && step - step_ > timeout_steps) { pending_ = false; got = true; out_.clear(); err_ = "timeout"; }
    return got;
  }
  bool pending() const { return pending_; }
  const std::string& instruction() const { return instr_; }
  const std::string& cls() const { return out_; }
  const std::string& err() const { return err_; }   // 마지막 실패 까닭 (빈 글 = 답을 받음)
  double ms() const { return ms_; }

 private:
  std::vector<std::string> classes_;
  std::string instr_, out_, err_;
  long id_ = 0;
  int step_ = 0;
  bool pending_ = false, done_ = false;
  double ms_ = 0;
};

}  // namespace limo_fire_explore
