// 판단 모델(llama-server /v1/systemone) 요청 · 답 · 보내는 쪽 틀 — 위층(train/explore/scripts/low_decider.h), 지시 클래스 풀기(class_ask.h),
// 실물 비차단 소켓(decider_client.h)과 시뮬 가짜 응답기(low_explore_check · dec_rollout)가 같이 쓴다. ROS 없음
#pragma once
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace lxd {

// 보낸 요청 하나
struct Req {
  long id = 0;
  int step = 0;
  std::string body;
  std::vector<int> cells;   // 글자 차례의 후보 서는 자리 칸
  std::vector<int> ids;     // 글자 차례의 안정 번호
  std::string why;          // 묻는 까닭 (period · start · goal failed or lost · approaching goal · current goal's frontier gone · much better candidate · max interval)
  std::string qid = "pick"; // 답을 읽을 질문 (pick = 후보 글자 · cls = 지시 클래스)
};
struct Answer {
  long id = 0;
  bool ok = false;
  std::string choice;   // 고른 것: pick 은 글자 ("A" …), 그 밖은 선택지 이름
  std::string err;
  double ms = 0;
};
class Transport {
 public:
  virtual ~Transport() = default;
  virtual void send(const Req& r) = 0;    // 기다리지 않음
  virtual bool poll(Answer* a) = 0;       // 와 있는 답 하나 (없으면 false) — 기다리지 않음
};

// 답 고르기: answers.<qid>.choice (llama-server /v1/systemone 의 choice 답). pick 은 글자 하나만 맞는 답
inline Answer parse_answer(long id, const std::string& raw, const std::string& qid = "pick") {
  Answer a;
  a.id = id;
  const auto j = nlohmann::json::parse(raw, nullptr, false);
  if (!j.is_object() || !j.contains("answers") || !j["answers"].is_object() || !j["answers"].contains(qid) || !j["answers"][qid].is_object() ||
      !j["answers"][qid].contains("choice") || !j["answers"][qid]["choice"].is_string()) {
    a.err = "answer shape";
    return a;
  }
  a.choice = j["answers"][qid]["choice"].get<std::string>();
  a.ok = qid == "pick" ? a.choice.size() == 1 && a.choice[0] >= 'A' && a.choice[0] <= 'Z' : !a.choice.empty();
  if (!a.ok) a.err = "unknown option " + a.choice;
  return a;
}

}  // namespace lxd
