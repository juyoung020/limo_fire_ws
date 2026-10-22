// CLIP BPE 토크나이저(영어 ASCII 프롬프트만). vocab.json·merges.txt 를 읽어 [32] 토큰·마스크를 만든다.
#pragma once
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

class ClipTok {
 public:
  ClipTok(const std::string& vocab_json, const std::string& merges_txt);
  // 시작·끝 토큰 포함, 길이 L 로 끝 토큰(49407) 채움. mask = 1 (유효)
  void encode(const std::string& text, int L, int* ids, int* mask) const;

 private:
  std::unordered_map<std::string, int> vocab_;
  std::map<std::pair<std::string, std::string>, int> rank_;
  std::vector<std::string> bpe(const std::string& word) const;
};
