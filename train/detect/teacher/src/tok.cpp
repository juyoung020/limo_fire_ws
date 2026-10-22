#include "tok.h"

#include <cctype>
#include <climits>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>

ClipTok::ClipTok(const std::string& vocab_json, const std::string& merges_txt) {
  std::ifstream fv(vocab_json);
  if (!fv) throw std::runtime_error("열 수 없음: " + vocab_json);
  auto j = nlohmann::json::parse(fv);
  for (auto& [k, v] : j.items()) vocab_[k] = v.get<int>();
  std::ifstream fm(merges_txt);
  if (!fm) throw std::runtime_error("열 수 없음: " + merges_txt);
  std::string line;
  int r = 0;
  while (std::getline(fm, line)) {
    if (line.empty() || line[0] == '#') continue;
    auto sp = line.find(' ');
    if (sp == std::string::npos) continue;
    rank_[{line.substr(0, sp), line.substr(sp + 1)}] = r++;
  }
}

std::vector<std::string> ClipTok::bpe(const std::string& word) const {
  // 글자 단위, 마지막 글자에 "</w>"
  std::vector<std::string> w;
  for (size_t i = 0; i < word.size(); ++i) w.push_back(std::string(1, word[i]));
  w.back() += "</w>";
  while (w.size() > 1) {
    int best = INT_MAX;
    size_t at = 0;
    for (size_t i = 0; i + 1 < w.size(); ++i) {
      auto it = rank_.find({w[i], w[i + 1]});
      if (it != rank_.end() && it->second < best) best = it->second, at = i;
    }
    if (best == INT_MAX) break;
    // 같은 쌍을 왼쪽부터 모두 합친다(원본 bpe 와 같음)
    std::string a = w[at], b = w[at + 1];
    std::vector<std::string> nw;
    for (size_t i = 0; i < w.size();) {
      if (i + 1 < w.size() && w[i] == a && w[i + 1] == b) {
        nw.push_back(a + b);
        i += 2;
      } else
        nw.push_back(w[i++]);
    }
    w.swap(nw);
  }
  return w;
}

void ClipTok::encode(const std::string& text, int L, int* ids, int* mask) const {
  // 소문자, 공백 정리, 정규식 [\p{L}]+ | [\p{N}] | [^\s\p{L}\p{N}]+ 의 ASCII 판
  std::vector<std::string> words;
  size_t i = 0;
  while (i < text.size()) {
    unsigned char c = (unsigned char)text[i];
    if (c >= 128) throw std::runtime_error("ASCII 프롬프트만: " + text);
    if (std::isspace(c)) {
      ++i;
      continue;
    }
    std::string w;
    if (std::isalpha(c)) {
      while (i < text.size() && std::isalpha((unsigned char)text[i])) w += (char)std::tolower(text[i++]);
    } else if (std::isdigit(c)) {
      w = text[i++];
    } else {
      while (i < text.size() && !std::isspace((unsigned char)text[i]) && !std::isalnum((unsigned char)text[i]))
        w += text[i++];
    }
    // 축약형 's 't 're 've 'm 'll 'd 는 영어 프롬프트에 안 쓴다고 보고 구두점 묶음으로 둔다
    words.push_back(w);
  }
  std::vector<int> out = {49406};
  for (auto& w : words)
    for (auto& t : bpe(w)) {
      auto it = vocab_.find(t);
      if (it == vocab_.end()) throw std::runtime_error("어휘에 없음: " + t);
      out.push_back(it->second);
    }
  out.push_back(49407);
  if ((int)out.size() > L) throw std::runtime_error("프롬프트가 너무 김: " + text);
  for (int k = 0; k < L; ++k) {
    ids[k] = k < (int)out.size() ? out[k] : 49407;
    mask[k] = k < (int)out.size() ? 1 : 0;
  }
}
