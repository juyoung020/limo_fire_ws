// SigLIP2 글 토크나이저 (Gemma BPE) — HF tokenizers 의 tokenizer.json 과 같은 결과를 C++ 만으로.
//   소문자로 (tokenizer_config do_lower_case) → 띄어쓰기를 '▁' 로 (normalizer Replace) → 한 글자(UTF-8)씩 기호 →
//   병합 순위가 가장 낮은 이웃 쌍부터 합침 (BPE merges, 순위 = 줄 번호) → 어휘에 없는 글자는 바이트 토큰 <0xXX> (byte_fallback) →
//   끝에 <eos> (post_processor) → 길이 max_len 까지 <pad> 로 채움 (SiglipProcessor padding="max_length", 64).
// 자료: siglip2_vocab.txt (줄 번호 = 토큰 번호) · siglip2_merges.txt ("왼쪽\t오른쪽", 줄 번호 = 순위) — README "SigLIP2" 의 변환으로 만든다.
#pragma once
#include <cctype>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace limo_fire_perception
{

class SiglipTokenizer
{
public:
  SiglipTokenizer(const std::string & vocab_file, const std::string & merges_file)
  {
    std::ifstream v(vocab_file);
    if (!v) {throw std::runtime_error("SigLIP2 어휘 파일 없음: " + vocab_file);}
    std::string line;
    int id = 0;
    while (std::getline(v, line)) {vocab_.emplace(unesc(line), id++);}
    std::ifstream m(merges_file);
    if (!m) {throw std::runtime_error("SigLIP2 병합 파일 없음: " + merges_file);}
    int rank = 0;
    while (std::getline(m, line)) {
      const auto tab = line.find('\t');
      if (tab != std::string::npos) {merges_.emplace(unesc(line.substr(0, tab)) + '\x01' + unesc(line.substr(tab + 1)), rank);}
      ++rank;
    }
    eos_ = at("<eos>");
    pad_ = at("<pad>");
  }

  // 글 → 토큰 번호 max_len 개 (<eos> 포함, 나머지 <pad>). 넘치면 자르고 마지막을 <eos>
  std::vector<int64_t> encode(const std::string & text, size_t max_len = 64) const
  {
    std::string s;
    for (unsigned char c : text) {s += (c < 0x80) ? static_cast<char>(std::tolower(c)) : static_cast<char>(c);}
    std::string norm;
    for (char c : s) {
      if (c == ' ') {norm += "\xE2\x96\x81";} else {norm += c;}   // '▁' U+2581
    }
    // UTF-8 글자 단위 기호
    std::vector<std::string> sym;
    for (size_t i = 0; i < norm.size(); ) {
      const unsigned char c = static_cast<unsigned char>(norm[i]);
      const size_t n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
      sym.push_back(norm.substr(i, n));
      i += n;
    }
    // 순위가 가장 낮은 이웃 쌍을 합치기를 더 없을 때까지
    while (sym.size() > 1) {
      int best = std::numeric_limits<int>::max();
      size_t bi = 0;
      for (size_t i = 0; i + 1 < sym.size(); ++i) {
        const auto it = merges_.find(sym[i] + '\x01' + sym[i + 1]);
        if (it != merges_.end() && it->second < best) {best = it->second; bi = i;}
      }
      if (best == std::numeric_limits<int>::max()) {break;}
      sym[bi] += sym[bi + 1];
      sym.erase(sym.begin() + static_cast<long>(bi) + 1);
    }
    std::vector<int64_t> ids;
    for (const auto & t : sym) {
      const auto it = vocab_.find(t);
      if (it != vocab_.end()) {ids.push_back(it->second); continue;}
      for (unsigned char b : t) {   // 바이트 토큰
        char buf[8];
        std::snprintf(buf, sizeof buf, "<0x%02X>", b);
        ids.push_back(at(buf));
      }
    }
    if (ids.size() + 1 > max_len) {ids.resize(max_len - 1);}
    ids.push_back(eos_);
    ids.resize(max_len, pad_);
    return ids;
  }

private:
  int at(const std::string & t) const
  {
    const auto it = vocab_.find(t);
    if (it == vocab_.end()) {throw std::runtime_error("SigLIP2 어휘에 없음: " + t);}
    return it->second;
  }
  static std::string unesc(const std::string & s)
  {
    std::string o;
    for (size_t i = 0; i < s.size(); ++i) {
      if (s[i] == '\\' && i + 1 < s.size()) {
        const char n = s[++i];
        o += n == 'n' ? '\n' : n == 't' ? '\t' : n == 'r' ? '\r' : n;
      } else {
        o += s[i];
      }
    }
    return o;
  }
  std::unordered_map<std::string, int> vocab_;
  std::unordered_map<std::string, int> merges_;
  int eos_ = 1, pad_ = 0;
};

}  // namespace limo_fire_perception
