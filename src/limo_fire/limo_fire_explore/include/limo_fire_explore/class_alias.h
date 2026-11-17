// 지시 글 → 물체 클래스 이름 ("의자로 가" → chair). explorer 와 저장 지도 object_goto(limo_fire_navigation) 가 같이 쓴다.
// 별칭은 config/explorer.yaml 의 aliases ("한글=영문클래스", 영문은 물체 지도 class_names 이름)
#pragma once

#include <cctype>
#include <string>
#include <utility>
#include <vector>

namespace limo_fire_explore {

using Aliases = std::vector<std::pair<std::string, std::string>>;   // (이름, 클래스)

// "한글=영문" 줄들 → 표 (= 없는 줄은 버림)
inline Aliases parse_aliases(const std::vector<std::string>& lines) {
  Aliases a;
  for (const auto& s : lines) {
    const auto eq = s.find('=');
    if (eq != std::string::npos) a.push_back({s.substr(0, eq), s.substr(eq + 1)});
  }
  return a;
}

// 비교용: 소문자, '_'·'-' 는 띄어쓰기, 앞뒤 빈칸 뺌
inline std::string norm_name(const std::string& s) {
  std::string o;
  for (char c : s) {
    if (c == '_' || c == '-') c = ' ';
    o += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  while (!o.empty() && o.back() == ' ') o.pop_back();
  while (!o.empty() && o.front() == ' ') o.erase(o.begin());
  return o;
}

// 글 전체가 별칭·클래스 이름과 같으면 그 클래스, 아니면 글 안에 든 가장 긴 이름의 클래스 ("가장 먼 의자로 가서 멈춰" → chair). 없으면 빈 글.
// known = 지금 물체 지도에 있는 클래스 이름 (영문 이름을 그대로 써도 찾게)
inline std::string class_of(const std::string& text, const Aliases& aliases, const std::vector<std::string>& known) {
  const std::string t = norm_name(text);
  Aliases names;
  for (const auto& a : aliases) names.emplace_back(norm_name(a.first), a.second);
  for (const auto& k : known) names.emplace_back(norm_name(k), k);
  for (const auto& n : names)
    if (n.first == t) return n.second;
  std::string best;
  size_t best_len = 0;
  for (const auto& n : names)
    if (n.first.size() > best_len && t.find(n.first) != std::string::npos) { best = n.second; best_len = n.first.size(); }
  return best;
}

}  // namespace limo_fire_explore
