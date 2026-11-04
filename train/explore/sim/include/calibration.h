// 보정 녹화 목록(analysis/calibration.yaml) 읽기 — motion_check · replay_check 가 같이 쓴다 (호스트)
#pragma once
#include <yaml-cpp/yaml.h>

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sim {

struct CalRec {
  std::string name, bag, role, floor;
  std::vector<std::pair<double, double>> skip;
  double replay_from = -1;
  std::string csv, lfrec, map, scene;   // data 폴더 안 파일
};

// role 이 빈 문자열이면 전부. section: recordings (구동·다시 돌리기) 또는 body (라이다 몸 참값 — motion_check --body)
inline std::vector<CalRec> load_calibration(const std::string& path, const std::string& role, const std::string& section = "recordings") {
  const YAML::Node y = YAML::LoadFile(path);
  const std::string dir = path.substr(0, path.find_last_of('/') + 1);
  const std::string data = dir + y["data"].as<std::string>() + "/";
  std::vector<CalRec> out;
  if (!y[section]) throw std::runtime_error(path + ": " + section + " 목록 없음");
  for (const auto& r : y[section]) {
    CalRec c;
    c.name = r["name"].as<std::string>();
    c.bag = r["bag"].as<std::string>();
    c.role = r["role"].as<std::string>();
    if (c.role != "fit" && c.role != "validate") throw std::runtime_error(path + ": " + c.name + " role 은 fit · validate");
    c.floor = r["floor"].as<std::string>();
    if (r["skip"])
      for (const auto& s : r["skip"]) c.skip.emplace_back(s[0].as<double>(), s[1].as<double>());
    if (r["replay_from"]) c.replay_from = r["replay_from"].as<double>();
    c.csv = data + c.name + ".csv";
    c.lfrec = data + c.name + ".lfrec";
    c.map = data + c.name + ".lfrec.map.yaml";
    c.scene = data + c.name + ".scene";
    if (role.empty() || c.role == role) out.push_back(c);
  }
  if (out.empty()) throw std::runtime_error(path + ": role " + role + " 녹화가 없음");
  return out;
}

// motion_check · replay_check 의 인자 형식 "파일@시작:끝,…"
inline std::string with_skips(const std::string& file, const std::vector<std::pair<double, double>>& skip) {
  std::string s = file;
  for (size_t i = 0; i < skip.size(); ++i)
    s += (i ? "," : "@") + std::to_string(skip[i].first) + ":" + std::to_string(skip[i].second);
  return s;
}

}  // namespace sim
