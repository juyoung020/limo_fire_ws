// 아래층 체크포인트 머리말만 읽어 신경망 구조 옵션을 알아낸다 (low_policy_run·combine_run — 평가 도구가 학습 때 구조를 따로 받지 않게).
// 형식은 ppo.cu save/load 와 같다 (형식 3): 매직 · 판 · 실수 바이트 · 블록 수 · (이름 길이 · 이름 · 원소 수) × 블록 수.
#pragma once
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

struct LowCkptMeta {
  bool wide = false;   // 넓은 지도 가지 (enc.cnn3.*)
  int gru_h = 0;       // 순환 핵 은닉 크기 (actor.gru.wh = 3H × H 원소), 없으면 0
};
inline LowCkptMeta read_low_ckpt_meta(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("체크포인트 없음: " + path);
  auto r = [&](void* d, size_t n) {
    f.read((char*)d, (std::streamsize)n);
    if (!f) throw std::runtime_error("체크포인트 머리말이 잘림: " + path);
  };
  uint32_t hdr[3];
  r(hdr, 12);
  int32_t nb;
  r(&nb, 4);
  LowCkptMeta m;
  for (int i = 0; i < nb; ++i) {
    int32_t ln;
    r(&ln, 4);
    std::string name(ln, '\0');
    r(name.data(), ln);
    int64_t n;
    r(&n, 8);
    if (name.rfind("enc.cnn3.", 0) == 0) m.wide = true;
    if (name == "actor.gru.wh") m.gru_h = (int)std::lround(std::sqrt((double)n / 3.0));
  }
  return m;
}
