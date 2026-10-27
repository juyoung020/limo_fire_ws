// 학습 캐시: 사진을 긴 변 S(640)로 줄여(양선형, cv2.INTER_LINEAR 와 같은 화소 중심) JPEG 로 다시 담은 한 파일.
// 한 번 만들어 두고 학습 때는 통째로 고정(pinned) 메모리에 올려 CPU 스레드가 JPEG 만 푼다.
//
// 파일: "YCACHE01" | u32 n | u32 S | Rec[n] | Box[...] | JPEG 덩어리
//       "YCACHE02" 는 끝에 마스크가 붙는다: Mask[n] | u64 길이 | 마스크 덩어리(yolods .msk 의 런 u32 들)
#pragma once
#include <cstdint>
#include <string>
#include <vector>

struct CacheRec {
  uint64_t jpg_off;  // JPEG 덩어리 안 위치
  uint32_t jpg_len;
  uint16_t w, h;     // 줄인 크기
  uint32_t box_off;  // Box 배열 안 위치
  uint16_t nbox;
  uint16_t orig_w, orig_h;  // 원본 크기(검증 때 좌표 되돌리기용)
  uint16_t pad;
};
struct CacheBox {
  float cls;
  float cx, cy, w, h;  // 0..1 (YOLO 라벨 그대로)
};

// 사진 하나의 마스크 번호 지도(사진 전체를 가로세로 따로 늘여 덮는 size×size, 0 배경, j+1 = 상자 j). 런 = 길이 << 8 | 번호
struct CacheMask {
  uint64_t off;  // 마스크 덩어리 안 위치(u32 단위)
  uint32_t nrun;
  uint16_t size, n;
};

class Cache {
 public:
  bool open(const std::string& path);  // 통째로 읽기(고정 메모리)
  ~Cache();
  int n() const { return (int)rec_.size(); }
  int S() const { return S_; }
  const CacheRec& rec(int i) const { return rec_[i]; }
  const CacheBox* boxes(int i) const { return box_.data() + rec_[i].box_off; }
  const uint8_t* jpg(int i) const { return blob_ + rec_[i].jpg_off; }
  bool has_masks() const { return !mask_.empty(); }
  const CacheMask& mask(int i) const { return mask_[i]; }
  const uint32_t* runs(int i) const { return runs_.data() + mask_[i].off; }
  std::vector<std::string> names;  // 같은 차례의 원래 이름(names.txt)

 private:
  int S_ = 0;
  std::vector<CacheRec> rec_;
  std::vector<CacheBox> box_;
  std::vector<CacheMask> mask_;
  std::vector<uint32_t> runs_;
  uint8_t* blob_ = nullptr;
  size_t blob_len_ = 0;
};

// 번호 지도 풀기: dst[size*size]
void decode_mask(const uint32_t* runs, uint32_t nrun, uint8_t* dst, int size);

// cv2.resize(INTER_LINEAR) 와 같은 양선형(화소 중심 맞춤, 앤티에일리어싱 없음), RGB u8
void resize_linear(const uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh);
