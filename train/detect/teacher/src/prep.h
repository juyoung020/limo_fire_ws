// 입력 전처리: 사진 읽기(stb_image) → 1008×1008 양선형(Pillow 와 같은 8 비트 고정소수점) → (x/255 − 0.5)/0.5, CHW float.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

// PIL Image.resize(BILINEAR) 와 같은 8 비트 RGB 리사이즈(Pillow Resample.c: 가로 다음 세로, 22 비트 고정소수점).
// ObjectSAM teacher/prep.cpp 에서 가져옴.
void pil_resize_rgb(const uint8_t* src, int sw, int sh, int sstride, uint8_t* dst, int dw, int dh);

// 사진 파일을 RGB 8 비트로 읽는다(jpg·png·gif·bmp). 실패하면 false
bool load_rgb(const std::string& path, std::vector<uint8_t>& rgb, int& w, int& h);

// rgb [h, w, 3] → out [3, S, S] float (S = 1008)
void prepare_sam3(const uint8_t* rgb, int w, int h, float* out, int S = 1008);
