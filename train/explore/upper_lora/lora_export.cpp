// lora_export — lora_train 어댑터(adapter.bin)를 llama.cpp LoRA GGUF 로 (llama-server --lora). llama.cpp src/llama-adapter.cpp 가 읽는 꼴:
// general.type "adapter" · general.architecture "lfm2" · adapter.type "lora" · adapter.lora.alpha, 텐서 "<기본 텐서>.lora_a" ne [in, r] · ".lora_b" ne [r, out] (f32)
#include <cstdio>
#include <string>
#include <vector>

#include "ggml.h"
#include "gguf.h"
#include "d1_ref.h"

int main(int argc, char** argv) {
  if (argc < 3) { std::fprintf(stderr, "lora_export adapter.bin out.gguf\n"); return 2; }
  std::FILE* f = std::fopen(argv[1], "rb");
  int r; float alpha;
  if (!f || std::fread(&r, 4, 1, f) != 1 || std::fread(&alpha, 4, 1, f) != 1) { std::fprintf(stderr, "어댑터를 못 읽음\n"); return 1; }
  const char* base[d1::kMats] = {"shortconv.in_proj", "shortconv.out_proj", "ffn_gate", "ffn_up", "ffn_down"};
  ggml_init_params ip{(size_t)1 << 30, nullptr, false};
  ggml_context* ctx = ggml_init(ip);
  gguf_context* g = gguf_init_empty();
  gguf_set_val_str(g, "general.type", "adapter");
  gguf_set_val_str(g, "general.architecture", "lfm2");
  gguf_set_val_str(g, "adapter.type", "lora");
  gguf_set_val_f32(g, "adapter.lora.alpha", alpha);
  int n = 0;
  for (int l = 0; l < 2; ++l)
    for (int m = 0; m < d1::kMats; ++m) {
      int on = 0;
      if (std::fread(&on, 4, 1, f) != 1) { std::fprintf(stderr, "어댑터가 짧음\n"); return 1; }
      if (!on) continue;
      const int in = d1::mat_in(m), out = d1::mat_out(m);
      ggml_tensor* A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in, r);
      ggml_tensor* B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, r, out);
      if (std::fread(A->data, 4, (size_t)in * r, f) != (size_t)in * r || std::fread(B->data, 4, (size_t)out * r, f) != (size_t)out * r) { std::fprintf(stderr, "어댑터가 짧음\n"); return 1; }
      const std::string nm = "blk." + std::to_string(28 + l) + "." + base[m] + ".weight";
      ggml_set_name(A, (nm + ".lora_a").c_str());
      ggml_set_name(B, (nm + ".lora_b").c_str());
      gguf_add_tensor(g, A);
      gguf_add_tensor(g, B);
      ++n;
    }
  std::fclose(f);
  if (!gguf_write_to_file(g, argv[2], false)) { std::fprintf(stderr, "못 씀\n"); return 1; }
  std::printf("%s: LoRA %d 쌍 · r %d · α %.1f\n", argv[2], n, r, alpha);
  gguf_free(g);
  ggml_free(ctx);
  return 0;
}
