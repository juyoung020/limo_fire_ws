// Embedder 판 고르기 (파라미터 backend). 새 판: 구현 파일 + 여기 이름 하나 + CMake 의 object_search 소스 목록
#include <stdexcept>

#include "limo_fire_perception/embedder.hpp"
#include "limo_fire_perception/trt_siglip.hpp"

namespace limo_fire_perception
{

std::unique_ptr<Embedder> make_embedder(const std::string & backend, const std::string & model_dir, const EmbedderParam & param)
{
  // 파일 이름만 쓰면 model_dir 에서
  auto path = [&](const std::string & v) {return v.empty() || v[0] == '/' ? v : model_dir + "/" + v;};
  if (backend == "siglip2_trt") {
    // 파라미터 이름은 "<판>.<칸>" (설정 파일에서 siglip2_trt: 아래에 묶임)
    const std::string p = backend + ".";
    TrtSiglip::Files f;
    f.vision_engine = path(param(p + "vision_engine", "siglip2_vision.engine"));
    f.text_engine = path(param(p + "text_engine", "siglip2_text_body.engine"));
    f.token_table = path(param(p + "token_table", "siglip2_tok_emb.f16"));
    f.vocab = path(param(p + "vocab_file", "siglip2_vocab.txt"));
    f.merges = path(param(p + "merges_file", "siglip2_merges.txt"));
    return std::make_unique<TrtSiglip>(f, std::stoi(param(p + "max_batch", "4")));
  }
  throw std::runtime_error("모르는 Embedder 판: '" + backend + "' (아는 것: siglip2_trt)");
}

}  // namespace limo_fire_perception
