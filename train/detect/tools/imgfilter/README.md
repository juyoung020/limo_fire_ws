# imgfilter — 크롤링 이미지 거르기 (Rust, candle + SigLIP)

AutoCrawler·icrawler·webcrawl 로 모은 사진에서 못 쓸 것을 버리고 남은 것만 `data/clean/<클래스>/` 에 하드링크로 모은다. 원본(`data/raw/`)은 지우지 않는다.

```bash
# 모델(3.5 GB, Apache-2.0): google/siglip-so400m-patch14-384 의 config.json·tokenizer.json·model.safetensors
#   → data/weights/siglip-so400m-patch14-384/
# 드라이버가 CUDA 13.0 까지라 12.8 툴킷으로 빌드 (13.2 로 빌드하면 CUDA_ERROR_UNSUPPORTED_PTX_VERSION)
PATH=/usr/local/cuda-12.8/bin:$PATH CUDA_HOME=/usr/local/cuda-12.8 CUDA_COMPUTE_CAP=120 \
  cargo build --release --features cuda
cd ../..   # train/detect
tools/imgfilter/target/release/imgfilter tools/imgfilter/prompts.json \
  data/weights/siglip-so400m-patch14-384 data/raw data/clean [클래스 …]
```

거르는 순서
1. **규칙**: 못 여는 파일, 짧은 변 < 200 px, 가로세로 비 > 3.
2. **내용**: SigLIP 제로샷. `prompts.json` 의 keep(받을 사진)·reject(버릴 사진) 설명 전부에 softmax, keep 확률 합 < 0.5 면 버림. 그림·아이콘·포스터·도면·관계없는 사진(사람·음식 등은 대개 "3D rendering" 칸으로 간다)이 빠진다.
3. **중복**: 이미지 임베딩 코사인 ≥ 0.97 이면 해상도 큰 것 하나만. 0.95 는 비슷한 다른 제품까지 지워서 올렸다.

판정은 `data/clean/<클래스>.tsv`(경로·판정·이유·keep 확률·가장 높은 설명). 문턱값·설명은 `prompts.json` 에서 바꾼다. RTX 5070 Ti 에서 5,000 장에 2 분 남짓.
