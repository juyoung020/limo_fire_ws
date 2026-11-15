# 위층 판단 모델(d1-3B, lfm2-d1) LoRA 학습 도구 — docs: README.md
# llama.cpp 는 저장소 밖 빌드를 씀 (LLAMA_DIR: 소스 + build/ — 서버를 돌리는 것과 같은 판). 받는 법은 scripts/README "준비: 모델과 서버"
cmake_minimum_required(VERSION 3.20)
project(upper_lora LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(LLAMA_DIR "$ENV{HOME}/models/decision/runtime/llama.cpp" CACHE PATH "llama.cpp 소스 (build/ 아래 빌드가 있어야 함)")
set(LB ${LLAMA_DIR}/build)
find_package(CUDAToolkit REQUIRED)

# d1_cache: 요청 본문(JSON, /v1/systemone 과 같은 꼴)마다 서버와 같은 프롬프트로 모델을 돌려 28 층 입력의 마지막 5 토큰 은닉과 선택지 글자 토큰·로짓을 저장
add_executable(d1_cache d1_cache.cpp)
target_include_directories(d1_cache PRIVATE ${LLAMA_DIR}/include ${LLAMA_DIR}/common ${LLAMA_DIR}/ggml/include ${LLAMA_DIR}/tools/server ${LLAMA_DIR}/tools/mtmd
                           ${LLAMA_DIR}/vendor ${LLAMA_DIR}/vendor/cpp-httplib ${LLAMA_DIR} ${LB}/tools/server)
target_link_libraries(d1_cache PRIVATE ${LB}/tools/server/libserver-context.a ${LB}/bin/libllama-common.so ${LB}/common/libllama-common-base.a ${LB}/bin/libmtmd.so
                      ${LB}/bin/libllama.so ${LB}/bin/libggml.so ${LB}/bin/libggml-base.so ${LB}/vendor/cpp-httplib/libcpp-httplib.a ${LB}/vendor/hash/libvendor-hash.a pthread)
set_target_properties(d1_cache PROPERTIES BUILD_RPATH ${LB}/bin)

# head_check: CPU 참조 층 계산이 llama.cpp 로짓과 같은지
add_executable(head_check head_check.cpp)
target_include_directories(head_check PRIVATE ${LLAMA_DIR}/ggml/include)
target_link_libraries(head_check PRIVATE ${LB}/bin/libggml-base.so ${LB}/bin/libggml.so)
set_target_properties(head_check PROPERTIES BUILD_RPATH ${LB}/bin)

# lora_train: LoRA 학습(28·29 층) + 기울기 검사 (--gradcheck)
add_executable(lora_train lora_train.cu)
target_include_directories(lora_train PRIVATE ${LLAMA_DIR}/ggml/include)
target_link_libraries(lora_train PRIVATE CUDA::cublas CUDA::cudart ${LB}/bin/libggml-base.so ${LB}/bin/libggml.so)
set_target_properties(lora_train PROPERTIES BUILD_RPATH ${LB}/bin CUDA_ARCHITECTURES 86)

# lora_export: 어댑터 → llama.cpp LoRA GGUF (llama-server --lora)
add_executable(lora_export lora_export.cpp)
target_include_directories(lora_export PRIVATE ${LLAMA_DIR}/ggml/include)
target_link_libraries(lora_export PRIVATE ${LB}/bin/libggml-base.so ${LB}/bin/libggml.so)
set_target_properties(lora_export PROPERTIES BUILD_RPATH ${LB}/bin)
