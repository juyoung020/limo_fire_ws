# CUDA 엄격 진단 (2026-11-16): 장치 코드에서 호스트 함수를 부르는 진단을 경고가 아니라 오류로.
#   20013 · 20015: constexpr __host__ 함수(람다 operator() 포함)를 __host__ __device__ · __device__ 함수에서 부름 — nvcc 는 경고만 내고 장치 쪽 값이
#                  틀리게 나온 적이 있다 (2026-11-16 mdp/graph.h — 몇 시간 잃음). --expt-relaxed-constexpr 를 켠 프로젝트에선 애초에 안 남
#   cross-execution-space-call: __host__ __device__ 에서 __host__ 함수 부름도 오류로 (nvcc --Werror)
# project(... CUDA) 뒤에 include. 이미 붙어 있으면(상위 디렉터리에서 물려받음) 또 안 붙인다. LIMO_CUDA_STRICT 는 직접 nvcc 를 부르는 명령(PTX 등)에 쓰는 목록
set(LIMO_CUDA_STRICT -Xcudafe --diag_error=20013 -Xcudafe --diag_error=20015 --Werror cross-execution-space-call)
if(NOT CMAKE_CUDA_FLAGS MATCHES "diag_error=20013")
  string(APPEND CMAKE_CUDA_FLAGS " -Xcudafe --diag_error=20013 -Xcudafe --diag_error=20015 --Werror cross-execution-space-call")
endif()
