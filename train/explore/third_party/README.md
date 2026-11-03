# third_party — 저장소에 넣지 않는 외부 헤더

## optix-dev (NVIDIA OptiX 9.0 헤더)

시뮬(`sim/`)이 RTX 의 RT 코어로 광선 순회를 할 때 쓴다. 없으면 CMake 가 소프트웨어 순회만 빌드한다(결과는 같은 규칙).

    git clone --depth 1 --branch v9.0.0 https://github.com/NVIDIA/optix-dev train/explore/third_party/optix-dev

- 출처: https://github.com/NVIDIA/optix-dev (공개 헤더 저장소, 태그 v9.0.0 = OptiX ABI 105)
- 라이선스: NVIDIA "Software Developer Kits, Samples and Tools License Agreement"(저장소 `LICENSE.txt`) — 헤더를 고쳐 쓰고 바이너리로 넣어 배포할 수 있으나 NVIDIA 하드웨어에서만 쓰고 헤더 자체 재배포 조건이 붙어 저장소에는 넣지 않는다(.gitignore)
- 런타임 `libnvoptix.so.1` 은 NVIDIA 드라이버에 들어 있다(이 PC 580.178 확인). 드라이버가 ABI 105 를 지원해야 한다
