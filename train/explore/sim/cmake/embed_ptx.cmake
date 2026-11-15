# PTX 파일 → C++ 바이트 배열 (빌드 때 cmake -P 로): 바이너리가 자기 빌드의 PTX 를 늘 품는다 — 빌드 폴더를 다시 지어도 옛 바이너리가 맞지 않는 PTX 를 읽지 않게.
#   cmake -DIN=<optix_rays.ptx> -DOUT=<optix_rays_ptx.cpp> -P embed_ptx.cmake
file(READ ${IN} hex HEX)
string(LENGTH "${hex}" n)
math(EXPR len "${n} / 2")
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," arr "${hex}")
string(REGEX REPLACE "((0x[0-9a-f][0-9a-f],){32})" "\\1\n" arr "${arr}")
file(WRITE ${OUT}.tmp "// 자동 생성 (sim/cmake/embed_ptx.cmake) — src/optix_rays.cu 의 PTX. 고치지 말 것\n#include <cstddef>\nnamespace sim {\nextern const unsigned char kOptixRaysPtx[] = {\n${arr}\n0x00};\nextern const std::size_t kOptixRaysPtxLen = ${len};\n}  // namespace sim\n")
file(RENAME ${OUT}.tmp ${OUT})
