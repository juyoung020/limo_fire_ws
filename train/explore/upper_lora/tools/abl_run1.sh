#!/bin/bash
# abl_run1.sh <이름> <장면.scene> <씨앗> [low_explore_check 인자…] — 위층 비교 판 하나. 지표는 $OUT/<이름>_<장면>_s<씨앗>.lat (--lat-log), 분석은 tools/abl_metrics.py
# BUILD = low_explore_check 빌드 폴더, OUT = 출력 폴더 (둘 다 필수). 이름에 밑줄(_)을 쓰지 말 것 (abl_metrics.py 가 첫 밑줄로 나눔).
# 주의: low_explore_check 는 실행 때 env.yaml 을 소스 트리(SIM_CONFIG_DIR)에서 읽음 — 배치 도는 동안 그 소스 트리를 다른 커밋으로 옮기지 말 것.
: "${BUILD:?BUILD 필요}" "${OUT:?OUT 필요}"
n=$1; sc=$2; sd=$3; shift 3
b=$(basename $sc .scene)
out=$OUT/${n}_${b}_s${sd}
[ -f $out.done ] && exit 0
rm -f $out.lat
cd $BUILD && ./low_explore_check $sc --seed $sd --people 0 --steps 20000 --max-ep 1 --lat-log $out.lat "$@" > $out.log 2>&1
echo "exit $?" >> $out.log
[ -s $out.lat ] && touch $out.done   # 비트 검사 불일치(종료 코드 ≠ 0)여도 지표는 씀
