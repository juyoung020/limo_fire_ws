#!/bin/bash
# 로봇 판단 모델 지연 표 (~/decider_test 안에서): 모델·서버 설정마다 서버를 띄우고, 요청 꼴마다 N 번 재고, 서버를 PID 로 끔. 결과 lat_table.tsv
# 쓰는 법: cd ~/decider_test && bash lat_table.sh [N=5]
cd ~/decider_test || exit 1
N=${1:-5}
BIN=./llama.cpp/build/bin/llama-server
MM=models/mmproj-d1-3B-Q8_0.gguf
OUT=lat_table.tsv
REQS="lat_long lat_comp"            # 지금 꼴 긴 글 · 압축 꼴(짧은 키 + 결정 기록 4 + C1 한 장)
# 설정: 이름|모델|서버 추가 인자
CFGS=(
  "Q8|models/d1-3B-Q8_0.gguf|"
  "Q8_fa|models/d1-3B-Q8_0.gguf|-fa on"
  "Q8_img128|models/d1-3B-Q8_0.gguf|--image-max-tokens 128"
  "Q4KM|models/d1-3B-Q4_K_M-rq.gguf|"
  "Q4KM_fa|models/d1-3B-Q4_K_M-rq.gguf|-fa on"
  "Q4KM_fa_img128|models/d1-3B-Q4_K_M-rq.gguf|-fa on --image-max-tokens 128"
  "Q4KM_fa_ub1024|models/d1-3B-Q4_K_M-rq.gguf|-fa on -ub 1024 -b 2048"
)
echo -e "cfg\treq\trun\twall_ms\tprompt_tokens\tcached" > $OUT
for c in "${CFGS[@]}"; do
  IFS='|' read -r name model extra <<< "$c"
  [ -f "$model" ] || { echo "$name: 모델 없음 ($model) — 건너뜀"; continue; }
  log=server_lat_$name.log
  $BIN -m $model --mmproj $MM --host 127.0.0.1 --port 8171 -ngl 99 -c 8192 -np 1 -lm dio $extra > $log 2>&1 &
  P=$!
  ok=0
  for i in $(seq 120); do sleep 1; grep -q "listening" $log && { ok=1; break; }; kill -0 $P 2>/dev/null || break; done
  if [ $ok = 0 ]; then echo "$name: 안 뜸 — $(grep -a -m1 -i 'error\|fail' $log)"; ps -o args= -p $P 2>/dev/null | grep -q llama-server && kill $P; continue; fi
  curl -s -o /dev/null -X POST -H 'Content-Type: application/json' --data-binary @lat_comp.json http://127.0.0.1:8171/v1/systemone   # 데우기
  for r in $REQS lat_comp2; do
    for k in $(seq $N); do
      f=$r.json; [ $r = lat_comp2 ] && f=lat_comp2.json
      if [ $r = lat_comp2 ]; then   # 캐시 확인: 압축 꼴 다음에 값 하나만 바뀐 요청
        curl -s -o /dev/null -X POST -H 'Content-Type: application/json' --data-binary @lat_comp.json http://127.0.0.1:8171/v1/systemone
      fi
      t=$(curl -s -o /dev/null -w '%{time_total}' -X POST -H 'Content-Type: application/json' --data-binary @$f http://127.0.0.1:8171/v1/systemone)
      tok=$(grep -a -o "n_tokens = [0-9]*, truncated" $log | tail -1 | grep -o "[0-9]*" | head -1)
      cached=$(grep -a -o "f_keep = [0-9.]*" $log | tail -1 | grep -o "[0-9.]*$")
      echo -e "$name\t$r\t$k\t$(awk "BEGIN{printf \"%.0f\", $t*1000}")\t$tok\t$cached" >> $OUT
    done
  done
  ps -o args= -p $P | grep -q "llama-server" && kill $P
  wait $P 2>/dev/null
  sleep 3
done
echo "== 중앙값 (ms)"
python3 -c "
import csv, statistics as S, collections
d = collections.defaultdict(list); tk = {}
for r in csv.DictReader(open('$OUT'), delimiter='\\t'):
    d[(r['cfg'], r['req'])].append(float(r['wall_ms'])); tk[(r['cfg'], r['req'])] = r['prompt_tokens']
for k in sorted(d): print(k[0], k[1], 'tokens', tk[k], 'median_ms', round(S.median(d[k])))
"
