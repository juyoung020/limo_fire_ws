#!/bin/bash
# 로봇 양자화 지연 표 (~/decider_test 안에서): 압축 요청 + 그림 128 토큰. 설정마다 서버를 띄우고 PID 로 끔. 결과 lat_table_q.tsv
# cold = 다른 요청(lat_long) 뒤 lat_comp (앞부분 캐시 거의 없음) · warm = lat_comp 뒤 값 하나 바뀐 lat_comp2 (패치된 앞부분 캐시)
# 쓰는 법: cd ~/decider_test && bash lat_table_q.sh [N=5] [설정 이름…]
cd ~/decider_test || exit 1
N=${1:-5}; shift
BIN=./llama.cpp/build/bin/llama-server
MM=models/mmproj-d1-3B-Q8_0.gguf
OUT=lat_table_q.tsv
ALL=("Q8|models/d1-3B-Q8_0.gguf" "Q6_K|models/d1-3B-Q6_K-rq.gguf" "Q5_K_M|models/d1-3B-Q5_K_M-rq.gguf" "Q4_K_M|models/d1-3B-Q4_K_M-rq.gguf")
[ -f $OUT ] || echo -e "cfg\treq\trun\twall_ms\tprompt_tokens\tprompt_ms" > $OUT
post() { curl -s -o /dev/null -w '%{time_total}' -X POST -H 'Content-Type: application/json' --data-binary @$1 http://127.0.0.1:8171/v1/systemone; }
for c in "${ALL[@]}"; do
  IFS='|' read -r name model <<< "$c"
  [ $# -gt 0 ] && ! printf '%s\n' "$@" | grep -qx "$name" && continue
  [ -f "$model" ] || { echo "$name: 모델 없음 — 건너뜀"; continue; }
  log=server_latq_$name.log
  LLAMA_DECISION_IMAGES_LAST=1 $BIN -m $model --mmproj $MM --host 127.0.0.1 --port 8171 -ngl 99 -c 8192 -np 1 -lm dio --image-max-tokens 128 > $log 2>&1 &
  P=$!
  ok=0
  for i in $(seq 180); do sleep 1; grep -q "listening" $log && { ok=1; break; }; kill -0 $P 2>/dev/null || break; done
  if [ $ok = 0 ]; then echo "$name: 안 뜸 — $(grep -a -m1 -i 'error\|fail' $log)"; ps -o args= -p $P 2>/dev/null | grep -q llama-server && kill $P; continue; fi
  post lat_comp.json > /dev/null   # 데우기
  for k in $(seq $N); do
    post lat_long.json > /dev/null
    t=$(post lat_comp.json)
    pm=$(grep -a "prompt eval time" $log | tail -1 | grep -o "= *[0-9.]* ms" | head -1 | grep -o "[0-9.]*")
    tok=$(grep -a -o "n_tokens = [0-9]*, truncated" $log | tail -1 | grep -o "[0-9]*" | head -1)
    echo -e "$name\tcold\t$k\t$(awk "BEGIN{printf \"%.0f\", $t*1000}")\t$tok\t$pm" >> $OUT
    t=$(post lat_comp2.json)
    pm=$(grep -a "prompt eval time" $log | tail -1 | grep -o "= *[0-9.]* ms" | head -1 | grep -o "[0-9.]*")
    echo -e "$name\twarm\t$k\t$(awk "BEGIN{printf \"%.0f\", $t*1000}")\t$tok\t$pm" >> $OUT
  done
  ps -o args= -p $P | grep -q "llama-server" && kill $P
  wait $P 2>/dev/null
  sleep 3
done
python3 -c "
import csv, statistics as S, collections
d = collections.defaultdict(list); pm = collections.defaultdict(list)
for r in csv.DictReader(open('$OUT'), delimiter='\t'):
    d[(r['cfg'], r['req'])].append(float(r['wall_ms']))
    if r['prompt_ms']: pm[(r['cfg'], r['req'])].append(float(r['prompt_ms']))
for k in sorted(d): print(k[0], k[1], 'n', len(d[k]), 'wall_median_ms', round(S.median(d[k])), 'prompt_eval_median_ms', round(S.median(pm[k])) if pm[k] else '-')
"
