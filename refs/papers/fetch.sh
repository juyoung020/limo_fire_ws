#!/usr/bin/env bash
# 선행 연구 PDF 받기 — sources.tsv 의 공개본을 원래 자리에 받는다. PDF 는 git 에 넣지 않는다(.gitignore).
#   refs/papers/fetch.sh            모두 받기 (이미 있고 체크섬이 맞으면 건너뜀)
#   refs/papers/fetch.sh smoke      경로에 "smoke" 가 든 것만
# 방식 browser: 출판사가 봇 확인·캡차를 걸어 curl 로 안 받힌다. 끝에 주소를 알려 주면 브라우저로 받아 그 경로에 둔다.
set -u
cd "$(dirname "$0")"
filter="${1:-}"
ok=0; skip=0; fail=(); manual=()

while IFS=$'\t' read -r path method sha url; do
  [[ -z "$path" || "$path" == \#* ]] && continue
  [[ -n "$filter" && "$path" != *"$filter"* ]] && continue

  if [[ -f "$path" ]] && { [[ "$sha" == "-" ]] || [[ "$(sha256sum < "$path" | cut -d' ' -f1)" == "$sha" ]]; }; then
    skip=$((skip + 1)); continue
  fi
  if [[ "$method" == browser ]]; then
    manual+=("$path  ←  $url"); continue
  fi

  mkdir -p "$(dirname "$path")"
  tmp="$path.part"
  if curl -fsSL --retry 2 --max-time 120 -o "$tmp" "$url" \
     && [[ "$(head -c4 "$tmp")" == "%PDF" ]] \
     && { [[ "$sha" == "-" ]] || [[ "$(sha256sum < "$tmp" | cut -d' ' -f1)" == "$sha" ]]; }; then
    mv "$tmp" "$path"; ok=$((ok + 1)); echo "받음  $path"
  else
    rm -f "$tmp"; fail+=("$path  ←  $url")
  fi
  sleep 1   # 같은 서버에 몰아서 요청하지 않게
done < sources.tsv

echo "받음 $ok · 이미 있음 $skip · 실패 ${#fail[@]} · 브라우저로 ${#manual[@]}"
for f in "${fail[@]}"; do echo "실패(PDF 가 아니거나 체크섬 다름 — 판이 바뀌었을 수 있음): $f"; done
for m in "${manual[@]}"; do echo "브라우저로 받아 두기: $m"; done
[[ ${#fail[@]} -eq 0 ]]
