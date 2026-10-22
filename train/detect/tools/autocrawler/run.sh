#!/usr/bin/env bash
# AutoCrawler(Naver, 원본 해상도)로 클래스별 키워드 이미지를 data/raw/autocrawler/<클래스>/ 에 받는다.
# 쓰기: ./run.sh [클래스 ...]   (없으면 keywords/*.txt 전부)   LIMIT=250 THREADS=4 로 조절
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
ac="$here/../external/AutoCrawler"
out="$here/../../data/raw/autocrawler"
limit="${LIMIT:-250}"
threads="${THREADS:-4}"

if [ ! -x "$here/.venv/bin/python" ]; then
  uv venv -q --python 3.12 "$here/.venv"
  uv pip install -q --python "$here/.venv/bin/python" "requests>=2.31" "selenium>=4.20" "webdriver-manager>=4.0"
fi

classes=("$@")
[ ${#classes[@]} -eq 0 ] && classes=($(cd "$here/keywords" && ls *.txt | sed 's/\.txt$//'))

mkdir -p "$out"
cd "$here"   # chrome-profile/ 이 여기 생긴다(git 제외)
for c in "${classes[@]}"; do
  # AutoCrawler 가 키워드 파일을 정렬해 덮어쓰므로 복사본을 넘긴다
  kw="$(mktemp)"; cp "keywords/$c.txt" "$kw"
  echo "== $c (키워드 $(wc -l < "$kw") 개, 키워드당 최대 $limit 장)"
  # 로그에 이미지 원본 URL 이 남는다 — 출처 기록으로 보관
  PYTHONPATH="$ac/src" "$here/.venv/bin/python" -m autocrawler \
    --keywords-file "$kw" --download-path "$out/$c" \
    --no-google --naver --full --limit "$limit" --threads "$threads" \
    2>&1 | tee -a "$out/$c.log" | grep -E "^(==|Done|Collect links done|Skipping)" || true
  rm -f "$kw"
  echo "== $c: $(find "$out/$c" -type f ! -name '*_done' | wc -l) 장"
done
