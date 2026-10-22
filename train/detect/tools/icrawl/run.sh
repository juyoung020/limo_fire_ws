#!/usr/bin/env bash
# icrawler(Bing·Baidu)로 클래스별 키워드 이미지를 data/raw/icrawl/<클래스>/<엔진>/<검색어>/ 에 받는다.
# 검색어는 ../autocrawler/keywords/<클래스>.txt 를 같이 쓴다.
# 쓰기: ./run.sh [클래스 ...]   BING=300 BAIDU=150 (검색어당 최대 장수, 0 이면 그 엔진 안 씀)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
kwdir="$here/../autocrawler/keywords"
out="$here/../../data/raw/icrawl"
bing="${BING:-300}"
baidu="${BAIDU:-150}"
py="$here/.venv/bin/python"

# icrawler 는 CLI 가 없는 라이브러리라 PyPI 판(MIT, external/icrawler 와 같은 0.6.x)을 깔아 한 줄로 부른다
if [ ! -x "$py" ]; then
  uv venv -q --python 3.12 "$here/.venv"
  uv pip install -q --python "$py" "icrawler==0.6.10"
fi

classes=("$@")
[ ${#classes[@]} -eq 0 ] && classes=($(cd "$kwdir" && ls *.txt | sed 's/\.txt$//'))

for c in "${classes[@]}"; do
  while IFS= read -r kw; do
    [ -z "$kw" ] && continue
    for pair in "Bing:$bing" "Baidu:$baidu"; do
      eng="${pair%%:*}"; n="${pair##*:}"
      [ "$n" -eq 0 ] && continue
      dir="$out/$c/${eng,,}/$kw"
      [ -f "$dir/.done" ] && continue
      mkdir -p "$dir"
      echo "== $c / $eng / $kw"
      # 로그에 이미지 원본 URL 이 남는다 — 출처 기록으로 보관
      "$py" -I -c 'import sys,logging
from icrawler.builtin import BingImageCrawler, BaiduImageCrawler
C = {"Bing": BingImageCrawler, "Baidu": BaiduImageCrawler}[sys.argv[1]]
C(downloader_threads=4, storage={"root_dir": sys.argv[3]}, log_level=logging.INFO).crawl(
    keyword=sys.argv[2], max_num=int(sys.argv[4]), min_size=(200, 200))' \
        "$eng" "$kw" "$dir" "$n" >> "$out/$c.log" 2>&1 || true
      touch "$dir/.done"
      echo "   $(find "$dir" -type f ! -name .done | wc -l) 장"
    done
  done < "$kwdir/$c.txt"
done
