# webcrawl

라이선스가 붙은 공개 이미지를 클래스별로 모은다. 검색 엔진(구글·네이버)을 긁지 않는다 — 약관 위반·저작권 문제.

| 출처 | 라이선스 처리 |
|---|---|
| Wikimedia Commons (분류 + 파일 검색) | 자유 라이선스만 있는 곳이라 모두 쓴다 |
| Openverse (CC 이미지 검색, 키 없음) | CC0·PDM·BY·BY-SA·BY-NC·BY-NC-SA 만. 변경 금지(ND) 제외. 키 없이는 검색어당 최대 240 개 |

```bash
cd train/detect/tools/webcrawl && cargo build --release
./target/release/webcrawl queries.json ../../data/raw/web            # 전부
./target/release/webcrawl queries.json ../../data/raw/web candle     # 한 클래스만
```

- 클래스·검색어·최대 장수는 `queries.json`. 다시 돌리면 이미 받은 것은 건너뛴다.
- 결과: `data/raw/web/<클래스>/*.jpg|png` + `manifest.jsonl`(파일마다 출처 페이지·라이선스·작성자). 발표·공개 때 이 목록으로 출처를 밝힌다.
- 라벨은 없다. 모은 뒤 상자 라벨을 따로 단다.
- 예의: User-Agent 에 저장소 주소, Commons API 1 초 간격, Openverse 3 초 간격, 429 면 기다렸다 다시.
