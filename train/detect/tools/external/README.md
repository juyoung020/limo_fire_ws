# external — 남의 이미지 크롤링 도구 (Python, 고치지 않음)

| 폴더 | 출처 | 커밋 | 라이선스 | 지원 |
|---|---|---|---|---|
| `AutoCrawler` | [YoongiKim/AutoCrawler](https://github.com/YoongiKim/AutoCrawler) (별 1,696, 2026-08 갱신) | `9d61d18` | Apache-2.0 | **Google, Naver** (Selenium) — 한국어 검색어에 맞음 |
| `icrawler` | [hellock/icrawler](https://github.com/hellock/icrawler) | `f2a326d` | MIT | Google, Bing, Baidu, Flickr(API 키) |
| `bing_image_downloader` | [gurugaurav/bing_image_downloader](https://github.com/gurugaurav/bing_image_downloader) | `d49b98a` | MIT | Bing |
| `google-images-download` | [hardikvasa/google-images-download](https://github.com/hardikvasa/google-images-download) | `9057dad` | MIT (setup.py) | Google |

주의
- **Google·Bing 이미지 검색을 프로그램으로 긁는 것은 두 회사 약관이 금지한다.** 쓰면 IP 가 막힐 수 있고, 책임은 돌리는 사람에게 있다.
- 검색 결과 사진은 대부분 저작권이 있다. "CC 라이선스" 필터도 원본 사이트 표시를 그대로 믿는 것이라 확인이 필요하다. 받은 사진은 출처 URL 을 남기고, 공개 저장소·발표 자료에 다시 싣지 않는다.
- 라이선스가 확실한 수집은 `../webcrawl`(Wikimedia Commons·Openverse) 와 icrawler 의 Flickr 크롤러(무료 API 키, 라이선스 필터)를 쓴다.
