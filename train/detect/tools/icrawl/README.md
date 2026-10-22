# icrawl — icrawler(Bing·Baidu)로 탐지 학습 이미지 더 모으기

`../autocrawler` 와 같은 검색어(`../autocrawler/keywords/*.txt`)로 Bing·Baidu 이미지 검색 결과를 받는다.

```bash
./run.sh                         # 전 클래스
BING=300 BAIDU=150 ./run.sh candle   # 검색어당 최대 장수, 0 이면 그 엔진 안 씀
```

- 결과: `../../data/raw/icrawl/<클래스>/<bing|baidu>/<검색어>/000001.jpg …`, 로그(원본 URL) `<클래스>.log`. 끝난 검색어(`.done`)는 건너뛴다.
- icrawler 는 CLI 가 없는 파이썬 라이브러리라 `run.sh` 안에서 한 줄(`python -c`)로 부른다. 우리 코드를 파이썬으로 쓰지 않는다는 규칙의 예외로, 라이브러리 호출만 한다. 패키지는 PyPI `icrawler==0.6.10`(MIT)을 `.venv` 에 깐다 — `../external/icrawler` 는 setuptools-scm 버전 파일이 없어 그대로는 import 가 안 된다.
- **Google 은 안 된다**: icrawler 의 GoogleImageCrawler 는 파싱 오류, `../external/google-images-download`(3.0.1)는 0 장. AutoCrawler 로 해 봐도 CAPTCHA 가 뜬다.
- Baidu 결과는 중국 사진이 섞인다(중국 비상구 표지·소화기도 모양이 비슷해 쓸 만함). 일부는 403 으로 못 받는다.
- 받은 뒤 `../imgfilter` 로 거른다. 저작권 주의는 `../external/README.md` 와 같다.
