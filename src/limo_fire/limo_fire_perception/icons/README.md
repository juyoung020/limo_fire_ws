# 지도 아이콘

`object_mapper` 가 RViz 지도 위에 붙이는 클래스별 배지. `<클래스>.obj`(지름 1 m 납작한 원판, 그림 위쪽 = +y) + `.mtl` + `.png`(512×512 텍스처).

| 파일 | 글리프 |
|---|---|
| fire | local_fire_department |
| smoke | foggy |
| fire_extinguisher | fire_extinguisher |
| exit_sign | directions_run |
| person | person |
| door | door_front |
| stairs | stairs |

글리프: Google [Material Symbols](https://github.com/google/material-design-icons) Rounded, Fill (npm `@material-symbols/svg-500@0.27.0`), Apache License 2.0.
흰 글리프를 클래스 색 그라데이션 원(흰 테두리) 위에 얹어 PNG 로 굽고(rsvg-convert + PIL), 64 각 원판 OBJ 에 입혔다. 색을 바꾸려면 PNG 만 다시 만들면 된다.
