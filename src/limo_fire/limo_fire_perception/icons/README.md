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

## COCO 80 종 (2026-11-17)

`config/yolo_coco.yaml` 의 80 종도 같은 방법·같은 글리프 묶음으로 만들었다. 바탕 색 = `class_colors`(색상환을 황금각 137.5° 씩 돌려 이웃 클래스끼리 다르게, person 은 위 화재 배지와 같은 파랑) — 지도 발자국·이름표 테두리와 같은 색.
person 은 위 화재 배지를 그대로 쓴다. 비슷한 글리프가 없는 것은 묶어 썼다(동물 = pets, 과일 = nutrition, 수저 = restaurant).

| 클래스 | 글리프 |
|---|---|
| bicycle | pedal_bike |
| car | directions_car |
| motorcycle | two_wheeler |
| airplane | flight |
| bus | directions_bus |
| train | train |
| truck | local_shipping |
| boat | sailing |
| traffic_light | traffic |
| fire_hydrant | fire_hydrant |
| stop_sign | do_not_disturb_on |
| parking_meter | local_parking |
| bench | chair_alt |
| bird | raven |
| cat | pets |
| dog | sound_detection_dog_barking |
| horse | bedroom_baby |
| sheep | pets |
| cow | pets |
| elephant | pets |
| bear | pets |
| zebra | pets |
| giraffe | pets |
| backpack | backpack |
| umbrella | beach_access |
| handbag | shopping_bag |
| tie | checkroom |
| suitcase | luggage |
| frisbee | album |
| skis | downhill_skiing |
| snowboard | snowboarding |
| sports_ball | sports_soccer |
| kite | paragliding |
| baseball_bat | sports_baseball |
| baseball_glove | sports_baseball |
| skateboard | skateboarding |
| surfboard | surfing |
| tennis_racket | sports_tennis |
| bottle | water_bottle |
| wine_glass | wine_bar |
| cup | local_cafe |
| fork | restaurant |
| knife | restaurant |
| spoon | restaurant |
| bowl | ramen_dining |
| banana | nutrition |
| apple | nutrition |
| sandwich | lunch_dining |
| orange | nutrition |
| broccoli | eco |
| carrot | nutrition |
| hot_dog | lunch_dining |
| pizza | local_pizza |
| donut | donut_large |
| cake | cake |
| chair | chair |
| couch | weekend |
| potted_plant | potted_plant |
| bed | bed |
| dining_table | table_restaurant |
| toilet | wc |
| tv | tv |
| laptop | laptop_windows |
| mouse | mouse |
| remote | settings_remote |
| keyboard | keyboard |
| cell_phone | smartphone |
| microwave | microwave |
| oven | oven |
| toaster | breakfast_dining |
| sink | wash |
| refrigerator | kitchen |
| book | menu_book |
| clock | schedule |
| vase | local_florist |
| scissors | content_cut |
| teddy_bear | toys |
| hair_drier | air |
| toothbrush | clean_hands |
