#!/usr/bin/env bash
# 가짜 ESP32 → 다리 → ROS 토픽을 끝까지 확인한다 (실물 없이). 쓰는 법: source install/setup.bash 뒤  scripts/fake_bridge_check.sh
# 확인: ⑤ sensors 장면에서 열화상 이미지·최고 온도·레이더 표적 ① sweep 에서 두 토픽이 나오고 거리가 0.2~4 m ② noecho 에서 range 가 inf ③ corrupt 에서 정상 줄만 ④ 다리를 먼저 켜도 이어짐
set -u
PKG=limo_fire_esp32_bridge
BIN=$(ros2 pkg prefix $PKG)/lib/$PKG   # ros2 run 은 껍데기라 죽여도 자식이 남는다 — 실행 파일을 바로 부른다
LINK=${LINK:-/tmp/fake_esp32_check}
fail=0
pids=()
cleanup() { for p in "${pids[@]}"; do kill "$p" 2>/dev/null; done; rm -f "$LINK"; }
trap cleanup EXIT

start_fake() { $BIN/fake_esp32 --link "$LINK" --scenario "$1" --duration 0 >/dev/null 2>&1 & pids+=($!); sleep 0.5; }
stop_fake() { kill "${pids[-1]}" 2>/dev/null; wait "${pids[-1]}" 2>/dev/null; unset 'pids[-1]'; }
ranges() { timeout 8 ros2 topic echo --csv --field range "$1" 2>/dev/null | head -n "${2:-20}"; }
check() { if eval "$2"; then echo "통과: $1"; else echo "실패: $1"; fail=1; fi; }

$BIN/esp32_bridge --ros-args --params-file "$(ros2 pkg prefix $PKG)/share/$PKG/config/esp32_bridge.yaml" -p port:="$LINK" >/tmp/bridge_check.log 2>&1 &
pids+=($!)
sleep 1   # 다리를 먼저 켬: 포트 없음 → 1 초마다 다시 엶

start_fake sweep
sleep 2.5
left=$(ranges /ultrasonic/front_left 20); right=$(ranges /ultrasonic/front_right 20)
check "sweep: front_left 값이 나옴" '[ -n "$left" ]'
check "sweep: front_right 값이 나옴" '[ -n "$right" ]'
check "sweep: front_left 0.2~4 m" 'echo "$left" | awk "{ if (\$1<0.2 || \$1>4.0) bad=1 } END { exit bad }"'
check "sweep: front_right ≈ 2.0 m" 'echo "$right" | awk "{ if (\$1<1.9 || \$1>2.1) bad=1 } END { exit bad }"'
stop_fake

start_fake noecho
sleep 2.5
r=$(ranges /ultrasonic/front_left 5)
check "noecho: range = inf" 'echo "$r" | grep -qi inf'
stop_fake

start_fake corrupt
sleep 2.5
r=$(ranges /ultrasonic/front_left 20)
check "corrupt: 나온 값은 모두 정상 줄의 1.0 m" 'echo "$r" | awk "{ if (\$1<0.99 || \$1>1.01) bad=1 } END { exit bad }" && [ -n "$r" ]'
stop_fake

start_fake sensors
sleep 2.5
img=$(timeout 8 ros2 topic echo --csv --field width /thermal/image 2>/dev/null | head -n 1)
mx=$(timeout 8 ros2 topic echo --csv --field data /thermal/max 2>/dev/null | head -n 5)
tg=$(timeout 8 ros2 topic echo --csv --field poses /radar/targets 2>/dev/null | head -n 3)
vel=$(timeout 8 ros2 topic echo --csv --field data /radar/velocities 2>/dev/null | head -n 3)
check "sensors: /thermal/image 폭 32" '[ "$img" = "32" ]'
check "sensors: /thermal/max ≈ 80 °C" '[ -n "$mx" ] && echo "$mx" | awk "{ if (\$1<79.9 || \$1>80.1) bad=1 } END { exit bad }"'
check "sensors: /radar/targets 가 나옴" '[ -n "$tg" ]'
check "sensors: /radar/velocities 가 나옴" '[ -n "$vel" ]'
stop_fake

if [ "$fail" = 0 ]; then echo "전부 통과"; else echo "실패 있음 (다리 로그 /tmp/bridge_check.log)"; fi
exit $fail
