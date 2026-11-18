#!/bin/sh
# 펌웨어의 호스트 시험(보드 없이): 프레임 조립·체크섬·레이더 부호
cd "$(dirname "$0")/.." && g++ -std=c++17 -Wall -Wextra -Iinclude test/test_sensor_frames.cpp -o /tmp/test_sensor_frames && /tmp/test_sensor_frames
