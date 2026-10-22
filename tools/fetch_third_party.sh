#!/bin/bash
# 외부 저장소를 워크스페이스 안에 받는다. 저장소 루트에서 실행: ./tools/fetch_third_party.sh
set -e
cd "$(dirname "$0")/.."
[ -d src/wego_limo_pro ] || git clone --depth 1 https://github.com/WeGo-Robotics/limo_pro.git src/wego_limo_pro
[ -d sim/wego_LimoIsaacSIM ] || git clone --depth 1 -b sim https://github.com/WeGo-Robotics/LimoIsaacSIM.git sim/wego_LimoIsaacSIM
echo "Isaac Sim USD(WegoLimoUSD.zip, 약 65MB)는 WeGo 가이드에서 직접 받아 sim/wego_LimoIsaacSIM/USD/ 에 푼다:"
echo "  https://goofy-pleasure-a84.notion.site/Limo-Isaac-Sim-1ff09b2ac4bf8086bdf5e94521f74946"
echo "  unzip WegoLimoUSD.zip -d sim/wego_LimoIsaacSIM/USD/"
