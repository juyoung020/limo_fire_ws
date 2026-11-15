# 분석 보조: 변형 B 가 무엇에 기대는지 — 검증 요청 50 상태(원래·뒤집은 순서 = 100 줄)를 고르고, 묶음마다 그 정보만 후보 사이에서 똑같게(중앙값·최빈값) 만들거나 그림을 회색으로 바꾼 사본
# 출력 <dst>/<묶음>/*.json 과 <dst>/<묶음>.txt (d1_cache --list 형식), <dst>/base.txt
import json, sys, os, io, base64, statistics as S
from PIL import Image
src, dst = sys.argv[1], sys.argv[2]
L = [l.split() for l in open(src)]
pairs = [(L[i], L[i + 1]) for i in range(0, len(L) - 1, 2) if '.rev.' in L[i + 1][0] and L[i + 1][0].replace('.rev', '') == L[i][0]]
step = max(1, len(pairs) // 50); pick = pairs[::step][:50]
G = {'time': ['travel_time_s', 'path_distance_m'], 'rate': ['new_area_per_second'],
     'gain': ['expected_new_area_ratio', 'frontier_width_m', 'open_depth_beyond_m'],
     'room': ['same_room_as_robot', 'other_frontiers_in_its_room'],
     'risk': ['narrowest_gap_on_way_m', 'clutter_on_way', 'people_near_way'],
     'hist': ['is_current_target', 'times_tried_as_goal', 'times_abandoned_no_progress_or_path_lost'],
     'dir': ['bearing_deg_left_positive', 'marked_in_camera_image'], 'cam': 0, 'map': 1}
def gray(uri):
    im = Image.open(io.BytesIO(base64.b64decode(uri.split(',', 1)[1])))
    b = io.BytesIO(); Image.new('RGB', im.size, (128, 128, 128)).save(b, 'PNG')
    return 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode()
os.makedirs(dst, exist_ok=True)
with open(f'{dst}/base.txt', 'w') as f:
    for a, b in pick: f.write(' '.join(a) + '\n' + ' '.join(b) + '\n')
for g, keys in G.items():
    os.makedirs(f'{dst}/{g}', exist_ok=True)
    with open(f'{dst}/{g}.txt', 'w') as f:
        for a, b in pick:
            for p, t in (a, b):
                d = json.load(open(p)); C = d['state']['candidates']
                if isinstance(keys, int): d['images'][keys] = gray(d['images'][keys])
                else:
                    for k in keys:
                        v = [c[k] for c in C if k in c]
                        if not v: continue
                        n = False if isinstance(v[0], bool) else (S.median(v) if isinstance(v[0], (int, float)) else v[0])
                        for c in C:
                            if k in c: c[k] = n
                q = f'{dst}/{g}/{os.path.basename(p)}'; json.dump(d, open(q, 'w')); f.write(f'{q} {t}\n')
print(len(pick), '상태', len(G), '묶음')
