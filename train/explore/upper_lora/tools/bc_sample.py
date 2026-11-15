# 분석 보조: --decider-log 결정 가운데 BC 캐시용 1500 개를 층별로 뽑음 (장면 · 후보 수 · "fuel 이 가장 가까운 후보가 아닌 것"), 검증은 장면으로 나눔
import json, glob, random, sys, collections
D = sys.argv[1]; out = sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
VAL = {'procthor_00581', 'hssd_102344022_l0'}   # 검증 장면 (학습에 안 씀)
rows = []
for f in glob.glob(D + '/*.jsonl'):
    for l in open(f):
        x = json.loads(l)
        if 'cands' not in x or x['label_fuel'] < 0 or x['n'] < 2: continue
        near = min(range(x['n']), key=lambda k: x['cands'][k]['t_s'])
        x['nb'] = 0 if x['n'] == 2 else 1 if x['n'] <= 4 else 2 if x['n'] <= 8 else 3
        x['nottop'] = int(x['label_fuel'] != near)
        rows.append(x)
random.seed(1)
st = collections.defaultdict(list)
for x in rows: st[(x['scene'], x['nb'], x['nottop'])].append(x)
for v in st.values(): random.shuffle(v)
pick = []; keys = sorted(st)
while len(pick) < min(N, len(rows)):   # 층마다 돌아가며 하나씩
    moved = False
    for k in keys:
        if st[k] and len(pick) < N: pick.append(st[k].pop()); moved = True
    if not moved: break
with open(out + '_train.txt', 'w') as ft, open(out + '_val.txt', 'w') as fv:
    for x in pick:
        f = fv if x['scene'] in VAL else ft
        rq = D + '/' + x['request']
        f.write(f"{rq} {x['label_fuel']}\n")
        f.write(f"{rq[:-13]}.rev.request.json {x['n'] - 1 - x['label_fuel']}\n")
c = collections.Counter((x['scene'] in VAL, x['nb'], x['nottop']) for x in pick)
print('결정', len(rows), '→ 뽑음', len(pick), '| 검증 장면', sum(x['scene'] in VAL for x in pick), '| fuel 이 가장 가까운 것 아님', sum(x['nottop'] for x in pick))
print(sorted(c.items()))
