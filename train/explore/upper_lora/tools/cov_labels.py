#!/usr/bin/env python3
"""cov_labels.py <라벨 폴더> <출력 앞머리> [--req-dir <다시 그린 요청 폴더>] [--val-frac 0.1] [--metric 2] [--horizon 180] [--mc 2000] [--seed 1]

dec_rollout 출력(<장면>_s<씨앗>.jsonl + _r<요청>.req.json)에서 판단 모델 LoRA 학습 목록을 만든다.
- 요청 본문: --req-dir 이면 dec_rollout --render 로 지금 노드 꼴로 다시 그린 것(같은 파일 이름), 아니면 굴릴 때 남긴 .req.json.
- 선택지 k (글자 'A'+k) 의 값 = 그 갈래들의 덮음 AUC 평균: 지표 --metric(기본 2 = 굵은 0.5 m @20 %)의 갈래 뒤 1 s 값들을
  --horizon 초까지 평균 (판이 일찍 끝나면 마지막 값을 이어 감). 요청에 실린 후보(최대 6)만 — jsonl 의 나머지 후보 갈래는 글자가 없어 버림.
- 부드러운 정답 q_k = P(k 가 가장 좋음): 선택지마다 평균 ± 표준오차(갈래 반복 사이, 아래 한계 0.002)인 정규분포에서 --mc 번 뽑아 1 등 비율.
  반복 잡음보다 작은 차이는 고르게 퍼지고, 분명한 차이만 한쪽으로 몰림.
- 검증은 장면으로 나눔(같은 장면의 요청이 학습·검증에 섞이지 않게): 장면 이름 해시로 --val-frac.
출력:
  <앞머리>_{train,val}.list   줄마다 "<요청 경로> <가장 좋은 선택지 번호>" (d1_cache --list)
  <앞머리>_{train,val}.soft   줄마다 "<요청 경로> n q_0 … q_{n-1}" (lora_train --soft — d1_cache 의 <캐시>.idx 로 맞춤)
  <앞머리>_{train,val}.auc    줄마다 "<요청 경로> n auc_0 … auc_{n-1} se" (평가: 고른 것의 AUC − 가장 좋은 AUC)
"""
import sys, os, json, glob, math, random, hashlib, statistics as S

a = sys.argv[1:]
src, pre = a[0], a[1]
opt = lambda k, d: type(d)(a[a.index(k) + 1]) if k in a else d
rdir = opt('--req-dir', '')
vf, metric, H, mc, seed = opt('--val-frac', 0.1), opt('--metric', 2), opt('--horizon', 180), opt('--mc', 2000), opt('--seed', 1)
rng = random.Random(seed)

def auc(o):
    if len(o) < 4 or len(o[3]) <= metric or not o[3][metric]: return None
    s = o[3][metric]; s = s + [s[-1]] * (H - len(s))
    return sum(s[:H]) / H

out = {sp: [open(f'{pre}_{sp}.{x}', 'w') for x in ('list', 'soft', 'auc')] for sp in ('train', 'val')}
n = {'train': 0, 'val': 0}; skip = 0; scenes = {'train': set(), 'val': set()}
for f in sorted(glob.glob(os.path.join(src, '*.jsonl'))):
    base = f[:-6]
    scene = os.path.basename(base).rsplit('_s', 1)[0]
    sp = 'val' if int(hashlib.md5(scene.encode()).hexdigest(), 16) % 1000 < vf * 1000 else 'train'
    for line in open(f):
        d = json.loads(line)
        rp = os.path.abspath(f"{base}_r{d['req']}.req.json")
        if rdir: rp = os.path.abspath(os.path.join(rdir, os.path.basename(rp)))
        if not os.path.exists(rp): skip += 1; continue
        req = json.load(open(rp))
        k = len(req['state']['candidates'])
        if k < 2: skip += 1; continue
        A, E, ok = [], [], True
        for i in range(k):
            v = [x for x in (auc(o) for o in d['cand'][i]['out']) if x is not None]
            if not v: ok = False; break
            A.append(S.mean(v)); E.append(max(0.002, S.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else 0.01))
        if not ok: skip += 1; continue
        win = [0] * k
        for _ in range(mc):
            x = [rng.gauss(A[i], E[i]) for i in range(k)]
            win[max(range(k), key=lambda i: x[i])] += 1
        q = [w / mc for w in win]
        best = max(range(k), key=lambda i: A[i])
        fl, fs, fa = out[sp]
        fl.write(f'{rp} {best}\n')
        fs.write(f'{rp} {k} ' + ' '.join(f'{p:.4f}' for p in q) + '\n')
        fa.write(f'{rp} {k} ' + ' '.join(f'{x:.5f}' for x in A) + f' {S.mean(E):.5f}\n')
        n[sp] += 1; scenes[sp].add(scene)
for v in out.values():
    for h in v: h.close()
print(f"학습 {n['train']} 요청 ({len(scenes['train'])} 장면) · 검증 {n['val']} 요청 ({len(scenes['val'])} 장면) · 뺌 {skip}")
