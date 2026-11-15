# 분석 보조: 같은 결정 상태에서 갈래마다 K 번 (rs<r>_d<d>_k<f|0|1>) → 갈래별 성공·끝 시간 평균·표준오차, 유의한 쌍만 (성공 수 차 ≥ 2, 또는 둘 다 성공 ≥ 2 번이고 평균 차 > 2 표준오차)
# 출력: 상태별 "가장 좋은 갈래 − fuel" 분포 · 유의하게 fuel 을 이기는 상태 수 · <out>_{train,val}_{list,pairs}.txt
import csv, glob, os, sys, math, collections, statistics as S
O, L, out, d = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
VAL = {'procthor_00581', 'hssd_102344022_l0'}
A = collections.defaultdict(lambda: collections.defaultdict(list)); N = {}; FK = {}
for f in glob.glob(f'{O}/rs*_d{d}_k*_s21_*_branch.tsv'):
    b = os.path.basename(f); p = b.split('_'); arm = p[2]; sc = b[len('_'.join(p[:3]) + '_s21_'):-len('_branch.tsv')]
    tsv = f[:-len('_branch.tsv')] + '.tsv'
    if not os.path.exists(tsv): continue
    E = {int(r['env']): (int(r['done']) == 1, float(r['t'])) for r in csv.DictReader(open(tsv), delimiter='\t')}
    for r in csv.DictReader(open(f), delimiter='\t'):
        if r['applied'] != '1': continue
        e = int(r['env']); k = int(r['k'])
        A[(sc, e)][k].append(E[e]); N[(sc, e)] = int(r['n_cand']); FK[(sc, e)] = int(r['fuel_k'])
def stat(v):
    ok = [t for o, t in v if o]
    m = S.mean(ok) if ok else float('nan'); se = (S.stdev(ok) / math.sqrt(len(ok))) if len(ok) >= 2 else float('inf')
    return len(ok), len(v), m, se
def better(a, b):   # a 가 b 보다 유의하게 나음?
    na, ka, ma, sa = stat(a); nb, kb, mb, sb = stat(b)
    if na - nb >= 2: return True
    if na >= 2 and nb >= 2 and abs(na - nb) < 2 and mb - ma > 2 * math.sqrt(sa ** 2 + sb ** 2): return True
    return False
gaps = []; win_over_fuel = 0; fuel_wins = 0; nst = 0
for part in ('train', 'val'):
    fl = open(f'{out}_{part}_list.txt', 'w'); fp = open(f'{out}_{part}_pairs.txt', 'w'); idx = 0; npair = 0; ns = 0
    for key in sorted(A):
        sc, e = key
        if (sc in VAL) != (part == 'val'): continue
        arms = {k: v for k, v in A[key].items() if len(v) >= 3}
        fk = FK[key]
        if fk not in arms or len(arms) < 2: continue
        nst += 1
        fm = stat(arms[fk])
        others = [k for k in arms if k != fk]
        best = min(others, key=lambda k: (-stat(arms[k])[0], stat(arms[k])[2]))
        bs = stat(arms[best])
        if fm[0] == fm[1] and bs[0] == bs[1]: gaps.append(bs[2] - fm[2])
        win_over_fuel += any(better(arms[k], arms[fk]) for k in others)
        fuel_wins += all(better(arms[fk], arms[k]) for k in others)
        P = [(a, b) for a in arms for b in arms if a != b and better(arms[a], arms[b])]
        if not P: continue
        rq = f'{L}/d{d}/{sc}_s21_e{e}_d{d}.request.json'
        if not os.path.exists(rq): continue
        n = N[key]; ns += 1
        fl.write(f'{rq} -1\n{rq[:-13]}.rev.request.json -1\n')
        for w, l in P: fp.write(f'{idx} {w} {l}\n{idx + 1} {n - 1 - w} {n - 1 - l}\n'); npair += 2
        idx += 2
    print(part, '유의한 쌍이 있는 상태', ns, '쌍', npair)
gaps.sort()
q = lambda p: gaps[int(p * (len(gaps) - 1))] if gaps else float('nan')
print(f'상태 {nst} 개 (갈래마다 ≥ 3 번) · 다른 갈래가 fuel 을 유의하게 이김 {win_over_fuel} · fuel 이 모두를 유의하게 이김 {fuel_wins}')
print(f'모두 성공한 상태의 (가장 좋은 갈래 − fuel) 평균 끝 시간 차: n {len(gaps)} · 10/50/90 % = {q(.1):+.1f} / {q(.5):+.1f} / {q(.9):+.1f} s')
