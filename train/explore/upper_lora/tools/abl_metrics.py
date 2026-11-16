#!/usr/bin/env python3
"""abl_metrics.py <dir> [--base NAME] [--metric 2] [--only 장면정규식]
.lat 파일(<이름>_<장면>_s<씨앗>.lat, low_explore_check --lat-log)에서 덮음 지표 표.
지표 번호(coverage.h 순서): 0 가는 칸, 1..4 굵은 0.5 m × 문턱 0.1/0.2/0.3/0.5, 5..8 굵은 1.0 m × 같은 문턱. 기본 2 = 0.5 m @20 %.
판이 끝난 뒤(성공·멈춤 둘 다)는 마지막 값을 그대로 이어 간다.
AUC@H = 1..H 초 덮음 평균 (0–1). tXX = 처음 XX % 닿은 초 (못 닿으면 이 표에서는 '-', 평균에선 빼고 개수를 적음).
짝 비교: 같은 장면·씨앗끼리 기준(--base) 과의 차이 평균 ± SE.
"""
import sys, os, glob, re, math, collections

def load(path, metric):
    l = open(path).read().strip().split('\n')[0].split('\t')
    ser = [list(map(float, x.split(','))) for x in l[-1].split(';')]
    d = int(l[1]); d = 0 if d == 11 else d   # 11 = rough_done (고친 lat-log) — 옛 로그는 0
    return dict(done=d, T=float(l[2]), fb=int(l[3]), s=ser[metric])

def at(s, t):   # t 초 (1 부터) 값, 끝 뒤는 마지막 값
    return s[min(t, len(s)) - 1] if s else 0.0

def auc(s, H): return sum(at(s, t) for t in range(1, H + 1)) / H
def reach(s, th): return next((i + 1 for i, v in enumerate(s) if v >= th), None)

def mse(xs):
    n = len(xs)
    if n == 0: return float('nan'), float('nan')
    m = sum(xs) / n
    se = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n) if n > 1 else float('nan')
    return m, se

def main():
    a = sys.argv[1:]
    d = a[0]; base = None; metric = 2
    if '--base' in a: base = a[a.index('--base') + 1]
    if '--metric' in a: metric = int(a[a.index('--metric') + 1])
    only = a[a.index('--only') + 1] if '--only' in a else None   # 장면 이름 정규식
    runs = collections.defaultdict(dict)   # name -> (scene, seed) -> rec
    for f in sorted(glob.glob(os.path.join(d, '*.lat'))):
        m = re.match(r'(.+?)_(.+)_s(\d+)\.lat$', os.path.basename(f))
        if not m: continue
        if only and not re.search(only, m.group(2)): continue
        try: r = load(f, metric)
        except Exception: continue
        runs[m.group(1)][(m.group(2), int(m.group(3)))] = r
    cols = ['AUC90', 'AUC180', 'c60', 'c120', 't50', 't80', 't90']
    def row(r):
        s = r['s']
        return dict(AUC90=auc(s, 90), AUC180=auc(s, 180), c60=at(s, 60), c120=at(s, 120), t50=reach(s, .5), t80=reach(s, .8), t90=reach(s, .9))
    print('# 판별 (지표 %d)' % metric)
    print('이름\t장면\t씨앗\t끝\tT\t' + '\t'.join(cols))
    for n in sorted(runs):
        for k in sorted(runs[n]):
            r = runs[n][k]; v = row(r)
            print('%s\t%s\t%d\t%d\t%.0f\t' % (n, k[0], k[1], r['done'], r['T']) + '\t'.join('-' if v[c] is None else ('%.3f' % v[c] if isinstance(v[c], float) else str(v[c])) for c in cols))
    print('\n# 이름별 평균 (판 수, 못 닿은 판은 t 평균에서 뺌)')
    print('이름\tn\t' + '\t'.join(cols))
    for n in sorted(runs):
        vs = [row(r) for r in runs[n].values()]
        out = []
        for c in cols:
            xs = [v[c] for v in vs if v[c] is not None]
            m, _ = mse(xs)
            out.append(('%.3f' % m if c[0] != 't' else '%.0f' % m) + ('' if len(xs) == len(vs) else '(%d)' % len(xs)))
        print('%s\t%d\t' % (n, len(vs)) + '\t'.join(out))
    # 끝 코드 (mdp/curriculums.h Done): 0 = 후보 모드 '대충 다 앎' 끝(rough_done, lat-log 에 0 으로 찍힘), 1 닫힘 성공, 2 부딪힘, 3 시간 끝, 4 이른 선언, 5 사람 닿음, 6 진척 없음
    print('\n# 끝 코드 나눔과 끝날 때 덮음 (아래층 실패와 위층 효과를 가르기)')
    print('이름\tn\t대충끝0\t닫힘1\t부딪힘2\t시간3\t선언4\t사람5\t진척없음6\t끝시각평균\t끝덮음평균\t끝덮음(성공판)\t끝덮음(실패판)')
    for n in sorted(runs):
        rs = list(runs[n].values())
        cnt = collections.Counter(r['done'] for r in rs)
        fin = [r['s'][-1] if r['s'] else 0.0 for r in rs]
        ok = [r['s'][-1] for r in rs if r['done'] in (0, 1) and r['s']]
        bad = [r['s'][-1] for r in rs if r['done'] not in (0, 1) and r['s']]
        f = lambda xs: '%.3f(%d)' % (sum(xs) / len(xs), len(xs)) if xs else '-'
        print('%s\t%d\t' % (n, len(rs)) + '\t'.join(str(cnt.get(c, 0)) for c in range(0, 7)) +
              '\t%.0f\t%.3f\t%s\t%s' % (sum(r['T'] for r in rs) / len(rs), sum(fin) / len(fin), f(ok), f(bad)))
    if base and base in runs:
        print('\n# 짝 차이 (이름 − %s), 같은 장면·씨앗만, 평균 ± SE' % base)
        print('이름\tn\tdAUC90\tdAUC180\tdc60\tdc120')
        for n in sorted(runs):
            if n == base: continue
            ks = [k for k in runs[n] if k in runs[base]]
            ds = {c: [row(runs[n][k])[c] - row(runs[base][k])[c] for k in ks] for c in ['AUC90', 'AUC180', 'c60', 'c120']}
            print('%s\t%d\t' % (n, len(ks)) + '\t'.join('%+.3f±%.3f' % mse(ds[c]) for c in ['AUC90', 'AUC180', 'c60', 'c120']))

main()
