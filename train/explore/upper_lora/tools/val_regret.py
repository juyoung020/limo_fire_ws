#!/usr/bin/env python3
"""val_regret.py <캐시> <.auc 파일> <pred 파일>… — 검증 요청에서 고른 선택지의 덮음 AUC 손해.

pred 파일: lora_train --pred (줄마다 "target n p_0 … p_{n-1}", 캐시 차례 — <캐시>.idx 로 요청 경로와 맞춤).
.auc 파일: tools/cov_labels.py (줄마다 "<요청 경로> n auc_0 … auc_{n-1} se").
표: 1 위 고름의 손해 = max auc − auc[argmax p] 평균, 기대 손해 = Σ p · (max auc − auc), 가장 좋은 것 맞춤 비율.
기준 줄: "늘 A"(규칙 cand 의 1 순위 = 글자 A), "고르게 무작위".
"""
import sys

cache, aucf, preds = sys.argv[1], sys.argv[2], sys.argv[3:]
idx = [l.split()[1] for l in open(cache + '.idx')]
A = {}
for l in open(aucf):
    t = l.split(); n = int(t[1]); A[t[0]] = [float(x) for x in t[2:2 + n]]

def row(name, picks):   # picks: 요청마다 확률 목록
    r = er = hit = 0.0; m = 0
    for p, P in zip(idx, picks):
        a = A[p]; b = max(a)
        k = max(range(len(P)), key=lambda i: P[i])
        r += b - a[k]; er += sum(P[i] * (b - a[i]) for i in range(len(a))); hit += a[k] == b; m += 1
    print(f'{name}\t{m}\t{r / m:.5f}\t{er / m:.5f}\t{hit / m:.3f}')

print('이름\tn\t1위손해\t기대손해\t최선맞춤')
row('늘 A', [[1.0] + [0.0] * (len(A[p]) - 1) for p in idx])
row('무작위', [[1.0 / len(A[p])] * len(A[p]) for p in idx])
for pf in preds:
    P = [[float(x) for x in l.split()[2:]] for l in open(pf)]
    assert len(P) == len(idx), (pf, len(P), len(idx))
    row(pf, P)
