# 분석 보조: BC 변형 B — 요청에서 "extra_total_time_if_chosen_first_s"(fuel 이 고른 후보는 0 이라 답이 새는 값)와 그 설명 문장을 뺀 사본
import json, sys, os
src, dst = sys.argv[1], sys.argv[2]
for part in ('train', 'val'):
    out = open(f'{os.path.dirname(dst)}/bcselB_{part}.txt', 'w')
    for l in open(f'{src}_{part}.txt'):
        p, t = l.split()
        b = json.load(open(p))
        for c in b['state']['candidates']: c.pop('extra_total_time_if_chosen_first_s', None)
        g = b['state'].get('candidate_fields', '')
        i = g.find('extra_total_time_if_chosen_first_s')
        if i >= 0:
            j = g.find(';', i)
            b['state']['candidate_fields'] = g[:i] + g[j + 2:]
        q = dst + '/' + os.path.basename(p)
        json.dump(b, open(q, 'w'))
        out.write(f'{q} {t}\n')
