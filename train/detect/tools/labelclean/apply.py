# SigLIP 검사 결과로 SAM 3 라벨에서 틀린 상자를 뺀다: pos < 문턱이면 그 검출을 지움. 메타(.meta.json)도 복사.
# 사용: apply.py <sam3.jsonl> <out.jsonl> <검사.jsonl>:<문턱> [...]
import json, shutil, sys
src, out = sys.argv[1], sys.argv[2]
drop = set()
for arg in sys.argv[3:]:
    path, thr = arg.rsplit(':', 1)
    for line in open(path):
        r = json.loads(line)
        if r['pos'] < float(thr): drop.add((r['img'], r['k']))
n = 0
with open(out, 'w') as fo:
    for line in open(src):
        r = json.loads(line)
        if 'det' in r:
            keep = [d for k, d in enumerate(r['det']) if (r['img'], k) not in drop]
            n += len(r['det']) - len(keep)
            r['det'] = keep
        fo.write(json.dumps(r, separators=(',', ':')) + '\n')
shutil.copy(src + '.meta.json', out + '.meta.json')
print(f'뺀 상자 {n} 개 (목록 {len(drop)})')
