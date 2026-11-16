#!/usr/bin/env python3
# 아래층 정책 판정 (보조 스크립트 — 학습 코드 아님): 검증 세트 2 장면 10 × 조건 p2·p8 을 low_policy_run 으로 돌려 자취를 모아 판정 줄을 표(TSV)에 붙인다.
# 판정 계측 (scripts/README "아래층 판정"): 벽·사람 /100 m = 판마다 첫 정답 충돌(시뮬 정답 — 센서 닿음으로 끝난 걸음 포함, low_run 자취 끝 줄 7 칸째 비트),
#   거리는 첫 정답 충돌까지 (없으면 판 전체). 사람/벽은 정답 비트로, 둘 다면 끝 코드로. 통과 = 벽 ≤ 0.20 · 사람 ≤ 0.10 · reach* ≥ 0.90 · 속도 ≥ 0.676 · 장면·조건마다 사람 ≤ 0.30.
#   (2026-11-16 전 판은 끝 줄에 비트가 없어 센서 닿음으로 끝난 걸음의 정답 충돌을 못 셌다 — 그 판의 표 줄은 safety 칸 '|oldmetric')
# 쓰기: judge_low.py 태그 체크포인트 --bin low_policy_run --scenes 검증세트/final --tsv 표.tsv --out 자취폴더 [--seed 1 --envs 32] [--env K=V,K=-]
#   --seed · --envs 를 기본(1 · 32)과 다르게 주면 태그 끝에 c (확인 다시 재기). --env: 관측 설정 덮기 (값 '-' 이면 뺌 — 체크포인트마다 다른 관측)
import argparse, subprocess, sys, os, re, time, math
ap=argparse.ArgumentParser()
ap.add_argument('tag'); ap.add_argument('ckpt')
ap.add_argument('--bin',required=True); ap.add_argument('--scenes',required=True); ap.add_argument('--tsv',required=True); ap.add_argument('--out',required=True)
ap.add_argument('--slip',type=int,default=0,help='1 = 실물처럼 닿은 뒤 바퀴 헛돎·IMU 충격 (SIM_CON_SLIP=1, 고르기 확인·시험 세트는 이것) — 태그 끝 s, 표 safety 칸 slip1');ap.add_argument('--seed',default='1'); ap.add_argument('--envs',default='32'); ap.add_argument('--env',default='')
ap.add_argument('--cwd',default=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # train/explore
A=ap.parse_args()
it=A.tag; ck=A.ckpt; seed=A.seed; nenv=A.envs
tag=(it if seed=='1' and nenv=='32' else it+'c')+('s' if A.slip else '')
PHYS='slip1' if A.slip else 'off'
E=A.cwd; TS=A.scenes; T=A.tsv; OUT=A.out
JUDGE=[f'{TS}/{n}.scene' for n in ['procthor_val2_00002_f','procthor_val2_00042_f','procthor_val2_00009_f','procthor_val2_00032_f','hssd_106366386_174226770_l0','hssd_107734176_176000019_l0','hssd_104348028_171512877_l0','hssd_104348361_171513414_l0','procthor_val2_00002_f-lowobs','hssd_106366386_174226770_l0-lowobs']]
HSSD=[]
GROUPS={'ProcTHOR':['procthor_val2_00002_f','procthor_val2_00042_f','procthor_val2_00009_f','procthor_val2_00032_f'],'HSSD':['hssd_106366386_174226770_l0','hssd_107734176_176000019_l0','hssd_104348028_171512877_l0','hssd_104348361_171513414_l0'],'낮은 상자':['procthor_val2_00002_f-lowobs','hssd_106366386_174226770_l0-lowobs']}
COND={'p2':('2',{}),'p8':('8',dict(SIM_PEOPLE_P_STAND='0',SIM_PEOPLE_P_IGNORE='1',SIM_PEOPLE_P_APPROACH='0.25',SIM_PEOPLE_P_CROSS='0.25'))}
rows=[]
for cn,(pk,extra) in COND.items():
    env=dict(os.environ); env.update(SIM_LOW_OBS_MOV='1',SIM_LOW_GOALONLY='1',SIM_LOW_WIDE='1',SIM_LOW_CAM_EVERY='1',SIM_LOW_TRACK_MISS='20',SIM_LOW_MOV4='1',SIM_LOW_BODY='1'); env.update(extra)
    if A.slip: env['SIM_CON_SLIP']='1'
    for kv in [x for x in A.env.split(',') if x]:   # 체크포인트마다 관측 설정 (K=V, 값 '-' 이면 뺌)
        k,v=kv.split('=',1)
        if v=='-': env.pop(k,None)
        else: env[k]=v
    tr=f'{OUT}/w3v_{tag}_{cn}.tsv'; log=f'{OUT}/w3v_{tag}_{cn}.log'
    cmd=[A.bin,'--mode','follow','--policy',ck,'--people',pk,'--episodes',nenv,'--per-env','2','--seed',seed,'--scenes']+JUDGE+HSSD+['--trace',tr]
    with open(log,'w') as f: rc=subprocess.call(cmd,cwd=E,env=env,stdout=f,stderr=subprocess.STDOUT)
    txt=open(log).read()
    res={}
    effs={}
    for blk in txt.split('#### ')[1:]:
        mm=re.match(r'단독 과제\(정책\) (\S+)',blk)
        me=re.search(r'도착 효율: 유효 목표 (\d+) \(도달 (\d+)\) · 실효 속도 평균 ([0-9.]+) · 중앙 ([0-9.]+) m/s · 1 m 당 ([0-9.]+) s · SCT ([0-9.]+) · SPL ([0-9.]+)',blk)
        if mm and me: effs[mm.group(1)]=[int(me.group(1)),int(me.group(2))]+[float(me.group(k)) for k in range(3,8)]
    for m in re.finditer(r'#### 단독 과제\(정책\) (\S+) · 사람 상한 (\d+).*?평균 속도 ([0-9.]+) m/s \| 100 m 당: 벽 충돌 [0-9.]+ · 사람 충돌 앞 ([0-9.]+) · 밖 ([0-9.]+).*?\n모두\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)', txt, re.S):
        fa,fb=float(m.group(4)),float(m.group(5))
        res[m.group(1)]=dict(speed=float(m.group(3)),front=(fa/(fa+fb) if fa+fb>0 else float('nan')),goals=int(m.group(6)),reachx=float(m.group(8)))
    eps={}
    for l in open(tr):
        p=l.rstrip('\n').split('\t')
        if p[0]=='scene': continue
        if len(p)==7:   # 판 끝 줄 (2026-11-16 계측 수정): 그 판의 정답 충돌 비트 (끝 걸음 포함) — 아직 안 센 충돌이면 셈 (센서 닿음으로 끝난 걸음 포함)
            k=(p[0],p[2],p[3]); e=eps.get(k)
            if e is None or e['done']: continue
            bits=int(p[6]); endc=int(p[5])
            if bits:
                e['done']=True; e['hit']=1; e['hp']=1 if (bits==2 or (bits==3 and endc==5)) else 0   # 정답 비트로 (둘 다면 끝 코드로 가름)
            continue
        if len(p)<30: continue
        k=(p[0],p[2],p[3]); x,y=float(p[6]),float(p[7])
        e=eps.setdefault(k,dict(px=x,py=y,d=0.0,hit=0,hp=0,done=False))
        if e['done']: continue
        e['d']+=math.hypot(x-e['px'],y-e['py']); e['px'],e['py']=x,y
        if p[25]=='1': e['done']=True; e['hit']=1; e['hp']=int(p[26]=='1')
    agg={}
    for (sc,_,_),e in eps.items():
        a=agg.setdefault(sc,[0.0,0,0,0]); a[0]+=e['d']; a[1]+=e['hit']-e['hp']; a[2]+=e['hp']; a[3]+=1
    for path in JUDGE+HSSD:
        sc=os.path.basename(path)[:-6]; r=res.get(sc); a=agg.get(sc,[0.0,0,0,0])
        if r is None: rows.append(dict(set='?',sc=sc,cn=cn,err=1)); continue
        rows.append(dict(eff=effs.get(sc),set='val2' if path in JUDGE else 'x',sc=sc,cn=cn,d=a[0],w=a[1],p=a[2],n=a[3],front=r['front'],reachx=r['reachx'],speed=r['speed'],goals=r['goals']))
ts=time.strftime('%Y-%m-%d %H:%M:%S')
out=[]
def effs_(e): return ['%.3f'%e[2],'%.3f'%e[3],'%.2f'%e[4],'%.3f'%e[5],'%.3f'%e[6]] if e else ['','','','','']
def line(set_,sc,cn,w,p,fr,rx,sp,g,n,ps,ef=None): out.append('\t'.join([ts,tag,PHYS,set_,sc,cn,'%.3f'%w,'%.3f'%p,fr,'%.3f'%rx,'%.3f'%sp,str(g),str(n),ps]+effs_(ef)))
for r in rows:
    if r.get('err'): out.append('\t'.join([ts,tag,PHYS,'?',r['sc'],r['cn'],'','','','','','','','parse-fail'])); continue
    w=100*r['w']/r['d'] if r['d'] else 0; p=100*r['p']/r['d'] if r['d'] else 0
    line(r['set'],r['sc'],r['cn'],w,p,'%.2f'%r['front'] if r['front']==r['front'] else '',r['reachx'],r['speed'],r['goals'],r['n'],'',r.get('eff'))
J=[r for r in rows if r.get('set')=='val2']
D=sum(r['d'] for r in J); W=sum(r['w'] for r in J); P=sum(r['p'] for r in J); G=sum(r['goals'] for r in J)
rx=sum(r['reachx']*r['goals'] for r in J)/G if G else 0; sp=sum(r['speed']*r['goals'] for r in J)/G if G else 0
w=100*W/D if D else 0; p=100*P/D if D else 0
worst=max((100*r['p']/r['d'] if r['d'] else 0) for r in J) if J else 0
ok= w<=0.20 and p<=0.10 and rx>=0.90 and sp>=0.676 and worst<=0.30 and len(J)==20
# 대략 95 % 구간 (포아송, 충돌 수 k 를 (√k ∓ 0.98)² · (√(k+1) + 0.98)² 로 — 거리로 나눔)
def ci(k,d):
    if d<=0: return (0.0,0.0)
    lo=max(0.0,(math.sqrt(k)-0.98))**2 if k>0 else 0.0; hi=(math.sqrt(k+1)+0.98)**2
    return (100*lo/d,100*hi/d)
wl,wh=ci(W,D); pl,ph=ci(P,D)
EJ=[r['eff'] for r in J if r.get('eff')]
ng=sum(e[0] for e in EJ); nr=sum(e[1] for e in EJ)
all_eff=[0,0,sum(e[2]*e[1] for e in EJ)/nr if nr else 0, sum(e[3]*e[1] for e in EJ)/nr if nr else 0, sum(e[4]*e[1] for e in EJ)/nr if nr else 0, sum(e[5]*e[0] for e in EJ)/ng if ng else 0, sum(e[6]*e[0] for e in EJ)/ng if ng else 0] if EJ else None
line('val2','ALL','p2+p8',w,p,'',rx,sp,G,sum(r['n'] for r in J),('PASS' if ok else 'fail')+' (벽 95%% %.2f~%.2f · 사람 %.2f~%.2f · 충돌 벽 %d 사람 %d · %.0f m)'%(wl,wh,pl,ph,W,P,D),all_eff)
with open(T,'a') as f:
    for o in out: f.write(o+'\n')
print(f'SET it {tag}: wall {w:.3f} [{wl:.2f}~{wh:.2f}] person {p:.3f} [{pl:.2f}~{ph:.2f}] (worst scene person {worst:.3f}) reach* {rx:.3f} speed {sp:.3f} -> {"PASS" if ok else "fail"}'+(f' | eff v {all_eff[2]:.3f} med {all_eff[3]:.3f} s/m {all_eff[4]:.2f} SCT {all_eff[5]:.3f} SPL {all_eff[6]:.3f}' if all_eff else ''), flush=True)

# 묶음별 보고 (판정 아님 — README): 같은 계산을 묶음 장면만으로
for gname,names in GROUPS.items():
    Jg=[r for r in J if r['sc'] in names]
    D=sum(r['d'] for r in Jg); W=sum(r['w'] for r in Jg); P=sum(r['p'] for r in Jg); G=sum(r['goals'] for r in Jg)
    rxg=sum(r['reachx']*r['goals'] for r in Jg)/G if G else 0; spg=sum(r['speed']*r['goals'] for r in Jg)/G if G else 0
    EJ=[r['eff'] for r in Jg if r.get('eff')]; ng=sum(e[0] for e in EJ); nr=sum(e[1] for e in EJ)
    sct=sum(e[5]*e[0] for e in EJ)/ng if ng else 0; spl=sum(e[6]*e[0] for e in EJ)/ng if ng else 0; ev=sum(e[2]*e[1] for e in EJ)/nr if nr else 0
    wl,wh=ci(W,D); pl,ph=ci(P,D)
    msg=f'GROUP {gname}: 벽 {100*W/D if D else 0:.3f} [{wl:.2f}~{wh:.2f}] 사람 {100*P/D if D else 0:.3f} [{pl:.2f}~{ph:.2f}] reach* {rxg:.3f} 속도 {spg:.3f} 실효 {ev:.3f} SCT {sct:.3f} SPL {spl:.3f} (목표 {G}, {D:.0f} m)'
    print(msg, flush=True)
    with open(T,'a') as f: f.write(ts+'\t'+tag+'\tgroup\t'+msg+'\n')
