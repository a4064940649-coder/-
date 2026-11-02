import sys; sys.path.insert(0,'/home/user/-/research/tw_stock_selection/agents/o_disposal_official_fullhist/replicate_o9')
import ev
from ev import *
d=load_events()
ev_=d[d.valid&d.adopt].sort_values(['si','announce','ord'])
ev_=ev_[(ev_.s_date>='2006-04-01')&(ev_.s_date<='2021-09-30')]
raw=C/PC-1
held={}
cnt={}
for r in ev_.itertuples():
    j=cix[r.code]; s=r.si; e=r.ei
    flags=[]
    if r.code in held and s<=held[r.code]: flags.append('held')
    if bear_before(s): flags.append('bear')
    if not okR[s,j]: flags.append('noR')
    elif R[s,j]>=lim(s)-0.01: flags.append('r3')
    if not (V60p[s,j]>=1e7 and PC[s,j]>=10): flags.append('liq')
    rr=raw[s,j]
    if flags==['r3'] or (flags==[] and not(rr<lim(s)-0.01)) or flags==['held']:
        print(r.code,r.s_date.date(),r.e_date.date(),flags,'R',round(R[s,j],4),'raw',round(rr,4),'C',C[s,j],'PC',PC[s,j], r.kind)
    if not flags:
        x,dl=exit_idx(j,e); held[r.code]=x
