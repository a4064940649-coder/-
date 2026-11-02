import sys; sys.path.insert(0,'/home/user/-/research/tw_stock_selection/agents/o_disposal_official_fullhist/replicate_o9')
from ev import *
d=load_events()
ev=d[d.valid&d.adopt].sort_values(['si','announce','ord'])
ev=ev[(ev.s_date>='2006-04-01')&(ev.s_date<='2021-09-30')]
for r in ev.itertuples():
    j=cix[r.code]; s=r.si; e=r.ei
    if bear_before(s): continue
    if not (V60p[s,j]>=1e7 and PC[s,j]>=10): 
        if not okR[s,j]: print('liqfail&noR',r.code,r.s_date.date(),V60p[s,j],PC[s,j])
        continue
    x,dl=exit_idx(j,e)
    if not okR[s,j] or x is None or x<=s or dl:
        print(r.code,r.s_date.date(),r.e_date.date(),'R_s',R[s,j],'C_s',C[s,j],'x',x,s,dl,'lastR',IDX[lastR[j]].date(),'lastpx',IDX[lastpx[j]].date())
