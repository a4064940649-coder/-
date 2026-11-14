# -*- coding: utf-8 -*-
"""凍結後：與研究員 trades.csv.gz 逐筆比對，追 E2/E4/E5 第一筆不同的交易（平盤觸價的浮點問題）。python3 -I -B compare_trades.py"""
import sys; sys.path.insert(0,'/home/user/-/research/tw_stock_selection/agents/o_ext_vcp_xs/replicate')
import rep_lib as L, numpy as np, pandas as pd
D=L.Data()
B='/home/user/-/research/tw_stock_selection/agents/o_ext_vcp_xs/'
t=pd.read_csv(B+'trades.csv.gz',dtype={'code':str},parse_dates=['signal_date','entry_date','exit_date'])
mt=pd.read_csv(B+'replicate/my_trades.csv.gz',dtype={'code':str},parse_dates=['sig_date','entry','exit'])
mt=mt[mt.idle=='cash']
sig=pd.read_csv(B+'replicate/my_signals.csv',dtype={'code':str},parse_dates=['date'])
pos={c:i for i,c in enumerate(D.codes)}
def show(code,a,b):
    j=pos[code]; sl=(D.dates>=a)&(D.dates<=b)
    print(pd.DataFrame({'O':D.O[sl,j],'H':D.H[sl,j],'L':D.L[sl,j],'C':D.C[sl,j],'F':D.F[sl,j],'A':D.A[sl,j],'ret':D.R[sl,j],'flag':D.FLAG[sl,j]},index=D.dates[sl]).to_string())
for rule,mr in [('E5_close_below_pivot','E5'),('E4_stop8_tp20','E4'),('E2_hold60','E2')]:
    A=t[t.rule==rule]; Bm=mt[mt.rule==mr]
    J=A.merge(Bm,left_on=['code','entry_date'],right_on=['code','entry'])
    d=J[(J.exit_date!=J.exit)|((J.ret_net-J.ret).abs()>1e-3)].sort_values('entry_date')
    print(rule,'n diff exits',len(d)); print(d[['code','signal_date','entry_date','exit_date','exit','reason_x','reason_y','exit_px_raw','exit_px','ret_net','ret','entry_open']].head(8).to_string())
print('--- 3515')
r=sig[(sig.code=='3515')&(sig.date=='2009-11-19')].iloc[0]; print(r.pivot, r.pivot_adj, r.close)
show('3515','2009-11-18','2009-11-27')
j=pos['3515']; i=list(D.dates).index(pd.Timestamp('2009-11-25')); print(repr(D.A[i,j]), repr(r.pivot*D.F[list(D.dates).index(pd.Timestamp('2009-11-19')),j]), repr(D.C[i,j]))
print('--- 2403'); show('2403','2010-08-10','2010-08-18'); show('2403','2010-09-14','2010-09-20')
print(repr(np.ceil(54.2*1.2/0.1-1e-7)*0.1), 54.2*1.2)
print('--- 2327'); show('2327','2014-05-08','2014-05-15'); print(14.9*1.2)
# E5: classify the 7 diffs: is their flag-day close == pivot?
A=t[t.rule=='E5_close_below_pivot']; Bm=mt[mt.rule=='E5']
J=A.merge(Bm,left_on=['code','entry_date'],right_on=['code','entry'])
d=J[(J.exit_date!=J.exit)]
for r in d.itertuples():
    j=pos[r.code]; s=sig[(sig.code==r.code)&(sig.date==r.signal_date)].iloc[0]
    ie=list(D.dates).index(r.exit_date)
    # previous bar of their exit
    k=ie-1
    while np.isnan(D.C[k,j]): k-=1
    print(r.code, str(r.signal_date)[:10], 'pivot',s.pivot,'F(d)/F(k)',D.F[list(D.dates).index(s.date),j]/D.F[k,j],'their flag-day close',D.C[k,j], 'raw equal:',abs(D.C[k,j]*D.F[k,j]-s.pivot_adj)<1e-9)
# E4 diffs: exact touch?
A=t[t.rule=='E4_stop8_tp20']; Bm=mt[mt.rule=='E4']
J=A.merge(Bm,left_on=['code','entry_date'],right_on=['code','entry'])
d=J[(J.exit_date!=J.exit)]
n=0
for r in d.itertuples():
    j=pos[r.code]; ie=list(D.dates).index(r.entry_date); im=list(D.dates).index(r.exit)
    tp=L.ceil_tick(D.O[ie,j]*1.2)*D.F[ie,j]
    touch=abs(D.AH[im,j]-tp)<1e-9*tp
    n+=touch
    print(r.code,str(r.entry_date)[:10],'mine',str(r.exit)[:10],r.reason_y,'theirs',str(r.exit_date)[:10],r.reason_x,'exact TP touch on my exit day:',touch, 'my ret %.3f their %.3f'%(r.ret,r.ret_net))
print('E4 exit-date diffs',len(d),'exact-touch',n)
