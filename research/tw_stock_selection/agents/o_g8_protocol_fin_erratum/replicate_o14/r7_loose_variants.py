# 寬鬆口徑 794 vs 802 差異的敏感度（只計數）
import sys; sys.argv=['x']
import importlib.util, numpy as np, pandas as pd, io, contextlib
spec=importlib.util.spec_from_file_location('r6','/home/user/-/research/tw_stock_selection/agents/o_g8_protocol_fin_erratum/replicate_o14/r6_erratum.py')
r6=importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(r6)
V,SN,C0,days,v0,N,COLS=r6.V,r6.SN,r6.C0,r6.days,r6.v0,r6.N,r6.COLS
def pool(d):
    t=v0.cross_section(d)['t']; v60=v0.V60[t]; px=V.K.Cn[t]
    base=np.isfinite(v60)&(v60>=1e7)&np.isfinite(px)&(px>=10)
    o=np.argsort(-np.where(base,v60,-np.inf),kind='stable'); top=np.zeros(N,bool); top[o[:min(300,base.sum())]]=True; return top&base
P={d:pool(d) for d in days}
def count(win, snaps, incl=True):
    tot=0
    for d in days:
        lo=d-pd.Timedelta(days=win)
        prev=[s for s in snaps if (lo<=s[0] if incl else lo<s[0]) and s[0]<=d]
        nxt=[s for s in snaps if d<s[0] and (s[0]<=d+pd.Timedelta(days=win) if incl else s[0]<d+pd.Timedelta(days=win))]
        s=prev[-1] if prev else (nxt[0] if nxt else None)
        for j in np.where(P[d]&C0)[0]:
            if s is None or COLS[j] not in s[1]: tot+=1
    return tot
print('base',count(731,SN))
print('excl boundary',count(731,SN,False))
print('730',count(730,SN),'732',count(732,SN))
print('no twse_list',count(731,SN[:-1]))
print('no d9d8662c',count(731,[s for i,s in enumerate(SN) if i!=1]))
print('d9d8662c dated commit 2011-10-03? n/a')
def count_pool(pf):
    tot=0
    for d in days:
        lo=d-pd.Timedelta(days=731)
        prev=[s for s in SN if lo<=s[0]<=d]; nxt=[s for s in SN if d<s[0]<=d+pd.Timedelta(days=731)]
        s=prev[-1] if prev else (nxt[0] if nxt else None)
        for j in np.where(pf(d)&C0)[0]:
            if s is None or COLS[j] not in s[1]: tot+=1
    return tot
def p_trade(d):
    cs=v0.cross_section(d); e=cs['e']; return P[d]&np.isfinite(V.K.On[e])&V.K.TRADED[e]
def p_s52(d):
    cs=v0.cross_section(d); return P[d]&np.isfinite(cs['s52'])
def p_all(d):
    cs=v0.cross_section(d); e=cs['e']; return P[d]&np.isfinite(cs['s52'])&np.isfinite(V.K.On[e])&V.K.TRADED[e]&~V.K.disposal_mask(d)
print('trade',count_pool(p_trade),'s52',count_pool(p_s52),'all-but-rev',count_pool(p_all))
