# 獨立複製員 o20：引擎層（只讀研究員輸出的 seed 平均淨值 npz + 公開資料），自寫統計
import json, numpy as np, pandas as pd
P='/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel/'
OUT='/home/user/-/research/tw_stock_selection/agents/o_g6_allpit_list_trace_stage2/'
idx=pd.read_parquet(P+'close_raw.parquet').index
t=pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv',parse_dates=['date']).set_index('date')
extra=t.index[(t.index>='2024-01-02')&(t.index<'2024-02-15')]
eidx=idx.union(extra)
assert len(eidx)==5611
e=pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv',parse_dates=['date']).set_index('date')['adj']
e=e.reindex(e.index.union(eidx)).ffill().reindex(eidx)
z=np.load(OUT+'out/b_align_seedavg_navs.npz')
def calann(v0,v1,d0,d1): return (v1/v0)**(365.25/(d1-d0).days)-1
p0=eidx.get_loc(pd.Timestamp('2006-04-10')); p1=eidx.get_loc(pd.Timestamp('2026-10-12'))
def jspan(j): return eidx[p0+j], eidx[min(p1+j,len(eidx)-1)]
def stats(nav, start=None, j=None):
    s=pd.Series(nav,index=eidx)
    a0,a1=jspan(j); s=s[(s.index>=a0)&(s.index<=a1)].ffill()
    if start: s=s[s.index>=start]
    d0,d1=s.index[0],s.index[-1]
    a=calann(s.iloc[0],s.iloc[-1],d0,d1); b=calann(e[d0],e[d1],d0,d1)
    # 月 t：月底超額
    m=s.resample('ME').last(); me=e[s.index].resample('ME').last()
    rm=m.pct_change().dropna(); re=me.pct_change().dropna(); x=(rm-re).dropna()
    tt=x.mean()/x.std(ddof=1)*np.sqrt(len(x))
    return 100*(a-b), tt, d0, d1
res={}
for cell in ['N1000','S100','O30']:
    for v in ['V0','V1','V2','V3']:
        ex=[];ts=[];ey=[];ends=[]
        for j in range(13):
            a,tt,d0,d1=stats(z[f'{cell}|{v}|J{j}'],j=j); ex.append(a); ts.append(tt); ends.append(str(d1.date()))
            ey.append(stats(z[f'{cell}|{v}|J{j}'],'2019-01-01',j=j)[0])
        ex=np.array(ex);ey=np.array(ey);ts=np.array(ts)
        res[f'{cell}|{v}']=dict(exc=list(np.round(ex,4)),exc_med=float(np.median(ex)),med_j=int(np.argsort(ex)[6]),
            t_med=float(np.median(ts)),exc_y19_med=float(np.median(ey)),ends=ends, start=str(d0.date()))
for k,v in res.items(): print(k, round(v['exc_med'],3), 'j',v['med_j'], 't',round(v['t_med'],3),'y19',round(v['exc_y19_med'],3), v['ends'][0], v['ends'][-1])
json.dump(res,open(OUT+'replicate_o20/r1_engine_navs.json','w'),ensure_ascii=False,indent=1)
