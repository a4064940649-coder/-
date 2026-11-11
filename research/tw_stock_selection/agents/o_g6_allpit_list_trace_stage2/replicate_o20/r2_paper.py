# 紙上：只讀 k_paper_windows.npz（研究員 K 角色輸出的逐窗報酬），自寫統計
import numpy as np, json
from scipy.optimize import brentq
OUT='/home/user/-/research/tw_stock_selection/agents/o_g6_allpit_list_trace_stage2/'
z=np.load(OUT+'out/k_paper_windows.npz')
def ann(r): return np.prod(1+r)**(12/len(r))-1
def t(x): return x.mean()/x.std(ddof=1)*np.sqrt(len(x))
res={}
for M in ['M0','M1','M2']:
  for V in ['ALLPIT_REAL_opt','ALLPIT_REAL_p90','ALLPIT_REAL_carry']:
    for c in ['normal','stress']:
      k=f'{M}|{V}|{c}|'; r=z[k+'r']; b=z[k+'r50']; per=z[k+'per']
      x=r-b; yrs=np.array([p[:4] for p in per])
      d=dict(n=len(r),ann_ia=100*ann(r),exc_ia=100*(ann(r)-ann(b)),t=t(x))
      # 三段：2006-04~2012-12 / 2013-01~2019-04 / 2019-05~
      s1=per<='2012-12'; s3=per>='2019-05'; s2=~s1&~s3
      d['seg']=[100*(ann(r[m])-ann(b[m])) for m in (s1,s2,s3)]
      # 拿掉最好 3 年（依日曆年超額）
      uy=sorted(set(yrs)); ye={y:np.prod(1+r[yrs==y])-np.prod(1+b[yrs==y]) for y in uy}
      best=sorted(uy,key=lambda y:-ye[y])[:3]; m=~np.isin(yrs,best); d['t_x_best3']=t(x[m]); d['best3']=best
      res[k[:-1]]=d
for k,d in res.items(): print(k, {a:(np.round(b,3) if not isinstance(b,list) else b) for a,b in d.items()})
# 名單 M1->M2 差
json.dump(res,open(OUT+'replicate_o20/r2_paper.json','w'),default=lambda o: o.tolist() if hasattr(o,'tolist') else float(o),indent=1)
