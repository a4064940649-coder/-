# AB FIX 衝擊成本 ≈ Σ impact_val ÷（40 seeds × 當年幣值級距）÷ 年數（近似：FIX 權益以級距計）
import pandas as pd, numpy as np, json
B='/home/user/-/research/tw_stock_selection/agents/o_open_impact_calib/'
r=pd.read_parquet(B+'data/cap_rows.parquet')
idx=pd.read_parquet(B+'data/px_traded.parquet').index
cpi=json.load(open(B+'raw/cpi_tw.json'))['index_2006_100']; cpi={int(k):v for k,v in cpi.items()}
r['yr']=idx[r.t.values].year
tiers={'30萬':3e5,'100萬':1e6,'300萬':3e6,'1000萬':1e7}
yrs=(pd.Timestamp('2026-10-12')-pd.Timestamp('2006-04-11')).days/365.25
for strat in ['AB','B10']:
  for ver in ['FIX','GROW']:
    out=[]
    for tn,tv in tiers.items():
        g=r[(r.strat==strat)&(r.ver==ver)&(r.tier==tn)]
        cap=g.yr.map(lambda y:tv*cpi[y]/cpi[2026])
        out.append(f'{tn}:{(g.impact_val/cap).sum()/40/yrs*100:.3f}')
    print(strat,ver,'impact cost %/yr (equity≈tier)',' '.join(out))
