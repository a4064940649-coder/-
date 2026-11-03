#!/usr/bin/env python3
"""反方審查（成本與可成交）o12：只讀研究員的檔案與資料，不改任何研究員檔案。python3 -I -B。
A. 營收訊號資料時點：每個 j=5 訊號日，前 100 中 y3_ab 有值的市值占比、可輪動產業 REV3 為 NaN 的比例（含近期）。
B. 每個 j 的驗證期紙上超額（對 0050、對 B），看規則書主推的 j≈5 是否落在中位數之下。
C. 可執行引擎相對 B 的「疊加成本」與紙上成本比較；扣成本前疊加超額的損益兩平成本。
D. 2019 起 100 萬 GROW 的累積財富差。"""
import sys, json, math
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
RES = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
W = pd.read_parquet(HERE / 'data' / 'oos_windows.parquet')
SEL = 'H52|k3|A15|M'
out = {}

def t(x):
    x = pd.Series(x).dropna(); return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))) if len(x) > 2 else np.nan

# ---------- B ----------
rows = []
for j in range(13):
    a = W[f'{SEL}|normal|J{j}'].dropna(); g = W[f'{SEL}|zero|J{j}'].dropna()
    e = W[f'0050|J{j}'].reindex(a.index); b = W[f'B|normal|J{j}'].reindex(a.index); bz = W[f'B|zero|J{j}'].reindex(g.index)
    m = (a.index >= '2017-01-01')
    rows.append(dict(j=j, n=int(m.sum()), vs0050=float((a - e)[m].mean() * 100), t0050=t((a - e)[m]), vsB=float((a - b)[m].mean() * 100), tB=t((a - b)[m]),
                     gross_vsB=float((g - bz)[g.index >= '2017-01-01'].mean() * 100),
                     cost_overlay=float(((g - bz) - (a - b)).dropna()[lambda s: s.index >= '2017-01-01'].mean() * 100)))
df = pd.DataFrame(rows).set_index('j')
out['B_perj_VAL'] = df.round(4).to_dict('index')
out['B_j5_rank_vs0050'] = int((df['vs0050'] < df.loc[5, 'vs0050']).sum())  # 幾個 j 比 j5 差
print(df.round(3))

# ---------- C ----------
E = RES['EXEC']['cells']; C = {}
for tier, mode in (('30萬', 'FIX'), ('100萬', 'FIX'), ('100萬', 'GROW'), ('300萬', 'GROW'), ('1000萬', 'GROW')):
    s = E[f'{SEL}|{tier}|{mode}|normal']; bb = E[f'B|{tier}|{mode}|normal']
    C[f'{tier}{mode}'] = dict(cost_sel=s['cost_pp_per_month_jmed'], cost_B=bb['cost_pp_per_month_jmed'],
                             overlay_cost=s['cost_pp_per_month_jmed'] - bb['cost_pp_per_month_jmed'],
                             vsB_VAL=s['VAL']['vsB_mean_jmed'], vsB_Y19=s['Y19']['vsB_mean_jmed'],
                             orders=s['orders_per_month_jmed'], orders_B=bb['orders_per_month_jmed'])
out['C_exec_overlay_cost'] = C
tc = RES['OOS']['turnover_cost']
out['C_paper_overlay_cost'] = tc[SEL]['cost_pp_m'] - tc['B']['cost_pp_m']
print(json.dumps(C, ensure_ascii=False, indent=1)); print('paper overlay cost', out['C_paper_overlay_cost'])

# ---------- D ----------
y = RES['EXEC']['cells'][f'{SEL}|100萬|GROW|normal']['Y19']
yrs = 7 + 10 / 12
out['D_wealth_100w_Y19'] = dict(sel=(1 + y['cagr_jmed']) ** yrs, e50=(1 + y['cagr0050']) ** yrs)
print(out['D_wealth_100w_Y19'])

# ---------- A ----------
import lcrun as R, lclib as X
R.init()
P = RES['PREREG']['chosen']['params']
days = X.jdays(5, '2014-01', '2026-10', '2026-11')[:-1]
A = []
for d in days:
    cs = R.cs_at(d, P); ST = cs['ST']
    y3 = X.TC.y3_ab(pd.Timestamp(d))[cs['U']].astype(float)
    cov = float(cs['mc'][np.isfinite(y3)].sum() / cs['mc'].sum())
    nan_share = float(ST['REV3'].isna().mean()) if len(ST) else np.nan
    nanB = float(ST.loc[ST['REV3'].isna(), 'B'].sum() / ST['B'].sum()) if len(ST) else np.nan
    A.append(dict(d=str(pd.Timestamp(d).date()), y3cov_top100=cov, rev3_nan_sectors=nan_share, rev3_nan_Bshare=nanB, n_sect=len(ST)))
A = pd.DataFrame(A).set_index('d')
A['yr'] = A.index.str[:4]
summ = A.groupby('yr')[['y3cov_top100', 'rev3_nan_sectors', 'rev3_nan_Bshare']].mean().round(3)
print(summ); print(A.tail(14).round(3))
out['A_rev_cov_by_year'] = summ.to_dict('index')
out['A_last14'] = A.tail(14).round(3).drop(columns='yr').to_dict('index')
(Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad') / 'review_exec_o12.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
