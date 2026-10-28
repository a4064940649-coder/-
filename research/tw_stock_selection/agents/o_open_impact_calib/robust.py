#!/usr/bin/env python3
"""穩健性與執行量（PREREG 後；讀 run_cap 輸出與 results.json 的 RESULTS；寫入 results.json 的 ROBUST）：
  (1) 整股開盤委託：每級距（AB、κ 點、FIX／GROW）的衝擊率（imp − 0.1%）中位數與 p90、Q/OA_hat > 10%／20% 的比例、2019 前後。
  (2) 分產業：衝擊成本占比對成交值占比（stock_list 產業別；FIX 300 萬與 1,000 萬、seed 0～39）。
  (3) 去掉最大成本股：最大 1 檔與前 10 檔占衝擊成本的比例。
  (4) 各模型的衝擊成本（占權益 %／年）並列：M3V 點、p10、p90、M2、M1、O5、壓力。
  (5) 分年：主欄 FIX 各級距對 0050 年超額為負的年數與最差 3 年。
用法：python3 -I -B robust.py
"""
import sys, json, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import ecore6 as E
res = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
assert 'PREREG' in res and hashlib.sha256(json.dumps(res['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == res['PREREG_sha256']
assert 'RESULTS' in res
C = res['RESULTS']['cells']
RW = pd.read_parquet(E.DATA / 'cap_rows.parquet')
D = E.Data()
RW['code'] = [D.CODES[int(j)] for j in RW.j]; RW['date'] = D.IDX[RW.t.astype(int).to_numpy()]
RW['imp_rate'] = RW.impact_val / RW.Q
out = {}
ex = {}
for (st, ti, ve), g in RW[RW.lot == 1].groupby(['strat', 'tier', 'ver']):
    pre = g[g.date < '2019-01-01']; post = g[g.date >= '2019-01-01']
    ex[f'{st}|{ti}|{ve}'] = dict(n_lot_orders_per_seed=float(len(g) / 40), imp_rate_median_pct=float(g.imp_rate.median() * 100), imp_rate_p90_pct=float(g.imp_rate.quantile(.9) * 100),
                                 Q_median=float(g.Q.median()), share_q_over_oa_gt10=float((g.q_over_oa > 0.10).mean()), share_q_over_oa_gt20=float((g.q_over_oa > 0.20).mean()),
                                 share_gt10_pre2019=float((pre.q_over_oa > 0.10).mean()) if len(pre) else None, share_gt10_post2019=float((post.q_over_oa > 0.10).mean()) if len(post) else None)
out['lot_orders'] = ex
sl = pd.read_csv(HERE / 'raw' / 'screener' / 'stock_list.csv', dtype=str)
ind = dict(zip(sl.code, sl.industry))
ind_out = {}
for ti in ('300萬', '1000萬'):
    g = RW[(RW.strat == 'AB') & (RW.tier == ti) & (RW.ver == 'FIX') & (RW.lot == 1)].copy()
    g['ind'] = g.code.map(ind).fillna('（無）')
    a = g.groupby('ind').agg(imp=('impact_val', 'sum'), val=('Q', 'sum'))
    a['imp_share'] = a.imp / a.imp.sum(); a['val_share'] = a.val / a.val.sum()
    ind_out[ti] = a.sort_values('imp_share', ascending=False).head(8)[['imp_share', 'val_share']].round(4).to_dict(orient='index')
    bc = g.groupby('code').impact_val.sum().sort_values(ascending=False)
    out[f'top_cost_stock_{ti}_FIX'] = dict(top1=bc.index[0], top1_share=float(bc.iloc[0] / bc.sum()), top10_share=float(bc.iloc[:10].sum() / bc.sum()), n_stocks=int(len(bc)))
out['industry_impact_share_FIX'] = ind_out
mods = {}
for ti in E.TIERS:
    for ve in ('GROW', 'FIX'):
        row = {}
        for v in ('M3V_point', 'M3V_p10', 'M3V_p90', 'M2', 'M1', 'M3V_VAL', 'O5', 'STRESS', 'Z02', 'NOZ', 'E1', 'E2'):
            c = C.get(f'AB|{v}|{ti}|{ve}')
            if c: row[v] = dict(impact=c['cost_pct_equity_per_yr']['impact'], odd_spread=c['cost_pct_equity_per_yr']['odd_spread'], CAGR=c['CAGR'], t=c['t'])
        mods[f'{ti}|{ve}'] = row
out['models_side_by_side_AB'] = mods
yr = {}
for st in ('AB', 'B10'):
    for ti in E.TIERS:
        c = C[f'{st}|M3V_point|{ti}|FIX']['yearly_excess_vs_0050_pp']
        s = pd.Series({int(k): v for k, v in c.items()})
        yr[f'{st}|{ti}|FIX'] = dict(n_neg_years=int((s < 0).sum()), n_years=int(len(s)), worst3={int(k): float(v) for k, v in s.nsmallest(3).items()},
                                    best3={int(k): float(v) for k, v in s.nlargest(3).items()}, y2024_26={int(k): float(v) for k, v in s.items() if k >= 2024})
out['yearly'] = yr
res['ROBUST'] = out
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
print(json.dumps(out, ensure_ascii=False, indent=1, default=float)[:8000])
