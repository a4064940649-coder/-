#!/usr/bin/env python3
"""事後診斷（不參與判定）：事件右尾依賴、只剔除 flag 3/4/5、優先順序隨機化、事件持有期內的同期 0050。執行：python3 -I check.py"""
import json, sys, runpy
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
# 重用 run.py 的全部物件（不寫檔：先備份 results.json，跑完還原再附加 posthoc）
bak = (HERE / 'results.json').read_text(encoding='utf-8')
navbak = (HERE / 'nav.parquet').read_bytes()
g = runpy.run_path(str(HERE / 'run.py'), run_name='bbo_check')
(HERE / 'results.json').write_text(bak, encoding='utf-8'); (HERE / 'nav.parquet').write_bytes(navbak)
L, EVENTS, W, event_stats, FL = g['L'], g['EVENTS'], g['W'], g['event_stats'], g['FL']
sim, daily_stats, ENTS, CAND, ROC, T = g['sim'], g['daily_stats'], g['ENTS'], g['CAND'], g['ROC'], g['T']
b1, s1 = L.cost_rates(0.001)
out = {}
KS = ('select_2005-01_2019-04', 'valid_2019-05_2026-10', 'y2024-01_2025-12')
for nm, tr in EVENTS.items():
    f345 = np.array([np.isin(FL[r.e:r.x + 1, r.j], [3, 4, 5]).any() for r in tr.itertuples()])
    o = {}
    st = event_stats(tr[~f345], 0.001); o['ex_flag345_only'] = {k: (st[k]['mean_abn_pct'], st[k]['t_month_cluster'], st[k]['n_trades']) for k in KS}
    for cap in (1.0, 2.0):
        t2 = tr.copy(); t2['ret'] = t2.ret.clip(upper=cap)
        st = event_stats(t2, 0.001); o[f'ret_capped_at_{int(cap*100)}pct'] = {k: (st[k]['mean_abn_pct'], st[k]['t_month_cluster']) for k in KS}
    net = (1 + tr.ret) * (1 - s1) / (1 + b1) - 1; abn = net - tr.ew
    o['median_abn_pct'] = {k: round(float(abn[(tr.d >= pd.Timestamp(W[k][0][0])) & (tr.d <= pd.Timestamp(W[k][0][1]))].median() * 100), 2) for k in KS}
    o['share_abn_from_top5pct_trades'] = {}
    for k in KS:
        m = (tr.d >= pd.Timestamp(W[k][0][0])) & (tr.d <= pd.Timestamp(W[k][0][1])); a = abn[m].sort_values()
        o['share_abn_from_top5pct_trades'][k] = round(float(a.iloc[-max(1, len(a) // 20):].sum() / a.sum()), 2) if a.sum() > 0 else None
    out[nm] = o
# 優先順序隨機化（A、B，滑價 0.1%）
rnd = {}
for vn in ('A_noFilter', 'B_taiex200'):
    orig = CAND[vn]; res_ = {k: [] for k in KS}
    for seed in range(10):
        rng = np.random.default_rng(seed)
        CAND[vn] = [rng.permutation(c) if len(c) else c for c in orig]
        nav, *_ = sim(vn, b1, s1)
        for k in KS:
            st = daily_stats(nav, W[k]); res_[k].append((st['CAGR'], st['t_vs_EW_m'], st['t_vs_0050_m']))
    CAND[vn] = orig
    rnd[vn] = {k: {'CAGR_mean': round(float(np.mean([x[0] for x in v])), 4), 'CAGR_min': round(float(np.min([x[0] for x in v])), 4), 'CAGR_max': round(float(np.max([x[0] for x in v])), 4),
                   't_vs_EW_mean': round(float(np.mean([x[1] for x in v])), 2), 't_vs_EW_max': round(float(np.max([x[1] for x in v])), 2),
                   't_vs_0050_mean': round(float(np.mean([x[2] for x in v])), 2)} for k, v in res_.items()}
out['random_priority_order'] = rnd
print(json.dumps(out, ensure_ascii=False, indent=1))
d = json.loads((HERE / 'results.json').read_text(encoding='utf-8')); d['posthoc_diag_not_for_verdict'] = out
(HERE / 'results.json').write_text(json.dumps(d, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
