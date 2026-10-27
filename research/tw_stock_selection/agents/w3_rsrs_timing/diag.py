#!/usr/bin/env python3
"""事後檢查（POSTHOC，不參與判定）：β 與 polyfit 一致、引擎永遠持有＝0050、主規格各成本、成本拖累、場外期間 0050 報酬。"""
import json, sys, runpy
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent
g = runpy.run_path(str(H / 'run.py'))
RJ = json.loads((H / 'results.json').read_text())
tx = g['tx']; out = {}
# 1) β／R² 與 np.polyfit 抽查
b, r2 = g['BR'][18]; rng = np.random.default_rng(1); errs = []
for i in rng.integers(18, len(tx), 200):
    lo = tx['low'].to_numpy()[i - 17:i + 1]; hi = tx['high'].to_numpy()[i - 17:i + 1]
    p = np.polyfit(lo, hi, 1); rr = np.corrcoef(lo, hi)[0, 1] ** 2
    errs.append(max(abs(p[0] - b.iloc[i]), abs(rr - r2.iloc[i])))
out['beta_r2_max_abs_err_vs_polyfit'] = float(max(errs))
# 2) 永遠持有＝0050 收對收
allin = np.ones(g['ND'], bool); r, held, tr = g['simulate'](allin, 'gross')
d = (r.iloc[2:] - g['R50D'].iloc[2:]).abs().max(); out['always_held_vs_0050_max_daily_diff'] = float(d)
# 3) 場外期間 0050 報酬（主規格）：場外時 0050 漲跌加總（以 0050 收對收，挑選/驗證）
held = g['MAIN_OBJ']['held']; r50 = g['R50D']
for w in ('select_price', 'valid_price'):
    m = g['win_mask'](held.index, w)
    out_r = r50[m & ~held.shift(1, fill_value=False).to_numpy()]
    in_r = r50[m & held.shift(1, fill_value=False).to_numpy()]
    out[f'{w}_0050_logret_when_out'] = round(float(np.log1p(out_r.fillna(0)).sum()), 3)
    out[f'{w}_0050_logret_when_in'] = round(float(np.log1p(in_r.fillna(0)).sum()), 3)
    out[f'{w}_0050_ann_when_out_pct'] = round(float(out_r.mean() * 252 * 100), 2)
    out[f'{w}_0050_ann_when_in_pct'] = round(float(in_r.mean() * 252 * 100), 2)
# 4) 進場後 0050 前 5 日／20 日報酬平均（訊號有無短期資訊）
C50 = g['C50']; ent = [e for e, _, _ in g['MAIN_OBJ']['trades']]; ext = [x for _, x, t in g['MAIN_OBJ']['trades'] if not t]
def fwd(ix, h): return float(np.nanmean([C50[min(i + h, len(C50) - 1)] / C50[i - 1] - 1 for i in ix]) * 100)
out['after_entry_fwd5_pct'] = round(fwd(ent, 5), 2); out['after_entry_fwd20_pct'] = round(fwd(ent, 20), 2)
out['after_exit_fwd5_pct'] = round(fwd(ext, 5), 2); out['after_exit_fwd20_pct'] = round(fwd(ext, 20), 2)
uc = np.nanmean([C50[i + 20] / C50[i - 1] - 1 for i in range(1, len(C50) - 20)]) * 100
out['uncond_fwd20_pct'] = round(float(uc), 2)
# 5) 平均持有天數、年切換數
tl = RJ['results']['main_trades']; out['main_n_trades_total'] = len(tl)
out['main_hold_median_days'] = float(np.median([t['hold_days'] for t in tl])); out['main_win_rate'] = round(float(np.mean([t['raw_pct'] > 0 for t in tl])), 3)
RJ['POSTHOC_diag'] = out
(H / 'results.json').write_text(json.dumps(RJ, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps(out, ensure_ascii=False, indent=1))
