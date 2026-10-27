#!/usr/bin/env python3
"""獨立檢查：(1) 分數與 np.polyfit 一致；(2) PRIMARY 目標權重分布與單檔最大權重；(3) 交易逐年勝率。執行：python3 -I check.py（需先跑 run.py）"""
import runpy, sys, io, contextlib, json
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
buf = io.StringIO()
# 重用 run.py 的設定（只到 setup 為止）：直接執行 run.py 的前半段
src = (HERE / 'run.py').read_text(encoding='utf-8').split("t_start = int(np.searchsorted")[0]
g = {'__file__': str(HERE / 'run.py'), '__name__': 'chk'}
with contextlib.redirect_stdout(buf): exec(compile(src, 'run_head', 'exec'), g)
G, SIGI, LOGA, SC, KS, ADJg, ATR20, COLS, POOL = (g[k] for k in ('G', 'SIGI', 'LOGA', 'SC', 'KS', 'ADJg', 'ATR20', 'COLS', 'POOL'))
out = {}
rng = np.random.default_rng(0); errs = []
for _ in range(300):
    k = rng.integers(10, len(SIGI)); j = rng.choice(np.flatnonzero(np.isfinite(SC['clenow'][k])))
    i = SIGI[k]; y = LOGA[i - 89:i + 1, j]; x = np.arange(90)
    b, a = np.polyfit(x, y, 1); r2 = np.corrcoef(x, y)[0, 1] ** 2
    errs.append(abs(((np.exp(b) ** 250 - 1) * r2) - SC['clenow'][k, j]))
out['score_vs_polyfit_max_abs_err'] = float(max(errs))
ok = KS['clenow'][0]
w = np.where(ok, 0.001 * ADJg[SIGI] / ATR20[SIGI], np.nan)
out['target_weight_pct_quantiles_eligible'] = {q: round(float(np.nanquantile(w, q) * 100), 2) for q in (0.05, 0.25, 0.5, 0.75, 0.95, 0.99)}
out['share_eligible_target_gt_10pct'] = round(float(np.nanmean(w > 0.10)), 3)
tr = pd.read_csv(HERE / 'trades_PRIMARY.csv.gz', dtype={'code': str})
tr['y'] = tr.d.str[:4]
out['trades_by_entry_year'] = tr.groupby('y').agg(n=('ret', 'size'), win=('ret', lambda s: round(float((s > 0).mean()), 2)),
                                                   mean_ret_pct=('ret', lambda s: round(float(s.mean() * 100), 1)),
                                                   med_hold=('hold', 'median')).to_dict('index')
print(json.dumps(out, ensure_ascii=False, indent=1))
(HERE / 'check.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
