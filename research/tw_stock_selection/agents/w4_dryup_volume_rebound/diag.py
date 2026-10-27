#!/usr/bin/env python3
"""事後診斷（非判定用）：(a) 不可成交的 t 收盤進場、純持有 10 日（無停損）扣成本前對等權；(b) 隨機抽 5 筆事件逐條核對條件；(c) 依訊號後隔夜跳空分組。"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as R  # noqa
from btlib import tstat
S = R.signals(0.5)
ew = R.ew_cc.to_numpy(); LCC = np.concatenate([[0.0], np.cumsum(np.log1p(ew))])
rows = []
for t, j in zip(*np.where(S)):
    if t + 10 >= R.T: continue
    a0, a1 = R.ACn[t, j], R.ACn[t + 10, j]
    ewr = np.expm1(LCC[t + 11] - LCC[t + 1])
    rows.append((R.IDX[t], a1 / a0 - 1 - ewr, R.ACn[t + 1, j] / a0 - 1 - ew[t + 1]))
d = pd.DataFrame(rows, columns=['sig', 'ab10_close', 'ab_day1_close'])
out = {}
for w, (a, b) in R.WIN.items():
    x = d[(d.sig >= a) & (d.sig <= b)]; g = x.groupby(x.sig.dt.to_period('M'))
    out[w] = {'n': len(x), 'tclose_H10_gross_ab_pct': round(x.ab10_close.mean() * 100, 3), 't_month': tstat(g.ab10_close.mean()),
              'day_t+1_close_ab_pct': round(x.ab_day1_close.mean() * 100, 3), 't_month_day1': tstat(g.ab_day1_close.mean())}
ev = pd.read_csv(HERE / 'events_K50_H10.csv.gz', parse_dates=['sig', 'entry'], dtype={'code': str})
ev['gapq'] = pd.qcut(ev['gap'], 4, labels=['Q1低', 'Q2', 'Q3', 'Q4高'])
out['by_overnight_gap_quartile'] = {str(k): {'n': int(len(g)), 'gap_pct': round(g.gap.mean() * 100, 2), 'ab_gross_pct': round(g.ab_gross.mean() * 100, 3)} for k, g in ev.groupby('gapq', observed=True)}
# spot check
rng = np.random.default_rng(0); chk = []
for i in rng.choice(len(ev), 5, replace=False):
    r = ev.iloc[i]; j = R.cols.index(r.code); t = R.IDX.get_loc(r.sig)
    vol = R.VOL.iloc[t - 10:t + 1, j]; vma20 = R.VMA20.iloc[t - 10:t + 1, j]
    chk.append({'code': r.code, 'sig': str(r.sig.date()), 'ret_t_pct': round((R.ADJn[t, j] / R.ADJn[t - 1, j] - 1) * 100, 2),
                'red': bool(R.CR.iat[t, j] > R.OP.iat[t, j]), 'vol_t/vma5_prev': round(R.VOL.iat[t, j] / R.VMA5.iat[t - 1, j], 2),
                'min_vol/vma20_prev10': round(float((vol.iloc[:-1] / vma20.iloc[:-1]).min()), 2),
                'adj/low60': round(R.ADJn[t, j] / R.LOW60.iat[t, j], 3), 'ma20_5d_chg_pct': round((R.MA20.iat[t, j] / R.MA20.iat[t - 5, j] - 1) * 100, 2),
                'hold': int(r.hold), 'why': r.why, 'raw_pct': round(r.raw * 100, 2)})
out['spot_check'] = chk
print(json.dumps(out, ensure_ascii=False, indent=1))
f = HERE / 'results.json'; D = json.loads(f.read_text(encoding='utf-8')); D['diagnostics_posthoc'] = out
f.write_text(json.dumps(D, ensure_ascii=False, indent=1), encoding='utf-8')
