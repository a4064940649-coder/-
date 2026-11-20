#!/usr/bin/env python3
"""S83 第 5 步（事後、僅描述）：合成溫度計「熱 − 冷」之後 1／3 個月超額的平均差，在驗證段用循環平移看是不是運氣；另報中位數、去掉 2020 的版本。
python3 -I -B posthoc2.py → results_posthoc2.json"""
import json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
T = pd.read_csv(HERE / 'thermometer_monthly.csv', index_col=0); mon = pd.PeriodIndex(T.index, freq='M')
OUT = {}
def diff(state, y, f=np.nanmean):
    h = y[(state == '熱') & np.isfinite(y)]; c = y[(state == '冷') & np.isfinite(y)]
    return f(h) - f(c) if len(h) and len(c) else np.nan
for nm, m in (('驗證 2016–2026', mon >= pd.Period('2016-01', 'M')), ('驗證去掉 2020', (mon >= pd.Period('2016-01', 'M')) & (mon.year != 2020)), ('訓練 2008–2015', mon <= pd.Period('2015-12', 'M')), ('2019–2026', mon >= pd.Period('2019-01', 'M'))):
    m = np.asarray(m); s = T.loc[m, 'state'].to_numpy(); r = {}
    for col, a in (('ex_month_pct', 12), ('ex_next_month_pct', 12), ('ex_next3_pct', 4)):
        y = T.loc[m, col].to_numpy(); d = diff(s, y); n = len(s)
        null = np.array([diff(np.roll(s, k), y) for k in range(12, n - 11)])
        r[col] = dict(hot_minus_cold_ann_pp=float(d * a), median_diff_ann_pp=float(diff(s, y, np.nanmedian) * a), p_shift_one_sided=float((1 + np.sum(null >= d)) / (1 + len(null))),
                      win_hot=float(np.nanmean(y[s == '熱'] > 0)), win_cold=float(np.nanmean(y[s == '冷'] > 0)), n_hot=int((s == '熱').sum()), n_cold=int((s == '冷').sum()))
    OUT[nm] = r
json.dump(OUT, open(HERE / 'results_posthoc2.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps(OUT, ensure_ascii=False, indent=1))
