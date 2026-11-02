#!/usr/bin/env python3
"""補充（只讀結果）：V7X 可執行（R4、R0）對主策略零衝擊 B10 j 版（navs_jscan_v7 AB_V7_B10|normal|J）的配對 t（訊號窗），13 個 j 中位數；
另報對同池等權（官方日 V7_B10|EW，只能做官方日窗）的年化差。"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import x_data as D, x_stats as S

JS = pd.read_parquet(D.O7 / 'navs_jscan_v7.parquet')
out = {}
for run in ('R0', 'R4', 'R4c'):
    f = HERE / 'out' / f'navs_oos_{run}.parquet'
    if not f.exists(): continue
    df = pd.read_parquet(f)
    for m in ('K5T', 'TICK'):
        for t in ('30萬', '100萬', '300萬', '1000萬'):
            for v in ('FIX', 'GROW'):
                k = f'{m}|normal|{t}|{v}'
                ts, ds, ts19 = [], [], []
                for j in range(13):
                    a = df[f'B10|J{j}|{k}']; b = JS[f'AB_V7_B10|normal|J{j}']
                    e = S.end_day('AB_V7_B10', j); sd = S.SIGD[('AB_V7_B10', j)]
                    sd = sd[sd < e].append(pd.DatetimeIndex([e]))
                    x = (a.reindex(sd).pct_change() - b.reindex(sd).pct_change()).iloc[1:]
                    x.index = sd[:-1]
                    ts.append(S.tstat(x)); ts19.append(S.tstat(x[x.index >= '2019-01-01'])); ds.append(float(x.mean() * 100))
                out[f'{run}|{k}'] = dict(t_vs_main_med=float(np.median(ts)), mean_pp_month_med=float(np.median(ds)), t_vs_main_Y19on_med=float(np.median(ts19)))
(HERE / 'out' / 'vs_main.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
for k, v in out.items(): print(k, {a: round(b, 2) for a, b in v.items()})
