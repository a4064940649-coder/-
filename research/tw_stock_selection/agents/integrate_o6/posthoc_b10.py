#!/usr/bin/env python3
"""integrate_o6 事後補充（非事前登記，不參與任何判定）：o5 教訓 2「穩健性要跑在 SOP 引用的口徑上」——
PREREG 只對 AB 基底報端點與逐年；本檔對 B10 基底（SOP 保守口徑）補報端點 t 與輸 0050 年數。只讀 navs.parquet，寫入 results.json 的 POSTHOC_B10 鍵。
不 import 引擎：窗＝plan_V6 訊號日收盤 → 下一訊號日收盤（與 v4lib.win_ret 相同），0050 取 data/etf_0050.csv。"""
import json, math
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
RJ = HERE / 'results.json'
RES = json.loads(RJ.read_text(encoding='utf-8'))
assert 'RESULTS' in RES and 'POSTHOC_B10' not in RES
NAV = pd.read_parquet(HERE / 'navs.parquet')
E = pd.read_csv(HERE / 'data/etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()
sd = pd.DatetimeIndex(list(pd.to_datetime(pd.read_csv(HERE / 'data/plan_V6.csv')['signal'])) + [pd.Timestamp('2026-10-12')])
def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; s = x.std(ddof=1); return float(x.mean() / s * math.sqrt(len(x)))
def win(nav):
    v = nav.reindex(nav.index.union(sd)).ffill().reindex(sd).to_numpy(); return pd.Series(v[1:] / v[:-1] - 1, index=sd[:-1])
E6 = E.reindex(E.index.union(NAV.index)).ffill().reindex(NAV.index)
out = {'note': '事後補充，非事前登記；不改任何判定'}
for nm in ('AB_V6', 'M_E50_20', 'M_E50_30', 'M_E50_50', 'M_2330_30'):
    for c in ('normal', 'stress'):
        nav = NAV[f'{nm}|B10|{c}']; x = (win(nav) - win(E6)).dropna(); x = x[x.index >= '2006-04-01']
        xs_ = x.sort_values(ascending=False); o = {'t_all': tstat(x)}
        for k in (1, 3, 5, 10): o[f'drop_best{k}'] = tstat(x.drop(xs_.index[:k]))
        o['start_2008'] = tstat(x[x.index >= '2008-01-01']); o['start_2014'] = tstat(x[x.index >= '2014-01-01']); o['to_2025_end'] = tstat(x[x.index <= '2025-12-31'])
        r = nav.pct_change(); r5 = E6.pct_change(); yrs = {}
        for y, g in r[r.index > nav.index[0]].groupby(r[r.index > nav.index[0]].index.year):
            yrs[int(y)] = (float((1 + g.fillna(0)).prod() - 1), float((1 + r5[g.index].fillna(0)).prod() - 1))
        o['years_lose_0050'] = int(sum(1 for a, b in yrs.values() if a < b))
        out[f'{nm}|B10|{c}'] = o
RES['POSTHOC_B10'] = out; RES['POSTHOC_B10_written_at'] = pd.Timestamp.now(tz='UTC').isoformat()
RJ.write_text(json.dumps(RES, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(out, ensure_ascii=False, indent=0))
