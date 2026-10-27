#!/usr/bin/env python3
"""主研究員補測（第 2 輪）：第三輪事前固定的擇時開關套在主策略長期樣本外。
規則（照 bt_more_methods.py，不改參數）：換股日收盤加權指數 < 200 日均線 → 下一期改持現金（另報改持 0050）。
策略淨值：revenue_longhistory/oos_nav.parquet（M2，滑價 0.1%，已含選股成本）。
切換成本：每次「出場」扣 0.4%（賣出含稅），每次「回場」扣 0.2%；改持 0050 時另計 ETF 買賣 0.0855%＋稅 0.1%。
換股日＝每月第一個 ≥ 11 日的交易日（與主策略相同）。月頻計算：最大回撤是月頻（日頻會更深），t＝對 0050 月超額。輸出 results.json。"""
import json
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent; A = H.parent
nav = pd.read_parquet(A / 'revenue_longhistory' / 'oos_nav.parquet')
tx = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date']).set_index('date')['close'].sort_index()
e50 = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()
idx = nav.index; tx = tx.reindex(tx.index.union(idx)).ffill(); ma = tx.rolling(200, min_periods=200).mean()
e50 = e50.reindex(idx).ffill()
reb = [d for d in pd.Series(idx, index=idx).groupby([idx.year, idx.month]).apply(lambda s: s[s.dt.day >= 11].iloc[0] if (s.dt.day >= 11).any() else None).dropna()]
reb = pd.DatetimeIndex(reb)

def run(col, idle):
    s = nav[col]; out = pd.Series(np.nan, index=reb[:-1]); bench = pd.Series(np.nan, index=reb[:-1]); state = []
    inn = True
    for a, b in zip(reb[:-1], reb[1:]):
        on = bool(tx.loc[a] >= ma.loc[a]) if np.isfinite(ma.loc[a]) else True
        r_s = s.loc[b] / s.loc[a] - 1; r_i = 0.0 if idle == 'cash' else e50.loc[b] / e50.loc[a] - 1
        c = 0.0
        if on and not inn: c += 0.002 + (0.00185 if idle == '0050' else 0)
        if not on and inn: c += 0.004 + (0.000855 if idle == '0050' else 0)
        out[a] = (r_s if on else r_i) - c; bench[a] = e50.loc[b] / e50.loc[a] - 1; inn = on; state.append(on)
    return out, bench, pd.Series(state, index=reb[:-1])

def stats(r, b, lo, hi):
    r, b = r[(r.index >= lo) & (r.index <= hi)], b[(b.index >= lo) & (b.index <= hi)]
    v = (1 + r).cumprod(); yrs = len(r) / 12
    ex = r - b
    return {'CAGR': round(float(v.iloc[-1] ** (1 / yrs) - 1), 4), 'maxDD_monthly': round(float((v / v.cummax() - 1).min()), 4),
            'bench_CAGR': round(float((1 + b).prod() ** (1 / yrs) - 1), 4), 't_vs_0050': round(float(ex.mean() / ex.std() * np.sqrt(len(ex))), 2), 'months': len(r)}
P = {'OOS_2006-04_2020-12': ('2006-04-01', '2020-12-31'), 'clean_2006-04_2013-01': ('2006-04-01', '2013-01-31'),
     '2021-03_2026-10': ('2021-03-01', '2026-10-31'), 'full_2006-04_2026-10': ('2006-04-01', '2026-10-31')}
BENCH = pd.Series([e50.loc[b] / e50.loc[a] - 1 for a, b in zip(reb[:-1], reb[1:])], index=reb[:-1])
res = {}
for col in ['M2_n30_slip0.001', 'M2_n10_slip0.001', 'M2_n30_slipstress']:
    for mode in ['no_switch', 'cash', '0050']:
        if mode == 'no_switch':
            s = nav[col]; r = pd.Series([s.loc[b] / s.loc[a] - 1 for a, b in zip(reb[:-1], reb[1:])], index=reb[:-1]); bch = BENCH; st = None
        else:
            r, bch, st = run(col, mode)
        res[f'{col}|{mode}'] = {k: stats(r, bch, *v) for k, v in P.items()}
        if st is not None: res[f'{col}|{mode}']['pct_months_out'] = round(float(1 - st.mean()), 3); res[f'{col}|{mode}']['switches'] = int((st != st.shift()).sum() - 1)
        yr = r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)
        res[f'{col}|{mode}']['yearly'] = {int(k): round(float(v), 3) for k, v in yr.items() if k in (2008, 2011, 2015, 2018, 2020, 2022)}
(H / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
for k, v in res.items():
    print(k, {p: (x['CAGR'], x['maxDD_monthly'], x['t_vs_0050']) for p, x in v.items() if p in P}, v.get('pct_months_out'), v['yearly'])
