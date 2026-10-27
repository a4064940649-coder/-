#!/usr/bin/env python3
"""(i) t 值：同一淨值用三種月窗（訊號日收盤窗、日曆月底窗、日頻）比較；並以 scipy 外的純公式驗算。"""
import sys, json, math; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, ia_core as K
nv = pd.read_parquet(K.HERE / 'navs.parquet')
out = {}
for key, a, b, Y, st in (('P1_long', '2006-04-01', '2020-12-31', K.yoy_from_long(), '2005-01-01'), ('P2_rev5y', '2021-03-01', '2026-10-31', K.yoy_from_rev5y(), '2021-01-01')):
    S = K.Strategy(Y, st); sd = S.sig_days + [S.end_day]
    for n in (30, 10):
        nav = nv[f'{key}|n{n}|0.001'].dropna(); e50 = K.E50.reindex(nav.index)
        o = K.stats(nav, a, b, sd, K.E50)
        # 日曆月
        me = nav.groupby(nav.index.to_period('M')).last(); be = e50.groupby(e50.index.to_period('M')).last()
        mr = me.pct_change(); br = be.pct_change()
        sel = (mr.index >= pd.Period(a, 'M')) & (mr.index <= pd.Period(b, 'M'))
        x = (mr - br)[sel].dropna()
        # 日頻
        dr = (nav.pct_change() - e50.pct_change()); dr = dr[(dr.index > pd.Timestamp(a)) & (dr.index <= pd.Timestamp(b))].dropna()
        # 純公式
        D = pd.DatetimeIndex([d for d in sd if d in nav.index]); p = nav.reindex(D); q = e50.reindex(D)
        ex = ((p.shift(-1) / p) - (q.shift(-1) / q)).iloc[:-1]; ex = ex[(ex.index >= pd.Timestamp(a)) & (ex.index <= pd.Timestamp(b))].dropna().to_numpy()
        mu = sum(ex) / len(ex); sdv = math.sqrt(sum((v - mu) ** 2 for v in ex) / (len(ex) - 1)); tm = mu / (sdv / math.sqrt(len(ex)))
        # Newey-West（lag 1）在訊號窗
        e0 = ex - ex.mean(); g0 = (e0 @ e0) / len(e0); g1 = (e0[1:] @ e0[:-1]) / len(e0); nw = ex.mean() / math.sqrt((g0 + 2 * 0.5 * g1) / len(e0))
        out[f'{key}|n{n}'] = {'t_signal_window': round(o['t_0050'], 2), 't_manual': round(tm, 2), 'n_windows': len(ex), 't_calendar_month': round(K.tstat(x), 2), 'n_cal': len(x),
                              't_daily': round(K.tstat(dr), 2), 't_signal_NW1': round(nw, 2), 'autocorr1': round(float(np.corrcoef(ex[1:], ex[:-1])[0, 1]), 3)}
        print(key, n, out[f'{key}|n{n}'])
(K.HERE / 'results_tcheck.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
