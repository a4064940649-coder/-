#!/usr/bin/env python3
"""主策略延伸到上櫃：(a) 只上櫃（前 300）、(b) 上市＋上櫃合併（前 400）、對照原上市版（前 300）。
另報：營收腿排除金融保險業（exFIN）、寬鬆滾動（診斷）。輸出 results.json（保留 PREREG）、navs.parquet。"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tplib as T  # noqa: E402
L = T.L


def windows(dates):
    D = pd.DatetimeIndex(dates); one = pd.Timedelta(days=1)
    d13 = D[D >= '2013-02-01'][0]; d14 = D[D >= '2014-04-01'][0]
    return {
        'S1_OOS_2008-05_2020-12': [('2008-05-01', '2020-12-31')],
        'S1a_clean_2008-05_2013-01sig': [('2008-05-01', d13 - one)],
        'IFRS_2013-02_2014-03sig': [(d13, d14 - one)],
        'S1b_2014-04_2020-12': [(d14, '2020-12-31')],
        'S2_recent_strict': [('2021-01-01', '2022-12-31'), ('2024-11-01', '2025-03-31'), ('2026-10-01', '2026-10-31')],
        'S2a_2021-01_2022-12': [('2021-01-01', '2022-12-31')],
        'HOLE1_2023-01_2024-10': [('2023-01-01', '2024-10-31')],
        'HOLE2_2025-04_2026-09': [('2025-04-01', '2026-09-30')],
        'S2_lenient_windows': [('2021-01-01', '2022-12-31'), ('2023-11-01', '2025-03-31'), ('2025-11-01', '2026-10-31')],
        'ALL_2021-01_2026-10': [('2021-01-01', '2026-10-31')],
        'ALL_2008-05_2026-10': [('2008-05-01', '2026-10-31')],
        'TWSE_ref_2006-04_2020-12': [(D[0], '2020-12-31')],
    }


def main():
    out = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
    out = {'PREREG': out['PREREG']}
    S = L.Setup(L.yoy_long(), start='2005-01-01')
    dates = S.dates; DATES = S.DATES; W = windows(dates)
    t0 = L.pos(dates[0]); t1 = L.pos(dates[-1])
    out['dates'] = {'first_signal': str(dates[0].date()), 'last_signal': str(dates[-1].date()), 'n': len(dates)}
    tw = T.twse_panel(); tp = T.load_tpex(); cb = T.combine(tw, tp)
    g_tw = L.GOOD
    cnt = tp['adj'].notna().sum(axis=1); med = cnt.rolling(21, center=True, min_periods=5).median()
    g_tp = (cnt >= 0.85 * med) & g_tw
    out['excluded_days_tpex'] = [str(d.date()) for d in T.IDX[(~g_tp) & (T.IDX >= '2007-05-01')]]
    TPCOLS = set(tp['adj'].columns)
    TWCOLS = set(tw['adj'].columns)
    unis = {
        'TWSE_top300': T.Universe(tw, 300, 'TWSE', good=g_tw),
        'TPEX_top300': T.Universe(tp, 300, 'TPEX', good=g_tp),
        'COMB_top400': T.Universe(cb, 400, 'COMB', good=g_tp),
        'TPEX_top300_lenient': T.Universe(tp, 300, 'TPEX', lenient=True, good=g_tp),
        'COMB_top400_lenient': T.Universe(cb, 400, 'COMB', lenient=True, good=g_tp),
    }
    navs = {}; ref = {}
    order = ['TWSE_top300', 'TPEX_top300', 'COMB_top400', 'TPEX_top300_lenient', 'COMB_top400_lenient']
    for var in ('base', 'exFIN'):
        for un in order:
            U = unis[un]
            Y = T.yoy_long(U.COLS)
            if var == 'exFIN':
                fin = T.fin_codes(U.COLS); Y = Y.copy(); Y[[c for c in Y.columns if c in fin]] = np.nan
            U.setup(Y, dates)
            for n in (30, 10):
                ev, pools = U.selection(n)
                key0 = f'{var}|{un}'
                if n == 30:
                    ew, _ = U.sim(pools, 0.0, 0.0, t0, t1); navs[f'{key0}|EW'] = ew
                    # 股池大小與上櫃占比
                    ps = {}; tpsh = {}
                    for k, (_, idx) in pools.items():
                        y = T.IDX[k].year; ps.setdefault(y, []).append(len(idx))
                        tpsh.setdefault(y, []).append(np.mean([U.COLS[j] in TPCOLS and U.COLS[j] not in TWCOLS for j in idx]) if len(idx) else np.nan)
                    out.setdefault('pool', {})[key0] = {'size_by_year': {y: round(float(np.mean(v)), 1) for y, v in ps.items()},
                                                        'tpex_only_share_by_year': {y: round(float(np.nanmean(v)), 3) for y, v in tpsh.items()}}
                ew = navs[f'{key0}|EW']
                # 容量：入選股訊號日 60 日均成交值
                v60 = U.v60d.to_numpy(); cap = {}; tpp = {}
                for k, (_, pk) in ev.items():
                    d = U.sig[k]; i = U.DATES.get_loc(d); vv = v60[i, pk]
                    cap.setdefault(d.year, []).append((np.nanmedian(vv), np.nanpercentile(vv, 10), np.nanmin(vv)))
                    tpp.setdefault(d.year, []).append(np.mean([U.COLS[j] not in TWCOLS for j in pk]))
                capd = {y: {'median_v60_億': round(float(np.median([a[0] for a in v])) / 1e8, 2), 'p10_v60_億': round(float(np.median([a[1] for a in v])) / 1e8, 2),
                            'min_v60_億': round(float(np.median([a[2] for a in v])) / 1e8, 2)} for y, v in cap.items()}
                for slip in (0.001, 0.003, 'stress'):
                    b, s = T.cost_rates(slip)
                    nav, tov = U.sim(ev, b, s, t0, t1)
                    key = f'{key0}|n{n}|slip{slip}'; navs[key] = nav
                    rkey = f'{var}|TWSE_top300|n{n}|slip{slip}'
                    if un == 'TWSE_top300': ref[(var, n, slip)] = nav
                    rnav = ref[(var, n, slip)]
                    bnav = ref[('base', n, slip)] if var == 'exFIN' else None
                    rec = {'turnover': round(tov, 3), 'skipped_limit_up': int(U.nblk), 'rebalances': len(ev), 'windows': {}}
                    for wk, sg in W.items():
                        st = T.stats(nav, sg, DATES, ew, None if un == 'TWSE_top300' else rnav)
                        if st and bnav is not None and un != 'TWSE_top300':
                            st2 = T.stats(nav, sg, DATES, None, bnav)
                            st['paired_vs_TWSE_base_pct_m'] = st2['paired_vs_TWSE_pct_m']; st['t_paired_vs_TWSE_base_m'] = st2['t_paired_vs_TWSE_m']
                        if st and un == 'TWSE_top300' and var == 'exFIN':
                            st2 = T.stats(nav, sg, DATES, None, ref[('base', n, slip)])
                            st['paired_vs_TWSE_base_pct_m'] = st2['paired_vs_TWSE_pct_m']; st['t_paired_vs_TWSE_base_m'] = st2['t_paired_vs_TWSE_m']
                        if st and wk != 'ALL_2008-05_2026-10':
                            for kk in ('yearly_pct', 'EW_yearly_pct', '0050_yearly_pct'): st.pop(kk, None)
                        rec['windows'][wk] = st
                    if slip == 0.001:
                        rec['capacity_by_signal_year'] = capd
                        rec['tpex_share_of_picks_by_year'] = {y: round(float(np.mean(v)), 3) for y, v in tpp.items()}
                    out.setdefault('runs', {})[key] = rec
                    w1 = rec['windows'].get('S1_OOS_2008-05_2020-12') or {}; w2 = rec['windows'].get('S2_recent_strict') or {}
                    print(key, 'S1', w1.get('CAGR'), w1.get('maxDD'), w1.get('t_vs_0050_m'), w1.get('t_vs_EW_m'), w1.get('t_paired_vs_TWSE_m'),
                          '| S2', w2.get('CAGR'), w2.get('t_vs_EW_m'), w2.get('t_paired_vs_TWSE_m'), flush=True)
                    (HERE / 'results.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    out['0050_windows'] = {k: L.daily_stats(L.E50.iloc[t0:t1 + 1], s, DATES) for k, s in W.items()}
    for k in out['0050_windows']:
        if out['0050_windows'][k] and k != 'ALL_2008-05_2026-10': out['0050_windows'][k].pop('yearly_pct', None)
    (HERE / 'results.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    pd.DataFrame({k: v for k, v in navs.items() if ('slip0.001' in k or k.endswith('EW'))}).to_parquet(T.TPP / 'tpex_main_navs.parquet')
    pd.DataFrame({k: v for k, v in navs.items() if 'slip0.001' in k and '|n30|' in k and 'lenient' not in k}).to_parquet(HERE / 'navs_n30.parquet')
    print('done')


if __name__ == '__main__':
    main()
