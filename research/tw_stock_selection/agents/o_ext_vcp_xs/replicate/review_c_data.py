# -*- coding: utf-8 -*-
"""審查 C：資料缺口影響。以官方日 K 重建替代面板後重跑訊號、事件研究、E3。python3 -I -B review_c_data.py"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import rep_lib as L, rep_alt as RA
HERE = os.path.dirname(os.path.abspath(__file__)) + '/'
D = L.Data()
sig0 = L.signals(D)
out = {}
def run(name, DD):
    s = L.signals(DD)
    o = dict(n=int(len(s)), by_year={int(k): int(v) for k, v in s.date.dt.year.value_counts().sort_index().items() if k >= 2015})
    a = set(zip(sig0.code, sig0.date)); b = set(zip(s.code, s.date))
    o['common'] = len(a & b); o['base_only'] = sorted([(c, str(d)[:10]) for c, d in a - b], key=lambda x: x[1]); o['alt_only'] = sorted([(c, str(d)[:10]) for c, d in b - a], key=lambda x: x[1])
    ev = L.event_study(DD, s)
    y = ev.date.dt.year
    for nm, m in (('all', None), ('from2019', y >= 2019)):
        for h in (20, 60):
            e = L.ev_summary(ev, h, m)
            o['ev_%s_h%d' % (nm, h)] = {k: e[k] for k in ('n', 'ex_etf', 't_etf', 'ex_pool', 't_pool')}
    for ex in ('E3', 'E5', 'E6'):
        for idle in ('cash', 'etf'):
            r = L.run_portfolio(DD, s, ex, idle)
            p = L.perf(DD, r)
            o['%s_%s' % (ex, idle)] = dict(full_ann=p['full']['ann'], full_ex=p['full']['ex'], full_t=p['full']['t'], y19_ann=p['2019']['ann'], y19_ex=p['2019']['ex'], y19_t=p['2019']['t'], etf19=p['2019']['etf'], trades=p['n_trades'], inv=p['invested'])
    if hasattr(DD, 'n_added_gap'):
        o['cells'] = dict(gap=DD.n_added_gap, after_last=DD.n_added_after_last, inside=DD.n_added_inside, removed=DD.n_removed, replaced=int(DD.replaced.sum()))
    out[name] = o
    print(name, o['n'], o['common'], len(o['base_only']), len(o['alt_only']), o.get('cells'))
    print('  base_only', o['base_only']); print('  alt_only', o['alt_only'])
    for k in ('E3_cash', 'E3_etf', 'E5_etf', 'E6_etf'): print('  ', k, {a: round(b, 4) for a, b in o[k].items()})
    for k in ('ev_all_h20', 'ev_from2019_h20', 'ev_from2019_h60'): print('  ', k, {a: round(b, 4) for a, b in o[k].items()})
    return s
run('base_panel', D)
A1 = RA.AltData(D, fill_gap=False, extend=False); s1 = run('official_ohlcv_nogapfill', A1); del A1
A2 = RA.AltData(D, fill_gap=True, extend=False); s2 = run('official_ohlcv_gapfilled', A2); del A2
A3 = RA.AltData(D, fill_gap=True, extend=True); s3 = run('official_ohlcv_gapfilled_extended', A3)
# 斷點股：面板最後一根在 2019-04～2019-05 者，官方接續後的訊號
last = pd.Series([D.dates[np.nonzero(D.has[:, j])[0][-1]] for j in range(D.N)], index=D.codes)
brk = last[(last >= '2019-04-01') & (last <= '2019-05-31')].index
ext = {c: int((~np.isnan(A3.C[:, list(D.codes).index(c)]) & (A3.dates > last[c])).sum()) for c in brk}
out['breakpoint'] = dict(n_codes=len(brk), n_with_official_after=int(sum(v > 0 for v in ext.values())), bars_added=int(sum(ext.values())),
                         signals_after_break=[(c, str(d)[:10]) for c, d in zip(s3.code, s3.date) if c in set(brk) and d > last[c]],
                         base_signals_from_these_codes=[(c, str(d)[:10]) for c, d in zip(sig0.code, sig0.date) if c in set(brk)])
print(out['breakpoint'])
json.dump(out, open(HERE + 'review_c_data.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
