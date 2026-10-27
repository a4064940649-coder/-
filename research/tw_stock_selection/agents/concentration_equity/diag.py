#!/usr/bin/env python3
"""事後診斷（不影響 PREREG 的判定）：
1) n30 名單內名次分層（1–10、11–20、21–30）；2) 每月從 n30 名單隨機抽 10 檔（200 次，零成本）看前 10 名落在分布哪裡；
3) 小 N 的報酬集中度：去掉該期貢獻最大的 5 筆持股後的毛額年化；4) 淨值濾網 E1 在挑選期去掉 2008 的表現。
寫入 results.json 的 POSTHOC_diagnostics。執行：python3 -I -B diag.py（需先跑 run.py）"""
import importlib.util, io, contextlib, json
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('run', H / 'run.py'); m = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(m)
L = m.L; RES = json.loads((H / 'results.json').read_text(encoding='utf-8'))
D = {'note': '事後診斷，不用於挑選或判定'}
W3 = {k: m.WIN[k] for k in ('P1', 'P2', 'P3', 'OOS_2006_2020')}
def cs(nav): return {w: (lambda o: {'CAGR': o['CAGR'], 'maxDD': o['maxDD'], 't_vs_EW_m': o.get('t_vs_EW_m')})(L.daily_stats(nav, s, m.DATES, None, m.ew)) for w, s in W3.items()}
ev30 = m.ev30
# 1) 名次分層
tr = {}
for a, b in ((0, 10), (10, 20), (20, 30)):
    ev = {t: p[a:b] for t, p in ev30.items()}
    g, _, _ = m.sim(ev, 'EW', 0.0, 0.0); c, _, _ = m.sim(ev, 'EW', m.BM, m.SM)
    tr[f'rank{a+1}-{b}'] = {'gross': cs(g), 'net': {w: v['CAGR'] for w, v in cs(c).items()}}
D['rank_tranches_within_n30'] = tr
# 2) 隨機 10 檔
rng = np.random.default_rng(20261027)
top = cs(m.sim({t: p[:10] for t, p in ev30.items()}, 'EW', 0.0, 0.0)[0])
draws = {w: [] for w in W3}
for _ in range(200):
    ev = {t: (np.sort(rng.choice(p, 10, replace=False)) if len(p) >= 10 else p) for t, p in ev30.items()}
    nav, _, _ = m.sim(ev, 'EW', 0.0, 0.0)
    for w, s in W3.items(): draws[w].append(L.daily_stats(nav, s, m.DATES)['CAGR'])
D['random10_of_n30_gross'] = {w: {'top10_CAGR': top[w]['CAGR'], 'random_median': round(float(np.median(v)), 4), 'random_p5': round(float(np.percentile(v, 5)), 4),
                                  'random_p95': round(float(np.percentile(v, 95)), 4), 'top10_percentile': round(float((np.array(v) < top[w]['CAGR']).mean()), 3)} for w, v in draws.items()}
# 3) 去掉貢獻最大的 5 筆持股
conc = {}
for n in (3, 5, 10, 30):
    _, _, book = m.sim(m.EVS[n], 'EW', 0.0, 0.0); cb = m.contributions(book)
    per = cb.groupby('ex')['contrib'].sum()
    o = {}
    for w, s in W3.items():
        c = cb[m.in_seg(pd.DatetimeIndex(cb['ex']), s)].copy(); p = per[m.in_seg(per.index, s)]
        yrs = (p.index[-1] - p.index[0]).days / 365.25 + 1 / 12
        full = float(np.prod(1 + p) ** (1 / yrs) - 1)
        top5 = c.nlargest(5, 'contrib'); p2 = p.copy()
        for _, r in top5.iterrows(): p2[r['ex']] -= r['contrib']
        o[w] = {'gross_CAGR_rebal_approx': round(full, 4), 'gross_CAGR_drop_top5_holdings': round(float(np.prod(1 + p2) ** (1 / yrs) - 1), 4),
                'top5': [f"{r['code']} {str(r['ex'].date())} {r['contrib']:+.3f}" for _, r in top5.iterrows()]}
    conc[f'EW{n}'] = o
D['return_concentration_drop_top5'] = conc
# 4) E1 去掉 2008
ex08 = {}
for n in (30, 10):
    for k in ('none', 'E1', 'S1'):
        nav = m.NAVS[f'EW{n}'] if k == 'none' else m.NAVS[f'n{n}_{k}']
        o = L.daily_stats(nav, [(m.first, '2007-12-31'), ('2009-01-01', '2013-12-31')], m.DATES)
        ex08[f'n{n}_{k}'] = {'CAGR': o['CAGR'], 'maxDD': o['maxDD'], 'Calmar': round(o['CAGR'] / abs(o['maxDD']), 3)}
D['P1_excluding_2008'] = ex08
RES['POSTHOC_diagnostics'] = D
(H / 'results.json').write_text(json.dumps(RES, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps(D, ensure_ascii=False, indent=1))
