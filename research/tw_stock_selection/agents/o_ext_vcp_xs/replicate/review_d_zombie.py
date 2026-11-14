# -*- coding: utf-8 -*-
"""審查（研究員沒講的問題）：下市股部位永遠不出場（殭屍部位）對各規格與 81 組鄰域的影響。
修正＝下市股在面板最後一根 K 棒收盤強制賣出。python3 -I -B review_d_zombie.py"""
import sys, os, json, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import rep_lib as L
HERE = os.path.dirname(os.path.abspath(__file__)) + '/'
D = L.Data()
sig = L.signals(D); sigA = L.signals(D, mode='adj')
out = {}
def one(s, ex, idle, fix, **kw):
    r = L.run_portfolio(D, s, ex, idle, delist_exit=fix, **kw)
    p = L.perf(D, r)
    z = [(o['code'], str(o['entry'])[:10], str(o['last_bar'])[:10]) for o in r['open_pos'] if o['last_bar'] < D.dates[-1]]
    return dict(full_ann=p['full']['ann'], full_ex=p['full']['ex'], full_t=p['full']['t'], y19_ann=p['2019']['ann'], y19_ex=p['2019']['ex'], y19_t=p['2019']['t'],
                inv=p['invested'], n_trades=p['n_trades'], zombies=z, full_cnt=p['stats']['full'], mdd=p['mdd'])
for sm, s in (('raw', sig), ('adj', sigA)):
    for ex in ('E1', 'E2', 'E3', 'E4', 'E5', 'E6'):
        for idle in ('cash', 'etf'):
            a = one(s, ex, idle, False); b = one(s, ex, idle, True)
            out['%s_%s_%s' % (sm, ex, idle)] = dict(spec=a, fixed=b)
            print(sm, ex, idle, 'full %+.2f (t %.2f) -> %+.2f (t %.2f) | 2019 %+.2f (t %.2f) -> %+.2f (t %.2f) | zombies %s' % (
                a['full_ex'] * 100, a['full_t'], b['full_ex'] * 100, b['full_t'], a['y19_ex'] * 100, a['y19_t'], b['y19_ex'] * 100, b['y19_t'], a['zombies']), flush=True)
b = one(sigA, 'E5', 'etf', True, stress=0.003); out['adj_E5_etf_fixed_stress'] = b
print('adj E5 etf fixed stress', {k: v for k, v in b.items() if k != 'zombies'})
grid = list(itertools.product((20, 25, 30), (8, 10, 12), (0.6, 0.7, 0.8), (1.3, 1.5, 2.0)))
rows = []
for sl, ml, vd, vb in grid:
    s = L.signals(D, dict(SegLen=sl, MaxLastDepth=ml, VolDryRatio=vd, VolBreak=vb))
    a = one(s, 'E3', 'etf', False); b = one(s, 'E3', 'etf', True); c = one(s, 'E3', 'cash', True)
    rows.append(dict(SegLen=sl, MaxLastDepth=ml, VolDryRatio=vd, VolBreak=vb, n_sig=len(s), n_zombie=len(a['zombies']), zombies=';'.join('%s@%s' % (z[0], z[1]) for z in a['zombies']),
                     etf_full_ex=a['full_ex'], etf_full_ex_fixed=b['full_ex'], etf_full_t_fixed=b['full_t'], etf_2019_ex=a['y19_ex'], etf_2019_ex_fixed=b['y19_ex'], etf_2019_t_fixed=b['y19_t'],
                     cash_full_ex_fixed=c['full_ex'], cash_2019_ex_fixed=c['y19_ex']))
    print(len(rows), flush=True)
g = pd.DataFrame(rows); g.to_csv(HERE + 'my_grid81_delistfix.csv', index=False)
def q(x): return dict(min=float(x.min()), p10=float(x.quantile(.1)), med=float(x.median()), p90=float(x.quantile(.9)), max=float(x.max()), pos=float((x > 0).mean()))
out['grid81_fixed'] = {c: q(g[c]) for c in ('etf_full_ex', 'etf_full_ex_fixed', 'etf_full_t_fixed', 'etf_2019_ex', 'etf_2019_ex_fixed', 'etf_2019_t_fixed', 'cash_full_ex_fixed', 'cash_2019_ex_fixed', 'n_zombie')}
d = g[(g.SegLen == 25) & (g.MaxLastDepth == 10) & (g.VolDryRatio == 0.7) & (g.VolBreak == 1.5)].iloc[0]
out['grid81_fixed']['default_rank_spec'] = int((g.etf_full_ex < d.etf_full_ex).sum()); out['grid81_fixed']['default_rank_fixed'] = int((g.etf_full_ex_fixed < d.etf_full_ex_fixed).sum())
out['grid81_fixed']['default_fixed'] = float(d.etf_full_ex_fixed)
out['grid81_fixed']['corr_spec_ex_vs_nzombie'] = float(np.corrcoef(g.etf_full_ex, g.n_zombie)[0, 1])
out['grid81_fixed']['mean_ex_by_nzombie'] = {int(k): float(v) for k, v in g.groupby('n_zombie').etf_full_ex.mean().items()}
out['grid81_fixed']['count_by_nzombie'] = {int(k): int(v) for k, v in g.n_zombie.value_counts().sort_index().items()}
print(json.dumps(out['grid81_fixed'], indent=1))
json.dump(out, open(HERE + 'review_d_zombie.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
