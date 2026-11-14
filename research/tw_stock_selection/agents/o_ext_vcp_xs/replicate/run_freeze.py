# -*- coding: utf-8 -*-
"""獨立複製：產生 my_numbers.json（凍結用）。python3 -I -B run_freeze.py
此檔執行時不讀研究員任何輸出檔。"""
import sys, os, json, time, hashlib, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import rep_lib as L

HERE = os.path.dirname(os.path.abspath(__file__)) + '/'
SCR = L.S + 'rep_vcp/'
os.makedirs(SCR, exist_ok=True)
t0 = time.time()
D = L.Data()
out = {'meta': {'note': '獨立複製員自寫；月 t 含最後不足月（2026-10），2024-01 因面板缺口併入 2024-02',
                'panel_dates': [str(D.dates[0])[:10], str(D.dates[-1])[:10]], 'T': int(D.T), 'N': int(D.N)}}

# ---------- 1. 訊號
sig, fun, single = L.signals(D, want_funnel=True)
out['signals'] = dict(n=int(len(sig)), n_codes=int(sig.code.nunique()), first=str(sig.date.min())[:10], last=str(sig.date.max())[:10],
                      by_year={int(k): int(v) for k, v in sig.date.dt.year.value_counts().sort_index().items()},
                      funnel=fun, single_pass={k: round(v, 5) for k, v in single.items()},
                      per_code_counts={int(k): int(v) for k, v in sig.code.value_counts().value_counts().sort_index().items()})
sig.drop(columns=['j', 'di']).to_csv(HERE + 'my_signals.csv', index=False)
sig660 = L.signals(D, min_bars=660)
out['signals']['n_min660'] = int(len(sig660))
sigA, funA, _ = L.signals(D, mode='adj', want_funnel=True)
ka = set(zip(sig.code, sig.date)); kb = set(zip(sigA.code, sigA.date))
out['signals_adj'] = dict(n=int(len(sigA)), both=len(ka & kb), raw_only=len(ka - kb), adj_only=len(kb - ka),
                          jaccard=len(ka & kb) / len(ka | kb), funnel=funA)
sigA.drop(columns=['j', 'di']).to_csv(HERE + 'my_signals_adj.csv', index=False)

# ---------- 2. 事件研究
def evblock(s):
    tr = L.tradable(D, s)
    ev = L.event_study(D, s)
    y = ev.date.dt.year
    blk = dict(n_sig=int(len(s)), lim_up=int(tr.lim_up.sum()), lock_up=int(tr.lock_up.sum()), no_open=int(tr.no_open.sum()),
               disp=int(tr.disp.sum()), tradable=int(tr.ok.sum()))
    for nm, m in (('all', None), ('sel_2006_2014', (y >= 2006) & (y <= 2014)), ('val_2015_2020', (y >= 2015) & (y <= 2020)),
                  ('hold_2021_2026', y >= 2021), ('from2019', y >= 2019)):
        for h in (5, 10, 20, 60):
            blk['%s_h%d' % (nm, h)] = L.ev_summary(ev, h, m)
    return blk, ev
out['event'], ev = evblock(sig)
ev.to_csv(HERE + 'my_events.csv', index=False)
out['event_adj'], _ = evblock(sigA)
out['ablation_event'] = {}
abl_sigs = {}
for ab in ('trend', 'contract', 'voldry', 'breakvol', 'rs', 'minvol'):
    s = L.signals(D, ablate=(ab,))
    abl_sigs[ab] = s
    b, _ = evblock(s)
    out['ablation_event'][ab] = {k: b[k] for k in ('n_sig', 'tradable', 'all_h20', 'all_h60')}
print('event done', time.time() - t0, flush=True)

# ---------- 3. 組合
out['portfolio'] = {}
trs = []
for ex in ('E1', 'E2', 'E3', 'E4', 'E5', 'E6'):
    for idle in ('cash', 'etf'):
        r = L.run_portfolio(D, sig, ex, idle)
        p = L.perf(D, r)
        tr = r['trades']
        p['trade_mean'] = float(tr.ret.mean()); p['trade_median'] = float(tr.ret.median()); p['trade_win'] = float((tr.ret > 0).mean())
        p['reasons'] = {k: int(v) for k, v in tr.reason.value_counts().items()}
        if idle == 'cash':
            pnl = (tr.proceeds - tr.cost)
            p['top10_share'] = float(pnl.sort_values(ascending=False).head(10).sum() / pnl.sum())
        out['portfolio']['%s_%s' % (ex, idle)] = p
        tr = tr.copy(); tr['rule'] = ex; tr['idle'] = idle; trs.append(tr)
        if ex == 'E3':
            r['nav'].to_csv(HERE + 'my_nav_E3_%s.csv' % idle, header=['nav'])
pd.concat(trs).to_csv(HERE + 'my_trades.csv.gz', index=False)
out['portfolio_adj'] = {}
for ex in ('E3', 'E5', 'E6'):
    for idle in ('cash', 'etf'):
        out['portfolio_adj']['%s_%s' % (ex, idle)] = L.perf(D, L.run_portfolio(D, sigA, ex, idle))
out['ablation_portfolio'] = {}
for ab, s in abl_sigs.items():
    for idle in ('cash', 'etf'):
        out['ablation_portfolio']['%s_%s' % (ab, idle)] = L.perf(D, L.run_portfolio(D, s, 'E3', idle))
print('portfolio done', time.time() - t0, flush=True)

# ---------- 4. 參數鄰域 81 組（E3）
grid = list(itertools.product((20, 25, 30), (8, 10, 12), (0.6, 0.7, 0.8), (1.3, 1.5, 2.0)))
rows = []
for sl, ml, vd, vb in grid:
    s = L.signals(D, dict(SegLen=sl, MaxLastDepth=ml, VolDryRatio=vd, VolBreak=vb))
    rc = L.perf(D, L.run_portfolio(D, s, 'E3', 'cash'))
    re = L.perf(D, L.run_portfolio(D, s, 'E3', 'etf'))
    rows.append(dict(SegLen=sl, MaxLastDepth=ml, VolDryRatio=vd, VolBreak=vb, n_sig=len(s),
                     cash_full_ex=rc['full']['ex'], cash_full_t=rc['full']['t'], cash_2019_ex=rc['2019']['ex'], cash_inv=rc['invested'],
                     etf_full_ex=re['full']['ex'], etf_full_t=re['full']['t'], etf_2019_ex=re['2019']['ex'], etf_2019_t=re['2019']['t'],
                     etf_inv=re['invested'], etf_full_ann=re['full']['ann'], etf_2019_ann=re['2019']['ann']))
    print(len(rows), time.time() - t0, flush=True)
g = pd.DataFrame(rows)
g.to_csv(HERE + 'my_grid81.csv', index=False)
def q(x): return dict(min=float(x.min()), p10=float(x.quantile(.1)), med=float(x.median()), p90=float(x.quantile(.9)), max=float(x.max()), pos=float((x > 0).mean()))
out['grid81'] = {c: q(g[c]) for c in ('n_sig', 'cash_full_ex', 'cash_full_t', 'cash_2019_ex', 'etf_full_ex', 'etf_full_t', 'etf_2019_ex', 'etf_2019_t')}
dflt = g[(g.SegLen == 25) & (g.MaxLastDepth == 10) & (g.VolDryRatio == 0.7) & (g.VolBreak == 1.5)].iloc[0]
out['grid81']['default_rank_etf_full_ex'] = int((g.etf_full_ex < dflt.etf_full_ex).sum())   # 0＝最後一名
out['grid81']['n_pos_etf_full_ex'] = int((g.etf_full_ex > 0).sum())
one = {}
for nm, vals in (('SegLen', (20, 25, 30)), ('MaxLastDepth', (8, 10, 12)), ('VolDryRatio', (0.6, 0.7, 0.8)), ('VolBreak', (1.3, 1.5, 2.0))):
    for v in vals:
        m = np.ones(len(g), bool)
        for k, dv in (('SegLen', 25), ('MaxLastDepth', 10), ('VolDryRatio', 0.7), ('VolBreak', 1.5)):
            m &= (g[k] == (v if k == nm else dv)).values
        r = g[m].iloc[0]
        one['%s=%s' % (nm, v)] = dict(n_sig=int(r.n_sig), cash_full_ex=float(r.cash_full_ex), etf_full_ex=float(r.etf_full_ex),
                                      etf_2019_ex=float(r.etf_2019_ex), etf_inv=float(r.etf_inv))
out['grid_1d'] = one

js = json.dumps(out, ensure_ascii=False, indent=1, sort_keys=True, default=float)
open(HERE + 'my_numbers.json', 'w', encoding='utf-8').write(js)
h = hashlib.sha256(js.encode('utf-8')).hexdigest()
open(HERE + 'my_numbers.sha256', 'w').write('%s  my_numbers.json\nfrozen_at=%s\n' % (h, time.strftime('%Y-%m-%d %H:%M:%S')))
print('FROZEN sha256', h, 'elapsed', time.time() - t0)
