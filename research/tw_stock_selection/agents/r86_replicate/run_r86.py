#!/usr/bin/env python3
"""R86：獨立重算 S86 八項數字。用法：python3 -I -B run_r86.py。輸出 r86_results.json、r86_yearly_dividends.csv、r86_payments_std.csv。
定義見各段註解；全部「拖累」＝（加了該項摩擦的 2019 起年化）−（沒加的 2019 起年化），單位百分點；2019 起＝2018-12-28 收盤 → 2026-10-22。"""
import sys, time, json, resource
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r86lib as L
T0 = time.time()
P, U, S, avA = L.load()
c = L.stlib.cagr
D0, DE = P.IDX[P.t0], P.IDX[-1]; D19 = P.IDX[P.IDX < pd.Timestamp('2019-01-01')][-1]
c19 = lambda nav: c(nav, D19, DE)
pp = lambda x: float(x * 100)
DV = L.dividends(P, 'est'); DVz = L.dividends(P, 'zero'); DVa = L.dividends(P, 'all')
SG = {s: L.s10lib.make_signal_s10(P, U, dict(S=S, av=avA), shift=s) for s in range(4)}
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
PATHS = [(cap, s) for s in range(4) for cap in CAPS]
R = dict(meta=dict(start=str(D0.date()), end=str(DE.date()), d19=str(D19.date()), flags=dict(EXRIGHT_FIX=True, DEADLINE_FIX=True, INS15_MASK=True, REDUCE_FIX=False, E50='old'),
                   div_info=DV['info']))

# ---- 0. 基準
nav0, lg0, st0 = L.s10lib.simulate_s10(P, SG[0], U, sat_w=0.7, sizing='W10', lock=False, cost=L.COSTS['strict'], capital=1e6)
rec = []; mf = []
nav1, lg1, st1 = L.sim(P, SG[0], U, DV, rec=rec, mf=mf)
e50 = pd.Series(P.E50, index=P.IDX).loc[D0:]
b0 = L.bench(P, DV)
R['base'] = dict(since2019=c19(nav0), full=c(nav0, D0, DE), n_closed=int((lg0['why'] != 'open_end').sum()),
                 copy_vs_orig_max_abs=float(np.abs(nav0.to_numpy() - nav1.to_numpy()).max()),
                 e50_since2019=c19(e50), bench_vs_e50_max_rel=float(np.abs(b0.to_numpy() / (e50.to_numpy() / e50.iloc[0] * 1e6) - 1).max()))
base = {k: c19(L.sim(P, SG[k[1]], U, DV, cap=k[0])[0]) for k in PATHS}
R['base']['paths20_mean'] = float(np.mean(list(base.values()))); R['base']['paths20_median'] = float(np.median(list(base.values())))
print('base', R['base'], round(time.time() - T0), flush=True)

# ---- 1. 現金股利占淨值（每年加總「當次股利 ÷ 前一日淨值」）
def div_table(rec, tag):
    d = pd.DataFrame(rec, columns=['t', 'j', 'D', 'navp', 'tx', 'pm'])
    d['date'] = P.IDX[d['t'].to_numpy()]; d['year'] = d['date'].dt.year; d['y'] = d['D'] / d['navp']
    d['code'] = [('0050' if j < 0 else str(P.COLS[j])) for j in d['j']]
    g = d.assign(stk=np.where(d['j'] >= 0, d['y'], 0.0), e50=np.where(d['j'] < 0, d['y'], 0.0)).groupby('year')[['y', 'stk', 'e50']].sum()
    g.columns = [f'{tag}_total', f'{tag}_stock', f'{tag}_0050']
    return d, g
pay, g_est = div_table(rec, 'strat')
recz = []; L.sim(P, SG[0], U, DVz, rec=recz); _, g_z = div_table(recz, 'zero')
reca = []; L.sim(P, SG[0], U, DVa, rec=reca); _, g_a = div_table(reca, 'all')
yb = pd.Series(DV['y50'], index=P.IDX).loc[D0:]; yb = yb[yb.index > D0]
g_b = yb.groupby(yb.index.year).sum().rename('e50_only')
# 個股部位的股利率＝個股股利 ÷ 個股平均權重
w = st1['stockfrac']; wy = w.groupby(w.index.year).mean().rename('avg_stock_weight')
yd = pd.concat([g_est, g_z[['zero_stock']], g_a[['all_stock']], g_b, wy], axis=1).fillna(0.0)
yd['stock_sleeve_yield'] = yd['strat_stock'] / yd['avg_stock_weight']
yd.index.name = 'year'; yd.to_csv(HERE / 'r86_yearly_dividends.csv')
pay.to_csv(HERE / 'r86_payments_std.csv', index=False)
m19 = yd.loc[2019:]
yrs = (DE - D19).days / 365.25
ev = DV['events']; held = pay[pay['j'] >= 0].merge(ev[['t', 'j', 'type', 'value', 'cash', 'prev_close']], on=['t', 'j'], how='left')
heldall = pd.DataFrame(reca, columns=['t', 'j', 'D', 'navp', 'tx', 'pm']); heldall = heldall[heldall['j'] >= 0].merge(ev[['t', 'j', 'type', 'value', 'cash']], on=['t', 'j'], how='left')
R['item1'] = dict(mean_calendar_years_2019_2026={k: pp(v) for k, v in m19.mean().items()},
                  per_elapsed_year={k: pp(m19[k].sum() / yrs) for k in ('strat_total', 'strat_stock', 'strat_0050', 'e50_only')},
                  n_payments_std=int(len(pay)), n_stock_payments=int((pay['j'] >= 0).sum()),
                  held_event_types_all_mode=heldall['type'].value_counts().to_dict(),
                  held_mixed=heldall[heldall['type'] == '權息'].assign(date=lambda d: P.IDX[d['t'].to_numpy()].astype(str), code=lambda d: P.COLS[d['j'].to_numpy()])[['date', 'code', 'value']].merge(
                      ev.assign(date=ev['date'].astype(str))[['date', 'code', 'cash']], on=['date', 'code'], how='left').to_dict('records'))
print('item1', R['item1']['mean_calendar_years_2019_2026'], R['item1']['per_elapsed_year'], flush=True)

# ---- 2. 股利稅（除息日直接從部位扣 tau × 股利）
TAUS = (0.05, 0.12, 0.20, 0.28)
it2 = dict(strat_std={}, strat_paths20={}, e50={})
for tau in TAUS:
    it2['strat_std'][tau] = pp(c19(L.sim(P, SG[0], U, DV, tau=tau)[0]) - base[(1e6, 0)])
    it2['strat_paths20'][tau] = pp(np.mean([c19(L.sim(P, SG[k[1]], U, DV, cap=k[0], tau=tau)[0]) - base[k] for k in PATHS]))
    it2['e50'][tau] = pp(c19(L.bench(P, DV, tau=tau)) - c19(b0))
R['item2'] = it2; print('item2', it2, flush=True)

# ---- 3. 補充保費（費率 2.11%、單筆股利滿 2 萬才扣；皆為假設）
PR, THR = 0.0211, 20000.0
def prem_fixed(y_event, year, size):
    amt = y_event * size
    p = np.where(amt >= THR, amt * PR, 0.0)
    s = pd.Series(p).groupby(np.asarray(year)).sum().reindex(range(2019, 2027)).fillna(0.0)
    return float(s.mean()), float(s.sum() / yrs)
it3 = dict(strat={}, e50={}, strat_paths20={})
recs20 = {}
for k in PATHS:
    r_ = []; L.sim(P, SG[k[1]], U, DV, cap=k[0], rec=r_); d = pd.DataFrame(r_, columns=['t', 'j', 'D', 'navp', 'tx', 'pm'])
    d['year'] = P.IDX[d['t'].to_numpy()].year; d['y'] = d['D'] / d['navp']; recs20[k] = d[d['year'] >= 2019]
p19 = pay[pay['year'] >= 2019]; b19 = yb[(yb > 0) & (yb.index.year >= 2019)]
for size in (1e6, 3e6, 1e7):
    it3['strat'][int(size)] = prem_fixed(p19['y'].to_numpy(), p19['year'].to_numpy(), size)
    it3['e50'][int(size)] = prem_fixed(b19.to_numpy(), b19.index.year, size)
    it3['strat_paths20'][int(size)] = float(np.mean([prem_fixed(d['y'].to_numpy(), d['year'].to_numpy(), size)[0] for d in recs20.values()]))
it3['div_share_paths20'] = {k2: pp(np.mean([d.loc[m_(d), 'y'].sum() for d in recs20.values()]) / 8) for k2, m_ in
                            (('total', lambda d: d['j'] > -9), ('stock', lambda d: d['j'] >= 0), ('e50', lambda d: d['j'] < 0))}
# 在回測帳戶裡照實際金額扣（帳戶會長大）的年化拖累
it3['drag_std'] = pp(c19(L.sim(P, SG[0], U, DV, prem=PR, thr=THR)[0]) - base[(1e6, 0)])
it3['drag_paths20'] = pp(np.mean([c19(L.sim(P, SG[k[1]], U, DV, cap=k[0], prem=PR, thr=THR)[0]) - base[k] for k in PATHS]))
it3['drag_e50'] = pp(c19(L.bench(P, DV, prem=PR, thr=THR)) - c19(b0))
R['item3'] = it3; print('item3', it3, flush=True)

# ---- 4. 股利延後 20 個交易日入帳
DLY = 20
R['item4'] = dict(strat_std=pp(c19(L.sim(P, SG[0], U, DV, delay=DLY)[0]) - base[(1e6, 0)]),
                  strat_paths20=pp(np.mean([c19(L.sim(P, SG[k[1]], U, DV, cap=k[0], delay=DLY)[0]) - base[k] for k in PATHS])),
                  e50=pp(c19(L.bench(P, DV, delay=DLY)) - c19(b0)))
print('item4', R['item4'], flush=True)

# ---- 5. 手續費 6 折／不打折（對 2.8 折）
R['item5'] = {}
for disc in (0.6, 1.0):
    R['item5'][disc] = dict(std=pp(c19(L.sim(P, SG[0], U, DV, disc=disc)[0]) - base[(1e6, 0)]),
                            paths20=pp(np.mean([c19(L.sim(P, SG[k[1]], U, DV, cap=k[0], disc=disc)[0]) - base[k] for k in PATHS])))
print('item5', R['item5'], flush=True)

# ---- 6. 20 元低消：新帳戶第一年（開戶日＝某年第一個交易日，之後 252 個交易日），多付的低消 ÷ 開戶本金
it6 = {}
T_full, t0_full = P.T, P.t0
starts = {y: int(np.searchsorted(P.IDX.values, np.datetime64(f'{y}-01-01'))) for y in range(2016, 2026)}
starts['2015-07'] = t0_full
for disc in (0.28, 0.6):
    for cap in (3e5, 5e5, 1e6):
        rows = {}
        for y, ts in starts.items():
            for s in range(4):
                P.t0 = ts; P.T = min(ts + 253, T_full)
                mf_ = []; n1 = L.sim(P, SG[s], U, DV, cap=cap, disc=disc, mf=mf_)[0]
                n2 = L.sim(P, SG[s], U, DV, cap=cap, disc=disc, minfee=0.0)[0]
                rows[(str(y), s)] = (sum(x for _, x in mf_) / cap * 100, len(mf_), (n2.iloc[-1] - n1.iloc[-1]) / cap * 100)
        P.t0, P.T = t0_full, T_full
        d = pd.DataFrame(rows, index=['excess_pct', 'n_minfee_orders', 'ret_diff_pct']).T
        d.index = pd.MultiIndex.from_tuples(d.index, names=['start', 'shift'])
        by = d.groupby('start').mean()
        g19 = by.loc[[str(y) for y in range(2019, 2026)]].mean(); g21 = by.loc[[str(y) for y in range(2021, 2026)]].mean()
        it6[f'{disc}|{int(cap)}'] = dict(mean_start_2019_2025=g19.to_dict(), mean_start_2021_2025=g21.to_dict(), start_2015_07=by.loc['2015-07'].to_dict(),
                                         by_start_excess_pct=by['excess_pct'].round(4).to_dict())
# 另一種看法：完整回測（2015-07 開戶）2019 起每年被低消多扣的錢 ÷ 當時淨值
for cap in (3e5, 5e5, 1e6):
    mf_ = []; n1 = L.sim(P, SG[0], U, DV, cap=cap, mf=mf_)[0]
    m = pd.DataFrame(mf_, columns=['t', 'x']); m['navp'] = n1.to_numpy()[m['t'].to_numpy() - P.t0 - 1]; m['year'] = P.IDX[m['t'].to_numpy()].year
    it6[f'fullrun|{int(cap)}'] = dict(pct_nav_per_year_2019on=float((m['x'] / m['navp'])[m['year'] >= 2019].sum() / yrs * 100),
                                      first252_pct_of_cap=float(m.loc[m['t'] <= P.t0 + 252, 'x'].sum() / cap * 100),
                                      cagr_diff_vs_nominfee=pp(c19(n1) - c19(L.sim(P, SG[0], U, DV, cap=cap, minfee=0.0)[0])))
R['item6'] = it6; print('item6', json.dumps(it6, ensure_ascii=False)[:1500], flush=True)

# ---- 7. 零股價差：每筆個股成交每邊再加 0.3%／0.6% 滑價
it7 = {}
for cap in (5e5, 1e6):
    b_ = [c19(L.sim(P, SG[s], U, DV, cap=cap)[0]) for s in range(4)]
    for add in (0.003, 0.006):
        cost = dict(cmult=1.0, xslip=0.003 + add, oddx=0.0)
        v_ = [c19(L.sim(P, SG[s], U, DV, cap=cap, cost=cost)[0]) for s in range(4)]
        it7[f'{int(cap)}|+{add}'] = dict(mean4=pp(np.mean(v_) - np.mean(b_)), shift0=pp(v_[0] - b_[0]), by_shift=[pp(a - b) for a, b in zip(v_, b_)], level_mean4=pp(np.mean(v_)))
    it7[f'{int(cap)}|base_mean4'] = pp(np.mean(b_))
R['item7'] = it7; print('item7', it7, flush=True)

# ---- 8. 三情境合計（標準路徑＝位移 0）。後＝稅＋補充保費（照帳戶實際金額）＋股利延後 20 日＋手續費折數
SC = {'小資': dict(cap=5e5, tau=0.05, disc=0.6), '一般': dict(cap=1e6, tau=0.12, disc=0.28), '高所得': dict(cap=5e6, tau=0.28, disc=0.28)}
it8 = {}
for name, a in SC.items():
    sb = c19(L.sim(P, SG[0], U, DV, cap=a['cap'])[0])
    full = dict(tau=a['tau'], prem=PR, thr=THR, delay=DLY)
    sa = c19(L.sim(P, SG[0], U, DV, cap=a['cap'], disc=a['disc'], **full)[0])
    sa2 = c19(L.sim(P, SG[0], U, DV, cap=a['cap'], disc=a['disc'], tau=a['tau'])[0])
    bb = c19(L.bench(P, DV, cap=a['cap'])); ba = c19(L.bench(P, DV, cap=a['cap'], **full)); ba2 = c19(L.bench(P, DV, cap=a['cap'], tau=a['tau']))
    it8[name] = dict(strat_before=pp(sb), strat_after=pp(sa), e50_before=pp(bb), e50_after=pp(ba), lead_before=pp(sb - bb), lead_after=pp(sa - ba),
                     taxfee_only=dict(strat_after=pp(sa2), e50_after=pp(ba2), lead_after=pp(sa2 - ba2)))
    if name == '小資':
        b4 = np.mean([c19(L.sim(P, SG[s], U, DV, cap=a['cap'])[0]) for s in range(4)])
        a4 = np.mean([c19(L.sim(P, SG[s], U, DV, cap=a['cap'], disc=a['disc'], **full)[0]) for s in range(4)])
        it8[name]['shift4_before'] = pp(b4); it8[name]['shift4_after'] = pp(a4)
        for add in (0.003, 0.006):
            cost = dict(cmult=1.0, xslip=0.003 + add, oddx=0.0)
            v = np.mean([c19(L.sim(P, SG[s], U, DV, cap=a['cap'], disc=a['disc'], cost=cost, **full)[0]) for s in range(4)])
            it8[name][f'oddslip+{add}'] = dict(strat=pp(v), lead=pp(v - ba))
R['item8'] = it8; print('item8', json.dumps(it8, ensure_ascii=False), flush=True)
R['meta']['secs'] = round(time.time() - T0); R['meta']['peak_mb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
json.dump(R, open(HERE / 'r86_results.json', 'w'), ensure_ascii=False, indent=1, default=str)
print('done', R['meta']['secs'], 's', R['meta']['peak_mb'], 'MB')
