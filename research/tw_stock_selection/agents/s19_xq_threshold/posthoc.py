#!/usr/bin/env python3
"""S19 事後分析（標「事後」，不影響判定）→ results_posthoc.json。
(1) 逐年：14 個主跑的年報酬；E0／XR／XU／XT 兩個資金格 20 條路徑的逐年配對差中位數（重跑路徑）；
(2) 門檻版每月持股檔數分布（逐年：滿 10 檔的月數、0 檔的月數）、候選檔數、季線上 SURGE ≥ 2.0 的檔數；
(3) 差最多的年份與原因：XT 沒買到的 XR 名單逐筆分類（SURGE < 2.5、不在腳本股池、滿 10 檔、其他），逐年筆數與下一個月超額；逐年損益貢獻差最大的股票；
(4) 動態門檻表 vs 當年實際的第 10／20 名中位數。python3 -I -B posthoc.py"""
import sys, json, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s19lib
from s19lib import s10lib, s4lib, stlib, COSTS

H = Path(__file__).resolve().parent
t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
RES = json.loads((H / 'results.json').read_text())
PRE = json.loads((H / 'PREREG.json').read_text())
DC = json.loads((H / 'data_check.json').read_text())
P = s19lib.load()
s19lib.e50_fix_suspension(P)
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
UX, Cadj, liqX = s19lib.script_universe(P, INNOV)
del Cadj, liqX, INNOV
S, _ = s19lib.rev_std(P)
months = list(S.index)
avA = s19lib.a10_rows(P, months); avB, _ = s19lib.b11_rows(P, months, 11)
RT_A = dict(S=S, av=avA); RT_B = dict(S=S, av=avB)
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'listed', 'NEARHI', 'V60', 'liq', 'has'):
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
bench = pd.Series(P.E50[P.t0:], index=P.IDX[P.t0:])
out = {}


def build(sid, s):
    if sid == 'E0':
        return s10lib.make_signal_s10(P, U, RT_A, shift=s)
    if sid == 'XR':
        return s10lib.make_signal_s10(P, U, RT_B, shift=s)
    if sid == 'XU':
        return s19lib.make_signal_thr(P, U, U, U, S, avB, 2.5, 2.0, shift=s)
    return s19lib.make_signal_thr(P, U, UX, None, S, avB, 2.5, 2.0, shift=s)


UNIV = {'E0': U, 'XR': U, 'XU': U, 'XT': UX}
CAPS = {'W10_70': (6e5, 8e5, 1e6, 1.5e6, 2e6), 'SAT': (4.2e5, 5.6e5, 7e5, 1.05e6, 1.4e6)}


def sim(sid, Sg, f, cap):
    if f == 'W10_70':
        return s10lib.simulate_s10(P, Sg, UNIV[sid], sat_w=0.7, sizing='W10', lock=False, cost=COSTS['strict'], capital=cap)
    return s10lib.simulate_s10(P, Sg, UNIV[sid], sat_w=1.0, sizing='W10', lock=False, cost=COSTS['strict'], capital=cap, sweep_min=float('inf'))


# (1) 逐年
yr_main = {k: {y: round(v['strat'] * 100, 2) for y, v in R['yearly'].items()} for k, R in RES['combos'].items()}
yr_main['0050'] = {y: round(v['e0050'] * 100, 2) for y, v in RES['combos']['E0|W10_70']['yearly'].items()}
out['yearly_main'] = yr_main
YP = {}
for s in range(4):
    for sid in ('E0', 'XR', 'XU', 'XT'):
        Sg = build(sid, s)
        for f in ('W10_70', 'SAT'):
            for cap in CAPS[f]:
                nav = sim(sid, Sg, f, cap)[0]
                YP[(sid, f, f'{int(cap/1e4)}萬|+{s}')] = {int(y): v['strat'] for y, v in stlib.yearly(nav, bench).items()}
        del Sg; gc.collect()
ypd = {}
for f in ('W10_70', 'SAT'):
    keys = [k[2] for k in YP if k[0] == 'E0' and k[1] == f]
    for a, b in (('XT', 'E0'), ('XR', 'E0'), ('XU', 'XR'), ('XT', 'XU')):
        d = {}
        for y in range(2016, 2027):
            v = np.array([YP[(a, f, k)][y] - YP[(b, f, k)][y] for k in keys]) * 100
            d[y] = dict(median=round(float(np.median(v)), 2), npos=int((v > 0).sum()))
        ypd[f'{a}-{b}|{f}'] = d
out['yearly_path_diff'] = ypd
rank_years = sorted(range(2019, 2027), key=lambda y: -abs(ypd['XT-E0|W10_70'][y]['median']))
out['years_largest_gap_W10_70'] = [(y, ypd['XT-E0|W10_70'][y]['median']) for y in rank_years]
print('逐年完成', f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)

# (2) 持股檔數分布、候選、季線上 SURGE ≥ 2.0 的檔數（位移 0）
hb = RES['combos']['XT|W10_70']['holdings_at_exec']['by_month']; he = RES['combos']['E0|W10_70']['holdings_at_exec']['by_month']
hs = RES['combos']['XT|SAT']['holdings_at_exec']['by_month']
L = RES['lists']
nth = s19lib.lma_nth(P, U, S, avB, ns=(10, 20))
ge2 = {}
for r in nth.itertuples():
    k = r.k; s = S.loc[pd.Period(r.month, 'M')].to_numpy()
    ge2[r.month] = int((np.isfinite(s) & (np.nan_to_num(s, nan=-9) >= 2.0) & P.F_MA60[k] & UX[k]).sum())
by_year = {}
for y in range(2016, 2027):
    ms = [m for m in hb if m.startswith(str(y))]
    if not ms:
        continue
    ic = [L['XT']['months'].index(m) for m in ms]
    by_year[y] = dict(n_months=len(ms), XT_W10_mean=round(float(np.mean([hb[m] for m in ms])), 2), XT_W10_full10=int(sum(hb[m] == 10 for m in ms)), XT_W10_zero=int(sum(hb[m] == 0 for m in ms)),
                      XT_SAT_mean=round(float(np.mean([hs[m] for m in ms if m in hs])), 2), E0_mean=round(float(np.mean([he[m] for m in he if m.startswith(str(y))])), 2),
                      XT_cand_mean=round(float(np.mean([L['XT']['n_cand'][i] for i in ic])), 2), XR_list_mean=round(float(np.mean([L['XR']['n_cand'][L['XR']['months'].index(m)] for m in ms if m in L['XR']['months']])), 2),
                      n_keep_ge2_mean=round(float(np.mean([ge2[m] for m in ms if m in ge2])), 1),
                      s10_med=round(float(nth[nth['month'].str.startswith(str(y))]['s10'].median()), 2), s20_med=round(float(nth[nth['month'].str.startswith(str(y))]['s20'].median()), 2))
out['holdings_by_year'] = by_year
hv = list(hb.values())
out['holdings_dist_XT_W10_70'] = {int(a): int(b) for a, b in zip(*np.unique(hv, return_counts=True))}
out['holdings_dist_XT_SAT'] = {int(a): int(b) for a, b in zip(*np.unique(list(hs.values()), return_counts=True))}
out['holdings_dist_E0_W10_70'] = {int(a): int(b) for a, b in zip(*np.unique(list(he.values()), return_counts=True))}

# (3) XT 沒買到的 XR 名單：原因分類（XT｜W10_70 主跑交易）
tr = pd.read_csv(H / 'trades.csv.gz', dtype={'code': str}, parse_dates=['sig_date', 'ent_date', 'exit_date'])
di = {d: i for i, d in enumerate(pd.DatetimeIndex(P.IDX))}
xt = tr[tr['run'] == 'XT|W10_70'].copy(); e0 = tr[tr['run'] == 'E0|W10_70'].copy(); xr = tr[tr['run'] == 'XR|W10_70'].copy()
for x in (xt, e0, xr):
    x['ent_t'] = x['ent_date'].map(di); x['exit_t'] = x['exit_date'].map(di); x['sig_t'] = x['sig_date'].map(di)
XRm = dict(zip(L['XR']['months'], L['XR']['sig_dates']))
XTt = L['XT']['targets']; XRt = L['XR']['targets']
krow = {m: di[pd.Timestamp(d)] for m, d in zip(L['XT']['months'], L['XT']['sig_dates'])}
rows_sorted = sorted(krow.values())
nxt = {k: (rows_sorted[i + 1] if i + 1 < len(rows_sorted) else None) for i, k in enumerate(rows_sorted)}


def fwd(k, j):
    k2 = nxt.get(k)
    if k2 is None or k2 + 1 >= P.T:
        return None
    r = P.AO[k2 + 1, j] / P.AO[k + 1, j] - 1; r0 = P.E50O[k2 + 1] / P.E50O[k + 1] - 1
    return float(r - r0) if np.isfinite(r) and np.isfinite(r0) else None


miss = []
for m, k in krow.items():
    t = k + 1
    held_before = set(xt[(xt['ent_t'] <= k) & ((xt['exit_t'] > k) | (xt['why'] == 'open_end'))]['code'])
    bought = set(xt[xt['sig_t'] == k]['code'])
    n_after = int(((xt['ent_t'] <= t) & ((xt['exit_t'] > t) | (xt['why'] == 'open_end'))).sum())
    s = S.loc[pd.Period(m, 'M')].to_numpy()
    for c in XRt.get(m, []):
        if c in held_before or c in bought:
            continue
        j = P.ci[c]
        if not UX[k, j]:
            why = '不在腳本股池（還原價／還原成交值）'
        elif not (np.isfinite(s[j]) and s[j] >= 2.5):
            why = 'SURGE < 2.5（研究版前 10 名但低於門檻）'
        elif n_after >= 10:
            why = '滿 10 檔、滿了不買'
        else:
            why = '其他（開盤漲停、整張成本超過目標、不到半個目標金額）'
        miss.append(dict(month=m, year=int(m[:4]) + (1 if m[5:] == '12' else 0), code=c, why=why, fwd=fwd(k, j), surge=float(s[j]),
                         xr_bought=bool(((xr['sig_t'] == k) & (xr['code'] == c)).any())))
md = pd.DataFrame(miss)


def agg(g):
    v = g['fwd'].dropna()
    return pd.Series(dict(n=len(g), n_xr_bought=int(g['xr_bought'].sum()), fwd_mean=round(float(v.mean()) * 100, 2) if len(v) else np.nan,
                          t=round(float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))), 2) if len(v) > 2 else np.nan))


out['miss_buy_by_reason'] = md.groupby('why').apply(agg).to_dict('index')
out['miss_buy_by_year_reason'] = {int(y): g.groupby('why').size().to_dict() for y, g in md.groupby('year')}
out['miss_buy_fwd_by_year'] = {int(y): round(float(g['fwd'].dropna().mean()) * 100, 2) for y, g in md.groupby('year')}
# 「多買」：XT 買、XR 名單沒有的，依原因（名次 11～20、21 名以後、不在研究股池）
extra = []
for r in xt.itertuples():
    m = [mm for mm, kk in krow.items() if kk == r.sig_t]
    if not m:
        continue
    m = m[0]
    if r.code in set(XRt.get(m, [])):
        continue
    k = int(r.sig_t); j = P.ci[r.code]; s = S.loc[pd.Period(m, 'M')].to_numpy()
    ok = U[k] & np.isfinite(s) & P.F_MA60[k]
    o = np.flatnonzero(ok)[np.argsort(-s[ok], kind='stable')]
    pos = {int(q): i + 1 for i, q in enumerate(o)}
    rk = pos.get(j)
    cat = '不在研究股池' if rk is None else ('研究名次 1～10（不應發生）' if rk <= 10 else ('研究名次 11～20' if rk <= 20 else '研究名次 21 以後'))
    extra.append(dict(month=m, code=r.code, cat=cat, rank=rk, ret=float(r.ret), fwd=fwd(k, j)))
ed = pd.DataFrame(extra)
out['extra_buy_by_cat'] = {c: dict(n=int(len(g)), trade_ret_mean=round(float(g['ret'].mean()) * 100, 2), trade_ret_median=round(float(g['ret'].median()) * 100, 2),
                                  fwd_mean=round(float(g['fwd'].dropna().mean()) * 100, 2)) for c, g in ed.groupby('cat')}
# 逐年損益貢獻差（XT − E0，W10_70 主跑；依出場年）
contrib = {}
for y in rank_years[:4]:
    a = xt[xt['exit_date'].dt.year == y].groupby('code')['pnl'].sum(); b = e0[e0['exit_date'].dt.year == y].groupby('code')['pnl'].sum()
    d = a.sub(b, fill_value=0).sort_values()
    contrib[y] = dict(XT_more=[(c, round(v)) for c, v in d.tail(5)[::-1].items()], E0_more=[(c, round(v)) for c, v in d.head(5).items()])
out['contrib_diff_top_years'] = contrib
# 2024 起：E0 的最賺 10 筆，XT 有沒有買到
big = e0[e0['ent_date'] >= '2024-01-01'].sort_values('pnl', ascending=False).head(10)
chk = []
for r in big.itertuples():
    hit = xt[(xt['code'] == r.code) & (xt['ent_date'] <= r.exit_date) & (xt['exit_date'] >= r.ent_date)]
    chk.append(dict(code=r.code, E0_ent=str(r.ent_date.date()), E0_exit=str(r.exit_date.date()), E0_ret=round(r.ret * 100, 1),
                    XT=[(str(h.ent_date.date()), str(h.exit_date.date()), round(h.ret * 100, 1)) for h in hit.itertuples()]))
out['E0_top10_since2024_in_XT'] = chk
big2 = xt[xt['ent_date'] < '2024-01-01'].sort_values('pnl', ascending=False).head(8)
out['XT_top8_before2024'] = [dict(code=r.code, ent=str(r.ent_date.date()), exit=str(r.exit_date.date()), ret=round(r.ret * 100, 1),
                                 in_E0=bool(((e0['code'] == r.code) & (e0['ent_date'] <= r.exit_date) & (e0['exit_date'] >= r.ent_date)).any())) for r in big2.itertuples()]

# (4) 動態門檻表 vs 當年實際
dyn = {int(y): v for y, v in DC['dynamic_table'].items()}
out['dynamic_vs_actual'] = {y: dict(table_min=dyn[y]['min_surge'], actual_s10=DC['drift_B11'][str(y)]['s10_med'] if str(y) in DC['drift_B11'] else None,
                                    table_exit=dyn[y]['exit_surge'], actual_s20=DC['drift_B11'][str(y)]['s20_med'] if str(y) in DC['drift_B11'] else None) for y in dyn}
out['meta'] = dict(runtime_s=round(time.time() - t_all, 1), maxrss_mb=rss())
(H / 'results_posthoc.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
print(json.dumps({k: out[k] for k in ('years_largest_gap_W10_70', 'holdings_by_year', 'miss_buy_by_reason', 'miss_buy_by_year_reason', 'miss_buy_fwd_by_year', 'extra_buy_by_cat',
                                     'contrib_diff_top_years', 'E0_top10_since2024_in_XT', 'XT_top8_before2024', 'holdings_dist_XT_W10_70', 'holdings_dist_XT_SAT')}, ensure_ascii=False, default=str))
print(json.dumps(out['yearly_path_diff'], ensure_ascii=False))
print(out['meta'])
