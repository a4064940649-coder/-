#!/usr/bin/env python3
"""R76＋R84 獨立複製（不讀、不 import S76／S84 的任何 .py）。重用 S75 的 s75lib（連續面板載入、SURGE、訊號、引擎包裝）。
python3 -I -B r76_r84.py → r76_r84_results.json、r76_signal_monthly.csv、r84_events.csv、r84_new_account.csv、r84_daily_analog.csv"""
import sys, json, math, resource
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 's75_continuous_2008_2026'))
import s75lib as K
from s75lib import cslib, stlib, s10lib

P, U = K.load_cont()
F, RT = K.surge(P, 'B2')
sg = K.signal(P, U, RT, 0)
nav, lg, st = K.sim(P, sg, U, 1e6)
IDX = P.IDX; D0, DE = IDX[P.t0], IDX[-1]
T0 = P.t0
e50 = pd.Series(P.E50, index=IDX)
bench = e50.loc[D0:] / e50.loc[D0] * 1e6
out = {}
cg = stlib.cagr
out['accept'] = dict(cagr=cg(nav, D0, DE), final=float(nav.iloc[-1]), n_trades=int(len(lg)), n_closed=int((lg['why'] != 'open_end').sum()),
                     start=str(D0.date()), end=str(DE.date()))
print(out['accept'], flush=True)
snav = pd.read_csv(HERE.parent / 's75_continuous_2008_2026' / 'nav_2008_2026.csv.gz', index_col=0, parse_dates=True)
out['accept']['max_abs_diff_vs_saved_nav'] = float((nav.round(2) - snav['V_BRIDGE']).abs().max())

def lastbefore(d): return IDX[IDX < pd.Timestamp(d)][-1]
def seg_diff(n, b, a, e):
    """a、e＝年；基底＝a 年之前最後一個交易日（沒有就起日）；終點＝e 年最後交易日（或資料末日）。"""
    d0 = max(lastbefore(f'{a}-01-01'), D0); d1 = IDX[IDX <= pd.Timestamp(f'{e}-12-31')][-1]
    return dict(base=str(d0.date()), end=str(d1.date()), strat=cg(n, d0, d1), e0050=cg(b, d0, d1), diff_pp=(cg(n, d0, d1) - cg(b, d0, d1)) * 100)
SEGS = {'early': (2008, 2015), 'late': (2016, 2026), '2008-2012': (2008, 2012), '2013-2015': (2013, 2015), '2016-2018': (2016, 2018), '2019-2026': (2019, 2026)}

# ================= R76
info = sg['info']; S = RT['S']
rows = []
def fwd(j, e, x):
    """執行日 e 開盤買 → 下一執行日 x 開盤賣（還原、無成本）；與引擎估值同式。開盤缺：進場缺→NaN；出場缺→用前收。"""
    g = P.CF[e, j] / P.O[e, j]
    if x > e + 1:
        g = g * np.prod(1 + P.R[e + 1:x, j], axis=0)
    ox = np.where(np.isfinite(P.O[x, j]) & (P.O[x, j] > 0), P.O[x, j] / P.REF[x, j], 1.0)
    return g * ox - 1
from scipy.stats import spearmanr, rankdata
for a, b in zip(info[:-1], info[1:]):
    k = a['k']; e, x = k + 1, b['k'] + 1
    m = pd.Period(a['month'], 'M')
    s = cslib.ins15_mask_row(P, m, k, S.loc[m].to_numpy())
    fin = U[k] & np.isfinite(s)
    jj = np.flatnonzero(fin)
    r = fwd(jj, e, x); ok = np.isfinite(r)
    ic = spearmanr(s[jj][ok], r[ok])[0]
    order = jj[np.argsort(-s[jj], kind='stable')]
    oma = order[P.F_MA60[k][order]]
    rr = dict(zip(jj, r))
    mean = lambda L: float(np.nanmean([rr[j] for j in L])) if len(L) else np.nan
    b50 = P.E50O[x] / P.E50O[e] - 1
    rows.append(dict(month=a['month'], sig_date=a['sig_date'], n_fin=int(fin.sum()), n_ma=int(len(oma)), n_tgt=a['n_tgt'], ic=ic, e0050=b50,
                     top10=mean(order[:10]), top20=mean(order[:20]), top50=mean(order[:50]), ma_top10=mean(oma[:10]), ma_top50=mean(oma[:50])))
M = pd.DataFrame(rows); M['early'] = M['sig_date'] < '2016-01-01'
M.to_csv(HERE / 'r76_signal_monthly.csv', index=False)
r76 = {}
for nm, mk in (('early', M['early']), ('late', ~M['early'])):
    x = M[mk]; d = dict(months=int(len(x)), ic_mean=float(x['ic'].mean()))
    for c in ('top10', 'top20', 'top50', 'ma_top10', 'ma_top50'):
        ex = (x[c] - x['e0050']).dropna()
        d[c + '_ex_ann_pp'] = float(ex.mean() * 12 * 100); d[c + '_t'] = float(ex.mean() / ex.std(ddof=1) * math.sqrt(len(ex)))
        d[c + '_n'] = int(len(ex))
    r76[nm] = d
r76['account_vs_0050'] = {k: seg_diff(nav, bench, *v) for k, v in SEGS.items()}
sf = st['stockfrac']
r76['stockfrac'] = dict(early=float(sf.loc[:'2015-12-31'].mean()), late=float(sf.loc['2016-01-01':].mean()),
                        **{k: float(sf.loc[f'{a}-01-01':f'{b}-12-31'].mean()) for k, (a, b) in list(SEGS.items())[2:]})
nt = pd.DataFrame(info)
r76['n_tgt'] = dict(early=float(nt[nt.sig_date < '2016-01-01'].n_tgt.mean()), late=float(nt[nt.sig_date >= '2016-01-01'].n_tgt.mean()))
# 5. 全期允許零股
ODD0 = cslib.ODD_DATE
cslib.ODD_DATE = pd.Timestamp('2000-01-01')
nav_odd, lg_odd, st_odd = K.sim(P, sg, U, 1e6)
cslib.ODD_DATE = ODD0
r76['odd_all'] = {k: seg_diff(nav_odd, bench, *v) for k, v in SEGS.items()}
r76['odd_all']['full_cagr'] = cg(nav_odd, D0, DE)
# 6. 去掉該段報酬率前 5% 的交易
def block_top(lgx, a, b, q=0.05, mode='ceil'):
    x = lgx[(lgx['ent_date'] >= a) & (lgx['ent_date'] <= b)].sort_values('ret', ascending=False)
    n = int(math.ceil(len(x) * q)) if mode == 'ceil' else int(round(len(x) * q))
    top = x.head(n)
    blk = {}
    for j, e, xx in zip(top['j'], top['ent_t'], top['exit_t']):
        blk.setdefault(int(j), []).append((int(e), int(xx)))
    return blk, top
def sim_block(blk):
    S2, ekw = s10lib.variant_S(sg, 'E0')
    return s10lib.simulate_s10(P, S2, U, cost=K.L.COSTS['strict'], capital=1e6, cap_pct=0.05, sat_w=0.7, sizing='W10', lock=False, block=blk, **ekw)
r76['drop_top5'] = {}
for nm, (a, b) in (('early', (2008, 2015)), ('late', (2016, 2026))):
    blk, top = block_top(lg, pd.Timestamp(f'{a}-01-01'), pd.Timestamp(f'{b}-12-31'))
    n2, l2, s2 = sim_block(blk)
    # 算術版：把那幾筆的損益從期末淨值直接扣掉（不重跑）
    d0 = max(lastbefore(f'{a}-01-01'), D0); d1 = IDX[IDX <= pd.Timestamp(f'{b}-12-31')][-1]
    yrs = (d1 - d0).days / 365.25
    v_end = nav.loc[d1] - top['pnl'].sum()
    arith = (v_end / nav.loc[d0]) ** (1 / yrs) - 1 - cg(bench, d0, d1)
    r76['drop_top5'][nm] = dict(n_trades=int(((lg['ent_date'] >= f'{a}-01-01') & (lg['ent_date'] <= f'{b}-12-31')).sum()), n_dropped=int(len(top)),
                                min_ret_dropped=float(top['ret'].min()), rerun_block=seg_diff(n2, bench, a, b), arith_minus_pnl_diff_pp=float(arith * 100),
                                dropped=[f"{c}@{str(d.date())}:{r:.2f}" for c, d, r in zip(top['code'], top['ent_date'], top['ret'])])
out['r76'] = r76
print(json.dumps(r76, indent=1, default=str)[:6000], flush=True)

# ================= R84
E = P.E50
ev = []
for i, a in enumerate(info):
    k = a['k']
    r10 = E[k] / E[k - 10] - 1; r20 = E[k] / E[k - 20] - 1; gap = E[k] / E[k - 59:k + 1].mean() - 1
    m = pd.Period(a['month'], 'M')
    s = cslib.ins15_mask_row(P, m, k, S.loc[m].to_numpy())
    fin = U[k] & np.isfinite(s)
    ev.append(dict(i=i, month=a['month'], sig_date=a['sig_date'], k=k, r10=r10, r20=r20, gap60=gap, event=bool(r10 <= -0.10 or r20 <= -0.10 or gap <= -0.08),
                   strict=bool(r10 <= -0.15 or r20 <= -0.15), n_ma=a['n_ma'], n_fin=int(fin.sum()), n_univ=int(U[k].sum()), ma_share_fin=a['n_ma'] / fin.sum(),
                   ma_share_univ=float((U[k] & P.F_MA60[k]).sum() / U[k].sum()), n_tgt=a['n_tgt'], stockfrac_exec_close=float(sf.iloc[k + 1 - T0]) if k + 1 < P.T else np.nan))
EV = pd.DataFrame(ev)
EV['first'] = EV['event'] & ~EV['event'].shift(1, fill_value=False)
def fwd12(series, d, mode='cal'):
    d = pd.Timestamp(d)
    if mode == 'cal':
        t = d + pd.DateOffset(years=1)
        if t > series.index[-1]: return np.nan
        return float(series.loc[:t].iloc[-1] / series.loc[d] - 1)
    i = series.index.get_loc(d)
    return float(series.iloc[i + 252] / series.iloc[i] - 1) if i + 252 < len(series) else np.nan
for md in ('cal', 'td252'):
    EV[f'acct12_{md}'] = [fwd12(nav, d, md) for d in EV['sig_date']]
    EV[f'e50_12_{md}'] = [fwd12(bench, d, md) for d in EV['sig_date']]
# 10. 新帳戶：訊號日收盤 100 萬現金 → 執行日開盤照當月名單買（每檔＝衛星 70% ÷ 10），其餘開盤全部買 0050；之後照現行規則。
def new_acct(k, odd_all):
    t0s = P.t0; P.t0 = k
    if odd_all: cslib.ODD_DATE = pd.Timestamp('2000-01-01')
    try:
        n, l, s_ = K.sim(P, sg, U, 1e6)
    finally:
        P.t0 = t0s; cslib.ODD_DATE = ODD0
    return n
for odd in (True, False):
    r12 = []; mdd = []
    for d, k in zip(EV['sig_date'], EV['k']):
        if pd.Timestamp(d) + pd.DateOffset(years=1) > DE or k + 1 >= P.T:
            r12.append(np.nan); mdd.append(np.nan); continue
        n = new_acct(int(k), odd)
        r12.append(fwd12(n, d, 'cal'))
        w = n.loc[d:pd.Timestamp(d) + pd.DateOffset(years=1)]
        mdd.append(float(w.min() / 1e6 - 1))
    tag = 'odd' if odd else 'lot'
    EV[f'new12_{tag}'] = r12; EV[f'new_minrel_{tag}'] = mdd
    print('new acct done', tag, flush=True)
EV.to_csv(HERE / 'r84_new_account.csv', index=False)
X = EV[EV['event']]
X.to_csv(HERE / 'r84_events.csv', index=False)
Fst = EV[EV['first']]
r84 = dict(n_signal_months=int(len(EV)), n_event=int(len(X)), event_months=X['sig_date'].tolist(), first_months=Fst['sig_date'].tolist(), n_strict=int(EV['strict'].sum()),
           min_r20=dict(v=float(EV['r20'].min()), d=EV.loc[EV['r20'].idxmin(), 'sig_date']),
           events=X.drop(columns=['i', 'k']).to_dict('records'),
           first_median={c: float(Fst[c].median()) for c in ('acct12_cal', 'e50_12_cal', 'acct12_td252', 'e50_12_td252', 'new12_odd', 'new12_lot')},
           first_min={c: float(Fst[c].min()) for c in ('acct12_cal', 'new12_odd', 'new12_lot', 'new_minrel_odd')},
           first_max={c: float(Fst[c].max()) for c in ('acct12_cal', 'new12_odd', 'new12_lot')},
           first_new_minrel_median=float(Fst['new_minrel_odd'].median()),
           all_months=dict(n=int(EV['new12_odd'].notna().sum()), new12_odd_median=float(EV['new12_odd'].median()), new12_lot_median=float(EV['new12_lot'].median()),
                           acct12_median=float(EV['acct12_cal'].median()), n_tgt_median=float(EV['n_tgt'].median()), ma_share_median_nonevent=float(EV[~EV['event']]['ma_share_fin'].median())))
# 11. 0050 日資料：7 個交易日累計 ≤ −18.4%
es = pd.Series(E, index=IDX).dropna()
r7 = es / es.shift(7) - 1
hit = r7[r7 <= -0.184]
r84['daily7'] = [dict(date=str(d.date()), r7=float(v), fwd12_cal=fwd12(es, d, 'cal'), fwd12_td252=fwd12(es, d, 'td252')) for d, v in hit.items()]
r84['daily7_range'] = [str(es.index[0].date()), str(es.index[-1].date())]
r84['daily7_near'] = [dict(date=str(d.date()), r7=float(v)) for d, v in r7.nsmallest(8).items()]
pd.DataFrame(r84['daily7']).to_csv(HERE / 'r84_daily_analog.csv', index=False)
out['r84'] = r84
out['maxrss_mb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
(HERE / 'r76_r84_results.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
print(json.dumps(r84, indent=1, default=str, ensure_ascii=False)[:9000])
print('maxrss', out['maxrss_mb'])
