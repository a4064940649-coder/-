#!/usr/bin/env python3
"""S82 大贏家依賴診斷。python3 -I -B run.py → results.json＋csv。
重用 S75 連續面板與引擎（s76lib → s75lib → …，全部不改）。只診斷；情境格（漏單、不敢買）照 PREREG.json。"""
import sys, json, time, math, inspect, hashlib, resource
from pathlib import Path
HERE = Path(__file__).resolve().parent
AG = HERE.parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(HERE / 'PREREG.json') == (HERE / 'PREREG.sha256').read_text().split()[0]
sys.path.insert(0, str(AG / 's76_early_vs_late_diagnosis'))
import numpy as np, pandas as pd
import s76lib as Q
from s76lib import K, R, s10lib, cslib

t0 = time.time()
P, U = K.load_cont()
F_, RT = K.surge(P, 'B2')
D0, DE = P.IDX[P.t0], P.IDX[-1]
T, N = P.T, P.N
bench = pd.Series(P.E50, index=P.IDX).loc[D0:]
D_SPLIT = P.IDX[P.IDX < pd.Timestamp(Q.SPLIT)][-1]
OUT = dict(task='S82 大贏家依賴診斷', start=str(D0.date()), end=str(DE.date()), prereg_sha256=sha(HERE / 'PREREG.json'))
jd = lambda v: None if (isinstance(v, float) and v != v) else (v.item() if hasattr(v, 'item') else str(v))

# ============================================================ 引擎副本：只多記「沒買到的原因」
_src = inspect.getsource(s10lib.simulate_s10)
_REPL = [("def simulate_s10(", "def simulate_s82("),
         ("exec_mode='open', trail=None, trail_daily=True, requal=None):", "exec_mode='open', trail=None, trail_daily=True, requal=None, why_log=None):"),
         ("                st['noslot'] += 1; continue", "                st['noslot'] += 1; why_log.append((t, j, 'noslot')); continue"),
         ("                st['blk_dropped'] += 1; continue", "                st['blk_dropped'] += 1; why_log.append((t, j, 'banned')); continue"),
         ("                st['blk_noopen'] += 1; continue", "                st['blk_noopen'] += 1; why_log.append((t, j, 'noopen')); continue"),
         ("                st['blk_limup'] += 1; continue", "                st['blk_limup'] += 1; why_log.append((t, j, 'limup')); continue"),
         ("                st['blk_lot' if budget >= 0.5 * target else 'blk_nocash'] += 1\n",
          "                st['blk_lot' if budget >= 0.5 * target else 'blk_nocash'] += 1; why_log.append((t, j, 'lot' if budget >= 0.5 * target else 'nocash'))\n"),
         ("                    if sh <= 0:\n                        continue\n", "                    if sh <= 0:\n                        why_log.append((t, j, 'cap')); continue\n"),
         ("                st['blk_nocash'] += 1\n", "                st['blk_nocash'] += 1; why_log.append((t, j, 'nocash'))\n"),
         ("                st['blk_min'] += 1\n", "                st['blk_min'] += 1; why_log.append((t, j, 'min'))\n")]
for a, b in _REPL:
    assert _src.count(a) == 1, a
    _src = _src.replace(a, b)
_ns = dict(vars(s10lib)); exec(_src, _ns)


def engine(sg, block=None, why_log=None):
    return _ns['simulate_s82'](P, sg, U, cost=Q.COSTS['strict'], capital=1e6, cap_pct=0.05, sat_w=0.7, sizing='W10', lock=False, block=block,
                               why_log=why_log if why_log is not None else [])


def perf(nav):
    return dict(cagr=K.cagr(nav, D0, DE), cagr_early=K.cagr(nav, D0, D_SPLIT), cagr_late=K.cagr(nav, D_SPLIT, DE), final=float(nav.iloc[-1]))


# ============================================================ 0 驗收
sg0 = K.signal(P, U, RT, 0)
sgc = Q.build_sig(P, U, RT)
nav_off, lg_off, st_off = K.sim(P, sg0, U, 1e6)
WHY = []
nav0, lg0, st0 = engine(sgc, why_log=WHY)
assert np.array_equal(nav_off.to_numpy(), nav0.to_numpy()) and len(lg0) == len(lg_off) == 632
s75 = json.load(open(AG / 's75_continuous_2008_2026' / 'results.json'))['versions']['V_BRIDGE']['full']
BASE = perf(nav0)
assert abs(BASE['final'] - 53078883.65) < 0.01 and abs(BASE['final'] - s75['final_nav']) < 1e-6 and abs(BASE['cagr'] - 0.2352) < 5e-5
b_perf = perf(bench.reindex(nav0.index))
OUT['accept'] = dict(BASE, s75_final=s75['final_nav'], n_trades=int(len(lg0)), n_closed=int((lg0['why'] != 'open_end').sum()), copy_engine_identical=True,
                     e0050=b_perf, blk={k: int(st0[k]) for k in ('n_sig', 'n_buy', 'blk_limup', 'blk_noopen', 'blk_lot', 'blk_cap', 'blk_nocash', 'blk_min', 'noslot')})
print('驗收', OUT['accept'], flush=True)

det = sgc['detail']; ks = [d['k'] for d in det]; k2i = {k: i for i, k in enumerate(ks)}
sigrows = sgc['sigrows']; RQ = sgc['RQ']
F1, b1 = Q.fwd_matrix(P, ks, 1)
F3, b3 = Q.fwd_matrix(P, ks, 3)
IND, INAMES = Q.load_industry(P)
# 市值代理（同 S12：股數落後 63 日 × 當時收盤 × 之後含息漲跌；缺 → 當日股數 × 收盤）。2006–2013 股數為回溯值。
SH = pd.read_parquet(AG / 'o_lc_sector_rotation_pit' / 'data' / 'shares_full.parquet')
SH = SH.reindex(SH.index.union(P.IDX)).ffill().reindex(P.IDX).reindex(columns=P.COLS).to_numpy(np.float64)
now_mc = SH * P.CF
MC = np.full((T, N), np.nan); MC[63:] = now_mc[:-63] * (P.G[63:] / P.G[:-63]); MC = np.where(np.isfinite(MC), MC, now_mc)
del SH, now_mc
e50 = P.E50
e50ma = pd.Series(e50).rolling(60, min_periods=60).mean().to_numpy()


def feat_at(k, j):
    """訊號日 k 收盤時看得到的特徵。"""
    g = P.G[max(0, k - 239):k + 1, j]
    amt = P.AMT[max(0, k - 19):k + 1, j]
    u = U[k]; mcu = MC[k][u]; mcu = mcu[np.isfinite(mcu)]
    return dict(ret20=float(P.RET20[k, j]), ret60=float(P.G[k, j] / P.G[k - 60, j] - 1) if k >= 60 else np.nan, px=float(P.CF[k, j]),
                amt20_m=float(np.nanmean(amt) / 1e6) if np.isfinite(amt).any() else np.nan,
                mc_e8=float(MC[k, j] / 1e8), mc_pct=float((mcu < MC[k, j]).mean()) if np.isfinite(MC[k, j]) and len(mcu) else np.nan,
                dist52=float(P.G[k, j] / np.nanmax(g) - 1) if np.isfinite(g).any() else np.nan,
                ind=INAMES[IND[k, j]] if IND[k, j] >= 0 else '缺',
                bull=bool(e50[k] > e50ma[k]), e50_ret60=float(e50[k] / e50[k - 60] - 1), e50_dd250=float(e50[k] / e50[max(0, k - 249):k + 1].max() - 1),
                breadth=float((P.F_MA60[k] & u).sum() / max(u.sum(), 1)))


# ============================================================ 1 交易表（632 筆）＋持有路徑
rows = []
for r in lg0.itertuples(index=False):
    j, k, e, x = int(r.j), int(r.sig_t), int(r.ent_t), int(r.exit_t)
    d = det[k2i[k]]
    opn = r.why == 'open_end'
    last = x if opn else x - 1
    g = np.empty(last - e + 1); g[0] = P.CF[e, j] / P.O[e, j]
    for t in range(e + 1, last + 1):
        g[t - e] = g[t - e - 1] * (1 + P.R[t, j])
    if opn:
        gx = g[-1]
    else:
        gx = g[-1] * (P.O[x, j] / P.REF[x, j] if np.isfinite(P.O[x, j]) and P.O[x, j] > 0 else 1.0)
    path = np.append(g, gx)
    pk = int(np.argmax(path)); gpk = float(path[pk])
    # 訊號列與續抱分類
    srs = [t for t in range(e, last + 1) if sigrows[t]]
    V = lambda s: g[s - e] * (P.O[s + 1, j] / P.REF[s + 1, j] if s + 1 < T and np.isfinite(P.O[s + 1, j]) and P.O[s + 1, j] > 0 else 1.0)
    lg_cls = dict(entry=0.0, rebuy=0.0, top10_nomom=0.0, r11_20=0.0, other=0.0)
    prev, cls = 1.0, 'entry'; n_rebuy = n_keep = 0
    for n_, s in enumerate(s for s in srs):
        v = V(s)
        lg_cls[cls] += math.log(v / prev); prev = v
        o = det[k2i[s]]['order']
        if j in RQ.get(s, ()):
            cls = 'rebuy'; n_rebuy += 1
        elif j in o[:10]:
            cls = 'top10_nomom'; n_keep += 1
        elif j in o[:20]:
            cls = 'r11_20'; n_keep += 1
        else:
            cls = 'other'                                      # 被標記出場（之後只剩賣出前的殘段，通常 0）
    lg_cls[cls] += math.log(gx / prev)
    g_first = V(srs[0]) if srs else gx
    post = {}
    for h in (20, 60):
        post[f'post{h}'] = float((P.CF[x, j] / P.O[x, j]) * P.G[x + h, j] / P.G[x, j] - 1) if (not opn and x + h < T and np.isfinite(P.O[x, j]) and P.O[x, j] > 0) else np.nan
    bh = (e50[x] if opn else P.E50O[x]) / P.E50O[e] - 1
    f = feat_at(k, j)
    rows.append(dict(code=str(r.code), sig_date=str(P.IDX[k].date()), ent_date=str(P.IDX[e].date()), exit_date=str(P.IDX[x].date()), why=r.why, j=j, sig_t=k, ent_t=e, exit_t=x,
                     ret=float(r.ret), pnl=float(r.pnl), cost=float(r.cost), contrib=float(r.pnl / nav0.iloc[e - 1 - P.t0]), hold_days=int(r.hold_days),
                     b0050=float(bh), ex=float(r.ret - bh), surge=float(r.score), rank=int(np.flatnonzero(d['order'] == j)[0]) + 1, n_tgt=int(len(d['tgt'])),
                     gap_open=float(P.O[e, j] / P.REF[e, j] - 1), gross=float(gx - 1), mfe=float(gpk - 1), mfe_day=pk, mfe_frac=float(pk / max(len(path) - 1, 1)),
                     mae=float(path.min() - 1), giveback=float(gx / gpk - 1), n_periods=len(srs), g_first=float(g_first - 1), n_rebuy=n_rebuy, n_keep=n_keep,
                     **{'lg_' + c: v for c, v in lg_cls.items()}, **post, **f))
TR = pd.DataFrame(rows)
TR['elec'] = TR['ind'].isin(Q.ELEC)
TR['year'] = pd.to_datetime(TR['ent_date']).dt.year
TR['era'] = np.where(TR['year'] < 2016, '2008–2015', np.where(TR['year'] < 2019, '2016–2018', '2019–2026'))
n = len(TR); n5 = int(round(n * 0.05)); n10 = int(round(n * 0.10))
TR['rk_ret'] = TR['ret'].rank(ascending=False, method='first').astype(int)
TR['rk_pnl'] = TR['pnl'].rank(ascending=False, method='first').astype(int)
TR['top5'] = TR['rk_ret'] <= n5; TR['top10'] = TR['rk_ret'] <= n10
TR['bot5'] = TR['ret'].rank(ascending=True, method='first') <= n5
TR.drop(columns=['j']).to_csv(HERE / 'trades_features.csv.gz', index=False)
assert abs(TR['pnl'].sum() - lg0['pnl'].sum()) < 1e-6


def skew(x):
    x = np.asarray(x, float); m = x.mean(); s = x.std(ddof=0)
    return float(((x - m) ** 3).mean() / s ** 3)


def dist(y):
    r = y['ret']; q = lambda p: float(r.quantile(p))
    o = dict(n=int(len(y)), mean=float(r.mean()), median=q(.5), win=float((r > 0).mean()), min=float(r.min()), p05=q(.05), p10=q(.1), p25=q(.25), p75=q(.75), p90=q(.9), p95=q(.95), p99=q(.99), max=float(r.max()),
             skew=skew(r), n_ge50=int((r >= .5).sum()), n_ge100=int((r >= 1).sum()), n_le_m20=int((r <= -.2).sum()),
             avg_win=float(r[r > 0].mean()), avg_loss=float(r[r <= 0].mean()), hold_median=float(y['hold_days'].median()), pnl_total=float(y['pnl'].sum()), contrib_total=float(y['contrib'].sum()),
             ret_sum=float(r.sum()))
    m5 = max(1, int(round(len(y) * .05))); m10 = max(1, int(round(len(y) * .10)))
    sr = y.sort_values('ret', ascending=False); sp = y.sort_values('pnl', ascending=False); pos = y.loc[y['pnl'] > 0, 'pnl'].sum()
    for nm, m in (('top5pct', m5), ('top10pct', m10)):
        o[nm] = dict(n=m, by_ret_pnl_share=float(sr['pnl'].head(m).sum() / y['pnl'].sum()), by_ret_contrib_share=float(sr['contrib'].head(m).sum() / y['contrib'].sum()),
                     by_ret_retsum_share=float(sr['ret'].head(m).sum() / r.sum()), by_pnl_pnl_share=float(sp['pnl'].head(m).sum() / y['pnl'].sum()),
                     by_pnl_share_of_gross_gain=float(sp['pnl'].head(m).sum() / pos), mean_ret_without=float(sr['ret'].iloc[m:].mean()), mean_ex_without=float(y.sort_values('ex', ascending=False)['ex'].iloc[m:].mean()),
                     contrib_without=float(sr['contrib'].iloc[m:].sum()))
    o['mean_ex'] = float(y['ex'].mean()); o['gross_loss_pnl'] = float(y.loc[y['pnl'] <= 0, 'pnl'].sum()); o['gross_gain_pnl'] = float(pos)
    return o


OUT['q1_distribution'] = {nm: dist(y) for nm, y in [('全期', TR)] + [(e_, TR[TR['era'] == e_]) for e_ in ('2008–2015', '2016–2018', '2019–2026')] + [('已平倉 624 筆', TR[TR['why'] != 'open_end'])]}
OUT['q1_cut'] = dict(n5=n5, n10=n10, top5_ret_min=float(TR.loc[TR['top5'], 'ret'].min()), top10_ret_min=float(TR.loc[TR['top10'], 'ret'].min()), bot5_ret_max=float(TR.loc[TR['bot5'], 'ret'].max()))
OUT['q1_top5_by_era'] = TR[TR['top5']].groupby('era').size().to_dict(); OUT['q1_top5_by_year'] = {int(a): int(b) for a, b in TR[TR['top5']].groupby('year').size().items()}
OUT['q1_trades_by_era'] = TR.groupby('era').size().to_dict()
cols_show = ['code', 'ent_date', 'exit_date', 'why', 'ret', 'pnl', 'contrib', 'hold_days', 'surge', 'rank', 'ret20', 'px', 'amt20_m', 'mc_e8', 'dist52', 'ind', 'mfe', 'gross', 'giveback', 'n_periods', 'g_first']
OUT['q1_top_list_by_pnl'] = TR.sort_values('pnl', ascending=False).head(15)[cols_show + ['rk_ret']].to_dict('records')
OUT['q1_top_list_by_ret'] = TR.sort_values('ret', ascending=False).head(32)[cols_show + ['rk_pnl']].to_dict('records')

FEATS = ['surge', 'rank', 'n_tgt', 'ret20', 'ret60', 'px', 'amt20_m', 'mc_e8', 'mc_pct', 'dist52', 'gap_open', 'e50_ret60', 'e50_dd250', 'breadth', 'hold_days', 'n_periods', 'mfe', 'mfe_day', 'mfe_frac', 'giveback', 'gross', 'g_first', 'b0050', 'mae']


def auc(pos, neg):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float); pos = pos[np.isfinite(pos)]; neg = neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return None
    rk = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def compare(y, flag, label):
    a, b = y[y[flag]], y[~y[flag]]
    o = dict(n_group=int(len(a)), n_rest=int(len(b)), bull_share=[float(a['bull'].mean()), float(b['bull'].mean())], elec_share=[float(a['elec'].mean()), float(b['elec'].mean())],
             semi_share=[float((a['ind'] == '半導體業').mean()), float((b['ind'] == '半導體業').mean())], rank1_3_share=[float((a['rank'] <= 3).mean()), float((b['rank'] <= 3).mean())])
    for f in FEATS:
        o[f] = dict(group_median=float(a[f].median()), rest_median=float(b[f].median()), group_mean=float(a[f].mean()), rest_mean=float(b[f].mean()), auc=auc(a[f], b[f]))
    return o


OUT['q1_features'] = dict(top5_vs_rest_all=compare(TR, 'top5', '前 5%'), top10_vs_rest_all=compare(TR, 'top10', '前 10%'),
                          top5_vs_rest_2019on=compare(TR[TR['era'] == '2019–2026'], 'top5', '前 5%（只看 2019 起）'), bot5_vs_rest_all=compare(TR, 'bot5', '最差 5%'),
                          bot5_vs_rest_2019on=compare(TR[TR['era'] == '2019–2026'], 'bot5', '最差 5%（只看 2019 起）'))


def ind_tab(y):
    g = y.groupby('ind').agg(n=('ret', 'size'), top5=('top5', 'sum'), bot5=('bot5', 'sum'), mean_ret=('ret', 'mean'), contrib=('contrib', 'sum')).sort_values('n', ascending=False)
    return [dict(ind=i, **{k: (int(v) if k in ('n', 'top5', 'bot5') else float(v)) for k, v in r_.items()}) for i, r_ in g.iterrows()]


OUT['q1_industry'] = ind_tab(TR)
OUT['q1_month_of_year'] = {int(m): dict(n=int(len(g_)), top5=int(g_['top5'].sum()), bot5=int(g_['bot5'].sum())) for m, g_ in TR.groupby(pd.to_datetime(TR['sig_date']).dt.month)}
OUT['q1_surge_bins'] = [dict(bin=str(b_), n=int(len(g_)), top5=int(g_['top5'].sum()), bot5=int(g_['bot5'].sum()), mean_ret=float(g_['ret'].mean()), median_ret=float(g_['ret'].median()))
                        for b_, g_ in TR.groupby(pd.cut(TR['surge'], [-np.inf, 2, 3, 4, 6, np.inf]), observed=True)]
OUT['q1_ret20_bins'] = [dict(bin=str(b_), n=int(len(g_)), top5=int(g_['top5'].sum()), bot5=int(g_['bot5'].sum()), mean_ret=float(g_['ret'].mean()), median_ret=float(g_['ret'].median()))
                        for b_, g_ in TR.groupby(pd.cut(TR['ret20'], [0.1, 0.2, 0.3, 0.5, np.inf]), observed=True)]
OUT['q1_rank_bins'] = [dict(rank=int(b_), n=int(len(g_)), top5=int(g_['top5'].sum()), bot5=int(g_['bot5'].sum()), mean_ret=float(g_['ret'].mean())) for b_, g_ in TR.groupby('rank')]
OUT['q1_bull'] = {str(b_): dict(n=int(len(g_)), top5=int(g_['top5'].sum()), bot5=int(g_['bot5'].sum()), mean_ret=float(g_['ret'].mean())) for b_, g_ in TR.groupby('bull')}
print('1 交易表', f'{time.time() - t0:.0f}s', flush=True)

# ============================================================ 2 出場規則抓到幾成
W = TR[TR['top5']]
q2 = {}
for nm, y in (('前 5%（32 筆）', W), ('前 10%（63 筆）', TR[TR['top10']]), ('最大浮盈 ≥50%', TR[TR['mfe'] >= .5]), ('最大浮盈 ≥100%', TR[TR['mfe'] >= 1.0]), ('全部', TR)):
    c = y[y['mfe'] > 0]
    q2[nm] = dict(n=int(len(y)), mfe_median=float(y['mfe'].median()), gross_median=float(y['gross'].median()), net_ret_median=float(y['ret'].median()),
                  capture_agg=float(y['gross'].sum() / y['mfe'].sum()), capture_median=float((c['gross'] / c['mfe']).median()), giveback_median=float(y['giveback'].median()), giveback_mean=float(y['giveback'].mean()),
                  hold_median=float(y['hold_days'].median()), periods_median=float(y['n_periods'].median()), mfe_day_median=float(y['mfe_day'].median()), mfe_frac_median=float(y['mfe_frac'].median()),
                  days_peak_to_exit_median=float((y['hold_days'] - y['mfe_day']).median()),
                  post20_median=float(y['post20'].median()), post60_median=float(y['post60'].median()), post60_gt20_share=float((y['post60'].dropna() > .2).mean()) if y['post60'].notna().any() else None,
                  post60_lt_m20_share=float((y['post60'].dropna() < -.2).mean()) if y['post60'].notna().any() else None, post60_n=int(y['post60'].notna().sum()))
OUT['q2_capture'] = q2
gb = TR[(TR['mfe'] >= .5) & (TR['gross'] < .2)]
OUT['q2_roundtrip'] = dict(n_mfe_ge50=int((TR['mfe'] >= .5).sum()), n_mfe_ge50_exit_lt20_gross=int(len(gb)), n_mfe_ge50_exit_lt20_net=int(((TR['mfe'] >= .5) & (TR['ret'] < .2)).sum()),
                           n_mfe_ge30=int((TR['mfe'] >= .3).sum()), n_mfe_ge30_exit_lt10=int(((TR['mfe'] >= .3) & (TR['gross'] < .1)).sum()), n_mfe_ge30_exit_le0=int(((TR['mfe'] >= .3) & (TR['gross'] <= 0)).sum()),
                           n_mfe_ge20=int((TR['mfe'] >= .2).sum()), n_mfe_ge20_exit_le0=int(((TR['mfe'] >= .2) & (TR['gross'] <= 0)).sum()),
                           list=gb[['code', 'ent_date', 'exit_date', 'mfe', 'gross', 'ret', 'mfe_day', 'hold_days', 'why']].to_dict('records'),
                           list_mfe30_exit_le0=TR[(TR['mfe'] >= .3) & (TR['gross'] <= 0)][['code', 'ent_date', 'exit_date', 'mfe', 'gross', 'ret', 'mfe_day', 'hold_days']].to_dict('records'))
# 續抱規則的貢獻
cut5 = OUT['q1_cut']['top5_ret_min']
hold = {}
for nm, y in (('前 5%（32 筆）', W), ('前 10%（63 筆）', TR[TR['top10']]), ('報酬 ≥50%', TR[TR['ret'] >= .5]), ('全部', TR)):
    tot = sum(y['lg_' + c].sum() for c in ('entry', 'rebuy', 'top10_nomom', 'r11_20', 'other'))
    hold[nm] = dict(n=int(len(y)), n_one_period=int((y['n_periods'] <= 1).sum()), n_multi_period=int((y['n_periods'] >= 2).sum()), periods_mean=float(y['n_periods'].mean()),
                    first_period_gross_median=float(y['g_first'].median()), final_gross_median=float(y['gross'].median()), first_period_gross_mean=float(y['g_first'].mean()), final_gross_mean=float(y['gross'].mean()),
                    n_first_period_ge50=int((y['g_first'] >= .5).sum()), n_first_period_ge_cut5=int((y['g_first'] >= cut5).sum()),
                    n_ever_kept_off_list=int((y['n_keep'] > 0).sum()), n_always_on_buy_list=int(((y['n_keep'] == 0) & (y['n_periods'] >= 2)).sum()),
                    log_gain_share={c: float(y['lg_' + c].sum() / tot) for c in ('entry', 'rebuy', 'top10_nomom', 'r11_20', 'other')})
OUT['q2_hold_rule'] = hold
print('2 出場', f'{time.time() - t0:.0f}s', flush=True)

# ============================================================ 3 錯過的大贏家（季線上前 10 名）
iv = {}
for r in lg0.itertuples(index=False):
    iv.setdefault(int(r.j), []).append((int(r.ent_t), int(r.exit_t), int(r.sig_t)))
whyd = {(t, j): w for t, j, w in WHY}
mrow = []
for i, d in enumerate(det):
    k, s, fin = d['k'], d['s'], d['fin']
    pool = d['order'][:10]; tg = set(int(j) for j in d['tgt'])
    oraw = np.flatnonzero(fin)[np.argsort(-s[fin], kind='stable')][:10]
    items = [(int(j), 'pool', p_ + 1) for p_, j in enumerate(pool)] + [(int(j), 'raw_below_ma', p_ + 1) for p_, j in enumerate(oraw) if not P.F_MA60[k, j]]
    for j, src, pos_ in items:
        heldnow = any(a <= k and b >= k + 1 for a, b, _ in iv.get(j, ()))
        bought = any(sk == k for _, _, sk in iv.get(j, ()))
        if src == 'raw_below_ma':
            stt = 'held' if heldnow else 'below_ma60'
        elif heldnow:
            stt = 'held'
        elif bought:
            stt = 'bought'
        elif j not in tg:
            stt = 'mom_fail'
        else:
            stt = whyd.get((k + 1, j), 'unknown')
        mrow.append(dict(i=i, sig_date=str(P.IDX[k].date()), code=str(P.COLS[j]), src=src, pos=pos_, status=stt, surge=float(s[j]), ret20=float(P.RET20[k, j]), px=float(P.CF[k, j]),
                         f1=float(F1[i, j]), f3=float(F3[i, j]), b3=float(b3[i]), ind=INAMES[IND[k, j]] if IND[k, j] >= 0 else '缺'))
MS = pd.DataFrame(mrow)
MS['era'] = np.where(MS['sig_date'] < Q.SPLIT, '前段 2008–2015', '後段 2016–2026')
MS['era3'] = np.where(MS['sig_date'] < '2016-01-01', '2008–2015', np.where(MS['sig_date'] < '2019-01-01', '2016–2018', '2019–2026'))
MS['big'] = MS['f3'] > 0.5
MS.to_csv(HERE / 'pool_top10_status.csv.gz', index=False)
assert (MS['status'] != 'unknown').all(), MS[MS['status'] == 'unknown']
q3 = {}
for nm, mk in (('全期', MS['era'] != ''), ('前段 2008–2015', MS['era'] == '前段 2008–2015'), ('後段 2016–2026', MS['era'] == '後段 2016–2026'), ('2016–2018', MS['era3'] == '2016–2018'), ('2019–2026', MS['era3'] == '2019–2026')):
    y = MS[mk & MS['f3'].notna()]; yp = y[y['src'] == 'pool']
    tab = {}
    for stt, g_ in y.groupby('status'):
        tab[stt] = dict(n=int(len(g_)), n_big=int(g_['big'].sum()), big_rate=float(g_['big'].mean()), f3_mean=float(g_['f3'].mean()), f3_median=float(g_['f3'].median()), ex3_mean=float((g_['f3'] - g_['b3']).mean()), px_median=float(g_['px'].median()))
    miss = yp[yp['big'] & ~yp['status'].isin(['bought', 'held'])]
    nb = int(yp['big'].sum())
    q3[nm] = dict(n_months=int(y['i'].nunique()), n_pool=int(len(yp)), n_pool_big=nb, by_status=tab,
                  big_bought=int((yp['big'] & (yp['status'] == 'bought')).sum()), big_held=int((yp['big'] & (yp['status'] == 'held')).sum()), big_missed=int(len(miss)),
                  missed_by_reason={k_: int(v) for k_, v in miss['status'].value_counts().items()}, missed_share_of_pool_big=float(len(miss) / nb) if nb else None,
                  missed_new_only_denominator=int((yp['big'] & (yp['status'] != 'held')).sum()),
                  below_ma_raw_big=int((y['big'] & (y['status'] == 'below_ma60')).sum()), below_ma_raw_n=int((y['status'] == 'below_ma60').sum()))
OUT['q3_missed'] = q3
OUT['q3_missed_list'] = MS[(MS['src'] == 'pool') & MS['big'] & ~MS['status'].isin(['bought', 'held'])][['sig_date', 'code', 'pos', 'status', 'surge', 'ret20', 'px', 'f1', 'f3', 'ind']].to_dict('records')
OUT['q3_note'] = dict(last3_signals_excluded=[det[i]['m'].__str__() for i in range(len(det) - 3, len(det))], big_def='執行日開盤 → 之後第 3 個執行日開盤的含息報酬 > 50%（不扣成本）')
print('3 錯過', f'{time.time() - t0:.0f}s', flush=True)

# ============================================================ 4 最大虧損
Lz = TR[TR['bot5']]
OUT['q4_losers'] = dict(n=int(len(Lz)), ret_max=float(Lz['ret'].max()), ret_min=float(Lz['ret'].min()), ret_mean=float(Lz['ret'].mean()), contrib_sum=float(Lz['contrib'].sum()), pnl_sum=float(Lz['pnl'].sum()),
                        by_era={k_: int(v) for k_, v in Lz.groupby('era').size().items()}, by_year={int(a): int(b) for a, b in Lz.groupby('year').size().items()},
                        one_period_share=float((Lz['n_periods'] <= 1).mean()), one_period_share_all=float((TR['n_periods'] <= 1).mean()),
                        mfe_lt5_share=float((Lz['mfe'] < .05).mean()), mfe_ge10_share=float((Lz['mfe'] >= .10).mean()), b0050_neg_share=float((Lz['b0050'] < 0).mean()), b0050_lt_m5_share=float((Lz['b0050'] < -.05).mean()),
                        b0050_neg_share_all=float((TR['b0050'] < 0).mean()), b0050_lt_m5_share_all=float((TR['b0050'] < -.05).mean()), ex_mean=float(Lz['ex'].mean()),
                        same_month_clusters={k_: int(v) for k_, v in Lz.groupby('sig_date').size().items() if v >= 2},
                        ret20_ge50_share=[float((Lz['ret20'] >= .5).mean()), float((TR['ret20'] >= .5).mean())], ret20_ge30_share=[float((Lz['ret20'] >= .3).mean()), float((TR['ret20'] >= .3).mean())],
                        first5d_loss_median=None,
                        list=Lz.sort_values('ret')[cols_show + ['b0050', 'mae']].to_dict('records'))
# 進場後 5 天的報酬（輸家是不是一開始就跌）
f5 = []
for r in TR.itertuples(index=False):
    e, j = r.ent_t, lg0['j'].iloc[r.Index] if hasattr(r, 'Index') else None,
    f5.append(np.nan)
g5 = np.array([(P.CF[min(e + 4, x if w == 'open_end' else x - 1), j] if False else np.nan) for e, x, w, j in zip(lg0['ent_t'], lg0['exit_t'], lg0['why'], lg0['j'])])
d5 = []
for e, x, w, j in zip(lg0['ent_t'], lg0['exit_t'], lg0['why'], lg0['j']):
    last = x if w == 'open_end' else x - 1
    t5 = min(e + 4, last)
    d5.append(float(P.CF[e, j] / P.O[e, j] * P.G[t5, j] / P.G[e, j] - 1))
TR['ret5d'] = d5
OUT['q4_losers']['first5d_median'] = [float(TR.loc[TR['bot5'], 'ret5d'].median()), float(TR.loc[~TR['bot5'], 'ret5d'].median())]
OUT['q4_losers']['first5d_neg_share'] = [float((TR.loc[TR['bot5'], 'ret5d'] < 0).mean()), float((TR.loc[~TR['bot5'], 'ret5d'] < 0).mean())]
OUT['q4_losers']['first5d_auc_for_loser'] = auc(-TR.loc[TR['bot5'], 'ret5d'], -TR.loc[~TR['bot5'], 'ret5d'])
OUT['q4_winners_first5d_median'] = [float(TR.loc[TR['top5'], 'ret5d'].median()), float(TR.loc[~TR['top5'], 'ret5d'].median())]
print('4 虧損', f'{time.time() - t0:.0f}s', flush=True)

# ============================================================ 5 情境（PREREG）
def delta(nav, **kw):
    p = perf(nav)
    return dict(p, d_full_pp=(p['cagr'] - BASE['cagr']) * 100, d_early_pp=(p['cagr_early'] - BASE['cagr_early']) * 100, d_late_pp=(p['cagr_late'] - BASE['cagr_late']) * 100, **kw)


def block_run(y):
    blk = {}
    for j, a, b in zip(y['j'], y['ent_t'], y['exit_t']):
        blk.setdefault(int(j), []).append((int(a), int(b)))
    nav, lg, st = engine(sgc, block=blk)
    return nav


def sig_variant(sig_new):
    s2 = dict(sgc); s2['sig'] = sig_new
    return s2


desc_tr = lambda y: [f'{c}@{e}({r_:+.0%})' for c, e, r_ in zip(y['code'], y['ent_date'], y['ret'])]
sc = {}
by_pnl = TR.sort_values('pnl', ascending=False); by_ret = TR.sort_values('ret', ascending=False)
for m in (1, 3, 5, 10):
    y = by_pnl.head(m); sc[f'A 禁買 依損益金額前 {m} 筆'] = delta(block_run(y), removed=desc_tr(y))
for m in (1, 3, 5, 10):
    y = by_ret.head(m); sc[f'A 禁買 依報酬率前 {m} 筆'] = delta(block_run(y), removed=desc_tr(y))
for nm, m in (('5%', n5), ('10%', n10)):
    y = by_ret.head(m); sc[f'A 禁買 依報酬率前 {nm}（{m} 筆）'] = delta(block_run(y), removed=desc_tr(y.head(10)))
for nm, m in (('最大贏家（損益第 1）進場月整月不買', 1), ('損益前 3 大各自進場月整月不買', 3)):
    y = by_pnl.head(m); sig = sgc['sig'].copy(); sig[sorted(set(y['sig_t'])), :] = False
    nav, lg, st = engine(sig_variant(sig))
    got = [f"{c}:{'後來仍買到 ' + str(P.IDX[int(lg.loc[(lg['j'] == j) & (lg['ent_t'] > e) & (lg['ent_t'] <= x), 'ent_t'].iloc[0])].date()) if ((lg['j'] == j) & (lg['ent_t'] > e) & (lg['ent_t'] <= x)).any() else '整段沒買到'}" for c, j, e, x in zip(y['code'], y['j'], y['ent_t'], y['exit_t'])]
    sc['B ' + nm] = delta(nav, months=[str(P.IDX[k].date()) for k in sorted(set(y['sig_t']))], winners=got, n_trades=int(len(lg)))
for thr in (0.3, 0.5, 0.8):
    sig = sgc['sig'] & ~(P.RET20 > thr)
    nav, lg, st = engine(sig_variant(sig))
    drop = TR[TR['ret20'] > thr]
    sc[f'C 近 20 日漲幅 >{int(thr * 100)}% 不敢買'] = delta(nav, n_base_trades_dropped=int(len(drop)), dropped_top5=int(drop['top5'].sum()), dropped_bot5=int(drop['bot5'].sum()), dropped_mean_ret=float(drop['ret'].mean()),
                                                     kept_mean_ret=float(TR.loc[TR['ret20'] <= thr, 'ret'].mean()), n_trades=int(len(lg)))
OUT['q5_scenarios'] = sc
print('5 情境 A–C', f'{time.time() - t0:.0f}s', flush=True)

# 逐筆留一（描述）
loo = []
for r in TR.itertuples(index=False):
    nav, lg, st = engine(sgc, block={int(r.j): [(int(r.ent_t), int(r.exit_t))]})
    loo.append((K.cagr(nav, D0, DE) - BASE['cagr']) * 100)
TR['loo_d_pp'] = loo
TR[['code', 'ent_date', 'exit_date', 'ret', 'pnl', 'contrib', 'rk_ret', 'rk_pnl', 'loo_d_pp']].sort_values('loo_d_pp').to_csv(HERE / 'leave_one_out.csv', index=False)
lo = TR['loo_d_pp']
OUT['q5_leave_one_out'] = dict(n=int(len(lo)), min=float(lo.min()), p01=float(lo.quantile(.01)), p05=float(lo.quantile(.05)), median=float(lo.median()), p95=float(lo.quantile(.95)), max=float(lo.max()),
                               n_le_m0_5=int((lo <= -.5).sum()), n_le_m0_2=int((lo <= -.2).sum()), n_le_m0_1=int((lo <= -.1).sum()), n_ge_0_1=int((lo >= .1).sum()),
                               worst10=TR.sort_values('loo_d_pp').head(10)[['code', 'ent_date', 'ret', 'pnl', 'contrib', 'rk_ret', 'rk_pnl', 'loo_d_pp']].to_dict('records'),
                               best5=TR.sort_values('loo_d_pp', ascending=False).head(5)[['code', 'ent_date', 'ret', 'pnl', 'loo_d_pp']].to_dict('records'),
                               corr_contrib=float(np.corrcoef(TR['contrib'], -TR['loo_d_pp'])[0, 1]))
print('5 留一', f'{time.time() - t0:.0f}s', flush=True)

# D 隨機漏單
rng = np.random.default_rng(82)
karr = np.array(ks)
top3_k = list(by_pnl.head(3)['sig_t']); top10_k = list(by_pnl.head(10)['sig_t']); top5pct_k = TR.loc[TR['top5'], 'sig_t'].to_numpy()
buy_months = np.array(sorted(set(TR['sig_t'])))
sim_rows = []; q5d = {}
NDRAW = 1000
for p_ in (0.10, 0.25):
    res = []
    for it in range(NDRAW):
        skip = karr[rng.random(len(karr)) < p_]
        sig = sgc['sig'].copy(); sig[skip, :] = False
        nav, lg, st = engine(sig_variant(sig))
        ss = set(int(v) for v in skip)
        pf = perf(nav)
        res.append(dict(p=p_, it=it, n_skip=len(skip), n_skip_buy_months=int(np.isin(buy_months, skip).sum()), cagr=pf['cagr'], cagr_early=pf['cagr_early'], cagr_late=pf['cagr_late'], final=pf['final'],
                        miss_top1=int(top3_k[0] in ss), miss_top3=int(sum(k_ in ss for k_ in top3_k)), miss_top10=int(sum(k_ in ss for k_ in top10_k)), miss_top5pct=int(sum(int(k_) in ss for k_ in top5pct_k)),
                        n_trades=int(len(lg))))
    X = pd.DataFrame(res); sim_rows.append(X)
    c = X['cagr']
    o = dict(n=NDRAW, mean=float(c.mean()), min=float(c.min()), p05=float(c.quantile(.05)), p25=float(c.quantile(.25)), median=float(c.median()), p75=float(c.quantile(.75)), p95=float(c.quantile(.95)), max=float(c.max()),
             d_median_pp=float((c.median() - BASE['cagr']) * 100), d_mean_pp=float((c.mean() - BASE['cagr']) * 100), d_p05_pp=float((c.quantile(.05) - BASE['cagr']) * 100), d_p95_pp=float((c.quantile(.95) - BASE['cagr']) * 100),
             share_below_base=float((c < BASE['cagr']).mean()), share_below_0050=float((c < b_perf['cagr']).mean()), share_d_worse_2pp=float((c < BASE['cagr'] - .02).mean()), share_d_worse_4pp=float((c < BASE['cagr'] - .04).mean()),
             early=dict(median=float(X['cagr_early'].median()), d_median_pp=float((X['cagr_early'].median() - BASE['cagr_early']) * 100), p05=float(X['cagr_early'].quantile(.05)), p95=float(X['cagr_early'].quantile(.95))),
             late=dict(median=float(X['cagr_late'].median()), d_median_pp=float((X['cagr_late'].median() - BASE['cagr_late']) * 100), p05=float(X['cagr_late'].quantile(.05)), p95=float(X['cagr_late'].quantile(.95))),
             mean_months_skipped=float(X['n_skip'].mean()), n_signal_months=int(len(karr)), final_median=float(X['final'].median()),
             by_miss_top3={int(m): dict(n=int(len(g_)), median=float(g_['cagr'].median()), d_median_pp=float((g_['cagr'].median() - BASE['cagr']) * 100)) for m, g_ in X.groupby('miss_top3')},
             by_miss_top1={int(m): dict(n=int(len(g_)), median=float(g_['cagr'].median()), d_median_pp=float((g_['cagr'].median() - BASE['cagr']) * 100)) for m, g_ in X.groupby('miss_top1')})
    A = np.column_stack([np.ones(len(X)), X['n_skip'], X['miss_top3'], X['miss_top5pct']]); beta, *_ = np.linalg.lstsq(A, (X['cagr'] - BASE['cagr']) * 100, rcond=None)
    pred = A @ beta; yv = (X['cagr'] - BASE['cagr']) * 100
    o['ols_d_pp'] = dict(const=float(beta[0]), per_month_skipped=float(beta[1]), per_top3_month_missed=float(beta[2]), per_top5pct_trade_month_missed=float(beta[3]), r2=float(1 - ((yv - pred) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()))
    A1 = np.column_stack([np.ones(len(X)), X['n_skip']]); b1_, *_ = np.linalg.lstsq(A1, yv, rcond=None)
    o['ols_only_n_skip_r2'] = float(1 - ((yv - A1 @ b1_) ** 2).sum() / ((yv - yv.mean()) ** 2).sum())
    q5d[f'p={int(p_ * 100)}%'] = o
    print('5 隨機', p_, f'{time.time() - t0:.0f}s', flush=True)
pd.concat(sim_rows).to_csv(HERE / 'skip_sim.csv.gz', index=False)
OUT['q5_random_skip'] = q5d

# ============================================================ 6 環境或訊號
SUB = [('2008–2012', '2008-01-01', '2012-12-31'), ('2013–2015', '2013-01-01', '2015-12-31'), ('2016–2018', '2016-01-01', '2018-12-31'), ('2019–2023', '2019-01-01', '2023-12-31'), ('2024–2026', '2024-01-01', '2026-12-31'),
       ('2008–2018', '2008-01-01', '2018-12-31'), ('2019–2026', '2019-01-01', '2026-12-31')]
GR = ['univ', 'univ_ma', 'ctrl_mom', 'pool10', 'buylist', 'raw10', 'pool11_50']
mm = []
for i, d in enumerate(det):
    k, s, fin = d['k'], d['s'], d['fin']
    f3, f1 = F3[i], F1[i]
    if not np.isfinite(b3[i]):
        continue
    u = U[k] & np.isfinite(f3)
    grp = {}
    grp['univ'] = u
    grp['univ_ma'] = u & P.F_MA60[k]
    with np.errstate(invalid='ignore'):
        grp['ctrl_mom'] = u & P.F_MA60[k] & (P.RET20[k] > 0.10)        # 對照組：季線上＋近 20 日漲 >10%，不看營收
    for nm, idx in (('pool10', d['order'][:10]), ('buylist', d['tgt']), ('raw10', np.flatnonzero(fin)[np.argsort(-s[fin], kind='stable')][:10]), ('pool11_50', d['order'][10:50])):
        m_ = np.zeros(N, bool); m_[idx] = True; grp[nm] = m_ & np.isfinite(f3)
    grp['ctrl_mom_ex'] = grp['ctrl_mom'] & ~grp['buylist']
    r = dict(i=i, sig_date=str(P.IDX[k].date()), b3=float(b3[i]))
    for nm, m_ in grp.items():
        r[nm + '_n'] = int(m_.sum()); r[nm + '_big'] = int((f3[m_] > .5).sum()); r[nm + '_big100'] = int((f3[m_] > 1.0).sum()); r[nm + '_sum'] = float(f3[m_].sum())
        f1m = f1[m_]; r[nm + '_big1'] = int((f1m > .3).sum())
    r['xs_sd3'] = float(np.nanstd(f3[u])); r['xs_p95'] = float(np.nanquantile(f3[u], .95)); r['xs_p99'] = float(np.nanquantile(f3[u], .99))
    mm.append(r)
MM = pd.DataFrame(mm); MM.to_csv(HERE / 'env_monthly.csv', index=False)
GR2 = GR + ['ctrl_mom_ex']
rngb = np.random.default_rng(8206)


def rates(x):
    o = {}
    for g in GR2:
        nn = x[g + '_n'].sum()
        o[g] = dict(n=int(nn), n_per_month=float(nn / len(x)), big_rate=float(x[g + '_big'].sum() / nn) if nn else None, big100_rate=float(x[g + '_big100'].sum() / nn) if nn else None,
                    big1_rate=float(x[g + '_big1'].sum() / nn) if nn else None, mean_f3=float(x[g + '_sum'].sum() / nn) if nn else None, n_big=int(x[g + '_big'].sum()), n_big100=int(x[g + '_big100'].sum()))
    for g in ('buylist', 'pool10', 'raw10'):
        for c in ('univ', 'ctrl_mom_ex'):
            o[f'lift_{g}_over_{c}'] = o[g]['big_rate'] / o[c]['big_rate'] if o[c]['big_rate'] else None
            o[f'lift1_{g}_over_{c}'] = o[g]['big1_rate'] / o[c]['big1_rate'] if o[c]['big1_rate'] else None
    o['xs_sd3'] = float(x['xs_sd3'].mean()); o['xs_p95'] = float(x['xs_p95'].mean()); o['xs_p99'] = float(x['xs_p99'].mean()); o['b3_mean'] = float(x['b3'].mean()); o['n_months'] = int(len(x))
    return o


def lift_of(x, g, c, key='big'):
    a = x[g + '_' + key].sum() / max(x[g + '_n'].sum(), 1); b = x[c + '_' + key].sum() / max(x[c + '_n'].sum(), 1)
    return a / b if b > 0 else np.nan


env = {}
for nm, a, b in SUB:
    x = MM[(MM['sig_date'] >= a) & (MM['sig_date'] <= b)]
    o = rates(x)
    for g, c in (('buylist', 'univ'), ('buylist', 'ctrl_mom_ex'), ('pool10', 'univ')):
        bs = [lift_of(x.iloc[rngb.integers(0, len(x), len(x))], g, c) for _ in range(2000)]
        o[f'lift_{g}_over_{c}_ci90'] = [float(np.nanquantile(bs, .05)), float(np.nanquantile(bs, .95))]
    env[nm] = o
OUT['q6_env'] = env
# 兩段升力的比值（2019–2026 ÷ 2008–2018）與自助法信賴區間；以及差（買進名單大漲率 − 對照組大漲率）
xa = MM[MM['sig_date'] < '2019-01-01']; xb = MM[MM['sig_date'] >= '2019-01-01']
cmpo = {}
for g, c in (('buylist', 'univ'), ('buylist', 'ctrl_mom_ex'), ('pool10', 'univ'), ('raw10', 'univ')):
    for key in ('big', 'big1'):
        la, lb = lift_of(xa, g, c, key), lift_of(xb, g, c, key)
        bs = []
        for _ in range(4000):
            ya = xa.iloc[rngb.integers(0, len(xa), len(xa))]; yb = xb.iloc[rngb.integers(0, len(xb), len(xb))]
            bs.append(lift_of(yb, g, c, key) / lift_of(ya, g, c, key))
        bs = np.array(bs, float); bs = bs[np.isfinite(bs)]
        cmpo[f'{g}_over_{c}_{key}'] = dict(lift_2008_2018=float(la), lift_2019_2026=float(lb), ratio=float(lb / la), ratio_ci90=[float(np.quantile(bs, .05)), float(np.quantile(bs, .95))], p_ratio_gt1=float((bs > 1).mean()))
OUT['q6_lift_compare'] = cmpo
# 反事實：2019–2026 買進名單的大漲檔數，若升力停在 2008–2018 水準
ra, rb = rates(xa), rates(xb)
nb_ = rb['buylist']['n']
cf = {}
for c in ('univ', 'ctrl_mom_ex'):
    la = ra['buylist']['big_rate'] / ra[c]['big_rate']
    cf[c] = dict(actual_big=rb['buylist']['n_big'], if_early_rate_all=float(nb_ * ra['buylist']['big_rate']), if_late_base_early_lift=float(nb_ * rb[c]['big_rate'] * la),
                 early_buylist_rate=ra['buylist']['big_rate'], late_buylist_rate=rb['buylist']['big_rate'], early_base=ra[c]['big_rate'], late_base=rb[c]['big_rate'], early_lift=float(la), late_lift=float(rb['buylist']['big_rate'] / rb[c]['big_rate']))
    tot = rb['buylist']['n_big'] - cf[c]['if_early_rate_all']
    cf[c]['share_env'] = float((cf[c]['if_late_base_early_lift'] - cf[c]['if_early_rate_all']) / tot) if tot else None
    cf[c]['share_lift'] = float((rb['buylist']['n_big'] - cf[c]['if_late_base_early_lift']) / tot) if tot else None
OUT['q6_counterfactual'] = cf
# 股池每年的「飆股」比例：年底時過去 250 個交易日含息漲幅 >100%
yy = {}
for y_ in range(2008, 2027):
    idx = np.flatnonzero(P.IDX.year == y_)
    k = idx[-1]; k0 = max(k - 250, 0)
    u = U[k] & np.isfinite(P.G[k]) & np.isfinite(P.G[k0])
    r_ = P.G[k][u] / P.G[k0][u] - 1
    yy[y_] = dict(n=int(u.sum()), gt100=float((r_ > 1).mean()), gt50=float((r_ > .5).mean()), n_gt100=int((r_ > 1).sum()), median=float(np.median(r_)), e50=float(e50[k] / e50[k0] - 1))
OUT['q6_univ_yearly_doublers'] = yy
OUT['q6_trade_tail_by_era'] = {e_: dict(n=int(len(g_)), ge50=int((g_['ret'] >= .5).sum()), ge100=int((g_['ret'] >= 1).sum()), ge50_rate=float((g_['ret'] >= .5).mean())) for e_, g_ in TR.groupby('era')}
print('6 環境', f'{time.time() - t0:.0f}s', flush=True)

OUT['runtime_s'] = round(time.time() - t0, 1); OUT['peak_rss_mb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
(HERE / 'results.json').write_text(json.dumps(OUT, ensure_ascii=False, indent=1, default=jd))
print('done', OUT['runtime_s'], OUT['peak_rss_mb'])
