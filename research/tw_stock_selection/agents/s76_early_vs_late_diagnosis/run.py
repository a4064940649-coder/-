#!/usr/bin/env python3
"""S76 前段（2008–2015）對後段（2016–2026）診斷。python3 -I -B run.py → results.json＋csv。
全部用 V_BRIDGE 的 SURGE（S71 B2 校正；2008-02～2013-01 與 2016-02 起和現行凍結規則逐格相同）。只診斷，不設計新規則。"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import s76lib as Q
from s76lib import K, R, s10lib, cslib

HERE = Path(__file__).resolve().parent
t0 = time.time()
P, U = K.load_cont()
F_, RT = K.surge(P, 'B2')
D0, DE = P.IDX[P.t0], P.IDX[-1]
bench = pd.Series(P.E50, index=P.IDX).loc[D0:]
OUT = dict(task='S76 前段對後段診斷', split=Q.SPLIT, start=str(D0.date()), end=str(DE.date()))

# ============================================================ 0 驗收：重現 S75
sg0 = K.signal(P, U, RT, 0)
sgc = Q.build_sig(P, U, RT)
assert np.array_equal(sg0['sig'], sgc['sig']) and np.array_equal(sg0['X'], sgc['X']) and np.array_equal(np.nan_to_num(sg0['rank'], nan=-9), np.nan_to_num(sgc['rank'], nan=-9))
nav0, lg0, st0 = K.sim(P, sg0, U, 1e6)
nav0b, lg0b, _ = Q.engine(P, U, sgc)
s75 = json.load(open(Q.AG / 's75_continuous_2008_2026' / 'results.json'))['versions']['V_BRIDGE']
OUT['accept'] = dict(final_nav=float(nav0.iloc[-1]), s75_final_nav=s75['full']['final_nav'], cagr=K.cagr(nav0, D0, DE), s75_cagr=s75['full']['cagr'],
                     n_trades=int(len(lg0)), custom_builder_same=bool(np.allclose(nav0.to_numpy(), nav0b.to_numpy(), rtol=0, atol=0)))
assert abs(nav0.iloc[-1] - s75['full']['final_nav']) < 1e-6 and OUT['accept']['custom_builder_same']
print('驗收', OUT['accept'], flush=True)

info = sgc['info']; det = sgc['detail']
ks = [d['k'] for d in det]; sd = pd.to_datetime([i['sig_date'] for i in info]); n = len(ks)
early = np.asarray(sd < pd.Timestamp(Q.SPLIT)); late = ~early
masks = {nm: Q.era_mask(sd, a, b) for nm, a, b in Q.ERAS + Q.SUBS}
F1, b1 = Q.fwd_matrix(P, ks, 1)
F3, b3 = Q.fwd_matrix(P, ks, 3)

# ============================================================ 1 訊號本身的選股力
e50 = P.E50; e50ma = pd.Series(e50).rolling(60, min_periods=60).mean().to_numpy()
rows = []; held = []
for i, d in enumerate(det):
    k, s, fin, tgt = d['k'], d['s'], d['fin'], d['tgt']
    oraw = np.flatnonzero(fin)[np.argsort(-s[fin], kind='stable')]
    oma = d['order']
    f1, f3 = F1[i], F3[i]
    ew = np.nanmean(f1[fin]) if np.isfinite(f1[fin]).any() else np.nan
    r = dict(i=i, month=str(d['m']), sig_date=str(P.IDX[k].date()), k=k, b0050=b1[i], b0050_3=b3[i], ew_univ=ew,
             n_listed=int(((P.MK[k] == 1) & P.has[k]).sum()), n_univ=int(U[k].sum()), n_surge=int(fin.sum()), n_ma=int(len(oma)), n_top10ma_mom=int(len(tgt)),
             bull_sig=bool(e50[k] > e50ma[k]), up_month=bool(b1[i] > 0) if np.isfinite(b1[i]) else None,
             frac_above_ma=float(len(oma) / max(fin.sum(), 1)), surge_top10_min=float(s[oraw[9]]) if len(oraw) >= 10 else np.nan,
             surge_top10ma_min=float(s[oma[9]]) if len(oma) >= 10 else np.nan, n_surge_ge3=int((s[fin] >= 3).sum()),
             ic=Q.spearman(np.where(fin, s, np.nan), f1), ic3=Q.spearman(np.where(fin, s, np.nan), f3),
             ic_ma=Q.spearman(np.where(np.isin(np.arange(P.N), oma), s, np.nan), f1),
             univ_big3=float(np.nanmean(f3[fin] > 0.5)) if np.isfinite(f3[fin]).any() else np.nan,
             univ_xs_sd=float(np.nanstd(f1[fin])) if np.isfinite(f1[fin]).any() else np.nan)
    for nm, o in (('raw', oraw), ('ma', oma)):
        for N_ in (10, 20, 50):
            g = o[:N_]
            r[f'{nm}{N_}'] = np.nanmean(f1[g]) if len(g) and np.isfinite(f1[g]).any() else np.nan
            r[f'{nm}{N_}_3'] = np.nanmean(f3[g]) if len(g) and np.isfinite(f3[g]).any() else np.nan
            r[f'{nm}{N_}_hit'] = float(np.nanmean((f1[g] > b1[i])[np.isfinite(f1[g])])) if len(g) and np.isfinite(f1[g]).any() else np.nan
    r['raw10_big3'] = float(np.nanmean(f3[oraw[:10]] > 0.5)) if np.isfinite(f3[oraw[:10]]).any() else np.nan
    r['ma10_big3'] = float(np.nanmean(f3[oma[:10]] > 0.5)) if len(oma) and np.isfinite(f3[oma[:10]]).any() else np.nan
    q = len(oraw) // 10
    r['dec_top'] = np.nanmean(f1[oraw[:q]]); r['dec_bot'] = np.nanmean(f1[oraw[-q:]])
    # 零件：前 10（季線上）拆成有過／沒過 20 日漲幅 >10%
    p10 = oma[:10]; okm = P.RET20[k][p10] > 0.10
    r['ma10_mom'] = np.nanmean(f1[p10[okm]]) if okm.any() and np.isfinite(f1[p10[okm]]).any() else np.nan       # ＝正式買進名單
    r['ma10_nomom'] = np.nanmean(f1[p10[~okm]]) if (~okm).any() and np.isfinite(f1[p10[~okm]]).any() else np.nan
    r['ma10_mom_3'] = np.nanmean(f3[p10[okm]]) if okm.any() and np.isfinite(f3[p10[okm]]).any() else np.nan
    # 前 10（不看季線）拆成季線上／下
    q10 = oraw[:10]; abv = P.F_MA60[k][q10]
    r['raw10_above'] = np.nanmean(f1[q10[abv]]) if abv.any() and np.isfinite(f1[q10[abv]]).any() else np.nan
    r['raw10_below'] = np.nanmean(f1[q10[~abv]]) if (~abv).any() and np.isfinite(f1[q10[~abv]]).any() else np.nan
    r['raw10_n_above'] = int(abv.sum())
    r['ma10_mom_px_med'] = float(np.nanmedian(P.CF[k][p10[okm]])) if okm.any() else np.nan
    r['ma10_mom_n_px_gt70'] = int((P.CF[k][p10[okm]] > 70).sum())
    rows.append(r)
M = pd.DataFrame(rows)
# 持有名單（E0 續抱：留在季線上前 20 名；最多 10 檔；不計成本、等權）
H = []; hs = []
for i, d in enumerate(det):
    top20 = set(int(j) for j in d['order'][:20])
    H = [j for j in H if j in top20]
    for j in d['tgt']:
        if len(H) < 10 and int(j) not in H:
            H.append(int(j))
    f = F1[i][H] if H else np.array([])
    hs.append(dict(held_n=len(H), held=np.nanmean(f) if len(f) and np.isfinite(f).any() else np.nan))
M = pd.concat([M, pd.DataFrame(hs)], axis=1)
M.to_csv(HERE / 'signal_power_monthly.csv', index=False)

GROUPS = ['raw10', 'raw20', 'raw50', 'ma10', 'ma20', 'ma50', 'ma10_mom', 'ma10_nomom', 'raw10_above', 'raw10_below', 'held']
sp = {}; tab = []
for nm, m in masks.items():
    x = M[m]
    o = dict(n_months=int(m.sum()), ic=Q.desc(x['ic'], 1), ic3=Q.desc(x['ic3'], 1), ic_ma=Q.desc(x['ic_ma'], 1),
             dec_spread=Q.desc(x['dec_top'] - x['dec_bot']), ew_univ_vs_0050=Q.desc(x['ew_univ'] - x['b0050']), b0050_ann_pp=float(x['b0050'].mean() * 1200))
    for g in GROUPS:
        o[g + '_vs0050'] = Q.desc(x[g] - x['b0050']); o[g + '_vsEW'] = Q.desc(x[g] - x['ew_univ'])
        tab.append(dict(period=nm, group=g, **{'vs0050_' + k: v for k, v in o[g + '_vs0050'].items()}, **{'vsEW_' + k: v for k, v in o[g + '_vsEW'].items()}))
    for g in ('raw10', 'raw20', 'raw50', 'ma10', 'ma20', 'ma50'):
        o[g + '_3m_vs0050'] = Q.desc(x[g + '_3'] - x['b0050_3'], 4); o[g + '_stock_hit'] = float(x[g + '_hit'].mean())
    o['ma10_mom_3m_vs0050'] = Q.desc(x['ma10_mom_3'] - x['b0050_3'], 4)
    o['big3'] = dict(univ=float(x['univ_big3'].mean()), raw10=float(x['raw10_big3'].mean()), ma10=float(x['ma10_big3'].mean()))
    o['counts'] = {c: float(x[c].mean()) for c in ('n_listed', 'n_univ', 'n_surge', 'n_ma', 'n_top10ma_mom', 'held_n', 'frac_above_ma', 'surge_top10_min', 'surge_top10ma_min', 'n_surge_ge3', 'raw10_n_above',
                                                      'univ_xs_sd', 'ma10_mom_px_med', 'ma10_mom_n_px_gt70')}
    o['counts']['coverage_surge_over_univ'] = float((x['n_surge'] / x['n_univ']).mean()); o['counts']['univ_over_listed'] = float((x['n_univ'] / x['n_listed']).mean())
    o['counts']['months_zero_target'] = int((x['n_top10ma_mom'] == 0).sum())
    o['regime'] = dict(bull_sig_share=float(x['bull_sig'].mean()), up_month_share=float(x['up_month'].dropna().astype(bool).mean()))
    sp[nm] = o
e_, l_ = M[early], M[late]
sp['diff_late_minus_early'] = {g: Q.welch(e_[g] - e_['b0050'], l_[g] - l_['b0050']) for g in GROUPS}
sp['diff_late_minus_early_vsEW'] = {g: Q.welch(e_[g] - e_['ew_univ'], l_[g] - l_['ew_univ']) for g in GROUPS}
sp['diff_late_minus_early']['ic'] = dict(diff=float(l_['ic'].mean() - e_['ic'].mean()), t=Q.welch(e_['ic'], l_['ic'])['t'])
OUT['signal_power'] = sp
pd.DataFrame(tab).to_csv(HERE / 'signal_power_by_period.csv', index=False)
# 逐年
yr = []
for y, x in M.groupby(pd.to_datetime(M['sig_date']).dt.year):
    yr.append(dict(year=int(y), n=len(x), ic=float(x['ic'].mean()), **{g + '_vs0050_ann_pp': float((x[g] - x['b0050']).mean() * 1200) for g in ('raw10', 'raw50', 'ma10', 'ma10_mom', 'held')},
                   ew_vs0050_ann_pp=float((x['ew_univ'] - x['b0050']).mean() * 1200), n_tgt=float(x['n_top10ma_mom'].mean()), bull_share=float(x['bull_sig'].mean()),
                   frac_above_ma=float(x['frac_above_ma'].mean()), n_univ=float(x['n_univ'].mean()), coverage=float((x['n_surge'] / x['n_univ']).mean())))
pd.DataFrame(yr).to_csv(HERE / 'signal_power_yearly.csv', index=False)
OUT['signal_power_yearly'] = yr
print('1 訊號力', f'{time.time() - t0:.0f}s', flush=True)

# ---- 市場狀態條件（訊號月層級）
reg = {}
for nm, m in (('前段', early), ('後段', late)):
    x = M[m & M['b0050'].notna().to_numpy()]
    o = {}
    for rn, c in (('bull_sig（訊號日 0050 在 60 日線上）', x['bull_sig'].astype(bool)), ('bear_sig', ~x['bull_sig'].astype(bool)), ('up_month（當期 0050 上漲）', x['up_month'].astype(bool)), ('down_month', ~x['up_month'].astype(bool))):
        y = x[c.to_numpy()]
        o[rn] = dict(n=int(len(y)), share=float(len(y) / len(x)), n_tgt=float(y['n_top10ma_mom'].mean()), frac_above_ma=float(y['frac_above_ma'].mean()),
                     **{g: Q.desc(y[g] - y['b0050']) for g in ('raw10', 'ma10', 'ma10_mom', 'held')}, ic=float(y['ic'].mean()))
    reg[nm] = o
OUT['regime_signal_level'] = reg

# ============================================================ 2 拆零件（引擎；同一個連續帳戶）
D_SPLIT = P.IDX[P.IDX < pd.Timestamp(Q.SPLIT)][-1]
CUTS = {'前段 2008–2015': (D0, D_SPLIT), '後段 2016–2026': (D_SPLIT, DE), '全期': (D0, DE),
        '2008–2012': (D0, K.last_before(P.IDX, '2013-01-01')), '2013–2015': (K.last_before(P.IDX, '2013-01-01'), D_SPLIT),
        '2016–2018': (D_SPLIT, K.last_before(P.IDX, '2019-01-01')), '2019–2026': (K.last_before(P.IDX, '2019-01-01'), DE)}


def acct(nav, lg, st, label):
    b = bench.reindex(nav.index)
    rs, rb = R.monthly_returns(nav, b)
    o = dict(label=label)
    for nm, (a, e) in CUTS.items():
        c1, c2 = K.cagr(nav, a, e), K.cagr(b, a, e)
        mm = (rs.index > a.to_period('M')) & (rs.index <= e.to_period('M')) if a != D0 else (rs.index <= e.to_period('M'))
        blk = R.stats_block(rs[mm], rb[mm])
        sf = st['stockfrac'].loc[a:e].dropna()
        o[nm] = dict(cagr=c1, cagr_0050=c2, diff_pp=(c1 - c2) * 100, ex_ann_pp=blk['ex_ann_pp'], t_nw6=blk['t_nw6'], te_pp=blk['te_ann_pp'], stockfrac=float(sf.mean()), maxdd=R.maxdd(nav.loc[a:e]))
    x = lg[lg['why'] != 'open_end']
    for nm, mk in (('前段 2008–2015', x['ent_date'] < Q.SPLIT), ('後段 2016–2026', x['ent_date'] >= Q.SPLIT)):
        y = x[mk]
        o[nm].update(n_trades=int(len(y)), trade_mean=float(y['ret'].mean()) if len(y) else None, trade_win=float((y['ret'] > 0).mean()) if len(y) else None,
                     avg_npos=float(K.npos_series(P, lg).loc[CUTS[nm][0]:CUTS[nm][1]].iloc[1:].mean()))
    return o


FR = dict(sat_w=1.0, cost='zero', capital=1e10, cap_pct=None)             # 無摩擦：不計成本、資金夠大（整張限制與 5% 成交值上限都不綁）
sgA = Q.build_sig(P, U, RT, use_ma=False, use_mom=False, keep_n=10)
sgB = Q.build_sig(P, U, RT, use_ma=True, use_mom=False, keep_n=10)
sgC = Q.build_sig(P, U, RT, use_ma=True, use_mom=True, keep_n=10)
LAD = [('L1 純 SURGE 前 10 名（每檔 10%、每月換、無成本）', sgA, FR),
       ('L2 ＋60 日線過濾', sgB, FR),
       ('L3 ＋近 20 日漲幅>10%（沒過的名額放 0050）', sgC, FR),
       ('L4 ＋出場規則（留在前 20 名續抱）', sgc, FR),
       ('L5 ＋權重 W10_70（每檔 7%、底倉 30% 0050）', sgc, dict(FR, sat_w=0.7)),
       ('L6 ＋嚴格成本（資金仍夠大）', sgc, dict(FR, sat_w=0.7, cost='strict')),
       ('L7 ＋100 萬實際帳戶（整張、5% 成交值上限）＝S75', sgc, dict(sat_w=0.7, cost='strict', capital=1e6, cap_pct=0.05))]
lad = []; NAVS = {}
for label, sg_, kw in LAD:
    nav, lg, st = Q.engine(P, U, sg_, **kw)
    lad.append(acct(nav, lg, st, label)); NAVS[label[:2]] = nav
OUT['ladder'] = lad
lrow = []
for i, o in enumerate(lad):
    r = dict(step=o['label'])
    for nm in CUTS:
        r[nm + '_cagr'] = o[nm]['cagr'] * 100; r[nm + '_vs0050_pp'] = o[nm]['diff_pp']; r[nm + '_stockfrac'] = o[nm]['stockfrac']
        r[nm + '_step_pp'] = (o[nm]['diff_pp'] - lad[i - 1][nm]['diff_pp']) if i else o[nm]['diff_pp']
        r[nm + '_ex_ann_pp'] = o[nm]['ex_ann_pp']; r[nm + '_t'] = o[nm]['t_nw6']
    for nm in ('前段 2008–2015', '後段 2016–2026'):
        r[nm + '_npos'] = o[nm]['avg_npos']; r[nm + '_trade_mean'] = o[nm]['trade_mean']; r[nm + '_trade_win'] = o[nm]['trade_win']; r[nm + '_n_trades'] = o[nm]['n_trades']
    lrow.append(r)
pd.DataFrame(lrow).to_csv(HERE / 'ladder.csv', index=False)
pd.DataFrame(NAVS).assign(e0050=bench.reindex(nav0.index) / bench.iloc[0] * 1e6).round(2).to_csv(HERE / 'ladder_nav.csv.gz')
# 零件換順序（檢查順序相依）：先加動能再加季線
alt = []
for label, kw_sig in (('純 SURGE＋20 日漲幅>10%（不看季線）', dict(use_ma=False, use_mom=True, keep_n=10)),
                      ('純 SURGE 前 10，續抱到跌出前 20（不看季線、不看動能）', dict(use_ma=False, use_mom=False, keep_n=20)),
                      ('季線＋續抱前 20（不看動能）', dict(use_ma=True, use_mom=False, keep_n=20))):
    nav, lg, st = Q.engine(P, U, Q.build_sig(P, U, RT, **kw_sig), **FR)
    alt.append(acct(nav, lg, st, label))
OUT['ladder_alt_order'] = alt
print('2 拆零件', f'{time.time() - t0:.0f}s', flush=True)

# ============================================================ 3 環境
env = {}
base = acct(nav0, lg0, st0, '現行（S75 V_BRIDGE，100 萬）')
env['base'] = base


def delta(o):
    return {nm: dict(cagr=o[nm]['cagr'], diff_pp=o[nm]['diff_pp'], delta_vs_base_pp=(o[nm]['cagr'] - base[nm]['cagr']) * 100, stockfrac=o[nm]['stockfrac']) for nm in CUTS}


# 3a 漲跌停 7%（2015-06-01 前）
cut = int(np.searchsorted(P.IDX.values, np.datetime64('2015-06-01'), side='left'))
up0, dn0 = P.LIMUP.copy(), P.LIMDN.copy()
ref = P.REF[:cut]
tk = cslib.tick(ref * 1.07); P.LIMUP[:cut] = np.floor(ref * 1.07 / tk + 1e-9) * tk
tk = cslib.tick(ref * 0.93); P.LIMDN[:cut] = np.ceil(ref * 0.93 / tk - 1e-9) * tk
nav, lg, st = K.sim(P, sg0, U, 1e6)
env['limit7'] = dict(delta(acct(nav, lg, st, '2015-06 前改用 7% 漲跌停')), blk_limup=int(st['blk_limup']), sell_defer_limdn=int(st['sell_defer_limdn']), base_blk_limup=int(st0['blk_limup']))
P.LIMUP, P.LIMDN = up0, dn0
# 3b 整張限制／資金規模
odd0 = cslib.ODD_DATE
cslib.ODD_DATE = pd.Timestamp('2000-01-01')
nav, lg, st = K.sim(P, sg0, U, 1e6)
env['odd_lot_all_period_1m'] = dict(delta(acct(nav, lg, st, '全期允許零股，100 萬')), blk_lot=int(st['blk_lot']), n_buy=int(st['n_buy']))
cslib.ODD_DATE = odd0
for cap in (3e6, 1e7):
    nav, lg, st = K.sim(P, sg0, U, cap)
    env[f'capital_{int(cap / 1e4)}w'] = dict(delta(acct(nav, lg, st, f'本金 {int(cap / 1e4)} 萬')), blk_lot=int(st['blk_lot']), blk_cap=int(st['blk_cap']), n_buy=int(st['n_buy']))
env['base_blk'] = {k: int(st0[k]) for k in ('n_sig', 'n_buy', 'blk_limup', 'blk_noopen', 'blk_lot', 'blk_cap', 'blk_nocash', 'blk_min', 'noslot')}
# 被擋掉的買進（現行帳戶）：名單上、沒持有、也沒買到
x = lg0.copy(); bl = []
k2i = {k: i for i, k in enumerate(ks)}
for i, d in enumerate(det):
    k = d['k']
    for j in d['tgt']:
        j = int(j)
        bought = ((x['j'] == j) & (x['sig_t'] == k)).any()
        heldnow = ((x['j'] == j) & (x['ent_t'] <= k) & (x['exit_t'] >= k + 1)).any()
        if not heldnow:
            bl.append(dict(i=i, sig_date=str(P.IDX[k].date()), code=str(P.COLS[j]), bought=bool(bought), px=float(P.CF[k, j]), fwd1=float(F1[i, j]), ex1=float(F1[i, j] - b1[i]), ex3=float(F3[i, j] - b3[i])))
BL = pd.DataFrame(bl); BL['early'] = pd.to_datetime(BL['sig_date']) < pd.Timestamp(Q.SPLIT)
BL.to_csv(HERE / 'new_targets_bought_or_blocked.csv', index=False)
env['blocked'] = {}
for nm, mk in (('前段', BL['early']), ('後段', ~BL['early']), ('2016-01～2020-10（整張時代的後段）', (~BL['early']) & (BL['sig_date'] < '2020-10-26'))):
    y = BL[mk]
    env['blocked'][nm] = dict(n_new_targets=int(len(y)), n_bought=int(y['bought'].sum()), n_blocked=int((~y['bought']).sum()), blocked_share=float((~y['bought']).mean()),
                              bought_ex1_mean_pct=float(y.loc[y['bought'], 'ex1'].mean() * 100), blocked_ex1_mean_pct=float(y.loc[~y['bought'], 'ex1'].mean() * 100) if (~y['bought']).any() else None,
                              bought_ex3_mean_pct=float(y.loc[y['bought'], 'ex3'].mean() * 100), blocked_ex3_mean_pct=float(y.loc[~y['bought'], 'ex3'].mean() * 100) if (~y['bought']).any() else None,
                              bought_px_median=float(y.loc[y['bought'], 'px'].median()), blocked_px_median=float(y.loc[~y['bought'], 'px'].median()) if (~y['bought']).any() else None)
OUT['env'] = env
print('3 環境', f'{time.time() - t0:.0f}s', flush=True)

# ============================================================ 3c 產業集中度＋去掉大贏家
IND, INAMES = Q.load_industry(P)
x = lg0[lg0['why'] != 'open_end'].copy()
x['ind'] = [INAMES[IND[t, j]] if IND[t, j] >= 0 else '缺' for t, j in zip(x['sig_t'], x['j'])]
x['elec'] = x['ind'].isin(Q.ELEC)
x['b'] = [P.E50O[b_] / P.E50O[a_] - 1 for a_, b_ in zip(x['ent_t'], x['exit_t'])]
x['ex'] = x['ret'] - x['b']
x['early'] = x['ent_date'] < Q.SPLIT
x.drop(columns=['j']).to_csv(HERE / 'trades_tagged.csv.gz', index=False)
ind = {}
for nm, mk in (('前段', x['early']), ('後段', ~x['early']), ('2016–2018', (~x['early']) & (x['ent_date'] < '2019-01-01')), ('2019–2026', x['ent_date'] >= '2019-01-01')):
    y = x[mk]; g = y.groupby('ind').agg(n=('ret', 'size'), pnl=('pnl', 'sum'), ex_mean=('ex', 'mean')).sort_values('pnl', ascending=False)
    tot = y['pnl'].sum(); pos_tot = y.loc[y['pnl'] > 0, 'pnl'].sum()
    srt = y.sort_values('pnl', ascending=False); srr = y.sort_values('ret', ascending=False); n5 = max(1, int(round(len(y) * 0.05)))
    ind[nm] = dict(n=int(len(y)), pnl_total=float(tot), trade_ret_mean=float(y['ret'].mean()), trade_ex_mean=float(y['ex'].mean()), trade_ex_median=float(y['ex'].median()), trade_ex_win=float((y['ex'] > 0).mean()),
                   trade_ex_t=Q.tstat(y['ex']), ret_p90=float(y['ret'].quantile(0.9)), ret_p95=float(y['ret'].quantile(0.95)), ret_max=float(y['ret'].max()),
                   share_ret_gt50=float((y['ret'] > 0.5).mean()), share_ret_gt100=float((y['ret'] > 1.0).mean()),
                   elec_n_share=float(y['elec'].mean()), elec_pnl_share=float(y.loc[y['elec'], 'pnl'].sum() / tot) if tot else None,
                   elec_ex_mean=float(y.loc[y['elec'], 'ex'].mean()), nonelec_ex_mean=float(y.loc[~y['elec'], 'ex'].mean()), nonelec_n=int((~y['elec']).sum()),
                   elec_ex_t=Q.tstat(y.loc[y['elec'], 'ex']), nonelec_ex_t=Q.tstat(y.loc[~y['elec'], 'ex']),
                   semi_n_share=float((y['ind'] == '半導體業').mean()), semi_pnl_share=float(y.loc[y['ind'] == '半導體業', 'pnl'].sum() / tot) if tot else None,
                   top_industries=[dict(ind=k_, n=int(v['n']), pnl_share=float(v['pnl'] / tot) if tot else None, ex_mean=float(v['ex_mean'])) for k_, v in g.head(6).iterrows()],
                   top5_pnl_share_of_total=float(srt['pnl'].head(5).sum() / tot) if tot else None, top5_pnl_share_of_gross_gain=float(srt['pnl'].head(5).sum() / pos_tot),
                   top5=[dict(code=str(c), ent=str(e.date()), ret=float(r_), pnl=float(p_), ind=i_) for c, e, r_, p_, i_ in zip(srt['code'].head(5), srt['ent_date'].head(5), srt['ret'].head(5), srt['pnl'].head(5), srt['ind'].head(5))],
                   trade_ret_mean_ex_top5pct=float(srr['ret'].iloc[n5:].mean()), trade_ex_mean_ex_top5pct=float(y.sort_values('ex', ascending=False)['ex'].iloc[n5:].mean()), n_top5pct=n5,
                   trade_ex_mean_ex_top5=float(y.sort_values('ex', ascending=False)['ex'].iloc[5:].mean()))
OUT['industry_and_winners'] = ind
# 逐年電子占比
OUT['elec_share_by_year'] = {int(y_): dict(n=int(len(g_)), elec=float(g_['elec'].mean()), semi=float((g_['ind'] == '半導體業').mean())) for y_, g_ in x.groupby(x['ent_date'].dt.year)}


def block_run(rows_):
    blk = {}
    for j, a, b in zip(rows_['j'], rows_['ent_t'], rows_['exit_t']):
        blk.setdefault(int(j), []).append((int(a), int(b)))
    nav, lg, st = Q.engine(P, U, sgc, block=blk)
    return delta(acct(nav, lg, st, 'block'))


xe, xl = x[x['early']], x[~x['early']]
rem = {}
for nm, pick in (('去掉前段前 5 大（依損益金額）', xe.sort_values('pnl', ascending=False).head(5)), ('去掉後段前 5 大（依損益金額）', xl.sort_values('pnl', ascending=False).head(5)),
                 ('去掉前段前 5 大（依報酬率）', xe.sort_values('ret', ascending=False).head(5)), ('去掉後段前 5 大（依報酬率）', xl.sort_values('ret', ascending=False).head(5)),
                 ('去掉前段前 5%（11 筆，依報酬率）', xe.sort_values('ret', ascending=False).head(int(round(len(xe) * 0.05)))), ('去掉後段前 5%（20 筆，依報酬率）', xl.sort_values('ret', ascending=False).head(int(round(len(xl) * 0.05)))),
                 ('去掉後段全部半導體業', xl[xl['ind'] == '半導體業']), ('去掉後段全部電子八類', xl[xl['elec']]), ('去掉前段全部電子八類', xe[xe['elec']])):
    rem[nm] = dict(block_run(pick), n_removed=int(len(pick)), removed=[f'{c}@{str(e.date())}({r_:+.0%})' for c, e, r_ in zip(pick['code'].head(8), pick['ent_date'].head(8), pick['ret'].head(8))])
OUT['remove_winners'] = rem
print('3c 產業與大贏家', f'{time.time() - t0:.0f}s', flush=True)

# ============================================================ 4 逐年＋市場狀態（帳戶層級）
b = bench.reindex(nav0.index)
rs, rb = R.monthly_returns(nav0, b)
ex = rs - rb
e50s = pd.Series(P.E50, index=P.IDX); ma = e50s.rolling(60, min_periods=60).mean()
above = (e50s > ma)
me = above.groupby(above.index.to_period('M')).last()
prev_above = me.shift(1).reindex(ex.index)
sf = st0['stockfrac'].groupby(st0['stockfrac'].index.to_period('M')).mean().reindex(ex.index)
MM = pd.DataFrame(dict(strat=rs, e0050=rb, ex=ex, up=rb > 0, prev_above=prev_above, stockfrac=sf))
MM.to_csv(HERE / 'account_monthly.csv')
acc = {}
for nm, mk in (('前段 2008–2015', MM.index < pd.Period('2016-01', 'M')), ('後段 2016–2026', MM.index >= pd.Period('2016-01', 'M'))):
    y = MM[mk]
    o = dict(all=Q.desc(y['ex']), up=Q.desc(y.loc[y['up'], 'ex']), down=Q.desc(y.loc[~y['up'], 'ex']),
             prev_above=Q.desc(y.loc[y['prev_above'] == True, 'ex']), prev_below=Q.desc(y.loc[y['prev_above'] == False, 'ex']),   # noqa: E712
             up_share=float(y['up'].mean()), prev_above_share=float((y['prev_above'] == True).mean()), stockfrac=float(y['stockfrac'].mean()),   # noqa: E712
             stockfrac_up=float(y.loc[y['up'], 'stockfrac'].mean()), stockfrac_down=float(y.loc[~y['up'], 'stockfrac'].mean()),
             e0050_ann_pp=float(y['e0050'].mean() * 1200), ex_per_unit_stock_ann_pp=float(y['ex'].mean() / y['stockfrac'].mean() * 1200))
    acc[nm] = o
# 重新加權：前段若有後段的多頭月份比例
E_, L_ = acc['前段 2008–2015'], acc['後段 2016–2026']
acc['reweight'] = dict(
    early_with_late_up_share_ann_pp=float((L_['up_share'] * E_['up']['mean_pct'] + (1 - L_['up_share']) * E_['down']['mean_pct']) * 12),
    late_with_early_up_share_ann_pp=float((E_['up_share'] * L_['up']['mean_pct'] + (1 - E_['up_share']) * L_['down']['mean_pct']) * 12),
    early_with_late_prev_above_share_ann_pp=float((L_['prev_above_share'] * E_['prev_above']['mean_pct'] + (1 - L_['prev_above_share']) * E_['prev_below']['mean_pct']) * 12),
    late_with_early_prev_above_share_ann_pp=float((E_['prev_above_share'] * L_['prev_above']['mean_pct'] + (1 - E_['prev_above_share']) * L_['prev_below']['mean_pct']) * 12))
OUT['account_regime'] = acc
yt = K.yearly_table(nav0, b)
ysf = st0['stockfrac'].groupby(st0['stockfrac'].index.year).mean()
ynp = K.npos_series(P, lg0).loc[D0:].groupby(lambda d: d.year).mean()
xx = lg0[lg0['why'] != 'open_end']
for r in yt:
    y = r['year']; t_ = x[x['ent_date'].dt.year == y]
    r.update(stockfrac=float(ysf.get(y, np.nan)), npos=float(ynp.get(y, np.nan)), n_trades=int(len(t_)), trade_ret_mean=float(t_['ret'].mean()) if len(t_) else None,
             trade_ex_mean=float(t_['ex'].mean()) if len(t_) else None, trade_win=float((t_['ret'] > 0).mean()) if len(t_) else None, pnl=float(t_['pnl'].sum()),
             up_month_share=float(MM.loc[MM.index.year == y, 'up'].mean()), prev_above_share=float((MM.loc[MM.index.year == y, 'prev_above'] == True).mean()))   # noqa: E712
OUT['yearly'] = yt
pd.DataFrame(yt).to_csv(HERE / 'yearly_diag.csv', index=False)

OUT['runtime_s'] = round(time.time() - t0, 1)
(HERE / 'results.json').write_text(json.dumps(OUT, ensure_ascii=False, indent=1, default=lambda v: None if (isinstance(v, float) and v != v) else (v.item() if hasattr(v, 'item') else str(v))))
print('done', OUT['runtime_s'])
