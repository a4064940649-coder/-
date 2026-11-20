#!/usr/bin/env python3
"""R75 稽核 (b)(c)(d)：在自建連續面板（cache_full）上重跑兩版，另做診斷用的對照跑（不是成績單數字）。
(b) 2008 年逐月持股、股票比重、帳戶與 0050 月報酬；起跑日效果；每個訊號日的股池大小與買不到原因；持股日的資料缺漏；逐筆交易對 data_panel 還原價交叉核對。
(c) 接縫（2014-12-31 → 2015-01-05）：兩版接縫日持股逐檔前後 5 個交易日的收盤／開盤／漲跌價差，對 pre 來源（Asoul 長表，2015 上半年重疊）與 data_panel（第三來源）核對；
    持股日報酬超過漲跌停幅度的格子（假跳動掃描）。
(d) 兩版 2016 年以後差距的來源：逐年差、交易集合差、對照跑（全期可買零股；2015-12-31 以兩版當時淨值全現金重新起跑）。"""
import sys, json, time, resource, gc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import r75lib as Q
from r75lib import R, L, stlib, s17lib, s10lib, cslib, s71lib
import numpy as np, pandas as pd

t0_ = time.time()
CN = 'cache_full'
Q.setup(CN, '2008-01-02', '2008-01')
P, U, fix = Q.load_all()
for a_ in ('H', 'L', 'VOL', 'HF', 'VOLF'):
    if hasattr(P, a_):
        delattr(P, a_)
gc.collect()
IDX = P.IDX; D0, DE = IDX[P.t0], IDX[-1]
bench = pd.Series(P.E50, index=IDX)
lastb = lambda d: IDX[IDX < pd.Timestamp(d)][-1]
row = lambda d: int(np.searchsorted(IDX.values, np.datetime64(pd.Timestamp(d)), side='left'))
out = {}
FS = {r: s71lib.rev_rule(P, r, '2008-01') for r in ('B2', 'MASK')}
SG = {r: s10lib.make_signal_s10(P, U, dict(S=FS[r]['S']['STD'], av=FS[r]['av']), shift=0) for r in FS}
RUN = {r: L.sim(P, SG[r], U, cap=1e6) for r in FS}
VER = {'B2': 'V_BRIDGE', 'MASK': 'V_MASK'}

# ============================================================ (b) 2008
nav, lg, st = RUN['B2']
b = {}
sf = st['stockfrac']
rows = []
for m in pd.period_range('2008-01', '2009-06', freq='M'):
    d = IDX[(IDX >= m.to_timestamp()) & (IDX <= m.to_timestamp(how='end'))]
    d = d[d >= D0]
    base = IDX[IDX < d[0]][-1] if d[0] > D0 else D0
    held = lg[(lg['ent_date'] <= d[-1]) & (lg['exit_date'] > d[-1])]
    rows.append(dict(month=str(m), nav_end=round(float(nav[d[-1]])), ret_nav=round(float(nav[d[-1]] / nav[base] - 1) * 100, 2), ret_0050=round(float(bench[d[-1]] / bench[base] - 1) * 100, 2),
                     n_hold_end=int(len(held)), codes_end=' '.join(held['code'].astype(str)), stockfrac_end=round(float(sf[d[-1]]) * 100, 1), stockfrac_avg=round(float(sf[d].mean()) * 100, 1)))
b['monthly'] = rows
# 起跑日：帳戶 2008-01-02 是現金、01-03 才買進 0050；基準 01-02 收盤就持有
d1 = IDX[P.t0 + 1]
b['start_effect'] = dict(first_day_after=str(d1.date()), nav_day1=float(nav[d1]), e50_day1_ret=float(bench[d1] / bench[D0] - 1),
                         y2008_nav=float(nav[lastb('2009-01-01')] / nav[D0] - 1), y2008_0050=float(bench[lastb('2009-01-01')] / bench[D0] - 1),
                         y2008_0050_from_day1=float(bench[lastb('2009-01-01')] / bench[d1] - 1), y2008_nav_from_day1=float(nav[lastb('2009-01-01')] / nav[d1] - 1))
# 分解：日報酬 = w(t-1)×股票袖珍報酬 + (1-w)×0050 → 反推股票部位的報酬貢獻
y = nav.loc[D0:lastb('2009-01-01')]; rb_ = bench.reindex(y.index).pct_change(); rn = y.pct_change(); w = sf.reindex(y.index).shift(1).fillna(0)
contrib_stock = (rn - (1 - w) * rb_).dropna()        # 股票部位（含全部交易成本、0050 買賣成本）對帳戶日報酬的貢獻
b['decomp_2008'] = dict(sum_log_nav=float(np.log1p(rn.dropna()).sum()), sum_stock_contrib=float(contrib_stock.sum()), sum_0050_contrib=float(((1 - w) * rb_).dropna().sum()),
                        avg_stockfrac=float(sf.reindex(y.index).iloc[1:].mean()), stock_sleeve_vs_0050_note='股票貢獻 − w×0050 ＝選股超額貢獻',
                        sum_w_times_0050=float((w * rb_).dropna().sum()))
# 每個訊號日（2008-02～2009-01）：股池、SURGE、名單、實際買進
S = FS['B2']['S']['STD']
sigtab = []
for x in SG['B2']['info']:
    if x['sig_date'] > '2009-02-28':
        break
    k = x['k']; s = S.loc[pd.Period(x['month'], 'M')].to_numpy()
    bought = lg[lg['sig_t'] == k]
    px = {c: float(P.O[k + 1, P.ci[c]]) for c in x['targets']}
    sigtab.append(dict(month=x['month'], sig_date=x['sig_date'], n_univ=int(U[k].sum()), n_listed_twse=int((P.MK[k] == 1).sum()), n_surge=int((U[k] & np.isfinite(s)).sum()),
                       n_ma60=x['n_ma'], n_tgt=x['n_tgt'], targets=x['targets'], open_next=px, n_bought=int(len(bought)), bought=list(bought['code'].astype(str)),
                       slot_budget=round(float(0.7 * nav.iloc[nav.index.get_loc(IDX[k])] / 10))))
b['signals_2008'] = sigtab
# 2008 交易
x8 = lg[lg['ent_date'] < '2009-01-01'].copy()
b['trades_2008'] = [dict(code=str(r.code), ent=str(r.ent_date.date()), exit=str(r.exit_date.date()), shares=int(r.shares), cost=round(float(r.cost)), ret=round(float(r.ret) * 100, 2), pnl=round(float(r.pnl))) for r in x8.itertuples()]
b['trades_2008_sum'] = dict(n=int(len(x8)), pnl=float(x8['pnl'].sum()), win=float((x8['ret'] > 0).mean()), avg_ret=float(x8['ret'].mean()))
# 持股日資料缺漏（全期 pre-2015）
hole = dict(pos_days=0, no_trade_days=0, zero_ret_days=0)
big = []
for vr in ('B2', 'MASK'):
    for r in RUN[vr][1].itertuples():
        j = P.ci[str(r.code)]
        for t in range(r.ent_t + 1, r.exit_t):
            lim = 0.0705 if IDX[t] < pd.Timestamp('2015-06-01') else 0.1005
            if abs(P.R[t, j]) > lim:
                big.append(dict(ver=VER[vr], code=str(r.code), date=str(IDX[t].date()), R=round(float(P.R[t, j]) * 100, 2), C=float(P.C[t, j]), prevC=float(P.CF[t - 1, j]), CHG=float(P.CHG[t, j]) if np.isfinite(P.CHG[t, j]) else None))
            if vr == 'B2' and IDX[t] < pd.Timestamp('2015-01-05'):
                hole['pos_days'] += 1; hole['no_trade_days'] += int(not P.has[t, j]); hole['zero_ret_days'] += int(P.R[t, j] == 0)
b['holding_data_holes_pre2015'] = hole
seen = set(); big2 = []
for z in big:
    key = (z['code'], z['date'])
    z2 = dict(z); z2.pop('ver')
    if key not in seen:
        seen.add(key); z2['vers'] = [q['ver'] for q in big if (q['code'], q['date']) == key]; big2.append(z2)
out['held_days_beyond_limit'] = sorted(big2, key=lambda z: z['date'])
# 逐筆交易對 data_panel（第三來源；adj＝還原收盤、含除權息）交叉核對：pre-2015 進出的已平倉交易
dp = Q.RS / 'data_panel/panel'
rd = lambda f: pd.read_parquet(dp / f).loc['2007-01-01':'2015-12-31']
DA, DC, DO = rd('adj.parquet'), rd('close_raw.parquet'), rd('open.parquet')
chk = []
for r in lg[(lg['why'] != 'open_end') & (lg['exit_date'] <= '2015-06-30')].itertuples():
    c = str(r.code)
    if c not in DA.columns:
        chk.append(dict(code=c, ent=str(r.ent_date.date()), note='not_in_data_panel')); continue
    e, x = r.ent_date, r.exit_date; xp = IDX[r.exit_t - 1]; j = P.ci[c]
    try:
        g_dp = (DC.at[e, c] / DO.at[e, c]) * (DA.at[xp, c] / DA.at[e, c]) * (DO.at[x, c] / DC.at[xp, c])
        g_dp_raw = DO.at[x, c] / DO.at[e, c]
    except KeyError:
        chk.append(dict(code=c, ent=str(e.date()), note='date_missing')); continue
    g_eng = (P.C[r.ent_t, j] / P.O[r.ent_t, j]) * float(np.prod(1 + P.R[r.ent_t + 1:r.exit_t, j])) * (P.O[r.exit_t, j] / P.REF[r.exit_t, j])
    chk.append(dict(code=c, ent=str(e.date()), exit=str(x.date()), eng_gross=float(g_eng - 1), dp_adj=float(g_dp - 1), dp_raw_open2open=float(g_dp_raw - 1), eng_net=float(r.ret),
                    o_ent_eng=float(P.O[r.ent_t, j]), o_ent_dp=float(DO.at[e, c]), o_exit_eng=float(P.O[r.exit_t, j]), o_exit_dp=float(DO.at[x, c])))
ck = pd.DataFrame([z for z in chk if 'eng_gross' in z])
ck['d_adj'] = ck['eng_gross'] - ck['dp_adj']; ck['d_px_ent'] = (ck['o_ent_eng'] / ck['o_ent_dp'] - 1).abs(); ck['d_px_exit'] = (ck['o_exit_eng'] / ck['o_exit_dp'] - 1).abs()
ck.to_csv(Q.TMP / 'audit_trades_vs_datapanel.csv', index=False)
out['trades_vs_data_panel_pre2015H1'] = dict(n=int(len(ck)), n_not_checked=int(len(chk) - len(ck)), not_checked=[z for z in chk if 'eng_gross' not in z][:20],
    entry_open_match=float((ck['d_px_ent'] < 1e-6).mean()), exit_open_match=float((ck['d_px_exit'] < 1e-6).mean()),
    mean_eng_gross=float(ck['eng_gross'].mean()), mean_dp_adj=float(ck['dp_adj'].mean()), mean_diff_pp=float(ck['d_adj'].mean() * 100), median_diff_pp=float(ck['d_adj'].median() * 100),
    n_abs_diff_gt_1pp=int((ck['d_adj'].abs() > 0.01).sum()), n_eng_higher_gt_1pp=int((ck['d_adj'] > 0.01).sum()), n_eng_lower_gt_1pp=int((ck['d_adj'] < -0.01).sum()),
    worst=ck.reindex(ck['d_adj'].abs().sort_values(ascending=False).index).head(12)[['code', 'ent', 'exit', 'eng_gross', 'dp_adj', 'dp_raw_open2open']].round(4).to_dict('records'),
    y2008=dict(n=int((ck['ent'] < '2009-01-01').sum()), mean_eng=float(ck.loc[ck['ent'] < '2009-01-01', 'eng_gross'].mean()), mean_dp=float(ck.loc[ck['ent'] < '2009-01-01', 'dp_adj'].mean())))
out['b'] = b

# ============================================================ (c) 接縫
c = {}
ks = row('2015-01-05'); assert str(IDX[ks - 1].date()) == '2014-12-31'
long_ = pd.read_csv(Q.RS / 'raw/pre2015/asoul_twse_2007_2015H1.csv.gz', dtype={'code': str}, parse_dates=['date'])
long_ = long_[(long_['date'] >= '2014-12-20') & (long_['date'] <= '2015-01-20')].set_index(['date', 'code'])
for vr in ('B2', 'MASK'):
    nv, lgv, stv = RUN[vr]
    held = lgv[(lgv['ent_t'] < ks) & (lgv['exit_t'] >= ks)]
    tab = []; nbad = 0
    for r in held.itertuples():
        cd = str(r.code); j = P.ci[cd]
        for t in range(ks - 5, ks + 5):
            d = IDX[t]
            a = long_.loc[(d, cd)] if (d, cd) in long_.index else None
            dpc = float(DC.at[d, cd]) if cd in DC.columns and d in DC.index else None
            rec = dict(code=cd, date=str(d.date()), side='pre' if t < ks else 'official', C=float(P.C[t, j]), O=float(P.O[t, j]), CHG=float(P.CHG[t, j]) if np.isfinite(P.CHG[t, j]) else None,
                       R_pct=round(float(P.R[t, j]) * 100, 3), C_asoul=float(a['close']) if a is not None else None, O_asoul=float(a['open']) if a is not None else None,
                       CHG_asoul=float(a['change']) if a is not None and np.isfinite(a['change']) else None, C_data_panel=dpc, MK=int(P.MK[t, j]))
            rec['ok'] = bool((rec['C_asoul'] is not None and abs(rec['C'] - rec['C_asoul']) < 1e-6 and abs(rec['O'] - rec['O_asoul']) < 1e-6) and (dpc is None or abs(dpc - rec['C']) < 1e-6))
            if t == ks:
                rec['R_from_closes_pct'] = round(float(P.C[t, j] / P.C[t - 1, j] - 1) * 100, 3)
                rec['ok'] = rec['ok'] and abs(rec['R_from_closes_pct'] - rec['R_pct']) < 1e-3
            nbad += int(not rec['ok']); tab.append(rec)
    c[VER[vr]] = dict(n_held_at_seam=int(len(held)), codes=list(held['code'].astype(str)), n_cells=len(tab), n_cells_not_ok=nbad, not_ok=[z for z in tab if not z['ok']],
                      nav_ret_around=[dict(date=str(IDX[t].date()), nav_ret_pct=round(float(nv[IDX[t]] / nv[IDX[t - 1]] - 1) * 100, 3), e50_ret_pct=round(float(P.E50[t] / P.E50[t - 1] - 1) * 100, 3),
                                           stockfrac=round(float(stv['stockfrac'][IDX[t]]), 4)) for t in range(ks - 5, ks + 5)])
    pd.DataFrame(tab).to_csv(Q.TMP / f'audit_seam_{VER[vr]}.csv', index=False)
# 全上市股接縫日：官方 CHG 推得的前收 vs pre 面板 2014-12-31 收盤
tw = (P.MK[ks] == 1) & P.has[ks] & P.has[ks - 1]
ref = P.C[ks] - P.CHG[ks]
dd = np.abs(ref - P.C[ks - 1])[tw & np.isfinite(ref)]
c['all_twse_seam_ref_vs_prev_close'] = dict(n=int(len(dd)), n_match=int((dd < 1e-6).sum()), n_mismatch=int((dd >= 1e-6).sum()),
                                           mismatch=[dict(code=str(P.COLS[j]), ref=float(ref[j]), prevC=float(P.C[ks - 1, j])) for j in np.flatnonzero(tw & np.isfinite(ref) & (np.abs(ref - P.C[ks - 1]) >= 1e-6))][:20],
                                           n_twse_official_first_day=int((P.MK[ks] == 1).sum()), n_pre_last_day=int((P.MK[ks - 1] == 1).sum()),
                                           pre_last_day_codes_gone=[str(P.COLS[j]) for j in np.flatnonzero((P.MK[ks - 1] == 1) & (P.MK[ks] == 0) & ~P.MK[ks:ks + 20].any(0))],
                                           official_first_day_twse_new=[str(P.COLS[j]) for j in np.flatnonzero((P.MK[ks] == 1) & ~P.MK[ks - 20:ks].any(0))],
                                           univ_before=int(U[ks - 1].sum()), univ_after=int(U[ks].sum()), univ_changed=int((U[ks - 1] != U[ks]).sum()))
out['c'] = c

# ============================================================ (d) 兩版 2016 後差距
d = {}
nA, lA, sA = RUN['B2']; nB, lB, sB = RUN['MASK']
yA, yB = Q.yearly(nA), Q.yearly(nB)
d['yearly_gap_pp'] = {y: round((yA[y] - yB[y]) * 100, 2) for y in yA if y >= 2013}
d16 = lastb('2016-01-01'); dodd = lastb('2020-10-26')
seg = lambda n, a, b_: stlib.cagr(n, a, b_)
d['cagr'] = {k: dict(bridge=seg(nA, a, b_), mask=seg(nB, a, b_), gap_pp=(seg(nA, a, b_) - seg(nB, a, b_)) * 100) for k, (a, b_) in
             {'2015-12-31~end': (d16, DE), '2015-12-31~2020-10-23(整股時代)': (d16, dodd), '2020-10-23~end(零股時代)': (dodd, DE), '2016-03-31~end': (lastb('2016-04-01'), DE)}.items()}
d['nav_at'] = {str(x.date()): dict(bridge=float(nA[x]), mask=float(nB[x]), ratio=float(nA[x] / nB[x])) for x in (d16, lastb('2017-01-01'), lastb('2018-01-01'), lastb('2019-01-01'), dodd, lastb('2021-01-01'), lastb('2022-01-01'), DE)}
# 交易集合（訊號日 ≥ 2016-02）
tA = lA[lA['sig_date'] >= '2016-02-01']; tB = lB[lB['sig_date'] >= '2016-02-01']
kA = set(zip(tA['code'].astype(str), tA['sig_t'])); kB = set(zip(tB['code'].astype(str), tB['sig_t']))
def side(tt, keys, other_nav):
    o = []
    for r in tt.itertuples():
        if (str(r.code), r.sig_t) in keys:
            o.append(dict(code=str(r.code), sig=str(r.sig_date.date()), ent=str(r.ent_date.date()), exit=str(r.exit_date.date()), open=float(P.O[r.ent_t, P.ci[str(r.code)]]), ret=round(float(r.ret) * 100, 2), pnl=round(float(r.pnl)),
                          cost=round(float(r.cost))))
    return o
onlyA = side(tA, kA - kB, None); onlyB = side(tB, kB - kA, None)
d['trades_post2016'] = dict(n_bridge=len(kA), n_mask=len(kB), n_common=len(kA & kB), only_bridge=onlyA, only_mask=onlyB,
                            only_bridge_n_pre_odd=int(sum(z['ent'] < '2020-10-26' for z in onlyA)), only_mask_n_pre_odd=int(sum(z['ent'] < '2020-10-26' for z in onlyB)),
                            only_bridge_pnl=float(sum(z['pnl'] for z in onlyA)), only_mask_pnl=float(sum(z['pnl'] for z in onlyB)))
# 共同交易的部位權重（成本 ÷ 進場前一日淨值）
mA = tA.assign(key=list(zip(tA['code'].astype(str), tA['sig_t']))).set_index('key'); mB = tB.assign(key=list(zip(tB['code'].astype(str), tB['sig_t']))).set_index('key')
com = sorted(kA & kB, key=lambda z: z[1])
wA = np.array([mA.at[k_, 'cost'] / nA[IDX[mA.at[k_, 'sig_t']]] for k_ in com]); wB = np.array([mB.at[k_, 'cost'] / nB[IDX[mB.at[k_, 'sig_t']]] for k_ in com])
rr = np.array([mA.at[k_, 'ret'] for k_ in com]); rrB = np.array([mB.at[k_, 'ret'] for k_ in com]); ed = np.array([str(mA.at[k_, 'ent_date'].date()) for k_ in com])
pre = ed < '2020-10-26'
d['common_trades'] = dict(n=len(com), n_pre_odd=int(pre.sum()), mean_w_bridge_pre=float(wA[pre].mean()), mean_w_mask_pre=float(wB[pre].mean()), mean_w_bridge_post=float(wA[~pre].mean()), mean_w_mask_post=float(wB[~pre].mean()),
                          max_abs_ret_diff_pp=float(np.abs(rr - rrB).max() * 100), n_ret_diff_gt_0_5pp=int((np.abs(rr - rrB) > 0.005).sum()),
                          sum_w_ret_bridge_pre=float((wA * rr)[pre].sum()), sum_w_ret_mask_pre=float((wB * rrB)[pre].sum()), sum_w_ret_bridge_post=float((wA * rr)[~pre].sum()), sum_w_ret_mask_post=float((wB * rrB)[~pre].sum()),
                          exit_date_diff_n=int(sum(mA.at[k_, 'exit_t'] != mB.at[k_, 'exit_t'] for k_ in com)))
# 衛星簿記（1 − B÷淨值）與股票比重的狀態差
sa = 1 - sA['B'] / nA; sb = 1 - sB['B'] / nB
d['state'] = {k: dict(sat_share_bridge=float(sa.loc[a:b_].mean()), sat_share_mask=float(sb.loc[a:b_].mean()), stockfrac_bridge=float(sA['stockfrac'].loc[a:b_].mean()), stockfrac_mask=float(sB['stockfrac'].loc[a:b_].mean()),
                      max_abs_sat_share_diff=float((sa - sb).loc[a:b_].abs().max()))
              for k, (a, b_) in {'2016-02~2020-10-23': ('2016-02-01', dodd), '2020-10-26~2021-12': ('2020-10-26', '2021-12-31'), '2022~end': ('2022-01-01', DE)}.items()}
stats_keys = ('blk_lot', 'blk_cap', 'blk_min', 'blk_limup', 'sell_partial_cap', 'noslot', 'rb_up', 'rb_down')
d['engine_counts_full'] = {VER[v]: {k: int(RUN[v][2][k]) for k in stats_keys} for v in RUN}

def rerun(rule, cap, start=None, first_month='2008-01', odd=None):
    old_t0, old_odd = P.t0, cslib.ODD_DATE
    try:
        if start is not None:
            P.t0 = row(start)
        if odd is not None:
            cslib.ODD_DATE = pd.Timestamp(odd)
        F = FS[rule] if first_month == '2008-01' else s71lib.rev_rule(P, rule, first_month)
        sg = SG[rule] if first_month == '2008-01' else s10lib.make_signal_s10(P, U, dict(S=F['S']['STD'], av=F['av']), shift=0)
        return L.sim(P, sg, U, cap=cap)
    finally:
        P.t0, cslib.ODD_DATE = old_t0, old_odd

# 對照 1：2015-12-31 以兩版當時淨值全現金重新起跑（訊號月 ≥ 2016-01，兩版訊號相同）→ 只剩「資金規模」不同
capA, capB = float(nA[d16]), float(nB[d16])
f = {}
for nm, cap in (('cap_bridge', capA), ('cap_mask', capB), ('cap_100w', 1e6)):
    n_, l_, s_ = rerun('MASK', cap, start=str(d16.date()), first_month='2016-01')
    f[nm] = dict(cap=cap, cagr=stlib.cagr(n_, d16, DE), cagr_to_odd=stlib.cagr(n_, d16, dodd), cagr_from_odd=stlib.cagr(n_, dodd, DE), blk_lot=int(s_['blk_lot']), n_closed=int((l_['why'] != 'open_end').sum()), yearly=Q.yearly(n_.loc[d16:]))
    n_o, l_o, s_o = rerun('MASK', cap, start=str(d16.date()), first_month='2016-01', odd='2000-01-01')
    f[nm + '_oddlot_always'] = dict(cap=cap, cagr=stlib.cagr(n_o, d16, DE), cagr_to_odd=stlib.cagr(n_o, d16, dodd), cagr_from_odd=stlib.cagr(n_o, dodd, DE), blk_lot=int(s_o['blk_lot']), n_closed=int((l_o['why'] != 'open_end').sum()))
d['fresh_start_2015_12_31'] = f
d['fresh_gap_pp'] = dict(whole_lot=(f['cap_bridge']['cagr'] - f['cap_mask']['cagr']) * 100, whole_lot_to_odd=(f['cap_bridge']['cagr_to_odd'] - f['cap_mask']['cagr_to_odd']) * 100,
                         whole_lot_from_odd=(f['cap_bridge']['cagr_from_odd'] - f['cap_mask']['cagr_from_odd']) * 100,
                         oddlot_always=(f['cap_bridge_oddlot_always']['cagr'] - f['cap_mask_oddlot_always']['cagr']) * 100)
# 對照 2：全期可買零股（診斷用，不是成績單）：兩版 2016 後年化差
g = {}
for vr in ('B2', 'MASK'):
    n_, l_, s_ = rerun(vr, 1e6, odd='2000-01-01')
    g[VER[vr]] = dict(final=float(n_.iloc[-1]), cagr_full=stlib.cagr(n_, D0, DE), cagr_2016_on=stlib.cagr(n_, d16, DE), nav_2015_12_31=float(n_[d16]), y2008=float(n_[lastb('2009-01-01')] / n_[D0] - 1),
                       avg_npos=float(s_['avg_npos']), avg_idle=float(s_['avg_idle_frac']), n_closed=int((l_['why'] != 'open_end').sum()), yearly=Q.yearly(n_))
g['gap_2016_on_pp'] = (g['V_BRIDGE']['cagr_2016_on'] - g['V_MASK']['cagr_2016_on']) * 100
d['oddlot_always_full'] = g
out['d'] = d
out['runtime_s'] = round(time.time() - t0_, 1); out['maxrss_mb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
(Q.TMP / 'audit_bcd.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
print('done', out['runtime_s'], out['maxrss_mb'])
