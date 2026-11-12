#!/usr/bin/env python3
"""探索（只用挑選期 ≤ 2014-12-31；描述統計，不算正式統計量）。資料：o20 的 seed 0、REAL_opt、100 萬一般成交日誌／槽表／換股表（只讀），
本題 data/cands_allseeds.csv.gz；ĉ 用 cas_lib 的同一函式（部位 P＝該換股日 avail×0.98／10，探索近似）。
E1 ĉ 校準；E2 毛 edge 對 ĉ 五等分（平均數為主、中位數並列）；E3 右尾槽 ĉ 分布（P2）；E4 遞補的直接代價（第 11～20 名新候選 vs 高成本新候選）；
E5 小額單（名目分布、最低手續費實際費率、零股占比、順延天數、順延成交價相對 e 日開盤漂移）。輸出 explore.json。
python3 -I -B explore.py"""
import sys, json, math
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats as SS
HERE = Path(__file__).resolve().parent; AG = HERE.parent
sys.path.insert(0, str(HERE)); import cas_lib as CL; sys.path.pop(0)
R0 = json.loads((HERE / 'results.json').read_text(encoding='utf-8')); assert CL.canon(R0['PREREG']) == R0['PREREG_sha256']
SEL_END = pd.Timestamp('2014-12-31')
E = CL.engine(); D = E['D']; X = E['XE']
O20 = AG / 'o_turnover_fixedcost_diagnosis/data'
T = pd.read_csv(O20 / 'tradelog_100n_REAL_opt_seed0.csv.gz', dtype={'code': str}); T['date'] = pd.to_datetime(T['date'])
SL = pd.read_csv(O20 / 'slots_100n_REAL_opt_seed0.csv.gz', dtype={'code': str}); SL['signal'] = pd.to_datetime(SL['signal'])
SW = pd.read_csv(O20 / 'swaps_exitlevel_100n_REAL_opt_seed0.csv.gz', dtype={'code': str}); SW['signal'] = pd.to_datetime(SW['signal'])
cfg = CL.cfg_for(E, '100萬', 'normal')
C0 = CL.load_cands('REAL_opt', 0)
pl0, cd0 = CL.to_plan(C0[C0.j == 0], D)
sim = E['CASim'](pl0, dict(cfg), settle=E['ST'], cands=cd0)
def chat(tt, js, P):
    tt = np.asarray(tt, int); js = np.asarray(js, int); P = np.asarray(P, float)
    out = np.empty(len(tt))
    for u in np.unique(P):
        m = P == u; out[m] = sim._chat_vec(tt[m], js[m], u)
    return out
T = T[T.date <= SEL_END + pd.Timedelta(days=40)]
T['cost'] = T.fee + T.tax + np.where(T.kind == 'O', T.notional * T.oddc, T.notional * T.im)
# 每個 (j,k) 的換股日 e 與 avail
RK = T[T.ctx == 'R'].groupby(['j', 'k']).agg(e=('date', 'min'), avail=('avail', 'first')).reset_index()
RK['ti'] = RK.e.map(D.TPOS)
out = {}
# ---------------- E1 ĉ 校準 ----------------
g = T.groupby(['j', 'k', 'code', 'cls']).agg(notional=('notional', 'sum'), cost=('cost', 'sum')).reset_index()
en = g[g.cls == 'entry'].merge(RK, on=['j', 'k']); ex = g[g.cls == 'exit']
en = en[en.e <= SEL_END]
exd = {}
for (j, c), gg in ex.groupby(['j', 'code']): exd[(j, c)] = gg.sort_values('k')[['k', 'notional', 'cost']].to_numpy()
rt = []
for r in en.itertuples():
    a = exd.get((r.j, r.code))
    if a is None: continue
    w = a[a[:, 0] > r.k]
    if len(w) == 0: continue
    rt.append((r.j, r.k, r.code, r.ti, r.avail, r.cost / r.notional + w[0, 2] / w[0, 1], r.cost / r.notional))
RT = pd.DataFrame(rt, columns=['j', 'k', 'code', 'ti', 'avail', 'rt_real', 'buy_real'])
RT['chat'] = chat(RT.ti, RT.code.map(D.CI), RT.avail * 0.98 / 10)
RT = RT[np.isfinite(RT.chat) & np.isfinite(RT.rt_real)]
sl = SS.linregress(RT.chat, RT.rt_real)
RT['dec'] = pd.qcut(RT.chat, 10, labels=False)
out['E1_calibration'] = dict(n=int(len(RT)), slope=float(sl.slope), intercept=float(sl.intercept), r2=float(sl.rvalue ** 2),
                             mean_chat=float(RT.chat.mean()), mean_real=float(RT.rt_real.mean()),
                             deciles=RT.groupby('dec').agg(chat=('chat', 'mean'), real=('rt_real', 'mean'), n=('chat', 'size')).round(5).to_dict('index'),
                             note='只用來確認 ĉ 定義沒寫錯，不據此調門檻')
# ---------------- E2/E3 槽表 ----------------
S2 = SL[SL.signal <= SEL_END].merge(RK[['j', 'k', 'ti', 'avail']], on=['j', 'k'], how='left')
S2 = S2[np.isfinite(S2.ti)]
S2['chat'] = chat(S2.ti.astype(int), S2.code.map(D.CI), S2.avail * 0.98 / 10)
S2['r_rel'] = S2.r - S2.groupby(['j', 'k']).r.transform('mean')
ent = S2[S2.is_entry == 1].copy(); ent['q'] = pd.qcut(ent.chat, 5, labels=False)
out['E2_gross_vs_chat_entries'] = ent.groupby('q').agg(chat_mean=('chat', 'mean'), r_mean=('r', 'mean'), r_med=('r', 'median'), rrel_mean=('r_rel', 'mean'),
                                                       rrel_med=('r_rel', 'median'), n=('r', 'size')).round(5).to_dict('index')
hi = ent[ent.q == 4].r_rel; lo = ent[ent.q == 0].r_rel
out['E2_q5_minus_q1'] = dict(rrel_mean_diff=float(hi.mean() - lo.mean()), welch_t=float(SS.ttest_ind(hi, lo, equal_var=False).statistic))
sw = SW[SW.signal <= SEL_END].copy()
ek = ent.groupby(['j', 'k']).chat.mean().rename('chat_entries_k').reset_index()
sw = sw.merge(ek, on=['j', 'k'], how='inner'); sw = sw[np.isfinite(sw.edge_B)]
sw['q'] = pd.qcut(sw.chat_entries_k, 5, labels=False, duplicates='drop')
out['E2_swap_edgeB_vs_chat'] = sw.groupby('q').agg(chat=('chat_entries_k', 'mean'), edge_mean=('edge_B', 'mean'), edge_med=('edge_B', 'median'),
                                                   rtc_mean=('rtc_B', 'mean'), n=('edge_B', 'size')).round(5).to_dict('index')
thr = S2.r.quantile(0.95); tail = S2[S2.r >= thr]; rest = S2[S2.r < thr]
ks = SS.ks_2samp(tail.chat, rest.chat)
out['E3_P2_righttail'] = dict(n_slots=int(len(S2)), r_thr=float(thr), n_tail=int(len(tail)), chat_tail_mean=float(tail.chat.mean()), chat_rest_mean=float(rest.chat.mean()),
                              chat_diff=float(tail.chat.mean() - rest.chat.mean()), chat_tail_med=float(tail.chat.median()), chat_rest_med=float(rest.chat.median()),
                              ks_stat=float(ks.statistic), ks_p=float(ks.pvalue), tail_entry_share=float(tail.is_entry.mean()), rest_entry_share=float(rest.is_entry.mean()),
                              tail_price_lt50=float((tail.price < 50).mean()), rest_price_lt50=float((rest.price < 50).mean()),
                              note='KS 把同一事件在 13 j 重複計數（約 4 次），p 值偏小；平均 ĉ 差為主')
# ---------------- E4 遞補的直接代價 ----------------
def winret(ti, ti2, j):
    o = D.O[ti, j]; o2 = D.O[ti2, j]
    if not np.isfinite(o): o = D.CF[ti - 1, j]
    if not np.isfinite(o2): o2 = D.CF[ti2, j]
    return float(o2 * np.exp(np.log(D.ADJ[ti + 1:ti2 + 1, j]).sum()) / o - 1)
pairs = {0.8: [], 0.9: []}
for jj, gj in C0.groupby('j'):
    pl, cd = CL.to_plan(gj, D)
    av = RK[RK.j == jj].set_index('ti').avail
    ex_list = [p[0] for p in pl]
    held = set(); ht = []; hj = []
    for i, (e, pk, s) in enumerate(pl):
        if D.IDX[s] > SEL_END: break
        cand, gu = cd[e]
        F = [x for x in cand if not (x in gu and x not in held)]
        base = F[:10]; assert base == [int(x) for x in pk]
        e2 = ex_list[i + 1] if i + 1 < len(ex_list) else None
        Pv = av.get(e, np.nan) * 0.98 / 10
        if e2 is not None and np.isfinite(Pv):
            lo_i = np.searchsorted(D.DATES, np.datetime64(D.IDX[s] - pd.DateOffset(months=36)), side='right')
            hta = np.array(ht, int); hja = np.array(hj, int); m = hta - 1 >= lo_i
            if len(set(hta[m].tolist())) >= 12:
                new = [x for x in base if x not in held]; R = [x for x in F[10:20] if x not in held]
                hc = sim._chat_vec(hta[m], hja[m], Pv)
                cn = dict(zip(new, sim._chat_vec(np.full(len(new), e), np.array(new, int), Pv))) if new else {}
                cr = dict(zip(R, sim._chat_vec(np.full(len(R), e), np.array(R, int), Pv))) if R else {}
                for q in (0.8, 0.9):
                    tau = np.nanpercentile(hc, 100 * q); RR = list(R)
                    for x in new:
                        if cn[x] > tau:
                            y = next((r for r in RR if cr[r] <= tau), None)
                            if y is None: continue
                            RR.remove(y)
                            pairs[q].append((jj, str(D.IDX[s].date()), D.COLS[x], D.COLS[y], winret(e, e2, x), winret(e, e2, y), float(cn[x]), float(cr[y]), float(tau),
                                             float(D.C[s, x]), float(D.C[s, y])))
        ht.extend([e] * len(base)); hj.extend(base); held = set(base)
E4 = {}
for q, L in pairs.items():
    P4 = pd.DataFrame(L, columns=['j', 'signal', 'x', 'y', 'r_x', 'r_y', 'c_x', 'c_y', 'tau', 'px_x', 'px_y'])
    P4['d_gross'] = P4.r_y - P4.r_x; P4['d_net'] = (P4.r_y - P4.c_y) - (P4.r_x - P4.c_x)
    bym = P4.groupby('signal')[['d_gross', 'd_net']].mean()
    E4[f'tau{int(q * 100)}'] = dict(n_pairs=int(len(P4)), n_dates=int(len(bym)), r_x_mean=float(P4.r_x.mean()), r_y_mean=float(P4.r_y.mean()),
                                    d_gross_mean=float(P4.d_gross.mean()), d_gross_med=float(P4.d_gross.median()), t_gross_bydate=CL.tstat(bym.d_gross),
                                    d_net_mean=float(P4.d_net.mean()), t_net_bydate=CL.tstat(bym.d_net), c_x_mean=float(P4.c_x.mean()), c_y_mean=float(P4.c_y.mean()),
                                    tau_mean=float(P4.tau.mean()), px_x_med=float(P4.px_x.median()), px_y_med=float(P4.px_y.median()),
                                    note='同一事件在 13 j 重複出現；t 依訊號日分群（日期不同的 j 視為不同群，仍偏大）')
    P4.to_csv(HERE / f'data/explore_ca_pairs_tau{int(q * 100)}.csv.gz', index=False, compression='gzip')
out['E4_replacement_cost'] = E4
# ---------------- E5 小額單 ----------------
T5 = T[T.date <= SEL_END].copy()
F_ = 20 / (0.001425 * 0.6)
rw = T5[T5.cls.isin(['rw_buy', 'rw_sell'])].groupby(['j', 'k', 'code']).notional.sum()
fl = T5[T5.shares > 0]
T5['minfee'] = (T5.fee <= 20.0 + 1e-9) & (T5.shares > 0)
dd = T5[T5.ctx == 'P'].merge(RK[['j', 'k', 'ti']], on=['j', 'k'])
dd['lag'] = dd.date.map(D.TPOS) - dd.ti
bp = dd[(dd.side > 0) & (dd.shares > 0)].copy()
bp['open_e'] = [D.O[t, D.CI[c]] for t, c in zip(bp.ti, bp.code)]
bp['drift'] = bp.px / bp.open_e - 1
days = T5.groupby('j').date.nunique()
ntd = {j: int(((D.IDX >= T5[T5.j == j].date.min()) & (D.IDX <= SEL_END)).sum()) for j in range(13)}
out['E5_small_orders'] = dict(
    F_star_pre2015=F_, rw_orders=int(len(rw)), rw_notional_q=rw.quantile([.1, .25, .5, .75, .9]).round(0).to_dict(), rw_lt_Fstar=float((rw < F_).mean()),
    fills=int(len(fl)), odd_share_fills=float((fl.kind == 'O').mean()), minfee_share_fills=float(T5.minfee[T5.shares > 0].mean()),
    minfee_eff_rate_mean=float((T5.fee / T5.notional)[T5.minfee].mean()), deferred_fill_share=float((fl.ctx == 'P').mean()),
    lag_days_q=dd.lag.quantile([.5, .75, .9, .95, .99]).to_dict(), lag_gt5_share_of_deferred_buys=float((bp.lag > 5).mean()),
    deferred_buy_drift_mean=float(bp.drift.mean()), deferred_buy_drift_med=float(bp.drift.median()),
    deferred_buy_drift_mean_lag_gt5=float(bp.drift[bp.lag > 5].mean()), deferred_buy_drift_mean_lag_le5=float(bp.drift[bp.lag <= 5].mean()),
    order_day_share_by_j=float(np.median([days[j] / ntd[j] for j in range(13)])))
(HERE / 'explore.json').write_text(CL.dumps(out), encoding='utf-8')
print(CL.dumps({k: v for k, v in out.items() if k != 'E1_calibration'})[:6000])
print(CL.dumps({k: v for k, v in out['E1_calibration'].items() if k != 'deciles'}))
