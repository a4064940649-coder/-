#!/usr/bin/env python3
"""R82／R83 獨立複製（不讀、不 import s82_*/s83_* 底下任何 .py）。
輸入：S75 trades.csv.gz、nav_2008_2026.csv.gz、連續面板快取（經 s75lib.load_cont 共用引擎載入）、S83 的 thermometer_monthly.csv（輸出資料，只拿來比對）。
用法：python3 -I -B rep.py A      → 統計類（項目 1–4、7、8–10），寫 results_A.json、thermo_mine.csv
      python3 -I -B rep.py B      → 需重跑帳戶的（項目 5、6），寫 results_B.json、skip_sim_mine.csv
挑哪些交易、怎麼統計全部在本檔自寫；只有「重跑帳戶」呼叫共用引擎 s10lib.simulate_s10（用它原本就有的 block 參數擋掉指定買進）。"""
import sys, json, time, resource, os
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
sys.path.insert(0, str(AG / 's75_continuous_2008_2026'))
import s75lib as K                                           # noqa: E402  共用引擎包裝（允許）
from s75lib import s10lib, L                                 # noqa: E402

MODE = sys.argv[1]
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
t00 = time.time()
tr = pd.read_csv(AG / 's75_continuous_2008_2026' / 'trades.csv.gz', dtype={'code': str}, parse_dates=['sig_date', 'ent_date', 'exit_date'])
tr = tr[tr.version == 'V_BRIDGE'].reset_index(drop=True)
assert len(tr) == 632
navf = pd.read_csv(AG / 's75_continuous_2008_2026' / 'nav_2008_2026.csv.gz', index_col=0, parse_dates=True)

P, U = K.load_cont()
F, RT = K.surge(P, 'B2')
sg = K.signal(P, U, RT, 0)
del F
print('loaded', round(time.time() - t00), 's', rss(), 'MB', flush=True)
IDX = P.IDX; T = P.T
D0, DE = IDX[P.t0], IDX[-1]
J = np.array([P.ci[c] for c in tr.code])
E = tr.ent_t.to_numpy(); X = tr.exit_t.to_numpy()
assert (IDX[E] == tr.ent_date.values).all()
KS = np.array([i['k'] for i in sg['info']])                  # 訊號列
EXEC = KS + 1                                                # 換股執行日
cagr = lambda s: float((s.iloc[-1] / s.iloc[0]) ** (365.25 / (s.index[-1] - s.index[0]).days) - 1)


def base_sim(block=None, sgx=None):
    S2, ekw = s10lib.variant_S(sgx or sg, 'E0')
    return s10lib.simulate_s10(P, S2, U, cost=s10lib.COSTS['strict'], capital=1e6, cap_pct=0.05, sat_w=0.7, sizing='W10', lock=False, block=block, **ekw)


if MODE == 'A':
    out = {}
    r = tr.ret.to_numpy(); pnl = tr.pnl.to_numpy()
    order = np.argsort(-r, kind='stable')
    n5 = int(np.ceil(632 * 0.05)); n10 = int(632 * 0.10)
    top5 = np.zeros(632, bool); top5[order[:n5]] = True
    top10 = np.zeros(632, bool); top10[order[:n10]] = True
    # ---------- 1
    out['1'] = dict(n=632, mean=r.mean(), median=float(np.median(r)), win=float((r > 0).mean()), n5=n5, thr5=float(r[order[n5 - 1]]), n10=n10,
                    top5_pnl_share=pnl[top5].sum() / pnl.sum(), top5_ret_share=r[top5].sum() / r.sum(),
                    top10_pnl_share=pnl[top10].sum() / pnl.sum(), top10_ret_share=r[top10].sum() / r.sum(),
                    rest569_pnl=float(pnl[~top10].sum()), top5_share_of_nav_gain=pnl[top5].sum() / (navf['V_BRIDGE'].iloc[-1] - 1e6))
    # ---------- 2（依進場日）
    y = tr.ent_date.dt.year.to_numpy()
    seg = {}
    for nm, a, b in (('2008-2015', 2008, 2015), ('2016-2018', 2016, 2018), ('2019-2026', 2019, 2026)):
        m = (y >= a) & (y <= b)
        seg[nm] = dict(n=int(m.sum()), mean=float(r[m].mean()), median=float(np.median(r[m])), max=float(r[m].max()), ge50=int((r[m] >= 0.5).sum()), ge100=int((r[m] >= 1.0).sum()))
    out['2'] = seg
    # ---------- 3
    hd = tr.hold_days.to_numpy()
    first = np.array([EXEC[EXEC > e][0] if (EXEC > e).any() else -1 for e in E])
    O, G, CF, REF = P.O, P.G, P.CF, P.REF

    def gross_to_open(j, e, x):
        """進場日開盤買 → x 日開盤賣的含息毛報酬倍數（x 日無開盤價就用參考價）。"""
        o = O[x, j]; k = o / REF[x, j] if np.isfinite(o) and o > 0 else 1.0
        return CF[e, j] / O[e, j] * G[x - 1, j] / G[e, j] * k

    def net_ret(i, x):
        j, e = J[i], E[i]
        o = O[x, j] if np.isfinite(O[x, j]) and O[x, j] > 0 else REF[x, j]
        gross = tr.shares[i] * O[e, j] * gross_to_open(j, e, x)
        sl = 0.5 * float(K.cslib.tick(o)) / o + 0.003
        g = gross * (1 - sl)
        return (g - g * s10lib.FEE - g * s10lib.TAX) / tr.cost[i] - 1
    chk = np.array([net_ret(i, X[i]) for i in range(632) if tr.why[i] == 'rule'])
    act = r[(tr.why == 'rule').to_numpy()]
    fr = np.array([net_ret(i, first[i]) if first[i] > 0 else np.nan for i in range(632)])
    fg = np.array([gross_to_open(J[i], E[i], first[i]) - 1 if first[i] > 0 else np.nan for i in range(632)])
    nper = np.array([int(((EXEC > e) & (EXEC <= x)).sum()) for e, x in zip(E, X)])    # 經過幾個換股執行日才出場
    out['3'] = dict(hold_median_top5=float(np.median(hd[top5])), hold_median_rest=float(np.median(hd[~top5])), hold_median_all=float(np.median(hd)),
                    first_check_net_median_top5=float(np.nanmedian(fr[top5])), first_check_gross_median_top5=float(np.nanmedian(fg[top5])),
                    actual_median_top5=float(np.median(r[top5])), n_top5_first_check_ge_thr=int((fr[top5] >= out['1']['thr5']).sum()),
                    n_top5_held_2plus_periods=int((nper[top5] >= 2).sum()),
                    approx_check_max_abs_err_vs_actual=float(np.abs(chk - act).max()), approx_check_median_abs_err=float(np.median(np.abs(chk - act))))
    # ---------- 4 最大浮盈（收盤價；相對進場成本，毛；持有期間＝進場日～出場前一日收盤）
    mfe = np.empty(632)
    for i in range(632):
        j, e, x = J[i], E[i], X[i]
        last = x if tr.why[i] == 'open_end' else x - 1
        path = CF[e, j] / O[e, j] * G[e:last + 1, j] / G[e, j]
        mfe[i] = tr.shares[i] * O[e, j] * path.max() / tr.cost[i] - 1
    cap = r / mfe
    big = mfe >= 0.5
    # 變體（毛報酬口徑：只看價格、不扣成本；相對進場開盤價）——追差異用
    gr = np.array([(CF[E[i], J[i]] / O[E[i], J[i]] * G[X[i], J[i]] / G[E[i], J[i]] - 1) if tr.why[i] == 'open_end' else gross_to_open(J[i], E[i], X[i]) - 1 for i in range(632)])
    mfe_g = np.array([(CF[E[i], J[i]] / O[E[i], J[i]] * G[E[i]:(X[i] if tr.why[i] == 'open_end' else X[i] - 1) + 1, J[i]] / G[E[i], J[i]]).max() - 1 for i in range(632)])
    mfe_gx = np.maximum(mfe_g, gr)                                            # 出場價高於所有收盤時，以出場報酬為最大浮盈
    out['4_gross_variant'] = dict(top5_capture_agg=float(gr[top5].sum() / mfe_g[top5].sum()), top5_capture_median=float(np.median((gr / mfe_g)[top5])),
                                  top5_capture_agg_floor=float(gr[top5].sum() / mfe_gx[top5].sum()), top5_capture_median_floor=float(np.median((gr / mfe_gx)[top5])),
                                  n_mfe_ge50=int((mfe_g >= 0.5).sum()), n_mfe_ge50_floor_by_exit=int((mfe_gx >= 0.5).sum()),
                                  n_floor_exit_lt20=int(((mfe_gx >= 0.5) & (gr < 0.2)).sum()), actual_gross_median_top5=float(np.median(gr[top5])))
    out['4'] = dict(top5_capture_agg=float(r[top5].sum() / mfe[top5].sum()), top5_capture_median=float(np.median(cap[top5])),
                    n_mfe_ge50=int(big.sum()), n_mfe_ge50_exit_lt20=int((big & (r < 0.2)).sum()),
                    n_mfe_ge50_closed_only=int((big & (tr.why != 'open_end').to_numpy()).sum()))
    pd.DataFrame(dict(code=tr.code, ent_date=tr.ent_date, exit_date=tr.exit_date, ret=r, pnl=pnl, hold_days=hd, top5=top5, top10=top10, first_check_net=fr, mfe_close=mfe,
                      capture=cap, n_periods=nper)).to_csv(HERE / 'trades_mine.csv', index=False)
    # ---------- 7 環境 vs 訊號：訊號日收盤 → 3 個訊號期後訊號日收盤的含息漲幅 > 50%
    rows = []
    for a, k in enumerate(KS):
        if a + 3 >= len(KS):
            break
        k3 = KS[a + 3]
        f_cc = G[k3] / G[k] - 1                                               # 訊號日收盤 → 3 期後訊號日收盤
        with np.errstate(invalid='ignore', divide='ignore'):
            o1 = O[k + 1]; o3 = O[k3 + 1]
            f_oo = (CF[k + 1] / o1) * (G[k3] / G[k + 1]) * np.where(np.isfinite(o3) & (o3 > 0), o3 / REF[k3 + 1], 1.0) - 1    # 執行日開盤 → 3 期後執行日開盤
        u = U[k]; b = sg['sig'][k]
        rows.append(dict(sig_date=IDX[k], buy_n_valid_oo=int((b & np.isfinite(f_oo)).sum()), univ_n=int(u.sum()), univ_big_cc=int((f_cc[u] > 0.5).sum()), buy_n=int(b.sum()), buy_big_cc=int((f_cc[b] > 0.5).sum()),
                         univ_n_oo=int((u & np.isfinite(f_oo)).sum()), univ_big_oo=int((f_oo[u] > 0.5).sum()), buy_big_oo=int((f_oo[b] > 0.5).sum())))
    env = pd.DataFrame(rows); env.to_csv(HERE / 'env_mine.csv', index=False)
    o7 = {}
    for nm, m in (('2008-2018', env.sig_date < '2019-01-01'), ('2019-2026', env.sig_date >= '2019-01-01')):
        e = env[m]
        o7[nm] = dict(months=int(len(e)), univ_cc=e.univ_big_cc.sum() / e.univ_n.sum(), buy_cc=e.buy_big_cc.sum() / e.buy_n.sum(), buy_n=int(e.buy_n.sum()),
                      univ_oo=e.univ_big_oo.sum() / e.univ_n_oo.sum(), buy_oo=e.buy_big_oo.sum() / e.buy_n.sum(),
                      buy_n_valid_oo=int(e.buy_n_valid_oo.sum()), buy_big_oo_n=int(e.buy_big_oo.sum()), buy_oo_valid=e.buy_big_oo.sum() / e.buy_n_valid_oo.sum())
    for v in ('cc', 'oo'):
        o7['ratio_' + v] = dict(pre=o7['2008-2018']['buy_' + v] / o7['2008-2018']['univ_' + v], post=o7['2019-2026']['buy_' + v] / o7['2019-2026']['univ_' + v])
    out['7'] = o7
    # ---------- 8 溫度計
    R_ = P.R
    Uf = U.astype(float); nU = Uf.sum(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        daily_big = ((R_ >= 0.065) & U).sum(1) / nU
    me = pd.Series(np.arange(T), index=IDX).groupby(IDX.to_period('M')).last()
    me = me[me.index >= pd.Period('2008-01', 'M')]
    rows = []
    for mth, t in me.items():
        u = U[t]
        r63 = G[t] / G[t - 63] - 1; r21 = G[t] / G[t - 21] - 1
        u63 = u & (P.nbar[t - 63] > 0)                                         # 變體：63 日前已掛牌才算
        rows.append(dict(month=str(mth), date=IDX[t], n=int(u.sum()), SPURT3=float((r63[u] > 0.5).mean()), SPURT3_listed63=float((r63[u63] > 0.5).mean()),
                         BIGUP21=float(np.nanmean(daily_big[t - 20:t + 1])), TAIL21=float(np.percentile(r21[u], 95) - np.median(r21[u]))))
    th = pd.DataFrame(rows).set_index('month'); th.to_csv(HERE / 'thermo_mine.csv')
    ref = pd.read_csv(AG / 's83_environment_thermometer' / 'thermometer_monthly.csv', index_col=0)
    assert list(ref.index) == list(th.index), (len(ref), len(th))
    assert (pd.to_datetime(ref.date).values == th.date.values).all()
    o8 = {}
    for c in ('SPURT3', 'BIGUP21', 'TAIL21'):
        d = th[c] - ref[c]
        o8[c] = dict(corr=float(np.corrcoef(th[c], ref[c])[0, 1]), mad=float(d.abs().mean()), max_abs=float(d.abs().max()), last_mine=float(th[c].iloc[-1]), last_ref=float(ref[c].iloc[-1]))
    o8['SPURT3_listed63_variant'] = dict(mad=float((th['SPURT3_listed63'] - ref['SPURT3']).abs().mean()))
    o8['n_months'] = len(th); o8['last_date'] = str(th.date.iloc[-1].date())
    out['8'] = o8
    # ---------- 9 訓練段三分位
    trn = th.loc['2008-01':'2015-12']
    out['9'] = {c: dict(cold=float(trn[c].quantile(1 / 3)), hot=float(trn[c].quantile(2 / 3)), n=int(len(trn))) for c in ('SPURT3', 'BIGUP21', 'TAIL21')}
    # ---------- 10 驗證段合成
    nv = navf[['V_BRIDGE', '0050']]
    pts = pd.concat([nv.iloc[:1], nv.groupby(nv.index.to_period('M')).tail(1)])
    mr = pts.pct_change().iloc[1:]; mr.index = mr.index.to_period('M').astype(str)
    ex = (mr['V_BRIDGE'] - mr['0050']).reindex(th.index)
    nh = sum((th[c] > out['9'][c]['hot']).astype(int) for c in ('SPURT3', 'BIGUP21', 'TAIL21'))
    nl = sum((th[c] < out['9'][c]['cold']).astype(int) for c in ('SPURT3', 'BIGUP21', 'TAIL21'))
    state = np.where(nh >= 2, '熱', np.where(nl >= 2, '冷', '溫'))
    th2 = th.assign(n_high=nh, n_low=nl, state=state, ex=ex); th2.to_csv(HERE / 'thermo_mine.csv')
    va = th2.loc['2016-01':'2026-10']
    out['10'] = {s: dict(n=int((va.state == s).sum()), ex_ann_pp=float(va.ex[va.state == s].mean() * 1200)) for s in ('冷', '溫', '熱')}
    out['10']['n_valid'] = int(len(va)); out['10']['state_agree_with_ref_all'] = float((th2.state.values == ref.state.values).mean())
    out['10']['state_agree_valid'] = float((va.state.values == ref.loc['2016-01':'2026-10'].state.values).mean())
    out['10']['ex_max_abs_diff_vs_ref_pp'] = float((ex * 100 - ref.ex_month_pct).abs().max())
    # 用原檔的門檻（四捨五入後的 FINDINGS 值）再分一次，作敏感度
    thr = dict(SPURT3=(0.0118, 0.0280), BIGUP21=(0.0159, 0.0251), TAIL21=(0.153, 0.189))
    nh2 = sum((va[c] > thr[c][1]).astype(int) for c in thr); nl2 = sum((va[c] < thr[c][0]).astype(int) for c in thr)
    st2 = np.where(nh2 >= 2, '熱', np.where(nl2 >= 2, '冷', '溫'))
    out['10']['with_published_thresholds'] = {s: dict(n=int((st2 == s).sum()), ex_ann_pp=float(va.ex[st2 == s].mean() * 1200)) for s in ('冷', '溫', '熱')}
    out['10']['hot_streak_months_to_last'] = int((th2.state[::-1] != '熱').to_numpy().argmax()) if (th2.state != '熱').any() else len(th2)
    out['rss_mb'] = rss(); out['runtime_s'] = round(time.time() - t00, 1)
    (HERE / 'results_A.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    print(json.dumps(out, ensure_ascii=False, indent=1, default=float))

if MODE == 'B':
    import multiprocessing as mp
    out = {}
    t1 = time.time()
    nav0, lg0, st0 = base_sim()
    print('base sim', round(time.time() - t1, 1), 's', flush=True)
    ref = navf['V_BRIDGE'].reindex(nav0.index)
    out['accept'] = dict(n_trades=int(len(lg0)), nav_max_rel_diff_vs_S75=float((nav0 / ref - 1).abs().max()), base_cagr=cagr(nav0),
                         trades_match=bool(len(lg0) == 632 and np.allclose(np.sort(lg0.ret.to_numpy()), np.sort(tr.ret.to_numpy()))))
    base = out['accept']['base_cagr']
    print(out['accept'], flush=True)
    r = tr.ret.to_numpy(); order = np.argsort(-r, kind='stable')

    def drop(ids, whole):
        """whole=True（主定義）：該檔在原交易的整個持有期間（進場日～出場日）都不准買 → 這筆交易完全消失，名額與錢照引擎規則給別檔或 0050。
        whole=False（第一次嘗試的定義）：只擋原進場那一天 → 下個月還在名單上就會被補買回來。"""
        blk = {}
        for i in ids:
            blk.setdefault(int(J[i]), []).append((int(E[i]), int(X[i]) if whole else int(E[i])))
        nav, lg, _ = base_sim(block=blk)
        key = set(zip(lg.j.to_numpy().tolist(), lg.ent_t.to_numpy().tolist()))
        rebought = sum(1 for i in ids if any((lg.j == J[i]) & (lg.ent_t > E[i]) & (lg.ent_t <= X[i])))
        return dict(cagr=cagr(nav), diff_pp=(cagr(nav) - base) * 100, n_trades=int(len(lg)), final=float(nav.iloc[-1]), n_rebought_within_original_hold=int(rebought))
    o5 = {}
    po = np.argsort(-tr.pnl.to_numpy(), kind='stable')
    for whole in (True, False):
        for nm, ids in (('top1', order[:1]), ('top3', order[:3]), ('top10', order[:10]), ('top5pct_32', order[:32]), ('pnl_top1', po[:1]), ('pnl_top3', po[:3])):
            k = nm + ('' if whole else '_entry_day_only')
            o5[k] = drop(ids, whole)
            print(k, o5[k], flush=True)
    out['5'] = o5
    (HERE / 'results_B.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    # ---------- 6 隨機漏買：每個訊號月以 p 機率漏掉當月全部新買進（該訊號列的買進訊號清空；出場矩陣 X 不動）
    NSIM = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    sig0 = sg['sig']

    def one(arg):
        p, seed = arg
        rng = np.random.default_rng(seed)
        skip = rng.random(len(KS)) < p
        sgx = dict(sg); s = sig0.copy(); s[KS[skip]] = False; sgx['sig'] = s
        nav, lg, _ = base_sim(sgx=sgx)
        return p, seed, int(skip.sum()), cagr(nav), len(lg)
    jobs = [(p, 820000 + int(p * 100) * 10000 + i) for p in (0.10, 0.25) for i in range(NSIM)]
    with mp.get_context('fork').Pool(3) as pool:
        res = pool.map(one, jobs, chunksize=5)
    sk = pd.DataFrame(res, columns=['p', 'seed', 'n_skipped_months', 'cagr', 'n_trades']); sk.to_csv(HERE / 'skip_sim_mine.csv', index=False)
    o6 = {}
    for p, g in sk.groupby('p'):
        q = g.cagr.quantile([0.05, 0.5, 0.95]).to_numpy()
        o6[str(p)] = dict(n=int(len(g)), median=float(q[1]), p5=float(q[0]), p95=float(q[2]), median_diff_pp=float((q[1] - base) * 100), mean_skipped=float(g.n_skipped_months.mean()))
    out['6'] = o6; out['n_signal_months'] = int(len(KS))
    out['rss_mb_parent'] = rss(); out['rss_mb_children'] = round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024); out['runtime_s'] = round(time.time() - t00, 1)
    (HERE / 'results_B.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    print(json.dumps(out, ensure_ascii=False, indent=1, default=float))
