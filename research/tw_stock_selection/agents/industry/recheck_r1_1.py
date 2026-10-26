#!/usr/bin/env python3
"""重測 r1_1：產業排序 rank IC（RS_52wH、DIFF_NH）的穩健性。

完全沿用 t4_ind_momentum.py 的訊號定義與 IC 計算（先重現 2.24/3.55、2.35/3.40），再做：
 (1) 挑選期 9 個訊號 IC t 的 Bonferroni / Holm 校正；
     置換檢定 A：每月把產業報酬在產業間隨機排列（9 個訊號共用同一排列）→ 單一訊號 p 與 max-t 族系 p；
     置換檢定 B：把股票的產業標籤隨機打亂（保持各產業檔數），整套重建產業指數與訊號 → 檢驗「產業結構」本身是否有資訊。
 (2) 剔除電子 8 類產業後重算挑選期 IC；另把 130 檔無法歸類（多為已下市電子股）隨機補進電子 8 類，看存活者偏誤的影響；
     另報剔除近年新分類（綠能環保、數位雲端、居家生活、運動休閒、油電燃氣業）。
     註：面板 2019–2023 只有存活者，沒有「2019 後下市股完整」的期間可用於挑選期，故改用補歸類的方法。
 (3) 分段 IC：2005–2012、2013–2019-04；驗證期 2019-05–2023、2024–2026。
 (4) 對 MOM6/MOM12 殘差化（每月橫斷面 rank 迴歸）、Fama-MacBeth、MOM6 條件雙重排序。
"""
import sys, json, math, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from scipy import stats as sst
from ilib import load, eligibility, period_returns, month_ends, prev_day, tstat

OUT = Path(__file__).resolve().parent
SEL = ('2005-01-01', '2019-04-30'); VAL = ('2019-05-01', '2026-12-31')
SIGS = ['MOM1', 'MOM3', 'MOM6', 'MOM12', 'MOM12_1', 'RS_52wH', 'RS_MA60', 'DIFF_NH', 'DIFF_DIST']
ELEC = ['半導體業', '電腦及週邊設備業', '光電業', '通信網路業', '電子零組件業', '電子通路業', '資訊服務業', '其他電子業']
NEWCAT = ['綠能環保', '數位雲端', '居家生活', '運動休閒', '油電燃氣業']
RNG = np.random.default_rng(20261026)


def prep():
    d = load(); adj, C, V60, RET = d['adj'], d['C'], d['V60'], d['RET']
    me = [x for x in month_ends(adj.index) if x >= pd.Timestamp('2004-01-01')]
    R = period_returns(adj, me)
    el, v = eligibility(me)
    pdays = prev_day(me, adj.index); DI = pd.DatetimeIndex(me)
    okd = ((V60.shift(1) >= 1e7) & (C.shift(1) >= 10)) & RET.notna()
    rd = RET.where(okd).clip(-0.5, 0.5)
    e = d['e50'].reindex(adj.index).ffill()
    A = adj.ffill(limit=5)
    mx252 = A.rolling(252, min_periods=240).max()
    nh = (A.rolling(21).max() >= mx252 - 1e-12).where(mx252.notna())
    dist = A / mx252
    nh = nh.reindex(pdays).set_axis(DI).astype(float); dist = dist.reindex(pdays).set_axis(DI)
    elR = el.reindex(index=R.index)
    return dict(adj=adj, me=me, R=R, el=el, pdays=pdays, DI=DI, okd=okd, rd=rd, e=e, nh=nh, dist=dist, elR=elR)


def gmean(X, OK, IND, cols):
    """產業平均：X 的非 NaN（且 OK）值依 IND 分組平均。回傳 (mean, count)。用矩陣乘法加速。"""
    labs = sorted(set(IND[cols]))
    M = np.zeros((len(cols), len(labs)))
    li = {k: i for i, k in enumerate(labs)}
    for j, c in enumerate(cols): M[j, li[IND[c]]] = 1.0
    Xv = X[cols].where(OK[cols]) if OK is not None else X[cols]
    msk = Xv.notna().values.astype(float)
    s = np.nan_to_num(Xv.values.astype(float)) @ M
    n = msk @ M
    with np.errstate(invalid='ignore', divide='ignore'):
        m = s / n
    m[n == 0] = np.nan
    return pd.DataFrame(m, index=X.index, columns=labs), pd.DataFrame(n, index=X.index, columns=labs)


def build(P, IND):
    """依 t4 定義重建 9 個訊號與產業持有期報酬。"""
    cols = [c for c in P['adj'].columns if c in IND.index]
    g, n = gmean(P['rd'], P['okd'], IND, cols)
    g = g.where(n >= 3).fillna(0.0)
    IDX = (1 + g).cumprod()
    pd_, DI = P['pdays'], P['DI']
    Ipd = IDX.reindex(pd_).set_axis(DI)
    ok = (n.rolling(252, min_periods=1).min().reindex(pd_).set_axis(DI) >= 3)
    S = {}
    for k in [1, 3, 6, 12]: S[f'MOM{k}'] = (Ipd / Ipd.shift(k) - 1).where(ok)
    S['MOM12_1'] = (Ipd.shift(1) / Ipd.shift(12) - 1).where(ok)
    RS = IDX.div(P['e'], axis=0)
    S['RS_52wH'] = (RS / RS.rolling(252, min_periods=200).max()).reindex(pd_).set_axis(DI).where(ok)
    S['RS_MA60'] = (RS / RS.rolling(60).mean() - 1).reindex(pd_).set_axis(DI).where(ok)
    el = P['el']
    S['DIFF_NH'], nn = gmean(P['nh'], el, IND, cols)
    S['DIFF_DIST'], _ = gmean(P['dist'], el, IND, cols)
    cnt = (el[cols].astype(float).values @ _onehot(IND, cols))
    cnt = pd.DataFrame(cnt, index=el.index, columns=S['DIFF_NH'].columns)
    R = P['R']
    IR, _ = gmean(R, P['elR'], IND, cols)
    out = {}
    for sn in SIGS:
        M = S[sn].reindex(columns=IR.columns)
        M = M.where(cnt.reindex(M.index) >= 5)
        if sn.startswith('DIFF'): M = M.where(nn.reindex(M.index) >= 5)
        out[sn] = M.reindex(R.index)
    return out, IR


def _onehot(IND, cols):
    labs = sorted(set(IND[cols])); li = {k: i for i, k in enumerate(labs)}
    M = np.zeros((len(cols), len(labs)))
    for j, c in enumerate(cols): M[j, li[IND[c]]] = 1.0
    return M


def ic_series(M, IR, drop=None, start=SEL[0]):
    """與 t4 相同：s＝訊號非 NaN 的產業，少於 10 產業略過；Spearman＝rank 後 Pearson。"""
    if drop: M = M.drop(columns=[c for c in drop if c in M.columns]); IR = IR.drop(columns=[c for c in drop if c in IR.columns])
    M = M[M.index >= pd.Timestamp(start)]; IR = IR.reindex(M.index)
    nsig = M.notna().sum(axis=1)
    msk = M.notna() & IR.notna()
    X = M.where(msk).rank(axis=1); Y = IR.where(msk).rank(axis=1)
    X = X.sub(X.mean(axis=1), axis=0); Y = Y.sub(Y.mean(axis=1), axis=0)
    ic = (X * Y).sum(axis=1) / np.sqrt((X ** 2).sum(axis=1) * (Y ** 2).sum(axis=1))
    return ic[nsig >= 10].dropna()


def seg(ic, a, b):
    x = ic[(ic.index >= pd.Timestamp(a)) & (ic.index <= pd.Timestamp(b))]
    return {'IC': round(float(x.mean()), 3), 't': tstat(x), 'n': int(len(x))}


def nwt(x, L=3):
    x = np.asarray(x, float); m = x.mean(); z = x - m; n = len(z)
    s = np.mean(z * z)
    for l in range(1, L + 1): s += 2 * (1 - l / (L + 1)) * np.mean(z[l:] * z[:-l])
    return round(float(m / math.sqrt(s / n)), 2)


def holm(p):
    k = len(p); order = np.argsort(p); adj = np.empty(k); run = 0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (k - i) * p[j])); adj[j] = run
    return adj


def main():
    t0 = time.time()
    P = prep(); IND0 = load()['IND']
    S, IR = build(P, IND0)
    res = {'notes': {'elec8': ELEC, 'newcat': NEWCAT}}
    # ---- 重現 ----
    ICs = {sn: ic_series(S[sn], IR) for sn in SIGS}
    rep = {}
    for sn in SIGS:
        ic = ICs[sn]
        rep[sn] = {'sel': seg(ic, *SEL), 'val': seg(ic, *VAL), 'sel_NW3_t': nwt(ic[:SEL[1]]), 'val_NW3_t': nwt(ic[VAL[0]:])}
    res['reproduce'] = rep
    print('reproduce', {k: (v['sel']['t'], v['val']['t']) for k, v in rep.items()}, f'{time.time()-t0:.0f}s')

    # ---- (1) 多重檢定 ----
    tsel = np.array([rep[s]['sel']['t'] for s in SIGS]); nsel = np.array([rep[s]['sel']['n'] for s in SIGS])
    p2 = np.array([2 * sst.t.sf(abs(t), n - 1) for t, n in zip(tsel, nsel)])
    p1 = np.array([sst.t.sf(t, n - 1) for t, n in zip(tsel, nsel)])
    mt = {}
    for nm, p in [('two_sided', p2), ('one_sided', p1)]:
        mt[nm] = {s: {'t': float(tsel[i]), 'p_raw': round(float(p[i]), 4), 'p_bonf9': round(float(min(1, 9 * p[i])), 4),
                      'p_holm9': round(float(holm(p)[i]), 4)} for i, s in enumerate(SIGS)}
    tval = np.array([rep[s]['val']['t'] for s in SIGS]); nval = np.array([rep[s]['val']['n'] for s in SIGS])
    pv2 = np.array([2 * sst.t.sf(abs(t), n - 1) for t, n in zip(tval, nval)])
    mt['validation_two_sided'] = {s: {'t': float(tval[i]), 'p_raw': round(float(pv2[i]), 5), 'p_bonf9': round(float(min(1, 9 * pv2[i])), 5),
                                      'p_holm9': round(float(holm(pv2)[i]), 5)} for i, s in enumerate(SIGS)}
    res['multiple_testing'] = mt
    print('MT sel two-sided', {s: (v['p_raw'], v['p_bonf9'], v['p_holm9']) for s, v in mt['two_sided'].items()})

    # 置換 A：每月產業報酬在產業間隨機排列
    NP = 5000
    sel_idx = [t for t in IR.index if pd.Timestamp(SEL[0]) <= t <= pd.Timestamp(SEL[1])]
    permIC = {s: np.full((NP, len(sel_idx)), np.nan) for s in SIGS}
    for ti, t in enumerate(sel_idx):
        y = IR.loc[t]; U = y.index[y.notna()]
        yv = y[U].values
        perm = np.argsort(RNG.random((NP, len(U))), axis=1)
        Yp = pd.DataFrame(yv[perm], columns=U)  # NP x |U|
        for s in SIGS:
            x = S[s].loc[t].dropna()
            if len(x) < 10: continue
            ii = [c for c in x.index if c in U]
            xr = sst.rankdata(x[ii].values); xr = xr - xr.mean()
            yr = sst.rankdata(Yp[ii].values, axis=1); yr = yr - yr.mean(axis=1, keepdims=True)
            permIC[s][:, ti] = (yr @ xr) / np.sqrt((yr ** 2).sum(axis=1) * (xr ** 2).sum())
    def tmat(A):
        n = np.sum(~np.isnan(A), axis=1)
        return np.nanmean(A, axis=1) / (np.nanstd(A, axis=1, ddof=1) / np.sqrt(n))
    PT = np.column_stack([tmat(permIC[s]) for s in SIGS])
    maxT = PT.max(axis=1)
    pa = {}
    for i, s in enumerate(SIGS):
        pa[s] = {'t_obs': float(tsel[i]), 'p_perm_one_sided': round(float((np.sum(PT[:, i] >= tsel[i]) + 1) / (NP + 1)), 4),
                 'p_perm_maxT_family9': round(float((np.sum(maxT >= tsel[i]) + 1) / (NP + 1)), 4)}
    pa['maxT_null_95pct'] = round(float(np.quantile(maxT, 0.95)), 2)
    pa['n_perm'] = NP
    res['perm_A_monthly_shuffle'] = pa
    print('permA', pa, f'{time.time()-t0:.0f}s')

    # 置換 B：股票產業標籤隨機打亂（保持各產業檔數）
    NB = 500
    codes = list(IND0.index); labs = IND0.values.copy()
    bt = {s: [] for s in SIGS}; bv = {s: [] for s in SIGS}
    for b in range(NB):
        INDb = pd.Series(RNG.permutation(labs), index=codes)
        Sb, IRb = build(P, INDb)
        for s in SIGS:
            ic = ic_series(Sb[s], IRb)
            bt[s].append(seg(ic, *SEL)['t']); bv[s].append(seg(ic, *VAL)['t'])
        if b % 20 == 0: print('permB', b, f'{time.time()-t0:.0f}s', flush=True)
    BT = np.array([bt[s] for s in SIGS]).T; BV = np.array([bv[s] for s in SIGS]).T
    pb = {}
    for i, s in enumerate(SIGS):
        pb[s] = {'t_sel_obs': float(tsel[i]), 'null_mean_t_sel': round(float(np.nanmean(BT[:, i])), 2),
                 'null_95pct_t_sel': round(float(np.nanquantile(BT[:, i], 0.95)), 2),
                 'p_sel': round(float((np.sum(BT[:, i] >= tsel[i]) + 1) / (NB + 1)), 4),
                 'p_sel_maxT9': round(float((np.sum(np.nanmax(BT, axis=1) >= tsel[i]) + 1) / (NB + 1)), 4),
                 't_val_obs': float(tval[i]), 'null_mean_t_val': round(float(np.nanmean(BV[:, i])), 2),
                 'null_95pct_t_val': round(float(np.nanquantile(BV[:, i], 0.95)), 2),
                 'p_val': round(float((np.sum(BV[:, i] >= tval[i]) + 1) / (NB + 1)), 4)}
    pb['n_perm'] = NB
    res['perm_B_stock_label_shuffle'] = pb
    print('permB', pb, f'{time.time()-t0:.0f}s')

    # ---- (2) 剔除電子／補歸類／剔除新分類 ----
    ex = {}
    for s in SIGS:
        ic_e = ic_series(S[s], IR, drop=ELEC); ic_n = ic_series(S[s], IR, drop=NEWCAT)
        ic_en = ic_series(S[s], IR, drop=ELEC + NEWCAT)
        ex[s] = {'exElec_sel': seg(ic_e, *SEL), 'exElec_val': seg(ic_e, *VAL), 'exElec_2005_2012': seg(ic_e, '2005-01-01', '2012-12-31'),
                 'exElec_2013_2019_04': seg(ic_e, '2013-01-01', SEL[1]),
                 'exNew_sel': seg(ic_n, *SEL), 'exNew_val': seg(ic_n, *VAL),
                 'exElecNew_sel': seg(ic_en, *SEL), 'exElecNew_val': seg(ic_en, *VAL)}
    res['exclusion'] = ex
    print('exElec sel', {s: ex[s]['exElec_sel']['t'] for s in SIGS})
    print('exNew sel', {s: ex[s]['exNew_sel']['t'] for s in SIGS})
    # 補歸類：130 檔未歸類（TDR 91 開頭除外）隨機分配到電子 8 類（依現有檔數比例）
    unc = [c for c in P['adj'].columns if c not in IND0.index and not c.startswith('91')]
    w = IND0[IND0.isin(ELEC)].value_counts().reindex(ELEC).values.astype(float); w /= w.sum()
    NI = 100; imp = {s: {'sel': [], 'val': [], 'exElec_sel': []} for s in ['RS_52wH', 'DIFF_NH', 'MOM6', 'MOM12']}
    for b in range(NI):
        INDi = pd.concat([IND0, pd.Series(RNG.choice(ELEC, size=len(unc), p=w), index=unc)])
        Si, IRi = build(P, INDi)
        for s in imp:
            ic = ic_series(Si[s], IRi)
            imp[s]['sel'].append(seg(ic, *SEL)['t']); imp[s]['val'].append(seg(ic, *VAL)['t'])
    res['impute_unclassified_to_elec'] = {'n_unclassified_imputed': len(unc), 'n_draws': NI,
                                          **{s: {k: {'mean_t': round(float(np.mean(v)), 2), 'min_t': round(float(np.min(v)), 2),
                                                     'max_t': round(float(np.max(v)), 2)} for k, v in d.items() if v}
                                             for s, d in imp.items()}}
    print('impute', res['impute_unclassified_to_elec'], f'{time.time()-t0:.0f}s')

    # ---- (3) 分段 ----
    segs = {'2005_2012': ('2005-01-01', '2012-12-31'), '2013_2019_04': ('2013-01-01', SEL[1]),
            '2019_05_2023': (VAL[0], '2023-12-31'), '2024_2026': ('2024-01-01', VAL[1])}
    res['subperiod'] = {s: {k: seg(ICs[s], *v) for k, v in segs.items()} for s in SIGS}
    print('subperiod', {s: {k: v['t'] for k, v in res['subperiod'][s].items()} for s in ['RS_52wH', 'DIFF_NH', 'MOM6', 'MOM12']})

    # ---- (4) 殘差化 / Fama-MacBeth / 雙重排序 ----
    def xs_tests(target, ctrls):
        rows = {}
        for t in IR.index:
            if t < pd.Timestamp(SEL[0]): continue
            df = pd.concat([S[target].loc[t].rename('x')] + [S[c].loc[t].rename(c) for c in ctrls] + [IR.loc[t].rename('y')], axis=1).dropna()
            if len(df) < 10: continue
            Rk = df.rank(); sd = Rk.std(ddof=0)
            if (sd < 1e-9).any(): continue
            Z = (Rk - Rk.mean()) / sd
            Xc = np.column_stack([np.ones(len(Z))] + [Z[c].values for c in ctrls])
            bx, *_ = np.linalg.lstsq(Xc, Z['x'].values, rcond=None)
            resid = Z['x'].values - Xc @ bx
            ic_res = np.corrcoef(resid, Z['y'].values)[0, 1] if resid.std() > 1e-9 else np.nan
            Xf = np.column_stack([Xc, Z['x'].values])
            bf, *_ = np.linalg.lstsq(Xf, Z['y'].values, rcond=None)
            # 雙重排序：依第一個控制變數分上下半，各半內依 target 上下半，報酬價差平均
            c0 = ctrls[0]; med = Z[c0].median(); spr = []
            for half in [Z[c0] > med, Z[c0] <= med]:
                sub = df[half.values]
                if len(sub) < 4: continue
                o = sub.sort_values('x'); h = len(o) // 2
                spr.append(o['y'].iloc[-h:].mean() - o['y'].iloc[:h].mean())
            rows[t] = {'ic_res': ic_res, 'fm_x': bf[-1], 'ds': np.mean(spr) if spr else np.nan,
                       'corr_x_c0': np.corrcoef(Z['x'], Z[c0])[0, 1], 'ic_raw': np.corrcoef(Z['x'], Z['y'])[0, 1]}
        D = pd.DataFrame(rows).T
        o = {}
        for nm, (a, b) in {'sel': SEL, 'val': VAL, '2005_2012': segs['2005_2012'], '2013_2019_04': segs['2013_2019_04'], '2024_2026': segs['2024_2026']}.items():
            x = D[(D.index >= pd.Timestamp(a)) & (D.index <= pd.Timestamp(b))]
            o[nm] = {'IC_raw_common': round(float(x['ic_raw'].mean()), 3), 't_IC_raw_common': tstat(x['ic_raw'].dropna()),
                     'IC_resid': round(float(x['ic_res'].mean()), 3), 't_IC_resid': tstat(x['ic_res'].dropna()),
                     'FM_coef': round(float(x['fm_x'].mean()), 4), 't_FM': tstat(x['fm_x'].dropna()),
                     'dsort_spread_pct_m': round(float(x['ds'].mean() * 100), 3), 't_dsort': tstat(x['ds'].dropna()),
                     'mean_rank_corr_with_' + ctrls[0]: round(float(x['corr_x_c0'].mean()), 2), 'n': int(len(x))}
        return o
    orth = {}
    for tg in ['RS_52wH', 'DIFF_NH']:
        orth[tg] = {'ctrl_MOM6': xs_tests(tg, ['MOM6']), 'ctrl_MOM12': xs_tests(tg, ['MOM12']),
                    'ctrl_MOM6_MOM12': xs_tests(tg, ['MOM6', 'MOM12']), 'ctrl_MOM1_3_6_12': xs_tests(tg, ['MOM6', 'MOM1', 'MOM3', 'MOM12'])}
    res['orthogonalize'] = orth
    for tg in orth:
        for k, v in orth[tg].items(): print(tg, k, {p: (v[p]['t_IC_resid'], v[p]['t_FM'], v[p]['t_dsort']) for p in ['sel', 'val']})
    p = OUT / 'recheck_r1_1.json'
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    print('saved', p, f'{time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
