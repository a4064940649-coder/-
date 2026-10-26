#!/usr/bin/env python3
"""檢定 3：產業間領先落後（上市股，月底換股）。

事前固定：
* 產業月報酬：產業等權日指數（成分＝前一日合格股）；預測變數用換股日前一交易日為止的上月報酬（lag 1 日）。
* 估計期 2005-01～2019-04，對每一對 A≠B：
    M1（主）：r_B(t) = a + b·r_A(t−1) + c·r_B(t−1) + d·r_M(t−1)（控制自身與大盤動能；r_M＝合格股等權）
    M2：r_B(t) = a + b·r_A(t−1)（簡單版）
  挑選 b 的 t ≥ 3 的配對（正向）；另報 t ≤ −3 的數目與獨立虛無下的期望假陽性數。
* 交易（2019-05～2026-10 驗證；也報 2005～2019-04 樣本內做參考）：每月底對有被選中預測者的產業 B，
  預測值＝用估計期係數算的模型擬合值（多個 A 取平均），買預測最強的 3 個產業：liq10（每產業流動前 10 檔、全部等權）／indEW。
  另一變體 M1_t2：門檻放寬為 t ≥ 2。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from scipy import stats as sst
from ilib import (load, eligibility, industry_daily_index, build_weights, run_weights, stats, brief, universe_ew,
                  period_returns, bench_returns, month_ends, prev_day, save, tstat)

SEL = ('2005-01-01', '2019-04-30'); VAL = ('2019-05-01', '2026-12-31')


def ols(y, X):
    df = pd.concat([y.rename('y')] + [x.rename(f'x{i}') for i, x in enumerate(X)], axis=1).dropna()
    if len(df) < 36: return np.full(len(X) + 1, np.nan), np.full(len(X) + 1, np.nan), len(df)
    Y = df.iloc[:, 0].values; Z = np.column_stack([np.ones(len(df))] + [df.iloc[:, i].values for i in range(1, df.shape[1])])
    b, *_ = np.linalg.lstsq(Z, Y, rcond=None); e = Y - Z @ b
    s2 = e @ e / (len(Y) - Z.shape[1]); se = np.sqrt(np.diag(s2 * np.linalg.pinv(Z.T @ Z)))
    return b, b / se, len(Y)


def main():
    d = load(); adj = d['adj']
    IDX = industry_daily_index(); N = d['IDX_N']; MKT = d['MKT_EW']
    me = [x for x in month_ends(adj.index) if x >= pd.Timestamp('2004-10-01')]
    pdays = prev_day(me, adj.index)
    I_me = IDX.reindex(me); I_pd = IDX.reindex(pdays).set_axis(pd.DatetimeIndex(me))
    okn = (N.reindex(pdays).set_axis(pd.DatetimeIndex(me)) >= 3) & (N.reindex(pdays).shift(1).set_axis(pd.DatetimeIndex(me)) >= 3)
    Y = (I_me.shift(-1) / I_me - 1).where(okn.shift(-1).fillna(False).astype(bool))   # 持有期 t→t+1（index=t）
    Xs = (I_pd / I_pd.shift(1) - 1).where(okn)                                          # 訊號（index=t，到 t 前一日）
    M_pd = MKT.reindex(pdays).set_axis(pd.DatetimeIndex(me)); XM = M_pd / M_pd.shift(1) - 1
    inds = [k for k in IDX.columns]
    sel_idx = (Y.index >= pd.Timestamp(SEL[0])) & (Y.index <= pd.Timestamp(SEL[1]))
    rows = []
    for B_ in inds:
        for A_ in inds:
            if A_ == B_: continue
            y = Y[B_][sel_idx]
            if y.notna().sum() < 60: continue
            if Xs[A_][sel_idx].notna().sum() < 60: continue
            b1, t1, n1 = ols(y, [Xs[A_][sel_idx], Xs[B_][sel_idx], XM[sel_idx]])
            b2, t2, n2 = ols(y, [Xs[A_][sel_idx]])
            rows.append({'A': A_, 'B': B_, 'M1_b': b1[1], 'M1_t': t1[1], 'M1_coef': list(b1), 'M2_b': b2[1], 'M2_t': t2[1], 'M2_coef': list(b2), 'n': n1})
    T = pd.DataFrame(rows)
    ntest = len(T); p3 = 1 - sst.norm.cdf(3); p2 = 1 - sst.norm.cdf(2)
    res = {'n_pairs_tested': ntest, 'expected_false_pos_t>=3_if_independent': round(ntest * p3, 2),
           'expected_false_pos_t>=2_if_independent': round(ntest * p2, 1),
           'M1_t>=3': int((T.M1_t >= 3).sum()), 'M1_t<=-3': int((T.M1_t <= -3).sum()), 'M1_t>=2': int((T.M1_t >= 2).sum()),
           'M2_t>=3': int((T.M2_t >= 3).sum()), 'M2_t<=-3': int((T.M2_t <= -3).sum()), 'M2_t>=2': int((T.M2_t >= 2).sum())}
    # 驗證期配對穩定度
    val_idx = Y.index >= pd.Timestamp(VAL[0])
    def val_t(r, model):
        y = Y[r['B']][val_idx]
        X = [Xs[r['A']][val_idx], Xs[r['B']][val_idx], XM[val_idx]] if model == 'M1' else [Xs[r['A']][val_idx]]
        b, t, n = ols(y, X); return round(float(b[1]), 3), round(float(t[1]), 2)
    pairs = {}
    for model, thr in [('M1', 3), ('M2', 3), ('M1', 2)]:
        S = T[T[f'{model}_t'] >= thr].sort_values(f'{model}_t', ascending=False)
        lst = []
        for _, r in S.iterrows():
            vb, vt = val_t(r, model)
            lst.append({'A': r['A'], 'B': r['B'], 'sel_b': round(r[f'{model}_b'], 3), 'sel_t': round(r[f'{model}_t'], 2), 'val_b': vb, 'val_t': vt})
        pairs[f'{model}_t{thr}'] = lst
    res['pairs'] = {k: v if len(v) <= 40 else v[:40] for k, v in pairs.items()}
    res['pairs_val_t_mean'] = {k: round(float(np.nanmean([x['val_t'] for x in v])), 2) if v else None for k, v in pairs.items()}
    res['pairs_val_share_positive'] = {k: round(float(np.mean([x['val_b'] > 0 for x in v if x['val_b'] == x['val_b']])), 2) if v else None for k, v in pairs.items()}
    neg = T[T.M1_t <= -3].sort_values('M1_t')
    res['M1_negative_pairs'] = [{'A': r['A'], 'B': r['B'], 'sel_t': round(r['M1_t'], 2)} for _, r in neg.iterrows()][:20]

    # 交易
    R = period_returns(adj, me); Bm = bench_returns(d['e50'], me)
    el, v = eligibility(me); U = universe_ew(R, el)
    IND = d['IND']
    cnt = pd.DataFrame({t: el.loc[t][el.loc[t]].index.map(lambda c: IND.get(c)).value_counts() for t in R.index}).T.fillna(0)
    res['trading'] = {}
    for key in ['M1_t3', 'M2_t3', 'M1_t2']:
        model = key[:2]; thr = int(key[-1])
        S = T[T[f'{model}_t'] >= thr]
        if S.empty: continue
        preds = {}
        for t in R.index:
            p = {}
            for B_, g in S.groupby('B'):
                vals = []
                for _, r in g.iterrows():
                    c = r[f'{model}_coef']
                    x = [Xs.at[t, r['A']]] + ([Xs.at[t, B_], XM[t]] if model == 'M1' else [])
                    if any(pd.isna(z) for z in x): continue
                    vals.append(c[0] + sum(ci * xi for ci, xi in zip(c[1:], x)))
                if vals and cnt.loc[t].get(B_, 0) >= 5: p[B_] = np.mean(vals)
            preds[t] = p
        for mode in ['liq10', 'indEW']:
            W = {}
            for t, p in preds.items():
                if len(p) < 1: continue
                top = [k for k, _ in sorted(p.items(), key=lambda kv: -kv[1])[:3]]
                W[t] = build_weights(t, top, el.loc[t], v.loc[t], mode)
            g, to = run_weights(W, R)
            res['trading'][f'{key}|{mode}'] = {'in_sample_ref': brief(stats(g, to, me, Bm, U, *SEL)), 'validation': brief(stats(g, to, me, Bm, U, *VAL)),
                                               'n_target_industries': int(S.B.nunique()),
                                               'validation_full': stats(g, to, me, Bm, U, *VAL)}
    save('results_t3.json', res)
    print({k: v for k, v in res.items() if k not in ('pairs', 'trading', 'M1_negative_pairs')})
    for k, v in res['pairs'].items():
        print(k, len(v)); [print('   ', x) for x in v[:15]]
    for k, v in res['trading'].items(): print(k, '\n  IS ', v['in_sample_ref'], '\n  VAL', v['validation'])


if __name__ == '__main__':
    main()
