#!/usr/bin/env python3
"""檢定 4：產業動能的不同定義（上市股，月底換股，訊號用前一交易日收盤）。

事前固定的訊號（產業層級，越大越好）：
* MOM1 / MOM3 / MOM6 / MOM12：產業等權指數近 1、3、6、12 個月報酬（另報 MOM12_1：12 個月扣最近 1 個月，作為對照）。
* 相對大盤強度（注意：單純「產業報酬−大盤報酬」與 MOMk 排序完全相同，故改用相對強度線 RS＝產業指數／0050 含息）：
    RS_52wH：RS 線 / 其 252 日最高；RS_MA60：RS 線 / 其 60 日均線 − 1。
* 擴散度：DIFF_NH＝產業內合格股中，近 21 個交易日曾創 252 日新高（還原價）的比例；DIFF_DIST＝產業內合格股 收盤/252 日最高 的平均。
組合：產業需 ≥5 檔合格股；買前 3 產業：liq10（每產業流動前 10 檔、全部等權）／indEW。
另報：前 3 − 後 3 產業等權毛報酬價差、產業層級 rank IC。
挑選 2005-01～2019-04、驗證 2019-05～2026-10；挑選準則＝挑選期淨報酬對 0050 的 t 值最高者。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from ilib import (load, eligibility, industry_daily_index, build_weights, run_weights, stats, brief, universe_ew,
                  period_returns, bench_returns, month_ends, prev_day, save, tstat)

SEL = ('2005-01-01', '2019-04-30'); VAL = ('2019-05-01', '2026-12-31')


def signals(me):
    d = load(); adj, IND = d['adj'], d['IND']
    IDX = industry_daily_index(); N = d['IDX_N']
    pdays = prev_day(me, adj.index); DI = pd.DatetimeIndex(me)
    Ipd = IDX.reindex(pdays).set_axis(DI)
    ok = (N.rolling(252, min_periods=1).min().reindex(pdays).set_axis(DI) >= 3)
    S = {}
    for k in [1, 3, 6, 12]:
        S[f'MOM{k}'] = (Ipd / Ipd.shift(k) - 1).where(ok)
    S['MOM12_1'] = (Ipd.shift(1) / Ipd.shift(12) - 1).where(ok)
    e = d['e50'].reindex(adj.index).ffill()
    RS = IDX.div(e, axis=0)
    S['RS_52wH'] = (RS / RS.rolling(252, min_periods=200).max()).reindex(pdays).set_axis(DI).where(ok)
    S['RS_MA60'] = (RS / RS.rolling(60).mean() - 1).reindex(pdays).set_axis(DI).where(ok)
    A = adj.ffill(limit=5)
    mx252 = A.rolling(252, min_periods=240).max()
    nh = (A.rolling(21).max() >= mx252 - 1e-12).where(mx252.notna())
    dist = A / mx252
    nh = nh.reindex(pdays).set_axis(DI); dist = dist.reindex(pdays).set_axis(DI)
    return S, nh, dist


def main():
    d = load(); adj, IND = d['adj'], d['IND']
    me = [x for x in month_ends(adj.index) if x >= pd.Timestamp('2004-01-01')]
    R = period_returns(adj, me); B = bench_returns(d['e50'], me)
    el, v = eligibility(me); U = universe_ew(R, el)
    S, nh, dist = signals(me)
    cols = [c for c in adj.columns if c in IND.index]
    elc = el[cols]
    def ind_mean(M):
        Mx = M[cols].where(elc)
        return Mx.T.groupby(IND[cols]).mean().T
    S['DIFF_NH'] = ind_mean(nh.astype(float))
    S['DIFF_DIST'] = ind_mean(dist)
    cnt = elc.T.groupby(IND[cols]).sum().T
    # 產業等權持有期報酬（合格股）
    IR = R[cols].where(elc).T.groupby(IND[cols]).mean().T
    res = {'configs': {}, 'spread': {}}
    for sn, M in S.items():
        M = M.where(cnt.reindex(M.index) >= 5)
        if sn.startswith('DIFF'):
            nn = nh[cols].where(elc).notna().T.groupby(IND[cols]).sum().T
            M = M.where(nn >= 5)
        top_by_t, sp, ic = {}, {}, {}
        for t in R.index:
            if t < pd.Timestamp(SEL[0]): continue
            s = M.loc[t].dropna()
            if len(s) < 10: continue
            o = s.sort_values(ascending=False)
            top_by_t[t] = list(o.index[:3])
            ir = IR.loc[t, o.index]
            sp[t] = ir.iloc[:3].mean() - ir.iloc[-3:].mean()
            ic[t] = s.rank().corr(IR.loc[t, s.index].rank())
        sp, ic = pd.Series(sp), pd.Series(ic)
        res['spread'][sn] = {'sel_top3_minus_bot3_pct_m': round(float(sp[:SEL[1]].mean() * 100), 3), 't_sel': tstat(sp[:SEL[1]]),
                             'val_top3_minus_bot3_pct_m': round(float(sp[VAL[0]:].mean() * 100), 3), 't_val': tstat(sp[VAL[0]:]),
                             'IC_sel': round(float(ic[:SEL[1]].mean()), 3), 't_IC_sel': tstat(ic[:SEL[1]]),
                             'IC_val': round(float(ic[VAL[0]:].mean()), 3), 't_IC_val': tstat(ic[VAL[0]:])}
        for mode in ['liq10', 'indEW']:
            W = {t: build_weights(t, k, el.loc[t], v.loc[t], mode) for t, k in top_by_t.items()}
            g, to = run_weights(W, R)
            res['configs'][f'{sn}|{mode}'] = {'selection': stats(g, to, me, B, U, *SEL), 'validation': stats(g, to, me, B, U, *VAL),
                                              'sub_2021_2026': brief(stats(g, to, me, B, U, '2021-01-01', '2026-12-31'))}
        res['configs'][f'{sn}|liq10']['latest_top3'] = {str(t.date()): k for t, k in list(top_by_t.items())[-3:]}
    cand = {k: v['selection']['t_vs_0050'] for k, v in res['configs'].items() if 't_vs_0050' in v['selection'] and not k.startswith('MOM12_1')}
    best = max(cand, key=cand.get)
    res['selected'] = {'config': best, 'selection': brief(res['configs'][best]['selection']), 'validation': brief(res['configs'][best]['validation'])}
    res['summary'] = {k: {'sel': brief(v['selection']), 'val': brief(v['validation'])} for k, v in res['configs'].items()}
    save('results_t4.json', res)
    for k, v in res['summary'].items(): print(k, '\n  SEL', v['sel'], '\n  VAL', v['val'])
    for k, v in res['spread'].items(): print(k, v)
    print('SELECTED', res['selected'])


if __name__ == '__main__':
    main()
