#!/usr/bin/env python3
"""檢定 1：產業營收循環（上市股，2021-03～2026-10）。

事前固定：
* 換股日：每月第一個 ≥11 日的交易日收盤；使用上個月營收（10 日前公告）。
* 產業加總營收：產業內有申報的上市股營收加總；年增率 = Σ營收 / Σ(營收/(1+yoy))－1（同一批股票）；需 ≥3 家申報。
* 訊號：S1 產業單月年增率；S1_3m 近 3 個月加總年增率；S2 加速度＝本月年增率－3 個月前年增率；
        S3 12 個月新高程度＝本月加總營收 / 前 11 個月最高（同一批股票），≥1 即創 12 個月新高。
* 組合：每期排序產業（產業需 ≥5 檔合格股），買前 3 名：liq10＝每產業流動性前 10 檔、全部等權；indEW＝產業等權。
* 挑選 2021-03～2023-12，驗證 2024-01～2026-10；挑選準則＝挑選期淨報酬對 0050 月超額平均最高者。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from ilib import load, eligibility, build_weights, run_weights, stats, brief, universe_ew, period_returns, bench_returns, save, X

SEL = ('2021-03-01', '2023-12-31'); VAL = ('2024-01-01', '2026-12-31')


def revenue_matrix():
    a = pd.read_csv(X / 'rev_5y.csv.gz', dtype={'code': str}); b = pd.read_csv(X / 'rev_hist.csv.gz', dtype={'code': str})
    r = pd.concat([a[['code', 'ym', 'revenue', 'yoy']], b[['code', 'ym', 'revenue', 'yoy']]]).drop_duplicates(['code', 'ym'], keep='last')
    r['p'] = pd.PeriodIndex(r['ym'], freq='M')
    REV = r.pivot_table(index='p', columns='code', values='revenue', aggfunc='last')
    YOY = r.pivot_table(index='p', columns='code', values='yoy', aggfunc='last')
    full = pd.period_range(REV.index.min() - 12, REV.index.max(), freq='M')
    REV = REV.reindex(full); YOY = YOY.reindex(full)
    ok = (YOY > -99.5) & (YOY < 5000) & (REV > 0)
    PREV = (REV / (1 + YOY / 100)).where(ok)
    # 反推一年前營收以補齊 2019-12～2020-11
    back = PREV.shift(-12)
    REVF = REV.where(REV > 0).combine_first(back)
    return REVF, PREV, REV.where(ok)


def industry_signals(IND, codes):
    REVF, PREV, REVOK = revenue_matrix()
    cols = [c for c in codes if c in REVF.columns and c in IND.index]
    REVF, PREV, REVOK = REVF[cols], PREV[cols], REVOK[cols]
    inds = sorted(IND[cols].unique())
    S1, S13, S3, NREP = {}, {}, {}, {}
    months = REVOK.index[REVOK.notna().sum(axis=1) > 100]
    for m in months:
        r1, r2, r3 = {}, {}, {}
        for k in inds:
            mem = [c for c in cols if IND[c] == k]
            cur, prv = REVOK.loc[m, mem], PREV.loc[m, mem]
            okm = cur.notna() & prv.notna()
            if okm.sum() >= 3: r1[k] = cur[okm].sum() / prv[okm].sum() - 1
            win = [m - i for i in range(3)]
            if all(w in REVOK.index for w in win):
                c3, p3 = REVOK.loc[win, mem], PREV.loc[win, mem]
                ok3 = (c3.notna() & p3.notna()).all()
                if ok3.sum() >= 3: r2[k] = c3.loc[:, ok3].values.sum() / p3.loc[:, ok3].values.sum() - 1
            hw = [m - i for i in range(12)]
            if all(w in REVF.index for w in hw):
                L = REVF.loc[hw, mem]; okh = L.notna().all()
                if okh.sum() >= 3:
                    s = L.loc[:, okh].sum(axis=1)
                    r3[k] = s.loc[m] / s.drop(m).max()
            NREP.setdefault(m, {})[k] = int(okm.sum())
        S1[m], S13[m], S3[m] = r1, r2, r3
    S1 = pd.DataFrame(S1).T; S13 = pd.DataFrame(S13).T; S3 = pd.DataFrame(S3).T
    S2 = S1 - S1.shift(3)
    return {'S1_yoy': S1, 'S1_3m_yoy': S13, 'S2_accel': S2, 'S3_12m_high': S3}


def main():
    d = load(); adj, IND = d['adj'], d['IND']
    ii = adj.index[adj.index >= pd.Timestamp('2021-03-01')]; s = pd.Series(ii, index=ii)
    ds = [g[g.dt.day >= 11].iloc[0] for _, g in s.groupby([ii.year, ii.month]) if (g.dt.day >= 11).any()]
    dates = ds
    R = period_returns(adj, dates); B = bench_returns(d['e50'], dates)
    el, v = eligibility(dates)
    U = universe_ew(R, el)
    sig = industry_signals(IND, list(adj.columns))
    # 產業合格股數
    cnt = pd.DataFrame({t: el.loc[t][el.loc[t]].index.map(lambda c: IND.get(c)).value_counts() for t in R.index}).T.fillna(0)
    res = {'note': '上市股；產業=stock_list 現行分類', 'dates': [str(x.date()) for x in dates], 'configs': {}}
    rank_log = {}
    for sn, S in sig.items():
        for mode in ['liq10', 'indEW']:
            W, picks = {}, {}
            for t in R.index:
                m = pd.Period(t, 'M') - 1
                if m not in S.index: continue
                sc = S.loc[m].dropna()
                sc = sc[[k for k in sc.index if cnt.loc[t].get(k, 0) >= 5]]
                if len(sc) < 10: continue
                top = list(sc.sort_values(ascending=False).index[:3])
                picks[t] = top
                W[t] = build_weights(t, top, el.loc[t], v.loc[t], mode)
            g, to = run_weights(W, R)
            res['configs'][f'{sn}|{mode}'] = {'selection': stats(g, to, dates, B, U, *SEL), 'validation': stats(g, to, dates, B, U, *VAL),
                                              'full': stats(g, to, dates, B, U)}
            if mode == 'liq10':
                rank_log[sn] = {str(t.date()): p for t, p in picks.items()}
                # 產業層級：前 3 名 vs 後 3 名產業等權（毛報酬）價差
                sp = {}
                for t in R.index:
                    m = pd.Period(t, 'M') - 1
                    if m not in S.index: continue
                    sc = S.loc[m].dropna(); sc = sc[[k for k in sc.index if cnt.loc[t].get(k, 0) >= 5]]
                    if len(sc) < 10: continue
                    o = sc.sort_values(ascending=False)
                    def iew(ks):
                        return np.mean([R.loc[t][[c for c in el.columns[el.loc[t].values] if IND.get(c) == k]].mean() for k in ks])
                    sp[t] = iew(o.index[:3]) - iew(o.index[-3:])
                sp = pd.Series(sp)
                from btlib import tstat
                res['configs'][f'{sn}|top3_minus_bottom3_gross'] = {
                    'selection_pct_m': round(float(sp[:SEL[1]].mean() * 100), 3), 't_sel': tstat(sp[:SEL[1]]),
                    'validation_pct_m': round(float(sp[VAL[0]:].mean() * 100), 3), 't_val': tstat(sp[VAL[0]:])}
    # 挑選
    cand = {k: v['selection']['ex_vs_0050_pct_m'] for k, v in res['configs'].items() if 'selection' in v and 'ex_vs_0050_pct_m' in v['selection']}
    best = max(cand, key=cand.get)
    res['selected_in_sample'] = best
    res['selected_validation'] = brief(res['configs'][best]['validation'])
    res['summary'] = {k: {'sel': brief(v['selection']), 'val': brief(v['validation'])} for k, v in res['configs'].items() if 'selection' in v}
    res['spreads'] = {k: v for k, v in res['configs'].items() if 'top3_minus' in k}
    res['latest_picks'] = {sn: dict(list(rl.items())[-3:]) for sn, rl in rank_log.items()}
    res['rank_log'] = rank_log
    save('results_t1.json', res)
    for k, v in res['summary'].items(): print(k, '\n  SEL', v['sel'], '\n  VAL', v['val'])
    for k, v in res['spreads'].items(): print(k, v)
    print('selected', best, res['selected_validation'])


if __name__ == '__main__':
    main()
