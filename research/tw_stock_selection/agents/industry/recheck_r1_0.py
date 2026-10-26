#!/usr/bin/env python3
"""重測 r1_0：檢定 4 事前選中規則 vs FINDINGS 所報 RS_52wH|liq10。

* 從頭重跑 t4_ind_momentum.py 的 18 個組態（同一訊號、同一權重函式，不讀舊 results_t4.json 的數字）。
* 依 t4 事前寫的挑選準則（挑選期淨報酬對 0050 t 值最高，排除 MOM12_1）排名，得出正式選中組態。
* 列出 DIFF_DIST|indEW（選中）、RS_52wH|liq10、RS_52wH|indEW 的挑選期／驗證期。
* RS_52wH|liq10 驗證期去掉 2026：stats(g,to,me,B,U,'2019-05-01','2025-12-31')。
* 另檢查：各組態挑選期樣本期數不同 → 用共同期數重排一次（穩健性）。
"""
import sys, json, math
from pathlib import Path
HERE = Path('/home/user/-/research/tw_stock_selection/agents/industry')
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
from ilib import (load, eligibility, build_weights, run_weights, stats, brief, universe_ew,
                  period_returns, bench_returns, month_ends, tstat, save)
from t4_ind_momentum import signals, SEL, VAL


def main():
    d = load(); adj, IND = d['adj'], d['IND']
    me = [x for x in month_ends(adj.index) if x >= pd.Timestamp('2004-01-01')]
    R = period_returns(adj, me); B = bench_returns(d['e50'], me)
    el, v = eligibility(me); U = universe_ew(R, el)
    S, nh, dist = signals(me)
    cols = [c for c in adj.columns if c in IND.index]
    elc = el[cols]
    def ind_mean(M):
        return M[cols].where(elc).T.groupby(IND[cols]).mean().T
    S['DIFF_NH'] = ind_mean(nh.astype(float)); S['DIFF_DIST'] = ind_mean(dist)
    cnt = elc.T.groupby(IND[cols]).sum().T
    runs, cfg = {}, {}
    for sn, M in S.items():
        M = M.where(cnt.reindex(M.index) >= 5)
        if sn.startswith('DIFF'):
            nn = nh[cols].where(elc).notna().T.groupby(IND[cols]).sum().T
            M = M.where(nn >= 5)
        top = {}
        for t in R.index:
            if t < pd.Timestamp(SEL[0]): continue
            s = M.loc[t].dropna()
            if len(s) < 10: continue
            top[t] = list(s.sort_values(ascending=False).index[:3])
        for mode in ['liq10', 'indEW']:
            W = {t: build_weights(t, k, el.loc[t], v.loc[t], mode) for t, k in top.items()}
            g, to = run_weights(W, R)
            k = f'{sn}|{mode}'
            runs[k] = (g, to)
            cfg[k] = {'sel': stats(g, to, me, B, U, *SEL), 'val': stats(g, to, me, B, U, *VAL)}
    # 事前準則排名
    cand = {k: c['sel']['t_vs_0050'] for k, c in cfg.items() if not k.startswith('MOM12_1')}
    rank = sorted(cand, key=cand.get, reverse=True)
    selected = rank[0]
    # 共同期數排名（穩健性）
    nets = {k: (g - to.reindex(g.index).fillna(0) * 0.005) for k, (g, to) in runs.items()}
    common = None
    for k in cand:
        ix = nets[k].dropna().loc[SEL[0]:SEL[1]].index
        common = ix if common is None else common.intersection(ix)
    t_common = {k: tstat(nets[k].reindex(common) - B.reindex(common)) for k in cand}
    rank_common = sorted(t_common, key=t_common.get, reverse=True)

    out = {'rank_by_sel_t': [{'rank': i + 1, 'config': k, 'sel_t_0050': cand[k], 'sel_n': cfg[k]['sel']['net']['periods'],
                              'sel_CAGR': cfg[k]['sel']['net']['CAGR'], 'sel_0050_CAGR': cfg[k]['sel']['0050']['CAGR'],
                              'val_CAGR': cfg[k]['val']['net']['CAGR'], 'val_t_0050': cfg[k]['val']['t_vs_0050'],
                              'val_beat': cfg[k]['val']['years_beat_0050']} for i, k in enumerate(rank)],
           'selected_ex_ante': selected,
           'rank_common_sample': {'n_common': int(len(common)), 'start': str(common[0].date()),
                                  'order': [(k, t_common[k]) for k in rank_common]},
           'val_rank_by_CAGR': sorted([(k, cfg[k]['val']['net']['CAGR']) for k in cfg], key=lambda x: -x[1])}
    focus = {}
    for k in [selected, 'RS_52wH|liq10', 'RS_52wH|indEW']:
        g, to = runs[k]
        f = {'sel': brief(cfg[k]['sel']), 'val': brief(cfg[k]['val']),
             'val_yearly': cfg[k]['val']['yearly_net'], 'val_yearly_0050': cfg[k]['val']['yearly_0050'],
             'rank_sel_t': rank.index(k) + 1 if k in rank else None}
        s25 = stats(g, to, me, B, U, '2019-05-01', '2025-12-31')
        f['val_2019_05_to_2025_12'] = brief(s25)
        f['val_2019_05_to_2025_12_EW_CAGR'] = s25.get('EW_univ', {}).get('CAGR')
        # 累積倍數 2019-05～2025-12
        net = (g - to.reindex(g.index).fillna(0) * 0.005).dropna()
        f['cum_mult_2019_05_2025_12'] = round(float(np.prod([1 + y / 100 for yy, y in s25['yearly_net'].items()])), 2)
        f['cum_mult_0050_2019_05_2025_12'] = round(float(np.prod([1 + y / 100 for yy, y in s25['yearly_0050'].items()])), 2)
        # 注意：stats 依「換股日（期初）」切；end='2025-12-31' 含 2025-12-31 起持有到 2026-01-30 的一期（即 2026 年 1 月）。
        # 嚴格「持有期在 2025 年底前結束」＝期初 ≤ 2025-11-30。
        s25s = stats(g, to, me, B, U, '2019-05-01', '2025-11-30')
        f['val_hold_2019_06_to_2025_12_strict'] = brief(s25s)
        f['cum_mult_strict'] = round(float(np.prod([1 + y / 100 for y in s25s['yearly_net'].values()])), 2)
        f['cum_mult_0050_strict'] = round(float(np.prod([1 + y / 100 for y in s25s['yearly_0050'].values()])), 2)
        s26 = stats(g, to, me, B, U, '2026-01-01', '2026-12-31')
        f['y2026'] = {'net_pct': s26['yearly_net'] if 'yearly_net' in s26 else None, '0050_pct': s26.get('yearly_0050')} \
            if 'yearly_net' in s26 else {'periods': s26.get('periods')}
        focus[k] = f
    # 2026 單獨（期數 <6 時 stats 不算，直接累積）
    for k in focus:
        g, to = runs[k]
        net = (g - to.reindex(g.index).fillna(0) * 0.005).dropna()
        n26 = net[net.index >= pd.Timestamp('2025-12-31')]  # 持有期結束在 2026（1–10 月）
        b26 = B.reindex(n26.index)
        focus[k]['y2026_raw'] = {'periods': int(len(n26)), 'net_pct': round(float(((1 + n26).prod() - 1) * 100), 1),
                                 '0050_pct': round(float(((1 + b26).prod() - 1) * 100), 1),
                                 'first': str(n26.index[0].date()), 'last': str(n26.index[-1].date())}
    out['focus'] = focus
    save('recheck_r1_0.json', out)
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
