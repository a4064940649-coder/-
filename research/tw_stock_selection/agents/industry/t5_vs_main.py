#!/usr/bin/env python3
"""檢定 5：產業訊號與中心主策略（bt_final_rule：52 週高點 × 近 3 月營收年增率，前 300 流動、處置濾網、30 檔）。

事前固定：
* 候選產業訊號（在主策略換股日的前一交易日計算；營收類用上月營收）：
    RS_52wH（產業相對 0050 強度線 / 252 日高）、DIFF_NH（產業內近 21 日創 52 週新高比例）、REV_S1_3m（產業近 3 月營收年增率，檢定 1）。
* (a) 相關性：各產業策略（前 3 產業 liq10，於主策略同一換股日）與主策略的月報酬、對 0050 超額報酬相關係數；50/50 混合績效。
* (b) 主策略只在強勢產業內選股：主策略池限制為 產業排名前 50%／前 1/3／剔除後 25% 的產業，其餘規則不變（30 檔）。
  挑選 2021～2023，驗證 2024～2026-10。主策略本身沿用其原始設計（訊號與成交皆為換股日收盤），只加產業遮罩。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
sys.path.insert(0, "/home/user/-/research/tw_stock_selection")
import bt_final_rule as M  # noqa: E402  (載入主策略的全域資料)
from ilib import load, eligibility, build_weights, run_weights, stats, brief, universe_ew, save, ann, yearly, tstat
from t4_ind_momentum import signals
from t1_revenue_cycle import industry_signals

SEL = ('2021-01-01', '2023-12-31'); VAL = ('2024-01-01', '2026-12-31')


def run_main(mask=None, n=30, rt=0.005):
    el0 = M.t300 & M.yoy3.notna() & M.s52.notna() & ~M.inD
    el = el0 if mask is None else (el0 & mask.reindex(index=el0.index, columns=el0.columns).fillna(False).astype(bool))
    # 分數沿用原規則：在（遮罩後）池內做百分位
    sc = 0.5 * M.s52.where(el).rank(axis=1, pct=True) + 0.5 * M.yoy3.where(el).rank(axis=1, pct=True)
    port, to, prev, npk = {}, {}, None, {}
    for t in M.R.index:
        ok = el.loc[t] & M.R.loc[t].notna()
        if ok.sum() < 30: continue
        pick = list(sc.loc[t][ok].sort_values(ascending=False).index[:n])
        port[t] = M.R.loc[t][pick].mean(); npk[t] = len(pick)
        to[t] = len(set(pick) - prev) / len(pick) if prev is not None else 1.0
        prev = set(pick)
    g = pd.Series(port); to = pd.Series(to)
    return g, to


def mk_stats(g, to, dates, B, U):
    return {'full': brief(stats(g, to, dates, B, U)), 'sel': brief(stats(g, to, dates, B, U, *SEL)), 'val': brief(stats(g, to, dates, B, U, *VAL))}


def main():
    d = load(); IND = d['IND']
    dates = M.dates; B = M.B
    U300 = pd.Series({t: M.R.loc[t][M.t300.loc[t] & M.R.loc[t].notna()].mean() for t in M.R.index})
    S, nh, dist = signals(dates)
    el, v = eligibility(dates)
    cols = [c for c in d['adj'].columns if c in IND.index]
    elc = el[cols]
    cnt = elc.T.groupby(IND[cols]).sum().T
    DIFF = nh[cols].astype(float).where(elc).T.groupby(IND[cols]).mean().T
    nn = nh[cols].where(elc).notna().T.groupby(IND[cols]).sum().T
    rev = industry_signals(IND, list(d['adj'].columns))['S1_3m_yoy']
    REV = pd.DataFrame({t: rev.loc[pd.Period(t, 'M') - 1] for t in dates if (pd.Period(t, 'M') - 1) in rev.index}).T
    SIG = {'RS_52wH': S['RS_52wH'].where(cnt >= 5), 'DIFF_NH': DIFF.where((cnt >= 5) & (nn >= 5)), 'REV_S1_3m': REV.where(cnt.reindex(REV.index) >= 5)}
    res = {}
    base_g, base_to = run_main()
    res['MAIN_base'] = mk_stats(base_g, base_to, dates, B, U300)
    base_net = base_g - base_to * 0.005
    # (a) 產業策略於同一換股日 + 相關
    Uel = universe_ew(M.R, el)
    res['corr'] = {}
    for sn, Sg in SIG.items():
        W = {}
        for t in M.R.index:
            if t not in Sg.index: continue
            s = Sg.loc[t].dropna()
            if len(s) < 10: continue
            W[t] = build_weights(t, list(s.sort_values(ascending=False).index[:3]), el.loc[t], v.loc[t], 'liq10')
        g, to = run_weights(W, M.R)
        net = (g - to * 0.005).dropna()
        common = net.index.intersection(base_net.index)
        b = B.reindex(common)
        blend = 0.5 * base_net.reindex(common) + 0.5 * net.reindex(common)
        res['corr'][sn] = {'industry_strategy': mk_stats(g, to, dates, B, Uel),
                           'corr_returns': round(float(np.corrcoef(net[common], base_net[common])[0, 1]), 3),
                           'corr_excess_vs_0050': round(float(np.corrcoef(net[common] - b, base_net[common] - b)[0, 1]), 3),
                           'blend_50_50': ann(blend, dates), 'main_alone_same_periods': ann(base_net.reindex(common), dates)}
    # (b) 主策略產業遮罩
    res['masked'] = {}
    for sn, Sg in SIG.items():
        pr = Sg.rank(axis=1, pct=True, ascending=True)   # 1 = 最強
        for name, rule in [('top50pct', pr > 0.5), ('top33pct', pr > 2 / 3), ('drop_bottom25pct', pr > 0.25)]:
            ind_ok = rule.where(Sg.notna(), True)  # 無訊號的產業不排除
            mask = pd.DataFrame({c: ind_ok[IND[c]] if (c in IND.index and IND[c] in ind_ok.columns) else pd.Series(True, index=ind_ok.index)
                                 for c in M.t300.columns}).reindex(pd.DatetimeIndex(dates)).fillna(True).astype(bool)
            g, to = run_main(mask)
            st = mk_stats(g, to, dates, B, U300)
            net = g - to * 0.005; dif = (net - base_net).dropna()
            st['diff_vs_base_pct_m'] = {'sel': round(float(dif[:SEL[1]].mean() * 100), 3), 't_sel': tstat(dif[:SEL[1]]),
                                        'val': round(float(dif[VAL[0]:].mean() * 100), 3), 't_val': tstat(dif[VAL[0]:])}
            res['masked'][f'{sn}|{name}'] = st
    cand = {k: v['diff_vs_base_pct_m']['sel'] for k, v in res['masked'].items()}
    best = max(cand, key=cand.get)
    res['masked_selected_in_sample'] = {'config': best, **res['masked'][best]}
    save('results_t5.json', res)
    print('MAIN', res['MAIN_base'])
    for k, v in res['corr'].items():
        print(k, 'corr', v['corr_returns'], 'corr_ex', v['corr_excess_vs_0050'], 'blend', v['blend_50_50'], 'main', v['main_alone_same_periods'])
        print('   ind strat', v['industry_strategy']['full'])
    for k, v in res['masked'].items():
        print(k, '\n  SEL', v['sel'], '\n  VAL', v['val'], '\n  diff', v['diff_vs_base_pct_m'])
    print('SELECTED', best)


if __name__ == '__main__':
    main()
