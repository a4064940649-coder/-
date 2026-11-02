#!/usr/bin/env python3
"""步驟 7（樣本外）：配置格（K5T 單一模型）。每格（4 級距 × FIX／GROW × 一般／壓力）：
先驗＝CV_FINAL 的 V*、s、k 套在該格的 j 中位數序列（全期族群窗）；
模擬：月頻 (0050 官方窗報酬, 該格超額殘差) 成對平穩 bootstrap（平均區塊 6），20 年 240 個月、20,000 條路徑，
每條路徑從先驗抽一個 α（共同亂數 seed 20261106）；主策略月報酬＝0050＋α＋殘差；每月再平衡並扣 sc_lib 成本（T.simulate）。
報告：P(α>0)、先驗均值、x*＝argmax E[log W]／年、平坦區間＝與最大值差 ≤ 0.25%／年的 x、10 年／20 年贏 0050 機率（x＝x*、100%）。
輸入：V7X（o_exec_engine_rewrite_v7x）若有正式淨值則用，否則 V7 navs_exec7 K5T（標註）。
用法：python3 -I -B s7_alloc.py"""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import fslib as F, prior as PR

T = F.T
B, HH, L, SEED = 20_000, 240, 6, 20261106
XS = np.round(np.arange(0, 1.01, 0.1), 2)


def v7x_navs():
    cand = list((F.AG / 'o_exec_engine_rewrite_v7x').glob('**/navs*v7x*.parquet'))
    return (pd.read_parquet(cand[0]), str(cand[0])) if cand else (None, 'V7 navs_exec7（V7X 未交件）')


def main():
    PRE = F.check_prereg('PREREG')
    res = F.load_res(); cv = res['CV_FINAL']
    X, lab, med, anc = F.family()
    navs, src = v7x_navs()
    O = F.official_series(); O.index = pd.PeriodIndex(O.index, freq='M')
    out = {'source': src, 'cv': {k: cv[k] for k in ('selected', 's', 'k')}, 'cells': {}}
    for cost in ('normal', 'stress'):
        for tier in F.TIERS:
            for mode in F.MODES:
                M, _, _ = F.cell_jmatrix(tier, mode, cost=cost, navs=navs)
                s = F.jmed(M); s.index = pd.PeriodIndex(s.index, freq='M')
                b = PR.build(np.ones(len(anc), bool), cv['selected'], cv['s'], cv['k'], s)
                common = s.index.intersection(O.index)
                r50 = O.r50.reindex(common).to_numpy(); e = s.reindex(common).to_numpy() / 100
                eps = e - e.mean()
                ix = T.sb_index(len(common), HH, L, B, SEED)
                r50p, ep = r50[ix], eps[ix]
                rng = np.random.default_rng(SEED)
                al = T.sample_dens(b['prior'], b['th'], B, rng, b['th'][1] - b['th'][0]) / 100
                elw, w10, w20 = {}, {}, {}
                for x in XS:
                    sim = T.simulate(r50p, ep, al, float(x))
                    elw[float(x)] = float(np.mean(sim['logW']) / 20 * 100)
                    w20[float(x)] = float(np.mean(sim['Wp'] > sim['W50']))
                    sim10 = T.simulate(r50p[:, :120], ep[:, :120], al, float(x))
                    w10[float(x)] = float(np.mean(sim10['Wp'] > sim10['W50']))
                best = max(elw, key=elw.get)
                flat = [x for x in elw if elw[best] - elw[x] <= 0.25]
                out['cells'][f'{tier}|{mode}|{cost}'] = dict(prior_cal=b['cal'], prior_raw=b['raw'], ahat=b['ahat'], n_months=len(common),
                                                             x_star=best, flat=[min(flat), max(flat)], flat_set=flat, ElogW=elw,
                                                             win10=w10, win20=w20, gap_0_vs_best=elw[best] - elw[0.0], gap_100_vs_best=elw[best] - elw[1.0])
                c = out['cells'][f'{tier}|{mode}|{cost}']
                print(tier, mode, cost, 'P>0', round(b['cal']['p_pos'], 3), 'mean', round(b['cal']['mean'], 3), 'raw', round(b['raw']['mean'], 3),
                      'x*', best, 'flat', c['flat'], 'win10@100', round(w10[1.0], 3), flush=True)
    out['utc'] = F.now_utc()
    F.save_key('ALLOC', out, need=['CV_FINAL'])


if __name__ == '__main__':
    main()
