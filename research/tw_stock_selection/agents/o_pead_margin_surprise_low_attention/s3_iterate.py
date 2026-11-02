#!/usr/bin/env python3
"""第 3 步：迭代（只用挑選期：公告季 ≤ 2019Q1，組合窗 2016-04-01～2019-08-30）。最多 8 個變體，全部記進 iterations.json。
選擇準則（事前寫死）：100 萬 GROW、K5T 一般成本、j＝0..9 的 j 中位數；對 0050 的月 t 最高，且鄰域（槽數 ±10、持有 40/60）不翻號。
不可調：SUE 公式、公告日規則、進場時點、成本模型；不加停損停利。"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from pcore import DATA, HERE, SEL_END_Q, load_panel, e50_on
from pengine import Market, run, perf

KSEL = SEL_END_Q[0] * 4 + SEL_END_Q[1] - 1
WIN = ('2016-04-01', '2019-08-30')
P = load_panel(); IDX = pd.DatetimeIndex(P['IDX']); e50 = e50_on(IDX); M = Market(P, e50)
E50 = pd.Series(e50, index=IDX)
E0 = pd.read_parquet(DATA / 'events.parquet')
E0 = E0[(E0['k'] <= KSEL) & E0['SUEP'].notna()]
assert E0['k'].max() <= KSEL

VARIANTS = {
    'v1': dict(sig='SUEP', att='low', mkt='both', S=20, H=60, note='事前主版本：SUE⊥ 前 1/5、低關注（成交值最低三分位且非 0050）、上市＋上櫃'),
    'v2': dict(sig='SUE', att='low', mkt='both', S=20, H=60, note='未正交化 SUE'),
    'v3': dict(sig='SUEP', att='non0050', mkt='both', S=20, H=60, note='關注度只用「非 0050」（全股池）'),
    'v4': dict(sig='SUEP', att='low', mkt='twse', S=20, H=60, note='只用上市'),
    'v5': dict(sig='SUEP', att='low', mkt='both', S=10, H=60, note='槽數 10（v1 鄰域）'),
    'v6': dict(sig='SUEP', att='low', mkt='both', S=30, H=60, note='槽數 30（v1 鄰域）'),
    'v7': dict(sig='SUEP', att='low', mkt='both', S=20, H=40, note='持有 40 日（v1 鄰域）'),
    'v8': dict(sig='SUE', att='non0050', mkt='both', S=20, H=60, note='未正交化 SUE＋全股池'),
}


def cand(E, v):
    m = E[f'top_{v["sig"]}'].copy()
    if v['att'] == 'low':
        m &= E['lowatt']
    elif v['att'] == 'non0050':
        m &= ~E['in0050']
    elif v['att'] == 'tercile':
        m &= E['att'] == 1
    if v['mkt'] == 'twse':
        m &= E['mkt'] == 'twse'
    return E[m]


def run_variant(E, v, K=1e6, mode='GROW', cost='normal', win=WIN, js=range(10)):
    C = cand(E, v)
    res = {}
    navs = {}
    for jd in js:
        ev = pd.DataFrame({'col': C['j'], 'e': C[f'e_j{jd}'], 'score': C[v['sig']]})
        ev = ev[ev['e'] >= 0]
        nav, st = run(M, ev, S=v['S'], H=v['H'], K=K, mode=mode, cost=cost, start=win[0], end=win[1])
        p = perf(nav, E50.loc[nav.index])
        p.update(buys=st['buys'], part_med=float(np.nanmedian(st['part'])) if st['part'] else None,
                 fill_med=float(np.median(st['filled_ratio'])) if st['filled_ratio'] else None,
                 cost_ann=float((st['fee'] + st['tax'] + st['imp']) / K / ((nav.index[-1] - nav.index[0]).days / 365.25)))
        res[jd] = p; navs[jd] = nav
    return res, navs


def summ(res):
    t = np.array([r['t_m'] for r in res.values()]); x = np.array([r['ex_ann'] for r in res.values()])
    return dict(t_med=float(np.median(t)), t_p10=float(np.quantile(t, 0.1)), ex_med=float(np.median(x)), ex_p10=float(np.quantile(x, 0.1)),
                cagr_med=float(np.median([r['cagr'] for r in res.values()])), mdd_med=float(np.median([r['mdd'] for r in res.values()])),
                cost_ann_med=float(np.median([r['cost_ann'] for r in res.values()])), fill_med=float(np.median([r['fill_med'] or 0 for r in res.values()])))


if __name__ == '__main__':
    it = {'criterion': '100萬 GROW、K5T 一般成本、j 中位數對 0050 月 t 最高，鄰域不翻號', 'window': WIN, 'variants': {}}
    for name, v in VARIANTS.items():
        res, _ = run_variant(E0, v)
        s = summ(res)
        it['variants'][name] = dict(spec=v, summary=s, per_j={int(k): r for k, r in res.items()})
        print(name, v['note'], json.dumps(s, ensure_ascii=False))
    # 選擇
    best = max(it['variants'], key=lambda n: it['variants'][n]['summary']['t_med'])
    it['best_by_t'] = best
    json.dump(it, open(HERE / 'iterations.json', 'w'), ensure_ascii=False, indent=1, default=float)
    print('best', best)
