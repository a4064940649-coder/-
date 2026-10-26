#!/usr/bin/env python3
"""重測 r1_1：ml 組「把模型分數和 52 週高點合成，也沒有比 52 週高點單獨好」。

事前寫下（全部報告，不挑）：
  流程：忠實重現 ml_price.py（17 個價格特徵、前 300 名流動性池、剔除鎖漲停、逐年外推、ridge / lgbm / lgbm_shallow / lgbm_top20、
        ens = ridge 與 lgbm 百分位平均、前 30 檔等權、成本 = 平均周轉 × 0.5%）。直接 import ml 組的 ml_common.py（不修改）。
  面板：orig = 原始面板（含 2018-01-13 星期六異常列，只有少數股有值 → V.rolling(60) 與 adj.rolling(252) 被打斷，2018 全年池為空）；
        fix  = load_panel 之後 drop 2018-01-13（adj、close、value、high、ret、open 全部）。orig 只用來對照質疑方引用的數字。
  合成（全部報告）：
        SCOREBLEND_{m}+hi52 = (模型分數池內百分位 + 52 週高點池內百分位) / 2，m ∈ {ridge, lgbm, lgbm_shallow, ens}
        PORTBLEND_50{m}_50hi52 = 模型組合與 52 週高點組合的每期淨報酬各半（周轉取平均）。
  期間：挑選期 2010-01～2019-04（換股日 2009-12-01～2019-04-30）、驗證期 2019-05～2026-10、全期 2010～2026。
  主要統計：配對差距 d_t = net_合成 − net_hi52（每期），報 平均（%/月）、t 值、勝率、期數；另報年化、最大回撤、對 0050 / 池內等權 t。
  穩健性（次要）：t+1 開盤進出（AO = open × adj / close_raw；沒開盤不可買），分數不變。
輸出：agents/ml/recheck_r1_1.json
"""
import json, time, sys
from pathlib import Path
ML = Path('/home/user/-/research/tw_stock_selection/agents/ml')
sys.path.insert(0, str(ML))
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
from ml_common import (load_panel, price_features, universe, stack, fit_predict, build_port, report,
                       PRICE_FEATS, period_returns, bench_returns, cut, tstat, P)
from btlib import month_ends, yearly

T0 = time.time()
BAD = pd.Timestamp('2018-01-13')
PER = {'SEL_2010-01_2019-04': ('2009-12-01', '2019-04-30'),
       'VAL_2019-05_2026-10': ('2019-05-01', None),
       'FULL_2010_2026': ('2009-12-01', None)}
MODELS = ['ridge', 'lgbm', 'lgbm_shallow', 'lgbm_top20']
BLEND_M = ['ridge', 'lgbm', 'lgbm_shallow', 'ens']
e50 = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj']


def open_returns(adj, C, O, dates):
    td = adj.index
    AO = O * adj / C
    A = adj.ffill(limit=30)
    D = pd.DatetimeIndex(dates)
    pos = td.get_indexer(D); assert (pos >= 0).all()
    d1 = [td[p + 1] if p + 1 < len(td) else pd.NaT for p in pos]
    ent = pd.DataFrame(np.nan, index=D, columns=adj.columns); ext = ent.copy()
    ok = [i for i, d in enumerate(d1) if d is not pd.NaT]
    dd = pd.DatetimeIndex([d1[i] for i in ok])
    ent.iloc[ok] = AO.reindex(dd).values
    ext.iloc[ok] = AO.reindex(dd).fillna(A.reindex(dd)).values
    R = (ext.shift(-1) / ent - 1).where(adj.reindex(D).notna())
    s = e50.sort_index()
    b = pd.Series([s[:d].iloc[-1] if d is not pd.NaT else np.nan for d in d1], index=D)
    return R.iloc[:-1], (b.shift(-1) / b - 1).iloc[:-1]


def run(variant, do_open):
    adj, C, V, H, R1 = load_panel()
    O = pd.read_parquet(P / 'open.parquet')
    n_bad = int(adj.loc[BAD].notna().sum()) if BAD in adj.index else 0
    if variant == 'fix':
        adj, C, V, H, R1, O = [x.drop(index=BAD, errors='ignore') for x in (adj, C, V, H, R1, O)]
    me = [d for d in month_ends(adj.index) if d >= pd.Timestamp('2005-01-01')]
    R = period_returns(adj, me); B = bench_returns(e50, me)
    F = price_features(adj, C, V, H, R1, me)
    el0, locked = universe(adj, C, V, H, R1, me)
    elig = (el0 & ~locked).reindex(R.index)
    F = {k: v.reindex(R.index) for k, v in F.items()}
    Xall = stack(F, elig, PRICE_FEATS)
    yall = (R.where(elig & R.notna()).rank(axis=1, pct=True) - 0.5).stack(future_stack=True).reindex(Xall.index)
    ti = list(R.index); nxt = dict(zip(me[:-1], me[1:]))
    pred = {m: pd.DataFrame(np.nan, index=R.index, columns=R.columns) for m in MODELS}
    dl = Xall.index.get_level_values(0)
    for Y in range(2010, 2027):
        test = [t for t in ti if nxt[t].year == Y]
        if not test: continue
        trm = (dl < test[0]) & yall.notna().values; tem = dl.isin(test)
        if pd.Series(dl[trm]).nunique() < 59: continue
        for m in MODELS:
            p, _ = fit_predict(m, Xall[trm], yall[trm], Xall[tem], PRICE_FEATS, seed=Y)
            s = pd.Series(p, index=Xall[tem].index).unstack()
            pred[m].loc[s.index, s.columns] = s.values
        print(f'[{variant}] {Y} train_months={pd.Series(dl[trm]).nunique()} test_months={pd.Series(dl[tem]).nunique()} {time.time()-T0:.0f}s', flush=True)
    rk = lambda S: S.where(elig).rank(axis=1, pct=True)
    pred['ens'] = (rk(pred['ridge']) + rk(pred['lgbm'])) / 2
    mask = pred['ridge'].notna()
    scores = {m: pred[m] for m in BLEND_M}
    scores['BASE_hi52'] = F['hi52'].where(mask)
    for m in BLEND_M:
        scores[f'SCOREBLEND_{m}+hi52'] = (rk(pred[m]) + rk(F['hi52'])).where(mask) / 2
    out = {'rows_on_2018-01-13_in_raw_panel': n_bad,
           'eligible_2018': {str(t.date()): int(v) for t, v in elig.sum(axis=1).items() if t.year == 2018}, 'eval': {}}
    evs = {'close': (R, B)}
    if do_open: evs['open_t+1'] = open_returns(adj, C, O, me)
    for ev, (RR, BB) in evs.items():
        ports = {k: build_port(s, RR, elig) for k, s in scores.items()}
        base = ports['BASE_hi52']
        for m in BLEND_M:
            p = ports[m]
            ports[f'PORTBLEND_50{m}_50hi52'] = {'gross': (p['gross'] + base['gross']) / 2, 'net': (p['net'] + base['net']) / 2,
                                                'net1': (p['net1'] + base['net1']) / 2, 'uni': p['uni'],
                                                'turnover': (p['turnover'] + base['turnover']) / 2}
        E = {}
        for pn, (a, b) in PER.items():
            E[pn] = {}
            hb = cut(base['net'], a, b)
            for k, p in ports.items():
                o = report(p, BB, me, a, b)
                net = cut(p['net'], a, b)
                d = (net - hb.reindex(net.index)).dropna()
                d1 = (cut(p['net1'], a, b) - cut(base['net1'], a, b).reindex(net.index)).dropna()
                r = {'periods': o['net'].get('periods'), 'CAGR': o['net'].get('CAGR'), 'maxDD': o['net'].get('maxDD'),
                     't_vs_0050': o['t_vs_0050'], 't_vs_EW': o['t_vs_EW'], 'turnover': o['turnover'],
                     '0050_CAGR': o['0050'].get('CAGR'), 'EW_CAGR': o['pool_EW'].get('CAGR')}
                if k != 'BASE_hi52':
                    r.update({'diff_vs_hi52_pct_m': round(float(d.mean() * 100), 3), 't_diff_vs_hi52': tstat(d),
                              'hit_vs_hi52': round(float((d > 0).mean()), 3), 'n_diff': int(len(d)),
                              't_diff_vs_hi52_cost1pct': tstat(d1)})
                    if pn == 'FULL_2010_2026':
                        yb, yh = yearly(net, me), yearly(hb, me)
                        r['yearly_net_minus_hi52_pp'] = {y: round(yb[y] - yh.get(y, float('nan')), 1) for y in yb}
                E[pn][k] = r
        out['eval'][ev] = E
    return out


res = {'design': __doc__}
res['orig'] = run('orig', do_open=False); print(f'orig done {time.time()-T0:.0f}s', flush=True)
res['fix'] = run('fix', do_open=True); print(f'fix done {time.time()-T0:.0f}s', flush=True)

SHOW = ['BASE_hi52'] + [f'SCOREBLEND_{m}+hi52' for m in BLEND_M] + [f'PORTBLEND_50{m}_50hi52' for m in BLEND_M] + BLEND_M
for v, evl in [('orig', ['close']), ('fix', ['close', 'open_t+1'])]:
    for ev in evl:
        for pn in PER:
            print(f'\n--- {v} / {ev} / {pn}')
            for k in SHOW:
                o = res[v]['eval'][ev][pn][k]
                print(f"{k:30s} n={o['periods']:3d} CAGR {o['CAGR']:7.2%} MDD {o['maxDD']:7.2%} t0050 {o['t_vs_0050']:5.2f} tEW {o['t_vs_EW']:5.2f}"
                      f" | d-hi52 {o.get('diff_vs_hi52_pct_m')}%/m t={o.get('t_diff_vs_hi52')} hit={o.get('hit_vs_hi52')} t(1%)={o.get('t_diff_vs_hi52_cost1pct')}")
print('\neligible 2018 orig', res['orig']['eligible_2018'])
print('eligible 2018 fix ', res['fix']['eligible_2018'])
for k in ['SCOREBLEND_lgbm+hi52', 'PORTBLEND_50lgbm_50hi52']:
    print(k, 'yearly diff (fix close)', res['fix']['eval']['close']['FULL_2010_2026'][k]['yearly_net_minus_hi52_pp'])
(ML / 'recheck_r1_1.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(f'done {time.time()-T0:.0f}s')
