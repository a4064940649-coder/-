#!/usr/bin/env python3
"""重測 r1_0：ml 組「2010～2019 ML 沒有贏過 52 週高點」結論。

不修改 ml 組檔案：ml_common.py / ml_price.py 已複製到 scratchpad/ml_recheck/，本程式從該副本 import，
並忠實重現 ml_price.py 的流程（特徵、池、逐年外推、四個模型、ens、對照組、分數合成、組合合成）。

變體：
  orig  ：原始面板（含 2018-01-13 星期六異常列，只有 46 檔有值）→ 應重現 101 期與 FINDINGS 表格。
  fix   ：load_panel 之後對 adj、C、V、H、R1（與 open）drop 2018-01-13 → 2010-01～2019-04 應為 113 期。
評估：
  close ：原法，換股日 t 收盤進出（period_returns(adj)）。
  open  ：t+1 開盤進出。調整後開盤 AO = open × adj / close_raw；進場 = t 之後第一個交易日的 AO（沒開盤就不可買），
          出場 = 下一換股日之後第一個交易日的 AO（沒開盤則用最後收盤 adj）。模型分數不變（標籤仍為收盤到收盤，
          訓練樣本最後一期在預測日收盤已知，不前視）。0050 用 t+1 日收盤到下期 t+1 日收盤（只有含息收盤序列，近似）。
另報：模型對 52 週高點的配對超額（每期 net_model − net_hi52）t 值、超額相關係數。
"""
import json, time, sys, math
from pathlib import Path
SCR = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/ml_recheck')
sys.path.insert(0, str(SCR))
import numpy as np, pandas as pd
from ml_common import (quintiles, ann, load_panel, price_features, universe, stack, fit_predict, build_port, ic_series, report,
                       PRICE_FEATS, period_returns, bench_returns, cut, tstat, P)
from btlib import month_ends

OUT = Path(__file__).resolve().parent
T0 = time.time()
BAD = pd.Timestamp('2018-01-13')
PER = {'2010-01_2019-04': ('2009-12-01', '2019-04-30'), '2010_2019': ('2009-12-01', '2019-11-30'),
       '2019-05_2026-10': ('2019-05-01', None), 'FULL_2010_2026': ('2009-12-01', None)}
MODELS = ['ridge', 'lgbm', 'lgbm_shallow', 'lgbm_top20']
e50 = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj']


def open_returns(adj, C, O, dates):
    """index = 換股日 t（最後一個除外）：t 後第一交易日開盤 → 下一換股日後第一交易日開盤。"""
    td = adj.index
    AO = O * adj / C
    A = adj.ffill(limit=30)
    pos = td.get_indexer(pd.DatetimeIndex(dates))
    assert (pos >= 0).all()
    d1 = [td[p + 1] if p + 1 < len(td) else pd.NaT for p in pos]
    ent = pd.DataFrame(np.nan, index=pd.DatetimeIndex(dates), columns=adj.columns)
    ext = ent.copy()
    ok = [i for i, d in enumerate(d1) if d is not pd.NaT]
    dd = pd.DatetimeIndex([d1[i] for i in ok])
    ent.iloc[ok] = AO.reindex(dd).values
    ext.iloc[ok] = AO.reindex(dd).fillna(A.reindex(dd)).values
    R = ext.shift(-1) / ent - 1
    R = R.where(adj.reindex(pd.DatetimeIndex(dates)).notna())
    s = e50.sort_index()
    b = pd.Series([s[:d].iloc[-1] if d is not pd.NaT else np.nan for d in d1], index=pd.DatetimeIndex(dates))
    return R.iloc[:-1], (b.shift(-1) / b - 1).iloc[:-1]


def run(variant):
    adj, C, V, H, R1 = load_panel()
    O = pd.read_parquet(P / 'open.parquet')
    if variant == 'fix':
        adj, C, V, H, R1, O = [x.drop(index=BAD) for x in (adj, C, V, H, R1, O)]
    me = month_ends(adj.index)
    me = [d for d in me if d >= pd.Timestamp('2005-01-01')]
    R = period_returns(adj, me)
    B = bench_returns(e50, me)
    F = price_features(adj, C, V, H, R1, me)
    el0, locked = universe(adj, C, V, H, R1, me)
    elig = (el0 & ~locked).reindex(R.index)
    F = {k: v.reindex(R.index) for k, v in F.items()}
    Xall = stack(F, elig, PRICE_FEATS)
    lab = (R.where(elig & R.notna()).rank(axis=1, pct=True) - 0.5).stack(future_stack=True)
    yall = lab.reindex(Xall.index)
    ti = list(R.index); nxt = dict(zip(me[:-1], me[1:]))
    pred = {m: pd.DataFrame(np.nan, index=R.index, columns=R.columns) for m in MODELS}
    dl = Xall.index.get_level_values(0)
    train_log = {}
    for Y in range(2010, 2027):
        test = [t for t in ti if nxt[t].year == Y]
        if not test: continue
        t0 = test[0]
        trm = (dl < t0) & yall.notna().values
        tem = dl.isin(test)
        if pd.Series(dl[trm]).nunique() < 59: continue
        Xtr, ytr, Xte = Xall[trm], yall[trm], Xall[tem]
        for m in MODELS:
            p, imp = fit_predict(m, Xtr, ytr, Xte, PRICE_FEATS, seed=Y)
            s = pd.Series(p, index=Xte.index).unstack()
            pred[m].loc[s.index, s.columns] = s.values
        train_log[Y] = {'train_months': int(pd.Series(dl[trm]).nunique()), 'train_rows': int(trm.sum()),
                        'test_months_with_rows': int(pd.Series(dl[tem]).nunique())}
        print(f'[{variant}] {Y}: {train_log[Y]}  {time.time()-T0:.0f}s', flush=True)
    rk = lambda S: S.where(elig).rank(axis=1, pct=True)
    pred['ens'] = (rk(pred['ridge']) + rk(pred['lgbm'])) / 2
    mask_pred = pred['ridge'].notna()
    scores = dict(pred)
    scores['BASE_hi52'] = F['hi52'].where(mask_pred)
    scores['BASE_mom12_1'] = F['mom12_1'].where(mask_pred)
    for m in ['ridge', 'lgbm', 'lgbm_shallow', 'ens']:
        scores[f'SCOREBLEND_{m}+hi52'] = (rk(pred[m]) + rk(F['hi52'])).where(mask_pred) / 2
    elig_n = elig.sum(axis=1)
    pool_2018 = {str(t.date()): int(elig_n.loc[t]) for t in elig_n.index if t.year == 2018}
    Ro, Bo = open_returns(adj, C, O, me)
    out = {'pool_size_2018': pool_2018, 'train_log': train_log, 'eval': {}}
    keep = {}
    for ev, (RR, BB) in {'close': (R, B), 'open': (Ro, Bo)}.items():
        ports = {k: build_port(s, RR, elig) for k, s in scores.items()}
        base = ports['BASE_hi52']
        for m in ['ridge', 'lgbm', 'lgbm_shallow', 'ens']:
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
                net = cut(p['net'], a, b); u = p['uni'].reindex(net.index); h = hb.reindex(net.index)
                d = (net - h).dropna()
                o2 = {'CAGR': o['net'].get('CAGR'), 'maxDD': o['net'].get('maxDD'), 'periods': o['net'].get('periods'),
                      'net_1pct_CAGR': o['net_1pct'].get('CAGR'), 'turnover': o['turnover'],
                      't_vs_0050': o['t_vs_0050'], 't_vs_EW': o['t_vs_EW'],
                      'ex0050_pct_m': o['excess_vs_0050_pct_m'], 'exEW_pct_m': o['excess_vs_EW_pct_m'],
                      'EW_CAGR': o['pool_EW'].get('CAGR'), '0050_CAGR': o['0050'].get('CAGR'),
                      'diff_vs_hi52_pct_m': round(float(d.mean() * 100), 3) if k != 'BASE_hi52' else None,
                      't_diff_vs_hi52': tstat(d) if k != 'BASE_hi52' else None,
                      'corr_exEW_vs_hi52_exEW': round(float((net - u).corr(h - u)), 3) if k != 'BASE_hi52' else None,
                      'yearly_net_pct': o['yearly_net_pct']}
                E[pn][k] = o2
        out['eval'][ev] = E
        keep[ev] = ports
    return out, keep


res = {'design': __doc__}
for v in ['orig', 'fix']:
    res[v], _ = run(v)
    print(f'=== {v} done {time.time()-T0:.0f}s', flush=True)

SHOW = ['ridge', 'lgbm', 'lgbm_shallow', 'lgbm_top20', 'ens', 'BASE_hi52', 'BASE_mom12_1',
        'SCOREBLEND_lgbm+hi52', 'SCOREBLEND_lgbm_shallow+hi52', 'PORTBLEND_50lgbm_50hi52']
for v in ['orig', 'fix']:
    for ev in ['close', 'open']:
        for pn in ['2010-01_2019-04', '2019-05_2026-10']:
            print(f'\n--- {v} / {ev} / {pn}')
            for k in SHOW:
                o = res[v]['eval'][ev][pn][k]
                print(f"{k:30s} n={o['periods']:3d} CAGR {o['CAGR']:7.2%} MDD {o['maxDD']:7.2%} t0050 {o['t_vs_0050']:5.2f} tEW {o['t_vs_EW']:5.2f}"
                      f" | diff-hi52 {o['diff_vs_hi52_pct_m']} t={o['t_diff_vs_hi52']} corr={o['corr_exEW_vs_hi52_exEW']}")
print('\npool 2018 orig', res['orig']['pool_size_2018'])
print('pool 2018 fix ', res['fix']['pool_size_2018'])
(OUT / 'recheck_r1_0.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(f'done {time.time()-T0:.0f}s')
