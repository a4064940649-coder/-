#!/usr/bin/env python3
"""第 5、6 步：樣本外（公告季 2019Q2～2026Q2）與全期，只跑一次；規格照 frozen_spec.txt，不得再改。
先斷言 PREREG 存在且 frozen_spec 雜湊相符。輸出 results.json 的 OOS 區塊、data/navs_*.parquet。"""
import json, sys, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from pcore import DATA, HERE, AG, load_panel, e50_on, tstat, sha
from pengine import Market, run, perf, monthly

R = json.load(open(HERE / 'results.json'))
assert 'PREREG' in R and R['PREREG'].get('sha256'), 'PREREG 不存在'
assert R['PREREG']['hashes']['frozen_spec.txt'] == sha(HERE / 'frozen_spec.txt'), 'frozen_spec 被改過'
assert R['PREREG']['hashes']['data/events.parquet'] == sha(DATA / 'events.parquet'), '事件表被改過'
from s3_iterate import VARIANTS, cand, M, E50, IDX   # 共用同一份面板（s3 只在 import 時載入資料，不跑迭代）
EA = pd.read_parquet(DATA / 'events.parquet'); EA = EA[EA['SUEP'].notna()]
K0 = 2019 * 4 + 1   # 2019Q2
WIN_OOS = ('2019-05-01', '2026-10-30'); WIN_FULL = ('2016-04-01', '2026-10-30')
SEG = {'VAL': ('2019-05-01', '2023-12-31'), 'HOLD': ('2024-01-01', '2026-10-30'), 'OOS': WIN_OOS}
SEG_FULL = {'FULL': WIN_FULL, 'SEL_2016_2018': ('2016-04-01', '2018-12-31'), 'Y2019_2026': ('2019-01-01', '2026-10-30')}
specs = {'MAIN': VARIANTS['v3'], 'R1': VARIANTS['v1'], 'R2': VARIANTS['v8'],
         'MAIN_S10': dict(VARIANTS['v3'], S=10), 'MAIN_S30': dict(VARIANTS['v3'], S=30), 'MAIN_H40': dict(VARIANTS['v3'], H=40)}
for n in ['v2', 'v4', 'v5', 'v6', 'v7']:
    specs['ITER_' + n] = VARIANTS[n]


def runs(v, E, win, K=1e6, mode='GROW', cost='normal'):
    C = cand(E, v)
    navs, parts, fills, costs = {}, [], [], []
    for jd in range(10):
        ev = pd.DataFrame({'col': C['j'], 'e': C[f'e_j{jd}'], 'score': C[v['sig']]}); ev = ev[ev['e'] >= 0]
        nav, st = run(M, ev, S=v['S'], H=v['H'], K=K, mode=mode, cost=cost, start=win[0], end=win[1])
        navs[jd] = nav; parts += st['part']; fills += st['filled_ratio']
        costs.append((st['fee'] + st['tax'] + st['imp']) / ((nav.index[-1] - nav.index[0]).days / 365.25))
    return navs, dict(part_q=[float(np.nanquantile(parts, q)) for q in (0.1, 0.5, 0.9)] if parts else None,
                      fill_med=float(np.median(fills)) if fills else None, cost_per_year_ntd_med=float(np.median(costs)))


def jstats(navs, seg):
    out = {}
    for name, (a, b) in seg.items():
        ps = [perf(n, E50.loc[n.index], a, b) for n in navs.values()]
        ps = [p for p in ps if p]
        if not ps:
            continue
        ex = np.array([p['ex_ann'] for p in ps]); t = np.array([p['t_m'] for p in ps])
        out[name] = dict(ex_med=float(np.median(ex)), ex_p10=float(np.quantile(ex, 0.1)), t_med=float(np.median(t)), t_p10=float(np.quantile(t, 0.1)),
                         cagr_med=float(np.median([p['cagr'] for p in ps])), b50=float(ps[0]['b50']), mdd_med=float(np.median([p['mdd'] for p in ps])),
                         n_m=ps[0]['n_m'])
    return out


def avg_nav(navs):
    r = pd.concat({j: n.pct_change() for j, n in navs.items()}, axis=1).mean(axis=1).fillna(0)
    return (1 + r).cumprod()


out = {'runs': {}}
NAVSTORE = {}
E_oos = EA[EA['k'] >= K0]
E_full = EA
for name, v in specs.items():
    for wname, E, win, seg in [('OOS', E_oos, WIN_OOS, SEG), ('FULL', E_full, WIN_FULL, SEG_FULL)]:
        cfgs = [(1e6, 'GROW', 'normal'), (1e6, 'GROW', 'stress')]
        if name == 'MAIN':
            cfgs += [(3e6, 'GROW', 'normal'), (1e7, 'GROW', 'normal'), (1e6, 'FIX', 'normal'), (3e6, 'FIX', 'normal'), (1e7, 'FIX', 'normal'),
                     (3e6, 'GROW', 'stress'), (1e7, 'GROW', 'stress')]
        for K, mode, cost in cfgs:
            navs, info = runs(v, E, win, K, mode, cost)
            key = f'{name}|{wname}|{int(K / 1e4)}w|{mode}|{cost}'
            out['runs'][key] = dict(stats=jstats(navs, seg), **info)
            NAVSTORE[key] = avg_nav(navs)
            if name == 'MAIN' and K == 1e6 and mode == 'GROW':
                for jd, n in navs.items():
                    NAVSTORE[f'{key}|j{jd}'] = n / n.iloc[0]
            print(key, json.dumps(out['runs'][key]['stats'].get('OOS', out['runs'][key]['stats'].get('FULL')), ensure_ascii=False))
nv = pd.DataFrame(NAVSTORE)
nv.to_parquet(DATA / 'navs_pead.parquet')

# ---------- 逐年、拿掉最好 3 年、回撤 ----------
def yearly_ex(n):
    y = n.resample('YE').last(); yb = E50.loc[n.index].resample('YE').last()
    first = n.iloc[0]; fb = E50.loc[n.index].iloc[0]
    ry = y.pct_change(); ry.iloc[0] = y.iloc[0] / first - 1
    rb = yb.pct_change(); rb.iloc[0] = yb.iloc[0] / fb - 1
    return pd.DataFrame({'pead': ry, '0050': rb, 'ex': ry - rb})


main_full = nv['MAIN|FULL|100w|GROW|normal'].dropna()
ye = yearly_ex(main_full)
out['yearly_MAIN_full_100w'] = {int(k.year): {c: float(v) for c, v in r.items()} for k, r in ye.iterrows()}
mex = (monthly(main_full) - monthly(E50.loc[main_full.index]).reindex(monthly(main_full).index)).dropna()
best3 = ye['ex'].sort_values().index[-3:].year
mx = mex[~mex.index.year.isin(best3)]
out['drop_best3_years'] = {'years': [int(x) for x in best3], 't': float(tstat(mx)), 'mean_m': float(mx.mean())}
out['MAIN_full_monthly_t_avgnav'] = float(tstat(mex))

# ---------- 組合：主策略 50/50、80/20 ----------
mn = pd.read_parquet(AG / 'main_v2' / 'navs_gapfix.parquet')['EXFIN|n10|normal'].dropna()
mn.index = pd.DatetimeIndex(mn.index)
def combo(pead_nav, w, win):
    a, b = win
    rp = monthly(pead_nav); rm = monthly(mn); r5 = monthly(E50)
    df = pd.concat({'p': rp, 'm': rm, 'b': r5}, axis=1).dropna()
    df = df[(df.index >= pd.Timestamp(a)) & (df.index <= pd.Timestamp(b) + pd.offsets.MonthEnd(0))]
    # 月再平衡成本：0.1% × 兩腿漂移後換手
    drift = (w * (1 + df['p'])) / (w * (1 + df['p']) + (1 - w) * (1 + df['m'])) - w
    c = w * df['p'] + (1 - w) * df['m'] - 0.001 * 2 * drift.abs()
    res = {'paired_vs_main_t': float(tstat(c - df['m'])), 'paired_vs_main_mean': float((c - df['m']).mean()),
           'combo_vs_0050_t': float(tstat(c - df['b'])), 'main_vs_0050_t': float(tstat(df['m'] - df['b'])),
           'pead_vs_0050_t': float(tstat(df['p'] - df['b'])),
           'combo_ann': float((1 + c).prod() ** (12 / len(c)) - 1), 'main_ann': float((1 + df['m']).prod() ** (12 / len(c)) - 1),
           'pead_ann': float((1 + df['p']).prod() ** (12 / len(c)) - 1), 'b50_ann': float((1 + df['b']).prod() ** (12 / len(c)) - 1),
           'corr_excess_pead_main': float(np.corrcoef(df['p'] - df['b'], df['m'] - df['b'])[0, 1]),
           'combo_mdd_m': float(((1 + c).cumprod() / (1 + c).cumprod().cummax() - 1).min()),
           'main_mdd_m': float(((1 + df['m']).cumprod() / (1 + df['m']).cumprod().cummax() - 1).min()), 'n_m': int(len(c))}
    return res
out['combo'] = {}
for w, nm in [(0.5, 'C50'), (0.2, 'C80')]:
    for seg, win in [('OOS', WIN_OOS), ('VAL', SEG['VAL']), ('HOLD', SEG['HOLD']), ('FULL', WIN_FULL)]:
        src = nv['MAIN|OOS|100w|GROW|normal'] if seg in ('OOS', 'VAL', 'HOLD') else nv['MAIN|FULL|100w|GROW|normal']
        # j 別組合的中位數
        per = []
        for jd in range(10):
            per.append(combo(nv[f'MAIN|OOS|100w|GROW|normal|j{jd}'].dropna() if seg != 'FULL' else src.dropna(), w, win))
            if seg == 'FULL':
                break
        out['combo'][f'{nm}|{seg}'] = {k: float(np.median([p[k] for p in per])) for k in per[0]}
        print(nm, seg, json.dumps(out['combo'][f'{nm}|{seg}']))
R['OOS'] = out
json.dump(R, open(HERE / 'results.json', 'w'), ensure_ascii=False, indent=1, default=float)
