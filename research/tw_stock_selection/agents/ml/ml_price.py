#!/usr/bin/env python3
"""A. 純價格機器學習選股（上市股，月底換股，2005～2026）。

事前固定（不調參）：
  特徵（17 個，換股日收盤以前）：動能 1/3/6/12 月、12-1 月、收盤/252 日最高、收盤/252 日最低、60 日波動、21 日最大單日漲幅、
        收盤相對 20/60/200 日均線、log 60 日均成交值、5/60 與 20/60 日成交值比、20/120 日成交股數比、Amihud 60 日非流動性。
        每期在可投資池內轉成橫斷面百分位 − 0.5。
  標籤：下月含息報酬在池內的橫斷面百分位 − 0.5。
  池：60 日均成交值 ≥ 1,000 萬、股價 ≥ 10、流動性前 300 名、有 252 日高點、剔除鎖漲停收盤。
  模型：ridge（alpha=100）、lgbm（400 樹、lr .03、31 葉、每葉 ≥300）、lgbm_shallow（200 樹、lr .05、7 葉、每葉 ≥500）、
        ens = ridge 與 lgbm 預測的橫斷面百分位平均。
  逐年外推：預測「持有期結束於 Y 年」的 12 期（換股日 Y−1 年 12 月底～Y 年 11 月底），
        訓練樣本 = 換股日早於第一個預測換股日的所有期（最後一期標籤正好結束於第一個預測換股日，無重疊），至少 5 年 → 2010 起。
  組合：前 30 檔等權，成本 = 平均周轉 × 0.5%（壓力 1%）。
  對照：同框架的 52 週高點單因子（hi52）、12-1 動能單因子、0050、池內等權；並測 50/50 組合合成與分數合成。
輸出：agents/ml/results_price.json
"""
import json, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from ml_common import (quintiles, ann, load_panel, price_features, universe, stack, fit_predict, build_port, ic_series, report, line,
                       PRICE_FEATS, period_returns, bench_returns, cut)
from btlib import month_ends

OUT = Path(__file__).resolve().parent
T0 = time.time()
adj, C, V, H, R1 = load_panel()
me = month_ends(adj.index)
me = [d for d in me if d >= pd.Timestamp('2005-01-01')]
R = period_returns(adj, me)
e50 = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj']
B = bench_returns(e50, me)
F = price_features(adj, C, V, H, R1, me)
el0, locked = universe(adj, C, V, H, R1, me)
elig = (el0 & ~locked).reindex(R.index)
F = {k: v.reindex(R.index) for k, v in F.items()}
Xall = stack(F, elig, PRICE_FEATS)
lab = (R.where(elig & R.notna()).rank(axis=1, pct=True) - 0.5).stack(future_stack=True)
yall = lab.reindex(Xall.index)
print(f'features built {time.time()-T0:.0f}s rows={len(Xall)}')

ti = list(R.index); nxt = dict(zip(me[:-1], me[1:]))
MODELS = ['ridge', 'lgbm', 'lgbm_shallow', 'lgbm_top20']  # lgbm_top20 = 事後診斷
pred = {m: pd.DataFrame(np.nan, index=R.index, columns=R.columns) for m in MODELS}
imps = {m: [] for m in MODELS}
dl = Xall.index.get_level_values(0)
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
        imps[m].append(imp.rename(Y))
    print(f'{Y}: train {pd.Series(dl[trm]).nunique()} months / {trm.sum()} rows, test {len(test)}  {time.time()-T0:.0f}s')

rk = lambda S: S.where(elig).rank(axis=1, pct=True)
pred['ens'] = (rk(pred['ridge']) + rk(pred['lgbm'])) / 2
mask_pred = pred['ridge'].notna()
scores = dict(pred)
scores['BASE_hi52'] = F['hi52'].where(mask_pred)
scores['BASE_mom12_1'] = F['mom12_1'].where(mask_pred)
for m in ['ridge', 'lgbm', 'ens']:
    scores[f'SCOREBLEND_{m}+hi52'] = (rk(pred[m]) + rk(F['hi52'])).where(mask_pred) / 2

ports = {k: build_port(s, R, elig) for k, s in scores.items()}
ics = {k: ic_series(s, R, elig) for k, s in scores.items()}
base = ports['BASE_hi52']
for m in ['ridge', 'lgbm', 'ens']:
    p = ports[m]
    ports[f'PORTBLEND_50{m}_50hi52'] = {'gross': (p['gross'] + base['gross']) / 2, 'net': (p['net'] + base['net']) / 2,
                                        'net1': (p['net1'] + base['net1']) / 2, 'uni': p['uni'], 'turnover': (p['turnover'] + base['turnover']) / 2}

PER = {'2010-01_2019-04': ('2009-12-01', '2019-04-30'), '2010_2019': ('2009-12-01', '2019-11-30'),
       '2019-05_2026-10': ('2019-05-01', None), 'FULL_2010_2026': ('2009-12-01', None)}
res = {'design': __doc__, 'periods': {}, 'correlation_vs_hi52': {}, 'feature_importance': {}, 'data_checks': {}}
for pn, (a, b) in PER.items():
    res['periods'][pn] = {}
    print(f'=== {pn}')
    for k, p in ports.items():
        o = report(p, B, me, a, b, ics.get(k)); res['periods'][pn][k] = o; print(line(k, o))
res['quintiles_gross_CAGR'] = {}
for k in ['ridge', 'lgbm', 'lgbm_shallow', 'lgbm_top20', 'ens', 'BASE_hi52', 'BASE_mom12_1']:
    Q = quintiles(scores[k], R, elig)
    for pn in ['2010-01_2019-04', '2019-05_2026-10']:
        a, b = PER[pn]; Qc = cut(Q, a, b)
        res['quintiles_gross_CAGR'].setdefault(pn, {})[k] = {q: ann(Qc[q], me).get('CAGR') for q in Qc.columns}
        print(pn, k, res['quintiles_gross_CAGR'][pn][k])
for pn in ['2010-01_2019-04', '2019-05_2026-10', 'FULL_2010_2026']:
    a, b = PER[pn]; res['correlation_vs_hi52'][pn] = {}
    for k in ['ridge', 'lgbm', 'lgbm_shallow', 'lgbm_top20', 'ens', 'BASE_mom12_1']:
        x, y = cut(ports[k]['net'], a, b), cut(base['net'], a, b).reindex(cut(ports[k]['net'], a, b).index)
        u = ports[k]['uni'].reindex(x.index)
        res['correlation_vs_hi52'][pn][k] = {'net_ret_corr': round(float(x.corr(y)), 3), 'excess_vs_EW_corr': round(float((x - u).corr(y - u)), 3),
                                             'avg_pick_overlap': round(float(np.mean([len(set(ports[k]['picks'][t]) & set(base['picks'][t])) / 30
                                                                                      for t in x.index if t in base['picks']])), 3)}
for m in MODELS:
    I = pd.concat(imps[m], axis=1)
    res['feature_importance'][m] = {'mean': I.mean(axis=1).sort_values(key=abs, ascending=False).round(4).to_dict(),
                                    'first_year': I.iloc[:, 0].round(4).to_dict(), 'last_year': I.iloc[:, -1].round(4).to_dict()}
# 資料檢查：入選股的極端單期報酬
for k in ['ridge', 'lgbm', 'ens', 'BASE_hi52']:
    rr = pd.Series({(t, c): R.loc[t, c] for t, ps in ports[k]['picks'].items() for c in ps})
    res['data_checks'][k] = {'max_single_stock_period_ret': round(float(rr.max()), 3), 'n_picks_ret_gt_100pct': int((rr > 1).sum()),
                             'n_picks_ret_lt_-50pct': int((rr < -0.5).sum()), 'top5': {f'{t.date()}_{c}': round(float(v), 3) for (t, c), v in rr.nlargest(5).items()}}
res['latest_picks'] = {k: {'date': str(max(ports[k]['picks']).date()), 'codes': ports[k]['picks'][max(ports[k]['picks'])]} for k in ['ridge', 'lgbm', 'ens']}
(OUT / 'results_price.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps({k: res[k] for k in ['correlation_vs_hi52', 'feature_importance', 'data_checks']}, ensure_ascii=False, indent=0)[:5000])
print(f'done {time.time()-T0:.0f}s')
