#!/usr/bin/env python3
"""S59 事後探索（PREREG 沒有登記；看過 results.json 之後才寫，一律標示為事後，不計入任何判定）：
  P1 逐年單位部位超額（月超額 ÷ 個股部位比重）與交易勝率；
  P2 策略自身動能（課程單元 02 因子動能）：過去 12 個月月超額平均 → 下個月／下 12 個月月超額；
  P3 滾動 12 個月已出場交易勝率（依出場月）→ 下 12 個月月超額；
  P4 候選「策略自身燈號」在 2011～2014 與樣本內的狀態（描述）。
輸出 results_posthoc.json。用法：python3 -I -B posthoc.py"""
import json
from pathlib import Path
import numpy as np, pandas as pd
import importlib.util

HERE = Path(__file__).resolve().parent
SCR = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s59')
spec = importlib.util.spec_from_file_location('s59run_nw', HERE / 'run.py')
src = (HERE / 'run.py').read_text()
ns = {}
exec(src[src.index('def nw_ols'):src.index('def nw_mean')], {'np': np, 'stats': __import__('scipy.stats', fromlist=['stats'])}, ns)
nw_ols = ns['nw_ols']

E = pd.read_csv(HERE / 'env_monthly.csv', index_col=0); E.index = pd.PeriodIndex(E.index, freq='M')
E = E.loc['2008-02':'2026-10']
Lm = pd.read_csv(HERE / 'light_monthly.csv', index_col=0); Lm.index = pd.PeriodIndex(Lm.index, freq='M')
E['seg'] = Lm['seg']; E['y'] = E['excess'] * 1200
T = pd.concat([pd.read_csv(SCR / 'oos_trades.csv', dtype={'code': str}), pd.read_csv(SCR / 'is_trades.csv', dtype={'code': str})])
T = T[T['why'] != 'open_end'].copy()
T['xm'] = pd.PeriodIndex(pd.to_datetime(T['exit_date']), freq='M')
out = {'note': '事後探索，PREREG 未登記；看過 results.json 後才設計；不得據以改規則'}

# P1
P1 = {}
for y, g in E.groupby(E.index.year):
    gg = g[g['W_ACT'] >= 0.05]
    t = T[T['xm'].dt.year == y]
    P1[str(y)] = dict(ann_excess_pp=float(g['y'].mean()), mean_w_act=float(g['W_ACT'].mean()),
                      per_unit_ann_pp=float(gg['y'].sum() / gg['W_ACT'].sum()) if len(gg) else None,
                      n_closed_trades=int(len(t)), win_rate=float((t['ret'] > 0).mean()) if len(t) else None,
                      mean_trade_ret=float(t['ret'].mean()) if len(t) else None)
out['P1_by_year'] = P1

# P2／P3：只用有訊號的月份（OOS＋IS），且過去 12 個月與未來 12 個月都不跨 STUCK
prim = E['seg'] != 'STUCK'
E['past12'] = E['y'].rolling(12, min_periods=12).mean().shift(1)
E['fut12'] = E['y'][::-1].rolling(12, min_periods=12).mean()[::-1]          # t…t+11
wr = T.groupby('xm')['ret'].agg(lambda r: (r > 0).sum()).reindex(E.index, fill_value=0)
nn = T.groupby('xm')['ret'].size().reindex(E.index, fill_value=0)
E['win12'] = (wr.rolling(12).sum() / nn.rolling(12).sum()).shift(1)
E['n12'] = nn.rolling(12).sum().shift(1)
clean = pd.Series(False, index=E.index)
for i, m in enumerate(E.index):
    w = E.index[max(0, i - 12):min(len(E), i + 12)]
    clean.iloc[i] = bool((E.loc[w, 'seg'] != 'STUCK').all()) and i >= 12 and i + 12 <= len(E)
for nm, x in (('P2_past12_excess', 'past12'), ('P3_past12_trade_winrate', 'win12')):
    o = {}
    for tgt in ('y', 'fut12'):
        m = clean & E[x].notna() & E[tgt].notna() & prim
        r = nw_ols(E.loc[m, tgt], np.column_stack([np.ones(m.sum()), E.loc[m, x]]), L=12 if tgt == 'fut12' else 6)
        o[tgt] = dict(T=r['T'], slope=float(r['b'][1]), t_nw=float(r['t'][1]), note='fut12 為重疊樣本，NW 落後 12；t 仍偏樂觀')
    m = clean & E[x].notna() & prim
    q = pd.qcut(E.loc[m, x], 3, labels=['低', '中', '高'])
    o['terciles_next_month_ann_pp'] = {str(k): dict(n=int((q == k).sum()), y=float(E.loc[m, 'y'][q == k].mean()), range=[float(E.loc[m, x][q == k].min()), float(E.loc[m, x][q == k].max())]) for k in ['低', '中', '高']}
    out[nm] = o

# P4 候選策略自身燈號：紅＝過去 12 個月已出場勝率 < 40% 或 過去 12 個月月超額平均 < 0；黃＝其中一項接近（勝率 40～45% 或 平均 0～3 個百分點）
def lamp(r):
    if not np.isfinite(r['win12']) or not np.isfinite(r['past12']):
        return None
    if r['win12'] < 0.40 or r['past12'] < 0:
        return '紅'
    if r['win12'] < 0.45 or r['past12'] < 3:
        return '黃'
    return '綠'


E['lamp'] = E.apply(lamp, axis=1)
P4 = {}
for nm, msk in (('OOS', E['seg'] == 'OOS'), ('IS', E['seg'] == 'IS'), ('Y2011_2013_01', (E.index >= pd.Period('2011-01', 'M')) & (E.index <= pd.Period('2013-01', 'M'))),
                ('primary', prim)):
    P4[nm] = {c: dict(n=int((msk & (E['lamp'] == c)).sum()), next_month_ann_pp=float(E.loc[msk & (E['lamp'] == c), 'y'].mean()) if (msk & (E['lamp'] == c)).any() else None)
              for c in ('綠', '黃', '紅')}
P4['latest'] = {str(m): dict(lamp=E.loc[m, 'lamp'], win12=float(E.loc[m, 'win12']), past12_pp=float(E.loc[m, 'past12'])) for m in E.index[-3:]}
P4['first_red_after_2010'] = str(E.index[(E['lamp'] == '紅') & (E.index >= pd.Period('2010-07', 'M'))][0])
out['P4_self_lamp'] = P4
# P5 樣本內超額對少數年份的依賴：依年平均超額由高到低剔除前 k 年
isy = E[E['seg'] == 'IS']
yrs = isy.groupby(isy.index.year)['y'].mean().sort_values(ascending=False)
out['P5_IS_drop_top_years'] = {str(k): dict(dropped=[int(x) for x in yrs.index[:k]], ann_excess_pp=float(isy.loc[~isy.index.year.isin(yrs.index[:k]), 'y'].mean()),
                                            months=int((~isy.index.year.isin(yrs.index[:k])).sum())) for k in range(0, 4)}
out['P5_years_positive'] = dict(OOS_2008_2012=int((pd.Series({int(k): v['ann_excess_pp'] for k, v in P1.items()}).loc[2008:2012] > 0).sum()),
                                IS_2016_2026=int((pd.Series({int(k): v['ann_excess_pp'] for k, v in P1.items()}).loc[2016:2026] > 0).sum()))
(HERE / 'results_posthoc.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
print(json.dumps(out, ensure_ascii=False, indent=1))
