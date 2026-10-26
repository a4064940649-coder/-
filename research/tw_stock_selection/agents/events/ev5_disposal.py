#!/usr/bin/env python3
"""檢定 5：處置股策略的變形（disposal_5y；date＝公告日、start/end＝處置期間；只用面板中的上市股）。
主策略：處置第一個交易日（start）收盤買、最後一個交易日（end）收盤賣，等權，來回 0.5%。
事前固定的分組（分界點只用挑選期決定）：
  G1 處置前 6 日漲幅：公告日收盤 / 公告日前 6 個交易日收盤 − 1（公告日盤後即知）；以挑選期事件的三分位數切三組。
  G2 第二次（再）處置：同一檔在本次處置第一天之前 30 個交易日內有另一筆處置結束 → 再處置；否則第一次。
  G3 處置期間長度（交易日數）：≤ 9 日、10 日、≥ 11 日。
  G4 市場多空：公告日加權指數收盤是否在 200 日均線之上。
  出場時點：最後一天收盤賣 vs 結束後第 1 天開盤賣（差額＝結束日收盤→次日開盤的隔夜跳空）、結束後第 1 天收盤賣（參考）。
共同：進場收盤鎖漲停 → 買不到，剔除（另報不剔除的原始版本，對照主研究 +4.4%）；異常報酬對流動性合格等權。
另報公告日流動性合格（60 日均成交值 ≥ 1,000 萬、股價 ≥ 10）子樣本。
日曆時間組合（閒置持現金／持 0050）：主策略與各組。挑選期 2021-10～2023-12、驗證期 2024～2026（依處置開始日）。
注意：2024-02-15～2025-09-23 的收盤為 Yahoo 13:20 近似，開盤為第一根 5 分 K 開盤。
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evlib as L

D = L.load(); idx = D['idx']; C = D['close_raw']; O = D['open']; adj = D['adj']; F = D['flag']
SPL = {'sel_2021-10_2023': ('2021-10-01', '2023-12-31'), 'val_2024_2026': ('2024-01-01', '2026-12-31')}
dp = pd.read_csv(L.X / 'disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['date', 'start', 'end'])
dp = dp[dp['code'].isin(C.columns)].sort_values(['start', 'code']).reset_index(drop=True)
tw = D['twii'].sort_index(); ma200 = tw.rolling(200).mean(); bull = (tw > ma200).reindex(idx, method='ffill')
cols = {c: j for j, c in enumerate(C.columns)}
An, Cn, On, LQ, LU, Fn = adj.to_numpy(), C.to_numpy(), O.to_numpy(), D['liq'].to_numpy(), D['limit_up_close'].to_numpy(), F.to_numpy()
car = L.Car(D, 'mkt_liq')
rows = []; prev_end = {}
for _, r in dp.iterrows():
    c = r['code']; j = cols[c]
    i_a = idx.searchsorted(r['date'], 'right') - 1; i_s = idx.searchsorted(r['start']); i_e = idx.searchsorted(r['end'], 'right') - 1
    second = c in prev_end and prev_end[c] >= i_s - 30
    prev_end[c] = max(prev_end.get(c, -1), i_e)
    if i_s >= len(idx) or i_e <= i_s or i_a < 6 or not np.isfinite(An[i_s, j]): continue
    o = {'date': idx[i_s], 'code': c, 'i_s': i_s, 'i_e': i_e, 'ndays': i_e - i_s + 1, 'second': bool(second),
         'run6': An[i_a, j] / An[i_a - 6, j] - 1 if np.isfinite(An[i_a - 6, j]) else np.nan, 'bull': bool(bull.iat[i_a]) if pd.notna(bull.iat[i_a]) else np.nan,
         'liq': bool(LQ[i_a, j]), 'lu_entry': bool(LU[i_s, j])}
    o['raw'], o['ab'], o['ab50'] = car(c, i_s, i_e)
    # 次日開盤出場：隔夜跳空（次日若為除權息／公司行動日則缺值）
    if i_e + 1 < len(idx) and np.isfinite(On[i_e + 1, j]) and np.isfinite(Cn[i_e, j]) and Fn[i_e + 1, j] in (0,):
        o['gap'] = On[i_e + 1, j] / Cn[i_e, j] - 1
        o['next_close_ab'] = car(c, i_s, i_e + 1)[1]
    else:
        o['gap'] = np.nan; o['next_close_ab'] = np.nan
    rows.append(o)
E = pd.DataFrame(rows)
sel = E[(E['date'] >= SPL['sel_2021-10_2023'][0]) & (E['date'] <= SPL['sel_2021-10_2023'][1])]
q1, q2 = sel['run6'].quantile([1 / 3, 2 / 3])
E['g_run6'] = np.where(E['run6'] <= q1, 'low', np.where(E['run6'] <= q2, 'mid', 'high'))
E['g_len'] = np.where(E['ndays'] <= 9, '<=9d', np.where(E['ndays'] == 10, '10d', '>=11d'))
T = E[~E['lu_entry']]           # 可成交版本
res = {'n_events_raw': int(len(E)), 'n_limit_up_entry_removed': int(E['lu_entry'].sum()), 'n_tradable': int(len(T)),
       'run6_tercile_cutoffs_from_selection': [round(float(q1), 4), round(float(q2), 4)],
       'counts_by_year': {int(y): int(n) for y, n in T.groupby(T['date'].dt.year).size().items()},
       'share_second': round(float(T['second'].mean()), 3), 'share_bull': round(float(T['bull'].mean()), 3)}
res['base_untradable_included'] = L.by_period(E, SPL)
res['base'] = L.by_period(T, SPL)
res['base_liquid'] = L.by_period(T[T['liq']], SPL)
res['yearly_base'] = L.yearly_ab(T)
groups = {}
for g in ('low', 'mid', 'high'): groups[f'run6_{g}'] = T[T['g_run6'] == g]
groups['first_disposal'] = T[~T['second']]; groups['second_disposal'] = T[T['second']]
for g in ('<=9d', '10d', '>=11d'): groups[f'len_{g}'] = T[T['g_len'] == g]
groups['bull_TAIEX>MA200'] = T[T['bull'] == True]; groups['bear_TAIEX<=MA200'] = T[T['bull'] == False]
res['groups'] = {k: L.by_period(g, SPL) for k, g in groups.items()}
def welch(a, b):
    a, b = a.dropna(), b.dropna()
    if len(a) < 3 or len(b) < 3: return None
    return {'diff_pct': round(float((a.mean() - b.mean()) * 100), 2), 't': round(float((a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b))), 2), 'n': [int(len(a)), int(len(b))]}
res['group_diff_tests'] = {}
for na, nb in [('run6_high', 'run6_low'), ('run6_mid', 'run6_low'), ('second_disposal', 'first_disposal'), ('len_10d', 'len_>=11d'), ('bull_TAIEX>MA200', 'bear_TAIEX<=MA200')]:
    o = {}
    for p, (a, b) in SPL.items():
        ga, gb = groups[na], groups[nb]
        o[p] = welch(ga[(ga['date'] >= a) & (ga['date'] <= b)]['ab'], gb[(gb['date'] >= a) & (gb['date'] <= b)]['ab'])
    res['group_diff_tests'][f'{na}_minus_{nb}'] = o
res['crosstab_second_vs_len'] = {str(k): int(v) for k, v in T.groupby(['second', 'g_len']).size().items()}
# 出場時點
ex = {}
for p, (a, b) in SPL.items():
    t = T[(T['date'] >= a) & (T['date'] <= b)].dropna(subset=['gap'])
    gm = t.groupby(t['date'].dt.to_period('M'))['gap'].mean()
    ex[p] = {'n': int(len(t)), 'mean_overnight_gap_after_end_pct': round(float(t['gap'].mean() * 100), 3), 'median_gap_pct': round(float(t['gap'].median() * 100), 3),
             't_month': L.tstat(gm), 'hit_gap>0': round(float((t['gap'] > 0).mean()), 3),
             'close_exit_mean_ab_pct': round(float(t['ab'].mean() * 100), 2), 'next_open_exit_mean_ab_approx_pct': round(float((t['ab'] + t['gap']).mean() * 100), 2),
             'next_close_exit_mean_ab_pct': round(float(t['next_close_ab'].mean() * 100), 2)}
res['exit_timing'] = ex
# 日曆時間組合
res['calendar'] = {}
for name, g in [('base', T)] + list(groups.items()):
    tr = list(zip(g['code'], g['i_s'], g['i_e']))
    for p, (a, b) in SPL.items():
        for idle in ('cash', '0050'):
            v = L.calendar_portfolio(D, tr, a, min(pd.Timestamp(b), idx[-1]), idle=idle)
            res['calendar'][f'{name}_idle{idle}_{p}'] = v
L.save('5_disposal', res)
print({k: v for k, v in res.items() if k not in ('groups', 'calendar', 'base', 'base_liquid', 'base_untradable_included', 'exit_timing')})
for k in ('base_untradable_included', 'base', 'base_liquid'):
    print(k, {p: (x.get('n'), x.get('mean_raw_pct'), x.get('mean_ab_pct'), x.get('median_ab_pct'), x.get('t_month'), x.get('net_ab_pct')) for p, x in res[k].items()})
for k, v in res['groups'].items(): print(k, {p: (x.get('n'), x.get('mean_raw_pct'), x.get('mean_ab_pct'), x.get('median_ab_pct'), x.get('t_month'), x.get('net_ab_pct')) for p, x in v.items()})
for p, v in ex.items(): print('exit', p, v)
for k, v in res['group_diff_tests'].items(): print('diff', k, v)
for k, v in res['calendar'].items(): print(k, v['strategy'], v['0050']['CAGR'], v['t_vs_0050'], v['years_beat_0050'], v.get('share_days_invested'))
