#!/usr/bin/env python3
"""檢定 4：除權息深入（exdiv_5y，2021-10 起；只用面板中的上市股）。事前固定規則：
A. 搶息：除息日 t 的前 11 個交易日收盤買（t-11）、前 1 日收盤賣（t-1）＝涵蓋除息前 10 個交易日報酬，不含股利。
   殖利率＝(前收 − 參考價) / 進場日收盤（股利金額與除息日通常在進場前已公告；此為假設）。
   分組固定：< 2%、2–4%、4–6%、≥ 6%；另報「只含現金股利（kind=息）」。
B. 填息 vs 貼息：除息日 t 至 t+4（共 5 個交易日）任一收盤 ≥ 除息前收 → 填息，否則貼息；
   t+5 收盤（第 6 日）進場，持有 20／60 個交易日。另報殖利率 ≥ 4% 子樣本。
共同：進場日流動性合格（60 日均成交值 ≥ 1,000 萬、股價 ≥ 10）；進場收盤鎖漲停者剔除；異常報酬對流動性合格等權；成本來回 0.5%。
日曆時間組合：A 的全部事件（與最佳殖利率組）、B 的填息組與貼息組；閒置持現金與持 0050 兩版本。
挑選期 2021-10～2023-12、驗證期 2024～2026。
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evlib as L

D = L.load(); idx = D['idx']; C = D['close_raw']; liq = D['liq']; lu = D['limit_up_close']
SPL = {'sel_2021-10_2023': ('2021-10-01', '2023-12-31'), 'val_2024_2026': ('2024-01-01', '2026-12-31')}
ex = pd.read_csv(L.X / 'exdiv_5y.csv.gz', dtype={'code': str}, parse_dates=['date']); ex = ex[ex['code'].isin(C.columns)]
pos = {d: i for i, d in enumerate(idx)}; cols = {c: j for j, c in enumerate(C.columns)}
Cn = C.to_numpy(); LQ = liq.to_numpy(); LU = lu.to_numpy()
car = L.Car(D, 'mkt_liq')
A, B = [], []
for _, r in ex.iterrows():
    i = pos.get(r['date']); j = cols[r['code']]
    if i is None or i < 12: continue
    div = r['prev_close'] - r['ref_price']
    # A
    i0 = i - 11
    if LQ[i0, j] and not LU[i0, j] and np.isfinite(Cn[i0, j]):
        raw, ab, ab50 = car(r['code'], i0, i - 1)
        A.append({'date': idx[i0], 'code': r['code'], 'kind': r['kind'], 'y': div / Cn[i0, j], 'raw': raw, 'ab': ab, 'ab50': ab50, 'i0': i0, 'i1': i - 1})
    # B
    if i + 5 < len(idx):
        w = Cn[i:i + 5, j]; i0 = i + 5
        if np.isfinite(w).sum() == 0 or not LQ[i0, j] or LU[i0, j] or not np.isfinite(Cn[i0, j]): continue
        filled = bool(np.nanmax(w) >= r['prev_close'] - 1e-9)
        o = {'date': idx[i0], 'code': r['code'], 'kind': r['kind'], 'y': div / r['prev_close'], 'filled': filled, 'i0': i0}
        for h in (20, 60):
            o[f'raw{h}'], o[f'ab{h}'], o[f'ab50_{h}'] = car(r['code'], i0, i0 + h)
        B.append(o)
A = pd.DataFrame(A); B = pd.DataFrame(B)
res = {'n_A': int(len(A)), 'n_B': int(len(B)), 'B_fill_rate': round(float(B['filled'].mean()), 3)}
bins = [-1, 0.02, 0.04, 0.06, 10]; labs = ['<2%', '2-4%', '4-6%', '>=6%']
A['yg'] = pd.cut(A['y'], bins, labels=labs)
res['A_pre_exdiv'] = {'all': L.by_period(A, SPL), 'cash_only': L.by_period(A[A['kind'] == '息'], SPL)}
for g in labs: res['A_pre_exdiv'][f'yield_{g}'] = L.by_period(A[A['yg'] == g], SPL)
# 穩健性（事後加報，非用來挑參數）：「權／權息」的 (前收−參考價) 含股票股利／現增，不是現金殖利率 → 只看 kind=息 的殖利率分組
for g in labs: res['A_pre_exdiv'][f'cash_only_yield_{g}'] = L.by_period(A[(A['yg'] == g) & (A['kind'] == '息')], SPL)
res['A_pre_exdiv']['stock_div_or_rights_(權,權息)'] = L.by_period(A[A['kind'] != '息'], SPL)
res['A_yearly_all'] = L.yearly_ab(A)
B['yg'] = np.where(B['y'] >= 0.04, 'hi', 'lo')
mk = lambda g, h: pd.DataFrame({'date': g['date'], 'raw': g[f'raw{h}'], 'ab': g[f'ab{h}'], 'ab50': g[f'ab50_{h}']})
res['B_fill'] = {}
for name, g in {'filled': B[B['filled']], 'unfilled': B[~B['filled']], 'filled_yield>=4%': B[B['filled'] & (B['yg'] == 'hi')],
                'unfilled_yield>=4%': B[~B['filled'] & (B['yg'] == 'hi')], 'all': B}.items():
    res['B_fill'][name] = {f'h{h}': L.by_period(mk(g, h), SPL) for h in (20, 60)}
# 填息 − 貼息 差（以進場月份配對）
for h in (20, 60):
    out = {}
    for p, (a, b) in SPL.items():
        bb = B[(B['date'] >= a) & (B['date'] <= b)]
        mm = bb.groupby([bb['date'].dt.to_period('M'), 'filled'])[f'ab{h}'].mean().unstack()
        dd = (mm[True] - mm[False]).dropna()
        out[p] = {'months': int(len(dd)), 'mean_filled_minus_unfilled_pct': round(float(dd.mean() * 100), 2), 't': L.tstat(dd)}
    res['B_fill'][f'diff_h{h}'] = out
# 日曆時間組合
res['calendar'] = {}
for name, tr in {'A_all': list(zip(A['code'], A['i0'], A['i1'])), 'A_yield>=6%': list(zip(A[A['yg'] == '>=6%']['code'], A[A['yg'] == '>=6%']['i0'], A[A['yg'] == '>=6%']['i1'])),
                 'A_cash_yield>=6%': list(zip(A[(A['yg'] == '>=6%') & (A['kind'] == '息')]['code'], A[(A['yg'] == '>=6%') & (A['kind'] == '息')]['i0'], A[(A['yg'] == '>=6%') & (A['kind'] == '息')]['i1'])),
                 'A_stock_or_rights': list(zip(A[A['kind'] != '息']['code'], A[A['kind'] != '息']['i0'], A[A['kind'] != '息']['i1'])),
                 'A_yield4-6%': list(zip(A[A['yg'] == '4-6%']['code'], A[A['yg'] == '4-6%']['i0'], A[A['yg'] == '4-6%']['i1'])),
                 'B_filled_h20': [(c, i, min(i + 20, len(idx) - 1)) for c, i in zip(B[B['filled']]['code'], B[B['filled']]['i0'])],
                 'B_unfilled_h20': [(c, i, min(i + 20, len(idx) - 1)) for c, i in zip(B[~B['filled']]['code'], B[~B['filled']]['i0'])]}.items():
    for idle in ('cash', '0050'):
        for p, (a, b) in SPL.items():
            res['calendar'][f'{name}_idle{idle}_{p}'] = L.calendar_portfolio(D, tr, a, min(pd.Timestamp(b), idx[-1]), idle=idle)
L.save('4_exdiv', res)
print({k: v for k, v in res.items() if k.startswith('n_') or k == 'B_fill_rate'})
for k, v in res['A_pre_exdiv'].items(): print('A_tmonth', k, {p: (x.get('t_month'), x.get('n_months')) for p, x in v.items()})
for k, v in res['A_pre_exdiv'].items(): print('A', k, {p: (x.get('n'), x.get('mean_ab_pct'), x.get('median_ab_pct'), x.get('hit_ab'), x.get('t_date'), x.get('net_ab_pct')) for p, x in v.items()})
for k, v in res['B_fill'].items():
    if k.startswith('diff'): print('B', k, v); continue
    for h, e in v.items(): print('B', k, h, {p: (x.get('n'), x.get('mean_ab_pct'), x.get('median_ab_pct'), x.get('t_month'), x.get('net_ab_pct')) for p, x in e.items()})
for k, v in res['calendar'].items(): print(k, v['strategy'], v['0050']['CAGR'], v['t_vs_0050'], v.get('excess_vs_EW_bp_on_invested_days'), v.get('t_vs_EW_invested'), v['share_days_invested'])
