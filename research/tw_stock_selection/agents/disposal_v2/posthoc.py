#!/usr/bin/env python3
"""事後檢查（看完 run.py 結果後才加，不是事前登記，不能當證據）：
1. 制度過渡修正：disposal_5y 對 2026-08-10 前開始、之後結束的記錄仍是原本 10／12 日的 end。
   依新制過渡規定（實施日前已滿 5／7 日者自 8/10 解除；未滿者執行到滿 5／7 日），把這些記錄的 end 改成：
   新天數＝5（原 ≤10 日）或 7（原 >10 日，當沖加重）；第 新天數 個交易日 < 2026-08-10 → end＝2026-08-07；否則 end＝第 新天數 個交易日。
   重算正式版逐筆統計與上市 10 槽。
2. 第二次處置排除（觀察）：正式版去掉 second 事件的逐筆統計。
3. 處置第 5 天收盤才買（10 日處置）：描述性，從逐日曲線算。
寫入 results.json 的 POSTHOC_not_preregistered。"""
import sys, json, runpy
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
G = runpy.run_path(str(HERE / 'run.py'), run_name='disposal_v2_run')   # 重跑主程式取得物件（會覆寫 results.json 的 RESULTS，內容相同）
L = G['L']; IDX = G['IDX']; TXI = G['TXI']
CH = pd.Timestamp('2026-08-10'); LASTOLD = pd.Timestamp('2026-08-07')


def fix_end(rec):
    rec = rec.copy(); n = 0
    for k, r in rec.iterrows():
        if r['start'] < CH <= r['end']:
            ln = G['tdays'](r['start'], r['end']); new = 5 if ln <= 10 else 7
            s = TXI.searchsorted(r['start']); e_new = TXI[s + new - 1]
            if e_new < CH: e_new = LASTOLD
            rec.at[k, 'end'] = e_new; rec.at[k, 'e_i'] = IDX.searchsorted(e_new, 'right') - 1; n += 1
    return rec, n


out = {}
L.set_market(L.TWSE)
recW, nW = fix_end(L.REC)
TWf, _ = G['build'](recW, G['CFG_OFF'], 'TWSE')
L.set_market(G['MP'])
recP, nP = fix_end(G['recP'])
TPf, _ = G['build'](recP, G['CFG_OFF'], 'TPEX')
L.set_market(L.TWSE)
for d in (TWf, TPf):
    d['second'] = d['second'].astype(bool)
out['transition_fix'] = {'records_changed_TWSE': nW, 'records_changed_TPEX': nP,
                         'TWSE': G['per'](TWf), 'TPEX': G['per'](TPf), 'POOLED': G['per'](pd.concat([TWf, TPf], ignore_index=True))}
nav, s, tr, held = G['run_sleeve'](TWf, '0050')
out['transition_fix']['sleeve_TWSE_0050'] = {'full': G['nstats'](nav), 'last12': G['nstats'](nav, '2025-10-22', '2026-10-22')}
TW = G['TW']; ALL = G['ALL']
out['exclude_second_TWSE'] = G['per'](TW[~TW['second']])
out['exclude_second_POOLED'] = G['per'](ALL[~ALL['second']])
nav, s, tr, held = G['run_sleeve'](TW[~TW['second']], '0050')
out['exclude_second_sleeve_TWSE_0050'] = {'full': G['nstats'](nav), 'sel': G['nstats'](nav, '2021-09-30', '2023-12-31'), 'val': G['nstats'](nav, '2023-12-29', '2026-10-22')}
# 第 5 天收盤才買（10 日處置，曲線上 day4→day9）
cv = [c for c in G['CV'] if c['N'] == 9]
def d5(items):
    if not items: return None
    a = np.array([c['ab'][9] - c['ab'][4] for c in items]); dts = pd.Series([c['date'] for c in items])
    m = pd.Series(a).groupby(dts.dt.to_period('M').to_numpy()).mean()
    return {'n': len(a), 'mean_gross_ab_pct': round(float(a.mean() * 100), 2), 't_month': G['tst'](m)}
out['buy_day5_close_hold_to_end_10day'] = {'sel_TWSE': d5([c for c in cv if c['mkt'] == 'TWSE' and c['date'] <= pd.Timestamp('2023-12-31')]),
                                           'val_pre_change': d5([c for c in cv if pd.Timestamp('2024-01-01') <= c['date'] < CH]),
                                           'last12': d5([c for c in cv if c['date'] >= pd.Timestamp('2025-10-23')])}
# 目前回撤
N = G['NAVS']['TWSE_K10_0050']; dd = N / N.cummax() - 1
out['sleeve_TWSE_0050_drawdown'] = {'max_dd_trough': str(dd.idxmin().date()), 'max_dd_pct': round(float(dd.min() * 100), 1),
                                     'peak_before': str(N[:dd.idxmin()].idxmax().date()), 'dd_at_end_pct': round(float(dd.iloc[-1] * 100), 1)}
R = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
R['POSTHOC_not_preregistered'] = out
(HERE / 'results.json').write_text(json.dumps(R, ensure_ascii=False, indent=1, default=G['conv']), encoding='utf-8')
K = ('n', 'mean_pct', 'median_pct', 'hit', 't_month', 'n_months')
for k in ('TWSE', 'TPEX', 'POOLED'):
    for p in ('val', 'last12', 'post_change'): print('FIX', k, p, {q: out['transition_fix'][k][p].get(q) for q in K})
print('FIX changed', nW, nP, out['transition_fix']['sleeve_TWSE_0050'])
for k in ('exclude_second_TWSE', 'exclude_second_POOLED'):
    for p in ('sel', 'val', 'last12'): print(k, p, {q: out[k][p].get(q) for q in K})
print(out['exclude_second_sleeve_TWSE_0050']); print(out['buy_day5_close_hold_to_end_10day']); print(out['sleeve_TWSE_0050_drawdown'])
