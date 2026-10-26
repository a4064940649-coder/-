#!/usr/bin/env python3
"""B 類可交易規則（事前固定，全部報告）。
B1：t−1 盤後「連續 2 個營業日注意」（t−1 當天連續天數剛好＝2），t 收盤買；t 盤後公告處置 → 持有到該處置鏈最後一天收盤；否則持有 5 個交易日收盤賣。
B2：t−1 盤後「近 10 個營業日內注意天數剛好達 5 日」（t−1 當天被列注意且 10 日計數＝5），進出場同 B1。
     兩者皆排除：t 當天已在處置期間（含 t−1 已公告處置）。
B3：B1／B2（與聯集）中 t 盤後沒有被公告處置的子樣本（事後分組，只看下行風險）。
B4：處置鏈結束後第 1 個交易日收盤買，持有 5／10 個交易日。
R0（對照，不列入判定）：處置第一天收盤買、處置鏈最後一天收盤賣（主研究的處置股策略）。
共同：異常報酬對同市場流動性合格等權；成本 0.5%，持有期間 (t, exit] 與處置期間重疊者 0.8%；進場日 ret ≥ 9% 買不到；
出場日 ret ≤ −9% 順延到第一個非跌停日；同一檔重疊事件只保留第一筆。上市用面板；上櫃 2025-09-24 起用 history.csv.gz 另列。
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import adlib as L

att, dp = L.load_events_raw()
TW = L.load_twse(); TP = L.load_tpex()
F = L.calendar(TW['idx'], att, TP['idx'])
codes = sorted(set(att['code']) | set(dp['code']) | set(TW['codes']) | set(TP['codes']))
AM = L.att_matrix(att, F, codes).to_numpy()
DPc = L.chains(dp, F)
cov, ann, ann_end = L.disposal_maps(DPc, F, codes)
nF = len(F)
streak = np.zeros(AM.shape, dtype=np.int32)
for i in range(nF):
    streak[i] = (streak[i - 1] + 1) * AM[i] if i else AM[i]
cnt10 = pd.DataFrame(AM.astype(np.int32)).rolling(10, min_periods=1).sum().to_numpy()
att_first = F.searchsorted(att['date'].min())


def plans(sig):
    out = []
    S, J = np.nonzero(sig)
    for s, j in zip(S, J):
        if s < att_first + 10 or s + 1 >= nF: continue
        if cov[s + 1, j] or ann[s, j]: continue
        c = codes[j]; t = F[s + 1]
        if ann[s + 1, j]:
            out.append({'code': c, 't': t, 'exit_date': F[ann_end[(c, s + 1)]], 'path': 'disposed'})
        else:
            out.append({'code': c, 't': t, 'hold_n': 5, 'path': 'not_disposed'})
    return out


sigB1 = AM & (streak == 2)
sigB2 = AM & (cnt10 == 5)
P = {'B1': plans(sigB1), 'B2': plans(sigB2)}
P['B12_union'] = P['B1'] + P['B2']   # 重疊由 simulate 剔除（同日同檔只留一筆）
# B4 / R0
ch = DPc.drop_duplicates(['code', 'chain_s'])
P4_5, P4_10, P0 = [], [], []
for r in ch.itertuples():
    if r.chain_e + 1 < nF:
        P4_5.append({'code': r.code, 't': F[r.chain_e + 1], 'hold_n': 5, 'path': 'post'})
        P4_10.append({'code': r.code, 't': F[r.chain_e + 1], 'hold_n': 10, 'path': 'post'})
    P0.append({'code': r.code, 't': F[r.chain_s], 'exit_date': F[r.chain_e], 'path': 'disposed'})
P['B4_hold5'] = P4_5; P['B4_hold10'] = P4_10; P['R0_disposal_hold'] = P0

res = {'notes': {'calendar_days': int(nF), 'n_chains': int(len(ch)), 'n_disposal_records': int(len(DPc))}}
E_all = {}
for mk, M in [('TWSE', TW), ('TPEX_from_2025-09-24', TP)]:
    mcodes = set(M['codes'])
    for name, pl in P.items():
        pl_m = [p for p in pl if p['code'] in mcodes and p['t'] >= M['idx'][0]]
        E, drop = L.simulate(M, pl_m, cov, F, codes)
        E_all[(mk, name)] = E
        key = f'{mk}|{name}'
        o = {'signals': len(pl_m), 'dropped': drop}
        if len(E):
            o['main'] = L.by_period(E)
            o['verdict'] = L.verdict(o['main']) if mk == 'TWSE' and not name.startswith('R0') else None
            o['liquid_only'] = L.by_period(E[E['liq']])
            if name.startswith('B1') or name.startswith('B2'):
                o['share_disposed_path'] = round(float((E['path'] == 'disposed').mean()), 3)
                o['by_path'] = {pth: L.by_period(g) for pth, g in E.groupby('path')}
                lu = [p for p in pl_m]
            o['yearly'] = {int(y): {'n': int(len(g)), 'net_pct': round(g['net'].mean() * 100, 2), 'median_net_pct': round(g['net'].median() * 100, 2)}
                           for y, g in E.groupby(E['date'].dt.year)}
        res[key] = o
# B3：未被處置的子樣本（下行風險）
for mk in ['TWSE', 'TPEX_from_2025-09-24']:
    for name in ['B1', 'B2', 'B12_union']:
        E = E_all[(mk, name)]
        if not len(E): continue
        g = E[E['path'] == 'not_disposed']
        res[f'{mk}|B3_not_disposed_from_{name}'] = {'main': L.by_period(g), 'vs_disposed_path': L.by_period(E[E['path'] == 'disposed'])}
# 進場日買不到者的路徑分布（選擇偏誤檢查）：重新以不剔除漲停的方式數
chk = {}
for name in ['B1', 'B2']:
    pl = [p for p in P[name] if p['code'] in set(TW['codes'])]
    R = TW['ret']; n_lu = {'disposed': [0, 0], 'not_disposed': [0, 0]}
    for p in pl:
        if p['t'] in R.index and p['code'] in R.columns:
            v = R.at[p['t'], p['code']]
            if np.isfinite(v): n_lu[p['path']][0] += 1; n_lu[p['path']][1] += int(v >= 0.09)
    chk[name] = {k: {'signals_with_price': a, 'limit_up_at_entry': b, 'share': round(b / a, 3) if a else None} for k, (a, b) in n_lu.items()}
res['limit_up_entry_by_path_TWSE'] = chk
L.save(res, 'results_B.json')
for k, v in res.items():
    if k in ('notes', 'limit_up_entry_by_path_TWSE'): print(k, v); continue
    m = v.get('main', {})
    print('\n==', k, v.get('signals'), v.get('dropped'), v.get('verdict'), 'disp_share', v.get('share_disposed_path'))
    for p, x in m.items():
        print('  ', p, x.get('n'), 'ab', x.get('mean_ab_pct'), 'net', x.get('mean_net_pct'), 'med', x.get('median_net_pct'), 'w5', x.get('winsor5_mean_net_pct'),
              'xt10', x.get('ex_top10_mean_net_pct'), 't_m', x.get('t_month'), 'p10', x.get('p10_pct'), 'hold', x.get('mean_hold_days'))
