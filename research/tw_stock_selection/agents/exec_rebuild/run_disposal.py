#!/usr/bin/env python3
"""C：處置股固定槽位模擬。輸出 results_disposal.json、disposal_nav.parquet。"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import exlib as L, dlib as D

OUT = L.HERE
res = {'_note': '窗格 full／select 對處置股實際從 2021-10-01 起；終點 2026-10-12（主策略最後換股日）。'}
navs = {}
# 原版重現（每日等權再平衡、來回 0.5%、閒置現金）
orig = pd.read_csv(L.HERE.parents[1] / 'output' / 'disposal_daily_returns.csv', index_col=0, parse_dates=True)
for col in ('0.005_cash', '0.005_0050'):
    r = orig[col]; nav = (1 + r).cumprod(); nav = pd.concat([pd.Series([1.0], index=[L.IDX[L.IDX.searchsorted(r.index[0]) - 1]]), nav])
    navs[f'orig_{col}'] = nav
    res[f'orig_{col}'] = {'whole_window_to_2026-10-13': L.daily_stats(nav, [('2021-10-01', '2026-10-31')], L.E50_NAV),
                         'windows_to_2026-10-12': L.all_windows(nav[nav.index <= L.dates[-1]], L.E50_NAV)}
    print(col, res[f'orig_{col}']['whole_window_to_2026-10-13']['CAGR'], flush=True)
# 原版同時持有檔數分布
ev_all, _ = D.build_events(ret_filter=False, delay_exit=False, drop_overlap=False, cost_first=0.005, cost_second=0.005)
hold = np.zeros(len(L.IDX), int)
dp = L.dsp[L.dsp['code'].isin(L.COLS)]
H = pd.DataFrame(False, index=L.IDX, columns=L.COLS)
for _, r in dp.iterrows():
    i0 = L.IDX.searchsorted(r['start']); i1 = L.IDX.searchsorted(r['end'], 'right') - 1
    if i0 >= len(L.IDX) or i1 <= i0 or not np.isfinite(L.ADJ_RAW[i0, L.CI[r['code']]]): continue
    H.iloc[i0 + 1:i1 + 1, L.CI[r['code']]] = True
w = (L.IDX >= '2021-10-01') & (L.IDX <= L.dates[-1]); n = H.sum(axis=1)[w]
res['orig_held_count'] = {'mean': round(float(n.mean()), 2), 'p50': float(n.median()), 'p90': float(n.quantile(.9)), 'max': int(n.max()),
                          'share_days_gt_10': round(float((n > 10).mean()), 3), 'share_days_0': round(float((n == 0).mean()), 3),
                          'distribution_share_days': {('>=15' if k >= 15 else int(k)): 0 for k in []}}
vc = n.clip(upper=15).value_counts().sort_index(); res['orig_held_count']['distribution_share_days'] = {(f'{int(k)}+' if k == 15 else int(k)): round(float(v / len(n)), 3) for k, v in vc.items()}

# 診斷 D0b：原版名單（不濾網、不去重疊、來回 0.5%），全額投入、只在名單變動日調回等權（不做每日再平衡）
Rn = L.RETN; Hn = H.to_numpy(); ids = np.where(w)[0]
navb = [1.0]; wv = np.zeros(len(L.COLS)); prevm = np.zeros(len(L.COLS), bool)
for t in ids:
    m = Hn[t]
    if not np.array_equal(m, prevm):
        k = m.sum(); new = (m & ~prevm).sum()
        wv = np.where(m, 1.0 / k, 0.0) if k else np.zeros_like(wv)
        cost = 0.005 * new / k if k else 0.0
    else:
        cost = 0.0
    rr = np.nan_to_num(Rn[t]); pr = float((wv * rr).sum()) if m.any() else 0.0
    navb.append(navb[-1] * (1 + pr - cost))
    if m.any(): wv = wv * (1 + rr) / (1 + pr)
    prevm = m
navb = pd.Series(navb, index=L.IDX[[ids[0] - 1] + list(ids)])
sb = L.daily_stats(navb, [('2021-10-01', '2026-10-31')], L.E50_NAV)
res['D0b_orig_list_fully_invested_rebalance_on_change_only'] = {k: sb[k] for k in ('CAGR', 'maxDD', 'vol', 't_vs_0050_m', 'yearly_pct')}
so = L.daily_stats(navs['orig_0.005_cash'][navs['orig_0.005_cash'].index <= L.dates[-1]], [('2021-10-01', '2026-10-31')], L.E50_NAV)
res['D0a_orig_daily_EW_same_span'] = {k: so[k] for k in ('CAGR', 'maxDD', 'vol', 't_vs_0050_m', 'yearly_pct')}
print('D0a', res['D0a_orig_daily_EW_same_span']['CAGR'], 'D0b', res['D0b_orig_list_fully_invested_rebalance_on_change_only'], flush=True)

# 拆解（診斷，資金 100 萬、閒置現金）
steps = {
    'D1_slots10_nofilter_rt0.5pct_nocap': dict(use_cap=False, ret_filter=False, delay_exit=False, drop_overlap=False, cost_first=0.005, cost_second=0.005),
    'D2_+entry/exit_filters_+drop_overlap': dict(use_cap=False, cost_first=0.005, cost_second=0.005),
    'D3_+cost_0.8/1.2pct': dict(use_cap=False),
    'D4_+capacity5pct (=正式版 100 萬)': dict(),
}
res['decomposition_1M_cash'] = {}
for k, kw in steps.items():
    nav, st = D.run_standalone(1e6, **kw)
    s = L.daily_stats(nav, [('2021-10-01', '2026-10-31')], L.E50_NAV)
    res['decomposition_1M_cash'][k] = {'CAGR': s['CAGR'], 'maxDD': s['maxDD'], 'vol': s['vol'], 't_vs_0050_m': s['t_vs_0050_m'],
                                       'mean_held': st['mean_held'], 'mean_invested_share': st['mean_invested_share'], 'skipped_no_slot': st['skipped_no_slot'], 'entered': st['entered'],
                                       'trade_net_mean_pct': st.get('trade_net_mean_pct'), 'trade_t': st.get('trade_t')}
    print(k, res['decomposition_1M_cash'][k], flush=True)
# 正式網格
for cap in (1e6, 5e6, 2e7):
    for idle in ('cash', '0050'):
        nav, st = D.run_standalone(cap, idle=idle)
        key = f'slots_K10_cap{int(cap/1e4)}萬_idle_{idle}'
        navs[key] = nav
        res[key] = {'diagnostics': st, 'windows': L.all_windows(nav, L.E50_NAV)}
        w = res[key]['windows']
        print(key, {k: (v['CAGR'], v['maxDD'], v.get('t_vs_0050_m')) for k, v in w.items() if v}, st['mean_held'], st['skipped_no_slot'], st['share_trades_capacity_bound'], flush=True)
(OUT / 'results_disposal.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
pd.DataFrame(navs).to_parquet(OUT / 'disposal_nav.parquet')
