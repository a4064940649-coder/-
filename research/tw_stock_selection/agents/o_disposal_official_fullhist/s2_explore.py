#!/usr/bin/env python3
"""第 2 步（探索）：只用挑選期 2021-10～2023 做描述（不碰 2006～2021-09）。→ explore.json"""
import sys, json, math
sys.dont_write_bytecode = True
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import core as K

SELP = 'SEL_2021-10_2023'
U = K.universe()
cuts = K.run6_cuts(U)
out = {'period': K.PER[SELP], 'run6_cuts_all_records_pct': [round(float(c * 100), 1) for c in cuts]}
df, st = K.build({}, U, SELP)
out['counts'] = {'n': int(len(df)), 'reconcile_disposal_v2': '81 筆 +3.07% t_month 1.74（disposal_5y＋共用面板）'}
out['P1_verbatim_SEL'] = {'net': K.S(df), 'gross': K.S(df, 'gross'), 'net50': K.S(df, 'net50'), 'raw_net': K.S(df, 'raw_net')}
# 成本 1.2% / 2%
for c in (0.012, 0.02):
    d2 = df.assign(netc=df['gross'] - c)
    out['P1_verbatim_SEL'][f'net_cost{c}'] = K.S(d2, 'netc')
# 路徑
P = K.path_split(df)
out['path_SEL'] = {'early': K.S(P, 'early'), 'late': K.S(P, 'late'), 'late_minus_early': K.S(P, 'late_m_early'),
                   'N_dist': {int(k): int(v) for k, v in P['N'].value_counts().sort_index().items()}}
# 子群（毛）
df['f_second'] = df['second']; df['f_20min'] = (df['match_min'] == 20); df['f_5min'] = (df['match_min'] == 5)
df['f_vdrop_big'] = df['vdrop'] <= df['vdrop'].median()
df['f_q5'] = df['run6'] >= cuts[3]
G = {}
for f in ('f_second', 'f_vdrop_big', 'f_q5'):
    G[f] = K.cluster_t(df, f, 'gross')
d520 = df[df['match_min'].isin([5, 20])].copy(); d520['f'] = d520['match_min'] == 20
G['20min_minus_5min'] = K.cluster_t(d520, 'f', 'gross')
G['match_min_counts'] = {str(k): int(v) for k, v in df['match_min'].value_counts(dropna=False).items()}
G['vdrop_median_pct'] = round(float(df['vdrop'].median() * 100), 1)
out['subgroups_SEL_gross'] = G
# 不開空頭開關：200 日均上下
dnb, _ = K.build({'bear': False}, U, SELP)
dnb['f_above'] = [not K.bear_before(K.IDX[s]) for s in dnb['s_i']]
out['above_ma_SEL_nobear'] = {'diff': K.cluster_t(dnb, 'f_above', 'gross'), 'n_below': int((~dnb['f_above']).sum())}
# 安慰劑
PL = K.placebo(df, U, cuts)
out['placebo_SEL'] = {'ev_minus_pl': K.S(PL, 'ev_minus_pl'), 'coverage': round(float(PL['pl_raw'].notna().mean()), 3),
                      'mode_counts': {str(k): int(v) for k, v in PL['pl_mode'].value_counts(dropna=False).items()},
                      'median_n_controls': float(PL['pl_n'].median()), 'pl_raw_mean_pct': round(float(PL['pl_raw'].mean() * 100), 2),
                      'ev_raw_mean_pct': round(float(PL['raw'].mean() * 100), 2)}
# 當沖撤出（描述）
try:
    dt = pd.read_csv('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/scr_extra/daytrade_5y.csv.gz',
                     dtype={'code': str}, parse_dates=['date'])
    dt = dt[(dt['dt_ratio'] >= 0) & (dt['dt_ratio'] <= 100)]
    dtp = dt.pivot_table(index='date', columns='code', values='dt_ratio', aggfunc='last')
    dtp.index = pd.DatetimeIndex(dtp.index).astype('datetime64[ns]')
    cs = set(map(str, dtp.columns)); pre, dur, ndays = [], [], []
    for r in df.itertuples():
        if str(r.code) not in cs: continue
        s = dtp[str(r.code)].dropna(); st_ = pd.Timestamp(r.start); en_ = pd.Timestamp(r.end)
        a = s[(s.index < st_) & (s.index >= st_ - pd.Timedelta(days=30))].median()
        sd = s[(s.index >= st_) & (s.index <= en_)]
        b = sd.median() if len(sd) else 0.0
        if np.isfinite(a): pre.append(a); dur.append(b); ndays.append(len(sd))
    out['daytrade_ratio_SEL'] = {'n': len(pre), 'share_events_with_any_daytrade_record_during': round(float(np.mean(np.array(ndays) > 0)), 3), 'pre30d_median_of_medians_pct': round(float(np.median(pre)), 2), 'during_median_of_medians_pct': round(float(np.median(dur)), 2)}
except Exception as e:
    out['daytrade_ratio_SEL'] = {'error': str(e)}
# 成交值降幅
out['value_drop_SEL'] = {'mean_pct': round(float(df['vdrop'].mean() * 100), 1), 'median_pct': round(float(df['vdrop'].median() * 100), 1)}
# 可成交性：每槽 10／30／100 萬的縮單比例
cap_tab = {}
for slot in (1e5, 3e5, 1e6):
    X = K.exec_cost_event(df, slot_2026=slot)
    cap_tab[f'slot_{int(slot/1e4)}萬'] = {'fill_median': round(float(X['fill'].median()), 3), 'fill_mean': round(float(X['fill'].mean()), 3),
                                        'share_capped': round(float((X['fill'] < 0.999).mean()), 3),
                                        'cost_x_mean_pct': round(float(X['cost_x'].mean() * 100), 2), 'net_x': K.S(X, 'net_x')}
out['capacity_SEL'] = cap_tab
# 9 種時點（100 萬、每槽 10 萬、可執行成本）
grid = {}
for e in ('C1', 'O2', 'C2'):
    for x in ('CL', 'CL1', 'OA'):
        d, _ = K.build({'entry': e, 'exit': x}, U, SELP)
        X = K.exec_cost_event(d, 1e5)
        grid[f'{e}x{x}'] = {'flat': K.S(d), 'exec': K.S(X, 'net_x')}
out['timing_grid_SEL'] = grid
vals = [v['exec'].get('mean_pct') for v in grid.values()]
out['timing_grid_SEL_summary'] = {'median_pct': float(np.median(vals)), 'p10_pct': float(np.percentile(vals, 10))}
(HERE / 'explore.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps(out, ensure_ascii=False, indent=1, default=str)[:7000])
