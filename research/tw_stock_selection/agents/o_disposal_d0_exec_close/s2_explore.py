#!/usr/bin/env python3
"""探索（只用挑選期 2021-10-01～2023-12-31）：機制是否看得到＋可執行摩擦的量級。輸出 explore.json。
用法：python3 -I -B s2_explore.py"""
import sys, json, math
sys.dont_write_bytecode = True
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import dlib as D

K = D.K
A, B = D.PER['SEL']
X = {}
ev, unf, st = D.build_x(dict(entry='C1', exit='CL'))
ev = D.sub(ev, A, B).reset_index(drop=True)
uf = D.sub(unf, A, B) if len(unf) else unf
X['n'] = {'filled': int(len(ev)), 'unfilled_lock': int(len(uf))}
# 1. 事件時間路徑（相對 s−1 收盤，對 EW 的累積毛超額；含未成交的鎖漲停事件另列）
def path(df, lo=-10, hi=12):
    out = {}
    for k in range(lo, hi + 1):
        v = []
        for r in df.itertuples():
            a, b = r.s_i - 1, r.s_i + k
            if b >= K.T or a < 0 or not K.TR[a, r.j]: continue
            b = min(b, K.LASTV[r.j])
            v.append(math.expm1(K.LA[b, r.j] - K.LA[a, r.j]) - math.expm1(K.LEW[b] - K.LEW[a]))
        out[k] = round(float(np.mean(v)) * 100, 2) if v else None
    return out
X['path_cum_excess_vs_EW_from_s-1_pct'] = {'filled': path(ev), 'unfilled_lock': path(uf) if len(uf) else None}
# 2. 資訊傳遞：公告日（s−1）報酬、s 日開盤跳空、s 日盤中
ann = []; gap = []; intra = []
for r in ev.itertuples():
    s, j = r.s_i, r.j
    ann.append(K.R[s - 1, j]); gap.append(K.O[s, j] / K.C[s - 1, j] - 1 if K.TR[s - 1, j] else np.nan); intra.append(K.C[s, j] / K.O[s, j] - 1)
q = lambda x: {k: round(float(v) * 100, 2) for k, v in zip(('mean', 'p25', 'median', 'p75'), [np.nanmean(x)] + list(np.nanpercentile(x, [25, 50, 75])))}
X['info_speed_pct'] = {'announce_day_s-1_ret': q(ann), 's_open_gap': q(gap), 's_open_to_close': q(intra)}
# 3. 可執行摩擦（每槽 3 萬／10 萬／30 萬／100 萬，主時點 C1×CL）
prm = D.default_prm()
fr = {}
for nm, slot in (('30萬', 3e4), ('100萬', 1e5), ('300萬', 3e5), ('1000萬', 1e6)):
    x = D.event_exec(ev, slot, prm)
    fr[nm] = {'mean_fill': round(float(x['fill'].mean()), 3), 'capped_share': round(float(x['capped'].mean()), 3),
              'odd_only_share': round(float((x['nl'] == 0).mean()), 3), 'mean_sell_days': round(float(x['sell_days'].mean()), 2),
              'mean_exec_cost_pct': round(float(((1 + x['raw']) - (1 + x['ret_x'])).mean()) * 100, 2),
              'net_x': D.S_ev(x, 'net_x'), 'gross_ideal': D.S_ev(x, 'gross')}
X['exec_by_tier_C1xCL'] = fr
# 4. 子群（毛額＝close-to-close 對 EW，月份群聚 t；挑選期）
ev = D.event_exec(ev, 1e5, prm)
ev['first'] = ~ev['second']
ev['mm20'] = ev['match_min'] >= 20
cuts = K.run6_cuts(D.U)
ev['run6_top'] = ev['run6'] >= cuts[3]
ev['vdrop_big'] = ev['vdrop'] <= ev['vdrop'].median()
ev['whole_lot'] = ev['nl'] >= 1
ps = K.path_split(ev.assign(i0=ev['i0'], ek=ev['ek']))
X['subgroups_gross'] = {f: D.cluster_t(ev.dropna(subset=['gross']), f, 'gross') for f in ('first', 'mm20', 'run6_top', 'vdrop_big', 'whole_lot')}
X['subgroups_net_x'] = {f: D.cluster_t(ev.dropna(subset=['net_x']), f, 'net_x') for f in ('first', 'mm20', 'run6_top', 'vdrop_big', 'whole_lot')}
X['path_split'] = {'early_mean_pct': round(float(ps['early'].mean()) * 100, 2), 'late_mean_pct': round(float(ps['late'].mean()) * 100, 2),
                   'late_minus_early_t': D.tstat(ps['late_m_early'].groupby(ps['date'].dt.to_period('M')).mean())}
# 5. 鎖漲停未成交事件（被成交規則剔除）的事後報酬（只描述，挑選期）
X['note'] = '全部只用挑選期；任何數字不得用來挑事件規則（事件規則不改），只用來理解機制與摩擦量級。'
json.dump(X, open(HERE / 'explore.json', 'w'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(X, ensure_ascii=False, indent=1, default=str)[:6000])
