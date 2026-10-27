"""步驟 4：挑選期迭代（只用 2008～2018）。研究員 8 版的確切定義不在凍結規格內，
這裡只能重做被選中的 V6（E 收盤買、持有 20 日、pressure ≥ 挑選期中位數），並另跑一個小格點當敏感度參考（非研究員的 V1～V8）。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import rlib as L

x = L.event_frame(); d = L.panel()
S = x[x.tradable & (x.win == 'select') & x.is_del & x.pressure.notna()].copy()
out = {'n_del_tradable_select': int(len(S)), 'cut_used': L.CUT,
       'cut_recomputed_median_tradable': round(float(S.pressure.median()), 4)}
hi = S[S.high]
out['V6'] = {'high': L.summ(hi, 'net20'), 'low': L.summ(S[~S.high], 'net20'), 'HL': L.hl(S, 'net20'),
             'high_stress': L.summ(hi, 'net20s'), 'high_gross_mean': round(hi.car20.mean() * 100, 2)}

# 敏感度格點（複製員自訂，不是研究員的 V1～V8）
idx = d['idx']; T = len(idx)
def ret_open_entry(r, h):
    t = int(r.t)
    if t + 1 + h > T - 1: return np.nan
    o = d['open'][r.code].iloc[t + 1]; c0 = d['close_raw'][r.code].iloc[t + 1]
    if not (np.isfinite(o) and o > 0 and np.isfinite(c0)): return np.nan
    # E+1 開盤買 → E+1+h 收盤（adj 比值 × E+1 開到收）
    g = (c0 / o) * (d['adjf'][r.code].iloc[t + 1 + h] / d['adj'][r.code].iloc[t + 1]) - 1
    ew = L.ew_bh(t, t + 1 + h)   # 近似：等權從 E 收盤起算
    return g - ew
grid = {}
for h in (5, 10, 20):
    k = f'car{h}'
    for cut in ('median', 'tertile_top'):
        thr = L.CUT if cut == 'median' else S.pressure.quantile(2 / 3)
        g = S[S.pressure >= thr].copy(); g['net'] = g[k] - L.COST
        grid[f'Eclose_h{h}_{cut}'] = L.summ(g, 'net')
g = S[S.high].copy(); g['net'] = [ret_open_entry(r, 20) - L.COST for r in g.itertuples()]
grid['E1open_h20_median'] = L.summ(g.dropna(subset=['net']), 'net')
out['grid_replicator_own'] = grid
L.dump(out, 'iterate.json')
import pprint; pprint.pprint(out, width=160)
