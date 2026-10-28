#!/usr/bin/env python3
"""反方審查（成本與可成交，o4）：只讀 import 研究員的 hb_core，不改任何研究員檔案。
檢查：(1) 出場名額開盤跌停（賣不掉）的比例；(2) 新買名額開盤貼近漲停（>= 漲停價 98%）的比例；
(3) 因處置被迫出場（需預收券）的名額；(4) 賣出端流動性（出場部位 / 60 日均成交值）；
(5) 凍結股仍在入選名單（引擎會對停牌股加碼）的次數；(6) 一般成本下配對差對額外滑價的線性外推。"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import hb_core as H
K, V, R = H.K, H.V, H.R
st = H.setup(); S = H.make_S(); P = H.Prep(S)
V60 = H._V60()
DS = st['DS']

def limit_down_open(t):
    lim = 0.07 if K.IDX[t] < pd.Timestamp('2015-06-01') else 0.10
    ref = K.ref_price(t)
    raw = ref * (1 - lim); tk = K.tick(raw)
    ld = np.ceil(raw / tk - 1e-9) * tk
    return np.isfinite(K.On[t]) & np.isfinite(ref) & (K.On[t] <= ld + 1e-6)

def near_up(t, frac=0.98):
    lim = 0.07 if K.IDX[t] < pd.Timestamp('2015-06-01') else 0.10
    ref = K.ref_price(t); lu = K.limit_up_price(ref, lim)
    return np.isfinite(K.On[t]) & np.isfinite(ref) & (K.On[t] >= frac * lu)

PER = {'SEL': (H.SEL_A, H.SEL_B), 'POST': (H.VAL_A, H.HOLD_B), 'ALL': (H.SEL_A, H.HOLD_B)}
out = {}
for nm, kw in (('BASE', {}), ('A20', dict(K_=20))):
    pl = H.planner(P, 10, **kw)
    rows = []
    for i in range(1, len(pl)):
        e, pk, _, d = pl[i]; prev = set(pl[i - 1][1].tolist()); cur = set(pk.tolist())
        exits = sorted(prev - cur); new = sorted(cur - prev)
        ldo = limit_down_open(e); nu = near_up(e); tr = K.TRADED[e] & np.isfinite(K.On[e])
        disp_d = set(DS.at(d).keys())
        frz_in_pk = [j for j in cur & prev if not tr[j]]
        v60 = V60[e - 1]
        liq_exit = [1e6 / v60[j] for j in exits if np.isfinite(v60[j]) and v60[j] > 0]
        liq_new = [1e6 / v60[j] for j in new if np.isfinite(v60[j]) and v60[j] > 0]
        rows.append(dict(d=d, n_exit=len(exits), n_new=len(new),
                         exit_ldo=sum(bool(ldo[j]) for j in exits if tr[j]),
                         new_near_up=sum(bool(nu[j]) for j in new),
                         exit_disp=sum(j in disp_d for j in exits),
                         frz_in_pk=len(frz_in_pk),
                         exit_over1=sum(x > 0.01 for x in liq_exit), exit_over5=sum(x > 0.05 for x in liq_exit),
                         new_over1=sum(x > 0.01 for x in liq_new)))
    df = pd.DataFrame(rows)
    o = {}
    for p, (a, b) in PER.items():
        x = df[(df.d >= a) & (df.d <= b)]
        ne, nn = x.n_exit.sum(), x.n_new.sum()
        o[p] = dict(months=int(len(x)), exits_per_m=float(x.n_exit.mean()), new_per_m=float(x.n_new.mean()),
                    exit_limitdown_open_share=float(x.exit_ldo.sum() / max(ne, 1)), months_with_ldo_exit=int((x.exit_ldo > 0).sum()),
                    new_near_limitup_share=float(x.new_near_up.sum() / max(nn, 1)),
                    exit_forced_disposal_share=float(x.exit_disp.sum() / max(ne, 1)), exit_disposal_per_yr=float(x.exit_disp.sum() / max(len(x), 1) * 12),
                    frozen_held_in_pick=int(x.frz_in_pk.sum()),
                    exit_pos_over1pct_v60_share=float(x.exit_over1.sum() / max(ne, 1)), exit_pos_over5pct_v60_share=float(x.exit_over5.sum() / max(ne, 1)),
                    new_pos_over1pct_v60_share=float(x.new_over1.sum() / max(nn, 1)),
                    trades_per_yr_names=float((x.n_exit + x.n_new).mean() * 12))
    out[nm] = o
# (6) 配對差對額外單邊滑價的外推（挑選期、跨循環），用研究員 oos.json 的 gross/normal/stress
oo = json.loads((HERE / 'oos.json').read_text(encoding='utf-8'))
ext = {}
for p in ('SEL', 'POST', 'ALL'):
    g = oo['stats']['A20|gross'][p]['paired']['mean_pp']; n = oo['stats']['A20|normal'][p]['paired']['mean_pp']; s = oo['stats']['A20|stress'][p]['paired']['mean_pp']
    slope = (s - n) / 0.3          # pp/月 每 1% 單邊額外滑價
    ext[p] = dict(gross=g, normal=n, stress=s, gain_per_1pct_slip=slope, extra_slip_to_zero_pct=(-n / slope if slope > 0 else None))
out['paired_slip_extrapolation'] = ext
print(json.dumps(out, indent=1, default=float, ensure_ascii=False))
