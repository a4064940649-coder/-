#!/usr/bin/env python3
"""P6：o1 題目的關鍵數字在修補後是否改變（照原程式、不改規格）。python3 -I -B p6.py <ORIG|R_DEL>
  * o_prepriced_growth：斷言修補沒有動到 2014-12 以前的任何格子 → 它的挑選期（2006～2013）與閘門在建構上不變。
  * o_capacity_tsmc_gap：只讀 import 它的 ocore（把 V.load_revenue 換成指定版本），重算 U300 w0、U300 w0.3、U51_300 w0 的
    SEL／VAL／HOLD／ALL，以及容量檢定（n10 入選股 101～300 組 − 1～100 組，ex_pool 月序列）SEL／VAL／ALL。
輸出 p6_<VER>.json。"""
import sys, json, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import osurv_core as Oc
V, K = Oc.V, Oc.K
VER = sys.argv[1]
RES = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
assert RES.get('PREREG'), '先執行 prereg.py'
out = {'version': VER}
YL0, Y50, Y5f0 = V.load_revenue()
YX, _ = Oc.load_yl(VER)
# o_prepriced_growth：2014-12 以前不變
a = YL0.loc[:pd.Period('2014-12', 'M')]
b = YX.reindex(index=a.index, columns=a.columns)
same = ((a == b) | (a.isna() & b.isna())).all().all()
extra_cols_before = [c for c in YX.columns if c not in YL0.columns and YX.loc[:pd.Period('2014-12', 'M'), c].notna().any()]
assert same and not extra_cols_before, '修補動到了 2014-12 以前'
out['prepriced_growth'] = {'cells_changed_le_2014_12': 0, 'note': '挑選期 2006～2013 與閘門（AVG|B1|n30 對等權 t < 1.5）在建構上完全不變；結論「不可用」不受影響'}
_orig = V.load_revenue
V.load_revenue = lambda: (YX, Y50, Y5f0)
sys.path.insert(0, str(HERE.parent / 'o_capacity_tsmc_gap'))
import ocore as OC
sys.path.pop(0)
assert OC.YL is YX
def strip(st): return {k: v for k, v in st.items() if k != 'yearly'}
KEYS = ('SEL', 'VAL', 'HOLD', 'ALL')
def full(univ, w, n=10):
    lo, hi = OC.UNIV[univ]
    S = OC.V2R(lo, hi)
    pl = S.plan(n)
    t0, t1, sd = S.bounds()
    ew, _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, t0, t1)
    nav, _ = OC.simulate_w(OC.tsmc_plan(pl, w), *OC.NORMAL, t0, t1)
    P = OC.periods(sd)
    return S, pl, {k: strip(OC.segstat(nav, ew, sd, *P[k])) for k in KEYS}
cap = {}
for u, w in (('U300', 0.0), ('U300', 0.3), ('U51_300', 0.0)):
    S, pl, st = full(u, w)
    cap[f'{u}|w{int(w*100)}'] = st
    if u == 'U300' and w == 0.0:
        df = OC.event_table(S, pl)
        SP = OC.SPLIT
        R = {'SEL': ('2006-01-01', str((SP - pd.Timedelta(days=1)).date())), 'VAL': (str(SP.date()), '2026-12-31'), 'ALL': ('2006-01-01', '2026-12-31')}
        cap['capacity_mml_ex_pool_n10'] = {k: OC.mml_stats(OC.mid_minus_large(df, 'ex_pool', lo, hi)) for k, (lo, hi) in R.items()}
out['capacity_tsmc_gap'] = cap
(HERE / f'p6_{VER}.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=lambda x: x.item() if hasattr(x, 'item') else str(x)), encoding='utf-8')
print(json.dumps(out, ensure_ascii=False, indent=1, default=str)[:3000])
