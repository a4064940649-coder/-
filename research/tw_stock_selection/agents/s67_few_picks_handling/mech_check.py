#!/usr/bin/env python3
"""S67 機械檢查（寫 PREREG 之前；只印「筆數／金額占比」，不計算也不輸出任何候選的報酬、淨值、回撤）：python3 -I -B mech_check.py → mech_check.json
M1 C1 倍數改成 1.0 → 與現行逐位相同（只回報 True／False）；
M2 C3 把放寬門檻設成 10%（補不進任何股票）→ 與現行逐位相同；
M3 C1：加倍的訊號列數、名單太少月份買進的單筆「成本 ÷ 進場前一日淨值」最大值應 ≤ 約 14%（另加成本與開盤跳動的少許誤差）、其他月份 ≤ 約 8%；
M4 C2：名單太少月份的買進筆數＝0；其他月份的訊號名單不變；
M5 C3：只在名單太少月份多出訊號，每月多出檔數＝min(可補檔數, 3 − 名單檔數)，補進者 RET20 介於 5%～10%、都在前 10 名內。"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s67lib as L
from s67lib import s10lib

H = Path(__file__).resolve().parent
P = L.load(); U, RT = L.base(P); L.s17lib.e50_fix_suspension(P); L.slim(P)
S = s10lib.make_signal_s10(P, U, RT, shift=0)
NT = {int(x['k']): int(x['n_tgt']) for x in S['info']}
cols = ['j', 'ent_t', 'exit_t', 'why', 'shares', 'cost', 'proceeds']
same = lambda a, b: bool(np.array_equal(a[0].to_numpy(), b[0].to_numpy(), equal_nan=True) and a[1][cols].equals(b[1][cols]))
cur = L.sim(P, S, U)
out = {}
_, mult, rec = L.apply_cand(P, U, RT, S, 'C1')
out['M1_mult1_identical'] = same(cur, L.sim(P, S, U, mult={k: 1.0 for k in mult}))
lo = L.C3_LO; L.C3_LO = L.MOM_LO
S3x = L.apply_cand(P, U, RT, S, 'C3')[0]; L.C3_LO = lo
out['M2_c3_lo10_identical'] = bool(np.array_equal(S3x['sig'], S['sig'])) and same(cur, L.sim(P, S3x, U))


def fracs(nav, lg):
    prev = nav.shift(1).reindex(lg['ent_date']).to_numpy()
    f = lg['cost'].to_numpy() / prev
    few = (lg['sig_t'].map(NT) <= L.FEW).to_numpy()
    return dict(n_few=int(few.sum()), max_few=float(f[few].max()) if few.any() else None, median_few=float(np.median(f[few])) if few.any() else None,
                n_other=int((~few).sum()), max_other=float(f[~few].max()), median_other=float(np.median(f[~few])))


out['cur_cost_frac'] = fracs(cur[0], cur[1])
n1, l1, st1 = L.sim(P, S, U, mult=mult)
out['M3_C1'] = dict(n_mult_rows=len(mult), engine_mult_rows=int(st1.get('n_mult_rows', 0)), cost_frac=fracs(n1, l1))
del n1
S2 = L.apply_cand(P, U, RT, S, 'C2')[0]
_, l2, _ = L.sim(P, S2, U)
fewk = [k for k, n in NT.items() if n <= L.FEW]; otherk = [k for k, n in NT.items() if n > L.FEW]
out['M4_C2'] = dict(n_buys_in_few_months=int((l2['sig_t'].map(NT) <= L.FEW).sum()), few_rows_cleared=bool(not S2['sig'][fewk].any()),
                    other_rows_same=bool(np.array_equal(S2['sig'][otherk], S['sig'][otherk])))
S3, _, rec3 = L.apply_cand(P, U, RT, S, 'C3')
ok = True; nadd = 0
for x, r in zip([x for x in S['info'] if x['n_tgt'] <= L.FEW], rec3):
    k = x['k']; pool, s, r20 = L.pool_of(P, U, RT, x)
    add = np.flatnonzero(S3['sig'][k] & ~S['sig'][k])
    nadd += len(add)
    ok &= all((j in pool) and (L.C3_LO < r20[j] <= L.MOM_LO) for j in add)
    with np.errstate(invalid='ignore'):
        navail = int(((r20[pool] > L.C3_LO) & ~(r20[pool] > L.MOM_LO)).sum())
    ok &= len(add) == min(navail, max(L.C3_FILL - x['n_tgt'], 0))
out['M5_C3'] = dict(ok=bool(ok), n_added_signals=int(nadd), other_rows_same=bool(np.array_equal(S3['sig'][otherk], S['sig'][otherk])),
                    few_months=len(rec3))
(H / 'mech_check.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps(out, ensure_ascii=False, indent=1))
