#!/usr/bin/env python3
"""S80 驗收（PREREG 之前跑；只碰現行格與機械性檢查，不輸出任何非現行格的報酬）。
A0：simulate_s80 不開掛鉤 ＝ L.sim（現行）逐位相同，且＝S75 存的 V_BRIDGE 淨值（存檔四捨五入到分）。
A1：tend 截斷的訓練段淨值＝全期跑的前段逐日相同（現行格）。
A2：cashmode 全 1（IDLE_CASH_ALWAYS）＝ s47lib.simulate_s47(idle='cash') 逐位相同（只比到訓練段末日、只報是否相同）。
A3：EXIT 格機械檢查（訓練段、只報筆數）：每個弱勢檢查日的次一交易日收盤，持股只剩「賣不掉而遞延」的。
另存：指標觸發天數（不含報酬）。python3 -I -B accept.py → accept.json"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import s80lib as S
from s80lib import K, L, s10lib
t0 = time.time()
P, U = K.load_cont()
F, RT = K.surge(P, 'B2')
sg = K.signal(P, U, RT, 0)
tr_end = int(np.searchsorted(P.IDX.values, np.datetime64(S.TRAIN_END), side='right'))   # 訓練段最後一列＋1
out = dict(panel=dict(T=P.T, N=P.N, first=str(P.IDX[P.t0].date()), last=str(P.IDX[-1].date())), train_end_row=str(P.IDX[tr_end - 1].date()), src_sha=S.SRC_SHA)
t1 = time.time()
n0, l0, s0 = L.sim(P, sg, U, cap=1e6)
out['sec_per_full_sim'] = round(time.time() - t1, 2)
n1, l1, s1 = S.sim(P, sg, U, cap=1e6)
ref = pd.read_csv(S.AG / 's75_continuous_2008_2026' / 'nav_2008_2026.csv.gz', index_col=0, parse_dates=True)['V_BRIDGE']
out['A0'] = dict(bitwise_vs_Lsim=bool(np.array_equal(n0.to_numpy(), n1.to_numpy()) and len(l0) == len(l1)), max_abs_vs_S75_csv=float((n1.round(2) - ref.reindex(n1.index)).abs().max()),
                 cagr_full=S.cagr(n1), maxdd_full=S.maxdd(n1))
n2, l2, s2 = S.sim(P, sg, U, cap=1e6, tend=tr_end)
a = n2.to_numpy(); b = n1.to_numpy()[:len(a)]
out['A1'] = dict(len=int(len(a)), last=str(n2.index[-1].date()), bitwise_prefix=bool(np.array_equal(a, b)))
IND, br = S.indicators(P, U); WE = S.week_ends(P)
c = S.CELL['IDLE_CASH_ALWAYS']
sgc, cm, szm, st, ch = S.apply_rule(P, U, sg, c, IND, WE)
n3, _, _ = S.sim(P, sgc, U, cap=1e6, cashmode=cm, tend=tr_end)
S2, ekw = s10lib.variant_S(sg, 'E0')
n4, _, _ = L.simulate_s47(P, S2, U, ndiv=None, idle='cash', cost=L.COSTS['strict'], capital=1e6, cap_pct=0.05, sat_w=0.7, sizing='W10', lock=False, **ekw)
out['A2'] = dict(bitwise_train_prefix=bool(np.array_equal(n3.to_numpy(), n4.to_numpy()[:len(n3)])))
# A3
chk = {}
for cid in ('EXIT_E50_M_E50MA120', 'EXIT_ALLCASH_W_E50MA120', 'EXIT_CASH_WR_BR40'):
    c = S.CELL[cid]
    sg2, cm, szm, st, ch = S.apply_rule(P, U, sg, c, IND, WE)
    nav, lg, sx = S.sim(P, sg2, U, cap=1e6, cashmode=cm, szm=szm, tend=tr_end)
    npos = K.npos_series(P, lg).to_numpy()
    cs = [t for t in np.flatnonzero(ch & st) if t + 1 < tr_end - 1]
    left = int(sum(npos[t + 1] > 0 for t in cs))
    chk[cid] = dict(weak_checks=len(cs), days_after_with_positions=left, sell_defer_limdn=int(sx['sell_defer_limdn']), sell_defer_noopen=int(sx['sell_defer_noopen']),
                    sell_partial_cap=int(sx['sell_partial_cap']), finite_nav=bool(np.isfinite(nav.to_numpy()).all()))
out['A3'] = chk
tr = slice(P.t0, tr_end); va = slice(tr_end, P.T)
out['trigger_days_frac'] = {k: dict(train=float(v[tr].mean()), valid=float(v[va].mean())) for k, v in IND.items()}
ks = np.flatnonzero(sg['sigrows'])
out['trigger_signal_months'] = {k: dict(train=int(v[ks[ks < tr_end]].sum()), n_train=int((ks < tr_end).sum()), valid=int(v[ks[ks >= tr_end]].sum()), n_valid=int((ks >= tr_end).sum())) for k, v in IND.items()}
out['breadth'] = dict(train_median=float(np.nanmedian(br[tr])), valid_median=float(np.nanmedian(br[va])))
out['runtime_s'] = round(time.time() - t0, 1)
(S.HERE / 'accept.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps(out, ensure_ascii=False, indent=1))
