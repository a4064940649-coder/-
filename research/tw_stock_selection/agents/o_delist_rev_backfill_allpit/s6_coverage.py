#!/usr/bin/env python3
"""步驟 2(b) 覆蓋表（凍結前；不算任何變體的報酬、t 或名單報酬差）。python3 -I -B s6_coverage.py <B_ver>
槽位覆蓋：U13 opt 放回槽 (訊號日 d, 代碼 c) 視為「已覆蓋」⇔ 引擎需要的每個月年增都有值（tw_revenue 原有或 rev_backfill）：
  d < 2021-03：m−1..m−3；若 d 是 D=S／D>S 日，再加 m−4（A_UB 回退一個月要用到）。d ≥ 2021-03：rev_5y 欄 m−1..m−3（同上加 m−4）。
  KY（口徑剔除，不放回）另列，不算覆蓋。
報酬權重：每槽權重＝|放回股下一期報酬|＋|該日被擠掉 C0 股下一期報酬的平均|（下一期＝訊號日收盤到下一訊號日收盤，含息還原價）。
只當權重用（絕對值），不計任何差或平均報酬。"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent
BV = sys.argv[1] if len(sys.argv) > 1 else 'B1'
sys.path.insert(0, str(AG / 'o_rev_avail_ds_pit_count')); import pclib as P; sys.path.pop(0)
V, K = P.V, P.K
RB = pd.read_csv(HERE / f'data/rev_backfill_{BV}.csv', dtype={'code': str})
RBY = {(c, y): v for c, y, v in zip(RB.code, RB.ym, RB.yoy_pct) if np.isfinite(v)}
YL, Y5 = P.YL, P.Y5
KY = {'1262', '1592', '6422', '6452'}
SL = pd.read_csv(HERE / 'data/demand_slots.csv', dtype={'code': str})
c0 = P.c0_cached(False)
C0 = {str(r['sig'].date()): r['picks'] for r in c0}
S = P.VU('opt', lo='2013-01-01'); L = P.u_lists(S)
sigs = sorted(C0)
def has_yoy(c, x, d):
    Y = Y5 if pd.Timestamp(d) >= V.SWITCH else YL
    try: v = Y.at[x, c]
    except KeyError: v = np.nan
    return bool(np.isfinite(v)) or (c, str(x)) in RBY
def needed(d):
    m = pd.Period(d, 'M'); ks = [1, 2, 3]
    if V.deadline_case(pd.Timestamp(d)) in ('D=S', 'D>S'): ks.append(4)
    return [m - k for k in ks]
ADJ = K.MYADJ if hasattr(K, 'MYADJ') else K.ADJ
def nxt_ret(c, d):
    i = sigs.index(d)
    if i + 1 >= len(sigs) or c not in ADJ.columns: return np.nan
    a = ADJ[c].loc[:pd.Timestamp(d)].dropna(); b = ADJ[c].loc[:pd.Timestamp(sigs[i + 1])].dropna()
    if not len(a) or not len(b): return np.nan
    return float(b.iloc[-1] / a.iloc[-1] - 1)
rows = []
for r in SL.itertuples():
    d, c = r.sig, r.code
    need = needed(d); miss = [str(x) for x in need if not has_yoy(c, x, d)]
    disp = sorted(set(C0[d]) - set(L[d]['picks']))
    rp = nxt_ret(c, d); rd = np.nanmean([abs(nxt_ret(x, d)) for x in disp]) if disp else 0.0
    rows.append(dict(sig=d, code=c, cause=r.cause, top10=r.top10, ky=c in KY, covered=(not miss) and c not in KY, n_missing=len(miss), missing=';'.join(miss),
                     w=(abs(rp) if np.isfinite(rp) else 0.0) + (rd if np.isfinite(rd) else 0.0)))
CV = pd.DataFrame(rows); CV.to_csv(HERE / f'data/coverage_slots_{BV}.csv', index=False)
def summ(x):
    return dict(n=int(len(x)), covered=int(x.covered.sum()), share=float(x.covered.mean()) if len(x) else None,
                wshare=float((x.w * x.covered).sum() / x.w.sum()) if x.w.sum() > 0 else None,
                ky=int(x.ky.sum()), share_incl_ky_resolved=float((x.covered | x.ky).mean()) if len(x) else None,
                wshare_incl_ky=float((x.w * (x.covered | x.ky)).sum() / x.w.sum()) if x.w.sum() > 0 else None)
out = dict(ver=BV, all_2013_2026=summ(CV), s2013_2018=summ(CV[CV.sig <= '2018-12-31']), s2019_2026=summ(CV[CV.sig >= '2019-01-01']),
           top10=summ(CV[CV.top10]), by_issuer={c: summ(g) for c, g in CV.groupby('code')},
           by_year={y: summ(g) for y, g in CV.groupby(CV.sig.str[:4])},
           note='權重只用絕對值；沒有計算任何變體報酬或差。')
P.jdump(out, HERE / f'coverage_{BV}.json')
print(json.dumps({k: v for k, v in out.items() if k not in ('by_issuer',)}, ensure_ascii=False, indent=0)[:3000])
print({c: (v['covered'], v['n']) for c, v in out['by_issuer'].items()})
