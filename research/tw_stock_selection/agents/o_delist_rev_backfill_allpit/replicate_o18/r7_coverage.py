#!/usr/bin/env python3
"""G6 複製員 r7：覆蓋率（凍結規格一.6）：U_opt 放回入選槽（r1）中，m−1～m−3 年增都有值（D=S／D>S 日加 m−4）者算已覆蓋；
年增＝tw_revenue／rev_5y 原值 ∪ 本複製員補回表。報酬權重＝|放回股下一期報酬|＋|被擠掉 C0 股下一期報酬平均|（訊號日收盤→下一訊號日收盤，ia 還原價）。
python3 -I -B r7_coverage.py"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
AG = HERE.parent.parent
SCR = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/rep_o18')
sys.path.insert(0, str(AG / 'o_rev_avail_ds_pit_count')); import pclib as P; sys.path.pop(0)
V, K = P.V, P.K
BF = pd.read_csv(HERE / 'my_backfill.csv', dtype={'code': str, 'ym': str}); BF = BF[BF.yoy_pct.notna()]
HAVE = set(zip(BF.code, BF.ym))
def has(Y, c, p):
    return (p in Y.index and c in Y.columns and np.isfinite(Y.at[p, c])) or (c, str(p)) in HAVE
SL = pd.read_csv(SCR / 'r1_slots.csv', dtype={'code': str, 'sig': str})
C0 = V.V2('gapfix', P.YL, P.Y5, exfin=True); pc = {p[3]: [P.COLS[j] for j in p[1]] for p in C0.plan(10, 'official')}
U = P.VU('opt', lo='2013-01-01'); pu = {p[3]: [P.COLS[j] for j in p[1]] for p in U.plan(10, 'official')}
SD = sorted(pc); nxt = {a: b for a, b in zip(SD[:-1], SD[1:])}
A = K.MYADJ
def r(c, a, b):
    try: x, y = A.at[a, c], A[c].loc[:b].dropna().iloc[-1]
    except Exception: return np.nan
    return y / x - 1 if np.isfinite(x) else np.nan
KY = {'1262', '1592', '6422', '6452'}
rows = []
for s, c in zip(SL.sig, SL.code):
    d = pd.Timestamp(s); m = pd.Period(d, 'M'); Y = P.Y5 if d >= V.SWITCH else P.YL
    ks = [1, 2, 3] + ([4] if V.deadline_case(d) in ('D=S', 'D>S') else [])
    cov = all(has(Y, c, m - k) for k in ks) and c not in KY
    b = nxt.get(d, V.END1012)
    out = [x for x in pc[d] if x not in pu[d]]
    w = abs(r(c, d, b)) + abs(np.nanmean([r(x, d, b) for x in out]) if out else 0.0)
    rows.append(dict(sig=s, code=c, covered=cov, ky=c in KY, w=w if np.isfinite(w) else 0.0))
CV = pd.DataFrame(rows)
def summ(x): return dict(n=len(x), covered=int(x.covered.sum()), share=float(x.covered.mean()), wshare=float((x.w * x.covered).sum() / x.w.sum()),
                         share_incl_ky=float((x.covered | x.ky).mean()), wshare_incl_ky=float((x.w * (x.covered | x.ky)).sum() / x.w.sum()))
top10 = list(SL.code.value_counts().index[:10])
out = dict(all=summ(CV), y1318=summ(CV[CV.sig <= '2018-12-31']), y19on=summ(CV[CV.sig >= '2019-01-01']), top10=summ(CV[CV.code.isin(top10)]),
           by_year={y: dict(n=len(g), covered=int(g.covered.sum()), share=float(g.covered.mean())) for y, g in CV.groupby(CV.sig.str[:4])},
           top10_issuer={c: f'{int(CV[CV.code == c].covered.sum())}/{int((CV.code == c).sum())}' for c in top10})
(HERE / 'r7_coverage.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(out, ensure_ascii=False, indent=1))
