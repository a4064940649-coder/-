#!/usr/bin/env python3
"""步驟 2(b)(c)(e) 補回值表（不看任何報酬）。版本 B<k>：python3 -I -B s5_backfill.py <news_ver> <tw_ver> <B_ver>
來源與閘門（gate_<news>_<tw>.json 的同期存活公司重疊比對；不合格的「來源×欄位×年」整段剔除）：
  A taiwanan（MOPS t21sc03 爬檔）：rev（當月）、prev（上月）、ly（去年同月）三欄各自依年過閘（|相對差| ≤ 0.1% 比例 ≥ 98%）。
  N 新聞（CTBC＋BOF＋ELD，解析器 <news_ver>）：只收「共識值」＝同 (代碼, 月) ≥ 2 則不同文句的水準互相在 0.6% 內且沒有衝突值；
    共識水準依年過閘（|相對差| ≤ 0.6% 比例 ≥ 98%，n ≥ 30）；共識年增（≥ 1 位小數）依年過閘（|差| ≤ 0.1pp 比例 ≥ 98%，n ≥ 30）。
水準優先序：tw_revenue（原有）＞ A.rev ＞ A.prev（下個月列）＞ A.ly（12 個月後列，限過閘年）＞ N 共識水準。
月年增 yoy_x（%）：lv_x 與 lv_{x−12} 都有 → 水準法（與引擎同構）；否則 x−12 ≥ 2013-01 時用 N 共識年增；其他 → 缺。
（2013 年月份的年增一律只用水準法：引擎對存活者是「2013 合併／2012 tw_revenue」的混合口徑，新聞年增是同口徑，兩者不同。）
B1 規則如上；KY（名稱含 -KY 的 1262、1592、6422、6452）不補值：依原始檔名稱證據屬 FOREIGN，與 C0 股池（不含 KY）口徑一致。
輸出 data/rev_backfill_<B>.csv（只含 tw_revenue 缺值、需求發行人的格）與 data/backfill_<B>_meta.json。"""
import sys, json, hashlib
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
NV, TV, BV = (sys.argv[1:4] + ['P4', 'T1', 'B1'][len(sys.argv[1:4]):]) if len(sys.argv) < 4 else sys.argv[1:4]
sys.path.insert(0, str(HERE)); import bflib as B; sys.path.pop(0)
LV = B.tw_levels()
def lv(c, p):
    try: return LV.at[p, c]
    except KeyError: return np.nan
G = json.loads((HERE / f'gate_{NV}_{TV}.json').read_text(encoding='utf-8'))
DEM = pd.read_csv(HERE / 'data/demand_cells.csv', dtype={'code': str})
DJ = json.loads((HERE / 'demand.json').read_text(encoding='utf-8'))
ISS = DJ['issuers']
KY = {'1262', '1592', '6422', '6452'}
# ---------------- A
T = B.taiwanan(TV)
gA = {f: {y: (v['ok'] is not None and v['ok'] >= 0.98 and v['n'] >= 30) for y, v in G['tw'][f]['by_year'].items()} for f in ('rev', 'prev', 'ly')}
A = {}
for r in T[T.code.isin(ISS)].itertuples():
    p = pd.Period(r.ym, 'M')
    for fld, sh in (('rev', 0), ('prev', 1), ('ly', 12)):
        v = getattr(r, fld); q = p - sh
        if np.isfinite(v) and v > 0 and gA[fld].get(str(q.year), False):
            A.setdefault((r.code, q), []).append((fld, float(v)))
# ---------------- N 共識
N = pd.read_parquet(HERE / f'data/news_vals_{NV}.parquet')
N['k40'] = N.seg.str[:40]
def consensus(df, col, tol, absmode=False):
    out = {}
    for (c, y), g in df[np.isfinite(df[col])].groupby(['code', 'ym']):
        g = g.drop_duplicates('k40'); v = g[col].to_numpy()
        if len(v) < 2: continue
        med = float(np.median(v))
        agree = (np.abs(v - med) <= tol) if absmode else (np.abs(v / med - 1) <= tol)
        if agree.all(): out[(c, pd.Period(y, 'M'))] = (med, int(len(v)), str(g.date.min()))
    return out
CL = consensus(N, 'rev_k', 0.006); CY = consensus(N, 'yoy', 0.1, absmode=True)
# 共識值閘門（存活者）
def gate_cons(C, kind):
    rows = []
    for (c, p), (v, n, d0) in C.items():
        t = lv(c, p)
        if kind == 'yoy':
            if p < pd.Period('2014-01', 'M'): continue
            t0 = lv(c, p - 12); t = (t / t0 - 1) * 100 if np.isfinite(t) and np.isfinite(t0) and t0 > 0 else np.nan
            if np.isfinite(t): rows.append((str(p.year), abs(v - t) <= 0.1))
        elif np.isfinite(t) and t > 0: rows.append((str(p.year), abs(v / t - 1) <= 0.006))
    d = pd.DataFrame(rows, columns=['yr', 'ok'])
    return {y: dict(n=int(len(g)), ok=float(g.ok.mean())) for y, g in d.groupby('yr')}
GN = {'level': gate_cons(CL, 'level'), 'yoy': gate_cons(CY, 'yoy')}
okN = {k: {y: (v['ok'] >= 0.98 and v['n'] >= 30) for y, v in d.items()} for k, d in GN.items()}
# ---------------- 水準表（需求發行人、2011-01～2025-12）
P0, P1 = pd.Period('2011-01', 'M'), pd.Period('2025-12', 'M')
months = pd.period_range(P0, P1, freq='M')
LVB, SRC = {}, {}
for c in ISS:
    if c in KY: continue
    for p in months:
        t = lv(c, p)
        if np.isfinite(t): LVB[(c, p)] = float(t); SRC[(c, p)] = 'tw_revenue'; continue
        a = A.get((c, p))
        if a:
            pri = {'rev': 0, 'prev': 1, 'ly': 2}; a = sorted(a, key=lambda z: pri[z[0]])
            LVB[(c, p)] = a[0][1]; SRC[(c, p)] = 'A.' + a[0][0]; continue
        n = CL.get((c, p))
        if n and okN['level'].get(str(p.year), False):
            LVB[(c, p)] = n[0]; SRC[(c, p)] = 'N.level'
rows = []
for c in ISS:
    if c in KY: continue
    for p in months:
        if np.isfinite(lv(c, p)) and np.isfinite(lv(c, p - 12)): continue     # 引擎原本就算得出年增 → 不需要補
        L1, L0 = LVB.get((c, p)), LVB.get((c, p - 12))
        y, ysrc, meth = np.nan, None, None
        if L1 is not None and L0 is not None and L0 > 0:
            y = (L1 / L0 - 1) * 100; meth = 'level'; ysrc = f'{SRC[(c, p)]}/{SRC[(c, p - 12)]}'
        elif p - 12 >= pd.Period('2013-01', 'M') and (c, p) in CY and okN['yoy'].get(str(p.year), False):
            y = CY[(c, p)][0]; meth = 'news_yoy'; ysrc = 'N.yoy'
        if L1 is None and not np.isfinite(y): continue
        rows.append(dict(code=c, ym=str(p), rev_k=L1, rev_src=SRC.get((c, p)), yoy_pct=y, yoy_src=ysrc, yoy_method=meth,
                         news_first_date=(CL.get((c, p)) or CY.get((c, p)) or (None, None, None))[2]))
RB = pd.DataFrame(rows).sort_values(['code', 'ym'])
f = HERE / f'data/rev_backfill_{BV}.csv'; RB.to_csv(f, index=False)
# 需求格覆蓋（水準格；yoy 格）
DEM['p'] = pd.PeriodIndex(DEM.ym, freq='M')
DEM['lev_cov'] = [(c, p) in LVB for c, p in zip(DEM.code, DEM.p)]
meta = dict(ver=BV, news_ver=NV, tw_ver=TV, gate_A=gA, gate_N_consensus=GN, ok_N=okN, n_consensus_level=len(CL), n_consensus_yoy=len(CY),
            n_rows=len(RB), n_yoy=int(RB.yoy_pct.notna().sum()), yoy_by_method=RB.yoy_method.value_counts().to_dict(),
            rev_src=RB.rev_src.value_counts().to_dict(), demand_level_cov=float(DEM.lev_cov.mean()),
            demand_level_cov_by_year=DEM.groupby(DEM.ym.str[:4]).lev_cov.mean().round(3).to_dict(),
            ky_excluded=sorted(KY), sha256=hashlib.sha256(f.read_bytes()).hexdigest())
(HERE / f'data/backfill_{BV}_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(meta, ensure_ascii=False, indent=0)[:3500])
