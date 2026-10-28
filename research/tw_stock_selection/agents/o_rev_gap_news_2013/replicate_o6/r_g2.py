#!/usr/bin/env python3
"""獨立複製 o6：G2 首次值佐證（自寫）。
AB_V4（40 seeds、mc 公告延遲）2013～2021 前 10 名股月；每格取 y3 實際用到的三個營收月（未公告者退一個月）。
嚴格佐證：首次公告窗（營收月次月月底 23:59 前）內 ≥2 個不同來源（CTBC／BOF／ELD／PTT）的提及一致
          （水準相對差 ≤0.5%；沒有水準時年增率差 ≤0.1pp）。git 時間戳路徑＝0。
寬鬆佐證：≥2 則不同文章一致（CTBC 兩資料夾同文算同一篇）。"""
import sys, json, math
from pathlib import Path
from itertools import combinations
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
AG = HERE.parent.parent
sys.path.insert(0, str(AG / 'o_v4_revenue_patch_official'))
import v4lib as L
V = L.V

M = pd.read_parquet(HERE / 'r_mentions.parquet')
M['p'] = pd.PeriodIndex(M.rev_ym, freq='M')
M['win_end'] = (M.p + 2).dt.start_time - pd.Timedelta(minutes=1)
M = M[M.ts <= M.win_end].sort_values('ts')

def agree(a, b):
    if np.isfinite(a.level_k) and np.isfinite(b.level_k):
        return abs(a.level_k - b.level_k) / max(a.level_k, b.level_k) <= 0.005
    if np.isfinite(a.yoy) and np.isfinite(b.yoy):
        return abs(a.yoy - b.yoy) <= 0.1
    return False

strict, loose, first = set(), set(), {}
for (c, p), g in M.groupby(['code', 'p']):
    rs = list(g.itertuples())
    rs = [r for r in rs if np.isfinite(r.level_k) or np.isfinite(r.yoy)]
    st = lo = False
    for a, b in combinations(rs, 2):
        if not agree(a, b): continue
        if a.art_id != b.art_id: lo = True
        if a.src != b.src: st = True
        if st and lo: break
    if st: strict.add((c, str(p)))
    if lo: loose.add((c, str(p)))
    if rs: first[(c, str(p))] = rs[0]

# 最新官方值
lv = pd.read_csv('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv', dtype={'code': str})
lv = lv.sort_values('market', key=lambda s: s != 'sii').drop_duplicates(['code', 'month'])
LV = {(c, m[:7]): v for c, m, v in zip(lv.code, lv.month, lv.revenue_k_twd) if np.isfinite(v) and v > 0}
r5 = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras/rev_5y.csv.gz', dtype={'code': str})
for c, y, v in zip(r5.code, r5.ym, r5.revenue):
    if (c, y) not in LV and np.isfinite(v) and v > 0: LV[(c, y)] = v
YFX, Y5F = L.load_rfix()
def fl(cells):
    n = dis = 0
    for k in cells:
        r = first[k]
        if np.isfinite(r.level_k) and k in LV:
            n += 1; dis += abs(r.level_k - LV[k]) / LV[k] > 0.005
        elif np.isfinite(r.yoy):
            p = pd.Period(k[1], 'M')
            o = YFX.at[p, k[0]] if (p in YFX.index and k[0] in YFX.columns) else np.nan
            if np.isfinite(o): n += 1; dis += abs(r.yoy - o) > 0.1
    return n, dis
def wilson(k, n, z=1.96):
    if n == 0: return [None, None]
    ph = k / n; den = 1 + z * z / n; ce = (ph + z * z / (2 * n)) / den; hw = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return [ce - hw, ce + hw]

XR = L.setup_px()
DMT = L.disp_mask_fn('trade')
rows = []
for seed in range(40):
    S = L.V4(YFX, Y5F, DMT, unavail=V.make_unavail('mc', seed))
    for e, picks, elidx, d in S.plan(10, 'official'):
        if not (pd.Timestamp('2013-01-01') <= d <= pd.Timestamp('2021-12-31')): continue
        U = S.unavail(d); m = pd.Period(d, 'M')
        for j in picks:
            b = 1 if U[j] else 0
            ms = [str(m - 1 - b - i) for i in range(3)]
            c = L.COLS[j]
            rows.append(dict(seed=seed, sig=d.strftime('%Y-%m-%d'), code=c,
                             ns=sum((c, x) in strict for x in ms), nl=sum((c, x) in loose for x in ms),
                             s_latest=(c, ms[0]) in strict, l_latest=(c, ms[0]) in loose))
T = pd.DataFrame(rows)
T.to_parquet(HERE / 'r_g2_top10.parquet', index=False)
n = len(T); k = int((T.ns == 3).sum())
ns, ds = fl(strict); nl, dl = fl(loose)
out = dict(n_top10_stock_month_seed=n, n_distinct_stock_months=int(T[['sig', 'code']].drop_duplicates().shape[0]),
           strict_all3=k / n, strict_all3_wilson=wilson(k, n), strict_latest=float(T.s_latest.mean()),
           loose_all3=float((T.nl == 3).mean()), loose_latest=float(T.l_latest.mean()),
           n_cells_strict=len(strict), n_cells_loose=len(loose),
           first_vs_latest_strict=[int(ds), int(ns), ds / max(1, ns), wilson(ds, ns)], first_vs_latest_loose=[int(dl), int(nl), dl / max(1, nl)],
           G2_pass=k / n >= 0.20)
print(json.dumps(out, ensure_ascii=False, indent=1))
json.dump(out, open(HERE / 'r_g2.json', 'w'), ensure_ascii=False, indent=1)
