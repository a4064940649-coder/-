#!/usr/bin/env python3
"""建置 V5 營收資料（只看資料，不看報酬）：
  data/fv_replace.csv：首次值替換格（iterations.json 選定的 F3）：code, ym, first（新聞首次值，千元）, first_date（最早一則文章日＝可得時點）,
                       lv_L（長期面板最新值：tw_revenue sii > taiwanan > rev_5y）, lv_5（rev_5y 最新值）, ratio_L＝first/lv_L, ratio_5＝first/lv_5（差 < 0.5% 時為 1）
  data/fv_mc_s{s}.parquet（s＝0～19）：首次值界限用的擾動格（未被新聞覆蓋的水準格，以機率 p 抽中，ratio＝1＋ε，ε 自 F3 替換格的實際相對差抽樣）
  data/fv_mc_neg.parquet／fv_mc_pos.parquet：同 s＝0 的抽中格，但 ε 一律取 −|ε|／＋|ε|（固定方向）
  data/fv_mc_meta.json"""
import json, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
import numpy as np, pandas as pd
D = HERE / 'data'
it = json.load(open(HERE / 'iterations.json'))
assert it['first_value_selected'] == 'F3'
A = pd.read_csv(D / 'fv_cells.csv', dtype={'code': str}, parse_dates=['first_date'])
m = (A.rel_first.abs() >= 0.005) & (A.rel_first.abs() <= 0.5) & (A.consensus | A.internal_ok)
F = A[m].copy()
LONG = '/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv'
X5 = '/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras/rev_5y.csv.gz'
TW = HERE.parent / 'o_rev_survivorship' / 'src_taiwanan.csv'
r5 = pd.read_csv(X5, dtype={'code': str}); L5 = {(c, y): float(v) for c, y, v in zip(r5.code, r5.ym, r5.revenue)}
F['first'] = F.earliest; F['lv_L'] = F.lv
F['lv_5'] = [L5.get((c, y), np.nan) for c, y in zip(F.code, F.ym)]
F['ratio_L'] = F['first'] / F.lv_L
F['ratio_5'] = np.where(np.isfinite(F.lv_5) & ((F['first'] / F.lv_5 - 1).abs() >= 0.005), F['first'] / F.lv_5, 1.0)
F[['code', 'ym', 'first', 'first_date', 'lv_L', 'lv_5', 'ratio_L', 'ratio_5', 'n', 'consensus', 'internal_ok']].to_csv(D / 'fv_replace.csv', index=False)
# ---------------- 首次值界限（MC）----------------
T30 = pd.read_csv(D / 'top30_cells.csv', dtype={'code': str})
t30 = set(zip(T30[T30.role == 'cur'].code, T30[T30.role == 'cur'].ym))
A['t30'] = [(c, y) in t30 for c, y in zip(A.code, A.ym)]
k = int((m & A.t30).sum()); n = int(A.t30.sum())
z = 1.959964; ph = k / n
wil_hi = (ph + z * z / (2 * n) + z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))) / (1 + z * z / n)
eps = F.rel_first.to_numpy()
d = pd.read_csv(LONG, dtype={'code': str, 'market': str}); d = d[d.market == 'sii']; d['ym'] = d.month.str[:7]
T = pd.read_csv(TW, dtype={'code': str})
univ = set(zip(d.code, d.ym)) | set(zip(T.code, T.ym)) | set(L5)
covered = set(zip(A.code, A.ym))
U = sorted(x for x in univ if x not in covered and '2005-01' <= x[1] <= '2026-09' and len(x[0]) == 4)
U = pd.DataFrame(U, columns=['code', 'ym'])
meta = {'p_rule': '前 30 名本期格的已佐證不一致率（F3）之 Wilson 95% 上界', 'k_top30': k, 'n_top30': n, 'p_hat': ph, 'p_used': wil_hi,
        'eps_source': 'F3 替換格的 first/latest−1（帶正負號），有放回抽樣', 'eps': eps.tolist(), 'n_universe_cells': int(len(U)), 'seeds': list(range(20))}
for s in range(20):
    rng = np.random.default_rng(10_000 + s)
    hit = rng.random(len(U)) < wil_hi
    e = rng.choice(eps, size=int(hit.sum()), replace=True)
    out = U[hit].copy(); out['ratio'] = 1.0 + e
    out.to_parquet(D / f'fv_mc_s{s}.parquet', index=False)
    if s == 0:
        o2 = out.copy(); o2['ratio'] = 1.0 - np.abs(e); o2.to_parquet(D / 'fv_mc_neg.parquet', index=False)
        o3 = out.copy(); o3['ratio'] = 1.0 + np.abs(e); o3.to_parquet(D / 'fv_mc_pos.parquet', index=False)
    meta.setdefault('n_cells_by_seed', []).append(int(hit.sum()))
json.dump(meta, open(D / 'fv_mc_meta.json', 'w'), ensure_ascii=False, indent=1)
print(len(F), 'replace cells;', meta['p_hat'], meta['p_used'], len(U), meta['n_cells_by_seed'][:5])
print(F[['code', 'ym', 'first', 'lv_L', 'lv_5', 'ratio_L', 'ratio_5', 'first_date']].to_string())
