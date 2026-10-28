#!/usr/bin/env python3
"""複製面板 vs 研究員 V6 面板：逐格比對 adj、open、close_raw（以及 ret、value、high、low 參考）。
比對範圍：2024-01-02 起所有「任一邊有值」的格；一致＝兩邊都有值且相對差 ≤ 1e-9（adj 也報「日比值」一致率，避免一格 ret 差異污染之後所有 adj）。"""
import json
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent; TOP = H.parent
out = {}
for k in ('adj', 'open', 'close_raw', 'high', 'low', 'value', 'ret', 'flag'):
    a = pd.read_parquet(H / 'data' / f'{k}.parquet'); b = pd.read_parquet(TOP / 'data' / f'{k}.parquet')
    assert a.index.equals(b.index) and a.columns.equals(b.columns)
    a, b = a.loc['2024-01-02':].to_numpy(), b.loc['2024-01-02':].to_numpy()
    anyv = np.isfinite(a) | np.isfinite(b)
    both = np.isfinite(a) & np.isfinite(b)
    eq = both & (np.abs(a - b) <= 1e-9 * np.maximum(np.abs(b), 1e-12) + (1e-12 if k in ('ret',) else 0))
    o = {'n_cells_any': int(anyv.sum()), 'agree': float(eq.sum() / anyv.sum()), 'n_disagree': int(anyv.sum() - eq.sum()),
         'n_only_rep': int((np.isfinite(a) & ~np.isfinite(b)).sum()), 'n_only_mine': int((~np.isfinite(a) & np.isfinite(b)).sum())}
    if k == 'adj':
        ra = a[1:] / a[:-1]; rb = b[1:] / b[:-1]; m = np.isfinite(ra) | np.isfinite(rb)
        e2 = np.isfinite(ra) & np.isfinite(rb) & (np.abs(ra - rb) <= 1e-9)
        o['daily_ratio_agree'] = float(e2.sum() / m.sum())
    out[k] = o
idx = pd.read_parquet(TOP / 'data' / 'ret.parquet').loc['2024-01-02':]
a = pd.read_parquet(H / 'data' / 'ret.parquet').loc['2024-01-02':]
d = ((a - idx).abs() > 1e-12) | (a.isna() != idx.isna())
dd = d.stack(); dd = dd[dd]
out['ret_disagree_examples'] = [(str(t.date()), c, None if pd.isna(a.at[t, c]) else float(a.at[t, c]), None if pd.isna(idx.at[t, c]) else float(idx.at[t, c])) for t, c in dd.index[:30]]
out['ret_disagree_by_date'] = {str(k.date()): int(v) for k, v in dd.groupby(level=0).size().sort_values(ascending=False).head(10).items()}
(H / 'rep_compare.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps({k: v for k, v in out.items() if not k.startswith('ret_dis')}, ensure_ascii=False, indent=0)); print(out['ret_disagree_by_date']); print(out['ret_disagree_examples'][:12])
