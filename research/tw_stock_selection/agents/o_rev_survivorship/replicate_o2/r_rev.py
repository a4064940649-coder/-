#!/usr/bin/env python3
"""複製員：建營收版本 ORIG／R_DEL／R_ALL（年增率寬表，index=Period M，columns=code），並算重疊期一致率。
輸出 my_yoy_{ORIG,R_DEL,R_ALL}.parquet、my_rev_meta.json。"""
import json
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent
LONG = Path('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv')
P = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
SLF = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv')
PCOLS = list(pd.read_parquet(P / 'close_raw.parquet').columns)

# ---- 長期營收（ORIG，和 main_v2 同一讀法：同代碼同月重複取 sii） ----
d = pd.read_csv(LONG, dtype={'code': str, 'market': str})
d['p'] = pd.PeriodIndex(pd.to_datetime(d['month']), freq='M')
d['pri'] = (d['market'] != 'sii').astype(int)
d = d.sort_values(['p', 'code', 'pri']).drop_duplicates(['p', 'code'])
lv = d.pivot(index='p', columns='code', values='revenue_k_twd')
lv = lv.reindex(pd.period_range(lv.index.min(), lv.index.max(), freq='M'))
YL = ((lv / lv.shift(12) - 1.0) * 100.0).where(lv.shift(12) > 0)

# ---- 來源 A ----
A = pd.read_csv(H / 'my_src.csv', dtype={'code': str})
A['p'] = pd.PeriodIndex(A['p'], freq='M')
la = A.pivot(index='p', columns='code', values='rev')
la = la.reindex(pd.period_range('2014-01', la.index.max(), freq='M'))
ly = A.pivot(index='p', columns='code', values='rev_ly').reindex(la.index)
ym = A.pivot(index='p', columns='code', values='yoy_mops').reindex(la.index)
base_self = la.shift(12)
V3 = ((la / base_self - 1) * 100).where(base_self > 0)
V2 = ((la / ly - 1) * 100).where(ly > 0)
V1 = ym
V4 = V3.where(base_self.notna(), V2)          # 來源內沒有 12 個月前 → 用 MOPS 當期公告的去年同月

# ---- 國內／KY 分類 ----
names = {}
for c, n in A.groupby('code')['name'].last().items(): names.setdefault(c, set()).add(str(n))
for c, n in d.groupby('code')['name'].last().items(): names.setdefault(c, set()).add(str(n))
SL = pd.read_csv(SLF, dtype=str)
for c, n in zip(SL['code'], SL['name']): names.setdefault(c, set()).add(str(n))
def foreign(c):
    nm = ' '.join(names.get(c, []))
    return ('KY' in nm.upper()) or ('DR' in nm.upper()) or c.startswith('91')
FOR = {c for c in names if foreign(c)}

# ---- 一致率（重疊期：兩邊都有值） ----
def agree(a, b, tol):
    a2, b2 = a.align(b, join='inner')
    m = a2.notna() & b2.notna()
    x, y = a2.to_numpy()[m.to_numpy()], b2.to_numpy()[m.to_numpy()]
    return int(len(x)), {str(t): float((np.abs(x - y) <= t * (np.abs(y) if rel else 1)).mean()) for t in tol for rel in [REL]}
meta = {'n_foreign_named': len(FOR)}
REL = True
n, r = agree(la.loc['2015-01':], lv, [0.0, 0.001, 0.005, 0.01])
meta['level_agree'] = {'n': n, 'rel_tol': r}
mpan = lambda X: X.reindex(columns=[c for c in X.columns if c in set(PCOLS)])
n, r = agree(mpan(la.loc['2015-01':]), mpan(lv), [0.0, 0.001, 0.005, 0.01]); meta['level_agree_panel'] = {'n': n, 'rel_tol': r}
REL = False
for k, Vk in (('V1', V1), ('V2', V2), ('V3', V3), ('V4', V4)):
    n, r = agree(Vk.loc['2015-01':], YL, [0.01, 0.05, 0.1, 0.5, 1.0])
    meta[f'yoy_agree_{k}'] = {'n': n, 'abs_pp_tol': r}
    n, r = agree(mpan(Vk.loc['2015-01':]), mpan(YL), [0.01, 0.05, 0.1, 0.5, 1.0]); meta[f'yoy_agree_{k}_panel'] = {'n': n, 'abs_pp_tol': r}

# ---- 修補：只補 NaN ----
def patch(Y, src, allow):
    Y = Y.copy()
    cols = [c for c in src.columns if c in allow]
    Y = Y.reindex(columns=Y.columns.union(pd.Index(cols)))
    S = src.loc['2015-01':, cols].reindex(index=Y.index)
    blk = Y[cols]
    fill = blk.isna() & S.notna()
    Y[cols] = blk.where(~fill, S)
    return Y, fill
out = {'ORIG': YL}
dom = {c for c in V4.columns if c not in FOR}
Rd, fd = patch(YL, V4, dom); out['R_DEL'] = Rd
Ra, fa = patch(YL, V4, set(V4.columns)); out['R_ALL'] = Ra
pc = set(PCOLS)
meta['filled'] = {'R_DEL': int(fd.values.sum()), 'R_ALL': int(fa.values.sum()),
                  'R_DEL_panel': int(fd[[c for c in fd.columns if c in pc]].values.sum()),
                  'R_ALL_panel': int(fa[[c for c in fa.columns if c in pc]].values.sum()),
                  'R_DEL_panel_codes': int((fd[[c for c in fd.columns if c in pc]].sum() > 0).sum()),
                  'R_ALL_panel_codes': int((fa[[c for c in fa.columns if c in pc]].sum() > 0).sum())}
for k, Y in out.items(): Y.columns = Y.columns.astype(str); Y.index = Y.index.astype(str); Y.to_parquet(H / f'my_yoy_{k}.parquet')
(H / 'my_foreign.json').write_text(json.dumps(sorted(FOR)), encoding='utf-8')
(H / 'my_rev_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(meta, ensure_ascii=False, indent=1))
