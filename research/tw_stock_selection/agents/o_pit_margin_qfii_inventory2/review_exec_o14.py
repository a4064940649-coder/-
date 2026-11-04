#!/usr/bin/env python3
"""反方審查 o14（成本與可成交／資料時點角度）：獨立重算 MARGN V1 覆蓋（下市股）、
QFIIS 月形成日規則、官方回溯值與 TEJ 不一致的分布。只讀研究員檔案，不寫任何檔。
執行：python3 -I -B review_exec_o14.py"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import glib

L, V, FD, LD = glib.denominator()
NEW = glib.new_listing_mask(L, FD, 130)
den = L & ~NEW
DEL = (LD < pd.Timestamp('2026-10-01')).reindex(L.columns).fillna(False)
out = {}

# ---------- 1. 獨立建 V1：只用原始餘額 ----------
core = pd.DataFrame(np.nan, index=L.index, columns=L.columns)
y = glib.load_yh()
print('TEJ cols', list(y.columns)[:12])
balcol = [c for c in y.columns if '餘額' in c and '融資' in c] or [c for c in y.columns if 'bal' in c.lower()]
print('TEJ balance col guess', balcol)
bc = balcol[0]
yw = y.pivot_table(index='date', columns='code', values=bc, aggfunc='first')
yw = yw.reindex(index=L.index, columns=L.columns)
src = pd.DataFrame(0, index=L.index, columns=L.columns)
core = core.fillna(yw); src[yw.notna()] = 4
b = glib.load_bb()
bcol = [c for c in b.columns if 'Margin' in c and ('Balance' in c or 'Today' in c)]
print('BB cols', list(b.columns)[:20])
if bcol:
    bw = b.pivot_table(index='date', columns='code', values=bcol[0], aggfunc='first').reindex(index=L.index, columns=L.columns)
    core = core.fillna(bw)
jm = glib.load_joeyc('margin-twse')
jw = jm.pivot_table(index='date', columns='code', values='margin_lots', aggfunc='first').reindex(index=L.index, columns=L.columns)
core = core.fillna(jw)
hit = core.notna() & den
Dm = pd.DataFrame(np.broadcast_to(DEL.values, L.shape), index=L.index, columns=L.columns)
seg = {}
for s, (a, bb) in glib.SEG.items():
    m = (L.index >= a) & (L.index <= bb)
    d_all = den.values[m]; h = hit.values[m]; dd = Dm.values[m]
    seg[s] = {'cov_full': round(h.sum() / d_all.sum(), 4),
              'cov_delisted': round((h & dd).sum() / (d_all & dd).sum(), 4)}
out['V1_recalc_without_margin_hist'] = seg
print('V1 recompute', seg)

# ---------- 2. S3「下市股」覆蓋的組成：TEJ 存活清單（2024-04）裡其實是之後下市的股票 ----------
tej_codes = set(y.code.astype(str).unique())
del_codes = DEL[DEL].index
del_in_tej = [c for c in del_codes if c in tej_codes]
ld_in_tej = LD.reindex(del_in_tej)
out['delisted_codes_total'] = int(len(del_codes))
out['delisted_codes_in_TEJ_survivor_list'] = int(len(del_in_tej))
out['their_last_day_min_max'] = [str(ld_in_tej.min()), str(ld_in_tej.max())]
out['their_last_day_before_2024_04'] = int((ld_in_tej < pd.Timestamp('2024-04-01')).sum())
# 真正「2024-04 前就下市」的股票的覆蓋
early = (LD < pd.Timestamp('2024-04-01')).reindex(L.columns).fillna(False)
De = pd.DataFrame(np.broadcast_to(early.values, L.shape), index=L.index, columns=L.columns)
for s, (a, bb) in glib.SEG.items():
    m = (L.index >= a) & (L.index <= bb)
    d_all = den.values[m]; h = hit.values[m]; dd = De.values[m]
    out.setdefault('cov_delisted_before_2024_04', {})[s] = round((h & dd).sum() / max((d_all & dd).sum(), 1), 4)
print('delisted-before-2024-04 coverage', out['cov_delisted_before_2024_04'])

# ---------- 3. QFIIS 形成日規則（11 日起第一個交易日）是否全部成立 ----------
jq = glib.load_joeyc('qfii-twse')
fds = sorted(set(jq.date))
bad = []
for d in fds:
    mdays = L.index[(L.index.to_period('M') == d.to_period('M')) & (L.index.day >= 11)]
    if len(mdays) == 0 or mdays[0] != d:
        bad.append(str(d.date()))
out['qfii_formation_dates'] = len(fds)
out['qfii_formation_rule_violations'] = bad
# 形成日的 QFIIS 檔數 vs 分母（含下市）
nq = jq.groupby('date').code.nunique()
out['qfii_rows_per_date_min_median_max'] = [int(nq.min()), int(nq.median()), int(nq.max())]

# ---------- 4. 官方回溯值 vs TEJ 不一致：集中在哪些日期 ----------
yj = y[['date', 'code', bc]].merge(jm[['date', 'code', 'margin_lots']], on=['date', 'code'])
yj['same'] = yj[bc].astype(float).round(0) == yj['margin_lots'].astype(float).round(0)
by = yj.groupby('date')['same'].mean()
out['joeyc_vs_TEJ_eq_rate'] = round(float(yj['same'].mean()), 4)
out['joeyc_vs_TEJ_worst_dates'] = {str(k.date()): round(float(v), 3) for k, v in by.nsmallest(6).items()}
out['joeyc_vs_TEJ_dates_below_99pct'] = int((by < 0.99).sum())
out['joeyc_vs_TEJ_n_dates'] = int(len(by))
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))

# ---------- 5. 分母本身的下市股數量是否偏少（分母存活者偏誤會讓覆蓋率被高估）----------
lt = pd.read_parquet(glib.AG / 'o_v7_listing_exec_engine/data/listed_twse.parquet')
lt = lt[glib.common(lt.columns)]; lt.index = pd.to_datetime(lt.index)
last_lt = lt[::-1].idxmax(); first_lt = lt.idxmax()
gone = last_lt[(lt.any()) & (last_lt < lt.index.max())]
print('listed_twse span', lt.index.min().date(), lt.index.max().date(), 'codes', int(lt.any().sum()), 'disappear before end', int(len(gone)))
print('disappear by year', gone.dt.year.value_counts().sort_index().to_dict())
ld = LD[L.any()]
print('denominator delisted (last<2026-10-01) by year', ld[ld < pd.Timestamp('2026-10-01')].dt.year.value_counts().sort_index().to_dict())
