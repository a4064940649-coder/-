#!/usr/bin/env python3
"""反方審查（資料與統計）o14：只讀研究員檔案，不改動。python3 -I -B 執行。
檢查：
 A. MARGN V1 覆蓋（全市場／下市／存活）從 raw 重算
 B. 2014 起分母：joeyc MI_INDEX vs listed_twse（GATE_SPEC 要求交叉並報差異，研究員未報）
 C. QFIIS 月快照在獨立分母（listed_twse）下的覆蓋，檢查「100%」是否為同源分母的循環
 D. 官方月快照 vs TEJ 不一致的日期集中度（事後更正 vs 日期錯置）
 E. 2014～2020 QFIIS 月快照「乾淨 PIT」主張：有無任何 2025 年前的獨立同期來源可核對
"""
import sys, json, glob
from pathlib import Path
HERE = Path('/home/user/-/research/tw_stock_selection/agents/o_pit_margin_qfii_inventory2')
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import glib

out = {}
L, V, FD, LD = glib.denominator()
NEW = glib.new_listing_mask(L, FD, 130)
DEL = LD < pd.Timestamp('2026-10-01')
days, codes = L.index, L.columns

def wide(df, val):
    return df.pivot_table(index='date', columns='code', values=val, aggfunc='last').reindex(index=days, columns=codes)

# ---- A. V1 重算 ----
yh = glib.load_yh(); bb = glib.load_bb(); mh = glib.load_mh(); jm = glib.load_joeyc('margin-twse')
jm['code'] = jm.code.astype(str).str.zfill(4)
B = wide(yh, 'mbal'); C = wide(bb, 'Margin_Balance'); E = wide(mh, 'margin_bal'); D = wide(jm, 'margin_lots')
V1 = D.combine_first(C).combine_first(E).combine_first(B)
den = (L & ~NEW).values
hit = V1.notna().values & den
dl = np.broadcast_to(DEL.reindex(codes).values, L.shape)
A = {}
for s, (a, b) in glib.SEG.items():
    m = (days >= a) & (days <= b)
    A[s] = {'full': round(hit[m].sum() / den[m].sum(), 4),
            'delisted': round((hit & dl)[m].sum() / (den & dl)[m].sum(), 4),
            'survivors': round((hit & ~dl)[m].sum() / (den & ~dl)[m].sum(), 4),
            'del_codes': int(((den & dl)[m].sum(axis=0) > 0).sum()),
            'surv_codes': int(((den & ~dl)[m].sum(axis=0) > 0).sum())}
m = (days >= '2008-09-01') & (days <= '2009-03-31')
A['crisis_2008_09_2009_03_full'] = round(hit[m].sum() / den[m].sum(), 4)
# 只有 TEJ（存活股快照 2024-04）在 S1/S2 是唯一來源：「存活股」覆蓋 96.7% 本身就是存活者定義的同義反覆
tej_codes = set(yh.code.unique())
A['tej_codes'] = len(tej_codes)
A['S1_survivors_not_in_TEJ'] = int(sum(1 for c in codes if c not in tej_codes and ((den & ~dl)[(days >= '2006-01-02') & (days <= '2012-12-31')][:, codes.get_loc(c)].sum() > 0)))
out['A_V1_recompute'] = A

# ---- B. 2014 起分母交叉 ----
lt = pd.read_parquet(HERE.parent / 'o_v7_listing_exec_engine/data/listed_twse.parquet')
lt = lt[glib.common(lt.columns)]; lt.index = pd.to_datetime(lt.index).astype('datetime64[ns]')
ov = days[(days >= '2014-01-02') & (days <= lt.index.max())].intersection(lt.index)
cc = sorted(set(codes) | set(lt.columns))
Lj = L.reindex(index=ov, columns=cc).fillna(False).astype(bool).values
Lt = lt.reindex(index=ov, columns=cc).fillna(False).astype(bool).values
out['B_denominator_2014plus'] = {
    'days': len(ov), 'joeyc_stockdays': int(Lj.sum()), 'listed_twse_stockdays': int(Lt.sum()),
    'in_listed_twse_not_joeyc': int((Lt & ~Lj).sum()), 'in_joeyc_not_listed_twse': int((Lj & ~Lt).sum()),
    'codes_only_listed_twse': int(((Lt & ~Lj).sum(axis=0) > 0).sum()),
    'codes_only_joeyc': int(((Lj & ~Lt).sum(axis=0) > 0).sum()),
    'note': 'GATE_SPEC denominator_sources 要求「與 listed_twse 交叉並報差異」；gate.json／FINDINGS 都沒有這個數字'}
# 只在 listed_twse 的股日，是否集中在下市前（停牌期）？
diffcols = [cc[i] for i in np.where((Lt & ~Lj).sum(axis=0) > 0)[0]]
out['B_denominator_2014plus']['sample_codes_only_listed_twse'] = diffcols[:15]

# ---- C. QFIIS 月快照：獨立分母 ----
jq = glib.load_joeyc('qfii-twse'); jq['code'] = jq.code.astype(str)
fdates = sorted(set(jq.date))
rows = []
for d in fdates:
    if d not in lt.index: continue
    den_t = lt.loc[d]; den_t = den_t[den_t.astype(bool)].index
    have = set(jq.loc[jq.date == d, 'code'])
    den_j = L.loc[d] if d in L.index else None
    den_j = set(den_j[den_j].index) if den_j is not None else set()
    miss_t = [c for c in den_t if c not in have]
    miss_j = [c for c in den_j if c not in have]
    rows.append((d, len(den_t), len(miss_t), len(den_j), len(miss_j), sum(1 for c in miss_t if DEL.get(c, False))))
cm = pd.DataFrame(rows, columns=['d', 'den_lt', 'miss_lt', 'den_joeyc', 'miss_joeyc', 'miss_lt_delisted']).set_index('d')
def ccov(x):
    return {'cov_vs_listed_twse': round(1 - x.miss_lt.sum() / x.den_lt.sum(), 5), 'cov_vs_joeyc_MI_INDEX': round(1 - x.miss_joeyc.sum() / x.den_joeyc.sum(), 5),
            'missing_stockmonths_vs_listed_twse': int(x.miss_lt.sum()), 'of_which_delisted': int(x.miss_lt_delisted.sum()), 'n_months': len(x)}
out['C_qfiis_monthly_independent_den'] = {'2014_2020': ccov(cm[cm.index <= '2020-12-31']), 'all': ccov(cm)}

# ---- D. 官方月快照 vs TEJ 不一致的日期分布 ----
Dm = D.loc[D.notna().any(axis=1)]
Bm = B.reindex(Dm.index)
mk = Dm.notna() & Bm.notna()
neq = (Dm != Bm) & mk
per_date = pd.DataFrame({'n': mk.sum(axis=1), 'neq': neq.sum(axis=1)})
per_date = per_date[per_date.n > 0]
per_date['rate'] = per_date.neq / per_date.n
top = per_date.sort_values('neq', ascending=False).head(8)
ratio = (Dm[neq] / Bm[neq]).stack()
out['D_official_vs_TEJ'] = {'overall_eq': round(1 - per_date.neq.sum() / per_date.n.sum(), 4),
    'n_dates': len(per_date), 'dates_with_neq_gt_10pct': int((per_date.rate > 0.10).sum()),
    'share_of_neq_in_top3_dates': round(top.neq.head(3).sum() / per_date.neq.sum(), 4),
    'top_dates': {str(k.date()): [int(v.n), int(v.neq)] for k, v in top.iterrows()},
    'eq_rate_excl_top3_dates': round(1 - (per_date.neq.sum() - top.neq.head(3).sum()) / (per_date.n.sum() - top.n.head(3).sum()), 4),
    'neq_ratio_median': float(ratio.median()), 'neq_absdiff_lots_median': float((Dm[neq] - Bm[neq]).abs().stack().median())}
# 同一天 TEJ 的「前一交易日」值是否等於官方（日期錯置測試）
Bprev = B.shift(1).reindex(Dm.index)
mk2 = neq & Bprev.notna()
out['D_official_vs_TEJ']['neq_cells_equal_TEJ_prev_day'] = round(float(((Dm == Bprev) & mk2).values.sum() / max(mk2.values.sum(), 1)), 4)
Bnext = B.shift(-1).reindex(Dm.index)
mk3 = neq & Bnext.notna()
out['D_official_vs_TEJ']['neq_cells_equal_TEJ_next_day'] = round(float(((Dm == Bnext) & mk3).values.sum() / max(mk3.values.sum(), 1)), 4)

# ---- E. QFIIS 2014～2020 的獨立同期核對 ----
vf = glib.load_vf()
out['E_qfiis_pre2025_crosscheck'] = {'voidful_first_date': str(vf.date.min().date()),
    'joeyc_qfii_first': str(min(fdates).date()), 'joeyc_commit_time': json.load(open(HERE / 'raw/manifest.json'))['sources']['joeycchen72/taiwan-market-history']['commit_time'],
    'independent_contemporaneous_source_2014_2020': 0}

print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
