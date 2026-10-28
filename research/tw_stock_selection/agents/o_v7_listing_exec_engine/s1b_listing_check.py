#!/usr/bin/env python3
"""步驟 1 檢查（只看資料）：三方交叉核對一致率、母體檔數對照官方家數、16 檔空洞清點、欄位外上市股、上市但無價資料的股日 → explore.json 的 LISTING 段。"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import v7common as C, srcload7 as S
LM = pd.read_parquet(C.DATA / 'listed_twse.parquet')
M = pd.read_csv(C.HERE / 'listing_master.csv', dtype=str)
cl6 = pd.read_parquet(C.V6DATA / 'close_raw.parquet'); src6 = pd.read_parquet(C.V6DATA / 'src.parquet')
COLS = list(cl6.columns)
out = {}
# ---- 三方交叉：每個 code-month（次月 10 日的狀態）----
rev = pd.read_csv(C.REV, dtype=str); rev['month'] = pd.to_datetime(rev.month)
trv = S.get('tasi_rev'); trv = trv[trv.pit]
bac = pd.concat([S.get('bacardi').assign(m='TWSE'), S.get('bacardi_tpex').assign(m='TPEX')])
def lm_at(codes, dates):
    idx = LM.index
    pos = np.searchsorted(idx.values, np.array(dates, dtype='datetime64[ns]'), side='right') - 1
    out = []
    for c, p in zip(codes, pos):
        out.append(bool(LM[c].iloc[p]) if (c in LM.columns and p >= 0) else False)
    return np.array(out)
def is_common(c): return isinstance(c, str) and len(c) == 4 and c.isdigit() and c[0] != '0'
# (1) tw_revenue（2013 起實為目前市場別）
r1 = rev[(rev.month >= '2021-01-01') & rev.code.map(is_common)].copy()
r1['d'] = r1.month + pd.offsets.MonthBegin(1) + pd.Timedelta(days=9)
r1['lm'] = lm_at(r1.code.tolist(), r1.d.tolist())
# (2) Tasi PIT 月檔
r2 = trv[trv.code.map(is_common)].copy(); r2['lm'] = lm_at(r2.code.tolist(), r2.added.tolist())
# (3) bacardi（逐日，月中第 10 個交易日取樣）
b = bac[bac.code.map(is_common)].copy(); b['ym'] = b.date.dt.to_period('M')
b = b.sort_values('date').groupby(['code', 'ym']).nth(9).reset_index(drop=True) if False else b.groupby(['code', 'ym']).head(1)
b['lm'] = lm_at(b.code.tolist(), b.date.tolist())
def agree(df, col, tw):
    x = df[col] == tw
    return {'n': int(len(df)), 'agree': float((x == df.lm).mean()), 'src_TW_lm_not': int((x & ~df.lm).sum()), 'src_OT_lm_TW': int((~x & df.lm).sum())}
out['three_way'] = {'tw_revenue_2021on(目前市場別)': agree(r1, 'market', 'sii'), 'tasi_rev_PIT_2024_04on': agree(r2, 'mkt', 'TWSE'),
                    'bacardi_2025_04on': agree(b, 'm', 'TWSE')}
# 不一致明細（Tasi PIT 與 bacardi）
dis2 = r2[(r2.mkt == 'TWSE') != r2.lm]; dis3 = b[(b.m == 'TWSE') != b.lm]
out['three_way_disagree_tasi_rev'] = dis2.groupby('code').agg(n=('ym', 'size'), first=('ym', 'min'), last=('ym', 'max'), mkt=('mkt', 'first'), name=('name', 'first')).reset_index().to_dict('records')
out['three_way_disagree_bacardi'] = dis3.groupby('code').agg(n=('ym', 'size'), first=('date', 'min'), last=('date', 'max'), m=('m', 'first')).reset_index().astype(str).to_dict('records')
dis1 = r1[(r1.market == 'sii') != r1.lm]
out['tw_revenue_disagree_codes'] = int(dis1.code.nunique())
# ---- 母體檔數 vs 官方 ----
I = S.get('isin')
cnt = {}
for v, g in I[I.file.str.startswith('isin_02')].groupby('version'):
    gg = g[g.market.str.startswith('上市') & g.group.isin(['股票', '普通股'])]
    codes = set(gg.code.str.strip()); c4 = {c for c in codes if is_common(c)}
    d = LM.index[LM.index <= v][-1]
    ours = LM.loc[d]; ours = set(ours[ours].index)
    cnt[str(v.date())] = {'isin_stocks_incl_TIB': len(codes), 'isin_common_4digit': len(c4), 'ours': len(ours), 'ours_ex91': len({c for c in ours if not c.startswith('91')}),
                          'isin_ex91': len({c for c in c4 if not c.startswith('91')}),
                          'only_isin': sorted(c4 - ours)[:20], 'only_ours': sorted(ours - c4)[:20],
                          'diff_pct': (len({c for c in ours if not c.startswith('91')}) - len({c for c in c4 if not c.startswith('91')})) / len({c for c in c4 if not c.startswith('91')}) * 100}
d = LM.index[LM.index <= pd.Timestamp('2026-01-30')][-1]; ours = LM.loc[d]; ours = {c for c in ours[ours].index}
cnt['2026-02-01（taiwan-md 研究稿引用證交所：掛牌 1,066 家＝本國 978＋外國 88）'] = {'official_total': 1066, 'official_domestic': 978, 'ours_all': len(ours),
    'ours_ex91': len({c for c in ours if not c.startswith('91')}), 'diff_pct_vs_1066_ex91': (len({c for c in ours if not c.startswith('91')}) - 1066) / 1066 * 100}
sl = pd.read_csv(C.SLIST, dtype=str); slt = {c for c in sl[sl.market == 'TWSE'].code if is_common(c)}
d = LM.index[-1]; ours = LM.loc[d]; ours = {c for c in ours[ours].index}
cnt['stock_list 快照（2026-10，TWSE 4 碼非 0 開頭）'] = {'stock_list': len(slt), 'ours_at_' + str(d.date()): len(ours), 'only_stock_list': sorted(slt - ours)[:20], 'only_ours': sorted(ours - slt)[:20]}
# 每個訊號日（年底附近）的上市普通股檔數
yr = {}
for y in range(2006, 2027):
    dd = LM.index[LM.index <= pd.Timestamp(f'{y}-12-31')][-1]
    s = LM.loc[dd]; yr[str(dd.date())] = {'listed': int(s.sum()), 'listed_ex91': int(sum(1 for c in s[s].index if not c.startswith('91'))),
                                          'listed_with_price_that_day': int(sum(1 for c in s[s].index if c in cl6.columns and pd.notna(cl6.at[dd, c]) if dd in cl6.index))}
cnt['year_end_counts'] = yr
out['population_counts'] = cnt
# ---- V6 上市空洞（V6 面板空白、但上市且 hschi/Tasi 有成交）----
hs = S.get('hschi'); ta = S.get('tasi')
tr = pd.concat([hs[['date', 'code']], ta[['date', 'code']], S.get('leochen')[['date', 'code']], S.get('bacardi')[['date', 'code']]]).drop_duplicates()
tr = tr[(tr.date >= C.W_A) & (tr.date <= C.W_B) & tr.code.isin(COLS)]
tr['listed'] = [bool(LM.at[d, c]) if d in LM.index else False for d, c in zip(tr.date, tr.code)]
tr['v6'] = [pd.notna(cl6.at[d, c]) if d in cl6.index else False for d, c in zip(tr.date, tr.code)]
hole = tr[tr.listed & ~tr.v6]
out['v6_listed_holes'] = {'n_codes': int(hole.code.nunique()), 'n_stock_days': int(len(hole)),
                          'by_code': hole.groupby('code').date.agg(['min', 'max', 'size']).astype(str).reset_index().to_dict('records')}
extra = tr[~tr.listed & tr.v6]
out['v6_has_price_but_not_listed'] = {'n_stock_days': int(len(extra)), 'by_code': extra.groupby('code').date.agg(['min', 'max', 'size']).astype(str).reset_index().to_dict('records')}
# ---- 上市但沒有任何價格資料的股日（2024-01～2025-09），附資料恢復後的 V60 名次 ----
anyp = set(zip(tr.date, tr.code))
W = LM.loc[C.W_A:C.W_B]
miss = []
for c in W.columns:
    s = W[c]; s = s[s]
    if not len(s): continue
    nd = [d for d in s.index if (d, c) not in anyp and d in set(C.taiex_cal())]
    if nd: miss.append((c, len(nd), str(min(nd).date()), str(max(nd).date())))
out['listed_without_any_price_window'] = sorted(miss, key=lambda x: -x[1])[:60]
out['n_listed_without_any_price_window'] = {'codes': len(miss), 'stock_days': int(sum(x[1] for x in miss))}
# ---- 欄位外的上市股 ----
out['listed_codes_outside_v4_columns'] = M[M.in_panel_cols == 'False'][['code', 'segments', 'evidence_srcs']].to_dict('records')
# ---- 創新板（TIB）----
tib = sorted({c for c, n in zip(trv.code, trv.name) if isinstance(n, str) and n.endswith('-創')} | set(I[I.market.str.contains('創新板')].code.str.strip()))
out['TIB_codes'] = tib
f = C.HERE / 'explore.json'; ex = C.jload(f); ex['LISTING'] = out; C.jdump(ex, f)
print(json.dumps(C.conv({k: out[k] for k in ('three_way', 'v6_listed_holes', 'n_listed_without_any_price_window')}), ensure_ascii=False)[:3000])
print(json.dumps(C.conv({k: v for k, v in cnt.items() if k != 'year_end_counts'}), ensure_ascii=False)[:3000])
print(yr)
