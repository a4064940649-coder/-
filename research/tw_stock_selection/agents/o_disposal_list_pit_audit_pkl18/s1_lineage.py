#!/usr/bin/env python3
"""S1：處置清單溯源鏈逐階段計數（年 × 月 × 階段）、封存內部一致性、announce 接回與 PIT 時點核對。不看任何報酬。
python3 -I -B s1_lineage.py → s1_lineage.json、data/lineage_year_month.csv、data/universe_announce.csv"""
import json, glob, hashlib, re
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
RAW = HERE / 'raw'
PANEL = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
(HERE / 'data').mkdir(exist_ok=True)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(RAW / 'lineage/disposal_official_universe.csv').startswith('2842f45d')
def roc(s):
    s = s.strip().lstrip('*'); y, m, d = s.split('/'); return pd.Timestamp(int(y) + 1911, int(m), int(d))
out = {}
# ---------- A：raw 年檔（含跨年重疊） ----------
rows = []; internal = {}
for f in sorted(glob.glob(str(RAW / 'twse_punish/twse_20??.json'))):
    y = int(Path(f).stem[-4:]); d = json.loads(Path(f).read_text(encoding='utf-8'))
    fl = d['fields']; ix = {k: fl.index(k) for k in fl}
    grp = {}
    for r in d['data']:
        a, b = r[ix['處置起迄時間']].replace(' ', '').split('～')
        rec = dict(file_year=y, no=int(r[ix['編號']]), announce=roc(r[ix['公布日期']]), code=str(r[ix['證券代號']]).strip(), name=str(r[ix['證券名稱']]).strip(),
                   cum=int(r[ix['累計']]), condition=r[ix['處置條件']], start=roc(a), end=roc(b), measure=r[ix['處置措施']].strip())
        rows.append(rec); grp.setdefault(rec['no'], []).append(rec)
    viol = []
    for no, g in grp.items():
        cums = {x['cum'] for x in g}; codes = {x['code'] for x in g}
        if len(codes) != 1 or len(cums) != 1 or list(cums)[0] != len(g):
            viol.append(dict(no=no, codes=sorted(codes), cum=sorted(cums), n_rows=len(g)))
    internal[y] = dict(total=int(d['total']), n_data=len(d['data']), count=int(d['count']), max_no=max(grp), n_groups=len(grp),
                       cum_eq_rows_violations=len(viol), viol_examples=viol[:5], title=d['title'])
A = pd.DataFrame(rows)
A['is_common'] = (A.code.str.len() == 4) & (~A.code.str.startswith('0')) & (~A.name.str.contains('DR'))
out['A_internal_by_file'] = internal
# 跨年重疊：同 (announce, code, start, end, measure) 在兩個年檔都出現
key = ['announce', 'code', 'start', 'end', 'measure']
dup = A.groupby(key).file_year.nunique()
out['A_rows'] = int(len(A)); out['A_unique_keys'] = int(len(dup)); out['A_keys_in_2files'] = int((dup > 1).sum())
# 跨年重疊是否完整：start 或 end 跨到下一年（或 announce 在前一年）的列，是否同時在兩個年檔
def years_touched(r):
    return set(range(min(r.announce.year, r.start.year), r.end.year + 1))
miss_overlap = []
fy = A.groupby(key).file_year.apply(set)
for k, s in fy.items():
    r = pd.Series(dict(zip(key, k)))
    # TWSE 查詢語意：處置期（或公布日）與查詢區間重疊 → 期間觸及的年份都應出現
    need = set(range(r.start.year, r.end.year + 1)) & set(range(2003, 2027))
    if not need <= s: miss_overlap.append(dict(code=k[1], announce=str(k[0].date()), start=str(k[2].date()), end=str(k[3].date()), in_files=sorted(s), need=sorted(need)))
out['A_overlap_semantics'] = dict(n_keys_missing_expected_file=len(miss_overlap), examples=miss_overlap[:10])
# ---------- B：rejudge（去重） ----------
B = pd.read_csv(RAW / 'lineage/disposal_hist_rejudge.csv', dtype={'code': str}, parse_dates=['announce', 'start', 'end'])
B['ordr'] = np.arange(len(B)); B['name'] = B['name'].fillna('')
B['is_common'] = (B.code.str.len() == 4) & (~B.code.str.startswith('0')) & (~B.name.str.contains('DR'))
Bk = B.set_index(key).index; Ak = pd.MultiIndex.from_frame(A[key].drop_duplicates())
out['B_rows'] = int(len(B)); out['B_eq_A_unique'] = bool(set(Bk) == set(Ak)); out['B_dup_keys'] = int(Bk.duplicated().sum())
# ---------- E：universe ----------
U = pd.read_csv(RAW / 'lineage/disposal_official_universe.csv', dtype={'code': str}, parse_dates=['start', 'end'])
out['U_rows'] = int(len(U))
J = U.merge(B[['ordr', 'announce', 'code', 'start', 'end', 'name', 'kind', 'regime', 'measure', 'condition', 'nth', 'in_panel', 'cum', 'file_year', 'n_tdays']],
            on='ordr', how='left', suffixes=('', '_b'))
ok_ordr = (J.code == J.code_b) & (J.start == J.start_b) & (J.end == J.end_b)
# (code, start) 一對一
cs_B = B[B.is_common].groupby(['code', 'start']).size()
cs_U = U.groupby(['code', 'start']).size()
J2 = U.merge(B[B.is_common][['code', 'start', 'announce', 'ordr']], on=['code', 'start'], how='left', suffixes=('', '_cs'))
out['announce_join'] = dict(by_ordr_match=int(ok_ordr.sum()), by_ordr_rate=float(ok_ordr.mean()),
                            code_start_keys_in_U=int(len(cs_U)), U_code_start_dups=int((cs_U > 1).sum()), B_common_code_start_dups=int((cs_B > 1).sum()),
                            B_common_code_start_dup_examples=[dict(code=c, start=str(s.date()), n=int(n)) for (c, s), n in cs_B[cs_B > 1].head(10).items()],
                            by_code_start_rows=int(len(J2)), by_code_start_unique_announce=bool(J2.groupby('ordr').announce.nunique().max() == 1),
                            by_code_start_ordr_agree=float((J2.ordr == J2.ordr_cs).groupby(J2.ordr).any().mean()))
# C 普通股 → E 的關係
C = B[B.is_common]
out['C_rows'] = int(len(C)); out['C_minus_U'] = int(len(set(C.ordr) - set(U.ordr))); out['U_minus_C'] = int(len(set(U.ordr) - set(C.ordr)))
out['U_kind_counts'] = J.kind.value_counts().to_dict(); out['U_in_panel'] = J.in_panel.value_counts().to_dict()
# ---------- PIT：announce vs start ----------
C_ = pd.read_parquet(PANEL / 'close_raw.parquet'); IDX = C_.index
cnt = C_.notna().sum(axis=1); TD = IDX[cnt >= 300]          # 交易日曆（剔除有值檔數 < 300 的日子）
def tpos(x): return TD.searchsorted(x)
J['ann_lt_start'] = J.announce < J.start
GAP = (J.announce < TD[0]) | ((J.announce >= '2024-01-01') & (J.announce <= '2024-02-15'))   # 面板日曆外（2004 前、2024-01-02～02-14 缺口）
J['in_cal'] = ~GAP
J['lag_td'] = [int(tpos(s) - tpos(a)) if ok else -1 for a, s, ok in zip(J.announce, J.start, J['in_cal'])]
J['lag_cal'] = (J.start - J.announce).dt.days
J['ann_is_td'] = J.announce.isin(TD)
J['start_is_td'] = J.start.isin(TD)
out['PIT'] = dict(n=int(len(J)), ann_lt_start=int(J.ann_lt_start.sum()), start_le_ann=int((~J.ann_lt_start).sum()),
                  lag_td_dist=J.lag_td.value_counts().sort_index().astype(int).to_dict(),
                  lag_cal_dist=J.lag_cal.value_counts().sort_index().astype(int).to_dict(),
                  ann_not_tradingday_in_cal=int((~J.ann_is_td & J.in_cal).sum()), start_not_tradingday_in_cal=int((~J.start_is_td & J.in_cal).sum()), n_outside_panel_calendar=int((~J.in_cal).sum()),
                  lag_td_gt1_examples=J[J.lag_td > 1][['code', 'announce', 'start', 'lag_td', 'lag_cal']].astype(str).head(15).to_dict('records'))
# 主策略訊號日 d（每月第一個 ≥ 11 日的交易日；BRIEF/SOP），成交日 e＝下一交易日
sig = []
for p, g in pd.Series(TD, index=TD).groupby(TD.to_period('M')):
    x = g[g.index.day >= 11]
    if len(x): sig.append(x.index[0])
sig = pd.DatetimeIndex(sig); sig = sig[(sig >= '2006-01-01') & (sig <= '2026-10-31')]
rel = []
for d in sig:
    t = TD.get_loc(d); e = TD[t + 1] if t + 1 < len(TD) else d
    x = J[((J.start <= d) & (J.end >= d)) | ((J.start <= e) & (J.end >= e))]
    for r in x.itertuples():
        rel.append(dict(signal=d, trade=e, code=r.code, announce=r.announce, start=r.start, end=r.end,
                        covers_d=bool(r.start <= d <= r.end), covers_e=bool(r.start <= e <= r.end),
                        ann_vs_d='lt' if r.announce < d else ('eq' if r.announce == d else 'gt'), start_eq_e=bool(r.start == e)))
R = pd.DataFrame(rel)
out['trade_date_relation'] = dict(n_signal_dates=int(len(sig)), n_rows_cover_d_or_e=int(len(R)),
                                 ann_vs_d=R.ann_vs_d.value_counts().to_dict(),
                                 covers_e_only=int((R.covers_e & ~R.covers_d).sum()),
                                 covers_e_only_ann_eq_d=int((R.covers_e & ~R.covers_d & (R.ann_vs_d == 'eq')).sum()),
                                 start_eq_e=int(R.start_eq_e.sum()), ann_gt_d=int((R.ann_vs_d == 'gt').sum()))
R.to_csv(HERE / 'data/signal_trade_cover_rows.csv', index=False)
# ---------- 逐年 × 月 × 階段 ----------
def ym(df, col): return df[col].dt.to_period('M')
tabs = {}
A1 = A.copy(); tabs['A_raw_rows_by_fileyear'] = A1.groupby('file_year').size()
for nm, df in {'B_rejudge': B, 'C_common': C, 'D_common_inpanel': C[C.in_panel], 'E_universe': J}.items():
    tabs[nm] = df.groupby(df.start.dt.to_period('M')).size()
LYM = pd.DataFrame({k: v for k, v in tabs.items() if k != 'A_raw_rows_by_fileyear'}).fillna(0).astype(int)
LYM.index.name = 'start_month'; LYM.to_csv(HERE / 'data/lineage_year_month.csv')
yr = LYM.groupby(LYM.index.year).sum()
# 每階段濾掉什麼（依 start 年）
B['nonc_type'] = np.select([B.code.str.startswith('0'), B.name.str.contains('DR'), B.code.str.len() == 6, B.code.str.len() == 5, B.code.str.len() == 4],
                           ['0開頭（ETF/權證/ETN）', 'DR', '6碼（權證）', '5碼（特別股等）', '普通股'], '其他')
filt = B.groupby([B.start.dt.year, 'nonc_type']).size().unstack(fill_value=0)
kinds = J.groupby([J.start.dt.year, 'kind']).size().unstack(fill_value=0)
out['year_stage'] = {int(y): {k: int(v) for k, v in r.items()} for y, r in yr.iterrows()}
out['year_filter_types'] = {int(y): {k: int(v) for k, v in r.items()} for y, r in filt.iterrows()}
out['year_universe_kind'] = {int(y): {k: int(v) for k, v in r.items()} for y, r in kinds.iterrows()}
out['announce_year_universe'] = J.groupby(J.announce.dt.year).size().astype(int).to_dict()
out['month_2018_2020_universe'] = {str(k): int(v) for k, v in LYM.loc['2018-01':'2020-12', 'E_universe'].items()}
out['raw_file_rows'] = {int(k): int(v) for k, v in tabs['A_raw_rows_by_fileyear'].items()}
out['raw_file_common_rows'] = {int(k): int(v) for k, v in A[A.is_common].groupby('file_year').size().items()}
# 2019 年 universe 列明細
out['universe_2019'] = J[J.start.dt.year == 2019][['code', 'name', 'announce', 'start', 'end', 'measure', 'condition', 'kind', 'nth']].astype(str).to_dict('records')
# universe＋announce 輸出（DISP_OFFICIAL_v2 的基底）
J[['code', 'ordr', 'announce', 'start', 'end', 'overlap_proxy', 'second30', 'take_R1', 'kind', 'regime', 'measure', 'condition', 'nth', 'lag_td']].to_csv(HERE / 'data/universe_announce.csv', index=False)
(HERE / 's1_lineage.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps({k: out[k] for k in ['A_rows', 'A_unique_keys', 'A_keys_in_2files', 'A_overlap_semantics', 'B_rows', 'B_eq_A_unique', 'B_dup_keys', 'U_rows', 'announce_join', 'C_rows', 'C_minus_U', 'U_minus_C', 'U_kind_counts', 'U_in_panel', 'PIT', 'trade_date_relation']}, ensure_ascii=False, default=str, indent=0)[:6000])
print({y: (v['cum_eq_rows_violations'], v['total'], v['count'], v['max_no'], v['n_groups']) for y, v in internal.items()})
print(yr.to_string())
