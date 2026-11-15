#!/usr/bin/env python3
"""S11 第 0 步：資料檢查（寫 PREREG 之前做；只看資料涵蓋與日期，不算任何報酬）。python3 -I -B s0_datacheck.py → data_check.json
檢查：(1) 季報 EPS／營收／毛利涵蓋哪些季；(2) 上市股池每季有幾檔有 SUE、SUE⊥；(3) 市場可得日 A 的來源與相對法定期限的分布（驗證期限）；
(4) 晚於訊號日才可得而被排除的檔數；(5) 下市股是否在資料內。"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s11lib
from s11lib import stlib, s4lib

H = Path(__file__).resolve().parent
P = s11lib.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22')
U0 = stlib.universe(P, 'twse')
INNOV, _ = s4lib.innov_panel(P)
U = U0 & ~INNOV
del U0, INNOV
FT = s11lib.fin_tables()
f = FT['fin']
out = {}
full = pd.read_parquet(s11lib.PEAD / 'fin_long.parquet', columns=['code', 'y', 'q', 'industry', 'eps_s', 'rev_s', 'gp_s'])
full = full[full['code'].str.fullmatch(r'[1-9]\d{3}')]
out['fin_range'] = dict(first=f"{int(full['y'].min())}Q{int(full[full['y'] == full['y'].min()]['q'].min())}",
                        last=f"{int(full['y'].max())}Q{int(full[full['y'] == full['y'].max()]['q'].max())}", rows=int(len(full)), codes=int(full['code'].nunique()))
cov = full.groupby(['y', 'q']).agg(n=('code', 'size'), eps=('eps_s', lambda s: int(s.notna().sum())), rev=('rev_s', lambda s: int(s.notna().sum())),
                                   gp=('gp_s', lambda s: int(s.notna().sum()))).reset_index()
out['fin_coverage_by_quarter'] = {f'{int(r.y)}Q{int(r.q)}': dict(rows=int(r.n), eps=int(r.eps), rev=int(r.rev), gp=int(r.gp)) for r in cov.itertuples()}
# 上市（面板上曾是 twse）且下市（最後一天不在面板）的公司
twse_ever = P.MK.max(axis=0) == 1
listed_last = P.MK[-1] > 0
delisted = set(P.COLS[twse_ever & ~listed_last])
fin_codes = set(full['code'])
out['delisted_twse'] = dict(n_delisted_twse_codes_in_panel=len(delisted), n_with_fin_rows=len(delisted & fin_codes),
                            n_with_sue=int(sum(1 for c in delisted if c in FT['SUE'].index and FT['SUE'].loc[c].notna().any())))
# 每季：上市股池、可選檔數、被排除的晚到檔數
rows = s11lib.quarter_rows(P, [k for k in FT['SUE'].columns if k >= 2015 * 4])
per = {}
for kq, k in sorted(rows.items()):
    w, sc, d = s11lib.q_scores(P, FT, U, kq, k, 'perp')
    w2, _, d2 = s11lib.q_scores(P, FT, U, kq, k, 'raw')
    dl_codes = [c for c in P.COLS[w2] if c in delisted]
    per[f'{kq // 4}Q{kq % 4 + 1}'] = dict(sig_date=str(P.IDX[k].date()), univ_twse=d['n_univ'], univ_nonfin=d['n_nonfin'], with_sue=d['n_sue'],
                                         late_excluded=d['n_late_excluded'], elig_perp=d['n_elig'], elig_raw=d2['n_elig'],
                                         coverage_raw=round(d2['n_elig'] / max(d['n_nonfin'], 1), 4), later_delisted_in_elig=len(dl_codes),
                                         corr_pct_sue_rsue=d.get('corr_pct_sue_rsue'))
out['per_quarter'] = per
valid = {k: v for k, v in per.items() if v['elig_perp'] >= 50}
out['first_signal_quarter'] = min(valid) if valid else None
out['n_signal_quarters'] = len(valid)
out['coverage_raw_min'] = min(v['coverage_raw'] for v in valid.values())
out['coverage_raw_median'] = float(np.median([v['coverage_raw'] for v in valid.values()]))
# A 相對法定期限（上市、非金融、有 EPS）
a = pd.read_parquet(s11lib.PEAD / 'avail_dates.parquet')
a = a[(a['mkt'] == 'twse') & a['code'].isin(full.loc[full['eps_s'].notna(), 'code'].unique())]
a = a.merge(full[['code', 'y', 'q', 'industry']], on=['code', 'y', 'q'], how='left')
a = a[~a['industry'].isin(s11lib.FIN_IND)]
a['dl'] = [s11lib.deadline(y, q) for y, q in zip(a['y'], a['q'])]
a['days_vs_dl'] = (pd.to_datetime(a['A']) - a['dl']).dt.days
# 期限順延後的交易日 k、訊號列 k+1
a['dl_td'] = [P.IDX[min(int(np.searchsorted(P.IDX.values, np.datetime64(d), side='left')), P.T - 1)] for d in a['dl']]
a['sig_td'] = [P.IDX[min(int(np.searchsorted(P.IDX.values, np.datetime64(d), side='left')) + 1, P.T - 1)] for d in a['dl']]
out['A_method_twse_nonfin'] = a['A_method'].value_counts().to_dict()
res = {}
for (m, qq), g in a.groupby([a['A_method'] == 'fsq', 'q']):
    x = g['days_vs_dl']
    res[f"{'fsq' if m else 'inferred'}|Q{qq}"] = dict(n=int(len(g)), share_A_le_deadline_td=float((pd.to_datetime(g['A']) <= g['dl_td']).mean()),
                                                     share_A_le_signal_day=float((pd.to_datetime(g['A']) <= g['sig_td']).mean()),
                                                     days_before_deadline_p10=float(-x.quantile(0.9)), days_before_deadline_median=float(-x.median()),
                                                     days_before_deadline_p90=float(-x.quantile(0.1)), max_days_late=float(x.max()))
out['A_vs_deadline'] = res
(H / 'data_check.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
print(json.dumps({k: v for k, v in out.items() if k not in ('fin_coverage_by_quarter', 'per_quarter')}, ensure_ascii=False, indent=1, default=str))
for k, v in per.items():
    print(k, v)
