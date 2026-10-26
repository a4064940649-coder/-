#!/usr/bin/env python3
"""驗證 TWSE 日線面板：
1. 方法一致性：Asoul 與 k66 重疊期（2010-01～2019-05）各自用 compute_returns 算日報酬，逐日比對。
2. 外部比對：tw_stocker 近似收盤（2024-02-15～2025-09-23）對 FinMind 官方收盤（python-actions.finmind type18，約 142 檔）。
3. 月報酬誤差：月底收盤算月報酬，面板 vs 官方。
輸出 research/tw_stock_selection/output/panel_validation.json
"""
import json, glob, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_twse_panel import read_exchange, compute_returns, is4, S

OUTD = Path(__file__).resolve().parent / 'output'; OUTD.mkdir(exist_ok=True)
P = S / 'panel'
res = {}
# 1) Asoul vs k66
cal = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date'])['date']
CAL = pd.DatetimeIndex(sorted(set(cal)))
bad = {pd.Timestamp('2017-08-03'), pd.Timestamp('2018-07-12')}
diffs, n_days, n_codes = [], 0, 0
codes = sorted(c for c in set(Path(p).stem for p in glob.glob(str(S / 'tsec_all/*.csv'))) & set(Path(p).stem for p in glob.glob(str(S / 'k66/*.csv'))) if is4(c))
rng = np.random.default_rng(0); sample = sorted(rng.choice(codes, size=min(200, len(codes)), replace=False))
for code in sample:
    a = read_exchange(S / f'tsec_all/{code}.csv', False).loc['2010-01-04':'2019-05-16']
    if a['plus'].any(): continue                      # 該期間為櫃買者略過
    k = read_exchange(S / f'k66/{code}.csv', True).loc['2010-01-04':'2019-05-16']
    a = a[~a.index.isin(bad)]
    if len(a) < 250 or len(k) < 250: continue
    out = {}
    for name, df, src in [('a', a, 1), ('k', k, 2)]:
        d = df.drop(columns='plus').copy(); d['src'] = src
        d = d[CAL.get_indexer(d.index) >= 0]
        r, _ = compute_returns(d, [(d.index[0], d.index[-1])], None, CAL)
        out[name] = pd.Series(r, index=d.index)
    j = pd.concat(out, axis=1, sort=True).dropna()
    j = j.loc[~j.index.isin([d + pd.Timedelta(days=i) for d in bad for i in range(1, 5)])]
    diffs.append((j['a'] - j['k']).abs().to_numpy()); n_days += len(j); n_codes += 1
dd = np.concatenate(diffs)
res['asoul_vs_k66_daily_returns'] = {'codes': n_codes, 'days': int(n_days), 'share_abs_diff_lt_0.01pct': round(float((dd < 1e-4).mean()), 5),
                                     'share_abs_diff_gt_1pct': round(float((dd > 0.01).mean()), 6), 'p999_abs_diff_pct': round(float(np.quantile(dd, 0.999) * 100), 3)}
print(res['asoul_vs_k66_daily_returns'], flush=True)
# 2) tw_stocker vs FinMind 官方收盤
C = pd.read_parquet(P / 'close_raw.parquet'); SRC = pd.read_parquet(P / 'src.parquet'); ADJ = pd.read_parquet(P / 'adj.parquet')
fm = {}
for f in glob.glob(str(S / 'finmind_type18/raw_daily_k_chart_flow_*.csv')):
    d = pd.read_csv(f, dtype={'stock_code': str}, usecols=['stock_code', '交易_日期', '收盤價_元'])
    code = d['stock_code'].iloc[0] if len(d) else None
    if code and code in C.columns:
        fm[code] = pd.Series(pd.to_numeric(d['收盤價_元'], errors='coerce').to_numpy(), index=pd.to_datetime(d['交易_日期'])).dropna().sort_index()
print('FinMind official closes for panel codes:', len(fm), flush=True)
rows, mret = [], []
for code, off in fm.items():
    pc = C[code]; m = SRC[code] == 3
    j = pd.concat([pc[m], off], axis=1, join='inner', keys=['p', 'o']).dropna()
    if len(j) < 50: continue
    e = (j['p'] / j['o'] - 1)
    rows.append({'code': code, 'n': len(j), 'med_abs_pct': float(e.abs().median() * 100), 'mean_signed_pct': float(e.mean() * 100), 'share_exact': float((e.abs() < 1e-6).mean())})
    # 月底收盤月報酬（不含息，兩邊同口徑比較價格報酬）
    pm = j['p'].groupby([j.index.year, j.index.month]).last(); om = j['o'].groupby([j.index.year, j.index.month]).last()
    mret.append(((pm.pct_change() - om.pct_change()).dropna() * 100).to_numpy())
t = pd.DataFrame(rows); mr = np.concatenate(mret) if mret else np.array([])
res['tw_stocker_vs_official_close'] = {'codes': int(len(t)), 'days_per_code_median': int(t['n'].median()), 'median_abs_err_pct_median': round(float(t['med_abs_pct'].median()), 3),
                                       'share_days_exact_median': round(float(t['share_exact'].median()), 3), 'mean_signed_err_pct_median': round(float(t['mean_signed_pct'].median()), 4),
                                       'monthly_price_return_err_pp': {'mean': round(float(mr.mean()), 4), 'mad': round(float(np.median(np.abs(mr))), 3), 'p95_abs': round(float(np.quantile(np.abs(mr), 0.95)), 3), 'n': int(len(mr))}}
print(res['tw_stocker_vs_official_close'], flush=True)
# 3) 面板基本統計
R = pd.read_parquet(P / 'ret.parquet'); F = pd.read_parquet(P / 'flag.parquet')
big = (R.abs() > 0.105) & (F == 0)
res['normal_days_beyond_10.5pct'] = int(big.values.sum())
res['normal_days_beyond_10.5pct_by_year'] = {int(y): int(v) for y, v in big.sum(axis=1).groupby(big.index.year).sum().items() if v > 0}
ew = R.mean(axis=1)
tx = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date']).set_index('date')['close'].pct_change()
j = pd.concat([ew, tx], axis=1, join='inner').dropna(); j.columns = ['ew', 'taiex']
res['corr_ew_panel_vs_taiex_daily'] = round(float(j.corr().iloc[0, 1]), 4)
res['corr_by_period'] = {k: round(float(j.loc[a:b].corr().iloc[0, 1]), 4) for k, (a, b) in {'2004-2009': ('2004', '2009'), '2010-2019-05': ('2010', '2019-05-16'), '2019-05-2023': ('2019-05-17', '2023'), '2024-02-2025-09': ('2024-02-15', '2025-09-23'), '2025-09-2026-10': ('2025-09-24', '2026')}.items()}
(OUTD / 'panel_validation.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps({k: res[k] for k in ['normal_days_beyond_10.5pct', 'normal_days_beyond_10.5pct_by_year', 'corr_ew_panel_vs_taiex_daily', 'corr_by_period']}, ensure_ascii=False))
