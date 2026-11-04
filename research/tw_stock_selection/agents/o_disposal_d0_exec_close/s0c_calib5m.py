#!/usr/bin/env python3
"""第 0 步：盤中結構校準（不讀任何事件報酬）。
Yahoo 5 分 K（voidful/tw_stocker，已去重）缺 13:30 收盤集合競價那一盤（最後一根 bar 多為 13:20／13:25），
所以「最後一盤占全日量」＝ 1 − Yahoo 全日量 ÷ 官方全日量，只能在官方日 K 與 Yahoo 重疊的 2025-09-24～2026-03-31 估。
（2024-02-15～2025-09-23 面板的 src 3 本身就是 Yahoo，沒有獨立官方量可比；該段只報 bar 結構描述。）
輸出 calib5m.json：
  - 處置日（依撮合分鐘）與非處置日：最後一盤占比、第一盤（第一根非零 bar）占比、最後一盤價格相對前一盤（Yahoo 最後非零 bar 收盤）的對數偏離。
  - frozen_shares：事前寫死的取法＝各組日占比中位數；組內 < 30 日 → 用全部處置日中位數。
用法：python3 -I -B s0c_calib5m.py"""
import sys, json, math
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
F = pd.read_parquet(HERE / 'raw' / 'yahoo5m_daily_features.parquet')
F['date'] = pd.to_datetime(F['date'])
L = pd.read_csv(HERE / 'raw' / 'disposal_hist_rejudge.csv', dtype={'code': str}, parse_dates=['announce', 'start', 'end'])
L['name'] = L['name'].fillna('')
L = L[(L.code.str.len() == 4) & (~L.code.str.startswith('0')) & (~L.name.str.contains('DR'))]
# 處置日標記（同檔多筆重疊取撮合分鐘最大者）
dd = []
for r in L.itertuples():
    for d in pd.bdate_range(r.start, r.end):
        dd.append((r.code, d, r.match_min if np.isfinite(r.match_min) else np.nan, int(r.nth), r.regime))
DD = pd.DataFrame(dd, columns=['code', 'date', 'mm', 'nth', 'regime']).sort_values('mm').drop_duplicates(['code', 'date'], keep='last')
F = F.merge(DD, on=['code', 'date'], how='left')
F['disp'] = F['mm'].notna()
codes = set(F.code)
H = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/history.csv.gz', dtype={'code': str}, parse_dates=['date'])
H = H[H.code.isin(codes)][['date', 'code', 'open', 'close', 'volume']].rename(columns={'open': 'o_off', 'close': 'c_off', 'volume': 'v_off'})
X = F.merge(H, on=['code', 'date'], how='inner')
X = X[(X.v_off > 0) & (X.vol > 0)]
X['sh_last'] = 1 - X['vol'] / X['v_off']
X['sh_first'] = X['vol_first_nz'] / X['v_off']
with np.errstate(divide='ignore', invalid='ignore'):
    X['dev_last_vs_prev'] = np.log(X['c_off'] / X['close_last_nz'])
X['yahoo_close_match'] = (X['close_last'] - X['c_off']).abs() < 1e-6


def mmkey(m):
    if not np.isfinite(m): return 'norm'
    if m <= 3: return '2'
    if m <= 7: return '5'
    if m <= 12: return '10'
    if m <= 22: return '20'
    return 'long'


X['grp'] = [mmkey(m) for m in X['mm']]
ok = X[(X.sh_last > -0.05) & (X.sh_last < 0.95)]   # 排除明顯量不一致（Yahoo 漏 bar／官方錯置）
out = {'window_overlap': [str(X.date.min().date()), str(X.date.max().date())], 'n_days_overlap': int(len(X)),
       'n_days_used': int(len(ok)), 'excluded_inconsistent': int(len(X) - len(ok)),
       'hm_last_dist_disp': F[F.disp].hm_last.value_counts().head(6).to_dict(),
       'hm_last_dist_norm': F[~F.disp].hm_last.value_counts().head(6).to_dict(),
       'hm_first_nz_dist_disp': F[F.disp].hm_first_nz.value_counts().head(6).to_dict(),
       'groups': {}}
for g, x in ok.groupby('grp'):
    q = lambda s: {k: round(float(v), 4) for k, v in zip(('p10', 'p25', 'median', 'p75', 'p90'), s.quantile([.1, .25, .5, .75, .9]))}
    out['groups'][g] = {'n_days': int(len(x)), 'n_codes': int(x.code.nunique()),
                        'sh_last': q(x.sh_last), 'sh_first': q(x.sh_first.clip(0, 1)),
                        'dev_last_vs_prev_bp': {k: round(v * 1e4, 1) for k, v in q(x.dev_last_vs_prev.dropna()).items()},
                        'abs_dev_last_vs_prev_bp_median': round(float(x.dev_last_vs_prev.abs().median() * 1e4), 1),
                        'mean_n_nonzero_bars': round(float(x.n_nz.mean()), 1)}
# 2024-02-15～2025-09-23（Yahoo only）：bar 結構
Y = F[(F.date <= '2025-09-23')]
out['yahoo_only_2024_2025'] = {
    g: {'n_days': int(len(x)), 'mean_n_nonzero_bars': round(float(x.n_nz.mean()), 1),
        'median_first_nz_share_of_yahoo': round(float((x.vol_first_nz / x.vol).median()), 4),
        'median_last_nz_share_of_yahoo': round(float((x.vol_last_nz / x.vol).median()), 4)}
    for g, x in Y.assign(grp=[mmkey(m) for m in Y['mm']]).groupby('grp')}
# 事前寫死的取法
disp = ok[ok.grp != 'norm']
pool_last = float(disp.sh_last.median()); pool_first = float(disp.sh_first.clip(0, 1).median())
fz = {}
for g in ('2', '5', '10', '20', 'long'):
    x = ok[ok.grp == g]
    fz[f'last_{g}'] = round(float(x.sh_last.median()), 4) if len(x) >= 30 else round(pool_last, 4)
    fz[f'first_{g}'] = round(float(x.sh_first.clip(0, 1).median()), 4) if len(x) >= 30 else round(pool_first, 4)
    fz[f'src_{g}'] = 'group' if len(x) >= 30 else 'pooled_disposal'
nm = ok[ok.grp == 'norm']
fz['close_norm'] = round(float(nm.sh_last.median()), 4)
fz['open_norm'] = round(float(nm.sh_first.clip(0, 1).median()), 4)
out['frozen_shares'] = fz
out['rule'] = '各組日占比中位數；組內 < 30 日用全部處置日中位數；占比 = 1 − Yahoo 量 ÷ 官方量（最後一盤）、第一根非零 bar 量 ÷ 官方量（第一盤）'
json.dump(out, open(HERE / 'calib5m.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps({k: out[k] for k in ('n_days_overlap', 'n_days_used', 'frozen_shares')}, ensure_ascii=False, indent=1))
for g, v in out['groups'].items(): print(g, v)
