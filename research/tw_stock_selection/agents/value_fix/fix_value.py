#!/usr/bin/env python3
"""面板成交值（value）異常：診斷 → 修正 → 驗證。輸出 value_fixed.parquet、fix_log.parquet、results_fix.json。

原因（見 FINDINGS）：voidful/tw_stocker 原始 5 分 K 檔在 2024-04-09～04-16、2025-04-30～2025-07-04 有「同一時間戳重複列」
（抓取程式把重疊的歷史窗重複附加），彙總成日 K 時直接 sum(Volume)，成交量被乘上重複倍數 k（2～10）。
另外 2025-01-21、2025-04-10 兩天原始檔缺下午／部分 K 棒，成交量偏低（收盤價也不是 13:20）。

修正（只改異常股日，src 3 期間）：
  1. 有當沖資料（dt_vol ≥ 1 張、dt_ratio > 0）且與去重後量相差在 3 倍以內 → 官方量（dt_vol ÷ dt_ratio）× 面板收盤 close_raw
  2. 否則，重複列股日 → 面板 value × （去重後量 ÷ 原始加總量）（保留面板原本的 Yahoo × 個股放大倍數口徑）
  3. 否則（截斷日，沒有當沖資料）→ 面板 value ÷ 當日截面「面板量 ÷ 官方量」中位數（相對個股基準）
"""
import json, glob
from pathlib import Path
import numpy as np, pandas as pd

S = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad')
P = S / 'panel'
OUT = Path(__file__).resolve().parent
S3A, S3B = pd.Timestamp('2024-02-15'), pd.Timestamp('2025-09-23')

V = pd.read_parquet(P / 'value.parquet'); C = pd.read_parquet(P / 'close_raw.parquet'); SRC = pd.read_parquet(P / 'src.parquet')
IDX, COLS = V.index, V.columns
res = {}

# ---------- 原始 5 分 K 彙總 ----------
A = pd.concat({Path(f).stem: pd.read_parquet(f) for f in sorted(glob.glob(str(S / 'vf_agg/*.parquet')))}, names=['code', 'date']).reset_index()
A['date'] = pd.to_datetime(A['date']).astype('datetime64[us]')
A['k'] = A['n_rows'] / A['n_ts']
res['raw_files'] = {'n_codes': int(A['code'].nunique()), 'codes_in_panel_src3': int(((SRC == 3).any()).sum())}
# 與 tw_stocker_daily（面板實際用的日 K）比對：原始加總量應完全相等
chk = []
for c in A['code'].unique()[::25]:
    f = S / f'tw_stocker_daily/{c}.csv'
    if not f.exists(): continue
    d = pd.read_csv(f, parse_dates=['date']).set_index('date')['volume']
    a = A[A['code'] == c].set_index('date')['vol_raw']
    j = pd.concat([a, d], axis=1, keys=['a', 'd']).dropna()
    chk.append(float((j['a'] == j['d']).mean()))
res['raw_sum_equals_tw_stocker_daily_share'] = round(float(np.mean(chk)), 5)

# ---------- 官方量（當沖還原） ----------
dt = pd.read_csv(S / 'scr_extra/daytrade_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
dt['date'] = dt['date'].astype('datetime64[us]')
dt = dt[(dt['dt_vol'] >= 1) & (dt['dt_ratio'] > 0)].copy()
dt['off_vol'] = dt['dt_vol'] / (dt['dt_ratio'] / 100) * 1000
dt = dt.drop_duplicates(['date', 'code'])

def stack(df, name):
    s = df.stack(); s.index.names = ['date', 'code']; return s.rename(name)

base = pd.concat([stack(V, 'val'), stack(C, 'close'), stack(SRC, 'src')], axis=1).reset_index()
base = base[(base['date'] >= S3A) & (base['date'] <= S3B) & (base['src'] == 3)]
M = base.merge(A, on=['date', 'code'], how='left').merge(dt[['date', 'code', 'off_vol', 'dt_vol', 'dt_ratio']], on=['date', 'code'], how='left')
M['pvol'] = M['val'] / M['close']
# 面板放大倍數 sc：value = vol_raw × close × sc（每檔常數）
M['sc_obs'] = M['val'] / (M['vol_raw'] * M['close'])
sc = M[M['vol_raw'] > 0].groupby('code')['sc_obs'].median()
dev = (M['sc_obs'] / M['code'].map(sc) - 1).abs()
res['panel_scale'] = {'median_sc': round(float(sc.median()), 4), 'share_rows_sc_constant_within_1e-6': round(float((dev < 1e-6).mean()), 5)}

# ---------- 異常股日 ----------
M['dup'] = M['k'] > 1.0001
r_off = M['pvol'] / M['off_vol']
normal = ~M['dup'] & M['off_vol'].notna()
b_i = r_off[normal].groupby(M.loc[normal, 'code']).median()              # 個股基準：正常日 面板量／官方量
M['b'] = M['code'].map(b_i)
M['rr'] = r_off / M['b']                                                 # 相對個股基準
day_rr = M[~M['dup']].groupby('date')['rr'].median()
day_nts = M.groupby('date')['n_ts'].median()
trunc_days = sorted(set(day_rr[(day_rr < 0.9)].index))                   # 無重複列但截面明顯偏低
over_days = sorted(set(day_rr[(day_rr > 1.2)].index))                    # 無重複列但截面明顯偏高
res['day_level_flags_no_dup'] = {'trunc_days(rr<0.9)': {str(d.date()): {'rr_med': round(float(day_rr[d]), 3), 'n_ts_med': float(day_nts[d])} for d in trunc_days},
                                 'over_days(rr>1.2)': {str(d.date()): {'rr_med': round(float(day_rr[d]), 3), 'n_ts_med': float(day_nts[d])} for d in over_days}}
# 截斷日：個股最後一根時間
for d in trunc_days:
    x = M[M['date'] == d]
    res['day_level_flags_no_dup']['trunc_days(rr<0.9)'][str(d.date())]['last_bar_time_top'] = x['last_ts'].value_counts().head(5).to_dict()
M['trunc'] = M['date'].isin(trunc_days)
M['anom'] = M['dup'] | M['trunc']

# 去重法選擇：重複股日裡，哪一種去重最接近官方（相對個股基準）
cmp = {}
for nm in ('vol_first', 'vol_last', 'vol_max'):
    q = M[nm] / M['vol_raw']
    rr = (M['pvol'] * q / M['off_vol']) / M['b']
    x = rr[M['dup'] & M['off_vol'].notna() & (M['dt_vol'] >= 10)]
    cmp[nm] = {'median': round(float(x.median()), 4), 'median_abs_log': round(float(np.log(x).abs().median()), 4), 'n': int(len(x))}
xn = M.loc[normal & (M['dt_vol'] >= 10), 'rr']
cmp['normal_days_reference'] = {'median': round(float(xn.median()), 4), 'median_abs_log': round(float(np.log(xn).abs().median()), 4), 'n': int(len(xn))}
res['dedup_rule_vs_official'] = cmp
best = min(('vol_first', 'vol_last', 'vol_max'), key=lambda k: cmp[k]['median_abs_log'])
res['dedup_rule_used'] = best
M['q'] = M[best] / M['vol_raw']
# 少數極冷門股日：最早那份抓取的 K 棒量全是 0、後來的重複列才有量 → 去重後變 0。這些改用 vol_max。
zq = (M['q'] == 0) & (M['vol_raw'] > 0)
M.loc[zq, 'q'] = M.loc[zq, 'vol_max'] / M.loc[zq, 'vol_raw']
res['n_rows_first_copy_zero_use_max'] = int((zq & M['dup']).sum())

# ---------- 修正 ----------
v_dd = M['pvol'] * M['q']                                               # 去重後的面板量
ok_off = M['off_vol'].notna() & ((M['off_vol'] / v_dd.where(M['dup'], M['pvol'])).between(1 / 3, 3) | M['trunc'])
M['method'] = np.select([M['anom'] & ok_off, M['anom'] & M['dup'] & M['q'].notna(), M['anom']], ['official_daytrade', 'dedup_scale', 'day_median_scale'], 'none')
M['val_fix'] = M['val']
m1 = M['method'] == 'official_daytrade'; M.loc[m1, 'val_fix'] = M.loc[m1, 'off_vol'] * M.loc[m1, 'close']
m2 = M['method'] == 'dedup_scale'; M.loc[m2, 'val_fix'] = M.loc[m2, 'val'] * M.loc[m2, 'q']
m3 = M['method'] == 'day_median_scale'
dmed = M[M['anom'] & M['off_vol'].notna()].groupby('date')['rr'].median()
M.loc[m3, 'val_fix'] = M.loc[m3, 'val'] / M.loc[m3, 'date'].map(dmed)
res['n_fixed_by_method'] = M.loc[M['anom'], 'method'].value_counts().to_dict()
res['anomaly_days'] = {'dup_days': sorted({str(d.date()) for d in M.loc[M['dup'], 'date']}), 'trunc_days': [str(d.date()) for d in trunc_days],
                       'n_stock_days': int(M['anom'].sum())}
res['dup_per_day'] = {str(d.date()): {'k_med': round(float(g['k'].median()), 2), 'k_min': round(float(g['k'].min()), 2), 'k_max': round(float(g['k'].max()), 2),
                                      'share_dup': round(float(g['dup'].mean()), 3)} for d, g in M[M['date'].isin(M.loc[M['dup'], 'date'].unique())].groupby('date')}

VF = V.copy()
fx = M[M['method'] != 'none'].pivot(index='date', columns='code', values='val_fix')
for c in fx.columns:
    s = fx[c].dropna(); VF.loc[s.index, c] = s.to_numpy()
VF.to_parquet(OUT / 'value_fixed.parquet')
M.loc[M['method'] != 'none', ['date', 'code', 'val', 'val_fix', 'method', 'k', 'q', 'n_rows', 'n_ts', 'last_ts', 'off_vol', 'dt_vol', 'dt_ratio']].to_parquet(OUT / 'fix_log.parquet')
res['changed_cells'] = int(((VF != V) & V.notna()).sum().sum())

# ---------- 驗證 ----------
def agree(x):
    e = np.log(x.dropna()); e = e[np.isfinite(e)]
    return {'n': int(len(e)), 'median_ratio': round(float(np.exp(e.median())), 4), 'p10': round(float(np.exp(e.quantile(.1))), 3), 'p90': round(float(np.exp(e.quantile(.9))), 3),
            'within5%': round(float((e.abs() < np.log(1.05)).mean()), 3), 'within20%': round(float((e.abs() < np.log(1.2)).mean()), 3)}
M['pvol_fix'] = M['val_fix'] / M['close']
liq = M['dt_vol'] >= 10
V_ = {}
V_['normal_days_panel_vs_official'] = agree((M['pvol'] / M['off_vol'])[normal & liq])
V_['anom_days_before_fix'] = agree((M['pvol'] / M['off_vol'])[M['anom'] & liq])
V_['anom_days_dedup_only_vs_official(未用官方量時的精度)'] = agree((M['pvol'] * M['q'] / M['off_vol'])[M['dup'] & liq])
V_['anom_days_after_fix_all'] = agree((M['pvol_fix'] / M['off_vol'])[M['anom'] & liq])
V_['anom_days_after_fix_fallback_rows_only'] = {'n_fallback_rows': int((M['anom'] & (M['method'] != 'official_daytrade')).sum()),
                                                'n_with_offvol_but_rejected': int((M['anom'] & (M['method'] != 'official_daytrade') & M['off_vol'].notna()).sum())}
dr_before = (M['pvol'] / M['off_vol'])[liq].groupby(M.loc[liq, 'date']).median()
dr_after = (M['pvol_fix'] / M['off_vol'])[liq].groupby(M.loc[liq, 'date']).median()
V_['daily_xsec_median_ratio'] = {'before_days_gt1.15': int((dr_before > 1.15).sum()), 'before_days_lt0.87': int((dr_before < 0.87).sum()),
                                 'after_days_gt1.15': int((dr_after > 1.15).sum()), 'after_days_lt0.87': int((dr_after < 0.87).sum()),
                                 'after_max': [str(dr_after.idxmax().date()), round(float(dr_after.max()), 3)], 'after_min': [str(dr_after.idxmin().date()), round(float(dr_after.min()), 3)],
                                 'after_outside_0.92_1.08': {str(d.date()): round(float(v), 3) for d, v in dr_after[(dr_after > 1.08) | (dr_after < 0.92)].items()}}
tot0 = V.sum(axis=1) / 1e8; tot1 = VF.sum(axis=1) / 1e8
segs = {'2024-03-01~04-08': ('2024-03-01', '2024-04-08'), '2024-04-09~04-16': ('2024-04-09', '2024-04-16'), '2024-04-17~05-31': ('2024-04-17', '2024-05-31'),
        '2025-03-01~04-29': ('2025-03-01', '2025-04-29'), '2025-04-30~07-04': ('2025-04-30', '2025-07-04'), '2025-07-07~09-23': ('2025-07-07', '2025-09-23'), '2025-09-24~10-31(src4)': ('2025-09-24', '2025-10-31')}
V_['market_value_median_1e8'] = {k: {'before': round(float(tot0[a:b].median())), 'after': round(float(tot1[a:b].median()))} for k, (a, b) in segs.items()}
# 60 日均值與流動性排名：以「官方量 × 收盤（有當沖資料者）」為參考
REF = V.copy()
ofv = M[M['off_vol'].notna()].pivot(index='date', columns='code', values='off_vol') * C.reindex(index=sorted(M.loc[M['off_vol'].notna(), 'date'].unique()))
ofv = ofv.reindex(columns=COLS)
for c in ofv.columns:
    s = ofv[c].dropna()
    if len(s): REF.loc[s.index, c] = s.to_numpy()
def rank_cmp(X, d):
    a = X.rolling(60).mean().loc[d]; r = REF.rolling(60).mean().loc[d]
    ok = a.notna() & r.notna() & (C.loc[d] >= 10)
    sp = float(a[ok].rank().corr(r[ok].rank()))
    t_a = set(a[ok & (a >= 1e7)].nlargest(300).index); t_r = set(r[ok & (r >= 1e7)].nlargest(300).index)
    return round(sp, 4), len(t_a ^ t_r) // 2
rk = {}
for d in ['2024-04-16', '2024-05-15', '2025-03-11', '2025-05-12', '2025-06-11', '2025-07-04', '2025-07-11', '2025-08-11', '2025-09-11', '2025-09-23']:
    dd = IDX[IDX <= pd.Timestamp(d)][-1]
    rk[str(dd.date())] = {'orig_spearman_vs_ref,top300_swaps': rank_cmp(V, dd), 'fixed': rank_cmp(VF, dd)}
V_['v60_rank_vs_official_reference'] = rk
res['validation'] = V_
res['notes'] = {'REF': 'src 3 期間有當沖資料的股日改用官方量 × close_raw，其餘維持面板值；只作為排名比對的參考',
                'b_i': '個股正常日「面板量÷官方量」中位數，用於扣掉 Yahoo 量 × 放大倍數的個股系統誤差'}
(OUT / 'results_fix.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps(res, ensure_ascii=False, default=str)[:12000])
