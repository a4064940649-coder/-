#!/usr/bin/env python3
"""S59 第 2 步：建月度環境變數（定義見 SPEC.json env_variables），只用本地公開資料。
輸出 agents/s59_regime_attribution/env_monthly.csv（月 t 一列；除 MKT_RET、SMB_RET、SURGE_TOP、NTGT、W_ACT 外皆為 t−1 月底可得）。
用法：python3 -I -B env.py（需先跑 engine.py oos／is）"""
import json, resource
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SCR = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s59')
DP = ROOT / 'data_panel' / 'panel'
E50_CSV = '/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv'
REV_CSV = '/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv'
IND_CSV = HERE.parent / 'o_lc_sector_rotation_pit' / 'raw' / 'industry_pit.csv'
ELEC = {'半導體業', '電腦及週邊設備業', '光電業', '通信網路業', '電子零組件業', '電子通路業', '資訊服務業', '其他電子業'}
MONTHS = pd.period_range('2007-01', '2026-10', freq='M')

close = pd.read_parquet(DP / 'close_raw.parquet')
value = pd.read_parquet(DP / 'value.parquet').reindex(index=close.index, columns=close.columns)
ret = pd.read_parquet(DP / 'ret.parquet').reindex(index=close.index, columns=close.columns)
# 資料修補（事後發現、非規格變更）：data_panel 缺 2024-01-02～2024-02-14 共 25 個交易日，2024-02-15 的日報酬橫跨缺口。
# 用官方面板快取（同一證交所日 K，2023-12-28 收盤與成交值逐格相同）補這段收盤、成交值，日報酬＝收盤對前一交易日收盤（1～2 月少有除權息）。
z = np.load(SCR / 'cache_is' / 'panel_cs.npz')
OI = pd.DatetimeIndex(z['IDX']); oc = [c for c in close.columns if c in set(z['COLS'])]
OC = pd.DataFrame(z['C'], index=OI, columns=z['COLS'])[oc]; OA = pd.DataFrame(z['AMT'], index=OI, columns=z['COLS'])[oc]
gap = OI[(OI > pd.Timestamp('2023-12-29')) & (OI <= pd.Timestamp('2024-02-15'))]
ORet = OC.ffill().pct_change().where(OC.notna())
new_idx = close.index.union(gap)
close = close.reindex(new_idx); value = value.reindex(new_idx); ret = ret.reindex(new_idx)
g14 = gap[gap < pd.Timestamp('2024-02-15')]
close.loc[g14, oc] = OC.loc[g14].to_numpy(); value.loc[g14, oc] = OA.loc[g14].to_numpy()
ret.loc[gap, oc] = ORet.loc[gap].to_numpy()
DATA_FIX = dict(gap_days_filled=int(len(g14)), ret_days_replaced=int(len(gap)), codes_in_official=len(oc), codes_total=int(close.shape[1]))
del z, OC, OA, ORet
idx = close.index
mper = idx.to_period('M')
last_day = pd.Series(idx, index=idx).groupby(mper).max()          # 每月最後交易日
pos = {d: i for i, d in enumerate(idx)}

# 0050
e = pd.read_csv(E50_CSV, parse_dates=['date']).set_index('date')['adj'].sort_index()
e_me = e.groupby(e.index.to_period('M')).last()
e_r = e.pct_change()

# 成交值與流動性
V = value.where(close.notna())
v60 = V.rolling(60, min_periods=40).mean()
tot = V.sum(axis=1, min_count=1)
codes = close.columns
ind = pd.read_csv(IND_CSV, dtype={'code': str}, parse_dates=['start', 'end'])


def industry_at(d):
    x = ind[(ind['start'] <= d) & (ind['end'].isna() | (ind['end'] >= d))].drop_duplicates('code', keep='last')
    return x.set_index('code')['industry'].reindex(codes)


# 月報酬（每檔：當月日報酬連乘，缺值記 0）
lr = np.log1p(ret.fillna(0.0).clip(lower=-0.99))
mret = np.expm1(lr.groupby(mper).sum())                            # 月×股
listed_m = close.notna().groupby(mper).any()                       # 當月有任何收盤
# 營收
r = pd.read_csv(REV_CSV, dtype={'code': str}, parse_dates=['month'])
r['pri'] = (r['market'] != 'sii').astype(int)
r = r.sort_values(['month', 'code', 'pri']).drop_duplicates(['month', 'code'], keep='first')
Rv = r.pivot(index='month', columns='code', values='revenue_k_twd').sort_index()
Rv.index = pd.PeriodIndex(Rv.index, freq='M')
Rv = Rv.reindex(pd.period_range(Rv.index.min(), Rv.index.max(), freq='M')).reindex(columns=codes)
Rv = Rv.where(Rv > 0)
YOY = np.log(Rv / Rv.shift(12))
YOY[YOY.index.year == 2013] = np.nan

rows = []
for m in MONTHS:
    pm = m - 1
    o = dict(month=str(m))
    if pm in last_day.index:
        d = last_day[pm]; i = pos[d]
        # 大盤
        if (pm - 12) in e_me.index and pm in e_me.index:
            o['MKT_MOM12'] = float(e_me[pm] / e_me[pm - 12] - 1)
        er = e_r.loc[:d].iloc[-63:]
        o['MKT_VOL'] = float(er.std(ddof=1) * np.sqrt(252)) if er.notna().sum() >= 50 else np.nan
        if m in e_me.index and pm in e_me.index:
            o['MKT_RET'] = float(e_me[m] / e_me[pm] - 1)
        # 成交值變化
        if i >= 250:
            o['TURN_CHG'] = float(np.log(tot.iloc[i - 62:i + 1].mean() / tot.iloc[i - 249:i + 1].mean()))
        # 電子權重（成交值）
        if i >= 59:
            vv = V.iloc[i - 59:i + 1].sum()
            g = industry_at(d)
            okc = g.notna() & (g != '未分類') & (vv > 0)
            o['ELEC_W'] = float(vv[okc & g.isin(ELEC)].sum() / vv[okc].sum())
        # 規模分組（流動性代理）→ 月 t 報酬
        if m in mret.index and i >= 60:
            c = close.iloc[i]; l60 = v60.iloc[i]
            elig = c.notna() & (c >= 10) & (l60 >= 1e7)
            rk = l60[elig].sort_values(ascending=False)
            big = rk.index[:50]; small = rk.index[150:]
            rm = mret.loc[m]
            o['SMB_RET'] = float(rm[small].mean() - rm[big].mean())
            o['SMALL_RET'] = float(rm[small].mean()); o['BIG_RET'] = float(rm[big].mean())
            o['n_small'] = int(len(small))
        # 營收離散度：營收月 t−2
        rm2 = m - 2
        if rm2 in YOY.index:
            lst = close.loc[last_day[rm2]] if rm2 in last_day.index else None
            y = YOY.loc[rm2]
            if lst is not None:
                y = y[lst.notna()]
            y = y.dropna()
            if len(y) >= 100:
                lo, hi = y.quantile(0.01), y.quantile(0.99)
                yc = y.clip(lo, hi)
                o['REV_DISP'] = float(yc.std(ddof=1)); o['REV_IQR'] = float(yc.quantile(0.75) - yc.quantile(0.25))
                o['REV_MED'] = float(yc.median()); o['n_rev'] = int(len(y))
    rows.append(o)
E = pd.DataFrame(rows).set_index('month')
E.index = pd.PeriodIndex(E.index, freq='M')
# SMB 12 個月動能（t−12…t−1）
smb = E['SMB_RET']
E['SMB_MOM12'] = np.expm1(np.log1p(smb).rolling(12, min_periods=12).sum().shift(1))

# 策略層：訊號月 → 報酬月（訊號日所在的日曆月）
sig = pd.concat([pd.read_csv(SCR / 'oos_signals.csv'), pd.read_csv(SCR / 'is_signals.csv')])
sig['m'] = pd.PeriodIndex(pd.to_datetime(sig['sig_date']), freq='M')
sig = sig.drop_duplicates('m', keep='last').set_index('m')
E['SURGE_TOP'] = sig['surge_top10'].reindex(E.index)
E['NTGT'] = sig['n_tgt'].reindex(E.index)
E['N_MA'] = sig['n_ma'].reindex(E.index)
dd = pd.concat([pd.read_csv(SCR / 'oos_daily.csv', index_col=0, parse_dates=True),
                pd.read_csv(SCR / 'is_daily.csv', index_col=0, parse_dates=True).loc['2015-07-01':]])
dd = dd[~dd.index.duplicated(keep='last')]
E['W_ACT'] = dd['w_act'].groupby(dd.index.to_period('M')).mean().reindex(E.index)
E['NPOS'] = dd['npos'].groupby(dd.index.to_period('M')).mean().reindex(E.index)
mo = pd.concat([pd.read_csv(SCR / 'oos_monthly.csv', index_col=0), pd.read_csv(SCR / 'is_monthly.csv', index_col=0)])
mo.index = pd.PeriodIndex(mo.index, freq='M')
E = E.join(mo[['strategy', 'b0050', 'excess']])
E.to_csv(HERE / 'env_monthly.csv', float_format='%.10g')
print(E.loc['2008-02':].describe().T.round(4).to_string())
(HERE / 'env_check.json').write_text(json.dumps(dict(data_fix=DATA_FIX, n_rows=int(len(E))), ensure_ascii=False, indent=1))
print('maxrss_mb', round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
