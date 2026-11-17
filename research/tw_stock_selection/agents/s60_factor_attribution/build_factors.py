#!/usr/bin/env python3
"""S60 第一步：自建上市月頻因子 2008-01～2026-10（形成日＝每月最後交易日，報酬＝下一個月），輸出 factors_monthly.csv、build_meta.json。
規格照 SPEC.json（sha256 必須相符）。python3 -I -B build_factors.py。只讀本地公開資料；不載入共用引擎。"""
import json, hashlib, time, resource
from pathlib import Path
import numpy as np, pandas as pd

H = Path(__file__).resolve().parent
ROOT = H.parent.parent                      # research/tw_stock_selection
AG = H.parent
assert hashlib.sha256((H / 'SPEC.json').read_bytes()).hexdigest() == (H / 'SPEC.sha256').read_text().split()[0], 'SPEC 不符'
PAN = ROOT / 'data_panel' / 'panel'
REV_CSV = Path('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv')
PEAD = AG / 'o_pead_margin_surprise_low_attention' / 'data'
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
t_all = time.time()
meta = {}

adj = pd.read_parquet(PAN / 'adj.parquet')
C = pd.read_parquet(PAN / 'close_raw.parquet').reindex(index=adj.index, columns=adj.columns)
COLS = np.array(adj.columns.astype(str))
IDX = adj.index
keep = np.array([c[:2] != '91' for c in COLS])
meta['n_codes'] = int(len(COLS)); meta['n_91_excluded'] = int((~keep).sum())
A = adj.ffill().to_numpy(float)              # 下市後停在最後價（下月報酬算到最後交易日）
Cn = C.to_numpy(float)
recent = C.notna().rolling(5, min_periods=1).max().to_numpy(bool)   # 前 5 個交易日內有收盤
del adj
T, N = A.shape

# 市值（同 S24：股數 t−63 × 收盤 t−63 × adj 漲跌）
sh = pd.read_parquet(AG / 'o_lc_sector_rotation_pit' / 'data' / 'shares_full.parquet').reindex(columns=COLS)
sh = sh.reindex(sh.index.union(IDX)).ffill().reindex(IDX).to_numpy(float)
Cf = C.ffill().to_numpy(float)
now = sh * Cf
LAG = 63
MC = np.full((T, N), np.nan)
MC[LAG:] = now[:-LAG] * (A[LAG:] / A[:-LAG])
MC = np.where(np.isfinite(MC), MC, now)
del sh, Cf, now, C

me = pd.Series(np.arange(T), index=IDX).groupby(IDX.to_period('M')).max()
me = me[me.index >= pd.Period('2006-01', 'M')]
months = list(me.index); mi = {m: i for i, m in enumerate(months)}
Ame = A[me.values]

# 營收年增（營收月份 m−1）
r = pd.read_csv(REV_CSV, dtype={'code': str}, parse_dates=['month'])
r['pri'] = (r['market'] != 'sii').astype(int)
r = r.sort_values(['month', 'code', 'pri']).drop_duplicates(['month', 'code'], keep='first')
Rv = r.pivot(index='month', columns='code', values='revenue_k_twd').sort_index()
Rv.index = pd.PeriodIndex(Rv.index, freq='M')
Rv = Rv.where(Rv > 0)
YOY = (Rv / Rv.shift(12, freq='M').reindex(Rv.index) - 1).replace([np.inf, -np.inf], np.nan)
YOY[YOY.index.year == 2013] = np.nan
YOY = YOY.reindex(columns=COLS)
del r, Rv

# 財報（PIT 可得日）
FIN_IND = {'bd', 'basi', 'ins', 'fh', 'mim'}
f = pd.read_parquet(PEAD / 'fin_long.parquet', columns=['code', 'y', 'q', 'industry', 'nip_s', 'opi_s', 'equity_parent', 'assets'])
f = f[f['code'].str.fullmatch(r'[1-9]\d{3}')].copy()
f['kq'] = f['y'] * 4 + f['q'] - 1
a = pd.read_parquet(PEAD / 'avail_dates.parquet', columns=['code', 'y', 'q', 'A'])
f = f.merge(a, on=['code', 'y', 'q'], how='left')
dl = {1: (5, 15), 2: (8, 14), 3: (11, 14)}
f['deadline'] = [pd.Timestamp(y + (q == 4), *(dl[q] if q < 4 else (3, 31))) for y, q in zip(f['y'], f['q'])]
f['avail'] = pd.to_datetime(f['A']).fillna(f['deadline'])
f = f.sort_values(['code', 'kq']).reset_index(drop=True)
g = f.groupby('code')
for src, dst in (('nip_s', 'ttm_ni'), ('opi_s', 'ttm_op')):
    f[dst] = g[src].transform(lambda s: s.rolling(4, min_periods=4).sum())
f['kq_l3'] = g['kq'].shift(3); f['kq_l4'] = g['kq'].shift(4); f['as_l4'] = g['assets'].shift(4)
f.loc[f['kq'] - f['kq_l3'] != 3, ['ttm_ni', 'ttm_op']] = np.nan
f['inv'] = np.where((f['kq'] - f['kq_l4'] == 4) & (f['as_l4'] > 0), f['assets'] / f['as_l4'] - 1, np.nan)
ind_last = f.groupby('code')['industry'].last()
fin_codes = set(ind_last[ind_last.isin(FIN_IND)].index)
nonfin = np.array([c not in fin_codes for c in COLS])
meta['fin_codes_excluded_from_value'] = int(len(fin_codes))
ci = {c: j for j, c in enumerate(COLS)}
fv = f[f['code'].isin(ci)]
BE, NI, OP, INV = (np.full((len(months), N), np.nan) for _ in range(4))
for m, i in mi.items():
    if m < pd.Period('2013-06', 'M'):
        continue
    d = IDX[me[m]]
    x = fv[fv['avail'] <= d]
    if not len(x):
        continue
    last = x.groupby('code').last()
    js = np.array([ci[c] for c in last.index])
    BE[i, js] = last['equity_parent'].to_numpy(float); NI[i, js] = last['ttm_ni'].to_numpy(float)
    OP[i, js] = last['ttm_op'].to_numpy(float); INV[i, js] = last['inv'].to_numpy(float)
del f, fv, g


def sort23(mc, ch, ret, elig, high_minus_low=True):
    e = elig & np.isfinite(ch)
    if e.sum() < 30:
        return np.nan, np.nan, 0
    med = np.median(mc[elig])
    lo, hi = np.quantile(ch[e], [0.3, 0.7])
    small = mc <= med
    grp = {}
    for sz, sm in (('S', small), ('B', ~small)):
        for cg, cm in (('L', ch <= lo), ('M', (ch > lo) & (ch < hi)), ('H', ch >= hi)):
            msk = e & sm & cm
            w = mc[msk]
            grp[sz + cg] = (float(np.sum(w * ret[msk]) / np.sum(w)) if msk.sum() else np.nan, int(msk.sum()))
    nmin = min(v[1] for v in grp.values())
    if nmin < 5:
        return np.nan, np.nan, nmin
    q = {k: v[0] for k, v in grp.items()}
    smb = (q['SL'] + q['SM'] + q['SH']) / 3 - (q['BL'] + q['BM'] + q['BH']) / 3
    hml = (q['SH'] + q['BH']) / 2 - (q['SL'] + q['BL']) / 2
    return smb, (hml if high_minus_low else -hml), nmin


keys = ('SMB_HML', 'SMB_MOM', 'SMB_RMOM', 'SMB_RMW', 'SMB_CMA', 'SMB_EP', 'HML', 'MOM', 'RMOM', 'RMW', 'CMA', 'EPF', 'MKT_VW_RAW', 'EW_RAW')
fac = {k: [] for k in keys}
cmin = {k: [] for k in ('HML', 'MOM', 'RMOM', 'RMW', 'CMA', 'EPF')}
urows, fm = [], []
for i in range(len(months) - 1):
    m, m1 = months[i], months[i + 1]
    if m1 < pd.Period('2008-01', 'M'):
        continue
    t = me[m]
    mc = MC[t]
    ret = Ame[i + 1] / Ame[i] - 1
    elig = keep & recent[t] & np.isfinite(mc) & (mc > 0) & np.isfinite(ret)
    mom = Ame[i - 1] / Ame[i - 12] - 1 if i >= 12 else np.full(N, np.nan)
    rm = m - 1
    ry = YOY.loc[rm].to_numpy(float) if rm in YOY.index else np.full(N, np.nan)
    bm = np.where(nonfin & (BE[i] > 0), BE[i] / mc, np.nan)
    ep = np.where(nonfin & (NI[i] > 0), NI[i] / mc, np.nan)
    op = np.where(nonfin & (BE[i] > 0), OP[i] / BE[i], np.nan)
    inv = np.where(nonfin, INV[i], np.nan)
    res = {}
    for nm, ch, hml in (('HML', bm, True), ('MOM', mom, True), ('RMOM', ry, True), ('RMW', op, True), ('CMA', inv, False), ('EPF', ep, True)):
        s_, h_, n_ = sort23(mc, ch, ret, elig, hml)
        res[nm] = (s_, h_); cmin[nm].append(n_)
    fac['SMB_HML'].append(res['HML'][0]); fac['SMB_MOM'].append(res['MOM'][0]); fac['SMB_RMOM'].append(res['RMOM'][0])
    fac['SMB_RMW'].append(res['RMW'][0]); fac['SMB_CMA'].append(res['CMA'][0]); fac['SMB_EP'].append(res['EPF'][0])
    for nm in ('HML', 'MOM', 'RMOM', 'RMW', 'CMA', 'EPF'):
        fac[nm].append(res[nm][1])
    w = mc[elig]
    fac['MKT_VW_RAW'].append(float(np.sum(w * ret[elig]) / np.sum(w)))
    fac['EW_RAW'].append(float(np.mean(ret[elig])))
    urows.append(dict(month=str(m1), n_elig=int(elig.sum()), n_bm=int((elig & np.isfinite(bm)).sum()), n_mom=int((elig & np.isfinite(mom)).sum()),
                      n_rmom=int((elig & np.isfinite(ry)).sum()), n_op=int((elig & np.isfinite(op)).sum()), n_inv=int((elig & np.isfinite(inv)).sum())))
    fm.append(m1)
FD = pd.DataFrame(fac, index=pd.PeriodIndex(fm, freq='M'))
FD['SMB'] = FD[['SMB_HML', 'SMB_MOM', 'SMB_RMOM']].mean(axis=1, skipna=False)
FD['SMB_MR'] = FD[['SMB_MOM', 'SMB_RMOM']].mean(axis=1, skipna=True)
FD['SMB_FF5'] = FD[['SMB_HML', 'SMB_RMW', 'SMB_CMA']].mean(axis=1, skipna=False)

# 策略與 0050
rf = 1.01 ** (1 / 12) - 1
nv = pd.read_csv(ROOT.parent.parent / 'xq_scripts' / 'site' / 'navs_official.csv.gz', parse_dates=['date']).set_index('date')
meta['official_nav'] = dict(start=str(nv.index[0].date()), end=str(nv.index[-1].date()), cols=list(nv.columns))
mend = nv.groupby(nv.index.to_period('M')).tail(1)
base = nv.iloc[[0]]
mm = pd.concat([base, mend])
mr = (mm / mm.shift(1) - 1).iloc[1:]
mr.index = mr.index.to_period('M')
off = pd.DataFrame({'R_E0': mr['E0|W10_70'], 'R_XT': mr['XT|W10_70'], 'R_0050': mr['0050']})
# data_panel 缺 2024-01-02～02-14（無任何來源），因子 2024-02 是兩個月報酬 → 策略同樣把缺的月份複利併入下一個月
gap = [p for p in off.index if p not in set(FD.index)]
meta['merged_gap_months'] = [str(p) for p in gap]
for p in gap:
    nxt = p + 1
    off.loc[nxt] = (1 + off.loc[p]) * (1 + off.loc[nxt]) - 1
off = off.drop(index=gap)
off['R_SATA'] = (off['R_E0'] - 0.3 * off['R_0050']) / 0.7
pre = pd.read_csv(AG / 's55_pre2015_oos' / 'oos_monthly.csv', index_col=0)
pre.index = pd.PeriodIndex(pre.index, freq='M')
pre = pd.DataFrame({'R_E0': pre['strategy'], 'R_0050': pre['b0050']})
pre['src'] = 's55_pre2015'; off['src'] = 'official'
S = pd.concat([pre, off]).sort_index()
out = FD.join(S, how='left')
out['RF'] = rf
out['MKT'] = out['R_0050'] - rf
out['MKT_VW'] = out['MKT_VW_RAW'] - rf
out.index = out.index.astype(str); out.index.name = 'month'
cols = ['src', 'RF', 'R_E0', 'R_XT', 'R_SATA', 'R_0050', 'MKT', 'MKT_VW', 'SMB', 'SMB_MR', 'SMB_FF5', 'HML', 'MOM', 'RMOM', 'RMW', 'CMA', 'EPF',
        'SMB_HML', 'SMB_MOM', 'SMB_RMOM', 'SMB_RMW', 'SMB_CMA', 'SMB_EP', 'EW_RAW', 'MKT_VW_RAW']
out[cols].to_csv(H / 'factors_monthly.csv', float_format='%.8f')
ud = pd.DataFrame(urows)
ud['y'] = ud['month'].str[:4]
meta['universe_counts_by_year_median'] = ud.drop(columns='month').groupby('y').median().astype(int).to_dict(orient='index')
meta['corner_min'] = {k: dict(min_pos=int(min([x for x in v if x > 0], default=0)), n_missing=int(np.sum(~np.isfinite(FD[k].to_numpy())))) for k, v in cmin.items()}
meta['factor_first_valid'] = {k: str(FD[k].first_valid_index()) for k in FD.columns}
meta['factor_missing_months'] = {k: [str(x) for x in FD.index[FD[k].isna()]] for k in ('RMOM', 'HML', 'RMW', 'CMA') if FD[k].isna().any() and FD[k].isna().sum() < 120}
meta['runtime_s'] = round(time.time() - t_all); meta['peak_rss_mb'] = rss()
(H / 'build_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str))
print(json.dumps({k: meta[k] for k in ('factor_first_valid', 'corner_min', 'runtime_s', 'peak_rss_mb')}, ensure_ascii=False, indent=1))
