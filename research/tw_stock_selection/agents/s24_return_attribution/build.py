#!/usr/bin/env python3
"""S24 第一步：重跑現行策略三條淨值（FULL、SAT50、SATCASH），自建上市月度因子（SMB、HML、EPF、MOM、RMOM、VW 市場、等權），
輸出 monthly.csv（月報酬）與 build_meta.json（驗收、股池檔數）。規格照 SPEC.json（sha256 必須相符）。
python3 -I -B build.py。單一行程；中間檔放 scratchpad/s24/。"""
import sys, json, hashlib, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd

H = Path(__file__).resolve().parent
AG = H.parent
assert hashlib.sha256((H / 'SPEC.json').read_bytes()).hexdigest() == (H / 'SPEC.sha256').read_text().split()[0], 'SPEC.json 與 SPEC.sha256 不符'
S24 = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s24')
S24.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(AG / 's20_persistent_growth'))
import s20lib                                              # noqa: E402
from s20lib import s17lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402
s20lib.S20 = S24; s17lib.S17 = S24; s10lib.S10 = S24; s9lib.S9 = S24; s8lib.S8 = S24; s7lib.S7 = S24; s6lib.S6 = S24; s5lib.S5 = S24; s4lib.S4 = S24
s3lib.S3 = S24; s2lib.S2 = S24; stlib.S1 = S24; cslib.TMP = S24
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
t_all = time.time()
meta = {}

# ---------------------------------------------------------------- 面板、訊號、三條淨值
P = s20lib.load()
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
F = s17lib.rev_family(P, defs=['STD'])
RT = dict(S=F['S']['STD'], av=F['av'])
Rv = F['Rv']
meta['e50_fix'] = s17lib.e50_fix_suspension(P)
print('載入完成', f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)
S0 = s10lib.make_signal_s10(P, U, RT, shift=0)
S2, ekw = s10lib.variant_S(S0, 'E0')
navs = {}
navs['FULL'] = s10lib.simulate_s10(P, S2, U, sat_w=0.7, sizing='W10', lock=False, cost=COSTS['strict'], capital=1e6, **ekw)[0]
navs['SAT50'] = s10lib.simulate_s10(P, S2, U, sat_w=1.0, sizing='W10', lock=False, cost=COSTS['strict'], capital=7e5, **ekw)[0]
navs['SATCASH'] = s10lib.simulate_s10(P, S2, U, sat_w=1.0, sizing='W10', lock=False, cost=COSTS['strict'], capital=7e5, sweep_min=float('inf'), **ekw)[0]
print('淨值完成', f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)
IDX = pd.DatetimeIndex(P.IDX)
t0 = P.t0
navs = {k: v.reindex(IDX).to_numpy(float) for k, v in navs.items()}   # 引擎回傳 t0 起的 Series → 面板長度陣列
dn = pd.DataFrame({k: v[t0:] for k, v in navs.items()}, index=IDX[t0:])
dn['E50'] = P.E50[t0:]
# 驗收 A1／A2
ref = pd.read_csv(AG / 's19_xq_threshold' / 'navs.csv.gz', parse_dates=['date']).set_index('date')
acc = {}
for mine, theirs in (('FULL', 'E0|W10_70'), ('SATCASH', 'E0|SAT')):
    a = dn[mine].reindex(ref.index); b = ref[theirs]
    rel = float((a / b - 1).abs().max())
    acc[mine] = dict(vs=theirs, max_rel_diff=rel, pass_=bool(rel < 1e-4), n=int(len(b)))
meta['acceptance_nav'] = acc
print('A1/A2', acc, flush=True)

# ---------------------------------------------------------------- 月底列
me_rows = pd.Series(np.arange(P.T), index=IDX).groupby(IDX.to_period('M')).max()
me_rows = me_rows[me_rows.index >= pd.Period('2015-01', 'M')]
months = list(me_rows.index)
mi = {m: i for i, m in enumerate(months)}

# 策略與 0050 月報酬（2015-07 起；第一個月用 t0 起算）
mrows = me_rows[me_rows.index >= pd.Period('2015-07', 'M')]
out = pd.DataFrame(index=mrows.index)
for k in ('FULL', 'SAT50', 'SATCASH'):
    v = navs[k]; prev = np.r_[v[t0], v[mrows.values[:-1]]]
    out['R_' + k] = v[mrows.values] / prev - 1
v = P.E50; prev = np.r_[v[t0], v[mrows.values[:-1]]]
out['R_0050'] = v[mrows.values] / prev - 1
out['last_date'] = IDX[mrows.values].strftime('%Y-%m-%d')

# ---------------------------------------------------------------- PIT 市值（同 s12lib.mcap_panel，股數落後 63 交易日）
sh = pd.read_parquet(AG / 'o_lc_sector_rotation_pit' / 'data' / 'shares_full.parquet')
sh = sh.reindex(sh.index.union(P.IDX)).ffill().reindex(P.IDX).reindex(columns=P.COLS).to_numpy(np.float64)
now = sh * P.CF
LAG = 63
MC = np.full((P.T, P.N), np.nan)
MC[LAG:] = now[:-LAG] * (P.G[LAG:] / P.G[:-LAG])
MC = np.where(np.isfinite(MC), MC, now)
del sh, now
gc.collect()

# ---------------------------------------------------------------- 財報：B/M、E/P（PIT 可得日）
PEAD = AG / 'o_pead_margin_surprise_low_attention' / 'data'
FIN_IND = {'bd', 'basi', 'ins', 'fh', 'mim'}
f = pd.read_parquet(PEAD / 'fin_long.parquet', columns=['code', 'y', 'q', 'industry', 'nip_s', 'equity_parent'])
f = f[f['code'].str.fullmatch(r'[1-9]\d{3}')].copy()
f['kq'] = f['y'] * 4 + f['q'] - 1
a = pd.read_parquet(PEAD / 'avail_dates.parquet', columns=['code', 'y', 'q', 'A'])
f = f.merge(a, on=['code', 'y', 'q'], how='left')
dl = {1: (5, 15), 2: (8, 14), 3: (11, 14)}
f['deadline'] = [pd.Timestamp(y + (q == 4), *(dl[q] if q < 4 else (3, 31))) for y, q in zip(f['y'], f['q'])]
f['avail'] = pd.to_datetime(f['A']).fillna(f['deadline'])
f = f.sort_values(['code', 'kq'])
# TTM 淨利：kq−3…kq 四季都有值且連續
g = f.groupby('code')
f['ttm'] = g['nip_s'].transform(lambda s: s.rolling(4, min_periods=4).sum())
f['kq_lag3'] = g['kq'].shift(3)
f.loc[f['kq'] - f['kq_lag3'] != 3, 'ttm'] = np.nan
ind_last = f.groupby('code')['industry'].last()
fin_codes = set(ind_last[ind_last.isin(FIN_IND)].index)
nonfin = np.array([c not in fin_codes for c in P.COLS])
meta['fin_codes_excluded_from_value'] = len(fin_codes)
ci = {c: j for j, c in enumerate(P.COLS)}
BE = np.full((len(months), P.N), np.nan); NI = np.full((len(months), P.N), np.nan)
fv = f[f['code'].isin(ci)]
for m, i in mi.items():
    d = IDX[me_rows[m]]
    x = fv[fv['avail'] <= d]
    if not len(x):
        continue
    last = x.groupby('code').last()               # 依 kq 排序後最後一筆＝最近一季已可得
    js = np.array([ci[c] for c in last.index])
    BE[i, js] = last['equity_parent'].to_numpy(float)
    NI[i, js] = last['ttm'].to_numpy(float)
del fv, f
# 營收年增率（PIT）
av = cslib.rev_avail_days(P, list(Rv.index), True)
YOY = (Rv / Rv.shift(12) - 1).replace([np.inf, -np.inf], np.nan)
YOY[YOY.index.year == 2013] = np.nan
YOYa = YOY.to_numpy(float)
RY = np.full((len(months), P.N), np.nan)
avk = np.array([av[rm] for rm in YOY.index])
for m, i in mi.items():
    t = me_rows[m]
    ok = np.flatnonzero(avk <= t)
    if not len(ok):
        continue
    RY[i] = YOYa[ok[-1]]                                    # 最近一個可得月份（所有公司同一個營收月份）
meta['rmom_note'] = '形成日用最近一個已可得營收月份（全部公司同一月份，通常是 m−1）的年增率'

# ---------------------------------------------------------------- 月度因子
Gme = P.G[me_rows.values]                           # 月底含息指數
INN = INNOV[me_rows.values]
uni_rows = []
fac = {k: [] for k in ('SMB_HML', 'SMB_MOM', 'SMB_RMOM', 'SMB_EP', 'HML', 'EPF', 'MOM', 'RMOM', 'MKT_VW', 'EW_ALL', 'EW_POOL')}
corner_min = {k: [] for k in ('HML', 'EPF', 'MOM', 'RMOM')}
fmonths = []


def sort23(mc, ch, ret, elig):
    """回傳 (smb, hml型, 最少檔數)。mc 中位數切大小（在 elig 股池內）、ch 30/70 切三組，市值加權。"""
    e = elig & np.isfinite(ch)
    if e.sum() < 30:
        return np.nan, np.nan, 0
    med = np.median(mc[elig])                       # 市值中位數：用該月整個股池（FF 用 NYSE 中位數；這裡用全股池）
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
    r = {k: v[0] for k, v in grp.items()}
    smb = (r['SL'] + r['SM'] + r['SH']) / 3 - (r['BL'] + r['BM'] + r['BH']) / 3
    hml = (r['SH'] + r['BH']) / 2 - (r['SL'] + r['BL']) / 2
    return smb, hml, nmin


for i in range(len(months) - 1):
    m = months[i]; m1 = months[i + 1]
    if m1 < pd.Period('2015-07', 'M'):
        continue
    t = me_rows[m]
    mc = MC[t]
    elig = (P.MK[t] == 1) & P.has[t] & P.not91 & ~INN[i] & np.isfinite(mc) & (mc > 0)
    ret = Gme[i + 1] / Gme[i] - 1
    elig &= np.isfinite(ret)
    bm = np.where(nonfin & (BE[i] > 0), BE[i] / mc, np.nan)
    ep = np.where(nonfin & (NI[i] > 0), NI[i] / mc, np.nan)
    mom = (Gme[i - 1] / Gme[i - 12] - 1) if i >= 12 else np.full(P.N, np.nan)
    ry = RY[i]
    s1, h, n1 = sort23(mc, bm, ret, elig)
    s2_, mo, n2 = sort23(mc, mom, ret, elig)
    s3_, rm_, n3 = sort23(mc, ry, ret, elig)
    s4_, epf, n4 = sort23(mc, ep, ret, elig)
    fac['SMB_HML'].append(s1); fac['HML'].append(h); fac['SMB_MOM'].append(s2_); fac['MOM'].append(mo)
    fac['SMB_RMOM'].append(s3_); fac['RMOM'].append(rm_); fac['SMB_EP'].append(s4_); fac['EPF'].append(epf)
    for k, n in (('HML', n1), ('MOM', n2), ('RMOM', n3), ('EPF', n4)):
        corner_min[k].append(n)
    w = mc[elig]
    fac['MKT_VW'].append(float(np.sum(w * ret[elig]) / np.sum(w)))
    fac['EW_ALL'].append(float(np.mean(ret[elig])))
    pool = U[t] & np.isfinite(ret)
    fac['EW_POOL'].append(float(np.mean(ret[pool])))
    uni_rows.append(dict(month=str(m1), n_elig=int(elig.sum()), n_bm=int((elig & np.isfinite(bm)).sum()), n_ep=int((elig & np.isfinite(ep)).sum()),
                         n_mom=int((elig & np.isfinite(mom)).sum()), n_rmom=int((elig & np.isfinite(ry)).sum()), n_pool=int(pool.sum())))
    fmonths.append(m1)
FD = pd.DataFrame(fac, index=pd.PeriodIndex(fmonths, freq='M'))
FD['SMB'] = FD[['SMB_HML', 'SMB_MOM', 'SMB_RMOM']].mean(axis=1, skipna=False)
out = out.join(FD)
out.index = out.index.astype(str)
out.index.name = 'month'
out.to_csv(H / 'monthly.csv', float_format='%.8f')
ud = pd.DataFrame(uni_rows)
meta['universe_counts'] = {c: dict(min=int(ud[c].min()), median=float(ud[c].median()), max=int(ud[c].max())) for c in ud.columns if c != 'month'}
meta['corner_min'] = {k: dict(min=int(np.min([x for x in v if x > 0])) if any(x > 0 for x in v) else 0,
                              n_missing=int(np.sum(~np.isfinite(FD[k].to_numpy())))) for k, v in corner_min.items()}
meta['factor_first_valid'] = {k: str(FD[k].first_valid_index()) for k in FD.columns}
meta['runtime_s'] = round(time.time() - t_all); meta['peak_rss_mb'] = rss()
(H / 'build_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str))
print(json.dumps({k: meta[k] for k in ('universe_counts', 'corner_min', 'factor_first_valid', 'runtime_s', 'peak_rss_mb')}, ensure_ascii=False, indent=1))
