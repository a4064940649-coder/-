#!/usr/bin/env python3
"""B1 交易日曆、B2 個股日 K 涵蓋、B3 價格合理性。python3 -I -B check_prices.py → parts/prices.json、tables/*.csv。只讀。"""
import sys; from pathlib import Path as _P; sys.path.insert(0, str(_P(__file__).resolve().parent))
import json, time, resource
import numpy as np, pandas as pd
from common import *                                    # noqa: E402

t0 = time.time()
(HERE / 'parts').mkdir(exist_ok=True); (HERE / 'tables').mkdir(exist_ok=True)
R = {}
z = np.load(TMP / 'off.npz', allow_pickle=False)
IDX = pd.DatetimeIndex(z['IDX']); COLS = z['COLS'].astype(str); MK = z['MK']
O, H, L, C, VOL, AMT, CHG = (z[k] for k in ['O', 'H', 'L', 'C', 'VOL', 'AMT', 'CHG'])
T, N = C.shape

# ---- pre2015（封存長表）
d = pd.read_csv(PRE, dtype={'code': str}); d['date'] = pd.to_datetime(d['date'])
PIDX_all = pd.DatetimeIndex(sorted(d['date'].unique()))
SEAM = pd.Timestamp('2015-01-05')

# ============================================================ B1 交易日曆
cal = PIDX_all[PIDX_all < SEAM].append(IDX)
assert cal.is_monotonic_increasing and cal.is_unique
def rd_dates(p, col='date'):
    x = pd.read_csv(p, parse_dates=[col]); return x
tw_eng = rd_dates(TWII_CSV); tw_fwd = rd_dates(FWD / 'raw/finlab/taiex_twii.csv')
e_eng = rd_dates(E50_CSV); e_fwd = rd_dates(FWD / 'raw/finlab/etf_0050.csv')
def cmp(calx, other, a, b, name):
    c = calx[(calx >= a) & (calx <= b)]; o = pd.DatetimeIndex(other)[(pd.DatetimeIndex(other) >= a) & (pd.DatetimeIndex(other) <= b)]
    miss = c.difference(o); extra = o.difference(c)
    return dict(ref=name, window=[str(a.date()), str(b.date())], n_panel=len(c), n_ref=len(o), n_in_panel_not_ref=len(miss), n_in_ref_not_panel=len(extra),
                in_panel_not_ref=[str(x.date()) for x in miss], in_ref_not_panel=[str(x.date()) for x in extra])
A0 = pd.Timestamp('2007-01-01')
b1 = dict(calendar=dict(first=str(cal[0].date()), last=str(cal[-1].date()), n=len(cal), n_pre=int((cal < SEAM).sum()), n_official=len(IDX),
                        weekend_days=[str(x.date()) for x in cal[cal.dayofweek >= 5]]))
b1['vs_twii_engine_file'] = cmp(cal, tw_eng['date'], A0, tw_eng['date'].max(), 'jonatw taiex_twii.csv（引擎讀的那一份，到 %s）' % tw_eng['date'].max().date())
b1['vs_twii_forward_copy'] = cmp(cal, tw_fwd['date'], A0, cal[-1], 'o_forward_infra_live/raw/finlab/taiex_twii.csv（每日更新）')
b1['vs_0050_engine_file'] = cmp(cal, e_eng['date'], A0, e_eng['date'].max(), 'jonatw etf_0050.csv（引擎讀的那一份）')
b1['vs_0050_forward_copy'] = cmp(cal, e_fwd['date'], A0, cal[-1], 'o_forward_infra_live/raw/finlab/etf_0050.csv（每日更新）')
b1['twii_nan_close'] = int(tw_fwd.loc[tw_fwd['date'] >= A0, 'close'].isna().sum()); b1['e50_nan_adj'] = int(e_fwd.loc[e_fwd['date'] >= A0, 'adj'].isna().sum())
# 兩份 TWII／0050 重疊期是否同值（上游有沒有改歷史）
m = tw_eng.merge(tw_fwd, on='date', suffixes=('_a', '_b')); b1['twii_two_copies'] = dict(n=len(m), n_close_diff=int((np.abs(m['close_a'] - m['close_b']) > 1e-6).sum()))
m = e_eng.merge(e_fwd, on='date', suffixes=('_a', '_b'))
rr = (m['adj_a'] / m['adj_b']); b1['e50_two_copies'] = dict(n=len(m), n_adj_diff=int((np.abs(m['adj_a'] - m['adj_b']) > 1e-6).sum()),
    ratio_min=float(rr.min()), ratio_max=float(rr.max()), n_ret_diff_gt_1bp=int((np.abs(m['adj_a'].pct_change() - m['adj_b'].pct_change()) > 1e-4).sum()))
# data_panel
dpc = pd.read_parquet(DP / 'close_raw.parquet')
dpi = dpc.index[(dpc.index >= A0)]
b1['vs_data_panel'] = cmp(cal, dpi, A0, dpi.max(), 'data_panel/panel/close_raw.parquet（舊研究面板；S75 只拿來補 2015 前的格）')
gap = pd.DatetimeIndex([pd.Timestamp(x) for x in b1['vs_data_panel']['in_panel_not_ref']])
b1['data_panel_gap_2024'] = dict(n=int(((gap >= '2024-01-02') & (gap <= '2024-02-14')).sum()), official_has_all=True,
                                official_days_2024_01_02_to_02_14=int(((IDX >= '2024-01-02') & (IDX <= '2024-02-14')).sum()))
# 引擎快取（暫存區）vs 從封存重建
if (ENG_CACHE / 'panel_cs.npz').exists():
    q = np.load(ENG_CACHE / 'panel_cs.npz', allow_pickle=False)
    qi = pd.DatetimeIndex(q['IDX']); qc = q['COLS'].astype(str)
    tt = np.flatnonzero(IDX.isin(qi)); jj = np.array([np.flatnonzero(COLS == c)[0] for c in qc if c in set(COLS)])
    eq = {}
    same_shape = len(tt) == len(qi) and len(jj) == len(qc)
    if same_shape:
        for k, arr in (('C', C), ('O', O), ('AMT', AMT), ('CHG', CHG), ('VOL', VOL), ('H', H), ('L', L)):
            a = arr[np.ix_(tt, jj)]; b = q[k]
            eq[k] = int((~((a == b) | (np.isnan(a) & np.isnan(b)))).sum())
        eq['MK'] = int((MK[np.ix_(tt, jj)] != q['MK']).sum())
    b1['engine_cache_vs_rebuild'] = dict(path=str(ENG_CACHE / 'panel_cs.npz'), sha256=sha_file(ENG_CACHE / 'panel_cs.npz'), T=len(qi), N=len(qc),
        first=str(qi[0].date()), last=str(qi[-1].date()), all_dates_in_rebuild=bool(same_shape), n_cells_diff=eq,
        codes_in_rebuild_not_cache=sorted(set(COLS) - set(qc)))
    del q
# pre 檔 2015 上半年日子 vs 官方
p15 = PIDX_all[PIDX_all >= SEAM]; o15 = IDX[IDX <= p15.max()]
b1['pre_file_2015H1_vs_official'] = dict(n_pre=len(p15), n_off=len(o15), sym_diff=[str(x.date()) for x in p15.symmetric_difference(o15)])
R['B1_calendar'] = b1
print('B1', {k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if not isinstance(vv, list) or len(vv) < 12}) for k, v in b1.items()}, flush=True)

# ============================================================ B2 涵蓋
has = np.isfinite(C) & (C > 0)
tw = MK == 1; ot = MK == 2
daily = pd.DataFrame(dict(date=IDX, n_listed_twse=tw.sum(1), n_priced_twse=(tw & has).sum(1), n_listed_tpex=ot.sum(1), n_priced_tpex=(ot & has).sum(1)))
pre = d[d['date'] < SEAM]
pg = pre.groupby('date'); pre_daily = pd.DataFrame(dict(n_listed_twse=pg.size(), n_priced_twse=pg['close'].apply(lambda s: int((s > 0).sum())))).reset_index()
pre_daily['n_listed_tpex'] = 0; pre_daily['n_priced_tpex'] = 0
dall = pd.concat([pre_daily, daily], ignore_index=True)
dall.to_csv(HERE / 'tables/daily_counts.csv', index=False)
def drops(s, dates, label):
    s = pd.Series(s.to_numpy(float), index=pd.DatetimeIndex(dates))
    med = s.rolling(20, min_periods=5).median().shift(1)
    bad = s[(s < med * 0.97)]
    return [dict(date=str(k.date()), n=int(v), prev20_median=float(med[k]), pct=round(float(v / med[k] - 1) * 100, 2), series=label) for k, v in bad.items()]
b2 = dict(drop_rule='當日檔數 < 前 20 個交易日中位數 × 0.97',
          drops_listed_twse=drops(dall['n_listed_twse'], dall['date'], '上市有列'), drops_priced_twse=drops(dall['n_priced_twse'], dall['date'], '上市有收盤價'),
          drops_listed_tpex=drops(daily['n_listed_tpex'], daily['date'], '上櫃有列'), drops_priced_tpex=drops(daily['n_priced_tpex'], daily['date'], '上櫃有收盤價'))
yr = {}
py = pre['date'].dt.year
for y in sorted(py.unique()):
    x = pre[py == y]; trd = x['volume'] > 0
    yr[int(y)] = dict(src='raw/pre2015（Asoul/tsec）', n_codes_twse=int(x['code'].nunique()), median_daily_rows=float(x.groupby('date').size().median()),
        median_daily_priced=float(x[x['close'] > 0].groupby('date').size().median()), n_rows=int(len(x)),
        miss_close_pct=float((~(x['close'] > 0)).mean() * 100), miss_open_pct=float((~(x['open'] > 0)).mean() * 100), miss_amount_pct=float(x['amount'].isna().mean() * 100),
        zero_amount_pct=float((x['amount'] == 0).mean() * 100), miss_change_pct=float(x['change'].isna().mean() * 100),
        traded_rows=int(trd.sum()), traded_miss_close_pct=float((~(x.loc[trd, 'close'] > 0)).mean() * 100), traded_miss_open_pct=float((~(x.loc[trd, 'open'] > 0)).mean() * 100),
        traded_miss_change_pct=float(x.loc[trd, 'change'].isna().mean() * 100), xflag_rows=int(x['xflag'].sum()))
oy = IDX.year
for y in sorted(set(oy)):
    r = oy == y; m = tw[r]; n = int(m.sum())
    c, o, a, ch, v = C[r][m], O[r][m], AMT[r][m], CHG[r][m], VOL[r][m]
    trd = v > 0
    yr.setdefault(int(y), {}).update({'official_' + k: vv for k, vv in dict(n_codes_twse=int(m.any(0).sum()), n_codes_tpex=int(ot[r].any(0).sum()),
        median_daily_rows=float(np.median(m.sum(1))), median_daily_priced=float(np.median((m & has[r]).sum(1))), n_rows=n,
        miss_close_pct=float((~(c > 0)).mean() * 100), miss_open_pct=float((~(o > 0)).mean() * 100), miss_amount_pct=float(np.isnan(a).mean() * 100),
        zero_amount_pct=float((a == 0).mean() * 100), miss_change_pct=float(np.isnan(ch).mean() * 100), traded_rows=int(trd.sum()),
        traded_miss_close_pct=float((~(c[trd] > 0)).mean() * 100), traded_miss_open_pct=float((~(o[trd] > 0)).mean() * 100),
        traded_miss_change_pct=float(np.isnan(ch[trd]).mean() * 100), miss_volume_pct=float(np.isnan(v).mean() * 100)).items()})
b2['by_year'] = yr
R['B2_coverage'] = b2
print('B2 drops', {k: len(v) for k, v in b2.items() if isinstance(v, list)}, flush=True)
for y, v in yr.items():
    print(y, {k.replace('official_', 'o_'): (round(vv, 3) if isinstance(vv, float) else vv) for k, vv in v.items() if k != 'src'}, flush=True)

# ============================================================ B3 價格合理性
E = pd.read_csv(TMP / 'events.csv', dtype=str, low_memory=False)
E = E[E['sub'] != 'stophalt']
E['d'] = pd.to_datetime(E['date'].fillna(E['file_date']), errors='coerce').fillna(pd.to_datetime(E['file_date']))
evset = set(zip(E['d'].dt.strftime('%Y-%m-%d'), E['stock_id']))
tw49 = pd.concat([pd.read_csv(f, dtype={'code': str}) for f in sorted(TWT49U.glob('*/*.csv.gz'))], ignore_index=True)
evset |= set(zip(tw49['date'].astype(str).str[:10], tw49['code']))
ev_unread_tse = set(json.loads((TMP / 'off_meta.json').read_text())['events']['exright']['unreadable_days'])
ev_unread_otc = set(json.loads((TMP / 'off_meta.json').read_text())['events']['otcexright']['unreadable_days'])

def limit_of(dates):
    return np.where(pd.DatetimeIndex(dates) < pd.Timestamp('2015-06-01'), 0.07, 0.10)

def scan(IDXx, COLSx, Cx, CHGx, MKx, VOLx, mkval, exmask, tag):
    hs = np.isfinite(Cx) & (Cx > 0) & (MKx == mkval)
    CF = pd.DataFrame(np.where(np.isfinite(Cx) & (Cx > 0), Cx, np.nan)).ffill().to_numpy()
    prevC = np.vstack([np.full((1, Cx.shape[1]), np.nan), CF[:-1]])
    prev_tr = np.vstack([np.zeros((1, Cx.shape[1]), bool), (np.isfinite(Cx) & (Cx > 0))[:-1]])
    nbar = np.cumsum(MKx > 0, axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        ret = Cx / prevC - 1
        ref = Cx - CHGx; rret = Cx / np.where(ref > 0, ref, np.nan) - 1
    lim = limit_of(IDXx)[:, None]
    big = hs & np.isfinite(ret) & (np.abs(ret) > lim + 1e-4)
    first5 = nbar <= 5
    resume = ~prev_tr
    cat = np.where(exmask, 'exright', np.where(first5, 'first5', np.where(resume, 'resume', 'unexplained')))
    ti, ji = np.nonzero(big)
    df = pd.DataFrame(dict(date=IDXx[ti], code=COLSx[ji], prev_close=prevC[ti, ji], close=Cx[ti, ji], ret_pct=ret[ti, ji] * 100, change=CHGx[ti, ji],
                           ret_vs_ref_pct=rret[ti, ji] * 100, volume=VOLx[ti, ji], limit_pct=lim[ti, 0] * 100, cat=cat[ti, ji], seg=tag))
    bigref = hs & np.isfinite(rret) & (np.abs(rret) > lim + 1e-4)
    return df, dict(n_priced_cells=int(hs.sum()), n_over_limit_vs_prev_close=int(big.sum()), by_cat={k: int(v) for k, v in df['cat'].value_counts().items()},
                    n_over_limit_vs_ref_price=int(bigref.sum()), n_over_limit_vs_ref_excl_first5=int((bigref & ~first5).sum()),
                    n_abs_ref_ret_gt_50pct=int((hs & (np.abs(rret) > 0.5)).sum()))

exm = np.zeros((T, N), bool)
di = {x: i for i, x in enumerate(IDX.strftime('%Y-%m-%d'))}; cj = {c: j for j, c in enumerate(COLS)}
for dd, cc in evset:
    if dd in di and cc in cj:
        exm[di[dd], cj[cc]] = True
df_tw, s_tw = scan(IDX, COLS, C, CHG, MK, VOL, 1, exm, '官方 上市 2015+')
df_ot, s_ot = scan(IDX, COLS, C, CHG, MK, VOL, 2, exm, '官方 上櫃 2015+')
# 上櫃除權息檔有些讀不到：這些日子的 unexplained 另標
df_ot.loc[(df_ot['cat'] == 'unexplained') & df_ot['date'].dt.strftime('%Y-%m-%d').isin(ev_unread_otc), 'cat'] = 'unverifiable_event_file_missing'
df_tw.loc[(df_tw['cat'] == 'unexplained') & df_tw['date'].dt.strftime('%Y-%m-%d').isin(ev_unread_tse), 'cat'] = 'unverifiable_event_file_missing'
s_tw['by_cat'] = {k: int(v) for k, v in df_tw['cat'].value_counts().items()}; s_ot['by_cat'] = {k: int(v) for k, v in df_ot['cat'].value_counts().items()}
# pre
pc = sorted(pre['code'].unique()); PI = PIDX_all[PIDX_all < SEAM]
pdi = {x: i for i, x in enumerate(PI)}; pcj = {c: j for j, c in enumerate(pc)}
shape = (len(PI), len(pc))
ti = pre['date'].map(pdi).to_numpy(); tj = pre['code'].map(pcj).to_numpy()
def wide(col, fillv=np.nan):
    a = np.full(shape, fillv); a[ti, tj] = pre[col].to_numpy(float); return a
pC, pCH, pV = wide('close'), wide('change'), wide('volume')
pMK = np.zeros(shape, np.int8); pMK[ti, tj] = 1
pX = np.zeros(shape, bool); pX[ti, tj] = pre['xflag'].to_numpy() > 0
for dd, cc in zip(tw49['date'].astype(str).str[:10], tw49['code']):
    k = pdi.get(pd.Timestamp(dd)); j = pcj.get(cc)
    if k is not None and j is not None:
        pX[k, j] = True
df_pre, s_pre = scan(PI, np.array(pc), pC, pCH, pMK, pV, 1, pX, 'pre2015 上市 2007–2014')
allb = pd.concat([df_pre, df_tw, df_ot], ignore_index=True)
allb['abs'] = allb['ret_pct'].abs()
allb.sort_values(['seg', 'abs'], ascending=[True, False]).drop(columns='abs').to_csv(HERE / 'tables/over_limit_all.csv', index=False)
un = allb[allb['cat'] == 'unexplained'].sort_values('abs', ascending=False)
un.drop(columns='abs').to_csv(HERE / 'tables/over_limit_unexplained.csv', index=False)
def top(dfx, n=30):
    return [dict(date=str(r.date.date()), code=r.code, prev_close=r.prev_close, close=r.close, ret_pct=round(r.ret_pct, 2), change=None if pd.isna(r.change) else r.change,
                 ret_vs_ref_pct=None if pd.isna(r.ret_vs_ref_pct) else round(r.ret_vs_ref_pct, 2), volume=None if pd.isna(r.volume) else int(r.volume), seg=r.seg) for r in dfx.head(n).itertuples()]
b3 = dict(rule='|收盤÷前一個有成交日收盤−1| > 漲跌停上限＋0.01 個百分點（2015-06-01 前 7%、之後 10%）；排除順序：除權息／減資／面額變更／分割事件日 → 掛牌後前 5 根 → 前一交易日無成交（恢復交易）',
          pre2015_twse=s_pre, official_twse=s_tw, official_tpex=s_ot,
          unexplained_by_year_twse={int(k): int(v) for k, v in un[un['seg'].str.contains('上市')].groupby(un['date'].dt.year).size().items()},
          unexplained_by_year_tpex={int(k): int(v) for k, v in un[un['seg'].str.contains('上櫃')].groupby(un['date'].dt.year).size().items()},
          top30_unexplained_twse=top(un[un['seg'].str.contains('上市')]), top30_unexplained_tpex=top(un[un['seg'].str.contains('上櫃')]))
# 不可能值
def imposs(o, h, l, c, v, a, m):
    tr = m & (np.nan_to_num(v) > 0)
    with np.errstate(invalid='ignore'):
        return dict(close_le0_with_volume=int((tr & ~(c > 0)).sum()), close_le0_any=int((m & np.isfinite(c) & (c <= 0)).sum()), amount_neg=int((m & (a < 0)).sum()), volume_neg=int((m & (v < 0)).sum()),
                    high_lt_low=int((m & (h < l)).sum()), close_gt_high=int((m & (c > h + 1e-9)).sum()), close_lt_low=int((m & (c < l - 1e-9)).sum()),
                    open_gt_high=int((m & (o > h + 1e-9)).sum()), open_lt_low=int((m & (o < l - 1e-9)).sum()),
                    priced_but_zero_volume=int((m & (c > 0) & ~(np.nan_to_num(v) > 0)).sum()), volume_pos_amount_zero_or_nan=int((tr & ~(a > 0)).sum()),
                    vwap_outside_low_high=int((tr & (a > 0) & (((a / v) > h * 1.001) | ((a / v) < l * 0.999))).sum()))
b3['impossible_official_twse'] = imposs(O, H, L, C, VOL, AMT, tw); b3['impossible_official_tpex'] = imposs(O, H, L, C, VOL, AMT, ot)
pO, pH, pL, pA = wide('open'), wide('high'), wide('low'), wide('amount')
b3['impossible_pre2015_twse'] = imposs(pO, pH, pL, pC, pV, pA, pMK == 1)
R['B3_sanity'] = b3
print('B3', json.dumps({k: v for k, v in b3.items() if not k.startswith('top30')}, ensure_ascii=False)[:3000], flush=True)
for r in b3['top30_unexplained_twse'][:30]:
    print(r)
R['runtime_s'] = round(time.time() - t0, 1); R['maxrss_mb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
jdump(R, HERE / 'parts/prices.json'); print('done', R['runtime_s'], R['maxrss_mb'])
