#!/usr/bin/env python3
"""S92 第 0 步（PREREG 之前）：①重現正式標準路徑（48.86%／34.46%／381 筆／20 條路徑中位數／與網站 navs_official 逐日差 0）；
②資料驗收（公告日欄位、涵蓋率、與 S88 市場可得日的差；應收／存貨有值率）；③時點（資料齡、公告距訊號日天數；不看報酬）；
④訓練期（訊號 2016-02～2021-12）報酬分組。驗證期（2022 起）的報酬在本步驟完全不計算、不輸出。輸出 data_check.json。python3 -I -B prep.py"""
import sys, json, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import s92lib as L
from s92lib import stlib

t0 = time.time(); rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P, U, RT = L.load()
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D0, DE, D19 = P.IDX[P.t0], P.IDX[-1], lastb('2019-01-01')
res = {}
# ---- ① 驗收
S0 = L.signal(P, U, RT, 0)
nav, lg, st = L.sim(P, S0, U, cap=1e6)
lg['code'] = lg['code'].astype(str)
site = pd.read_csv('/home/user/-/xq_scripts/site/navs_official.csv.gz', index_col=0, parse_dates=True)['E0|W10_70']
cm = nav.index.intersection(site.index)
got = dict(since2019=stlib.cagr(nav, D19, DE), full=stlib.cagr(nav, D0, DE), maxdd=stlib.maxdd(nav, D0, DE), n_closed=int((lg['why'] != 'open_end').sum()), n_all=int(len(lg)),
           site_days=int(len(site)), common_days=int(len(cm)), site_same_index=bool(site.index.equals(nav.index)), site_max_abs_diff=float((nav.round(2) - site).abs().max()))
paths = {}
for s in L.SHIFTS:
    S = S0 if s == 0 else L.signal(P, U, RT, s)
    for cap in L.CAPS:
        paths[f'{int(cap/1e4)}萬|+{s}'] = stlib.cagr(L.sim(P, S, U, cap=cap)[0], D19, DE)
    if s:
        del S
got['paths_median'] = float(np.median(list(paths.values())))
T = L.TARGET
ok = (abs(got['since2019'] - T['since2019']) < 1e-9 and abs(got['full'] - T['full']) < 1e-9 and abs(got['maxdd'] - T['maxdd']) < 1e-9 and got['n_closed'] == T['n_closed']
      and got['n_all'] == T['n_all'] and abs(got['paths_median'] - T['paths_median']) < 1e-9 and got['site_max_abs_diff'] == 0 and got['site_same_index'])
res['acceptance'] = dict(got=got, target=T, ok=bool(ok))
print('驗收', res['acceptance'], rss(), 'MB', flush=True)
assert ok

# ---- ② 資料驗收（不看報酬）
FN = L.Fin(P)
fd = L.tload('filing_dates')
fd4 = fd[fd['code'].str.fullmatch(r'[1-9]\d{3}')].copy()
dc = dict(rows=int(len(fd)), rows_4digit=int(len(fd4)), doc_type=fd['doc_type'].value_counts().to_dict(), corrected=fd['corrected'].value_counts().to_dict(),
          year_min=int(fd['year'].min()), year_max=int(fd['year'].max()))
g = fd4.groupby(['code', 'year', 'season'])
first = g['uploaded_at'].min()
dc['code_quarters'] = int(len(first)); dc['rows_per_code_quarter_mean'] = float(g.size().mean())
re = fd4[fd4['doc_type'].str.contains('重編')]
f_non = fd4[~fd4['doc_type'].str.contains('重編')].groupby(['code', 'year', 'season'])['uploaded_at'].min()
dc['restated_rows'] = int(len(re)); dc['code_quarters_only_restated'] = int(len(first.index.difference(f_non.index)))
j = pd.concat([first.rename('F'), f_non.rename('F0')], axis=1).dropna()
dc['code_quarters_restated_earlier_than_original'] = int((j['F'] < j['F0']).sum())
dl = pd.Series([L.deadline(y * 4 + q - 1) for (_, y, q) in first.index], index=first.index)
lag = (first.dt.normalize() - dl).dt.days
qs = lambda s: {k: float(v) for k, v in s.quantile([.01, .05, .25, .5, .75, .95, .99]).items()}
dc['first_upload_minus_deadline_days'] = dict(quantiles=qs(lag), share_after_deadline=float((lag > 0).mean()), n_over_30d_late=int((lag > 30).sum()),
                                              note='晚於期限 30 天以上的多半是後來補傳／更正後覆蓋的上傳時間（原始首傳時間已看不到）→ 會被當成「較晚才可得」，方向保守')
a = pd.read_parquet(L.PEAD / 'data' / 'avail_dates.parquet', columns=['code', 'y', 'q', 'A', 'A_method'])
x = first.rename('F').reset_index().merge(a.rename(columns={'y': 'year', 'q': 'season'}), on=['code', 'year', 'season'], how='inner')
x['d'] = (x['F'].dt.normalize() - pd.to_datetime(x['A'])).dt.days
dc['F_minus_A_days_all_stocks'] = dict(n=int(len(x)), quantiles=qs(x['d']), same_day=float((x['d'] == 0).mean()), F_earlier=float((x['d'] < 0).mean()), F_later=float((x['d'] > 0).mean()),
                                       within3=float((x['d'].abs() <= 3).mean()))
res['filing_dates'] = dc
# fs_xbrl 有值率（按業別；只算面板內 4 碼代號）
xb = L.tload('fs_xbrl', columns=['period', 'code', 'industry_template', 'report', 'inventories', 'accounts_receivable', 'accounts_receivable_related', 'notes_receivable', 'revenue_ytd', 'cost_of_sales_ytd'])
xb = xb[xb['code'].isin(set(map(str, P.COLS)))]
xb['ar_any'] = xb[['accounts_receivable', 'accounts_receivable_related', 'notes_receivable']].notna().any(axis=1)
res['fs_xbrl'] = dict(rows_in_panel=int(len(xb)), report=xb['report'].value_counts().to_dict(),
                      by_template={t: dict(rows=int(len(z)), codes=int(z['code'].nunique()), ar=float(z['accounts_receivable'].notna().mean()), ar_any=float(z['ar_any'].mean()),
                                           inventories=float(z['inventories'].notna().mean()), revenue=float(z['revenue_ytd'].notna().mean()), cogs=float(z['cost_of_sales_ytd'].notna().mean()))
                                   for t, z in xb.groupby('industry_template')})
# fs_xbrl 單季營收 vs fin_long 單季營收（交叉檢查）
both = np.isfinite(FN.xrev) & np.isfinite(FN.rev) & (np.abs(FN.rev) > 0)
with np.errstate(invalid='ignore', divide='ignore'):
    rel = np.abs(FN.xrev[both] / FN.rev[both] - 1)
res['fs_xbrl']['rev_q_vs_fin_long'] = dict(n=int(both.sum()), within_0p1pct=float((rel < 1e-3).mean()), within_1pct=float((rel < 1e-2).mean()))
print(json.dumps(res['filing_dates'], ensure_ascii=False, default=str)[:3000]); print(json.dumps(res['fs_xbrl'], ensure_ascii=False), flush=True)

# ---- ③ 交易逐筆特徵（時點；不看報酬）
rows = []
for k, jx in zip(lg['sig_t'].to_numpy(), lg['j'].to_numpy()):
    k = int(k); jx = int(jx); d = P.IDX[k]
    X = L.featx(P, FN, k, 'fd'); Xs = L.featx(P, FN, k, 's88'); A = L.feat88(P, U, FN, k, 'fd'); B = L.feat88(P, U, FN, k, 's88')
    r = {n: float(X[n][jx]) for n in ('ar_g', 'inv_g', 'rev_g', 'ar_gap', 'inv_gap', 'dso', 'dio', 'dso_chg', 'dio_chg', 'kq', 'age_days', 'since_filing')}
    r.update(kq_s88x=float(Xs['kq'][jx]), age_s88x=float(Xs['age_days'][jx]), tpl=str(FN.tpl.iloc[jx]), has_ar=bool(np.isfinite(X['ar'][jx])), has_inv=bool(np.isfinite(X['inv'][jx])),
             kq88_fd=float(A['kq'][jx]), age88_fd=float(A['age_days'][jx]), kq88_s88=float(B['kq'][jx]), age88_s88=float(B['age_days'][jx]),
             R09_fd=bool(A['ACC_HI'][jx] | A['OCF_NEG'][jx]), R09_s88=bool(B['ACC_HI'][jx] | B['OCF_NEG'][jx]),
             ACC_HI_fd=bool(A['ACC_HI'][jx]), OCF_NEG_fd=bool(A['OCF_NEG'][jx]), ACC_HI_s88=bool(B['ACC_HI'][jx]), OCF_NEG_s88=bool(B['OCF_NEG'][jx]))
    # 該筆「進場當下最近一季」＝S88 估法挑的那一季，有沒有實際公告日
    i88 = FN.pick(d, 's88', 's88')[jx]; ifd = FN.pick(d, 'fd', 'x')[jx]
    r['s88_quarter_has_FD'] = bool(i88 >= 0 and not np.isnat(FN.FD[jx, i88]))
    r['fd_quarter_has_FD'] = bool(ifd >= 0 and not np.isnat(FN.FD[jx, ifd]))
    r['F_minus_A_s88q'] = float((FN.FD[jx, i88] - FN.A[jx, i88]).astype(float)) if (i88 >= 0 and not np.isnat(FN.FD[jx, i88]) and not np.isnat(FN.A[jx, i88])) else np.nan
    rows.append(r)
TF = pd.concat([lg.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
TF['per'] = np.where(TF['sig_date'] < pd.Timestamp('2016-02-01'), 'pre', np.where(TF['sig_date'] < pd.Timestamp('2022-01-01'), 'train', 'valid'))
TF['sig_month'] = TF['sig_date'].dt.month
tm = {}
for nm, z in (('all', TF), ('train', TF[TF['per'] == 'train']), ('valid', TF[TF['per'] == 'valid'])):
    sf = z['since_filing']
    tm[nm] = dict(n=int(len(z)), s88_quarter_has_actual_filing_date=float(z['s88_quarter_has_FD'].mean()), fd_quarter_has_actual_filing_date=float(z['fd_quarter_has_FD'].mean()),
                  has_any_quarter_fd=float(np.isfinite(z['kq']).mean()), has_any_quarter_s88=float(np.isfinite(z['kq88_s88']).mean()),
                  F_minus_A=dict(n=int(z['F_minus_A_s88q'].notna().sum()), quantiles=qs(z['F_minus_A_s88q'].dropna()), same_day=float((z['F_minus_A_s88q'].dropna() == 0).mean()),
                                 within3=float((z['F_minus_A_s88q'].dropna().abs() <= 3).mean()), F_later_than_A=float((z['F_minus_A_s88q'].dropna() > 0).mean())),
                  age_median_s88=float(z['age88_s88'].median()), age_median_fd=float(z['age88_fd'].median()), age_median_fd_xbrl=float(z['age_days'].median()), age_median_s88_xbrl=float(z['age_s88x'].median()),
                  age_mean_s88=float(z['age88_s88'].mean()), age_mean_fd=float(z['age88_fd'].mean()),
                  n_newer_quarter_with_fd=int((z['kq88_fd'] > z['kq88_s88']).sum()), n_older_quarter_with_fd=int((z['kq88_fd'] < z['kq88_s88']).sum()), n_same_quarter=int((z['kq88_fd'] == z['kq88_s88']).sum()),
                  since_filing_quantiles=qs(sf.dropna()), n_since_le5=int((sf <= 5).sum()), n_since_le10=int((sf <= 10).sum()), n_since_le30=int((sf <= 30).sum()),
                  n_since_31_60=int(((sf > 30) & (sf <= 60)).sum()), n_since_gt60=int((sf > 60).sum()), n_since_na=int(sf.isna().sum()),
                  R09_rate_s88=float(z['R09_s88'].mean()), R09_rate_fd=float(z['R09_fd'].mean()), R09_flip=int((z['R09_s88'] != z['R09_fd']).sum()),
                  avail={c: float(np.isfinite(z[c]).mean()) for c in ('ar_gap', 'inv_gap', 'dso_chg', 'dio_chg')}, has_ar=float(z['has_ar'].mean()), has_inv=float(z['has_inv'].mean()),
                  tpl=z['tpl'].value_counts().to_dict(),
                  newer_by_sig_month={int(m): [int((q['kq88_fd'] > q['kq88_s88']).sum()), int(len(q))] for m, q in z.groupby('sig_month')})
res['timing'] = tm
print(json.dumps(tm, ensure_ascii=False, indent=1), flush=True)

# ---- ④ 訓練期報酬分組（只有訊號日 2016-02-01～2021-12-31）
Z = TF[TF['per'] == 'train'].copy()
hi, lo = Z['ret'].quantile(0.95), Z['ret'].quantile(0.05)
Z['pnl_rank'] = Z['pnl'].rank(ascending=False, method='first')
tp = float(Z['pnl'].sum())


def gstat(z):
    if not len(z):
        return dict(n=0)
    return dict(n=int(len(z)), win=float((z['ret'] > 0).mean()), mean=float(z['ret'].mean()), median=float(z['ret'].median()), n_big_win=int((z['ret'] >= hi).sum()), n_big_lose=int((z['ret'] <= lo).sum()),
                n_top10_pnl=int((z['pnl_rank'] <= 10).sum()), pnl_share=float(z['pnl'].sum() / tp))


G = dict(ALL=gstat(Z), thresholds=dict(big_win_ret_ge=float(hi), big_lose_ret_le=float(lo)), terciles={})
for c in ('ar_gap', 'inv_gap', 'dso_chg', 'dio_chg'):
    v = Z[c]; q1, q2 = v.quantile(1 / 3), v.quantile(2 / 3)
    G['terciles'][c] = dict(cut=[float(q1), float(q2)], low=gstat(Z[v <= q1]), mid=gstat(Z[(v > q1) & (v <= q2)]), high=gstat(Z[v > q2]), na=gstat(Z[v.isna()]))
    G[c + '_thr'] = {}
    thr = (0.0, 0.10, 0.20, 0.30, 0.50) if c.endswith('gap') else (0.0, 10.0, 20.0, 30.0)
    for xthr in thr:
        G[c + '_thr'][str(xthr)] = dict(above=gstat(Z[v > xthr]), not_above=gstat(Z[~(v > xthr)]))
for xthr in (0.0, 0.10, 0.20, 0.30):
    both = (Z['ar_gap'] > xthr) & (Z['inv_gap'] > xthr); either = (Z['ar_gap'] > xthr) | (Z['inv_gap'] > xthr)
    G[f'both_gap>{xthr}'] = gstat(Z[both]); G[f'either_gap>{xthr}'] = gstat(Z[either])
sf = Z['since_filing']
G['since_filing'] = {'<=10': gstat(Z[sf <= 10]), '11-30': gstat(Z[(sf > 10) & (sf <= 30)]), '31-60': gstat(Z[(sf > 30) & (sf <= 60)]), '>60': gstat(Z[sf > 60]), 'na': gstat(Z[sf.isna()])}
G['age_fd'] = {'<=75': gstat(Z[Z['age_days'] <= 75]), '76-120': gstat(Z[(Z['age_days'] > 75) & (Z['age_days'] <= 120)]), '>120': gstat(Z[Z['age_days'] > 120])}
G['R09_s88=1'] = gstat(Z[Z['R09_s88']]); G['R09_fd=1'] = gstat(Z[Z['R09_fd']]); G['R09_s88=0'] = gstat(Z[~Z['R09_s88']]); G['R09_fd=0'] = gstat(Z[~Z['R09_fd']])
G['newer_quarter_with_fd'] = gstat(Z[Z['kq88_fd'] > Z['kq88_s88']])
res['train_returns_by_group'] = G
res['meta'] = dict(runtime_s=round(time.time() - t0, 1), maxrss_mb=rss())
(H / 'data_check.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
pr = lambda v: {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()}
print('ALL', pr(G['ALL']), G['thresholds'])
for c, v in G['terciles'].items():
    print(c, [round(q, 3) for q in v['cut']])
    for g_ in ('low', 'mid', 'high', 'na'):
        print('   ', g_, pr(v[g_]))
for k_, v in G.items():
    if k_.endswith('_thr'):
        for t_, w in v.items():
            print(k_, t_, 'above', pr(w['above']))
    elif k_ in ('since_filing', 'age_fd'):
        for t_, w in v.items():
            print(k_, t_, pr(w))
    elif k_ not in ('ALL', 'thresholds', 'terciles'):
        print(k_, pr(v))
print(res['meta'])
