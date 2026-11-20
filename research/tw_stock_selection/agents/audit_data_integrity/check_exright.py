#!/usr/bin/env python3
"""B4 除權息涵蓋。python3 -I -B check_exright.py → parts/exright.json、tables/exright_*.csv。只讀。"""
import sys; from pathlib import Path as _P; sys.path.insert(0, str(_P(__file__).resolve().parent))
import json, time, resource, hashlib
import numpy as np, pandas as pd
from common import *                                    # noqa: E402

t0 = time.time(); R = {}
meta = json.loads((TMP / 'off_meta.json').read_text())
z = np.load(TMP / 'off.npz', allow_pickle=False)
IDX = pd.DatetimeIndex(z['IDX']); COLS = z['COLS'].astype(str); MK = z['MK']; C = z['C']; CHG = z['CHG']
T, N = C.shape
is4 = lambda s: s.str.fullmatch(r'[1-9]\d{3}')

# ---- TWT49U 封存
sums = {}
for l in (TWT49U.parent / 'SHA256SUMS').read_text().strip().split('\n'):
    h, p = l.split(); sums[p.lstrip('*').lstrip('./')] = h
files = sorted(TWT49U.glob('*/*.csv.gz'))
bad_sha = [str(f.relative_to(TWT49U.parent)) for f in files if sums.get(str(f.relative_to(TWT49U.parent))) != sha_file(f)]
tw = pd.concat([pd.read_csv(f, dtype={'code': str}).assign(file=f.name) for f in files], ignore_index=True)
tw['date'] = pd.to_datetime(tw['date'])
tw4 = tw[is4(tw['code'])]
R['twt49u'] = dict(n_files=len(files), n_in_sha256sums=len(sums), sha_mismatch=bad_sha, files_not_in_sums=[str(f.relative_to(TWT49U.parent)) for f in files if str(f.relative_to(TWT49U.parent)) not in sums],
    first=str(tw['date'].min().date()), last=str(tw['date'].max().date()), n_rows=len(tw), n_rows_4digit=len(tw4), n_dup_date_code=int(tw.duplicated(['date', 'code']).sum()),
    by_year_all={int(k): int(v) for k, v in tw.groupby(tw['date'].dt.year).size().items()}, by_year_4digit={int(k): int(v) for k, v in tw4.groupby(tw4['date'].dt.year).size().items()},
    by_type={str(k): int(v) for k, v in tw4['type'].value_counts().items()},
    months_missing=[str(m) for m in pd.period_range(tw['date'].min(), tw['date'].max(), freq='M') if m not in set(tw['date'].dt.to_period('M'))],
    n_bad_values=int((~(tw4['ref_price'] > 0) | ~(tw4['prev_close'] > 0)).sum()))

# ---- 官方倉庫 exright
E = pd.read_csv(TMP / 'events.csv', dtype=str, low_memory=False)
X = E[E['sub'] == 'exright'].copy(); X['date'] = pd.to_datetime(X['date'].fillna(X['file_date']))
for c in ('pre_close', 'ref_price', 'value'):
    X[c] = pd.to_numeric(X[c], errors='coerce')
X4 = X[is4(X['stock_id'])]
em = meta['events']
R['repo_exright'] = dict(raw_archive=em['exright']['raw'], tsd=em['exright']['tsd'], unreadable_days=em['exright']['unreadable_days'],
    n_rows=len(X), n_rows_4digit=len(X4), first_read=str(X['date'].min().date()), last_read=str(X['date'].max().date()),
    n_file_date_ne_row_date=int((X['date'].dt.strftime('%Y-%m-%d') != X['file_date']).sum()),
    by_year_all={int(k): int(v) for k, v in X.groupby(X['date'].dt.year).size().items()}, by_year_4digit={int(k): int(v) for k, v in X4.groupby(X4['date'].dt.year).size().items()},
    by_kind={str(k): int(v) for k, v in X4['kind'].value_counts().items()}, n_dup_date_code=int(X.duplicated(['date', 'stock_id']).sum()),
    other_event_files={k: dict(tsd=v['tsd'], raw_archive_missing_local=v['raw']['missing_local'], n_unreadable=len(v['unreadable_days'])) for k, v in em.items() if k != 'exright'})

# ---- 兩個來源交叉（重疊期、四碼）
a0, a1 = max(tw4['date'].min(), X4['date'].min()), min(tw4['date'].max(), X4['date'].max())
unread = set(pd.to_datetime(em['exright']['unreadable_days']))
A = tw4[(tw4['date'] >= a0) & (tw4['date'] <= a1)].drop_duplicates(['date', 'code'], keep='last').set_index(['date', 'code'])
B = X4[(X4['date'] >= a0) & (X4['date'] <= a1)].drop_duplicates(['date', 'stock_id'], keep='last').rename(columns={'stock_id': 'code'}).set_index(['date', 'code'])
both = A.index.intersection(B.index); onlyA = A.index.difference(B.index); onlyB = B.index.difference(A.index)
onlyA_unread = [k for k in onlyA if k[0] in unread]
j = A.loc[both].join(B.loc[both], lsuffix='_t', rsuffix='_r')
R['cross_twt49u_vs_repo'] = dict(window=[str(a0.date()), str(a1.date())], n_twt49u=len(A), n_repo=len(B), n_both=len(both), n_only_twt49u=len(onlyA), n_only_twt49u_on_unreadable_repo_days=len(onlyA_unread),
    n_only_repo=len(onlyB), match_rate_of_twt49u_pct=round(len(both) / len(A) * 100, 3), match_rate_of_repo_pct=round(len(both) / len(B) * 100, 3),
    ref_price_equal_pct=round(float((np.abs(j['ref_price_t'] - j['ref_price_r']) < 0.006).mean() * 100), 3), prev_close_equal_pct=round(float((np.abs(j['prev_close'] - j['pre_close']) < 0.006).mean() * 100), 3),
    value_equal_pct=round(float((np.abs(j['value_t'] - j['value_r']) < 0.0006).mean() * 100), 3),
    only_twt49u_sample=[(str(k[0].date()), k[1]) for k in list(onlyA)[:20]], only_repo_sample=[(str(k[0].date()), k[1]) for k in list(onlyB)[:20]])

# ---- 事件日不是交易日（颱風假順延）
cal = set(IDX)
for nm, df, cc in (('twt49u', tw4[tw4['date'] >= IDX[0]], 'code'), ('repo_exright', X4, 'stock_id')):
    nt = df[~df['date'].isin(cal) & (df['date'] <= IDX[-1])]
    rows = []
    for r in nt.itertuples():
        k = int(np.searchsorted(IDX.values, np.datetime64(r.date)))
        jx = np.flatnonzero(COLS == getattr(r, cc))
        if k < T and len(jx):
            c, ch = C[k, jx[0]], CHG[k, jx[0]]
            rows.append(dict(event_date=str(r.date.date()), code=getattr(r, cc), next_trading_day=str(IDX[k].date()), ref_price=r.ref_price, close=c, change=ch,
                             k_ref=None if not np.isfinite(c - ch) else round(float(c - ch), 2), k_ref_matches=bool(np.isfinite(c - ch) and abs(c - ch - r.ref_price) < 0.011)))
    R['events_on_non_trading_days_' + nm] = dict(n=len(nt), days=sorted(set(str(x.date()) for x in nt['date'])), n_next_day_k_has_ref=sum(1 for x in rows if x['k_ref_matches']), rows=rows[:60])

# ---- 與日 K 比對（官方 2015+，上市）
has = np.isfinite(C) & (C > 0)
CF = pd.DataFrame(np.where(has, C, np.nan)).ffill().to_numpy(); prevC = np.vstack([np.full((1, N), np.nan), CF[:-1]])
prev_tr = np.vstack([np.zeros((1, N), bool), has[:-1]])
di = {d: i for i, d in enumerate(IDX)}; cj = {c: k for k, c in enumerate(COLS)}
def mask_of(dates, codes):
    m = np.zeros((T, N), bool)
    for d, c in zip(dates, codes):
        i = di.get(d); k = cj.get(c)
        if i is not None and k is not None:
            m[i, k] = True
    return m
mT = mask_of(tw4['date'], tw4['code']); mR = mask_of(X4['date'], X4['stock_id'])
O = E[E['sub'].isin(['reduce', 'parvalue', 'etfsplit'])]; mO = mask_of(pd.to_datetime(O['date'].fillna(O['file_date'])), O['stock_id'])
twse = MK == 1
with np.errstate(invalid='ignore'):
    jump = has & twse & np.isfinite(prevC) & (np.nan_to_num(CHG) == 0) & (np.abs(C - prevC) > 1e-9)     # 漲跌價差＝0 但收盤對前一成交收盤有變
yrs = IDX.year
def by_year(m):
    return {int(y): int(m[yrs == y].sum()) for y in sorted(set(yrs))}
cat = dict(in_twt49u=jump & mT, in_repo_only=jump & mR & ~mT, in_other_event=jump & mO & ~mT & ~mR, prev_untraded=jump & ~mT & ~mR & ~mO & ~prev_tr)
cat['unexplained'] = jump & ~mT & ~mR & ~mO & prev_tr
ti, ji = np.nonzero(cat['unexplained'])
unex = pd.DataFrame(dict(date=IDX[ti], code=COLS[ji], prev_close=prevC[ti, ji], close=C[ti, ji], ret_pct=(C[ti, ji] / prevC[ti, ji] - 1) * 100))
unex['after_twt49u_archive'] = unex['date'] > tw['date'].max(); unex['repo_file_unreadable'] = unex['date'].isin(unread)
unex.to_csv(HERE / 'tables/exright_zero_change_unexplained.csv', index=False)
ev_any = (mT | mR) & twse & has
with np.errstate(invalid='ignore'):
    ev_zero = ev_any & (np.nan_to_num(CHG) == 0)
REF = np.full((T, N), np.nan)
for d, c, r in list(zip(X4['date'], X4['stock_id'], X4['ref_price'])) + list(zip(tw4['date'], tw4['code'], tw4['ref_price'])):
    i = di.get(d); k = cj.get(c)
    if i is not None and k is not None:
        REF[i, k] = r
with np.errstate(invalid='ignore'):
    ev_k_ok = ev_any & ~ev_zero & (np.abs(C - CHG - REF) < 0.011)
R['vs_daily_k_official_twse'] = dict(
    n_zero_change_but_close_moved=int(jump.sum()), by_category={k: int(v.sum()) for k, v in cat.items()},
    explained_by_exright_or_event_pct=round(float((cat['in_twt49u'] | cat['in_repo_only'] | cat['in_other_event']).sum() / jump.sum() * 100), 3),
    explained_incl_prev_untraded_pct=round(float((jump & ~cat['unexplained']).sum() / jump.sum() * 100), 3),
    unexplained_by_year=by_year(cat['unexplained']), in_repo_only_by_year=by_year(cat['in_repo_only']),
    unexplained_top=[dict(date=str(r.date.date()), code=r.code, prev_close=r.prev_close, close=r.close, ret_pct=round(r.ret_pct, 2), after_twt49u_archive=bool(r.after_twt49u_archive))
                     for r in unex.reindex(unex['ret_pct'].abs().sort_values(ascending=False).index).head(30).itertuples()],
    events_traded_twse=int(ev_any.sum()), events_with_zero_change_in_k=int(ev_zero.sum()), events_k_change_already_uses_ref=int(ev_k_ok.sum()),
    events_other=int((ev_any & ~ev_zero & ~ev_k_ok).sum()),
    events_zero_change_pct=round(float(ev_zero.sum() / ev_any.sum() * 100), 2), events_zero_change_by_year=by_year(ev_zero), events_by_year=by_year(ev_any))
# 封存結束後的事件（只在官方倉庫）
late = X4[X4['date'] > tw['date'].max()]
late_tw = late[late['stock_id'].map(lambda c: c in cj and (MK[:, cj[c]] == 1).any())]
R['after_twt49u_archive'] = dict(archive_last=str(tw['date'].max().date()), price_last=str(IDX[-1].date()), repo_last_event_file=em['exright']['tsd']['last'],
    n_events_4digit=len(late), n_events_4digit_twse=len(late_tw), events=[dict(date=str(r.date.date()), code=r.stock_id, kind=r.kind, pre_close=r.pre_close, ref_price=r.ref_price, value=r.value) for r in late_tw.itertuples()],
    n_trading_days_uncovered=int((IDX > tw['date'].max()).sum()))
# 名單股／持股是否受影響
try:
    lj = json.loads((AG / 's18_live_list/lists/2026-11_exright.json').read_text())
    top = [str(x.get('code')) for x in lj.get('top20', lj.get('rows', []))] if isinstance(lj, dict) else []
    hold = pd.read_csv(SITE / 'state/holdings_after_2026-11.csv', dtype=str)
    hc = [c for c in hold.iloc[:, 0].astype(str)]
    R['after_twt49u_archive']['hit_current_holdings'] = sorted(set(late_tw['stock_id']) & set(hc)); R['after_twt49u_archive']['holdings'] = hc
    R['after_twt49u_archive']['list_meta_exright'] = lj.get('meta', {}).get('exright') if isinstance(lj, dict) else None
except Exception as e:
    R['after_twt49u_archive']['holdings_check_error'] = repr(e)

# ---- 2014：pre 檔 xflag vs TWT49U
d = pd.read_csv(PRE, dtype={'code': str}); d['date'] = pd.to_datetime(d['date'])
d = d.sort_values(['code', 'date'])
d['pc'] = d.groupby('code')['close'].transform(lambda s: s.where(s > 0).ffill().shift(1))
xf = d[d['xflag'] > 0]
w0, w1 = tw4['date'].min(), pd.Timestamp('2014-12-31')
xs = set(zip(xf.loc[(xf['date'] >= w0) & (xf['date'] <= w1), 'date'], xf.loc[(xf['date'] >= w0) & (xf['date'] <= w1), 'code']))
ts = set(zip(tw4.loc[tw4['date'] <= w1, 'date'], tw4.loc[tw4['date'] <= w1, 'code']))
pre_codes = set(d['code'])
ts_in = {k for k in ts if k[1] in pre_codes}
xfy = xf[(xf['close'] > 0) & xf['pc'].notna()].copy(); xfy['gap'] = xfy['close'] / xfy['pc'] - 1
mkt = d[(d['close'] > 0) & d['pc'].notna()].assign(r=lambda q: q['close'] / q['pc'] - 1).groupby('date')['r'].median()
xfy['gap_ex_mkt'] = xfy['gap'] - xfy['date'].map(mkt)
R['pre2015_xflag'] = dict(
    note='2015 前的來源檔（Asoul/tsec）把除權息日的漲跌價差記成 X，沒有參考價；S75／R55 面板把這些日子的漲跌價差設 0＝當日報酬記 0',
    xflag_by_year={int(k): int(v) for k, v in xf.groupby(xf['date'].dt.year).size().items()},
    mean_close_vs_prev_close_pct_by_year={int(k): round(float(v) * 100, 3) for k, v in xfy.groupby(xfy['date'].dt.year)['gap'].mean().items()},
    mean_gap_minus_market_median_pct_by_year={int(k): round(float(v) * 100, 3) for k, v in xfy.groupby(xfy['date'].dt.year)['gap_ex_mkt'].mean().items()},
    check_2014=dict(window=[str(w0.date()), str(w1.date())], n_xflag=len(xs), n_twt49u_in_pre_codes=len(ts_in), n_both=len(xs & ts_in),
                    xflag_found_in_twt49u_pct=round(len(xs & ts_in) / len(xs) * 100, 2), twt49u_found_in_xflag_pct=round(len(xs & ts_in) / len(ts_in) * 100, 2),
                    only_xflag=sorted((str(a.date()), b) for a, b in xs - ts_in)[:20], only_twt49u=sorted((str(a.date()), b) for a, b in ts_in - xs)[:20]),
    gap_2007_to_2014_01_15=dict(n_xflag_events=int(((xf['date'] < w0)).sum()), ref_price_available=0,
                                 note='這段只知道哪一天除權息（X 記號），沒有參考價與息值；兩個除權息來源都沒有 2014-01-16 以前的上市資料'))
# 引擎實際修了幾格（讀正式結果）
try:
    ro = json.loads((AG / 'official_nav/out/res_official.json').read_text())
    R['engine_exright_info'] = dict(exright_info=ro.get('exright_info'), panel_end=ro.get('end'), T=ro.get('T'), N=ro.get('N'))
except Exception as e:
    R['engine_exright_info'] = repr(e)
R['runtime_s'] = round(time.time() - t0, 1); R['maxrss_mb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
jdump(R, HERE / 'parts/exright.json')
def show(o, depth=0):
    return json.dumps(o, ensure_ascii=False, default=str)[:1800]
for k, v in R.items():
    print('##', k, show(v))
