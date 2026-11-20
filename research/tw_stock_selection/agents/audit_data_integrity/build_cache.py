#!/usr/bin/env python3
"""把官方日 K（封存倉庫＋暫存區較新 clone 多出來的日子）讀成寬表快取（暫存區，稽核結束刪除）。只讀倉庫、不連網。
python3 -I -B build_cache.py → TMP/off.npz、TMP/off_meta.json、TMP/e50_raw.csv、TMP/events.csv、TMP/revhist.csv"""
import io, json, time, resource, re
import numpy as np, pandas as pd
import sys; from pathlib import Path as _P; sys.path.insert(0, str(_P(__file__).resolve().parent))
from common import *                                    # noqa: E402

t0 = time.time()
TMP.mkdir(parents=True, exist_ok=True)
meta = {}
have_raw = local_objects(RAW_REPO); have_tsd = local_objects(TSD)
raw_items = ls_tree(RAW_REPO, 'data/universe/daily'); tsd_items = ls_tree(TSD, 'data/universe/daily')
isday = lambda p: re.fullmatch(r'\d{4}-\d{2}-\d{2}\.csv', p.rsplit('/', 1)[1]) is not None
raw_d = {p.rsplit('/', 1)[1][:10]: s for s, p in raw_items if isday(p)}
tsd_d = {p.rsplit('/', 1)[1][:10]: s for s, p in tsd_items if isday(p)}
common_days = sorted(set(raw_d) & set(tsd_d))
diff_days = [d for d in common_days if raw_d[d] != tsd_d[d]]
meta['daily'] = dict(raw_n=len(raw_d), raw_first=min(raw_d), raw_last=max(raw_d), tsd_n=len(tsd_d), tsd_first=min(tsd_d), tsd_last=max(tsd_d),
                     raw_missing_local=sum(1 for s in raw_d.values() if s not in have_raw),
                     tsd_missing_local=sum(1 for s in tsd_d.values() if s not in have_tsd),
                     tsd_missing_local_days=sorted(d for d, s in tsd_d.items() if s not in have_tsd)[:50],
                     only_in_raw=sorted(set(raw_d) - set(tsd_d)), only_in_tsd=sorted(set(tsd_d) - set(raw_d)),
                     n_common=len(common_days), n_common_blob_diff=len(diff_days), common_blob_diff_days=diff_days)
print('daily', {k: v for k, v in meta['daily'].items() if not isinstance(v, list)}, flush=True)

# 來源：封存倉庫的全部日子＋tsd 多出來的日子（本機有 blob 才讀）
plan = [('raw', d, raw_d[d]) for d in sorted(raw_d)] + [('tsd', d, tsd_d[d]) for d in sorted(set(tsd_d) - set(raw_d))]
cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'change']
days, etf, rowcnt, unread = [], [], [], []
for src, repo, have in (('raw', RAW_REPO, have_raw), ('tsd', TSD, have_tsd)):
    items = [(s, d) for sr, d, s in plan if sr == src]
    for d, b in read_blobs(repo, items, have):
        if b is None:
            unread.append((src, d)); continue
        try:
            x = pd.read_csv(io.BytesIO(b), dtype={'stock_id': str}, usecols=['date', 'stock_id', 'market'] + cols)
        except Exception as e:
            unread.append((src, d, repr(e)[:80])); continue
        if not len(x):
            unread.append((src, d, 'empty')); continue
        if (x['date'] != d).any():
            unread.append((src, d, 'date_mismatch'))
        z = x[x['stock_id'] == '0050']
        if len(z):
            etf.append((d,) + tuple(pd.to_numeric(z[c], errors='coerce').iloc[0] for c in ['open', 'high', 'low', 'close', 'volume', 'change']))
        n_all = len(x); n_dup = int(x.duplicated('stock_id').sum())
        x = x[x['stock_id'].str.fullmatch(r'[1-9]\d{3}') & x['market'].isin(['twse', 'tpex'])].drop_duplicates('stock_id', keep='last')
        vals = x[cols].apply(pd.to_numeric, errors='coerce').to_numpy(float)
        days.append((d, x['stock_id'].to_numpy(), np.where(x['market'].to_numpy() == 'twse', 1, 2).astype(np.int8), vals))
        rowcnt.append((d, src, n_all, n_dup, len(x)))
days.sort(key=lambda r: r[0])
codes = np.array(sorted(set(np.concatenate([r[1] for r in days]))))
ci = {c: j for j, c in enumerate(codes)}
T, N = len(days), len(codes)
A = {k: np.full((T, N), np.nan) for k in ['O', 'H', 'L', 'C', 'VOL', 'AMT', 'CHG']}
MK = np.zeros((T, N), np.int8)
for t, (d, cs, mk, vals) in enumerate(days):
    j = np.array([ci[c] for c in cs])
    for k, col in zip(['O', 'H', 'L', 'C', 'VOL', 'AMT', 'CHG'], range(7)):
        A[k][t, j] = vals[:, col]
    MK[t, j] = mk
idx = pd.to_datetime([r[0] for r in days]).values.astype('datetime64[ns]')
np.savez_compressed(TMP / 'off.npz', IDX=idx, COLS=codes.astype(str), MK=MK, **A)
pd.DataFrame(etf, columns=['date', 'open', 'high', 'low', 'close', 'volume', 'change']).sort_values('date').to_csv(TMP / 'e50_raw.csv', index=False)
pd.DataFrame(rowcnt, columns=['date', 'src', 'n_rows_all', 'n_dup_stock_id', 'n_rows_4digit']).sort_values('date').to_csv(TMP / 'rowcnt.csv', index=False)
meta['panel'] = dict(T=T, N=N, first=days[0][0], last=days[-1][0], unread=unread)
print('panel', T, N, len(unread), f'{time.time()-t0:.0f}s', flush=True)
del A, MK, days

# 事件檔（除權息、分割、面額、減資；上市＋上櫃）：用 tsd（較新）；本機沒有的 blob 記下來
ev, evmeta = [], {}
for sub in ('exright', 'otcexright', 'etfsplit', 'parvalue', 'reduce', 'otcparvalue', 'otcreduce', 'stophalt'):
    info = {}
    for nm, repo, have in (('raw', RAW_REPO, have_raw), ('tsd', TSD, have_tsd)):
        it = [(s, p) for s, p in ls_tree(repo, 'data/universe/' + sub) if isday(p)]
        info[nm] = dict(n_files=len(it), first=min((p.rsplit('/', 1)[1][:10] for s, p in it), default=None),
                        last=max((p.rsplit('/', 1)[1][:10] for s, p in it), default=None), missing_local=sum(1 for s, p in it if s not in have))
        info[nm + '_items'] = {p.rsplit('/', 1)[1][:10]: s for s, p in it}
    # 讀：tsd 有 blob 用 tsd，否則同 sha 在 raw 有就用 raw
    ti, ri = info.pop('tsd_items'), info.pop('raw_items')
    info['common_blob_diff'] = sum(1 for d in set(ti) & set(ri) if ti[d] != ri[d])
    alld = sorted(set(ti) | set(ri)); n_unread = 0
    from_tsd = [(ti[d], d) for d in alld if d in ti and ti[d] in have_tsd]
    from_raw = [(ri[d] if d not in ti else ti[d], d) for d in alld if not (d in ti and ti[d] in have_tsd) and ((ti.get(d) or ri.get(d)) in have_raw)]
    got = set()
    for repo, items in ((TSD, from_tsd), (RAW_REPO, from_raw)):
        for d, b in read_blobs(repo, items):
            if b is None:
                continue
            got.add(d)
            try:
                x = pd.read_csv(io.BytesIO(b), dtype=str)
            except Exception:
                continue
            x['file_date'] = d; x['sub'] = sub
            ev.append(x)
    info['unreadable_days'] = sorted(set(alld) - got)
    evmeta[sub] = info
    print(sub, {k: v for k, v in info.items() if k != 'unreadable_days'}, 'unreadable', len(info['unreadable_days']), flush=True)
E = pd.concat(ev, ignore_index=True)
E.to_csv(TMP / 'events.csv', index=False)
meta['events'] = evmeta

# 月營收彙總（data/mops/revenue_hist）
rv, rmeta = [], {}
for nm, repo, have in (('raw', RAW_REPO, have_raw), ('tsd', TSD, have_tsd)):
    it = [(s, p) for s, p in ls_tree(repo, 'data/mops/revenue_hist') if p.endswith('.csv')]
    rmeta[nm] = dict(n_files=len(it), missing_local=sum(1 for s, p in it if s not in have),
                     first=min(p.rsplit('/', 1)[1] for s, p in it), last=max(p.rsplit('/', 1)[1] for s, p in it))
    rmeta[nm + '_items'] = {p.rsplit('/', 1)[1]: s for s, p in it}
ti, ri = rmeta.pop('tsd_items'), rmeta.pop('raw_items')
rmeta['common_blob_diff'] = sorted(f for f in set(ti) & set(ri) if ti[f] != ri[f])
src_used = {}
for f in sorted(set(ti) | set(ri)):
    if f in ti and ti[f] in have_tsd:
        src_used[f] = ('tsd', ti[f])
    elif f in ri and ri[f] in have_raw:
        src_used[f] = ('raw', ri[f])
rmeta['stale_from_raw'] = sorted(f for f, (r, s) in src_used.items() if r == 'raw' and f in ti and ti[f] != s)
rmeta['unreadable'] = sorted(set(ti) | set(ri) - set(src_used)) if False else sorted((set(ti) | set(ri)) - set(src_used))
for repo, nm in ((TSD, 'tsd'), (RAW_REPO, 'raw')):
    items = [(s, f) for f, (r, s) in src_used.items() if r == nm]
    for f, b in read_blobs(repo, items):
        if b is None:
            continue
        x = pd.read_csv(io.BytesIO(b), dtype=str)
        x = x.rename(columns={'當月營收': 'rev', '去年當月營收': 'rev_ly'})[['stock_id', 'period', 'market', 'rev', 'rev_ly']]
        x['file'] = f; x['from'] = nm
        rv.append(x)
pd.concat(rv, ignore_index=True).to_csv(TMP / 'revhist.csv', index=False)
meta['revenue_hist'] = rmeta
print('revenue_hist', {k: (v if not isinstance(v, list) or len(v) < 8 else len(v)) for k, v in rmeta.items()}, flush=True)
meta['heads'] = {nm: git(repo, 'log', '-1', '--format=%H %cI %s').stdout.strip() for nm, repo in (('raw', RAW_REPO), ('tsd', TSD))}
meta['runtime_s'] = round(time.time() - t0, 1); meta['maxrss_mb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
jdump(meta, TMP / 'off_meta.json')
print('done', meta['runtime_s'], 's', meta['maxrss_mb'], 'MB')
