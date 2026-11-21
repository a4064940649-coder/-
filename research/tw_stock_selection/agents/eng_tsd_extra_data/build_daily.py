#!/usr/bin/env python3
"""每日逐股目錄 → 一個精簡長表 parquet（date, code, 整數欄…；依日期、代號排序，zstd 壓縮）。
用法：python3 -I -B build_daily.py <臨時clone> <輸出資料夾> <來源目錄名（data/universe 底下）> <輸出檔名（不含副檔名）> [起始日]
會順便寫 <輸出檔名>.info.json：列數、涵蓋、缺日（對 data/universe/daily 的交易日清單）、空檔、重複鍵、sha256、上游 commit。
"""
import sys, io, os, json, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tsdlib
import pandas as pd, numpy as np

c, out, d, name = sys.argv[1:5]
start = sys.argv[5] if len(sys.argv) > 5 else '0000'
os.makedirs(out, exist_ok=True)
head = tsdlib.head(c)
src = 'data/universe/' + d
allf = tsdlib.ls(c, src)
e = [(p, s) for p, s in allf if p.endswith('.csv') and p.rsplit('/', 1)[1][:4].isdigit() and p.rsplit('/', 1)[1][:10] >= start]
other = [p.rsplit('/', 1)[1] for p, s in allf if (p, s) not in set(e)]
cal = sorted(p.rsplit('/', 1)[1][:10] for p, s in tsdlib.ls(c, 'data/universe/daily') if p.endswith('.csv') and p.rsplit('/', 1)[1][:4].isdigit())
tsdlib.fetch_shas(c, [s for _, s in e])
STR = {'stock_id', 'note', 'name', 'reason', 'adjust_from', 'restore_date', 'date', 'investor', 'item', 'kind'}
parts = []; empty = []; hdrs = {}; raw_bytes = 0; perday = {}
for p, b in tsdlib.iter_blobs(c, e):
    day = p.rsplit('/', 1)[1][:10]; raw_bytes += len(b)
    t = b.decode('utf-8-sig')
    h = t.split('\n', 1)[0].strip()
    hdrs.setdefault(h, [day, day, 0]); hdrs[h][1] = day; hdrs[h][2] += 1
    df = pd.read_csv(io.StringIO(t), dtype=str, keep_default_na=False)
    if len(df) == 0:
        empty.append(day); continue
    assert (df['date'] == day).all(), (p, df['date'].unique()[:3])
    perday[day] = len(df); parts.append(df)
df = pd.concat(parts, ignore_index=True); del parts
df = df.rename(columns={'stock_id': 'code'})
notint = {}
for col in df.columns:
    if col in STR or col == 'code':
        continue
    s = df[col].str.strip().str.replace(',', '', regex=False)
    v = pd.to_numeric(s.replace('', np.nan), errors='coerce')
    nbad = int((v.isna() & (s != '')).sum())
    if nbad: notint[col] = dict(unparseable=nbad, sample=s[v.isna() & (s != '')].unique()[:5].tolist())
    if v.notna().all() and (v == v.round()).all():
        df[col] = v.astype('int64')
    elif (v.dropna() == v.dropna().round()).all():
        df[col] = v.astype('Int64')
    else:
        df[col] = v.astype('float64')
df['date'] = pd.to_datetime(df['date']).dt.date
key = ['date', 'code'] if 'code' in df.columns else ['date']
ndup = int(df.duplicated(key).sum()) if 'code' in df.columns else None
df = df.sort_values(key, kind='stable').reset_index(drop=True)
for col in df.columns:
    if col in STR and col != 'date': df[col] = df[col].astype('category')
if 'code' in df.columns: df['code'] = df['code'].astype('category')
f = f'{out}/{name}.parquet'
df.to_parquet(f, compression='zstd', compression_level=9, index=False)
days = sorted(perday)
lo, hi = days[0], days[-1]
calr = [x for x in cal if lo <= x <= hi]
missing = sorted(set(calr) - set(days))
extra = sorted(set(days) - set(cal))
pc = pd.Series(perday)
info = dict(file=f'{name}.parquet', bytes=os.path.getsize(f), sha256=tsdlib.sha256_file(f), upstream_repo=tsdlib.URL, upstream_commit=head, source_dir=src,
            source_csv_files=len(e), source_raw_bytes=raw_bytes, other_files_in_dir=other, rows=len(df), first_date=lo, last_date=hi, days_with_rows=len(days),
            n_codes=int(df['code'].nunique()) if 'code' in df.columns else None, columns={k: str(v) for k, v in df.dtypes.items()},
            headers_seen={h: v for h, v in hdrs.items()}, duplicate_date_code_rows=ndup, non_numeric=notint,
            calendar='data/universe/daily 的檔名日期（同倉庫的日 K 交易日）', calendar_days_in_range=len(calr),
            missing_days=missing, n_missing_days=len(missing), empty_files=empty, days_not_in_calendar=extra,
            rows_per_day=dict(min=int(pc.min()), median=float(pc.median()), max=int(pc.max()), min_day=pc.idxmin()),
            rows_per_year={str(y): int(n) for y, n in df.groupby(pd.to_datetime(df['date']).dt.year).size().items()},
            days_per_year={y: sum(1 for x in days if x[:4] == y) for y in sorted({x[:4] for x in days})})
json.dump(info, open(f'{out}/{name}.info.json', 'w'), ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: v for k, v in info.items() if k not in ('rows_per_year',)}, ensure_ascii=False, default=str)[:2500])
