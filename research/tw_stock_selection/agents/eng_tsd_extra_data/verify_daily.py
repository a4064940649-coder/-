#!/usr/bin/env python3
"""驗證日 K 精簡面板＝原始日檔：逐檔、逐列、逐欄比對（不是抽樣，是 clone 裡有內容的全部日檔），外加各數值欄總和。
用法：python3 -I -B verify_daily.py <有日檔內容的 clone> <tsd_extra 資料夾> <結果 json> [<本機日 K 封存 clone，用來比對 blob sha>]
比對規則：文字欄（name、market、limit、price_basis）逐字相同；key 欄＝date_code；整數欄 int(原字串)＝面板值；
小數欄 Decimal(原字串)＝Decimal(面板浮點數的最短表示)；原檔空白 ⇔ 面板空值。
"""
import sys, os, io, csv, json, subprocess, glob, math
from decimal import Decimal
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tsdlib
import pyarrow.parquet as pq

c, out, resf = sys.argv[1:4]
arch = sys.argv[4] if len(sys.argv) > 4 else None
INTS = ['volume', 'amount', 'shares', 'transactions']; FLTS = ['open', 'high', 'low', 'close', 'change', 'last_price']; STRS = ['name', 'market', 'limit', 'price_basis']
ents = [(p, s) for p, s in tsdlib.ls(c, 'data/universe/daily') if p.rsplit('/', 1)[1][:4].isdigit()]
local = set(subprocess.run(['git', '-C', c, 'cat-file', '--batch-all-objects', '--unordered', '--batch-check=%(objectname)'], capture_output=True, env=tsdlib.ENV_NOFETCH).stdout.decode().split())
have = [(p, s) for p, s in ents if s in local]
res = dict(upstream_commit=tsdlib.head(c), source_files_in_tree=len(ents), source_files_compared=len(have))
by_month = {}
for p, s in have:
    by_month.setdefault(p.rsplit('/', 1)[1][:7], []).append((p, s))
nrow = 0; nfield = 0; bad = []; files_ok = 0
sum_src = {k: 0 for k in INTS}; sum_src.update({k: Decimal(0) for k in FLTS}); sum_pan = dict(sum_src)
nonnull_src = {k: 0 for k in INTS + FLTS}; nonnull_pan = dict(nonnull_src)
for m in sorted(by_month):
    f = f'{out}/daily/{m[:4]}/{m}.parquet'
    t = pq.read_table(f).to_pydict()
    idx = {}
    for i, d in enumerate(t['date']):
        idx.setdefault(d.isoformat(), []).append(i)
    for p, body in tsdlib.iter_blobs(c, by_month[m]):
        day = p.rsplit('/', 1)[1][:10]
        rows = list(csv.DictReader(io.StringIO(body.decode('utf-8-sig'))))
        ii = idx.get(day, [])
        fb = 0
        if len(ii) != len(rows):
            bad.append((day, 'rows', len(rows), len(ii))); continue
        # 面板依 (date, code) 排序；原檔照自己的順序 → 用 code 對
        pos = {t['code'][i]: i for i in ii}
        if len(pos) != len(ii):
            bad.append((day, 'dup_code')); continue
        for r in rows:
            i = pos.get(r['stock_id'])
            if i is None:
                bad.append((day, r['stock_id'], 'missing')); fb += 1; continue
            if r['key'] != day + '_' + r['stock_id'] or r['date'] != day: bad.append((day, r['stock_id'], 'key')); fb += 1
            for k in STRS:
                nfield += 1
                if t[k][i] != r[k]: bad.append((day, r['stock_id'], k, r[k], t[k][i])); fb += 1
            for k in INTS:
                nfield += 1; v = t[k][i]; sv = r[k].strip()
                if sv == '':
                    if v is not None: bad.append((day, r['stock_id'], k, sv, v)); fb += 1
                else:
                    nonnull_src[k] += 1; sum_src[k] += int(sv)
                    if v is None or int(sv) != v: bad.append((day, r['stock_id'], k, sv, v)); fb += 1
                if v is not None: nonnull_pan[k] += 1; sum_pan[k] += v
            for k in FLTS:
                nfield += 1; v = t[k][i]; sv = r[k].strip()
                if sv == '':
                    if v is not None: bad.append((day, r['stock_id'], k, sv, v)); fb += 1
                else:
                    nonnull_src[k] += 1; sum_src[k] += Decimal(sv)
                    if v is None or Decimal(repr(v)) != Decimal(sv): bad.append((day, r['stock_id'], k, sv, v)); fb += 1
                if v is not None: nonnull_pan[k] += 1; sum_pan[k] += Decimal(repr(v))
        nrow += len(rows); files_ok += (fb == 0)
res.update(rows_compared=nrow, fields_compared=nfield, files_fully_identical=files_ok, mismatches=len(bad), mismatch_sample=[list(map(str, b)) for b in bad[:40]],
           column_sums_source={k: str(v) for k, v in sum_src.items()}, column_sums_panel={k: str(v) for k, v in sum_pan.items()},
           column_sums_equal=all(sum_src[k] == sum_pan[k] for k in sum_src), nonnull_counts_equal=nonnull_src == nonnull_pan, nonnull_counts=nonnull_src)
# 面板總列數（全部月檔）對 來源列數
tot = sum(pq.read_metadata(f).num_rows for f in glob.glob(f'{out}/daily/*/*.parquet'))
res['panel_total_rows'] = tot
if arch:
    a = dict((p, s) for p, s in tsdlib.ls(arch, 'data/universe/daily') if p.rsplit('/', 1)[1][:4].isdigit())
    h = dict(ents)
    same = sum(1 for p in a if h.get(p) == a[p]); diff = sorted(p.rsplit('/', 1)[1][:10] for p in a if p in h and h[p] != a[p])
    res['archive_vs_upstream_head'] = dict(archive_commit=tsdlib.head(arch), archive_daily_files=len(a), same_blob_sha=same, different_blob_sha=len(diff),
        different_days=diff, only_in_archive=sorted(p for p in a if p not in h)[:20], upstream_days_after_archive=sum(1 for p in h if p not in a),
        note='blob sha 相同＝檔案內容逐位元相同。封存是研究引擎建面板時用的版本；不同的日子代表上游事後改過那天的日檔。')
json.dump(res, open(resf, 'w'), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in res.items() if k != 'mismatch_sample'}, ensure_ascii=False)[:3000]); print(res['mismatch_sample'][:8])
