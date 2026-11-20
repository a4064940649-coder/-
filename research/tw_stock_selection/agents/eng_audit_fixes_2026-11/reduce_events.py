#!/usr/bin/env python3
"""把官方日 K 倉庫（tsd，git HEAD，只讀）的上市減資恢復買賣（reduce＝TWTAUU）、面額變更（parvalue）事件整理成一張表。
python3 -I -B reduce_events.py <tsd> → tables/reduce_parvalue_events.csv、tables/reduce_events_meta.json"""
import sys, os, io, csv, json, subprocess
from pathlib import Path
ENV = dict(os.environ, GIT_NO_LAZY_FETCH='1', GIT_CONFIG_COUNT='1', GIT_CONFIG_KEY_0='protocol.allow', GIT_CONFIG_VALUE_0='never')
tsd = sys.argv[1]; OUT = Path(__file__).resolve().parent / 'tables'
git = lambda *a, inp=None: subprocess.run(['git', '-C', tsd, *a], input=inp, capture_output=True, env=ENV)
local = set(x.split()[0] for x in git('cat-file', '--batch-check', '--batch-all-objects').stdout.decode().splitlines())
rows = []; meta = {}
for kind in ('reduce', 'parvalue', 'otcreduce', 'otcparvalue'):
    files = {}
    for ln in git('ls-tree', '-r', 'HEAD', f'data/universe/{kind}').stdout.decode().splitlines():
        m, p = ln.split('\t'); f = p.rsplit('/', 1)[1]
        if f.endswith('.csv') and f[:4].isdigit():
            files[f[:10]] = m.split()[2]
    un = sorted(d for d, s in files.items() if s not in local)
    meta[kind] = dict(n_files=len(files), first=min(files), last=max(files), unreadable_days=un)
    for d in sorted(files):
        if files[d] in local:
            b = git('cat-file', 'blob', files[d]).stdout.decode('utf-8')
            for r in csv.DictReader(io.StringIO(b)):
                rows.append(dict(date=r['date'], code=r['stock_id'].strip(), kind=kind, reason=(r.get('reason') or '').strip(), pre_close=r['pre_close'], ref_price=r['ref_price'],
                                 limit_up=r.get('limit_up', ''), limit_down=r.get('limit_down', ''), halt_date=r.get('halt_date', '') or ''))
rows.sort(key=lambda r: (r['date'], r['code'], r['kind']))
with open(OUT / 'reduce_parvalue_events.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator='\n'); w.writeheader(); w.writerows(rows)
meta['n_rows'] = {k: sum(1 for r in rows if r['kind'] == k) for k in meta}
meta['tsd_head'] = git('rev-parse', 'HEAD').stdout.decode().strip()
(OUT / 'reduce_events_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1))
print(json.dumps(meta, ensure_ascii=False, indent=1))
