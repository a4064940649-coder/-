#!/usr/bin/env python3
"""稽核獨立重建（一）：從官方日 K 倉庫（git HEAD，不抓網路）抽出 0050 原始日 K 與事件檔。
用法：python3 -I -B extract_raw.py
輸出：data/e50_raw_<repo>.csv、data/ev_<kind>.csv（exright 只留 0050；etfsplit、reduce、parvalue 全留）、data/extract_log.json"""
import subprocess, os, io, json, csv
from pathlib import Path
HERE = Path(__file__).resolve().parent
OUT = HERE / 'data'; OUT.mkdir(exist_ok=True)
REPOS = {'tsd': '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/tsd',
         'raw': '/home/user/-/research/tw_stock_selection/raw/jb7k6295r7-bot_tw-stock-data'}
ENV = dict(os.environ, GIT_NO_LAZY_FETCH='1')


def blobs(repo, sub):
    ls = subprocess.run(['git', '-C', repo, 'ls-tree', 'HEAD', sub + '/'], capture_output=True, text=True, check=True, env=ENV).stdout
    items = [(l.split()[2], l.split('\t')[1]) for l in ls.strip().split('\n') if l.endswith('.csv') and '/_' not in l]
    have = set(subprocess.run(['git', '-C', repo, 'cat-file', '--batch-all-objects', '--batch-check=%(objectname)', '--unordered'], capture_output=True, text=True, check=True, env=ENV).stdout.split())
    p = subprocess.Popen(['git', '-C', repo, 'cat-file', '--batch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=ENV)
    for sha, path in items:
        if sha not in have:
            yield path, None
            continue
        p.stdin.write((sha + '\n').encode()); p.stdin.flush()
        hdr = p.stdout.readline().decode().split()
        if len(hdr) < 3 or hdr[1] != 'blob':
            yield path, None
            continue
        n = int(hdr[2]); data = p.stdout.read(n); p.stdout.read(1)
        yield path, data.decode('utf-8-sig')
    p.stdin.close(); p.wait()


log = {'head': {k: subprocess.run(['git', '-C', r, 'rev-parse', 'HEAD'], capture_output=True, text=True, env=ENV).stdout.strip() for k, r in REPOS.items()}}
for name, repo in REPOS.items():
    rows = []; miss = []; n = 0
    for path, txt in blobs(repo, 'data/universe/daily'):
        n += 1
        if txt is None:
            miss.append(path); continue
        for r in csv.DictReader(io.StringIO(txt)):
            if r.get('stock_id') == '0050':
                rows.append([r['date'], r['market'], r['open'], r['high'], r['low'], r['close'], r['volume'], r['change']])
    with open(OUT / f'e50_raw_{name}.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['date', 'market', 'open', 'high', 'low', 'close', 'volume', 'change']); w.writerows(sorted(rows))
    log[f'daily_{name}'] = dict(files=n, unreadable=miss, rows_0050=len(rows))
    print(name, 'daily', n, len(miss), len(rows), flush=True)
for kind in ('exright', 'etfsplit', 'reduce', 'parvalue'):
    for name, repo in REPOS.items():
        rows = []; miss = []; n = 0; hdrs = set()
        for path, txt in blobs(repo, 'data/universe/' + kind):
            n += 1
            if txt is None:
                miss.append(path); continue
            for r in csv.DictReader(io.StringIO(txt)):
                if kind == 'exright' and r.get('stock_id') != '0050':
                    continue
                hdrs.add(tuple(r.keys()))
                rows.append(dict(r, file=path.split('/')[-1]))
        keys = []
        for r in rows:
            for k in r:
                if k not in keys: keys.append(k)
        with open(OUT / f'ev_{kind}_{name}.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
        log[f'{kind}_{name}'] = dict(files=n, unreadable=miss, rows=len(rows), n_header_variants=len(hdrs))
        print(kind, name, n, miss, len(rows), flush=True)
json.dump(log, open(OUT / 'extract_log.json', 'w'), ensure_ascii=False, indent=1)
