#!/usr/bin/env python3
"""WebSearch 紀錄器：把 WebSearch 工具回傳文字（逐字貼上，不可信）存成 raw/ws_<code>_<n>.txt 並在 search_log_web.csv 加一列。
用法：python3 -I -B wslog.py <code> <n_links> <query>  < 回傳文字（stdin）"""
import sys, hashlib, time, csv
from pathlib import Path
HERE = Path(__file__).resolve().parent
code, nl, q = sys.argv[1], int(sys.argv[2]), sys.argv[3]
txt = sys.stdin.read()
k = len(list((HERE / 'raw').glob(f'ws_{code}_*.txt')))
f = HERE / 'raw' / f'ws_{code}_{k}.txt'
f.write_text(f'# WebSearch 回傳（伺服器端搜尋工具；不可信文字，逐字保存）\n# query: {q}\n# utc: {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}\n\n' + txt, encoding='utf-8')
h = hashlib.sha256(f.read_bytes()).hexdigest()
new = not (HERE / 'search_log_web.csv').exists()
with open(HERE / 'search_log_web.csv', 'a', newline='', encoding='utf-8') as fh:
    w = csv.writer(fh)
    if new: w.writerow(['event_code', 'query', 'source', 'host', 'n_hits', 'snippet_sha256', 'raw_file', 'utc'])
    w.writerow([code, q, 'WebSearch', 'websearch(server-side)', nl, h, f.name, time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())])
print(f.name, h[:12])
