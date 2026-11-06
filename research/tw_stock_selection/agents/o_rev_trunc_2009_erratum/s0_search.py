#!/usr/bin/env python3
"""第 0 步：GitHub 公開 repo 搜尋（code／repositories），逐筆記入 raw/search_log.jsonl（只記中繼資料，不下載內容）。
用法：python3 -I -B s0_search.py code|repo '<query>' [...]"""
import sys, json, time, urllib.request, urllib.parse, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
LOG = HERE / 'raw' / 'search_log.jsonl'
def get(url):
    req = urllib.request.Request(url, headers={'Accept': 'application/vnd.github+json', 'User-Agent': 'o15rev'})
    with urllib.request.urlopen(req, timeout=40) as r: return json.loads(r.read())
kind = sys.argv[1]
for q in sys.argv[2:]:
    ep = 'code' if kind == 'code' else 'repositories'
    url = f'https://api.github.com/search/{ep}?q={urllib.parse.quote(q)}&per_page=50'
    for att in range(3):
        try:
            d = get(url); err = None; break
        except Exception as e:
            d = {}; err = repr(e); time.sleep(20)
    items = []
    for it in d.get('items', []):
        if ep == 'code':
            items.append(dict(repo=it['repository']['full_name'], path=it['path']))
        else:
            items.append(dict(repo=it['full_name'], size_kb=it['size'], pushed=it['pushed_at'], created=it['created_at'], desc=(it.get('description') or '')[:90]))
    rec = dict(ts=datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat() + 'Z', kind=kind, q=q, total=d.get('total_count'), err=err, items=items)
    with open(LOG, 'a', encoding='utf-8') as f: f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f'### [{kind}] {q} total={rec["total"]} err={err}')
    for it in items: print('  ', ' | '.join(str(v) for v in it.values()))
    time.sleep(6.5 if ep == 'code' else 2.5)
