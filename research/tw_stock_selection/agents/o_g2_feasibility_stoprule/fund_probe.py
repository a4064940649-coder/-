#!/usr/bin/env python3
"""基金資料來源探查：GitHub repo／code 搜尋與檔案下載；每次連線記入 raw/access_log.json（含 UTC 時間、狀態碼）。
用法：python3 -I -B fund_probe.py search_repo "<q>" | search_code "<q>" | get <url> <raw相對路徑>
下載內容一律放 raw/（視為不可信），本程式不執行任何下載內容。
"""
import json, sys, time, datetime, urllib.request, urllib.parse
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / 'raw'; RAW.mkdir(exist_ok=True)
LOG = RAW / 'access_log.json'

def log(entry):
    L = json.loads(LOG.read_text()) if LOG.exists() else []
    entry['utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
    L.append(entry); LOG.write_text(json.dumps(L, ensure_ascii=False, indent=1))

def fetch(url, timeout=40):
    req = urllib.request.Request(url, headers={'User-Agent': 'tw-research-center/1.0', 'Accept': 'application/vnd.github+json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            b = r.read(); log({'url': url, 'status': r.status, 'bytes': len(b)}); return r.status, b
    except urllib.error.HTTPError as e:
        log({'url': url, 'status': e.code, 'error': str(e)}); return e.code, b''
    except Exception as e:
        log({'url': url, 'status': None, 'error': repr(e)[:300]}); return None, b''

if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd in ('search_repo', 'search_code'):
        kind = 'repositories' if cmd == 'search_repo' else 'code'
        q = sys.argv[2]
        st, b = fetch(f'https://api.github.com/search/{kind}?q={urllib.parse.quote(q)}&per_page=30')
        if st == 200:
            d = json.loads(b)
            print('total', d.get('total_count'))
            for it in d.get('items', []):
                if kind == 'repositories':
                    print(it['full_name'], '|', it.get('pushed_at'), '|', it.get('size'), '|', (it.get('description') or '')[:90])
                else:
                    print(it['repository']['full_name'], '|', it['path'])
        else:
            print('status', st)
    elif cmd == 'get':
        url, rel = sys.argv[2], sys.argv[3]
        st, b = fetch(url)
        print('status', st, len(b))
        if st == 200:
            p = RAW / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b)
