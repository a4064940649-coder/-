#!/usr/bin/env python3
"""步驟 2(b) 原始子集：taiwanan/TW_Stock_Exchange_Crawler（GitHub 公開 clone，唯讀；只用 git show 讀檔，不執行其中程式）
monthly_raw_data/<code>.csv 每檔「最後存在」的版本（被刪的檔取刪除前一版）逐行原樣存入 raw/taiwanan_monthly_lastver.csv.gz
（欄：code, commit, commit_utc, path, line_no, line）。python3 -I -B s2b_extract_taiwanan.py"""
import subprocess, csv, gzip, hashlib, json, re
from pathlib import Path
HERE = Path(__file__).resolve().parent; AG = HERE.parent
REPO = AG / 'o_rev_survivorship/raw/taiwanan_TW_Stock_Exchange_Crawler'
def git(*a): return subprocess.run(['git', '-C', str(REPO), *a], capture_output=True, check=True).stdout
log = git('log', '--format=@%H %cI', '--name-status', '--', 'monthly_raw_data').decode()
last, cur, when = {}, None, {}
for line in log.splitlines():
    if line.startswith('@'):
        cur, t = line[1:].split(' '); when[cur] = t; continue
    if not line.strip(): continue
    st, path = line.split('\t')[0], line.split('\t')[-1]
    if not path.endswith('.csv') or path in last: continue
    last[path] = (cur + '^', cur) if st.startswith('D') else (cur, cur)
out = HERE / 'raw/taiwanan_monthly_lastver.csv.gz'; n = 0
with gzip.open(out, 'wt', encoding='utf-8', newline='') as g:
    w = csv.writer(g); w.writerow(['code', 'commit', 'commit_utc', 'path', 'line_no', 'line'])
    for path, (rev, c0) in sorted(last.items()):
        code = Path(path).stem
        if not re.fullmatch(r'\d{4}', code): continue
        full = git('rev-parse', rev).decode().strip()
        txt = git('show', f'{rev}:{path}').decode('utf-8-sig', errors='replace')
        for i, ln in enumerate(txt.splitlines()):
            w.writerow([code, full, when.get(c0), path, i, ln]); n += 1
man = dict(repo='taiwanan/TW_Stock_Exchange_Crawler', head=git('rev-parse', 'HEAD').decode().strip(), n_files=len(last), n_lines=n,
           out_sha256=hashlib.sha256(out.read_bytes()).hexdigest(), rule=__doc__)
(HERE / 'raw/taiwanan_manifest.json').write_text(json.dumps(man, ensure_ascii=False, indent=0), encoding='utf-8')
print(man['n_files'], n, out.stat().st_size)
