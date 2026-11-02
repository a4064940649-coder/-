# 獨立複製 o11：列出營收檔所有 commit 版本的 blob，批次抓取（只讀資料，不執行下載內容）
import subprocess, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, 'raw')
def git(repo, *a, inp=None):
    return subprocess.run(['git', '-c', 'core.quotepath=off', '-C', os.path.join(RAW, repo)] + list(a),
                          input=inp, capture_output=True, text=True, check=True).stdout
def versions(repo, path):
    out = git(repo, 'log', '--full-history', '--format=%H %cI %P', '--', path)
    res = []
    for ln in out.strip().splitlines():
        h, t, *par = ln.split()
        try:
            b = git(repo, 'rev-parse', f'{h}:{path}').strip()
        except subprocess.CalledProcessError:
            b = None  # 刪除
        res.append(dict(commit=h, T=t, nparents=len(par), blob=b))
    return res[::-1]
man = {}
tfiles = [l for l in git('tasi', 'log', '--all', '--format=', '--name-only', '--', '營收/').splitlines() if l.strip()]
tfiles = sorted(set(tfiles))
man['tasi'] = {f: versions('tasi', f) for f in tfiles}
ufiles = ['data/extras/revenue.csv', 'data/extras/rev_hist.csv.gz', 'data/extras/rev_5y.csv.gz']
man['screener'] = {f: versions('screener', f) for f in ufiles}
json.dump(man, open(os.path.join(HERE, 'manifest.json'), 'w'), ensure_ascii=False, indent=0)
for repo in ['tasi', 'screener']:
    blobs = sorted({v['blob'] for f in man[repo].values() for v in f if v['blob']})
    miss = git(repo, 'cat-file', '--batch-check', '--no-lazy-fetch' if False else '--batch-check', inp='\n'.join(blobs) + '\n') if False else None
    print(repo, 'files', len(man[repo]), 'blobs', len(blobs), 'merges', sum(v['nparents'] > 1 for f in man[repo].values() for v in f))
    for i in range(0, len(blobs), 100):
        subprocess.run(['git', '-C', os.path.join(RAW, repo), '-c', 'fetch.negotiationAlgorithm=noop', 'fetch', '-q', '--no-tags',
                        '--no-write-fetch-head', '--recurse-submodules=no', '--filter=blob:none', 'origin'] + blobs[i:i+100], check=True)
print('done')
