#!/usr/bin/env python3
"""官方資料倉庫（tw-stock-data）的「一次只抓一個目錄」小工具。用 python3 -I -B 執行。

做法：臨時 clone（--filter=blob:none --depth=1 --no-checkout）只有目錄表、沒有檔案內容；
fetch_paths() 一次把指定目錄的檔案內容批次抓回來（一個請求，不是一檔一檔抓）；
iter_blobs() 用 git cat-file --batch 直接從 git 物件讀內容（不 checkout、不落地成檔案，下載內容不會被當成程式執行）；
讀取時一律 GIT_NO_LAZY_FETCH=1：沒抓到的檔會報錯，不會偷偷一個一個去抓。
"""
import os, subprocess, hashlib

URL = 'https://github.com/jb7k6295r7-bot/tw-stock-data'
ENV_NOFETCH = dict(os.environ, GIT_NO_LAZY_FETCH='1')


def clone(dst):
    subprocess.run(['git', 'clone', '-q', '--filter=blob:none', '--depth=1', '--no-checkout', URL, dst], check=True)
    return head(dst)


def head(c):
    return subprocess.run(['git', '-C', c, 'rev-parse', 'HEAD'], check=True, capture_output=True, env=ENV_NOFETCH).stdout.decode().strip()


def ls(c, path, recursive=True):
    """回傳 [(路徑, blob sha)]，只列檔案。只讀目錄表，不抓內容。"""
    a = ['git', '-C', c, '-c', 'core.quotePath=false', 'ls-tree', 'HEAD']
    if recursive:
        a.insert(-1, '-r')
    r = subprocess.run(a + [path], check=True, capture_output=True, env=ENV_NOFETCH).stdout.decode()
    out = []
    for ln in r.splitlines():
        meta, p = ln.split('\t', 1)
        m = meta.split()
        if m[1] == 'blob':
            out.append((p, m[2]))
    return out


def fetch_shas(c, shas):
    """批次把這些 blob 抓回來（同 git 自己補抓缺檔的做法，但一次送全部）。"""
    shas = sorted(set(shas))
    if not shas:
        return
    r = subprocess.run(['git', '-C', c, '-c', 'fetch.negotiationAlgorithm=noop', 'fetch', 'origin', '--no-tags',
                        '--no-write-fetch-head', '--recurse-submodules=no', '--filter=blob:none', '--stdin', '-q'],
                       input=('\n'.join(shas) + '\n').encode(), capture_output=True)
    if r.returncode != 0:
        raise RuntimeError('git fetch 失敗：' + r.stderr.decode()[-2000:])


def iter_blobs(c, ents):
    """ents=[(路徑, sha)]；依序 yield (路徑, bytes)。串流讀，不把全部內容放進記憶體。"""
    p = subprocess.Popen(['git', '-C', c, 'cat-file', '--batch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=ENV_NOFETCH)
    import threading
    def feed():
        p.stdin.write(('\n'.join(s for _, s in ents) + '\n').encode()); p.stdin.close()
    t = threading.Thread(target=feed, daemon=True); t.start()
    for path, sha in ents:
        hd = p.stdout.readline().decode().split()
        if len(hd) != 3 or hd[0] != sha or hd[1] != 'blob':
            raise RuntimeError(f'讀不到內容（沒抓到？）：{path} {hd}')
        n = int(hd[2]); body = p.stdout.read(n); p.stdout.read(1)
        yield path, body
    p.wait()


def sha256_file(f):
    h = hashlib.sha256()
    with open(f, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()
