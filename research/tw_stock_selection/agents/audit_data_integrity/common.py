#!/usr/bin/env python3
"""audit_data_integrity 共用：路徑、git 唯讀讀檔（不連網）、sha256。一律 python3 -I -B；只讀。"""
import os, io, subprocess, hashlib, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RS = Path('/home/user/-/research/tw_stock_selection')
AG = RS / 'agents'
SITE = Path('/home/user/-/xq_scripts/site')
SCR_OLD = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad')     # 引擎寫死的暫存區
SCR_NEW = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad')
TMP = SCR_NEW / 'audit_integrity'
RAW_REPO = RS / 'raw/jb7k6295r7-bot_tw-stock-data'      # 官方日 K 封存（blobless，.gitignore）
TSD = SCR_OLD / 'tsd'                                    # 每日更新用的同一倉庫較新 clone（暫存區）
TWT49U = RS / 'raw/twt49u/twse-exright'
PRE = RS / 'raw/pre2015/asoul_twse_2007_2015H1.csv.gz'
DP = RS / 'data_panel/panel'
REV_CSV = Path('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv')
E50_CSV = Path('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv')
TWII_CSV = Path('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv')
FWD = AG / 'o_forward_infra_live'
ENG_CACHE = SCR_OLD / 'eng/cache'
DELIST = AG / 'o_infra_v3_rebuild/delist_official.csv'
ENV = dict(os.environ, GIT_NO_LAZY_FETCH='1', GIT_CONFIG_COUNT='1', GIT_CONFIG_KEY_0='protocol.allow', GIT_CONFIG_VALUE_0='never',
           GIT_TERMINAL_PROMPT='0')


def sha_file(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git(repo, *a):
    return subprocess.run(['git', '-C', str(repo)] + list(a), capture_output=True, text=True, env=ENV)


def ls_tree(repo, sub):
    """[(blob sha, path)]；只列 blob。"""
    out = git(repo, 'ls-tree', 'HEAD', sub.rstrip('/') + '/').stdout
    r = []
    for l in out.strip().split('\n'):
        if not l:
            continue
        meta, path = l.split('\t')
        m = meta.split()
        if m[1] == 'blob':
            r.append((m[2], path))
    return r


def local_objects(repo):
    """本機實際有的物件 id 集合（不會觸發抓取）。"""
    out = subprocess.run(['git', '-C', str(repo), 'cat-file', '--batch-all-objects', '--batch-check=%(objectname)', '--unordered'],
                         capture_output=True, text=True, env=ENV).stdout
    return set(out.split())


def read_blobs(repo, items, have=None):
    """items＝[(sha, path)]；逐一產生 (path, bytes 或 None)。have 給了就先濾掉本機沒有的（回 None），保證不連網。"""
    todo = [(s, p) for s, p in items if have is None or s in have]
    miss = [(s, p) for s, p in items if have is not None and s not in have]
    for s, p in miss:
        yield p, None
    if not todo:
        return
    pr = subprocess.Popen(['git', '-C', str(repo), 'cat-file', '--batch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL, env=ENV)
    for s, p in todo:
        pr.stdin.write((s + '\n').encode()); pr.stdin.flush()
        hdr = pr.stdout.readline().decode().split()
        if len(hdr) < 3 or hdr[1] != 'blob':
            yield p, None
            continue
        n = int(hdr[2]); data = pr.stdout.read(n); pr.stdout.read(1)
        yield p, data
    pr.stdin.close(); pr.wait()


def jdump(obj, p):
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str))
