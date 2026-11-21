#!/usr/bin/env python3
"""兩來源比對共用函式（只讀、不連網）。GitHub 倉庫一律讀 git HEAD、GIT_NO_LAZY_FETCH=1（本機沒有的檔回 None，不向遠端抓）。"""
import csv, gzip, io, os, re, subprocess
from pathlib import Path

ENV = dict(os.environ, GIT_NO_LAZY_FETCH='1', GIT_CONFIG_COUNT='1', GIT_CONFIG_KEY_0='protocol.allow', GIT_CONFIG_VALUE_0='never')
IS4 = re.compile(r'^[1-9]\d{3}$')
EX_COLS = ['date', 'code', 'prev_close', 'ref_price', 'value', 'type']


def git(tsd, *a):
    return subprocess.run(['git', '-C', str(tsd), *a], capture_output=True, env=ENV)


def ls(tsd, sub):
    out = {}
    for ln in git(tsd, 'ls-tree', '-r', 'HEAD', sub).stdout.decode().splitlines():
        meta, path = ln.split('\t', 1); out[path] = meta.split()[2]
    return out


def blob(tsd, sha):
    r = git(tsd, 'cat-file', 'blob', sha)
    return r.stdout if r.returncode == 0 else None


def head(tsd):
    r = git(tsd, 'log', '-1', '--format=%H %cI')
    return r.stdout.decode().split() if r.returncode == 0 else (None, None)


def last_daily(tsd):
    """倉庫最新一天日 K 的日期（看檔名，不讀內容）。"""
    d = sorted(p.rsplit('/', 1)[1][:10] for p in ls(tsd, 'data/universe/daily') if re.search(r'/\d{4}-\d{2}-\d{2}\.csv$', p))
    return d[-1] if d else None


def num(s):
    try:
        return float(str(s).replace(',', '').strip())
    except ValueError:
        return float('nan')


# ---------------------------------------------------------------- 月營收
def repo_revenue(tsd, month, market='twse'):
    """回傳 ({代號: 列}, 來源說明)；檔案不在或讀不到回 (None, 原因)。"""
    path = f'data/mops/revenue_hist/{month}_{market}.csv'
    f = ls(tsd, 'data/mops/revenue_hist')
    if path not in f:
        return None, f'倉庫 HEAD 沒有 {path}'
    b = blob(tsd, f[path])
    if b is None:
        return None, f'{path} 本機讀不到內容'
    return {r['stock_id'].strip(): r for r in csv.DictReader(io.StringIO(b.decode('utf-8-sig')))}, f'{path}@{f[path][:12]}'


def csv_revenue(path):
    with open(path, newline='', encoding='utf-8-sig') as fh:
        return {r['stock_id'].strip(): r for r in csv.DictReader(fh)}


def cmp_revenue(A, B, only4=False):
    """A＝GitHub 倉庫、B＝官方。比當月營收（數值相等才算一致）；另外數其他欄位不同的家數。"""
    if only4:
        A = {c: r for c, r in A.items() if IS4.match(c)}; B = {c: r for c, r in B.items() if IS4.match(c)}
    both = sorted(set(A) & set(B)); diff = []; other = {}
    for c in both:
        a, b = num(A[c].get('當月營收', '')), num(B[c].get('當月營收', ''))
        if not (a == b or (a != a and b != b)):
            diff.append(dict(code=c, name=A[c].get('name', ''), repo=A[c].get('當月營收', ''), official=B[c].get('當月營收', '')))
        for k in A[c]:
            if k not in ('當月營收', 'name', 'market', 'period', '產業別') and k in B[c] and A[c][k] != B[c][k] and num(A[c][k]) != num(B[c][k]):
                other.setdefault(k, []).append(dict(code=c, repo=A[c][k], official=B[c][k]))
    nm = lambda S, c: (S[c].get('name') or '').strip()
    return dict(n_repo=len(A), n_official=len(B), n_both=len(both),
                only_repo=[f'{c} {nm(A, c)}' for c in sorted(set(A) - set(B))], only_official=[f'{c} {nm(B, c)}' for c in sorted(set(B) - set(A))],
                revenue_diff=diff, other_col_diff={k: v[:20] for k, v in other.items()}, n_other_col_diff={k: len(v) for k, v in other.items()})


# ---------------------------------------------------------------- 除權息
def archive_exright(d, start, end):
    """讀封存 twse-exright/YYYY/YYYY-MM.csv.gz 在 [start, end] 的列；回傳 (列, 封存最後一個事件日)。"""
    rows = []; last = ''
    for p in sorted(Path(d).glob('*/*.csv.gz')):
        ym = p.name[:7]
        with gzip.open(p, 'rt', encoding='utf-8') as fh:
            rr = list(csv.DictReader(fh))
        if rr:
            last = max(last, max(r['date'] for r in rr))
        if start[:7] <= ym <= end[:7]:
            rows += [r for r in rr if start <= r['date'] <= end]
    return rows, last


def csv_exright(path):
    with open(path, newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def _trim(s):
    s = str(s).strip().replace(',', '')
    if not re.fullmatch(r'-?\d+(\.\d+)?', s):
        return s
    if '.' in s:
        s = s.rstrip('0').rstrip('.')
    return s


def repo_exright(tsd, start, end):
    """GitHub 倉庫 data/universe/exright/<日>.csv（欄 date,stock_id,…,pre_close,ref_price,value,kind）→ 封存欄名。回傳 (列, 讀不到的日子)。"""
    rows = []; bad = []
    for p, sha in sorted(ls(tsd, 'data/universe/exright').items()):
        d = p.rsplit('/', 1)[1][:10]
        if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', d) or not (start <= d <= end):
            continue
        b = blob(tsd, sha)
        if b is None:
            bad.append(d); continue
        for r in csv.DictReader(io.StringIO(b.decode('utf-8-sig'))):
            rows.append(dict(date=r['date'], code=r.get('stock_id', r.get('code', '')), prev_close=_trim(r.get('pre_close', r.get('prev_close', ''))),
                             ref_price=_trim(r['ref_price']), value=_trim(r['value']), type=r.get('kind', r.get('type', ''))))
    return rows, bad


def cmp_events(A, B, cols=EX_COLS, key=('date', 'code')):
    """逐筆比對（同一天同代號多筆時依全部欄位排序後配對）。A、B 是 dict 列。"""
    def idx(R):
        o = {}
        for r in R:
            o.setdefault(tuple(r[k] for k in key), []).append(tuple(str(r.get(c, '')) for c in cols))
        return {k: sorted(v) for k, v in o.items()}
    a, b = idx(A), idx(B)
    diff = [dict(key=' '.join(k), a=[list(x) for x in a[k]], b=[list(x) for x in b[k]]) for k in sorted(set(a) & set(b)) if a[k] != b[k]]
    return dict(n_a=len(A), n_b=len(B), n_same=sum(len(a[k]) for k in set(a) & set(b) if a[k] == b[k]),
                only_a=[' '.join(k) for k in sorted(set(a) - set(b))], only_b=[' '.join(k) for k in sorted(set(b) - set(a))], diff=diff)


# ---------------------------------------------------------------- 官方快照索引
def latest_snapshot(root, kind, tag):
    """index.csv 裡這個 tag 最後一次成功（200）的紀錄；回傳 (紀錄, 快照檔路徑) 或 (None, None)。"""
    idx = Path(root) / kind / 'index.csv'
    if not idx.exists():
        return None, None
    with open(idx, newline='', encoding='utf-8') as fh:
        rows = [r for r in csv.DictReader(fh) if r['tag'] == tag and r['http_status'] == '200' and r['sha256']]
    if not rows:
        return None, None
    r = rows[-1]
    return r, Path(root) / kind / (r['stored_file'] or r['same_as'])


def covering_snapshot(root, kind, prefix, need_end):
    """找視窗結束日 ≥ need_end 的最後一次成功快照（tag＝<prefix>_<起>_<迄>）；回傳 (紀錄, 路徑, 起, 迄) 或 (None, None, None, None)。"""
    idx = Path(root) / kind / 'index.csv'
    if not idx.exists():
        return None, None, None, None
    best = None
    with open(idx, newline='', encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            m = re.fullmatch(re.escape(prefix) + r'_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})', r['tag'])
            if m and r['http_status'] == '200' and r['sha256'] and m.group(2) >= need_end and m.group(1) <= need_end:
                best = (r, Path(root) / kind / (r['stored_file'] or r['same_as']), m.group(1), m.group(2))
    return best or (None, None, None, None)
