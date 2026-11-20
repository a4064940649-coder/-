#!/usr/bin/env python3
"""把官方日 K 倉庫（tsd）的上市除權息事件檔 data/universe/exright/YYYY-MM-DD.csv（讀 git HEAD）
轉成研究中心 TWT49U 封存格式（twse-exright/YYYY/YYYY-MM.csv.gz；欄位 date,code,prev_close,ref_price,value,type）。

用法（python3 -I -B）：
  convert_exright.py verify  <tsd> <封存資料夾 twse-exright> <結果 json>          # 重疊期逐欄比對
  convert_exright.py build   <tsd> <輸出資料夾> <起始日 YYYY-MM-DD> [<結束日>]   # 產生月檔（只寫到輸出資料夾）
只讀 tsd；git 一律 GIT_NO_LAZY_FETCH=1（本機沒有內容的檔記在 unreadable，不當成沒有事件）。
"""
import sys, os, io, csv, gzip, json, subprocess, collections, hashlib
from pathlib import Path

ENV = dict(os.environ, GIT_NO_LAZY_FETCH='1', GIT_CONFIG_COUNT='1', GIT_CONFIG_KEY_0='protocol.allow', GIT_CONFIG_VALUE_0='never')
HEADER = 'date,code,prev_close,ref_price,value,type'


def git(tsd, *a, inp=None):
    return subprocess.run(['git', '-C', str(tsd), *a], input=inp, capture_output=True, env=ENV)


def num(s):
    """倉庫是固定小數位（80.00、1.807197）；封存是去掉尾端 0 的寫法（80、1.807197）。只做字串處理，不經浮點數。"""
    s = s.strip()
    if s == '':
        return ''
    neg = s.startswith('-'); s = s.lstrip('+-')
    if '.' in s:
        s = s.rstrip('0').rstrip('.')
    s = s.lstrip('0') or '0'
    if s.startswith('.'):
        s = '0' + s
    return ('-' if neg and s != '0' else '') + s


def load_repo(tsd):
    """回傳（{日期: [列…]}, 讀不到內容的日期清單, HEAD commit）。一次 cat-file --batch，不逐檔開行程。"""
    head = git(tsd, 'rev-parse', 'HEAD').stdout.decode().strip()
    ls = git(tsd, 'ls-tree', '-r', 'HEAD', 'data/universe/exright').stdout.decode().splitlines()
    ent = []
    for ln in ls:
        meta, path = ln.split('\t'); sha = meta.split()[2]; name = path.rsplit('/', 1)[1]
        if name.endswith('.csv') and name[:4].isdigit():
            ent.append((name[:-4], sha))
    local = set(x.split()[0] for x in git(tsd, 'cat-file', '--batch-check', '--batch-all-objects').stdout.decode().splitlines())
    have = [(d, s) for d, s in ent if s in local]
    unreadable = sorted(d for d, s in ent if s not in local)
    r = git(tsd, 'cat-file', '--batch', inp=('\n'.join(s for _, s in have) + '\n').encode())
    buf = r.stdout; pos = 0; out = {}
    for d, s in have:
        nl = buf.index(b'\n', pos); hd = buf[pos:nl].decode().split()
        assert hd[0] == s and hd[1] == 'blob', hd
        n = int(hd[2]); body = buf[nl + 1:nl + 1 + n].decode('utf-8'); pos = nl + 1 + n + 1
        rows = list(csv.DictReader(io.StringIO(body)))
        for x in rows:
            assert x['date'] == d, (d, x)
        out[d] = rows
    return out, unreadable, head


def convert(rows):
    """倉庫列 → 封存列（字串）。排序：日期、代號（同封存）。"""
    o = [(x['date'], x['stock_id'].strip(), num(x['pre_close']), num(x['ref_price']), num(x['value']), x['kind'].strip()) for x in rows]
    return sorted(o, key=lambda r: (r[0], r[1]))


def read_archive(d):
    out = collections.defaultdict(list)
    for f in sorted(Path(d).glob('*/*.csv.gz')):
        with gzip.open(f, 'rt', encoding='utf-8', newline='') as fh:
            rd = csv.reader(fh); h = next(rd)
            assert ','.join(h) == HEADER, (f, h)
            for r in rd:
                out[r[0]].append(tuple(r))
    return out


def verify(tsd, arc, outp):
    repo, unreadable, head = load_repo(tsd)
    A = read_archive(arc)
    a_last = max(A); a_first = min(A)
    res = dict(tsd_head=head, archive_first=a_first, archive_last=a_last, repo_first_readable=min(repo), repo_last=max(repo),
               repo_unreadable_days=unreadable)
    days = sorted(d for d in repo if d <= a_last)
    n_same = n_diff = n_rows = 0; diffs = []; order_diff = 0
    only_repo_days = []; fld = collections.Counter()
    for d in days:
        conv = convert(repo[d]); arc_rows = A.get(d)
        if arc_rows is None:
            only_repo_days.append(d); continue
        if conv != arc_rows:
            if sorted(conv) == sorted(arc_rows):
                order_diff += 1
            ka = {(r[0], r[1]): r for r in arc_rows}; kc = {(r[0], r[1]): r for r in conv}
            for k in sorted(set(ka) | set(kc)):
                a, c = ka.get(k), kc.get(k)
                if a != c:
                    n_diff += 1
                    if a and c:
                        for i, nm in enumerate(HEADER.split(',')):
                            if a[i] != c[i]:
                                fld[nm] += 1
                    if len(diffs) < 200:
                        diffs.append(dict(key=k, archive=a, converted=c))
                else:
                    n_same += 1
        else:
            n_same += len(conv)
        n_rows += len(arc_rows)
    lo = min(repo)
    only_arc_days = sorted(d for d in A if lo <= d <= a_last and d not in repo and d not in set(unreadable))
    res.update(overlap_days_compared=len(days) - len(only_repo_days), archive_rows_on_compared_days=n_rows, rows_identical=n_same,
               rows_different=n_diff, diff_by_field=dict(fld), days_same_rows_but_order_differs=order_diff,
               days_only_in_repo=only_repo_days, days_only_in_archive_within_repo_range=only_arc_days,
               archive_rows_on_days_only_in_archive=sum(len(A[d]) for d in only_arc_days), diffs_sample=diffs)
    # 整月文字檔逐位元（解壓後）比對：倉庫那個月每天都讀得到才比
    by_m = collections.defaultdict(list)
    for d in sorted(repo):
        by_m[d[:7]].extend(convert(repo[d]))
    un_m = {d[:7] for d in unreadable}
    m_same = m_diff = 0; md = []
    for f in sorted(Path(arc).glob('*/*.csv.gz')):
        m = f.name[:7]
        if m not in by_m or m in un_m or m < lo[:7] or (m == lo[:7] and lo[8:] != '01' and any(d[:7] == m and d < lo for d in A)):
            continue
        txt = gzip.open(f, 'rb').read()
        if txt == render(by_m[m]):
            m_same += 1
        else:
            m_diff += 1; md.append(m)
    res.update(month_files_bytes_identical_after_gunzip=m_same, month_files_different=m_diff, month_files_different_list=md)
    Path(outp).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != 'diffs_sample'}, ensure_ascii=False, indent=1))
    return res


def render(rows):
    return (HEADER + '\n' + ''.join(','.join(r) + '\n' for r in rows)).encode('utf-8')


def build(tsd, outd, start, end='9999-12-31'):
    repo, unreadable, head = load_repo(tsd)
    bad = [d for d in unreadable if start <= d <= end]
    if bad:
        raise SystemExit(f'倉庫有 {len(bad)} 個事件檔本機讀不到內容：{bad}；不產生（不把讀不到當成沒有事件）')
    by_m = collections.defaultdict(list)
    for d in sorted(repo):
        if start <= d <= end:
            by_m[d[:7]].extend(convert(repo[d]))
    man = dict(tsd_head=head, start=start, end=end, repo_last_event_day=max(repo), files=[])
    for m, rows in sorted(by_m.items()):
        p = Path(outd) / m[:4] / f'{m}.csv.gz'; p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, 'wb') as fh:                       # mtime=0、檔名不寫入：同輸入永遠產生同位元組
            with gzip.GzipFile(filename='', mode='wb', fileobj=fh, mtime=0, compresslevel=9) as g:
                g.write(render(rows))
        man['files'].append(dict(path=str(p.relative_to(outd)), rows=len(rows), first=rows[0][0], last=rows[-1][0],
                                 n_4digit=sum(1 for r in rows if len(r[1]) == 4 and r[1][0] in '123456789' and r[1].isdigit()),
                                 sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    (Path(outd) / 'MANIFEST.json').write_text(json.dumps(man, ensure_ascii=False, indent=1))
    print(json.dumps(man, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    a = sys.argv[1:]
    if a and a[0] == 'verify':
        verify(a[1], a[2], a[3])
    elif a and a[0] == 'build':
        build(*a[1:])
    else:
        sys.exit(__doc__)
