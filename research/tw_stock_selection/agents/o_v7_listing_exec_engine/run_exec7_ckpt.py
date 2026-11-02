#!/usr/bin/env python3
"""run_exec7.py 的「只加斷點續跑」包裝（2026-11-02 補跑；PREREG 之後新增，不在 PREREG 雜湊清單內——已揭露）。
原因：2026-10-28 的 run_exec7.py 在 exec 17,000／19,520 時因工作階段結束被中止，沒有任何輸出。
作法：讀入 run_exec7.py 原始碼（先斷言 sha256 與 PREREG 登記值相同），只把 run_pool 換成會把每 200 筆結果
寫進 data/exec_ckpt/<tag>/*.pkl、重啟時跳過已完成工作的版本，其餘程式碼（工作清單、模擬、彙整、輸出）一字不改地以 __main__ 執行。
計算結果與一次跑完相同（工作是確定性的；只有平均的加總順序可能不同，差 ~1e-16）。"""
import sys, json, hashlib, pickle, os
from pathlib import Path
HERE = Path(__file__).resolve().parent
SRC = HERE / 'run_exec7.py'
res = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
h = hashlib.sha256(SRC.read_bytes()).hexdigest()
assert res['PREREG']['hashes'][str(SRC)] == h, 'run_exec7.py 與 PREREG 不符'
src = SRC.read_text(encoding='utf-8')
OLD = '''def run_pool(fn, jobs, tag):
    out = []
    with mp.get_context('fork').Pool(4) as P:
        for i, r in enumerate(P.imap_unordered(fn, jobs, chunksize=4)):
            out.append(r)
            if i % 1000 == 0: log(tag, i, '/', len(jobs))
    return out
'''
NEW = '''def run_pool(fn, jobs, tag):
    import pickle as _pk
    ck = C.DATA / 'exec_ckpt' / ('dry_' + tag if DRY else tag); ck.mkdir(parents=True, exist_ok=True)
    kf = (lambda r: (r[0], r[1])) if tag == 'ideal' else (lambda r: r[0])
    done = {}
    for f in sorted(ck.glob('part_*.pkl')):
        for r in _pk.loads(f.read_bytes()): done[repr(kf(r))] = r
    todo = [j for j in jobs if repr(j) not in done]
    log(tag, 'checkpoint loaded', len(done), 'todo', len(todo), '/', len(jobs))
    buf = []; npart = len(list(ck.glob('part_*.pkl')))
    def flush():
        nonlocal buf, npart
        if not buf: return
        tmp = ck / f'tmp_{npart:05d}.pkl'; tmp.write_bytes(_pk.dumps(buf, protocol=4)); tmp.rename(ck / f'part_{npart:05d}.pkl'); npart += 1; buf = []
    if todo:
        with mp.get_context('fork').Pool(4) as P:
            for i, r in enumerate(P.imap_unordered(fn, todo, chunksize=4)):
                done[repr(kf(r))] = r; buf.append(r)
                if len(buf) >= 200: flush()
                if i % 1000 == 0: log(tag, i, '/', len(todo))
        flush()
    return [done[repr(j)] for j in jobs]
'''
assert src.count(OLD) == 1
src = src.replace(OLD, NEW)
g = globals(); g['__file__'] = str(SRC)
exec(compile(src, str(SRC), 'exec'), g)
