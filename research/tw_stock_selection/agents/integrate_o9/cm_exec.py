#!/usr/bin/env python3
"""C 腿可執行逐窗報酬：唯讀 import o_capmatched_pit_close/s6_exec.py（它自己斷言 C 的 PREREG 與程式雜湊），PREREG 之前執行，
用它的 run_exec、PLANS、TIERS，逐 j 逐窗輸出報酬（原檔只存彙總）。先用彙總重現 exec_{cfg}.json（差 ≤1e-9）。
python3 -I -B cm_exec.py <a|b>  → out/cm_exec_<cfg>.npz"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
CMD = HERE.parent / 'o_capmatched_pit_close'
CFG = sys.argv[1]
assert CFG in ('a', 'b')
# 在 PREREG 之前執行：只產生 C 腿自己的可執行逐窗報酬（C 研究已公布其彙總），不做任何混合；輸出 npz 的 sha256 登記進 PREREG。
sys.argv = ['s6_exec.py', CFG]
sys.path.insert(0, str(CMD))
import s6_exec as S6
import numpy as np
from multiprocessing import Pool

def job(a):
    k, tn, tier, mode, extra = a
    pl, days, b = S6.PLANS[k]
    r, acc = S6.run_exec(days, pl, tier, mode, extra)
    return k, tn, mode, extra, r, np.asarray(b)

if __name__ == '__main__':
    jobs = [(k, tn, tier, mode, ex) for k in S6.DS for tn, tier, mode in S6.TIERS for ex in (0.0, 0.005)]
    with Pool(4) as pool: res = pool.map(job, jobs, chunksize=4)
    out = {}; chk = {}
    for k, tn, mode, ex, r, b in res:
        tag = 'K5T' if ex == 0 else 'K5T+0.5%'
        out[f'{k}|{tn}|{mode}|{tag}'] = r; out[f'{k}|r5'] = b
        out[f'{k}|days'] = np.array([str(d.date()) if hasattr(d, 'date') else str(d) for d in S6.PLANS[k][1]])
    ref = json.loads((CMD / 'out' / f'exec_{CFG}.json').read_text(encoding='utf-8'))
    mx = 0.0
    for tn, tier, mode in S6.TIERS:
        for tag in ('K5T', 'K5T+0.5%'):
            ks = [k for k in S6.DS if k != 'OFF']
            mm = np.median([(out[f'{k}|{tn}|{mode}|{tag}'] - out[f'{k}|r5']).mean() * 100 for k in ks])
            tt = np.median([S6.P.tstat(out[f'{k}|{tn}|{mode}|{tag}'] - out[f'{k}|r5']) for k in ks])
            R = ref[f'{tn}|{mode}|{tag}']
            mx = max(mx, abs(mm - R['vs0050_mean_pp_jmed']), abs(tt - R['vs0050_t_jmed']))
    print(CFG, 'reproduce exec json max diff', mx, flush=True)
    assert mx <= 1e-9, mx
    np.savez_compressed(HERE / 'out' / f'cm_exec_{CFG}.npz', **out)
    (HERE / 'out' / f'cm_exec_{CFG}_check.json').write_text(json.dumps(dict(reproduce_maxdiff=mx)), encoding='utf-8')
