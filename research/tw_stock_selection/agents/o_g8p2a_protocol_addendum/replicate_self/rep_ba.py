#!/usr/bin/env python3
"""自我複製（不是 G6）：換 seed 重算 b_a(A1) 的驅動格 q_0.9525（MA3、MA2、AR10 × 9σ × s 0.78，P＝200,000），估 b_a 的 MC 變異。
用法：python3 -I -B replicate_self/rep_ba.py"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import numpy as np
import g8alib as L
out = {}
for seed in (2026110791, 2026110792, 2026110793):
    qs = {}
    for d in ('MA3', 'MA2', 'AR10'):
        for si in range(9):
            X = L.gen(d, 0.78 - L.DW, L.SIG_GRID[si], 200_000, 24, L.cell_rng(seed, d, si, 4, 2, 0))
            qs[f'{d}|{L.SIG_LABEL[si]}|0.78'] = float(np.quantile(L.bstar_combined(L.all_stats(X), 'A1'), 0.9525))
    k = max(qs, key=qs.get); out[str(seed)] = dict(b_a=max(L.B_V2, qs[k]), cell=k)
ba = [v['b_a'] for v in out.values()]
prot = json.loads((HERE / 'G8_PROTOCOL_v2a.json').read_text(encoding='utf-8'))['params']['b_a']
res = dict(seeds=out, mean=float(np.mean(ba)), sd=float(np.std(ba, ddof=1)), protocol_b_a=prot, diff=float(np.mean(ba) - prot),
           note='三組 seed 的 b_a（max over 27 個驅動格）；sd 是 MC 變異（含 max 運算），不是 G6 獨立複製')
(HERE / 'replicate_self' / 'rep_ba.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
print(res)
