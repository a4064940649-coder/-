#!/usr/bin/env python3
"""自我複製（非獨立複製員）：只讀 G8_PROTOCOL.json 的文字與參數，不 import g8lib，重寫模擬與規則，重跑 OC（P＝20,000、另一組 seed），
與協定內 OC 表比較是否在 2 倍 MC 標準誤（合併）以內。用法：python3 -I -B rep_oc_from_protocol.py"""
import json, math, hashlib
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / 'G8_PROTOCOL.json').read_text(encoding='utf-8'))
sd, c, b, cd = (P['params'][k] for k in ('sigma_design', 'c', 'b', 'c_d'))
e = np.array(P['data']['resid_post2019']); e = e - e.mean()
H, NS = 24, 20_000


def paths(dgp, a, rng):
    if dgp == 'D1': return a + sd * rng.normal(size=(NS, H))
    if dgp == 'D2': return a + sd * rng.standard_t(4, size=(NS, H)) / math.sqrt(2)
    n = len(e); I = np.zeros((NS, H), int); I[:, 0] = rng.integers(0, n, NS)
    for t in range(1, H):
        new = rng.random(NS) < 1 / 6
        I[:, t] = np.where(new, rng.integers(0, n, NS), (I[:, t - 1] + 1) % n)
    return a + e[I]


def decide(x):
    """單一路徑：回傳 ('up'|'down'|None, 月)。"""
    sig = c * sd; refit = False; S = 0.0
    for n in range(1, H + 1):
        S += x[n - 1]
        look = n in (12, 18, 24)
        if look and not refit:
            s = np.std(x[:n], ddof=1)
            if s > 1.5 * sd: refit = True; sig = max(c * sd, s)
        z = S / (sig * math.sqrt(n))
        if look and z <= -cd: return 'down', n
        if n in (12, 24) and z >= b * math.sqrt(24 / n): return 'up', n
    return None, 0


rows = {}; worst = 0.0; ok = True
for gi, g in enumerate(['D1', 'D2', 'D3']):
    for a in (-1.0, 0.0, 1.0, 2.0):
        rng = np.random.default_rng([777, gi, int((a + 3) * 10)])
        X = paths(g, a, rng)
        res = [decide(x) for x in X]
        pu = np.mean([r[0] == 'up' for r in res]); pd_ = np.mean([r[0] == 'down' for r in res])
        ref = P['oc']['table_primary'][f'{g}|{a:+.1f}']['24']
        Pf = P['oc']['P']
        for nm, mine, theirs in (('up', pu, ref['up']), ('down', pd_, ref['down'])):
            se = math.sqrt(max(mine * (1 - mine), 1e-9) / NS + max(theirs * (1 - theirs), 1e-9) / Pf)
            zz = abs(mine - theirs) / se
            worst = max(worst, zz); ok &= zz <= 2.0
            rows[f'{g}|{a:+.1f}|{nm}'] = dict(rep=round(float(mine), 4), protocol=theirs, z=round(float(zz), 2))
out = dict(protocol_sha256=hashlib.sha256((HERE / 'G8_PROTOCOL.json').read_bytes()).hexdigest(), P=NS, all_within_2se=bool(ok), worst_z=round(worst, 2), cells=rows,
           note='同一研究員以協定文字重寫（不 import g8lib）；不是獨立複製員')
print(json.dumps(out, ensure_ascii=False))
res = json.loads((HERE / 'results.json').read_text(encoding='utf-8')); res['SELF_REPLICATION_OC'] = out
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
