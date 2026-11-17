#!/usr/bin/env python3
"""S46 事後描述（未登記、不影響判定）：各格 20 條路徑在 2016-03～2018-12、2019～2023、2024 起三段與逐年的年化中位數，現行在每段的名次。
重用 run.py 的載入與訊號（逐行複製）。
原 run.py 說明：S46 進場動能門檻 × 回看期 × 深度 家族檢查（規格 SPEC.json、判定 PREREG.json，寫於任何報酬計算之前）。
用法：python3 -I -B run.py [--quick]（--quick 只跑現行格驗收）
鏈＝agents/deadline_engine_fix/run_engine.py 的 ALL：s20lib→s17lib→s10lib、EXRIGHT_FIX／DEADLINE_FIX／INS15_MASK 全開、e50_fix_suspension、
股池 twse & ~創新板、STD SURGE、make_signal_s10(depth, mom_lo, nsel=10)、E0、simulate_s10(strict, sat_w=0.7, W10, lock=False)。
回看期 L ≠ 20：產生訊號時把 P.RET20 暫時換成 RET_L，產生完換回。暫存 scratchpad/s46/（結束刪除）。不改任何既有檔案。"""
import sys, json, time, resource, gc, itertools, hashlib
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
TMP = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s46')
CACHE = TMP / 'cache'
CACHE.mkdir(parents=True, exist_ok=True)
QUICK = '--quick' in sys.argv
t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

sys.path.insert(0, str(AG / 's20_persistent_growth'))
import s20lib                                              # noqa: E402
from s20lib import s17lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402
for mod, attr in ((s20lib, 'S20'), (s17lib, 'S17'), (s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'), (s7lib, 'S7'), (s6lib, 'S6'), (s5lib, 'S5'),
                  (s4lib, 'S4'), (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP')):
    setattr(mod, attr, CACHE)
assert cslib.DEADLINE_FIX is False and cslib.INS15_MASK is False
cslib.EXRIGHT_FIX = True

P = s20lib.load()
assert P.exright_fix
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), '面板不符'
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
del INNOV
F = s17lib.rev_family(P, defs=['STD'])                     # 開關關閉下建（同 run_engine）
s17lib.e50_fix_suspension(P)
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq'):
    if hasattr(P, a):
        delattr(P, a)
S0 = F['S']['STD']; months = list(S0.index)
del F; gc.collect()
cslib.DEADLINE_FIX = True; cslib.INS15_MASK = True         # 正式設定：三個全開
AV = cslib.rev_avail_days(P, months, True)
RT = dict(S=S0, av=AV)
RET = {20: P.RET20}
for L in (10, 40):
    RET[L] = P.G / cslib.shift(P.G, L) - 1
print('載入', f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)

lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
TB = P.IDX[-1]; D19 = lastb('2019-01-01'); D24 = lastb('2024-01-01')
FULL0 = P.IDX[P.t0]
CELL = dict(sat_w=0.7, sizing='W10', lock=False)
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
GATES = (0.0, 0.05, 0.10, 0.15, 0.20); LBS = (10, 20, 40); DEPTHS = (8, 10, 12)
CUR = (0.10, 20, 10)
cells = [CUR] if QUICK else list(itertools.product(GATES, LBS, DEPTHS))
cid = lambda c: f'g{int(round(c[0]*100))}_L{c[1]}_D{c[2]}'


def signal(c, shift):
    g, L, D = c
    orig = P.RET20
    P.RET20 = RET[L]
    try:
        sg = s10lib.make_signal_s10(P, U, RT, shift=shift, depth=D, mom_lo=g, nsel=10)
    finally:
        P.RET20 = orig
    return sg


def metrics(nav):
    return dict(since2019=stlib.cagr(nav, D19, TB), full=stlib.cagr(nav, FULL0, TB), maxdd_full=stlib.maxdd(nav, FULL0, TB),
                maxdd_2019=stlib.maxdd(nav, D19, TB), seg19_23=stlib.cagr(nav, D19, D24), seg24=stlib.cagr(nav, D24, TB))



SEG = {'2016-03~2018-12': ('2016-02-26', '2018-12-28'), '2019~2023': (None, None), '2024起': (None, None)}
def segs(nav):
    o = {'2016-03~2018-12': stlib.cagr(nav, lastb('2016-03-01'), D19), '2019~2023': stlib.cagr(nav, D19, D24), '2024起': stlib.cagr(nav, D24, TB)}
    for y in range(2016, 2027):
        a = lastb(f'{y}-01-01'); b = lastb(f'{y+1}-01-01') if y < 2026 else TB
        o[str(y)] = float(nav.loc[b] / nav.loc[a] - 1)
    return o
R = {}
for c in cells:
    k = cid(c); paths = []
    for s in range(4):
        sg = signal(c, s); S2, ekw = s10lib.variant_S(sg, 'E0')
        for cap in CAPS:
            nav, lg, st = s10lib.simulate_s10(P, S2, U, cost=COSTS['strict'], capital=cap, **CELL, **ekw)
            paths.append(segs(nav))
        del sg, S2
    R[k] = {f: float(np.median([p[f] for p in paths])) for f in paths[0]}
    print(k, flush=True)
fields = list(R[cid(CUR)])
rank = {f: int(1 + sum(R[k][f] > R[cid(CUR)][f] for k in R)) for f in fields}
fmed = {f: float(np.median([R[k][f] for k in R])) for f in fields}
out = dict(note='事後描述，未登記，不影響判定；數字＝各格 20 條路徑中位數（段年化／逐年報酬）', current_rank=rank, family_median=fmed,
           current={f: R[cid(CUR)][f] for f in fields}, current_minus_median_pp={f: (R[cid(CUR)][f] - fmed[f]) * 100 for f in fields}, cells=R,
           run_sha256=sha(Path(__file__)))
(HERE / 'results_posthoc.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
for f in fields:
    print(f, rank[f], round(R[cid(CUR)][f]*100, 2), round(fmed[f]*100, 2), round(out['current_minus_median_pp'][f], 2))
