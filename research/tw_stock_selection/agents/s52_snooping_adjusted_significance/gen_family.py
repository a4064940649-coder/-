#!/usr/bin/env python3
"""S52：重建近親族群 S46（45 格）與 S48（24 格）的主路徑日淨值（100 萬、位移 +0），並讀 S45、S47 既有 navs.csv.gz，
輸出四家族＋正式 E0＋0050 的月底淨值 family_monthly_nav.csv.gz（本資料夾）。
引擎鏈完全照抄 agents/s46_momentum_gate_family/run.py 與 agents/s48_exit_rule_family/run.py（唯讀 import，不改任何檔）。
用法：python3 -I -B gen_family.py"""
import sys, json, time, resource, gc, itertools, hashlib
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
ROOT = AG.parent.parent.parent                      # /home/user/-
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s52')
CACHE = TMP / 'cache'
CACHE.mkdir(parents=True, exist_ok=True)
t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for f in ('SPEC', 'PREREG'):
    assert sha(HERE / f'{f}.json') == (HERE / f'{f}.sha256').read_text().split()[0], f'{f}.json 與登記的 sha256 不符'

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
FMA = {}
with np.errstate(invalid='ignore'):
    for M in (20, 40, 60, 120):
        FMA[M] = P.G > stlib.rmean(P.G, M)
assert np.array_equal(FMA[60], P.F_MA60)
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
del INNOV
F = s17lib.rev_family(P, defs=['STD'])
s17lib.e50_fix_suspension(P)
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq'):
    if hasattr(P, a):
        delattr(P, a)
S0 = F['S']['STD']; months = list(S0.index)
del F; gc.collect()
cslib.DEADLINE_FIX = True; cslib.INS15_MASK = True
AV = cslib.rev_avail_days(P, months, True)
RT = dict(S=S0, av=AV)
RET = {20: P.RET20}
for L in (10, 40):
    RET[L] = P.G / cslib.shift(P.G, L) - 1
print('載入', f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)
CELL = dict(sat_w=0.7, sizing='W10', lock=False)
TB = P.IDX[-1]; D19 = P.IDX[P.IDX < pd.Timestamp('2019-01-01')][-1]


def to_series(nav):
    if isinstance(nav, pd.Series):
        return nav.astype(float)
    if isinstance(nav, pd.DataFrame):
        return nav.iloc[:, 0].astype(float)
    a = np.asarray(nav, float)
    return pd.Series(a, index=P.IDX[len(P.IDX) - len(a):])


NAV = {}
# ---- S46：門檻 × 回看期 × 深度
GATES = (0.0, 0.05, 0.10, 0.15, 0.20); LBS = (10, 20, 40); DEPTHS = (8, 10, 12)
for g, L, D in itertools.product(GATES, LBS, DEPTHS):
    orig = P.RET20; P.RET20 = RET[L]
    try:
        sg = s10lib.make_signal_s10(P, U, RT, shift=0, depth=D, mom_lo=g, nsel=10)
    finally:
        P.RET20 = orig
    S2, ekw = s10lib.variant_S(sg, 'E0')
    nav, lg, st = s10lib.simulate_s10(P, S2, U, cost=COSTS['strict'], capital=1e6, **CELL, **ekw)
    k = f'S46:g{int(round(g*100))}_L{L}_D{D}'
    NAV[k] = to_series(nav)
    if (g, L, D) == (0.10, 20, 10):
        cur_since19 = stlib.cagr(nav, D19, TB); cur_ncl = int((lg['why'] != 'open_end').sum())
    del sg, S2; gc.collect()
print('S46 完成', f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)

# ---- S48：排名門檻 N × 出場均線 M（出場矩陣照 S48 build_exits）
NS = (15, 20, 25, 30, None); MS = (20, 40, 60, 120, None)
ALLC = [(n, m) for n in NS for m in MS if not (n is None and m is None)]
sg = s10lib.make_signal_s10(P, U, RT, shift=0, depth=10, mom_lo=0.10, nsel=10)
X = {c: np.zeros((P.T, P.N), bool) for c in ALLC}
ks = []
for m in RT['S'].index:
    k0 = RT['av'][m]
    if k0 >= P.T or k0 < 61:
        continue
    s = RT['S'].loc[m].to_numpy()
    if int((U[k0] & np.isfinite(s)).sum()) < 50:
        continue
    k = k0
    s = cslib.ins15_mask_row(P, m, k, s)
    fin = U[k] & np.isfinite(s)
    ok_ma = fin & P.F_MA60[k]
    o_ma = np.flatnonzero(ok_ma)[np.argsort(-s[ok_ma], kind='stable')]
    sn = np.nan_to_num(s, nan=-np.inf)
    ks.append(int(k))
    for (n, mm) in ALLC:
        keep = fin.copy()
        if n is not None:
            thr = s[o_ma[n - 1]] if len(o_ma) >= n else -np.inf
            keep &= sn >= thr
        if mm is not None:
            keep &= FMA[mm][k]
        X[(n, mm)][k, :] = ~keep
assert ks == [int(x['k']) for x in sg['info']], '訊號列不符'
A1 = bool(np.array_equal(X[(20, 60)], sg['XV']['E0']))
assert A1, 'S48 出場矩陣驗收失敗'
base = dict(sg); del base['XV']
for c in ALLC:
    S2 = dict(base); S2['X'] = X[c]
    nav, lg, st = s10lib.simulate_s10(P, S2, U, cost=COSTS['strict'], capital=1e6, **CELL)
    NAV[f"S48:N{c[0] if c[0] else 'none'}_M{c[1] if c[1] else 'none'}"] = to_series(nav)
del sg, X, base; gc.collect()
print('S48 完成', f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)

# ---- S45、S47 既有日淨值；正式淨值
off = pd.read_csv(ROOT / 'xq_scripts' / 'site' / 'navs_official.csv.gz', index_col=0, parse_dates=True)
D = pd.DataFrame(NAV)
D.index = pd.to_datetime(D.index)
D = D.loc[D.index >= off.index[0]]
for fam, fn in (('S45', 's45_universe_threshold_family'), ('S47', 's47_allocation_family')):
    x = pd.read_csv(AG / fn / 'navs.csv.gz', index_col=0, parse_dates=True).drop(columns=['0050'])
    x.columns = [f'{fam}:{c}' for c in x.columns]
    D = D.join(x, how='outer')
assert D.index.equals(off.index), '日期不符'

# ---- 驗收
ref = json.loads((AG / 'deadline_engine_fix' / 'out' / 'results.json').read_text())
acc = dict(cur_since2019=cur_since19, ref_since2019=ref['standard']['ALL']['since2019'], cur_n_closed=cur_ncl, ref_n_closed=ref['standard']['ALL']['n_closed'])
acc['since2019_bitwise'] = bool(cur_since19 == acc['ref_since2019'] and cur_ncl == acc['ref_n_closed'])
e0 = off['E0|W10_70']
for k in ('S46:g10_L20_D10', 'S48:N20_M60', 'S45:A1000_P10', 'S47:S70_D10'):
    acc[f'maxreldiff_vs_official[{k}]'] = float((D[k] / D[k].iloc[0] * e0.iloc[0] / e0 - 1).abs().max())
acc['ok'] = acc['since2019_bitwise'] and all(v < 1e-7 for kk, v in acc.items() if kk.startswith('maxreldiff'))   # 正式 CSV 四捨五入到 0.01 元（相對 ≈ 5e-9）
print('驗收', json.dumps(acc, ensure_ascii=False), flush=True)
assert acc['ok'], '驗收失敗'

D.insert(0, 'OFFICIAL:E0|W10_70', e0)
D.insert(0, '0050', off['0050'])
ME = D.groupby(D.index.to_period('M')).tail(1)
ME = pd.concat([D.iloc[[0]], ME])                         # 第一列＝2015-07-01 起始基底
ME.to_csv(HERE / 'family_monthly_nav.csv.gz', float_format='%.6f')
(HERE / 'gen_family_check.json').write_text(json.dumps(dict(acceptance=acc, n_cols=int(D.shape[1]), rows=int(ME.shape[0]),
    run_sha256=sha(Path(__file__)), runtime_s=round(time.time() - t0, 1), maxrss_mb=rss()), ensure_ascii=False, indent=1))
print('done', ME.shape, f'{time.time()-t0:.0f}s', rss(), 'MB')
