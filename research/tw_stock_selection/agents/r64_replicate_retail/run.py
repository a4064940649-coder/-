#!/usr/bin/env python3
"""R64：獨立重現 S64（散戶行為代價）與 S62（小資金）會寫進散戶建議表的關鍵數字。
用法：python3 -I -B run.py [--quick]（--quick：亂數情境只抽 10 次）。沒有讀 s62lib／s64lib／兩題 run.py。"""
import sys, json, time, resource, gc, math, shutil, csv
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r64lib as L                                          # noqa: E402
from r64lib import s10lib, COSTS, simulate_r64, cagr, maxdd  # noqa: E402

QUICK = '--quick' in sys.argv
ND = 10 if QUICK else 200
t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P, U, RT = L.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22')
TB = P.IDX[-1]; D19 = P.IDX[P.IDX < pd.Timestamp('2019-01-01')][-1]; FULL0 = P.IDX[P.t0]
iD19 = int(P.IDX.get_loc(D19))
assert str(D19.date()) == '2018-12-28' and iD19 == 980
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6); SHIFTS = (0, 1, 2, 3)
KW = dict(sat_w=0.7, sizing='W10', lock=False, cost=COSTS['strict'])
SG = {s: L.build_sig(P, U, RT, s) for s in SHIFTS}
E50 = pd.Series(P.E50, index=P.IDX)
E50_19 = cagr(E50, D19, TB)
print('載入', f'{time.time()-t0:.0f}s', rss(), 'MB', 'E50_19', E50_19, flush=True)
pp = lambda x: round(x * 100, 4)
med = lambda v: float(np.median(np.asarray(v, float)))
res = dict(meta=dict(quick=QUICK, draws=ND, e50_since2019=E50_19))


def met(nav):
    return dict(c19=cagr(nav, D19, TB), cfull=cagr(nav, FULL0, TB), ddfull=maxdd(nav, FULL0, TB), dd19=maxdd(nav, D19, TB))


# ---------------- 第 0 步：逐位重現現行格＋副本驗收
BASE = {}; ok_copy = True
for s in SHIFTS:
    for cap in CAPS:
        nav, lg, st = s10lib.simulate_s10(P, SG[s], U, capital=cap, **KW)
        BASE[(s, cap)] = met(nav)
        nav2, lg2, _ = simulate_r64(P, SG[s], U, capital=cap, **KW)
        ok_copy &= bool(np.array_equal(nav.to_numpy(), nav2.to_numpy(), equal_nan=True) and len(lg) == len(lg2))
        if s == 0 and cap == 1e6:
            std = dict(since2019=BASE[(s, cap)]['c19'], full=BASE[(s, cap)]['cfull'], maxdd=BASE[(s, cap)]['ddfull'], n_closed=int((lg['why'] != 'open_end').sum()))
            # 永不觸發的 stop／gap 也應逐位相同
            nav3, _, st3 = simulate_r64(P, SG[s], U, capital=cap, stop=(0.99, 6), gapx=10.0, **KW)
            ok_copy &= bool(np.array_equal(nav.to_numpy(), nav3.to_numpy(), equal_nan=True))
            nav4, _, _ = simulate_r64(P, SG[s], U, capital=cap, start_t=None, fee_rate=0.001425 * 0.28, min_odd=20.0, odd_from='hist', sweep_min=20000.0, **KW)
            ok_copy &= bool(np.array_equal(nav.to_numpy(), nav4.to_numpy(), equal_nan=True))
TARGET = dict(since2019=0.4875617247721724, n_closed=381)
res['step0'] = dict(std=std, target=TARGET, exact=bool(std['since2019'] == TARGET['since2019'] and std['n_closed'] == 381), copy_identical=ok_copy,
                    base_median_c19=med([b['c19'] for b in BASE.values()]), base_median_ddfull=med([b['ddfull'] for b in BASE.values()]),
                    base_median_dd19=med([b['dd19'] for b in BASE.values()]))
print('第 0 步', res['step0'], f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)
assert res['step0']['exact'] and ok_copy, '第 0 步不符'
PK = [(s, cap) for s in SHIFTS for cap in CAPS]
B19 = np.array([BASE[k]['c19'] for k in PK])


def cell20(fn):
    """fn(s, cap) → nav, st；回傳 20 條統計。"""
    M = []; extra = []
    for s, cap in PK:
        nav, st = fn(s, cap); M.append(met(nav)); extra.append(st)
    c = np.array([m['c19'] for m in M]); d = c - B19
    ddf = np.array([m['ddfull'] for m in M]) - np.array([BASE[k]['ddfull'] for k in PK])
    dd19 = np.array([m['dd19'] for m in M]) - np.array([BASE[k]['dd19'] for k in PK])
    return dict(c19_median=med(c), diff_median_pp=med(d) * 100, n_neg=int((d < 0).sum()), cfull_median=med([m['cfull'] for m in M]),
                ddfull_median=med([m['ddfull'] for m in M]), dd19_median=med([m['dd19'] for m in M]),
                ddfull_diff_median_pp=med(ddf) * 100, n_ddfull_shallower=int((ddf > 0).sum()), dd19_diff_median_pp=med(dd19) * 100,
                n_dd19_shallower=int((dd19 > 0).sum()), excess_vs_0050_pp=(med(c) - E50_19) * 100), extra


# ---------------- S64 ②：回落停手
res['s64_stop'] = {}
for X in (0.15, 0.20, 0.25):
    for M in (3, 6, 12):
        def fn(s, cap):
            nav, lg, st = simulate_r64(P, SG[s], U, capital=cap, stop=(X, M), **KW)
            return nav, st
        r, ex = cell20(fn)
        ev = [e for st in ex for e in st['stop_ev']]
        r['stops_per_path_median'] = med([len(st['stop_ev']) + (st['stopped_end'] is not None) for st in ex])
        r['resume_time'] = sum(1 for e in ev if e[2] == 'time'); r['resume_newhigh'] = sum(1 for e in ev if e[2] == 'newhigh')
        r['halt_day_frac_median'] = med([(sum(b - a for a, b, _ in st['stop_ev']) + ((P.T - 1 - st['stopped_end']) if st['stopped_end'] is not None else 0)) / (P.T - 1 - P.t0) for st in ex])
        res['s64_stop'][f'X{int(X*100)}_M{M}'] = r
        print('stop', X, M, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, f'{time.time()-t0:.0f}s', flush=True)

# ---------------- S64 ④：開盤漲超過 x 不買
res['s64_gap'] = {}
for x in (0.03, 0.05):
    def fn(s, cap):
        nav, lg, st = simulate_r64(P, SG[s], U, capital=cap, gapx=x, **KW)
        return nav, st
    r, ex = cell20(fn)
    r['blk_gap_median'] = med([st.get('blk_gap', 0) for st in ex])
    res['s64_gap'][f'gap{int(x*100)}'] = r
    print('gap', x, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, f'{time.time()-t0:.0f}s', flush=True)


def rand_summary(C):
    """C：draws × 20 的 2019 起年化。"""
    D = C - B19[None, :]
    pm = np.median(C, axis=1)
    return dict(draws=int(C.shape[0]), c19_median_of_draw_medians=med(pm), c19_p10=float(np.percentile(pm, 10)), c19_p90=float(np.percentile(pm, 90)),
                diff_median_all_pp=med(D) * 100, diff_median_of_draw_medians_pp=med(np.median(D, axis=1)) * 100,
                diff_of_medians_pp=(med(pm) - med(B19)) * 100, frac_neg=float((D < 0).mean()))


# ---------------- S64 ③：每月以機率 p 漏做（同一組漏做月份套到 20 條）
MONTHS = [x['month'] for x in SG[0]['info']]
for s in SHIFTS:
    assert set(x['month'] for x in SG[s]['info']) <= set(MONTHS)


def miss_S(S, missed):
    S2 = dict(S); sig = S['sig'].copy(); X = S['X'].copy(); info = []
    for x in S['info']:
        if x['month'] in missed:
            sig[x['k']] = False; X[x['k']] = False
        else:
            info.append(x)
    S2['sig'] = sig; S2['X'] = X; S2['info'] = info
    return S2


# 驗收：空集合逐位相同；全部漏做買進 0 筆
_n, _l, _ = simulate_r64(P, miss_S(SG[0], set()), U, capital=1e6, **KW)
assert abs(cagr(_n, D19, TB) - std['since2019']) == 0
_n, _l, _st = simulate_r64(P, miss_S(SG[0], set(MONTHS)), U, capital=1e6, **KW)
assert _st['n_buy'] == 0
res['s64_miss'] = {}
for p in ((0.25,) if QUICK else (0.25, 0.10, 0.50)):
    C = np.zeros((ND, len(PK)))
    for i in range(ND):
        rng = np.random.default_rng([64, 1, int(round(p * 1000)), i])
        missed = set(m for m, u in zip(MONTHS, rng.random(len(MONTHS))) if u < p)
        for s in SHIFTS:
            S2 = miss_S(SG[s], missed)
            for cap in CAPS:
                nav, _, _ = simulate_r64(P, S2, U, capital=cap, **KW)
                C[i, PK.index((s, cap))] = cagr(nav, D19, TB)
            del S2
        if i % 20 == 0:
            print('miss', p, i, f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)
    res['s64_miss'][f'p{int(p*100)}'] = rand_summary(C)
    print('miss', p, res['s64_miss'][f'p{int(p*100)}'], flush=True)

# ---------------- S64 ①：隨機只買名單一半（每個訊號列獨立抽 ceil(n/2) 檔）
C = np.zeros((ND, len(PK)))
for i in range(ND):
    rng = np.random.default_rng([64, 3, i])
    for s in SHIFTS:
        S2 = dict(SG[s]); sig = SG[s]['sig'].copy()
        for x in SG[s]['info']:
            k = x['k']; tg = np.flatnonzero(sig[k]); n = len(tg)
            if n == 0:
                continue
            keep = rng.choice(n, size=math.ceil(n / 2), replace=False)
            drop = np.ones(n, bool); drop[keep] = False
            sig[k, tg[drop]] = False
        S2['sig'] = sig
        for cap in CAPS:
            nav, _, _ = simulate_r64(P, S2, U, capital=cap, **KW)
            C[i, PK.index((s, cap))] = cagr(nav, D19, TB)
        del S2, sig
    if i % 20 == 0:
        print('rand', i, f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)
res['s64_rand_half'] = rand_summary(C)
print('rand', res['s64_rand_half'], flush=True)

# ---------------- S62：2018-12-28 全現金新起跑
FEE0 = 0.001425
SCEN = {   # 折扣、零股低消、零股每邊額外滑價、零股制度
    'R0': dict(disc=0.28, mo=20.0, oddx=0.0, odd='all'),
    'A': dict(disc=0.28, mo=1.0, oddx=0.003, odd='all'),
    'B': dict(disc=0.6, mo=1.0, oddx=0.003, odd='all'),
    'C': dict(disc=1.0, mo=20.0, oddx=0.006, odd='all'),
    'D': dict(disc=0.6, mo=20.0, oddx=0.0, odd='never'),
    'B_hist': dict(disc=0.6, mo=1.0, oddx=0.003, odd='hist'),
}
res['s62'] = {}


def s62_cell(sc, base, sweep_mode='path'):
    c = SCEN[sc]; out = []; blk = []; npos = []
    for s in SHIFTS:
        for m in (0.8, 0.9, 1.0, 1.1, 1.2):
            cap = base * m
            sw = {'path': cap * 0.02, 'nominal': base * 0.02, 'fixed': 20000.0}[sweep_mode]
            nav, lg, st = simulate_r64(P, SG[s], U, sat_w=0.7, sizing='W10', lock=False, capital=cap, cost=dict(cmult=1.0, xslip=0.003, oddx=c['oddx']),
                                       start_t=iD19, fee_rate=FEE0 * c['disc'], min_odd=c['mo'], odd_from=c['odd'], sweep_min=sw)
            assert nav.index[0] == D19 and nav.iloc[0] == cap
            out.append(cagr(nav, D19, TB)); npos.append(st['avg_npos'])
            f = st['blk_lot'] + st['blk_min'] + st['blk_nocash']
            blk.append(f / max(f + st['n_buy'], 1))
    v = np.array(out)
    return dict(c19_median=med(v), c19_min=float(v.min()), c19_max=float(v.max()), excess_pp=(med(v) - E50_19) * 100, n_beat_0050=int((v > E50_19).sum()),
                blocked_ratio_median=med(blk), avg_npos_median=med(npos))


for base in (5e4, 1e5, 2e5, 3e5, 5e5, 1e6):
    for sc in SCEN:
        r = s62_cell(sc, base)
        res['s62'][f'{int(base/1e4)}萬|{sc}'] = r
    print('s62', base, {sc: (round(res['s62'][f'{int(base/1e4)}萬|{sc}']['c19_median'] * 100, 2), round(res['s62'][f'{int(base/1e4)}萬|{sc}']['excess_pp'], 2)) for sc in SCEN}, f'{time.time()-t0:.0f}s', flush=True)
res['s62_sens_sweep'] = {f'10萬|B|sweep={m}': s62_cell('B', 1e5, m) for m in ('nominal', 'fixed')}
res['s62_sens_sweep'].update({f'10萬|D|sweep={m}': s62_cell('D', 1e5, m) for m in ('nominal', 'fixed')})

# ---------------- 對照表
g = res
ROWS = [
    ('0', '現行 100 萬 2019 起年化（%）', 48.76, std['since2019'] * 100, 0.005),
    ('0', '現行 100 萬已出場筆數', 381, std['n_closed'], 0),
    ('1', 'S64 隨機只買一半：2019 起（%）', 40.06, g['s64_rand_half']['c19_median_of_draw_medians'] * 100, 0.6),
    ('1', 'S64 隨機只買一半：相對照做（百分點）', -6.00, g['s64_rand_half']['diff_median_all_pp'], 0.6),
    ('2', 'S64 回落 20% 停手等 6 個月：2019 起（%）', 41.49, g['s64_stop']['X20_M6']['c19_median'] * 100, 0.3),
    ('2', 'S64 回落 20% 停手等 6 個月：相對照做（百分點）', -4.76, g['s64_stop']['X20_M6']['diff_median_pp'], 0.3),
    ('3', 'S64 漏做 25%：2019 起（%）', 43.79, g['s64_miss']['p25']['c19_median_of_draw_medians'] * 100, 0.6),
    ('3', 'S64 漏做 25%：相對照做（百分點）', -2.12, g['s64_miss']['p25']['diff_median_all_pp'], 0.6),
    ('4', 'S64 開盤漲超過 3% 不買：2019 起（%）', 45.57, g['s64_gap']['gap3']['c19_median'] * 100, 0.3),
    ('4', 'S64 開盤漲超過 3% 不買：相對照做（百分點）', -1.63, g['s64_gap']['gap3']['diff_median_pp'], 0.3),
    ('5', 'S62 10 萬 B 基準：2019 起（%）', 43.57, g['s62']['10萬|B']['c19_median'] * 100, 0.3),
    ('5', 'S62 10 萬 B 基準：對 0050 超額（百分點）', 14.66, g['s62']['10萬|B']['excess_pp'], 0.3),
    ('6', 'S62 10 萬只買整張：2019 起（%）', 29.38, g['s62']['10萬|D']['c19_median'] * 100, 0.3),
    ('6', 'S62 10 萬只買整張：對 0050 超額（百分點）', 0.47, g['s62']['10萬|D']['excess_pp'], 0.3),
    ('7', 'S62 100 萬 R0：2019 起（%）', 46.70, g['s62']['100萬|R0']['c19_median'] * 100, 0.3),
]
cmp_ = []
for no, name, claim, mine, tol in ROWS:
    d = mine - claim
    cmp_.append(dict(編號=no, 項目=name, 原宣稱=claim, 重現值=round(mine, 4), 差=round(d, 4), 容許=tol, 符合=bool(abs(d) <= tol + 1e-9)))
res['compare'] = cmp_
res['all_pass'] = bool(all(r['符合'] for r in cmp_))
res['peak_mb'] = rss(); res['secs'] = round(time.time() - t0)
sfx = '_quick' if QUICK else ''
json.dump(res, open(HERE / f'results{sfx}.json', 'w'), ensure_ascii=False, indent=1, default=str)
with open(HERE / f'compare{sfx}.csv', 'w', newline='', encoding='utf-8-sig') as fh:
    w = csv.DictWriter(fh, fieldnames=list(cmp_[0])); w.writeheader(); w.writerows(cmp_)
for r in cmp_:
    print(r)
print('all_pass', res['all_pass'], res['secs'], 's', res['peak_mb'], 'MB')
if not QUICK:
    shutil.rmtree(L.TMP, ignore_errors=True)
