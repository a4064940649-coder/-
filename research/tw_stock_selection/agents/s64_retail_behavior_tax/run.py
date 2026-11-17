#!/usr/bin/env python3
"""S64 散戶行為偏差與稅後報酬：規格 SPEC.json、判讀 PREREG.json（寫於任何情境報酬計算之前，sha256 啟動時核對）。
用法：python3 -I -B run.py [--quick]（--quick 只跑驗收 A1～A5）
暫存 scratchpad/s64/（結束刪除）。"""
import sys, json, time, resource, gc, hashlib
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
QUICK = '--quick' in sys.argv
t0c = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for f in ('SPEC', 'PREREG'):
    assert sha(HERE / f'{f}.json') == (HERE / f'{f}.sha256').read_text().split()[0], f'{f}.json 與登記的 sha256 不符'
sys.path.insert(0, str(HERE))
import s64lib as L                                         # noqa: E402
from s64lib import s10lib, stlib, cslib, COSTS, PATHS, pkey, simulate_s64   # noqa: E402

P, U, RT = L.load()
T = P.T; IDXV = pd.DatetimeIndex(P.IDX)
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
TB = P.IDX[-1]; D19 = lastb('2019-01-01'); FULL0 = P.IDX[P.t0]
i19 = int(IDXV.get_loc(D19)); it0 = int(P.t0)
YR19 = (TB - D19).days / 365.25; YRF = (TB - FULL0).days / 365.25
E50 = np.asarray(P.E50, float)
BENCH = dict(since2019=float((E50[-1] / E50[i19]) ** (1 / YR19) - 1), full=float((E50[-1] / E50[it0]) ** (1 / YRF) - 1),
             maxdd_full=float((E50[it0:] / np.maximum.accumulate(E50[it0:]) - 1).min()), maxdd_2019=float((E50[i19:] / np.maximum.accumulate(E50[i19:]) - 1).min()))
KW = dict(sat_w=0.7, sizing='W10', lock=False, cost=COSTS['strict'])
print('載入', f'{time.time()-t0c:.0f}s', rss(), 'MB', BENCH, flush=True)


def met(nav):
    v = nav.to_numpy()                                     # index＝P.IDX[t0:]
    a = i19 - it0
    return (float((v[-1] / v[a]) ** (1 / YR19) - 1), float((v[-1] / v[0]) ** (1 / YRF) - 1),
            float((v / np.maximum.accumulate(v) - 1).min()), float((v[a:] / np.maximum.accumulate(v[a:]) - 1).min()))


SB = {s: L.base_signal(P, U, RT, s) for s in L.SHIFTS}
MONTHS = sorted({x['month'] for s in L.SHIFTS for x in SB[s]['info']})
K_OF = {s: {x['month']: int(x['k']) for x in SB[s]['info']} for s in L.SHIFTS}
DYm, DYl, DY50, DIVN = L.load_dividends(P)
RES = {m: L.resume_rows(P, m) for m in (3, 6, 12)}

# ---------------------------------------------------------------- 現行＋驗收
ref = json.loads((L.AG / 'deadline_engine_fix' / 'out' / 'results.json').read_text())
BASE = np.zeros((20, 4)); BASE_N = np.zeros(20, int)
BNAV = {}; BEV = {}
acc = dict(A2=True, A3=True, A4=True, A5=True)
for i, (cap, s) in enumerate(PATHS):
    nav, lg, st = s10lib.simulate_s10(P, SB[s], U, capital=cap, **KW)
    BASE[i] = met(nav); BASE_N[i] = int((lg['why'] != 'open_end').sum())
    nv = nav.to_numpy()
    n2, _, st2 = simulate_s64(P, SB[s], U, capital=cap, **KW)
    acc['A2'] &= bool(np.array_equal(nv, n2.to_numpy()))
    n3, _, st3 = simulate_s64(P, SB[s], U, capital=cap, beh=dict(DY=DYm, DY50=DY50), **KW)
    acc['A3'] &= bool(np.array_equal(nv, n3.to_numpy()))
    n3l, _, st3l = simulate_s64(P, SB[s], U, capital=cap, beh=dict(DY=DYl, DY50=DY50), **KW)
    acc['A3'] &= bool(np.array_equal(nv, n3l.to_numpy()))
    BNAV[i] = np.concatenate([np.full(it0, np.nan), nv]); BEV[i] = dict(m=st3['div_ev'], l=st3l['div_ev'], e50=st3['div50_ev'])
    if cap == 1e6:
        n4, _, st4 = simulate_s64(P, L.skip_months(SB[s], set()), U, capital=cap, beh=dict(stop=(0.99, RES[6]), gap=10.0), **KW)
        acc['A4'] &= bool(np.array_equal(nv, n4.to_numpy()) and not st4['stop_ev'] and st4.get('blk_gap', 0) == 0)
        n5, _, st5 = simulate_s64(P, L.skip_months(SB[s], set(K_OF[s].values())), U, capital=cap, **KW)
        acc['A5'] &= bool(st5['n_buy'] == 0)
    if cap == 1e6 and s == 0:
        acc['main'] = dict(since2019=BASE[i, 0], full=BASE[i, 1], maxdd_full=BASE[i, 2], n_closed=int(BASE_N[i]))
        acc['A1_main_bitwise'] = bool(BASE[i, 0] == ref['standard']['ALL']['since2019'] == L.ACC['since2019'] and BASE_N[i] == ref['standard']['ALL']['n_closed'] == 381
                                      and stlib.cagr(nav, FULL0, TB) == ref['standard']['ALL']['full'] and stlib.maxdd(nav, FULL0, TB) == ref['standard']['ALL']['maxdd_full']
                                      and stlib.cagr(nav, D19, TB) == BASE[i, 0])
    assert abs(stlib.cagr(nav, D19, TB) - BASE[i, 0]) < 1e-12
    rp = ref['paths']['ALL'][pkey(cap, s)]
    acc.setdefault('A1_paths_bitwise', True)
    acc['A1_paths_bitwise'] &= bool(stlib.cagr(nav, D19, TB) == rp['since2019'] and stlib.maxdd(nav, FULL0, TB) == rp['maxdd_full'])
acc['ok'] = bool(acc['A1_main_bitwise'] and acc['A1_paths_bitwise'] and acc['A2'] and acc['A3'] and acc['A4'] and acc['A5'])
print('驗收', json.dumps(acc, ensure_ascii=False), f'{time.time()-t0c:.0f}s', flush=True)
assert acc['ok'], '驗收失敗'
if QUICK:
    print('quick ok', rss(), 'MB'); sys.exit(0)

pp = lambda x: float(x) * 100


def lab_ret(med, frac_neg):
    return '明顯傷害' if (med <= -2 and frac_neg >= 0.75) else ('事後看反而較好（不採用，描述）' if (med >= 2 and frac_neg <= 0.25) else '差不多')


def lab_dd(med, n_shallow, n):
    return '回撤較淺' if (med >= 2 and n_shallow >= 0.75 * n) else ('回撤較深' if (med <= -2 and n_shallow <= 0.25 * n) else '差不多')


def summ_det(M, extra=None):
    """M：20×4（since2019, full, dd_full, dd_2019）。"""
    d = (M - BASE) * 100
    o = dict(kind='det', since2019=float(np.median(M[:, 0])), full=float(np.median(M[:, 1])), maxdd_full=float(np.median(M[:, 2])), maxdd_2019=float(np.median(M[:, 3])),
             since2019_min=float(M[:, 0].min()), since2019_max=float(M[:, 0].max()), maxdd_full_worst=float(M[:, 2].min()),
             excess19_pp=pp(np.median(M[:, 0]) - BENCH['since2019']), excess_full_pp=pp(np.median(M[:, 1]) - BENCH['full']),
             d19_pp=float(np.median(d[:, 0])), d19_n_neg=int((d[:, 0] < 0).sum()), d19_min=float(d[:, 0].min()), d19_max=float(d[:, 0].max()),
             dfull_pp=float(np.median(d[:, 1])), dfull_n_neg=int((d[:, 1] < 0).sum()),
             ddd_full_pp=float(np.median(d[:, 2])), ddd_n_shallower=int((d[:, 2] > 1e-9).sum()), ddd_2019_pp=float(np.median(d[:, 3])),
             main=dict(since2019=float(M[2, 0]), full=float(M[2, 1]), maxdd_full=float(M[2, 2])))
    o['label'] = lab_ret(o['d19_pp'], o['d19_n_neg'] / 20); o['label_dd'] = lab_dd(o['ddd_full_pp'], o['ddd_n_shallower'], 20)
    if extra:
        o.update(extra)
    return o


def summ_rand(A, extra=None):
    """A：ndraw×20×4。每次亂數先取 20 條中位數，再對次數取中位數／P10／P90。"""
    D = (A - BASE[None]) * 100
    dm = np.median(A, axis=1); dd = np.median(D, axis=1)   # ndraw×4
    q = lambda x, p: float(np.percentile(x, p))
    o = dict(kind='rand', n_draw=int(A.shape[0]),
             since2019=q(dm[:, 0], 50), since2019_p10=q(dm[:, 0], 10), since2019_p90=q(dm[:, 0], 90),
             full=q(dm[:, 1], 50), full_p10=q(dm[:, 1], 10), full_p90=q(dm[:, 1], 90),
             maxdd_full=q(dm[:, 2], 50), maxdd_full_p10=q(dm[:, 2], 10), maxdd_2019=q(dm[:, 3], 50), maxdd_2019_p10=q(dm[:, 3], 10),
             maxdd_full_worst=float(A[:, :, 2].min()),
             excess19_pp=pp(q(dm[:, 0], 50) - BENCH['since2019']), excess19_p10_pp=pp(q(dm[:, 0], 10) - BENCH['since2019']),
             excess_full_pp=pp(q(dm[:, 1], 50) - BENCH['full']), excess_full_p10_pp=pp(q(dm[:, 1], 10) - BENCH['full']),
             d19_pp=q(dd[:, 0], 50), d19_p10=q(dd[:, 0], 10), d19_p90=q(dd[:, 0], 90), d19_frac_neg=float((D[:, :, 0] < 0).mean()),
             d19_frac_draw_neg=float((dd[:, 0] < 0).mean()),
             dfull_pp=q(dd[:, 1], 50), dfull_p10=q(dd[:, 1], 10), dfull_frac_neg=float((D[:, :, 1] < 0).mean()),
             ddd_full_pp=q(dd[:, 2], 50), ddd_full_p10=q(dd[:, 2], 10), ddd_frac_shallower=float((D[:, :, 2] > 1e-9).mean()), ddd_2019_pp=q(dd[:, 3], 50),
             frac_lose_0050_since2019=float((dm[:, 0] < BENCH['since2019']).mean()),
             main_path=dict(since2019=q(A[:, 2, 0], 50), since2019_p10=q(A[:, 2, 0], 10), full=q(A[:, 2, 1], 50), maxdd_full=q(A[:, 2, 2], 50)),
             draws_since2019=[round(float(x), 5) for x in dm[:, 0]])
    o['label'] = lab_ret(o['d19_pp'], o['d19_frac_neg']); o['label_dd'] = lab_dd(o['ddd_full_pp'], o['ddd_frac_shallower'] * 20, 20)
    if extra:
        o.update(extra)
    return o


def run20(make_S=None, beh=None, collect=None):
    M = np.zeros((20, 4)); ex = []
    Sc = {}
    for i, (cap, s) in enumerate(PATHS):
        if s not in Sc:
            Sc[s] = make_S(s) if make_S else SB[s]
        nav, lg, st = simulate_s64(P, Sc[s], U, capital=cap, beh=beh, **KW)
        M[i] = met(nav)
        if collect:
            ex.append(collect(lg, st, Sc[s]))
    return M, ex


SC = {}
NDRAW = 200

# ---------------------------------------------------------------- 1 漏做
def draw_skip(rng, p):
    r = rng.random(len(MONTHS))
    return {m for m, x in zip(MONTHS, r) if x < p}


for p in (0.10, 0.25, 0.50):
    A = np.zeros((NDRAW, 20, 4)); nsk = []
    for d in range(NDRAW):
        sk = draw_skip(np.random.default_rng([64, 1, int(round(p * 1000)), d]), p)
        nsk.append(len(sk))
        A[d], _ = run20(lambda s: L.skip_months(SB[s], {K_OF[s][m] for m in sk if m in K_OF[s]}))
    SC[f'SKIP_{int(p*100)}'] = summ_rand(A, dict(group='1_漏做', name=f'漏做 {int(p*100)}% 的月份', n_months=len(MONTHS), skipped_months_mean=float(np.mean(nsk))))
    print('SKIP', p, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in SC[f'SKIP_{int(p*100)}'].items() if k not in ('draws_since2019', 'main_path')}, f'{time.time()-t0c:.0f}s', rss(), flush=True)

# ---------------------------------------------------------------- 2 大跌停手
def col_stop(lg, st, S):
    ev = st['stop_ev']
    days = sum(((e[2] if e[2] is not None else T - 1) - e[0]) for e in ev)
    return dict(n=len(ev), frac=days / (T - 1 - it0), n_time=sum(1 for e in ev if e[3] == 'time'), n_high=sum(1 for e in ev if e[3] == 'newhigh'),
                ev=[(str(IDXV[e[0]].date()), round(e[1], 4), str(IDXV[e[2]].date()) if e[2] is not None else None, e[3]) for e in ev])


for X in (0.15, 0.20, 0.25):
    for Mo in (3, 6, 12):
        M, ex = run20(beh=dict(stop=(X, RES[Mo])), collect=col_stop)
        SC[f'STOP_{int(X*100)}_{Mo}M'] = summ_det(M, dict(group='2_大跌停手', name=f'回落 {int(X*100)}% 停手、等 {Mo} 個月或創新高',
                                                     n_stops_median=float(np.median([e['n'] for e in ex])), stopped_frac_median=float(np.median([e['frac'] for e in ex])),
                                                     resume_time_total=int(sum(e['n_time'] for e in ex)), resume_newhigh_total=int(sum(e['n_high'] for e in ex)),
                                                     main_events=ex[2]['ev']))
        print(f'STOP {X} {Mo}', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in SC[f'STOP_{int(X*100)}_{Mo}M'].items() if k not in ('main_events', 'main')}, flush=True)

# ---------------------------------------------------------------- 3 挑股
A = np.zeros((NDRAW, 20, 4))
for d in range(NDRAW):
    A[d], _ = run20(lambda s: L.pick_half(P, SB[s], 'rand', np.random.default_rng([64, 3, d, s])))
SC['PICK_RAND'] = summ_rand(A, dict(group='3_挑股', name='名單只買一半（隨機挑）'))
print('PICK_RAND', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in SC['PICK_RAND'].items() if k not in ('draws_since2019', 'main_path')}, f'{time.time()-t0c:.0f}s', flush=True)
for mode, nm in (('lowpx', '只買股價最低的一半'), ('bigamt', '只買成交值最大（名字熟）的一半'), ('highpx', '（對照）只買股價最高的一半'), ('smallamt', '（對照）只買成交值最小的一半')):
    M, ex = run20(lambda s: L.pick_half(P, SB[s], mode), collect=lambda lg, st, S: S['pick_stats'])
    o = summ_det(M, dict(group='3_挑股', name=nm, pick_stats=ex[2]))
    o['vs_rand_since2019_pp'] = pp(o['since2019'] - SC['PICK_RAND']['since2019']); o['vs_rand_full_pp'] = pp(o['full'] - SC['PICK_RAND']['full'])
    o['pctile_in_rand_draws'] = float((np.array(SC['PICK_RAND']['draws_since2019']) < o['since2019']).mean())
    SC[f'PICK_{mode.upper()}'] = o
    print('PICK', mode, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items() if k != 'main'}, flush=True)

# ---------------------------------------------------------------- 4 追高不買
for x in (0.03, 0.05):
    M, ex = run20(beh=dict(gap=x), collect=lambda lg, st, S: (st.get('blk_gap', 0), st['n_buy']))
    SC[f'GAP_{int(x*100)}'] = summ_det(M, dict(group='4_追高不買', name=f'開盤漲超過 {int(x*100)}% 就不買', blocked_median=float(np.median([e[0] for e in ex])),
                                               n_buy_median=float(np.median([e[1] for e in ex])), blocked_main=int(ex[2][0]), n_buy_main=int(ex[2][1])))
    print('GAP', x, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in SC[f'GAP_{int(x*100)}'].items() if k != 'main'}, flush=True)

# ---------------------------------------------------------------- 6 照做程度組合
LV = {'L2': dict(name='L2 大致照做：漏做 10%＋開盤漲超過 5% 不買', p=0.10, gap=0.05, half=False, stop=None),
      'L3': dict(name='L3 打折照做：漏做 25%＋開盤漲超過 3% 不買＋隨機只買一半', p=0.25, gap=0.03, half=True, stop=None),
      'L4': dict(name='L4 隨興：漏做 50%＋隨機只買一半＋回落 20% 停手等 6 個月', p=0.50, gap=None, half=True, stop=(0.20, RES[6]))}
for li, (lid, c) in enumerate(LV.items()):
    A = np.zeros((NDRAW, 20, 4))
    beh = {k: v for k, v in (('gap', c['gap']), ('stop', c['stop'])) if v is not None}
    for d in range(NDRAW):
        rng = np.random.default_rng([64, 6, li + 2, d])
        sk = draw_skip(rng, c['p'])
        seeds = rng.integers(0, 2 ** 31, size=4)

        def mk(s):
            S = L.pick_half(P, SB[s], 'rand', np.random.default_rng(int(seeds[s]))) if c['half'] else SB[s]
            return L.skip_months(S, {K_OF[s][m] for m in sk if m in K_OF[s]})
        A[d], _ = run20(mk, beh=beh or None)
    SC[lid] = summ_rand(A, dict(group='6_照做程度', name=c['name']))
    print(lid, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in SC[lid].items() if k not in ('draws_since2019', 'main_path')}, f'{time.time()-t0c:.0f}s', rss(), flush=True)

# ---------------------------------------------------------------- 5 稅後
YEAR = IDXV.year.to_numpy(); MON = IDXV.month.to_numpy()
SETTLE = {}
for y in range(int(YEAR[it0]), int(YEAR[-1]) + 1):
    w = np.flatnonzero((YEAR == y + 1) & (MON == 5))
    SETTLE[y] = int(w[-1]) if len(w) else T - 1
REG = {'併入5%': ('int', 0.05), '併入12%': ('int', 0.12), '併入20%': ('int', 0.20), '併入30%': ('int', 0.30), '分開28%': ('sep', 0.28), '只扣補充保費': ('nhi', 0.0)}
NHI_RATE, NHI_MIN, CREDIT, CREDIT_CAP = 0.0211, 20000.0, 0.085, 80000.0


def after_tax(navabs, ev, a, f0, reg):
    """navabs：長度 T 的稅前淨值；ev：[(t, 稅前尺度的股利金額)]；a：起算列；f0：起始比例。回傳（稅後年化, 稅後最大回撤, 統計）。"""
    kind, r = reg
    f = float(f0); fa = np.full(T, np.nan); steps = []
    ysum = {}; tot = dict(div=0.0, nhi=0.0, tax=0.0, n_ev=0, n_nhi=0)

    def settle(y):
        nonlocal f
        D = ysum.pop(y)
        tx = 0.0 if kind == 'nhi' else (r * D - min(CREDIT * D, CREDIT_CAP) if kind == 'int' else r * D)
        s = SETTLE[y]
        f -= tx / navabs[s]; steps.append((s, f)); tot['tax'] += tx

    for t, amt in sorted(e for e in ev if e[0] > a):
        for y in sorted(ysum):
            if SETTLE[y] <= t and y < YEAR[t]:
                settle(y)
        act = f * amt
        ysum[int(YEAR[t])] = ysum.get(int(YEAR[t]), 0.0) + act
        tot['div'] += act; tot['n_ev'] += 1
        if act >= NHI_MIN:
            fee = NHI_RATE * act
            f -= fee / navabs[t]; steps.append((t, f)); tot['nhi'] += fee; tot['n_nhi'] += 1
    for y in sorted(ysum):
        settle(y)
    fa[a:] = f0
    for s, fv in sorted(steps, key=lambda x: x[0]):
        fa[s:] = fv
    v = (fa * navabs)[a:]
    yrs = (IDXV[-1] - IDXV[a]).days / 365.25
    return float((v[-1] / v[0]) ** (1 / yrs) - 1), float((v / np.maximum.accumulate(v) - 1).min()), tot


def ev50_hold(cap, a):
    nv = np.full(T, np.nan); nv[a:] = cap * E50[a:] / E50[a]
    return nv, [(int(t), float(nv[t - 1] * DY50[t])) for t in np.flatnonzero(DY50 > 0) if t > a]


TAX = dict(rules=dict(nhi_rate=NHI_RATE, nhi_min=NHI_MIN, credit=CREDIT, credit_cap=CREDIT_CAP, settle='次年 5 月最後交易日；未結算年度在最後一天扣'),
           div_events=DIVN, variants={})
for vid, (dk, sh50) in {'主口徑（權息全額當現金、0050 配息 100% 股利所得）': ('m', 1.0), '權息當 0（下界）': ('l', 1.0), '0050 配息 50% 屬股利所得': ('m', 0.5),
                         '權息當 0＋0050 配息 50%': ('l', 0.5)}.items():
    V = {}
    for win, a in (('full', it0), ('since2019', i19)):
        W = {}
        for rid, reg in REG.items():
            s_at = np.zeros(20); s_dd = np.zeros(20); s_pre = np.zeros(20); e_at = np.zeros(20); e_pre = np.zeros(20); e_dd = np.zeros(20)
            tots = []; etots = []
            for i, (cap, s) in enumerate(PATHS):
                ev = [(t, x) for t, j, x in BEV[i][dk]] + [(t, x * sh50) for t, x in BEV[i]['e50']]
                f0 = 1.0 if a == it0 else cap / BNAV[i][a]
                s_at[i], s_dd[i], tt = after_tax(BNAV[i], ev, a, f0, reg); tots.append(tt)
                s_pre[i] = BASE[i, 0 if a == i19 else 1]
                nv, e5 = ev50_hold(cap, a)
                e_at[i], e_dd[i], et = after_tax(nv, [(t, x * sh50) for t, x in e5], a, 1.0, reg); etots.append(et)
                e_pre[i] = BENCH['since2019' if a == i19 else 'full']
            yrs = (IDXV[-1] - IDXV[a]).days / 365.25
            W[rid] = dict(strategy_after=float(np.median(s_at)), strategy_pre=float(np.median(s_pre)), strategy_drag_pp=float(np.median((s_at - s_pre) * 100)),
                          strategy_drag_min_pp=float(((s_at - s_pre) * 100).min()), strategy_drag_max_pp=float(((s_at - s_pre) * 100).max()),
                          strategy_maxdd_after=float(np.median(s_dd)),
                          e50_after=float(np.median(e_at)), e50_pre=float(np.median(e_pre)), e50_drag_pp=float(np.median((e_at - e_pre) * 100)),
                          e50_drag_100w_pp=float((e_at[2] - e_pre[2]) * 100), e50_maxdd_after=float(np.median(e_dd)),
                          excess_after_pp=float(np.median((s_at - e_at) * 100)), excess_pre_pp=float(np.median((s_pre - e_pre) * 100)),
                          strategy_main=dict(after=float(s_at[2]), drag_pp=float((s_at[2] - s_pre[2]) * 100), div_total=tots[2]['div'], nhi_total=tots[2]['nhi'], tax_total=tots[2]['tax'],
                                             n_ev=tots[2]['n_ev'], n_nhi=tots[2]['n_nhi']),
                          e50_main=dict(after=float(e_at[2]), div_total=etots[2]['div'], nhi_total=etots[2]['nhi'], tax_total=etots[2]['tax'], n_ev=etots[2]['n_ev'], n_nhi=etots[2]['n_nhi']),
                          strategy_div_per_year_main=tots[2]['div'] / yrs, by_cap_drag_pp={pkey(c, 0): float((s_at[j] - s_pre[j]) * 100) for j, (c, sft) in enumerate(PATHS) if sft == 0})
        V[win] = W
    TAX['variants'][vid] = V
MAINV = TAX['variants']['主口徑（權息全額當現金、0050 配息 100% 股利所得）']
for win in ('since2019', 'full'):
    for rid, w in MAINV[win].items():
        print('TAX', win, rid, {k: round(v, 4) for k, v in w.items() if isinstance(v, float)}, flush=True)
# 標準路徑逐年股利（稅前尺度）
ev = pd.DataFrame(BEV[2]['m'], columns=['t', 'j', 'a']); e5 = pd.DataFrame(BEV[2]['e50'], columns=['t', 'a'])
nvs = pd.Series(BNAV[2][it0:], index=IDXV[it0:])
ye = nvs.groupby(nvs.index.year).last()
TAX['main_path_yearly'] = {int(y): dict(stock_div=float(ev.loc[YEAR[ev.t] == y, 'a'].sum()), e50_div=float(e5.loc[YEAR[e5.t] == y, 'a'].sum()), year_end_nav=float(ye[y])) for y in ye.index}
TAX['main_path_events'] = dict(n_stock=len(ev), n_stock_mixed=len(BEV[2]['m']) - len(BEV[2]['l']), n_e50=len(e5), stock_single_max=float(ev.a.max()),
                               stock_frac_ge_2w=float((ev.a >= NHI_MIN).mean()), n_closed=int(BASE_N[2]))

# ---------------------------------------------------------------- 排名
base = dict(since2019=float(np.median(BASE[:, 0])), full=float(np.median(BASE[:, 1])), maxdd_full=float(np.median(BASE[:, 2])), maxdd_2019=float(np.median(BASE[:, 3])),
            since2019_min=float(BASE[:, 0].min()), since2019_max=float(BASE[:, 0].max()), maxdd_full_worst=float(BASE[:, 2].min()),
            excess19_pp=pp(np.median(BASE[:, 0]) - BENCH['since2019']), excess_full_pp=pp(np.median(BASE[:, 1]) - BENCH['full']),
            main=dict(since2019=float(BASE[2, 0]), full=float(BASE[2, 1]), maxdd_full=float(BASE[2, 2]), n_closed=int(BASE_N[2])),
            paths={pkey(c, s): dict(since2019=float(BASE[i, 0]), full=float(BASE[i, 1]), maxdd_full=float(BASE[i, 2]), maxdd_2019=float(BASE[i, 3]), n_closed=int(BASE_N[i])) for i, (c, s) in enumerate(PATHS)})
rows = [(k, v['name'], v['d19_pp'], v['dfull_pp'], v['ddd_full_pp'], v['label']) for k, v in SC.items()]
for rid in REG:
    rows.append((f'TAX_{rid}', f'稅後（{rid}）', MAINV['since2019'][rid]['strategy_drag_pp'], MAINV['full'][rid]['strategy_drag_pp'], 0.0, '稅'))
rows.sort(key=lambda x: x[2])
head = ['SKIP_25', 'STOP_20_6M', 'PICK_RAND', 'GAP_3', 'TAX_併入12%']
rank = dict(headline=[dict(id=r[0], name=r[1], d19_pp=r[2], dfull_pp=r[3], ddd_full_pp=r[4]) for r in rows if r[0] in head],
            full=[dict(id=r[0], name=r[1], d19_pp=r[2], dfull_pp=r[3], ddd_full_pp=r[4], label=r[5]) for r in rows])
res = dict(spec_sha256=sha(HERE / 'SPEC.json'), prereg_sha256=sha(HERE / 'PREREG.json'), run_sha256=sha(Path(__file__)), lib_sha256=sha(HERE / 's64lib.py'), src_sha256=L.SRC_SHA,
           flags=dict(EXRIGHT_FIX=cslib.EXRIGHT_FIX, DEADLINE_FIX=cslib.DEADLINE_FIX, INS15_MASK=cslib.INS15_MASK),
           panel=dict(T=int(P.T), N=int(P.N), start=str(FULL0.date()), d19=str(D19.date()), end=str(TB.date()), n_signal_months=len(MONTHS)),
           acceptance=acc, bench_0050=BENCH, baseline=base, scenarios=SC, tax=TAX, ranking=rank, path_keys=[pkey(c, s) for c, s in PATHS],
           runtime_s=round(time.time() - t0c, 1), maxrss_mb=rss())
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
print('BASE', {k: v for k, v in base.items() if k != 'paths'})
for r in rows:
    print(f'{r[2]:+7.2f} {r[3]:+7.2f} dd{r[4]:+6.2f}', r[0], r[1], r[5])
print('done', res['runtime_s'], 's', res['maxrss_mb'], 'MB')
