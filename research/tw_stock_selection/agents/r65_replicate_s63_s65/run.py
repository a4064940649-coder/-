#!/usr/bin/env python3
"""R65：獨立重現 S63（D、E、G）與 S65（A、D、E）的關鍵數字。python3 -I -B run.py → results.json、compare.csv。"""
import sys, json, csv, time, resource, gc, shutil
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import r65lib as L
from r65lib import s10lib, stlib, cslib, COSTS, simulate_s47, simulate_r65

t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P, U, RT = L.load()
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq'):
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
print('載入', P.T, P.N, P.IDX[-1].date(), f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)
IDX = pd.DatetimeIndex(P.IDX)
D0, DE = IDX[P.t0], IDX[-1]
D19 = IDX[IDX < pd.Timestamp('2019-01-01')][-1]
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6); SHIFTS = (0, 1, 2, 3)
STRICT = COSTS['strict']
res = dict(meta=dict(T=int(P.T), N=int(P.N), start=str(D0.date()), end=str(DE.date()), d19=str(D19.date()),
                     flags=dict(EXRIGHT_FIX=cslib.EXRIGHT_FIX, DEADLINE_FIX=cslib.DEADLINE_FIX, INS15_MASK=cslib.INS15_MASK)))

# ---------------- 第 0 步：現行逐位
S0 = s10lib.make_signal_s10(P, U, RT, shift=0)
nav, lg, st = s10lib.simulate_s10(P, S0, U, sat_w=0.7, cost=STRICT, capital=1e6, **L.KW)
c19 = stlib.cagr(nav, D19, DE); ncl = int((lg['why'] != 'open_end').sum())
res['step0'] = dict(since2019=c19, full=stlib.cagr(nav, D0, DE), maxdd=stlib.maxdd(nav, D0, DE), n_closed=ncl,
                    bitwise=bool(c19 == 0.4875617247721724 and ncl == 381))
print('step0', res['step0'], flush=True)
nav2, lg2, _ = simulate_r65(P, S0, U, sat_w=0.7, cost=STRICT, capital=1e6, **L.KW)
res['step0']['copy_default_bitwise'] = bool(np.array_equal(nav.to_numpy(), nav2.to_numpy()))
nav3, _, _ = simulate_s47(P, S0, U, sat_w=0.7, cost=STRICT, capital=1e6, ndiv=10, slots=10, **L.KW)
res['step0']['s47_div10_bitwise'] = bool(np.array_equal(nav.to_numpy(), nav3.to_numpy()))
print(res['step0'], flush=True)
bench = pd.Series(P.E50[P.t0:], index=IDX[P.t0:])
res['meta']['e50_since2019'] = stlib.cagr(bench, D19, DE); res['meta']['e50_full'] = stlib.cagr(bench, D0, DE)

# ---------------- S63：C、D、E、G、A（20 條路徑）
QM = {x['month'] for x in S0['info'] if pd.Timestamp(x['sig_date']).month in (1, 4, 7, 10)}     # 位移 0 訊號日落在 1/4/7/10 月的營收月份
res['s63_D_rows'] = dict(n=len(QM), first=min(x['sig_date'] for x in S0['info'] if x['month'] in QM))
cells = {c: {} for c in 'ACDEG'}
def mt(nav):
    return dict(c19=stlib.cagr(nav, D19, DE), full=stlib.cagr(nav, D0, DE), dd=stlib.maxdd(nav, D0, DE))
for s in SHIFTS:
    Sg = s10lib.make_signal_s10(P, U, RT, shift=s)
    Sd = L.quarterly_S(P, Sg, QM)
    Se = s10lib.make_signal_s10(P, U, RT, shift=s, nsel=5)
    Sa = L.empty_S(P, Sg)
    for cap in CAPS:
        key = f'{int(cap/1e4)}萬|+{s}'
        kw = dict(cost=STRICT, capital=cap, **L.KW)
        cells['C'][key] = mt(s10lib.simulate_s10(P, Sg, U, sat_w=0.7, **kw)[0])
        cells['D'][key] = mt(s10lib.simulate_s10(P, Sd, U, sat_w=0.7, **kw)[0])
        cells['E'][key] = mt(simulate_s47(P, Se, U, sat_w=0.7, ndiv=5, slots=5, **kw)[0])
        cells['G'][key] = mt(s10lib.simulate_s10(P, Sg, U, sat_w=0.35, **kw)[0])
        if s == 0:
            cells['A'][key] = mt(s10lib.simulate_s10(P, Sa, U, sat_w=0.7, **kw)[0])
    del Sg, Sd, Se, Sa; gc.collect()
res['s63'] = {}
for c, d in cells.items():
    res['s63'][c] = dict(n_paths=len(d), median={m: float(np.median([v[m] for v in d.values()])) for m in ('c19', 'full', 'dd')},
                         std=d['100萬|+0'], paths=d)
    print('S63', c, {k: round(v * 100, 2) for k, v in res['s63'][c]['median'].items()}, 'std', {k: round(v * 100, 2) for k, v in d['100萬|+0'].items()}, flush=True)

# ---------------- S65：每個換股月新起跑（位移 0、零股全期）
info = S0['info']
ks = [x['k'] for x in info]
starts = [i for i, x in enumerate(info) if x['k'] + 1 < P.T and pd.Timestamp('2016-02-01') <= IDX[x['k'] + 1] <= pd.Timestamp('2025-10-31')]
Sa = L.empty_S(P, S0)
E50 = pd.Series(P.E50, index=IDX)
DDN = (E50 / E50.rolling(252, min_periods=1).max() - 1).to_numpy()       # 含當天的近 252 個交易日最高
def horizon(k, months):
    lim = IDX[k] + pd.DateOffset(months=months)
    if lim > DE:
        return None
    return int(np.searchsorted(IDX.values, np.datetime64(lim), side='right') - 1)
def run_from(S, k, cap=1e6, inj=None, inj_base=True, odd_all=True):
    nv = simulate_r65(P, S, U, sat_w=0.7, cost=STRICT, capital=cap, start_t=k, odd_all=odd_all, inj=inj, inj_base=inj_base, **L.KW)[0]
    w = np.full(P.T, np.nan); w[k:] = nv.to_numpy()
    return w
def mdd(w):
    return float((w / np.maximum.accumulate(w) - 1).min())
cacheA = {}; cacheL = {}
def WA(k, odd_all=True):
    if (k, odd_all) not in cacheA:
        cacheA[(k, odd_all)] = run_from(S0, k, odd_all=odd_all)
    return cacheA[(k, odd_all)]
rows = []
for i in starts:
    k = ks[i]; h12 = horizon(k, 12); h36 = horizon(k, 36)
    W = {}
    W['A'] = WA(k)
    W['L0'] = run_from(Sa, k)
    for name, n, base in (('D', 6, True), ('D_nobase', 6, False), ('C', 3, True)):
        part = 1e6 / n
        inj = {ks[i + q] + 1: part for q in range(1, n)}
        w = run_from(S0, k, cap=part, inj=inj, inj_base=base)
        un = np.zeros(P.T)
        for q in range(1, n):
            un[:ks[i + q] + 1] += part                         # 第 q 份在 ks[i+q]+1 開盤投入；之前算未投入現金
        W[name] = w + un
        assert abs(part * n - 1e6) < 1e-6
    # E：每月訊號列看
    wait = None
    for q in range(0, 13):
        if i + q >= len(ks):
            break
        if DDN[ks[i + q]] <= -0.10 or q == 12:
            wait = q; break
    if wait is None:                                           # 清單用完仍未觸發（不應發生）
        w = np.full(P.T, 1e6); w[:k] = np.nan; wait = -1
    else:
        kk = ks[i + wait]
        w = WA(kk).copy(); w[k:kk] = 1e6
    W['E'] = w
    r = dict(i=i, month=info[i]['month'], k=k, sig_date=str(IDX[k].date()), h12=str(IDX[h12].date()), h36=str(IDX[h36].date()) if h36 else None,
             E_wait=wait, n_tgt=info[i]['n_tgt'])
    yrs = (IDX[h36] - IDX[k]).days / 365.25 if h36 else None
    for m, w in W.items():
        r[f'r12_{m}'] = float(w[h12] / 1e6 - 1); r[f'mdd12_{m}'] = mdd(w[k:h12 + 1])
        if h36:
            r36 = w[h36] / 1e6 - 1
            r[f'ann36_{m}'] = float((1 + r36) ** (1 / yrs) - 1); r[f'ann36y3_{m}'] = float((1 + r36) ** (1 / 3) - 1)
    rows.append(r)
df = pd.DataFrame(rows)
df.to_csv(H / 'starts.csv', index=False)
n12 = len(df); d36 = df[df['h36'].notna()]; n36 = len(d36)
out = dict(n12=n12, n36=n36, first=df['sig_date'].iloc[0], last=df['sig_date'].iloc[-1], last36=d36['sig_date'].iloc[-1],
           E_immediate=float((df['E_wait'] == 0).mean()), E_full_wait=float((df['E_wait'] == 12).mean()))
for m in ('A', 'C', 'D', 'D_nobase', 'E', 'L0'):
    ex12 = (df[f'r12_{m}'] - df['r12_L0']) * 100
    ex36 = (d36[f'ann36_{m}'] - d36['ann36_L0']) * 100
    ex36y3 = (d36[f'ann36y3_{m}'] - d36['ann36y3_L0']) * 100
    out[m] = dict(r12_med=float(df[f'r12_{m}'].median()), r12_min=float(df[f'r12_{m}'].min()), r12_p10=float(np.percentile(df[f'r12_{m}'], 10)),
                  ex12_med=float(ex12.median()), ex12_p10=float(np.percentile(ex12, 10)), lose12=float((ex12 < 0).mean()),
                  ann36_med=float(d36[f'ann36_{m}'].median()), ann36_min=float(d36[f'ann36_{m}'].min()),
                  ex36_med=float(ex36.median()), ex36_p10=float(np.percentile(ex36, 10)), ex36_min=float(ex36.min()), lose36=int((ex36 < 0).sum()),
                  ex36y3_med=float(ex36y3.median()), ex36y3_p10=float(np.percentile(ex36y3, 10)))
    print('S65', m, {k: (round(v * 100, 2) if k.startswith(('r12', 'ann36')) else round(v, 2)) for k, v in out[m].items()}, flush=True)
res['s65'] = out

# ---------------- 對照表
cmp = []
def add(no, name, claim, mine, tol=0.3):
    d = mine - claim
    cmp.append(dict(編號=no, 項目=name, 原宣稱=claim, 重現值=round(mine, 4), 差=round(d, 4), 容許=tol, 符合=bool(abs(d) <= tol + 1e-9)))
s63 = res['s63']; A = out['A']
add('0', '現行 100 萬 2019 起年化（%，逐位）', 48.75617247721724, res['step0']['since2019'] * 100, 0.0)
add('0', '現行已出場筆數', 381, res['step0']['n_closed'], 0)
add('1', 'S63 D 季換股：2019 起（%）', 28.93, s63['D']['median']['c19'] * 100)
add('1', 'S63 D 季換股：全期（%）', 21.11, s63['D']['median']['full'] * 100)
add('1', 'S63 D 季換股：全期最大回撤（%）', -38.68, s63['D']['median']['dd'] * 100)
add('2', 'S63 E 只持 5 檔：2019 起（%）', 57.22, s63['E']['median']['c19'] * 100)
add('2', 'S63 E 只持 5 檔：最大回撤（%）', -44.91, s63['E']['median']['dd'] * 100)
add('3', 'S63 G 衛星 35%：2019 起（%）', 36.71, s63['G']['median']['c19'] * 100)
add('3', 'S63 G 衛星 35%：最大回撤（%）', -30.42, s63['G']['median']['dd'] * 100)
add('4', 'S65 A：12 個月報酬中位數（%）', 29.07, A['r12_med'] * 100)
add('4', 'S65 A：12 個月報酬最差（%）', -15.07, A['r12_min'] * 100)
add('4', 'S65 A：12 個月超額中位數（百分點）', 14.63, A['ex12_med'])
add('4', 'S65 A：36 個月年化中位數（%）', 35.97, A['ann36_med'] * 100)
add('4', 'S65 A：36 個月年化超額中位數（百分點）', 18.07, A['ex36_med'])
add('4', 'S65 A：36 個月年化超額 P10（百分點）', 4.73, A['ex36_p10'])
add('4', 'S65 A：36 個月輸 0050 的起始月數', 0, A['lose36'], 0)
add('4', 'S65 起始月數（12 個月）', 117, n12, 0)
add('4', 'S65 起始月數（36 個月）', 93, n36, 0)
add('5', 'S65 D 分 6 個月：36 個月年化超額中位數（百分點）', 14.96, out['D']['ex36_med'])
add('6', 'S65 E 等回落：36 個月年化超額中位數（百分點）', 14.44, out['E']['ex36_med'])
add('6', 'S65 E 等回落：36 個月年化超額 P10（百分點）', -3.35, out['E']['ex36_p10'])
add('附', 'S63 C 現行 20 條中位數 2019 起（%）', 46.40, s63['C']['median']['c19'] * 100)
add('附', 'S63 A 只買 0050 2019 起（%）', 28.91, s63['A']['median']['c19'] * 100)
add('附', 'S63 A 只買 0050 全期（%）', 21.11, s63['A']['median']['full'] * 100)
add('附', 'S63 E 只持 5 檔：全期（%）', 39.04, s63['E']['median']['full'] * 100)
add('附', 'S63 G 衛星 35%：全期（%）', 26.29, s63['G']['median']['full'] * 100)
add('附', 'S65 A：12 個月報酬 P10（%）', 1.10, A['r12_p10'] * 100)
add('附', 'S65 0050：12 個月報酬中位數（%）', 19.68, out['L0']['r12_med'] * 100)
add('附', 'S65 0050：36 個月年化中位數（%）', 15.02, out['L0']['ann36_med'] * 100)
add('附', 'S65 C 分 3 個月：36 個月年化超額中位數（百分點）', 16.01, out['C']['ex36_med'])
add('附', 'S65 D：36 個月年化超額 P10（百分點）', 3.81, out['D']['ex36_p10'])
add('附', 'S65 D：12 個月報酬中位數（%）', 21.38, out['D']['r12_med'] * 100)
add('附', 'S65 E：12 個月報酬中位數（%）', 1.04, out['E']['r12_med'] * 100)
res['compare'] = cmp
res['all_pass_required'] = bool(all(r['符合'] for r in cmp if r['編號'] != '附'))
res['peak_mb'] = rss(); res['secs'] = round(time.time() - t_all)
json.dump(res, open(H / 'results.json', 'w'), ensure_ascii=False, indent=1, default=str)
with open(H / 'compare.csv', 'w', newline='', encoding='utf-8-sig') as fh:
    w = csv.DictWriter(fh, fieldnames=list(cmp[0])); w.writeheader(); w.writerows(cmp)
for r in cmp:
    print(r)
print('all_pass_required', res['all_pass_required'], res['secs'], 's', res['peak_mb'], 'MB')
shutil.rmtree(L.TMP, ignore_errors=True)
