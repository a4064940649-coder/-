#!/usr/bin/env python3
"""S87 驗證期（凍結後只跑一次）：重新起跑（只保留營收月份 ≥ 2021-12 的訊號；2022 前全在 0050，載具的第一次換倉在 2022 第一個交易日），
量 2021-12-30 → 2026-10-22。判定 V1～V4（S25 標準）只看凍結格；其餘 15 格與描述項依 PREREG 只作描述。
results.json 已存在就拒絕重跑。乾跑 --dry：凍結格換成現行。python3 -I -B validate.py [--dry]"""
import sys, json, hashlib, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s87lib as L
from s87lib import s10lib, s2lib, s17lib, stlib

H = Path(__file__).resolve().parent
DRY = '--dry' in sys.argv
if not DRY:
    for n in ('PREREG', 'FROZEN'):
        assert hashlib.sha256((H / f'{n}.json').read_bytes()).hexdigest() == (H / f'{n}.sha256').read_text().split()[0], f'{n}.json 與 sha256 不符'
    PRE = json.loads((H / 'PREREG.json').read_text())
    for n, v in PRE['code_sha256_at_write'].items():
        assert hashlib.sha256((H / n).read_bytes()).hexdigest() == v, f'{n} 在登記後被改過'
    assert not (H / 'results.json').exists(), 'results.json 已存在，拒絕重跑'
    FZ = json.loads((H / 'FROZEN.json').read_text())
    FZID, SELECTED, T0 = FZ['frozen_for_validation'], FZ['selected'], FZ['T0_pass']
else:
    FZID, SELECTED, T0 = L.CELLS[0]['id'], 'DRY', False
OUT = L.TMP / 'dry' if DRY else H
OUT.mkdir(parents=True, exist_ok=True)
t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = L.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗'
U, RT = L.base(P)
s17lib.e50_fix_suspension(P)
L.slim(P); gc.collect()
bench = pd.Series(P.E50, index=P.IDX)
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D0, DE = P.IDX[P.t0], P.IDX[-1]
D19, D24, TB = lastb('2019-01-01'), lastb('2024-01-01'), lastb('2022-01-01')
iTB = int(np.flatnonzero(P.IDX == TB)[0])
E = L.etf_panel(P)
med = lambda v: float(np.median(v))
SEGV = ('2022', '2024-07~2025-04')
CELL = {c['id']: c for c in L.CELLS}


def veh_valid(cid):                                         # 驗證：決定日 ≥ 2021-12-30（第一次換倉＝2022 第一個交易日）
    if DRY:
        return None, []
    dec, rec = L.decisions(P, E, CELL[cid], first_dec_t=iTB)
    return L.vehicle(P, E, dec), rec


def veh_cont(cid):                                          # 全期連續
    if DRY:
        return None
    dec, _ = L.decisions(P, E, CELL[cid])
    return L.vehicle(P, E, dec)


S0 = s10lib.make_signal_s10(P, U, RT, shift=0)
n1, l1, _ = L.sim(P, S0, U, cap=1e6, fresh=True)
ratio = float(n1.loc[TB] / 1e6)
assert bool((l1['ent_date'] >= pd.Timestamp('2022-01-01')).all())
FCAP = {cap: cap / ratio for cap in L.CAPS}
VF, REC = veh_valid(FZID)
res = dict(PREREG_sha256=(H / 'PREREG.sha256').read_text().split()[0] if not DRY else 'DRY', FROZEN_sha256=(H / 'FROZEN.sha256').read_text().split()[0] if not DRY else 'DRY',
           dry=DRY, frozen=FZID, selected=SELECTED, T0_pass=T0, fresh_capital_scale=ratio,
           meta=dict(start=str(D0.date()), valid_from=str(TB.date()), end=str(DE.date())),
           bench_0050=dict(valid=stlib.cagr(bench, TB, DE), valid_dd=stlib.maxdd(bench, TB, DE), seg=s17lib.seg_dd(bench)))
if VF is not None:
    vs = pd.Series(VF['Dc'], index=P.IDX)
    res['vehicle_frozen_valid'] = dict(cagr=stlib.cagr(vs, TB, DE), maxdd=stlib.maxdd(vs, TB, DE), y2022_2023=stlib.cagr(vs, TB, D24), since2024=stlib.cagr(vs, D24, DE),
                                       e50_y2022_2023=stlib.cagr(bench, TB, D24), e50_since2024=stlib.cagr(bench, D24, DE), n_switch=VF['n_trades'], cost_sum=VF['cost_sum'],
                                       turnover=VF['turnover'], seg=s17lib.seg_dd(vs), months=REC)

# ---- 主跑（位移 0、100 萬換算）與禁買名單
navC, lgC, _ = L.sim(P, S0, U, cap=FCAP[1e6], fresh=True)
navF, lgF, _ = L.sim(P, S0, U, V=VF, cap=FCAP[1e6], fresh=True)
blk = {}
tops = {}
for nm, lg in (('CUR', lgC), ('FZ', lgF)):
    b, x = s2lib.block_from(lg, 5)
    tops[nm] = [str(c) for c in x['code']]
    for j, iv in b.items():
        blk.setdefault(j, [])
        for z in iv:
            if z not in blk[j]:
                blk[j].append(z)
res['block_union'] = dict(top5=tops, n_stocks=len(blk), n_intervals=int(sum(len(v) for v in blk.values())))
mC = navC.loc[TB:DE].groupby(navC.loc[TB:DE].index.to_period('M')).last().pct_change().dropna()
mF = navF.loc[TB:DE].groupby(navF.loc[TB:DE].index.to_period('M')).last().pct_change().dropna()
mB = bench.loc[TB:DE].groupby(bench.loc[TB:DE].index.to_period('M')).last().pct_change().dropna()
lose = (mC - mB) < 0
res['main_fresh'] = dict(cur=dict(cagr=stlib.cagr(navC, TB, DE), maxdd=stlib.maxdd(navC, TB, DE), seg=s17lib.seg_dd(navC)),
                         frozen=dict(cagr=stlib.cagr(navF, TB, DE), maxdd=stlib.maxdd(navF, TB, DE), seg=s17lib.seg_dd(navF)),
                         monthly_diff_mean_pp=float((mF - mC).mean() * 100),
                         months_cur_loses_0050=int(lose.sum()), diff_in_those_months_mean_pp=float((mF - mC)[lose].mean() * 100),
                         diff_in_other_months_mean_pp=float((mF - mC)[~lose].mean() * 100),
                         months_0050_down=int((mB < 0).sum()), diff_when_0050_down_mean_pp=float((mF - mC)[mB < 0].mean() * 100))

# ---- 20 條路徑：凍結格的 V1～V4；其餘 15 格只作描述（年化、回撤）
ALL = [FZID] + ([c for c in CELL if c != FZID] if not DRY else [])
VV = {FZID: VF}
for c in ALL[1:]:
    VV[c] = veh_valid(c)[0]
PT = {c: {} for c in ['CUR'] + ALL}
B3 = {'CUR': {}, FZID: {}}
for s in L.SHIFTS:
    S = S0 if s == 0 else s10lib.make_signal_s10(P, U, RT, shift=s)
    for cap in L.CAPS:
        k = f'{int(cap/1e4)}萬|+{s}'
        for c in ['CUR'] + ALL:
            nav = L.sim(P, S, U, V=(None if c == 'CUR' else VV[c]), cap=FCAP[cap], fresh=True)[0]
            sd = s17lib.seg_dd(nav)
            PT[c][k] = dict(valid=stlib.cagr(nav, TB, DE), dd=stlib.maxdd(nav, TB, DE), y2223=stlib.cagr(nav, TB, D24), y24=stlib.cagr(nav, D24, DE),
                            **{f'seg_{g}': sd[g] for g in SEGV})
        for c in ('CUR', FZID):
            nav = L.sim(P, S, U, V=(None if c == 'CUR' else VF), cap=FCAP[cap], fresh=True, block=blk)[0]
            B3[c][k] = stlib.cagr(nav, TB, DE)
    print('位移', s, f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)
keys = list(PT['CUR']); assert len(keys) == 20
g = lambda c, f: np.array([PT[c][k][f] for k in keys])
summ = {'CUR': dict(valid_median=med(g('CUR', 'valid')), dd_median=med(g('CUR', 'dd')))}
for c in ALL:
    d = (g(c, 'valid') - g('CUR', 'valid')) * 100; dd = (g(c, 'dd') - g('CUR', 'dd')) * 100
    summ[c] = dict(valid_median=med(g(c, 'valid')), dd_median=med(g(c, 'dd')), paired_median_pp=med(d), npos=int((d > 0).sum()), paired_min_pp=float(d.min()),
                   paired_max_pp=float(d.max()), paired_dd_median_pp=med(dd), y2223_median_pp=med((g(c, 'y2223') - g('CUR', 'y2223')) * 100),
                   y24_median_pp=med((g(c, 'y24') - g('CUR', 'y24')) * 100),
                   calmar_diff_median=med(g(c, 'valid') / np.abs(g(c, 'dd')) - g('CUR', 'valid') / np.abs(g('CUR', 'dd'))),
                   **{f'seg_{sg}_median_pp': med((g(c, f'seg_{sg}') - g('CUR', f'seg_{sg}')) * 100) for sg in SEGV})
d3 = (np.array([B3[FZID][k] for k in keys]) - np.array([B3['CUR'][k] for k in keys])) * 100
f = summ[FZID]
V = dict(V1=dict(npos=f['npos'], ok=bool(f['npos'] >= 15)), V2=dict(median_pp=f['paired_median_pp'], ok=bool(f['paired_median_pp'] >= 2.0)),
         V3=dict(median_pp=med(d3), npos=int((d3 > 0).sum()), ok=bool(med(d3) > 0)),
         V4=dict(**{sg: f[f'seg_{sg}_median_pp'] for sg in SEGV}, ok=bool(all(f[f'seg_{sg}_median_pp'] >= -2.0 for sg in SEGV))))
overall = bool(SELECTED not in ('不改', 'DRY') and T0 and all(V[k]['ok'] for k in V))
res.update(paths=PT, block3=B3, summary=summ, verdict=dict(T0=T0, selected=SELECTED, **V, overall_pass=overall))

# ---- 描述：全期連續（凍結格 vs 現行，20 條）
if not DRY:
    VC = veh_cont(FZID)
    cont = {'CUR': {}, FZID: {}}
    for s in L.SHIFTS:
        S = S0 if s == 0 else s10lib.make_signal_s10(P, U, RT, shift=s)
        for cap in L.CAPS:
            k = f'{int(cap/1e4)}萬|+{s}'
            for c, v in (('CUR', None), (FZID, VC)):
                nav = L.sim(P, S, U, V=v, cap=cap)[0]
                cont[c][k] = dict(since2019=stlib.cagr(nav, D19, DE), full=stlib.cagr(nav, D0, DE), maxdd=stlib.maxdd(nav, D0, DE))
    gc_ = lambda c, fld: np.array([cont[c][k][fld] for k in keys])
    res['continuous'] = dict(std={c: cont[c]['100萬|+0'] for c in cont},
                             since2019_paired_median_pp=med((gc_(FZID, 'since2019') - gc_('CUR', 'since2019')) * 100),
                             full_paired_median_pp=med((gc_(FZID, 'full') - gc_('CUR', 'full')) * 100),
                             maxdd_paired_median_pp=med((gc_(FZID, 'maxdd') - gc_('CUR', 'maxdd')) * 100))
    # ---- 描述：ETF 層級「順風」之後一個月對 0050 的超額（池 SX、回看 12、含流動性門檻）
    me = L.month_ends(P); ci = {c: i for i, c in enumerate(E['codes'])}
    rows = []
    for q in range(12, len(me) - 1):
        d, d0, d1 = int(me[q]), int(me[q - 12]), int(me[q + 1])
        r50 = E['TR'][d, 0] / E['TR'][d0, 0] - 1; n50 = E['TR'][d1, 0] / E['TR'][d, 0] - 1
        for c in L.POOLS['SX']:
            m = ci[c]
            a = E['AMT'][max(d - L.LIQ_WIN + 1, 0):d + 1, m]
            if not (E['HAS'][d, m] and E['HAS'][d1, m] and np.isfinite(E['TR'][d0, m]) and np.isfinite(a).sum() >= L.LIQ_MINOBS and np.nanmean(a) >= L.LIQ_MIN):
                continue
            r = E['TR'][d, m] / E['TR'][d0, m] - 1
            rows.append(dict(dec=P.IDX[d], code=c, kind='S' if c in L.STYLE else 'X', tail=bool(r > 0 and r > r50), head=bool(r < 0 and r < r50),
                             nx=float(E['TR'][d1, m] / E['TR'][d, m] - 1 - n50)))
    X = pd.DataFrame(rows)
    desc = {}
    for nm, sub in (('train', X[X.dec <= TB]), ('valid', X[X.dec >= TB]), ('all', X)):
        o = {}
        for kind in ('S', 'X', 'SX'):
            z = sub if kind == 'SX' else sub[sub.kind == kind]
            for lab, zz in (('tail', z[z.tail]), ('not_tail', z[~z.tail])):
                mm = zz.groupby('dec')['nx'].mean()
                o[f'{kind}_{lab}'] = dict(n=int(len(zz)), n_months=int(len(mm)), mean_pp=float(mm.mean() * 100) if len(mm) else None,
                                          t=(float(mm.mean() / (mm.std(ddof=1) / np.sqrt(len(mm)))) if len(mm) > 2 and mm.std(ddof=1) > 0 else None))
        desc[nm] = o
    res['etf_tailwind_next_month_excess_vs_0050'] = desc
res['meta'].update(runtime_s=round(time.time() - t_all, 1), maxrss_mb=rss())
(OUT / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
print('凍結格', FZID, json.dumps(summ[FZID], ensure_ascii=False))
print('判定', json.dumps(res['verdict'], ensure_ascii=False))
print(res['meta'])
