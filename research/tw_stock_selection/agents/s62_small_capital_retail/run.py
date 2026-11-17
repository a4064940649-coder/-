#!/usr/bin/env python3
"""S62 主程式：python3 -I -B run.py → results.json、navs.csv.gz。SPEC.json／PREREG.json 的 sha256 必須相符。單行程。
36 格摩擦 × 6 個本金（5／10／20／30／50 萬＋參考 100 萬）× 20 條路徑，另加 B 的兩個敏感度（歷史零股制度、閒錢門檻固定 2 萬）。"""
import sys, json, hashlib, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s62lib as L
from s62lib import s10lib, stlib, cslib

H = Path(__file__).resolve().parent
for n in ('SPEC', 'PREREG'):
    assert hashlib.sha256((H / f'{n}.json').read_bytes()).hexdigest() == (H / f'{n}.sha256').read_text().split()[0], f'{n}.json 與 sha256 不符'
SPEC = json.loads((H / 'SPEC.json').read_text())
for n, v in SPEC['code_sha256_at_write'].items():
    assert hashlib.sha256((H / n).read_bytes()).hexdigest() == v, f'{n} 在登記後被改過'
t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P, U, RT = L.setup()
print('載入完成', f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D0, DE, D19, D21 = P.IDX[P.t0], P.IDX[-1], lastb('2019-01-01'), lastb('2021-01-01')
T19 = int(np.searchsorted(P.IDX.values, np.datetime64(D19)))
YRS = (DE - D19).days / 365.25
bench = pd.Series(P.E50[T19:], index=P.IDX[T19:])
B19, B21, BDD = stlib.cagr(bench, D19, DE), stlib.cagr(bench, D21, DE), stlib.maxdd(bench, D19, DE)
ALLCAPS = L.CAPS + (L.REF_CAP,)
SC = {L.gid(g): g for g in L.GRID}
NAMED = {}
for k, v in L.SCEN.items():
    g = {a: v[a] for a in ('disc', 'min_odd', 'oddx', 'odd')}
    if 'sweep_fixed' in v or v['odd'] == 'hist':
        SC[k] = v; NAMED[k] = k
    else:
        NAMED[k] = L.gid(g); assert NAMED[k] in SC
ck = lambda c: f'{int(round(c / 1e4))}萬'


def metrics(nav, lg, st):
    blk = st['blk_lot'] + st['blk_min'] + st['blk_nocash']
    cost = st['fee'] + st['tax'] + st['slip'] + st['e50_cost']
    mn = float(nav.mean())
    return dict(since2019=stlib.cagr(nav, D19, DE), since2021=stlib.cagr(nav, D21, DE), maxdd=stlib.maxdd(nav, D19, DE), avg_npos=float(st['avg_npos']),
                fills_py=len(st['fills']) / YRS, e50_orders_py=st['e50_orders'] / YRS, n_buy=int(st['n_buy']), blocked=int(blk),
                blocked_frac=float(blk / (st['n_buy'] + blk)) if st['n_buy'] + blk else float('nan'),
                blk_lot=int(st['blk_lot']), blk_min=int(st['blk_min']), blk_nocash=int(st['blk_nocash']),
                cost_py=float(cost / YRS / mn), fee_py=float(st['fee'] / YRS / mn), tax_py=float(st['tax'] / YRS / mn), slip_py=float(st['slip'] / YRS / mn),
                e50_py=float(st['e50_cost'] / YRS / mn), odd_buy_frac=float(st['odd_buy_val'] / st['buy_val']) if st['buy_val'] else float('nan'),
                idle_frac=float(st['avg_idle_frac']), end_mult=float(nav.iloc[-1] / nav.iloc[0]))


def yearly(nav):
    y = nav.groupby(nav.index.year).last(); out = {}; prev = nav.iloc[0]
    for k in y.index:
        if k >= 2019:
            out[int(k)] = float(y.loc[k] / prev - 1)
        prev = y.loc[k]
    return out


res = dict(spec_sha256=(H / 'SPEC.sha256').read_text().split()[0], prereg_sha256=(H / 'PREREG.sha256').read_text().split()[0],
           flags=dict(EXRIGHT_FIX=cslib.EXRIGHT_FIX, DEADLINE_FIX=cslib.DEADLINE_FIX, INS15_MASK=cslib.INS15_MASK), src_sha256=L.SRC_SHA,
           window=dict(start=str(D19.date()), end=str(DE.date()), years=YRS, start2021=str(D21.date())),
           bench_0050=dict(since2019=B19, since2021=B21, maxdd=BDD, yearly=yearly(bench)), named=NAMED, main=NAMED[L.MAIN],
           cells={s: {ck(c): dict(paths={}) for c in ALLCAPS} for s in SC}, acceptance={})
navs = {'0050': bench / bench.iloc[0]}
cols = ['j', 'ent_t', 'exit_t', 'why', 'shares', 'cost', 'proceeds']
for s in L.SHIFTS:
    sg = s10lib.make_signal_s10(P, U, RT, shift=s)
    if s == 0:
        nav, lg, st = L.sim_official(P, sg, U)
        a0 = dict(since2019=stlib.cagr(nav, D19, DE), full=stlib.cagr(nav, D0, DE), maxdd=stlib.maxdd(nav, D0, DE), n_closed=int((lg['why'] != 'open_end').sum()))
        a0['ok'] = bool(all(a0[k] == L.ACC[k] for k in L.ACC))
        nav1, lg1, _ = L.sim(P, sg, U, 1e6, None, None, defaults=True)
        nav2, lg2, _ = L.sim(P, sg, U, 1e6, dict(disc=0.28, min_odd=20.0, oddx=0.0, odd='hist', sweep_fixed=20000.0), None)
        res['acceptance'] = dict(A0=a0, A1=bool(np.array_equal(nav.to_numpy(), nav1.to_numpy()) and lg[cols].equals(lg1[cols])),
                                 A2=bool(np.array_equal(nav.to_numpy(), nav2.to_numpy()) and lg[cols].equals(lg2[cols])))
        res['official_100萬_2015起跑'] = dict(a0, avg_npos=float(st['avg_npos']))
        assert a0['ok'] and res['acceptance']['A1'] and res['acceptance']['A2'], f"驗收失敗 {res['acceptance']}"
    for sid, sc in SC.items():
        for c in ALLCAPS:
            for m in L.MULTS:
                nav, lg, st = L.sim(P, sg, U, c * m, sc, T19)
                r = res['cells'][sid][ck(c)]
                r['paths'][f'x{m}|+{s}'] = metrics(nav, lg, st)
                if s == 0 and m == 1.0:
                    r['main'] = dict(r['paths'][f'x{m}|+{s}'], n_closed=int((lg['why'] != 'open_end').sum()), yearly=yearly(nav))
                    if sid in NAMED.values():
                        navs[f'{sid}|{ck(c)}'] = nav / nav.iloc[0]
    del sg; gc.collect()
    print('位移', s, f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)

KEYS = ('since2019', 'since2021', 'maxdd', 'avg_npos', 'fills_py', 'e50_orders_py', 'blocked_frac', 'cost_py', 'fee_py', 'tax_py', 'slip_py', 'e50_py', 'odd_buy_frac', 'idle_frac')
for sid in SC:
    for c in ALLCAPS:
        r = res['cells'][sid][ck(c)]; pp = list(r['paths'].values())
        g = lambda k: np.array([v[k] for v in pp], dtype=float)
        r['summary'] = dict({f'{k}_median': float(np.nanmedian(g(k))) for k in KEYS}, since2019_min=float(g('since2019').min()), since2019_max=float(g('since2019').max()),
                            maxdd_worst=float(g('maxdd').min()), above_0050=int((g('since2019') > B19).sum()), above_0050_2021=int((g('since2021') > B21).sum()),
                            excess=float(np.median(g('since2019')) - B19), excess2021=float(np.median(g('since2021')) - B21), n=len(pp))

# ---- 判定（事前規則）
ref = res['cells'][NAMED['R0_現行嚴成本']]['100萬']['summary']['excess']
res['reference_excess_100萬_R0'] = ref


def judge(sid, key='excess', akey='above_0050'):
    out = {}
    for c in L.CAPS:
        sm = res['cells'][sid][ck(c)]['summary']
        ex = sm[key]
        worse = bool(ex <= 0); half = bool(ex <= 0.5 * ref)
        tier = '不建議（直接買 0050）' if worse else ('可行' if (not half and sm[akey] >= 15) else '可行但打折')
        out[ck(c)] = dict(excess=ex, above=sm[akey], worse_than_0050=worse, lost_half=half, tier=tier)
    w = [c for c in L.CAPS if out[ck(c)]['worse_than_0050']]; h = [c for c in L.CAPS if out[ck(c)]['lost_half']]
    mono = lambda xs: all(out[ck(c)]['excess'] <= out[ck(d)]['excess'] + 1e-12 for c, d in zip(L.CAPS[:-1], L.CAPS[1:]))
    out['_threshold'] = dict(worse_max=ck(max(w)) if w else None, half_max=ck(max(h)) if h else None,
                             worse_contiguous=bool(w == list(L.CAPS[:len(w)])), half_contiguous=bool(h == list(L.CAPS[:len(h)])), excess_monotone=bool(mono(None)))
    return out


res['judge'] = {k: judge(NAMED[k]) for k in NAMED}
res['judge_2021window'] = {k: judge(NAMED[k], 'excess2021', 'above_0050_2021') for k in ('B_基準', 'A_樂觀', 'C_保守', 'D_只買整張')}
res['peak_mb'] = rss(); res['secs'] = round(time.time() - t_all)
json.dump(res, open(H / 'results.json', 'w'), ensure_ascii=False, indent=1)
pd.DataFrame(navs).round(6).to_csv(H / 'navs.csv.gz')
print('0050', round(B19, 4), round(B21, 4), round(BDD, 4), '參考超額', round(ref, 4))
for k in NAMED:
    print(k)
    for c in ALLCAPS:
        sm = res['cells'][NAMED[k]][ck(c)]['summary']
        print(f"  {ck(c):>5} 中位 {sm['since2019_median']*100:6.2f}% [{sm['since2019_min']*100:6.2f},{sm['since2019_max']*100:6.2f}] 超額 {sm['excess']*100:6.2f} 勝 {sm['above_0050']:2d}/20 "
              f"回撤 {sm['maxdd_median']*100:6.2f} 2021超額 {sm['excess2021']*100:6.2f} 持股 {sm['avg_npos_median']:.2f} 筆/年 {sm['fills_py_median']:.0f}+{sm['e50_orders_py_median']:.0f} "
              f"成本 {sm['cost_py_median']*100:.2f}% 買不到 {sm['blocked_frac_median']*100:.1f}% 零股占買 {sm['odd_buy_frac_median']*100:.0f}%")
    print('  ', res['judge'][k]['_threshold'])
print('done', res['secs'], 's', res['peak_mb'], 'MB')
