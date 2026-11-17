#!/usr/bin/env python3
"""S63 主跑：python3 -I -B run.py is ; python3 -I -B run.py oos ; python3 -I -B run.py combine
is／oos 各自一個行程（樣本窗是模組變數），結果先放 scratchpad/s63/part_<which>.json；combine 套 PREREG 的層、排名、建議規則 → results.json。
先驗 SPEC／PREREG／code.sha256。不改任何既有檔案。"""
import sys, json, time, resource, gc, hashlib
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for f in ('SPEC', 'PREREG'):
    assert sha(HERE / f'{f}.json') == (HERE / f'{f}.sha256').read_text().split()[0], f'{f}.json 與登記的 sha256 不符'
for line in (HERE / 'code.sha256').read_text().strip().split('\n'):
    h, f = line.split()
    assert sha(HERE / f) == h, f'{f} 與登記的 sha256 不符'
import s63lib as X                                          # noqa: E402
MODE = sys.argv[1]
t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
RANKED = ['A', 'B12', 'C', 'D', 'E', 'F', 'G']
CFG = dict(C=dict(sg='C', sat_w=0.7), D=dict(sg='D', sat_w=0.7), E=dict(sg='E', sat_w=0.7, div=5, slots=5), F=dict(sg='F', sat_w=1.0),
           F70=dict(sg='F', sat_w=0.7), G=dict(sg='C', sat_w=0.35))
LOOKS = dict(A=0, B=0, B12=0, C=12, D=4, E=12, F=12, F70=12, G=12)


def run(which):
    P, U, RT, Rv, fix = X.build(which)
    bench = pd.Series(P.E50, index=P.IDX)
    D0 = P.IDX[P.t0]
    if which == 'is':
        TB = P.IDX[-1]; D19 = P.IDX[P.IDX < pd.Timestamp('2019-01-01')][-1]
        WIN = dict(full=(D0, TB), s2019=(D19, TB))
    else:
        TB = P.IDX[P.IDX <= pd.Timestamp('2013-01-31')][-1]
        WIN = dict(oos=(D0, TB))
    months_all = len(pd.period_range(D0, P.IDX[-1], freq='M')); years_all = (P.IDX[-1] - D0).days / 365.25
    R = {c: dict(paths={}) for c in ['A', 'B', 'B12'] + list(CFG)}
    navs = {}
    acc = {}
    RTQ = X.rt_quarterly(P, RT)

    def effort(lg, st):
        f = st['fills']
        ns = sum(1 for x in f if x[2] == 'S')
        days = len(set(x[0] for x in f))
        orders = st['n_buy'] + ns + st['rb_trim_orders'] + st['e50_orders']
        cl = int((lg['why'] != 'open_end').sum()) if len(lg) else 0
        return dict(n_buy=int(st['n_buy']), n_sell=int(ns), n_trim=int(st['rb_trim_orders']), n_e50_orders=int(st['e50_orders']), n_closed=cl,
                    orders_per_month=orders / months_all, stock_days_per_month=days / months_all, closed_per_year=cl / years_all,
                    avg_npos=float(st['avg_npos']), avg_idle_frac=float(st['avg_idle_frac']), months=months_all,
                    hold_days_mean=float(lg.loc[lg['why'] != 'open_end', 'hold_days'].mean()) if cl else None)

    def blocks(nav):
        return {w: X.stats_block(nav, bench, a, b) for w, (a, b) in WIN.items()}

    # ---- A：空訊號（與位移無關）
    SA = X.empty_signal(P)
    for cap in X.CAPS:
        nav, lg, st = X.sim(P, SA, U, cap=cap)
        R['A']['paths'][f'{int(cap/1e4)}萬'] = blocks(nav)
        if cap == 1e6:
            R['A']['std'] = blocks(nav); R['A']['effort'] = effort(lg, st); navs['A'] = nav
            late = {}
            for w, (a, b) in WIN.items():                      # 第一筆買進晚 2 天：只有窗的基底日＝回測起點時才有差
                if a == D0:
                    s = nav.loc[a:b]; f = P.E50O[P.t0 + 1] / P.E50O[P.t0 + 3]
                    yrs = (s.index[-1] - s.index[0]).days / 365.25
                    late[w] = (((s.iloc[-1] * f / s.iloc[0]) ** (1 / yrs) - 1) - X.stlib.cagr(s)) * 100
                else:
                    late[w] = 0.0
            R['A']['late2_pp'] = late
    # ---- B／B12：定期定額
    for sft in X.SHIFTS:
        o = {}; o12 = {}
        for w, (a, b) in WIN.items():
            d = X.dca(P, a, b, 10000.0, shift=sft); d3 = X.dca(P, a, b, 30000.0, shift=sft)
            lump_mult = float(bench.loc[b] / bench.loc[a])
            yrs = (b - a).days / 365.25
            o[w] = dict(irr=d['irr'], multiple=d['multiple'], invested=d['invested'], final=d['final'], n_buys=d['n_buys'], irr_3wan=d3['irr'], multiple_3wan=d3['multiple'],
                        lump_multiple_nocost=lump_mult, lump_cagr_nocost=float(lump_mult ** (1 / yrs) - 1),
                        acct_maxdd_ref=float((d['nav'].iloc[1:] / d['nav'].iloc[1:].cummax() - 1).min()))
            d12 = X.dca(P, a, b, 1e6 / 12, shift=sft, n_months=12, lump=1e6)
            o12[w] = X.stats_block(d12['nav'], bench, a, b); o12[w]['n_buys'] = d12['n_buys']
            if sft == 0:
                navs[f'B12_{w}'] = d12['nav']
        R['B']['paths'][f'+{sft}'] = o; R['B12']['paths'][f'+{sft}'] = o12
    R['B']['std'] = R['B']['paths']['+0']; R['B12']['std'] = R['B12']['paths']['+0']
    R['B']['late2_pp'] = {w: (R['B']['paths']['+2'][w]['irr'] - R['B']['paths']['+0'][w]['irr']) * 100 for w in WIN}
    R['B12']['late2_pp'] = {w: (R['B12']['paths']['+2'][w]['cagr'] - R['B12']['paths']['+0'][w]['cagr']) * 100 for w in WIN}
    R['B']['effort'] = dict(orders_per_month=1.0, stock_days_per_month=0.0, closed_per_year=0.0, note='每月 1 筆 0050（可設自動扣款）')
    R['B12']['effort'] = dict(orders_per_month=12 / months_all, stock_days_per_month=0.0, closed_per_year=0.0, note='前 12 個月每月 1 筆，之後不動')
    # ---- C～G
    for sft in X.SHIFTS:
        SG = dict(C=X.s10lib.make_signal_s10(P, U, RT, shift=sft), D=X.s10lib.make_signal_s10(P, U, RTQ, shift=sft),
                  E=X.s10lib.make_signal_s10(P, U, RT, shift=sft, nsel=5), F=X.make_signal_lazy(P, U, RT, Rv, shift=sft))
        if sft == 0:
            acc['A4_F_rows_same'] = bool([x['k'] for x in SG['F']['info']] == [x['k'] for x in SG['C']['info']])
            sig_info = {k: dict(n_rows=len(v['info']), mean_tgt=float(np.mean([x['n_tgt'] for x in v['info']])), first=v['info'][0]['sig_date'], last=v['info'][-1]['sig_date'])
                        for k, v in SG.items()}
            n2, _, _ = X.L.simulate_s47(P, SG['C'], U, ndiv=10, idle='e50', slots=10, cost=X.COSTS['strict'], capital=1e6, cap_pct=0.05, sat_w=0.7, sizing='W10', lock=False)
        for c, cf in CFG.items():
            for cap in X.CAPS:
                nav, lg, st = X.sim(P, SG[cf['sg']], U, sat_w=cf['sat_w'], div=cf.get('div', 10), slots=cf.get('slots', 10), cap=cap)
                R[c]['paths'][f'{int(cap/1e4)}萬|+{sft}'] = blocks(nav)
                if sft == 0 and cap == 1e6:
                    R[c]['std'] = blocks(nav); R[c]['effort'] = effort(lg, st); navs[c] = nav
                    if c == 'C':
                        acc['A3_s47_copy_bitwise'] = bool(np.array_equal(nav.to_numpy(), n2.to_numpy(), equal_nan=True))
                        if which == 'is':
                            a1 = dict(since2019=R[c]['std']['s2019']['cagr'], full=R[c]['std']['full']['cagr'], maxdd=R[c]['std']['full']['maxdd'], n_closed=R[c]['effort']['n_closed'])
                            acc['A1'] = a1; acc['A1_bitwise'] = bool(all(a1[k] == X.L.ACC[k] for k in X.L.ACC))
                        else:
                            acc['A2'] = dict(ex_ann_pp=R[c]['std']['oos']['ex_ann_pp'], t_nw6=R[c]['std']['oos']['t_nw6'], s55=dict(ex_ann_pp=1.27, t_nw6=0.51))
                            acc['A2_ok'] = bool(abs(acc['A2']['ex_ann_pp'] - 1.27) <= 0.01 and abs(acc['A2']['t_nw6'] - 0.51) <= 0.01)
                del nav, lg, st
        del SG; gc.collect()
        print(which, '位移', sft, f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)
    for c in CFG:
        R[c]['late2_pp'] = {w: float(np.mean([R[c]['paths'][f'{int(cap/1e4)}萬|+2'][w]['cagr'] - R[c]['paths'][f'{int(cap/1e4)}萬|+0'][w]['cagr'] for cap in X.CAPS]) * 100)
                            for w in WIN}
        R[c]['late1_pp'] = {w: float(np.mean([R[c]['paths'][f'{int(cap/1e4)}萬|+1'][w]['cagr'] - R[c]['paths'][f'{int(cap/1e4)}萬|+0'][w]['cagr'] for cap in X.CAPS]) * 100)
                            for w in WIN}
    # ---- 路徑中位數
    for c in R:
        if c == 'B':
            R[c]['path'] = {w: {k: float(np.median([p[w][k] for p in R[c]['paths'].values()])) for k in ('irr', 'multiple')} for w in WIN}
            continue
        R[c]['path'] = {w: dict(n=len(R[c]['paths']), **{k: float(np.median([p[w][k] for p in R[c]['paths'].values()])) for k in ('cagr', 'maxdd', 'calmar', 'geo_diff_pp', 'ex_ann_pp', 't_nw6')},
                                cagr_min=float(min(p[w]['cagr'] for p in R[c]['paths'].values())), cagr_max=float(max(p[w]['cagr'] for p in R[c]['paths'].values())),
                                maxdd_worst=float(min(p[w]['maxdd'] for p in R[c]['paths'].values())),
                                n_beat_0050=int(sum(p[w]['geo_diff_pp'] > 0 for p in R[c]['paths'].values())))
                        for w in WIN}
    bstat = {w: dict(cagr=X.stlib.cagr(bench.loc[a:b]), maxdd=float((bench.loc[a:b] / bench.loc[a:b].cummax() - 1).min()), a=str(a.date()), b=str(b.date())) for w, (a, b) in WIN.items()}
    for w in bstat:
        bstat[w]['calmar'] = bstat[w]['cagr'] / abs(bstat[w]['maxdd'])
    acc['ok'] = bool(acc['A3_s47_copy_bitwise'] and acc['A4_F_rows_same'] and (acc.get('A1_bitwise', True)) and acc.get('A2_ok', True))
    print('驗收', json.dumps(acc, ensure_ascii=False), flush=True)
    assert acc['ok'], '驗收失敗'
    out = dict(which=which, panel=dict(T=P.T, N=P.N, t0=str(D0.date()), last=str(P.IDX[-1].date())), bench_nocost=bstat, signals=sig_info, acceptance=acc, cells=R,
               runtime_s=round(time.time() - t0, 1), maxrss_mb=rss())
    (X.SCR / f'part_{which}.json').write_text(json.dumps(out, ensure_ascii=False))
    df = pd.DataFrame({'0050': bench.reindex(navs['C'].index), **{k: v for k, v in navs.items()}}).round(2); df.index.name = 'date'
    if which == 'oos':
        df = df.loc[:TB]
    df.to_csv(HERE / f'navs_{which}.csv.gz')
    print('done', which, out['runtime_s'], 's', out['maxrss_mb'], 'MB')


def combine():
    I = json.loads((X.SCR / 'part_is.json').read_text()); O = json.loads((X.SCR / 'part_oos.json').read_text())
    T = {}
    for c in I['cells']:
        ci, co = I['cells'][c], O['cells'][c]
        if c == 'B':
            T[c] = dict(full=ci['path']['full'], s2019=ci['path']['s2019'], oos=co['path']['oos'], std=dict(full=ci['std']['full'], s2019=ci['std']['s2019'], oos=co['std']['oos']),
                        late2_pp=dict(**ci['late2_pp'], **co['late2_pp']), effort=ci['effort'], looks_per_year=LOOKS[c])
            continue
        T[c] = dict(full=ci['path']['full'], s2019=ci['path']['s2019'], oos=co['path']['oos'],
                    std=dict(full=ci['std']['full'], s2019=ci['std']['s2019'], oos=co['std']['oos']),
                    late2_pp=dict(**ci['late2_pp'], **co['late2_pp']), late1_pp=dict(**ci.get('late1_pp', {}), **co.get('late1_pp', {})),
                    effort=ci['effort'], effort_oos=co['effort'], looks_per_year=LOOKS[c])
    G = {}
    for c in RANKED:
        g_is = bool(T[c]['full']['geo_diff_pp'] >= -0.5 and T[c]['s2019']['geo_diff_pp'] >= -0.5)
        g_oos = bool(T[c]['oos']['geo_diff_pp'] >= -0.5)
        G[c] = dict(is_pass=g_is, oos_pass=g_oos, layer=1 if (g_is and g_oos) else (2 if (g_is or g_oos) else 3))
    cal = {c: T[c]['full']['calmar'] for c in RANKED}; eff = {c: T[c]['effort']['orders_per_month'] for c in RANKED}
    rk_cal = {c: 1 + sum(cal[x] > cal[c] for x in RANKED) for c in RANKED}
    rk_eff = {c: 1 + sum(eff[x] < eff[c] for x in RANKED) for c in RANKED}
    order = sorted(RANKED, key=lambda c: (G[c]['layer'], rk_cal[c] + rk_eff[c], -cal[c], -T[c]['late2_pp']['s2019']))
    ranking = [dict(rank=i + 1, cell=c, layer=G[c]['layer'], is_pass=G[c]['is_pass'], oos_pass=G[c]['oos_pass'], calmar_full=cal[c], calmar_rank=rk_cal[c],
                    orders_per_month=eff[c], effort_rank=rk_eff[c], score=rk_cal[c] + rk_eff[c]) for i, c in enumerate(order)]
    rec = {}
    for c in RANKED:
        if c == 'C':
            continue
        c1 = G[c]['layer'] <= G['C']['layer']; c2 = eff[c] <= 0.5 * eff['C']; c3 = cal[c] >= 0.9 * cal['C']
        rec[c] = dict(layer_ok=bool(c1), effort_ok=bool(c2), calmar_ok=bool(c3), recommended_option=bool(c1 and c2 and c3),
                      cagr2019_minus_C_pp=(T[c]['s2019']['cagr'] - T['C']['s2019']['cagr']) * 100,
                      costly=bool((T[c]['s2019']['cagr'] - T['C']['s2019']['cagr']) * 100 < -5))
    res = dict(spec_sha256=sha(HERE / 'SPEC.json'), prereg_sha256=sha(HERE / 'PREREG.json'), code_sha256={f: sha(HERE / f) for f in ('s63lib.py', 'run.py')},
               bench_nocost=dict(**I['bench_nocost'], **O['bench_nocost']), signals=dict(is_=I['signals'], oos=O['signals']),
               acceptance=dict(is_=I['acceptance'], oos=O['acceptance']), table=T, gates=G, ranking=ranking, recommendation=rec,
               paths=dict(is_={c: I['cells'][c]['paths'] for c in I['cells']}, oos={c: O['cells'][c]['paths'] for c in O['cells']}),
               runtime_s=dict(is_=I['runtime_s'], oos=O['runtime_s']), maxrss_mb=dict(is_=I['maxrss_mb'], oos=O['maxrss_mb']))
    (HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
    pc = lambda v: f'{v*100:.2f}%'
    print('0050（不含成本）', {w: (pc(v['cagr']), pc(v['maxdd'])) for w, v in res['bench_nocost'].items()})
    print('格 | 2019起 年化/回撤/Calmar/幾何差 | 全期 年化/回撤/Calmar/幾何差/算術超額/NWt(標準) | 筆/年 單/月 日/月 | 晚2天(2019/全期) | 樣本外 年化/幾何差/算術超額/NWt/回撤')
    for c in T:
        t = T[c]
        if c == 'B':
            print('B', {w: (pc(t[w]['irr']), round(t[w]['multiple'], 2)) for w in ('s2019', 'full', 'oos')}, 'std', {w: (round(t['std'][w]['lump_multiple_nocost'], 2), pc(t['std'][w]['lump_cagr_nocost']), t['std'][w]['invested'], pc(t['std'][w]['irr_3wan']), pc(t['std'][w]['acct_maxdd_ref'])) for w in ('s2019', 'full', 'oos')}, 'late2', t['late2_pp'])
            continue
        e = t['effort']
        print(c, '|', pc(t['s2019']['cagr']), pc(t['s2019']['maxdd']), f"{t['s2019']['calmar']:.2f}", f"{t['s2019']['geo_diff_pp']:+.2f}", f"std {t['std']['s2019']['ex_ann_pp']:+.2f}/{t['std']['s2019']['t_nw6']:.2f}",
              '|', pc(t['full']['cagr']), pc(t['full']['maxdd']), f"{t['full']['calmar']:.2f}", f"{t['full']['geo_diff_pp']:+.2f}", f"std {t['std']['full']['ex_ann_pp']:+.2f}/{t['std']['full']['t_nw6']:.2f}",
              '|', f"{e['closed_per_year']:.1f} {e['orders_per_month']:.2f} {e['stock_days_per_month']:.2f}",
              '|', f"{t['late2_pp']['s2019']:+.2f}/{t['late2_pp']['full']:+.2f}",
              '|', pc(t['oos']['cagr']), f"{t['oos']['geo_diff_pp']:+.2f}", f"std {t['std']['oos']['ex_ann_pp']:+.2f}/{t['std']['oos']['t_nw6']:.2f}", pc(t['oos']['maxdd']), f"late2 {t['late2_pp']['oos']:+.2f}",
              f"n_beat {t['s2019']['n_beat_0050']}/{t['full']['n_beat_0050']}/{t['oos']['n_beat_0050']} of {t['full']['n']}")
    print(json.dumps(ranking, ensure_ascii=False, indent=0)); print(json.dumps(rec, ensure_ascii=False, indent=0))


if MODE == 'combine':
    combine()
else:
    run(MODE)
