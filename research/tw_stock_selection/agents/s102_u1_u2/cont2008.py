#!/usr/bin/env python3
"""S102 長期描述（不參與判定）：U1、U2 各自的凍結格放進 2008 起連續引擎（S75 v3 口徑：7% 漲跌幅＋2008-10 跌幅 3.5%、0050 銜接檔、REDUCE_FIX）。
做法照 agents/s75_continuous_2008_2026/v3_2026-11-25_limit7/run_v3.py 與 S100 cont2008.py（只讀 import s75lib；漲跌停價只在記憶體覆寫；不改任何既有檔）。
先原樣重現 v3：V_BRIDGE／V_MASK 標準路徑淨值與 nav_2008_2026_v3.csv.gz 逐日差 0、交易筆數相同（V_BRIDGE 627 筆）；
並驗收 make_signal_u(fill=None)＝K.signal、simulate_u(rebuy=None)＝K.sim（逐日逐筆）。不符就中止。
再跑：現行 vs 兩個凍結格 × 2 版本 × 20 條路徑。輸出 results_2008.json、nav_2008_frozen_vs_current.csv.gz、yearly_2008.csv。已有 results_2008.json 就拒絕重跑。
用法：python3 -I -B cont2008.py [--dry]（--dry：凍結格＝現行，只驗基準與流程，不寫檔）"""
import sys, json, time, resource, gc, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
AG = HERE.parent
S75 = AG / 's75_continuous_2008_2026'; V3 = S75 / 'v3_2026-11-25_limit7'; V2 = S75 / 'v2_2026-11-21'
DRY = '--dry' in sys.argv
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(S75))
import numpy as np, pandas as pd
if not DRY:
    assert not (HERE / 'results_2008.json').exists(), 'results_2008.json 已存在'
    PSHA = sha(HERE / 'PREREG.json'); assert PSHA == (HERE / 'PREREG.sha256').read_text().split()[0]
    for ln in (HERE / 'code.sha256').read_text().splitlines():
        h, f = ln.split()
        assert sha(HERE / f) == h, f'{f} 與登記時不同'
    assert sha(HERE / 'FROZEN.json') == (HERE / 'FROZEN.sha256').read_text().split()[0]
    FZJ = json.loads((HERE / 'FROZEN.json').read_text())
    FZC = {tp: FZJ['topics'][tp]['frozen_for_validation'] for tp in ('U1', 'U2')}
else:
    FZC = dict(U1='CUR', U2='CUR')
import s75lib as K
from s75lib import R, cslib, s10lib
import s102core as CORE
import s102sim
from s10lib import COSTS

t0 = time.time(); rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
assert cslib.REDUCE_FIX is False and cslib.E50_SOURCE is None
old = pd.read_csv(cslib.E50_CSV, parse_dates=['date']).set_index('date')['adj'].sort_index()
new = pd.read_csv(cslib.E50_OFFICIAL_CSV, parse_dates=['date']).set_index('date')['adj'].sort_index()
seam = new.index[0]; assert str(seam.date()) == '2015-01-05' and seam in old.index
sp = pd.concat([old.loc[:seam].iloc[:-1] * float(new.loc[seam] / old.loc[seam]), new])
ex = pd.read_csv(V2 / 'e50_spliced.csv', parse_dates=['date']).set_index('date')['adj']
assert sp.index.equals(ex.index) and float((sp - ex).abs().max()) < 1e-9, '0050 銜接檔與 v2 不同'
cslib.E50_SOURCE = V2 / 'e50_spliced.csv'; cslib.REDUCE_FIX = True
P, U = K.load_cont()
assert P.reduce_fix and P.exright_fix
D0, DE = P.IDX[P.t0], P.IDX[-1]
assert (str(D0.date()), str(DE.date())) == ('2008-01-02', '2026-10-22')
D19 = K.last_before(P.IDX, '2019-01-01'); DH1 = K.last_before(P.IDX, '2015-07-01')
benchE = pd.Series(P.E50, index=P.IDX)
bench = benchE.loc[D0:]; bench = bench / bench.iloc[0] * 1e6
P.RET5 = CORE.ret5(P)


def limits(ref, up_pct, dn_pct):                              # 逐行同 run_v3.py
    u = ref * (1 + up_pct); t = cslib.tick(u); u = np.floor(u / t + 1e-9) * t
    d = ref * (1 - dn_pct); t = cslib.tick(d); d = np.ceil(d / t - 1e-9) * t
    return u, d


iv = P.IDX.values
cut = int(np.searchsorted(iv, np.datetime64('2015-06-01'), side='left')); assert str(P.IDX[cut].date()) == '2015-06-01'
h0 = int(np.searchsorted(iv, np.datetime64('2008-10-13'), side='left')); h1 = int(np.searchsorted(iv, np.datetime64('2008-10-24'), side='right'))
u, d = limits(P.REF, 0.10, 0.10)
assert np.array_equal(u, P.LIMUP, equal_nan=True) and np.array_equal(d, P.LIMDN, equal_nan=True)
u7, d7 = limits(P.REF[:cut], 0.07, 0.07)
LU = P.LIMUP.copy(); LU[:cut] = u7; LD = P.LIMDN.copy(); LD[:cut] = d7
_, d35 = limits(P.REF[h0:h1], 0.07, 0.035); LD[h0:h1] = d35
P.LIMUP, P.LIMDN = LU, LD
del u, d, u7, d7, d35; gc.collect()

ref_nav = pd.read_csv(V3 / 'nav_2008_2026_v3.csv.gz', index_col=0, parse_dates=True)
ref_res = json.loads((V3 / 'results_v3.json').read_text())
STD = '100萬|+0'
WINS = dict(full=(D0, DE), y2008_2015H1=(D0, DH1), y2015H2_2018=(DH1, D19), since2019=(D19, DE))
KW = lambda cap: dict(sat_w=0.7, sizing='W10', lock=False, cost=COSTS['strict'], capital=cap, cap_pct=0.05)   # 同 S100 cont2008（＝s47lib.sim 傳給 simulate_s10 的參數）
NM = ['CUR'] + [f'FZ_{tp}' for tp in FZC]
CELL = dict(CUR='CUR', **{f'FZ_{tp}': FZC[tp] for tp in FZC})
YEARS = (2011, 2012, 2013, 2019)


def run(sg, cell, cap):
    if cell == 'CUR' or CORE.CELLS[cell]['topic'] == 'U1':
        return K.sim(P, sg, U, cap)
    rb = dict(CORE.CELLS[cell]['rebuy']); rb['RET5'] = P.RET5
    return s102sim.simulate_u(P, sg, U, rebuy=rb, **KW(cap))


def extra_mask(lg, sg, st, cell):
    if cell == 'CUR':
        return np.zeros(len(lg), bool)
    if CORE.CELLS[cell]['topic'] == 'U1':
        fk = sg['FILL']; return np.array([int(j) in set(fk.get(int(k), [])) for j, k in zip(lg['j'], lg['sig_t'])], bool)
    return lg['rb'].to_numpy(bool)


def metrics(nav, lg, st):
    o = {k: K.cagr(nav, a, b) for k, (a, b) in WINS.items()}
    o.update(maxdd=R.maxdd(nav), maxdd_2008_2015H1=R.maxdd(nav.loc[D0:DH1]), n_all=int(len(lg)), stockfrac_full=float(np.nanmean(st['stockfrac'].to_numpy())),
             stockfrac_2008_2015H1=float(np.nanmean(st['stockfrac'].loc[D0:DH1].to_numpy())), roll252_sr_neg=CORE.roll_sr_neg(nav), top5_share=CORE.top5_share(lg)['share_of_trade_pnl'])
    return o


res = dict(frozen=FZC, engine='S75 v3（7% 漲跌幅＋2008-10 跌幅 3.5%）', versions={}, e0050={k: K.cagr(bench, a, b) for k, (a, b) in WINS.items()},
           e0050_roll252_sr_neg=CORE.roll_sr_neg(bench))
NAVOUT = {}; YR = []
for v, rule in K.VERSIONS.items():
    F, RT = K.surge(P, rule)
    PATH = {nm: {} for nm in NM}; std = {}
    for s in K.SHIFTS:
        sg0 = K.signal(P, U, RT, s)
        SG = {nm: CORE.make_signal_u(P, U, RT, shift=s, fill=(CORE.CELLS[CELL[nm]].get('fill') if CELL[nm] != 'CUR' else None)) for nm in NM}
        if s == 0:                                              # 驗收：複製訊號＝原訊號；複製引擎（不開買回）＝原引擎
            a2 = bool(np.array_equal(SG['CUR']['sig'], sg0['sig']) and np.array_equal(SG['CUR']['rank'], sg0['rank'], equal_nan=True) and np.array_equal(SG['CUR']['X'], sg0['X']))
            n1, l1, _ = K.sim(P, sg0, U, 1e6); n2, l2, _ = s102sim.simulate_u(P, SG['CUR'], U, **KW(1e6))
            cols = ['j', 'sig_t', 'ent_t', 'exit_t', 'cost', 'proceeds', 'why', 'shares']
            a1 = bool(np.array_equal(n1.to_numpy(), n2.to_numpy(), equal_nan=True) and l1[cols].equals(l2[cols]))
            del n1, l1, n2, l2
        for nm in NM:
            for cap in K.CAPS:
                nav, lg, st = run(SG[nm], CELL[nm], cap)
                key = f'{int(cap / 1e4)}萬|+{s}'
                PATH[nm][key] = metrics(nav, lg, st)
                if key == STD:
                    std[nm] = (nav, lg, st, extra_mask(lg, SG[nm], st, CELL[nm]))
                del nav, lg, st
        del sg0, SG; gc.collect()
    navC, lgC, stC, _ = std['CUR']
    base = dict(nav_max_abs_diff=float((navC.round(2) - ref_nav[v]).abs().max()), n_all=int(len(lgC)), n_all_ref=int(ref_res['versions'][v]['n_all']) if 'versions' in ref_res else None,
                cagr=K.cagr(navC, D0, DE), cagr_2008_2015H1=K.cagr(navC, D0, DH1), A1_sim_copy=a1, A2_signal_copy=a2)
    base['pass'] = bool(base['nav_max_abs_diff'] == 0 and (base['n_all_ref'] is None or base['n_all'] == base['n_all_ref']) and a1 and a2)
    print(v, '基準重現', base, f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)
    assert base['pass'], 'v3 基準重現失敗'
    keys = list(PATH['CUR'])
    b = bench.reindex(navC.index)
    yC = K.yearly_table(navC, b)
    R_ = dict(baseline=base, std={nm: PATH[nm][STD] for nm in NM}, topics={})
    for tp in FZC:
        nm = f'FZ_{tp}'
        navF, lgF, stF, emF = std[nm]

        def pair(f):
            a = np.array([PATH[nm][k][f] for k in keys]); c = np.array([PATH['CUR'][k][f] for k in keys]); dd = (a - c) * 100
            return dict(median=float(np.median(dd)), npos=int((dd > 0).sum()), min=float(dd.min()), max=float(dd.max()), fz_median=float(np.median(a)), cur_median=float(np.median(c)))
        yF = K.yearly_table(navF, b)
        for a_, f_ in zip(yC, yF):
            YR.append(dict(version=v, topic=tp, cell=FZC[tp], year=a_['year'], current=a_['strat'], frozen=f_['strat'], e0050=a_['e0050'], frozen_minus_current_pp=(f_['strat'] - a_['strat']) * 100,
                           current_minus_0050_pp=a_['diff_pp'], frozen_minus_0050_pp=f_['diff_pp']))
        yd = {r['year']: r for r in YR if r['version'] == v and r['topic'] == tp}
        x = lgF[emF]; pre = x[x['ent_date'] <= DH1]
        tstat = lambda y: dict(n=int(len(y)), mean_ret=float(y['ret'].mean()) if len(y) else None, median_ret=float(y['ret'].median()) if len(y) else None,
                               win=float((y['ret'] > 0).mean()) if len(y) else None, pnl_sum=float(y['pnl'].sum()))
        R_['topics'][tp] = dict(cell=FZC[tp], pairs={f: pair(f) for f in ('full', 'y2008_2015H1', 'y2015H2_2018', 'since2019', 'maxdd', 'maxdd_2008_2015H1', 'stockfrac_full', 'roll252_sr_neg', 'top5_share')},
                                std=dict(CUR=PATH['CUR'][STD], FZ=PATH[nm][STD]), top5pct=dict(CUR=CORE.top5_share(lgC), FZ=CORE.top5_share(lgF)),
                                years={y: dict(current_minus_0050_pp=yd[y]['current_minus_0050_pp'], frozen_minus_0050_pp=yd[y]['frozen_minus_0050_pp'], frozen_minus_current_pp=yd[y]['frozen_minus_current_pp']) for y in YEARS},
                                years_beat_0050=dict(CUR=int(sum(r['current'] > r['e0050'] for r in yd.values())), FZ=int(sum(r['frozen'] > r['e0050'] for r in yd.values())), n=len(yd)),
                                extra_trades=dict(all=tstat(x), y2008_2015H1=tstat(pre), after=tstat(x[x['ent_date'] > DH1])))
        NAVOUT[f'{v}|{tp}凍結格'] = navF
    R_['paths'] = PATH
    res['versions'][v] = R_
    NAVOUT[v + '|現行'] = navC
    print(v, 'DRY' if DRY else json.dumps({tp: {k: R_['topics'][tp]['pairs'][k] for k in ('y2008_2015H1', 'full', 'since2019', 'maxdd', 'roll252_sr_neg', 'top5_share')} for tp in FZC}, ensure_ascii=False),
          f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)
    del std, F, RT, PATH; gc.collect()
res['meta'] = dict(runtime_s=round(time.time() - t0, 1), maxrss_mb=rss(), dry=DRY)
if DRY:
    assert all(abs(res['versions'][v]['topics'][tp]['pairs'][f]['max']) == 0 and res['versions'][v]['topics'][tp]['pairs'][f]['min'] == 0
               for v in res['versions'] for tp in FZC for f in ('full', 'y2008_2015H1'))
    print('DRY OK', {v: res['versions'][v]['baseline'] for v in res['versions']}, res['meta'])
else:
    (HERE / 'results_2008.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    df = pd.DataFrame(NAVOUT).round(2); df['0050'] = bench.reindex(df.index).round(2); df.index.name = 'date'; df.to_csv(HERE / 'nav_2008_frozen_vs_current.csv.gz')
    pd.DataFrame(YR).round(6).to_csv(HERE / 'yearly_2008.csv', index=False)
    vb = res['versions']['V_BRIDGE']
    for tp in FZC:
        print(tp, json.dumps({k: vb['topics'][tp][k] for k in ('top5pct', 'years', 'years_beat_0050', 'extra_trades')}, ensure_ascii=False))
    print(res['meta'])
