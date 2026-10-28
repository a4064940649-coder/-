#!/usr/bin/env python3
"""步驟 4：正式重跑（PREREG 後一次跑完，不改規格）。V4 引擎（v4lib，一字不改）＋V6 面板。
欄位：V6（＝V4 規格在 V6 面板）、AB_V6（40 seeds）、V6_B10、AB_V6_B10；另跑 V4 的全部敏感度欄（供 SPA 族群替換）。
輸出 navs_v6.parquet、seed_navs_v6.parquet、plan_*.csv、results.json 的 RESULTS_RUN。"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import v6common as C
RES = C.check_prereg()
fz = json.loads((C.HERE / 'data_freeze_v6.json').read_text(encoding='utf-8'))
for k, v in fz['v6_panel'].items(): assert C.sha(v['path']) == v['sha256'], k
import v6env, v6stats as ST
L = v6env.load('v6')
K, V = L.K, L.V
T0 = time.time()
def log(*a): print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)
fz4 = L.check_freeze()
XR = L.setup_px()
YFX, Y5F = L.load_rfix(); YRD = L.load_rdel(); Y5 = L.y5_orig()
GAP = L.gap_dict(); QP = L.qproxy_fn()
DMT, DMS = L.disp_mask_fn('trade'), L.disp_mask_fn('signal')
cells = pd.read_csv(C.V4DIR / 'rfix_cells.csv', dtype={'code': str})
YNO13 = YFX.copy()
_lv = pd.read_csv(K.LONG, dtype={'code': str, 'market': str}); _lv = _lv[_lv.market == 'sii']
_has = set(zip(_lv.code, _lv.month.str[:7]))
for c, y in zip(cells.code, cells.ym):
    base = str(pd.Period(y, 'M') - 12)
    if y[:4] == '2013' or (y[:4] == '2014' and (c, base) not in _has):
        YNO13.at[pd.Period(y, 'M'), c] = np.nan
E50 = K.E50
def mk(**kw):
    YL = kw.pop('YL', YFX); Y5x = kw.pop('Y5', Y5F); dm = kw.pop('dm', DMT)
    return L.V4(YL, Y5x, dm, **kw)
SPECS = {'V4': dict(), 'V4_B05': dict(gap=GAP, pfix=0.5), 'V4_B10': dict(gap=GAP, pfix=1.0), 'V4_BQ': dict(gap=GAP, yproxy=QP),
         'V4_SIG': dict(dm=DMS), 'V4_RDEL': dict(YL=YRD, Y5=Y5), 'V4_NO2013': dict(YL=YNO13), 'V4_SPEC0': dict(fin2=False, exclude_foreign=False),
         'O3REP': dict(YL=YRD, Y5=Y5, dm=DMS, fin2=False, exclude_foreign=False)}
navs, plans, logs, nblk = {}, {}, {}, {}
t0 = t1 = sd0 = None
for k, kw in SPECS.items():
    S = mk(**kw); ta, tb, sd = S.bounds(); pl = S.plan(10, 'official')
    if t0 is None: t0, t1, sd0 = ta, tb, sd
    assert (ta, tb) == (t0, t1) and sd == sd0
    for c in ('normal', 'stress'):
        navs[f'{k}|{c}'], logs[f'{k}|{c}'] = L.sim_plan(pl, ta, tb, c, XR)
    navs[f'{k}|EW'], _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, ta, tb)
    plans[k] = pl; nblk[k] = S.nblk
log('specs done')
sd = sd0
SLIPS = [round(0.0010 + 0.0002 * i, 4) for i in range(16)]
seed_navs = {}
def ab(kw, tag, slips=()):
    rets = {c: [] for c in ('normal', 'stress')}; rs = {s: [] for s in slips}; ewr = []; nold = []
    for seed in range(40):
        S = mk(unavail=V.make_unavail('mc', seed), **kw); pl = S.plan(10, 'official')
        for c in rets:
            nav, _ = L.sim_plan(pl, t0, t1, c, XR); rets[c].append(nav.pct_change())
            if tag in ('AB_V4', 'AB_V4_B10'): seed_navs[f'{tag}|{c}|s{seed}'] = nav
        for s in slips:
            nav, _ = L.sim_plan(pl, t0, t1, ('slip', s), XR); rs[s].append(nav.pct_change())
        ew, _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, t0, t1); ewr.append(ew.pct_change())
        nold.append(np.mean([S.cross_section(d)['nold'] for d in S.sig_days]))
        if seed == 0: plans[tag + '_s0'] = pl
    return {c: ST.mean_nav(rets[c]) for c in rets}, {s: ST.mean_nav(rs[s]) for s in slips}, ST.mean_nav(ewr), float(np.mean(nold))
SLIPN = {}
for k in ('V4', 'V4_B05', 'V4_B10', 'V4_BQ'):
    nv, ns, ew, no = ab(SPECS[k], f'AB_{k}', slips=SLIPS if k in ('V4', 'V4_B10') else ())
    for c in nv: navs[f'AB_{k}|{c}'] = nv[c]
    navs[f'AB_{k}|EW'] = ew; SLIPN[k] = ns
    log('AB', k)
for k in ('V4', 'V4_B10'):
    S = mk(unavail=V.make_unavail('worst'), **SPECS[k]); pl = S.plan(10, 'official')
    for c in ('normal', 'stress'): navs[f'ABW_{k}|{c}'], _ = L.sim_plan(pl, t0, t1, c, XR)
    navs[f'ABW_{k}|EW'], _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, t0, t1)
navs['V4|fullfee'], _ = L.sim_plan(plans['V4'], t0, t1, ('fullfee', 0.001), XR)
navs['V4_B10|fullfee'], _ = L.sim_plan(plans['V4_B10'], t0, t1, ('fullfee', 0.001), XR)
slip_main = {k: {s: L.sim_plan(plans[k], t0, t1, ('slip', s), XR)[0] for s in SLIPS} for k in ('V4', 'V4_B10')}
log('slips done')

# ---------------- 可執行版（o5 移植公式：容量研究的「可執行 AB_V4 − 零衝擊 AB_V4」逐日差加到 V6 欄）----------------
CAP = C.AG / 'o_exec_capacity_tiers'
EX = pd.read_parquet(CAP / 'navs_abavg.parquet'); BEN = pd.read_parquet(CAP / 'data' / 'bench.parquet')
rzi = BEN['AB_V4|normal'].pct_change()
TIERS = [f'main|{t}|{m}|A' for t in ('30萬', '100萬', '300萬', '1000萬') for m in ('FIX', 'GROW')]
def exec_nav(base, col):
    drag = (EX[col].pct_change() - rzi).reindex(base.index).fillna(0.0)   # V6 新增的 2024-01 日期沒有拖累資料 → 0
    return (1 + base.pct_change().fillna(0) + drag).cumprod()
for b in ('AB_V4', 'AB_V4_B10'):
    for col in TIERS:
        navs[f'EXEC_{b}|{col}'] = exec_nav(navs[f'{b}|normal'], col)

# ---------------- 統計 ----------------
def ren(k): return k.replace('V4', 'V6') if not k.startswith('O3REP') else 'O3REP_V6' + k[5:]
def full(nav, ew, extra=True):
    o = ST.segs(nav, ew, E50, sd)
    if extra:
        o['endpoint'] = ST.endpoint(nav, E50, sd); o['yearly'] = ST.yearly(nav, E50)
        o['yearly_lose_to_0050'] = [y for y, v in o['yearly'].items() if v['strat'] < v['0050']]
    return o
tab = {}
for k in list(SPECS) + ['AB_V4', 'AB_V4_B05', 'AB_V4_B10', 'AB_V4_BQ', 'ABW_V4', 'ABW_V4_B10']:
    for c in ('normal', 'stress'):
        tab[ren(f'{k}|{c}')] = full(navs[f'{k}|{c}'], navs[f'{k}|EW'], extra=k in ('V4', 'V4_B10', 'AB_V4', 'AB_V4_B10', 'AB_V4_B05', 'AB_V4_BQ'))
for k in ('V4|fullfee', 'V4_B10|fullfee'): tab[ren(k)] = full(navs[k], navs[k.split('|')[0] + '|EW'], extra=False)
for k in [x for x in navs if x.startswith('EXEC_')]:
    tab[ren(k)] = full(navs[k], navs[k.split('|')[0][5:] + '|EW'], extra=True)
out = {'n_patched_cells_P': L._STATE['npatch'], 'n_exit_rules_X': len(XR), 'n_signal_days': len(sd) - 1,
       'signal_days_2023_12_to_2024_03': [str(d.date()) for d in sd if pd.Timestamp('2023-12-01') <= d <= pd.Timestamp('2024-03-31')], 'table': tab}
out['turnover'] = {ren(k): logs[k]['turnover'] for k in logs if k.endswith('normal')}
out['nblk_total'] = {ren(k): v for k, v in nblk.items()}
# 逐 seed t（AB 欄，一般與壓力）
out['seed_t'] = {ren(f'{b}|{c}'): ST.seed_t([seed_navs[f'{b}|{c}|s{s}'] for s in range(40)], E50, sd) for b in ('AB_V4', 'AB_V4_B10') for c in ('normal', 'stress')}
# 成本損益兩平
def be(fn, ewk):
    pts = [(s, ST.tstat(ST.xs(fn(s), E50, sd))) for s in SLIPS]
    def cross(th):
        for (s1, a), (s2, b) in zip(pts, pts[1:]):
            if a >= th > b: return s1 + (s2 - s1) * (a - th) / (a - b)
        return None if pts[0][1] < th else float('inf')
    return {'grid': pts, 'slip_at_t2.3': cross(2.3), 'slip_at_t2.0': cross(2.0)}
out['breakeven'] = {'V6': be(lambda s: slip_main['V4'][s], 'V4|EW'), 'V6_B10': be(lambda s: slip_main['V4_B10'][s], 'V4_B10|EW'),
                    'AB_V6': be(lambda s: SLIPN['V4'][s], 'AB_V4|EW'), 'AB_V6_B10': be(lambda s: SLIPN['V4_B10'][s], 'AB_V4_B10|EW'),
                    'fullfee_t': {'V6': tab['V6|fullfee']['ALL']['t_0050'], 'V6_B10': tab['V6_B10|fullfee']['ALL']['t_0050']}}
# V6 − V4 配對差（同一格點：V4 的訊號日＋終點；另報 V6 格點）
N4 = pd.read_parquet(C.V4DIR / 'navs_v4.parquet')
sd4 = [pd.Timestamp(x) for x in pd.read_csv(C.V4DIR / 'plan_V4.csv')['signal']] + [pd.Timestamp(ST.B0)]
pair = {}
for k6, k4 in (('AB_V4|normal', 'AB_V4|normal'), ('AB_V4_B10|normal', 'AB_V4_B10|normal'), ('V4|normal', 'V4|normal'), ('V4_B10|normal', 'V4_B10|normal'),
               ('AB_V4|stress', 'AB_V4|stress')):
    a, b = navs[k6], N4[k4].dropna()
    pair[ren(k6) + ' − V4'] = {p: ST.paired(a, b, sd4, *ST.PER[p]) for p in ('ALL', 'SEL', 'VAL', 'HOLD')}
    pair[ren(k6) + ' − V4'].update(CAGR_diff_ALL=ST.cagr_mdd(a, *ST.PER['ALL'])[0] - ST.cagr_mdd(b, *ST.PER['ALL'])[0],
                                   CAGR_diff_HOLD=ST.cagr_mdd(a, *ST.PER['HOLD'])[0] - ST.cagr_mdd(b, *ST.PER['HOLD'])[0],
                                   daily_identical_to_2023=bool(np.max(np.abs(a[a.index <= '2023-12-29'] / b[b.index <= '2023-12-29'].reindex(a[a.index <= '2023-12-29'].index) - 1)) <= 1e-12))
out['paired_V6_minus_V4'] = pair
# 名單差異（2023-12 起）與最大貢獻股
P4 = pd.read_csv(C.V4DIR / 'plan_V4.csv', dtype=str)
p4 = {s: set(x.split(',')) for s, x in zip(P4.signal, P4.picks)}
p6 = {str(p[3].date()): set(K.COLS[j] for j in p[1]) for p in plans['V4']}
out['picks_diff_V6_vs_V4'] = {d: {'v6_only': sorted(p6[d] - p4.get(d, set())), 'v4_only': sorted(p4.get(d, set()) - p6[d])}
                              for d in sorted(p6) if d >= '2023-12-01' and p6[d] != p4.get(d, set())}
out['n_months_picks_differ_2024_2025'] = sum(1 for d in out['picks_diff_V6_vs_V4'] if d <= '2025-10-31')
out['robust'] = {'top_contrib_ALL': V.top_contrib(plans['V4'], t1, ST.A0, ST.B0), 'top_contrib_HOLD': V.top_contrib(plans['V4'], t1, '2024-01-01', ST.B0)}
con, per = V.stock_contrib(plans['V4'], t1)
SL = V.SL; ind = {}
for c, v in con.items():
    i = SL['industry'].get(c) if c in SL.index else None
    if not isinstance(i, str): i = '（已下市或無產業別）'
    ind[i] = ind.get(i, 0) + v
tot = sum(con.values())
out['robust']['industry_share_top10'] = sorted(((k, round(v / tot, 3)) for k, v in ind.items()), key=lambda x: -x[1])[:10]
C.save_key('RESULTS_RUN', out)
NV = pd.DataFrame({ren(k): v for k, v in navs.items()})
NV.to_parquet(C.HERE / 'navs_v6.parquet')
pd.DataFrame({ren(k): v for k, v in seed_navs.items()}).to_parquet(C.HERE / 'seed_navs_v6.parquet')
for k in ('V4', 'V4_B10', 'AB_V4_s0', 'AB_V4_B10_s0'):
    pd.DataFrame([(str(p[3].date()), str(K.IDX[p[0]].date()), ','.join(K.COLS[j] for j in p[1])) for p in plans[k]],
                 columns=['signal', 'exec', 'picks']).to_csv(C.HERE / f'plan_{ren(k)}.csv', index=False)
log('done', {k: (round(tab[k]['ALL']['CAGR'], 3), round(tab[k]['ALL']['t_0050'], 3)) for k in ('V6|normal', 'AB_V6|normal', 'AB_V6_B10|normal')})
