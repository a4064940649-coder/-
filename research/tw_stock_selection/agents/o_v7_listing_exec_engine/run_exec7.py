#!/usr/bin/env python3
"""步驟 5：可執行引擎 V7（PREREG 後一次跑完；DRY=1 時只跑 2006-04～2009-12 的少量格做流程檢查、不輸出數字）。
(A) 引擎恆等檢查：ecore7.ideal（flags 全關）逐名單重算，40 seeds 平均須與 navs_v7（官方日）及 navs_jscan_v7（各 j）逐日相同（< 1e-9）。
(B) 零衝擊拆解（引擎一般成本）：ideal ＋ 處置期賣出（disp）、＋ 併購結算延遲（mdelay，0050 計值；另報 0 報酬保守版）、兩者皆開；AB 與 B10、官方日與 j＝0～12。
(C) 可執行格：{AB, B10} × {官方日 seeds 0～39, j＝0～12 seeds 0～19} × 4 級距 × {FIX, GROW} × {一般, 壓力} × {K5T 上界, TICK 下界}；
    另加敏感度 K5（o5 原式、無跳動單位下限）：官方日、一般成本、seeds 0～19。
    全部格的 opts：漲跌停順延、零股賣成本下限 0、2% 現金、2015 前手續費 B、零股量上限（part 1.0）、處置期賣出、併購結算延遲；手續費 A（2.8 折、最低 20 元）。
    壓力：每邊再加 0.4%、全期手續費 B（6 折）、零股壓力。
輸出：navs_exec7.parquet（各格 seeds 平均日淨值）、data/exec_acc7.json（成本記帳平均）、results.json 的 RESULTS_EXEC。"""
import sys, os, json, time, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import multiprocessing as mp
import v7common as C
DRY = os.environ.get('DRY') == '1'
if not DRY: C.check_prereg()
import v7env, v7stats as ST, ecore7 as E
T0 = time.time()
def log(*a): print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)
L = v7env.load('v7'); K = L.K
XR = L.setup_px()
D = E.Data(L, XR)
SPEC = json.loads((C.DATA / 'model_spec7.json').read_text(encoding='utf-8'))
MODELS = {k: E.Model(SPEC, k) for k in ('K5T', 'TICK', 'K5')}
END = pd.Timestamp('2026-10-12')
def to_plan(g): return [(int(e), [D.CI[c] for c in pk.split(',')], sg) for e, pk, sg in zip(g.exec_idx, g.picks, g.signal)]
PDIR = Path(os.environ['DRY_DIR']) if DRY else C.DATA
PA = pd.read_csv(PDIR / 'plans_ab7.csv', dtype={'picks': str})
PJ = pd.read_csv(PDIR / 'plans_j7.csv', dtype={'picks': str})
JEND = json.loads((PDIR / 'jscan_end_days.json').read_text(encoding='utf-8'))
PLANS = {}; BOUNDS = {}; GRID = {}
for (st, s), g in PA.groupby(['strat', 'seed']):
    key = ('AB' if st == 'AB_V7' else 'B10', 'OFF', int(s)); PLANS[key] = to_plan(g)
    BOUNDS[key[:2]] = (int(K.IDX.get_loc(pd.Timestamp(g.signal.iloc[0]))), int(K.IDX.get_loc(END)))
    GRID[key[:2]] = [pd.Timestamp(x) for x in g.signal] + [END]
for (st, j, s), g in PJ.groupby(['strat', 'j', 'seed']):
    key = ('AB' if st == 'AB_V7' else 'B10', f'J{j}', int(s)); PLANS[key] = to_plan(g)
    de = pd.Timestamp(JEND[f'{st}|J{j}'])
    BOUNDS[key[:2]] = (int(K.IDX.get_loc(pd.Timestamp(g.signal.iloc[0]))), int(K.IDX.get_loc(de)))
    GRID[key[:2]] = [pd.Timestamp(x) for x in g.signal] + [de]
if DRY:
    cut = int(K.IDX.searchsorted(pd.Timestamp('2009-12-31')))
    BOUNDS = {k: (a, min(b, cut)) for k, (a, b) in BOUNDS.items()}
NORMAL = dict(lim=True, odd0=True, cash=0.02, feemix=True, zpart=1.0, disp=True, mdelay=True)
STRESS = dict(lim=True, odd0=True, cash=0.02, feemix=False, zpart=1.0, disp=True, mdelay=True, extra=0.004, odd_stress=True)

def job_ideal(a):
    key, mode = a
    t0, t1 = BOUNDS[key[:2]]
    b, s = L.cost_rates('normal')
    fl = {'base': {}, 'disp': dict(disp=True), 'mdelay': dict(mdelay=True), 'both': dict(disp=True, mdelay=True)}[mode]
    nav, info = E.ideal(D, PLANS[key], t0, t1, b, s, **fl)
    return key, mode, nav.pct_change().to_numpy(dtype=np.float64), info

def job_exec(a):
    key, model, cost, tier, ver = a
    t0, t1 = BOUNDS[key[:2]]
    opts = NORMAL if cost == 'normal' else STRESS
    nav, acc = E.Sim(D, MODELS[model], opts).run(PLANS[key], t0, t1, E.TIERS[tier], ver, 'A' if cost == 'normal' else 'B')
    keep = ('tax', 'fee', 'impact', 'odd_spread', 'disp_cost', 'pct_tax', 'pct_fee', 'pct_impact', 'pct_odd_spread', 'pct_disp_cost',
            'n_orders', 'n_cap_hit', 'n_zcap_hit', 'n_lim_block', 'deferred_days', 'n_disp_sells', 'n_merger_delay')
    return a, nav.pct_change().to_numpy(dtype=np.float64), {k: float(acc.get(k, 0.0)) for k in keep}

def run_pool(fn, jobs, tag):
    out = []
    with mp.get_context('fork').Pool(4) as P:
        for i, r in enumerate(P.imap_unordered(fn, jobs, chunksize=4)):
            out.append(r)
            if i % 1000 == 0: log(tag, i, '/', len(jobs))
    return out

E50 = K.E50
def stats(nav, grp):
    sd = GRID[grp]; t0, t1 = BOUNDS[grp]
    a, b = K.IDX[t0], K.IDX[t1]
    cg, md = ST.cagr_mdd(nav, a, b); c50 = ST.cagr_mdd(E50.reindex(nav.index).ffill(), a, b)[0]
    x = ST.xs(nav, E50, sd, a, b)
    o = {'CAGR': cg, 'maxDD': md, 'E50_CAGR': c50, 'CAGR_minus_0050': cg - c50, 't_0050': ST.tstat(x), 'mean_pp': float(x.mean() * 100)}
    for lab, (pa, pb) in (('Y19on', ('2019-01-01', b)), ('SEL', ('2006-04-01', '2018-12-31')), ('FROM2008', ('2008-01-01', b)), ('HOLD', ('2024-01-01', b))):
        xx = ST.xs(nav, E50, sd, pa, pb); o[f't_{lab}'] = ST.tstat(xx)
        o[f'CAGR_{lab}'] = ST.cagr_mdd(nav, pa, pb)[0]; o[f'E50_CAGR_{lab}'] = ST.cagr_mdd(E50.reindex(nav.index).ffill(), pa, pb)[0]
    xh = ST.xs(nav, E50, sd, ST.HOLD_CAL_END, b, attr='end'); o['t_HOLD_cal'] = ST.tstat(xh); o['HOLD_cal_cumlog'] = float(np.log1p(ST.win(nav, sd).r.reindex(xh.index)).sum() - np.log1p(ST.win(E50.reindex(nav.index).ffill(), sd).r.reindex(xh.index)).sum()) if len(xh) else None
    return o

if __name__ == '__main__':
    groups = sorted({k[:2] for k in PLANS})
    seeds_off = list(range(40)); seeds_j = list(range(20))
    if DRY: groups = [('AB', 'OFF'), ('B10', 'J3')]; seeds_off = [0, 1]; seeds_j = [0]
    # ---------- (A)(B) ideal ----------
    ij = [((st, g, s), m) for (st, g) in groups for s in (seeds_off if g == 'OFF' else list(range(40)) if not DRY else [0]) for m in ('base', 'disp', 'mdelay', 'both') if (st, g, s) in PLANS]
    res_i = run_pool(job_ideal, ij, 'ideal')
    agg = {}
    infos = {}
    for (key, mode, r, info) in res_i:
        agg.setdefault((key[0], key[1], mode), []).append(r)
        infos.setdefault((key[0], key[1], mode), []).append(info)
    NI = {}
    for (st, g, mode), rs in agg.items():
        t0, t1 = BOUNDS[(st, g)]
        NI[f'{st}|{g}|{mode}'] = pd.Series((1 + np.nan_to_num(np.mean(rs, axis=0), nan=0.0)).cumprod(), index=K.IDX[t0:t1 + 1])
    out = {'dry': DRY}
    if DRY:
        chk = {}
        for (st, g) in groups:
            key = (st, g, 0); t0, t1 = BOUNDS[(st, g)]
            ref, _ = L.sim_plan([(p[0], p[1], None, None) for p in PLANS[key]], t0, t1, 'normal', XR)
            nav, _ = E.ideal(D, PLANS[key], t0, t1, *L.cost_rates('normal'))
            chk[f'{st}|{g}|s0'] = float(np.max(np.abs(nav / ref - 1)))
        out['ideal_identity_maxrel'] = chk
    if not DRY:
        NV = pd.read_parquet(C.HERE / 'navs_v7.parquet'); NJ = pd.read_parquet(C.HERE / 'navs_jscan_v7.parquet')
        chk = {}
        for (st, g) in groups:
            nav = NI[f'{st}|{g}|base']
            ref = NV[f'AB_V7{"_B10" if st == "B10" else ""}|normal'] if g == 'OFF' else NJ[f'AB_V7{"_B10" if st == "B10" else ""}|normal|{g}']
            ref = ref.dropna(); ref = ref[(ref.index >= nav.index[0]) & (ref.index <= nav.index[-1])]
            chk[f'{st}|{g}'] = float(np.max(np.abs(nav.reindex(ref.index) / nav.iloc[0] / (ref / ref.iloc[0]) - 1)))
        out['ideal_identity_maxrel'] = chk
        assert max(chk.values()) < 1e-9, chk
    dec = {}
    for (st, g) in groups:
        b = stats(NI[f'{st}|{g}|base'], (st, g))
        row = {'base': {k: b[k] for k in ('CAGR', 't_0050', 't_Y19on')}}
        for m in ('disp', 'mdelay', 'both'):
            o = stats(NI[f'{st}|{g}|{m}'], (st, g)); row[m] = {k: o[k] for k in ('CAGR', 't_0050', 't_Y19on')}
            row[m]['dCAGR_pp'] = o['CAGR'] - b['CAGR']
        row['n_disp_sells_per_seed'] = float(np.mean([i['n_disp_sells'] for i in infos[(st, g, 'disp')]]))
        row['n_merger_delay_per_seed'] = float(np.mean([i['n_merger_delay'] for i in infos[(st, g, 'mdelay')]]))
        dec[f'{st}|{g}'] = row
    out['decomp_zero_impact'] = dec
    log('ideal done')
    # ---------- (C) exec ----------
    jobs = []
    for (st, g) in groups:
        ss = seeds_off if g == 'OFF' else seeds_j
        for s in ss:
            for model in ('K5T', 'TICK'):
                for cost in ('normal', 'stress'):
                    for tier in (E.TIERS if not DRY else ['30萬', '1000萬']):
                        for ver in ('FIX', 'GROW'):
                            jobs.append(((st, g, s), model, cost, tier, ver))
        if g == 'OFF':
            for s in ss[:20]:
                for tier in (E.TIERS if not DRY else ['30萬']):
                    for ver in ('FIX', 'GROW'):
                        jobs.append(((st, g, s), 'K5', 'normal', tier, ver))
    log('exec jobs', len(jobs))
    res = run_pool(job_exec, jobs, 'exec')
    agg = {}; accs = {}; seedt = {}
    for (a, r, acc) in res:
        key, model, cost, tier, ver = a
        k = (key[0], key[1], model, cost, tier, ver)
        agg.setdefault(k, []).append(r); accs.setdefault(k, []).append(acc)
        if key[1] == 'OFF':
            t0, t1 = BOUNDS[key[:2]]
            nv = pd.Series((1 + np.nan_to_num(r, nan=0.0)).cumprod(), index=K.IDX[t0:t1 + 1])
            seedt.setdefault(k, []).append(ST.tstat(ST.xs(nv, E50, GRID[key[:2]], K.IDX[t0], K.IDX[t1])))
    NE = {}; TAB = {}
    for k, rs in agg.items():
        t0, t1 = BOUNDS[k[:2]]
        nav = pd.Series((1 + np.nan_to_num(np.mean(rs, axis=0), nan=0.0)).cumprod(), index=K.IDX[t0:t1 + 1])
        name = '|'.join(k); NE[name] = nav
        o = stats(nav, k[:2]); o['acc'] = {kk: float(np.mean([a[kk] for a in accs[k]])) for kk in accs[k][0]}; o['n_seeds'] = len(rs)
        yrs = (K.IDX[t1] - K.IDX[t0]).days / 365.25
        o['acc']['cost_pp_per_year'] = {kk: o['acc']['pct_' + kk] / yrs * 100 for kk in ('tax', 'fee', 'impact', 'odd_spread', 'disp_cost')}
        if k in seedt:
            ts = np.array(seedt[k]); o['seed_t'] = {'median': float(np.median(ts)), 'min': float(ts.min()), 'max': float(ts.max()), 'n_below_2.0': int((ts < 2.0).sum()), 'n_below_2.3': int((ts < 2.3).sum())}
        TAB[name] = o
    out['table'] = TAB
    # j 分布摘要（每個 strat × model × cost × tier × ver）
    summ = {}
    for st in ('AB', 'B10'):
        for model in ('K5T', 'TICK'):
            for cost in ('normal', 'stress'):
                for tier in E.TIERS:
                    for ver in ('FIX', 'GROW'):
                        rows = [TAB.get('|'.join((st, f'J{j}', model, cost, tier, ver))) for j in range(13)]
                        rows = [r for r in rows if r is not None]
                        if not rows: continue
                        f = lambda key, q: float(np.percentile([r[key] for r in rows], q))
                        summ['|'.join((st, model, cost, tier, ver))] = {
                            'n_j': len(rows), 'CAGR_median': f('CAGR', 50), 'CAGR_p10': f('CAGR', 10), 't_median': f('t_0050', 50), 't_p10': f('t_0050', 10),
                            'excess_vs_0050_median_pp': f('CAGR_minus_0050', 50), 'excess_vs_0050_p10_pp': f('CAGR_minus_0050', 10),
                            'E50_CAGR_median': f('E50_CAGR', 50), 'maxDD_median': f('maxDD', 50), 't_Y19on_median': f('t_Y19on', 50),
                            'CAGR_HOLD_median': f('CAGR_HOLD', 50), 'E50_CAGR_HOLD_median': f('E50_CAGR_HOLD', 50), 't_HOLD_median': f('t_HOLD', 50),
                            'CAGR_by_j': [r['CAGR'] for r in rows], 't_by_j': [r['t_0050'] for r in rows]}
    out['jsumm'] = summ
    # 實盤預期表（事前登記：主欄＝B10 可執行 j 中位數、一般成本；並列上界 K5T／下界 TICK；兩模型對「是否贏 0050」相反 → 無法判斷）
    tbl = {}
    for tier in E.TIERS:
        for ver in ('FIX', 'GROW'):
            u = summ.get(f'B10|K5T|normal|{tier}|{ver}'); l = summ.get(f'B10|TICK|normal|{tier}|{ver}')
            if not u or not l: continue
            wu, wl = u['excess_vs_0050_median_pp'] > 0, l['excess_vs_0050_median_pp'] > 0
            tbl[f'{tier}|{ver}'] = {'CAGR_K5T(上界成本)': u['CAGR_median'], 'CAGR_TICK(下界成本)': l['CAGR_median'], 'CAGR_p10_K5T': u['CAGR_p10'], 'CAGR_p10_TICK': l['CAGR_p10'],
                                    't_K5T': u['t_median'], 't_TICK': l['t_median'], '0050_CAGR': u['E50_CAGR_median'],
                                    'excess_K5T_pp': u['excess_vs_0050_median_pp'], 'excess_TICK_pp': l['excess_vs_0050_median_pp'],
                                    'beat_0050': '贏' if (wu and wl) else ('輸' if (not wu and not wl) else '無法判斷'),
                                    'stress_CAGR_K5T': summ.get(f'B10|K5T|stress|{tier}|{ver}', {}).get('CAGR_median'), 'stress_CAGR_TICK': summ.get(f'B10|TICK|stress|{tier}|{ver}', {}).get('CAGR_median')}
    out['live_expect_table'] = tbl
    if not DRY:
        pd.DataFrame(NE).to_parquet(C.HERE / 'navs_exec7.parquet')
        pd.DataFrame(NI).to_parquet(C.HERE / 'navs_ideal7.parquet')
        C.save_key('RESULTS_EXEC', out)
    else:
        log('DRY OK', len(TAB), 'cells', sorted(out['ideal_identity_maxrel'].items()) if 'ideal_identity_maxrel' in out else '')
    log('done')
