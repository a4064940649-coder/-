# R4/R5 彙整（解封後）：逐格最差表（凍結版、F1–F5 修正版、原引擎口徑診斷版）、P3 30 萬、P4 歸因、原引擎逐 j／逐年比對
# 用法：python3 -I -B audit_run.py J analysis.py
import sys, os, json, pickle, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import xdata, xstats
R = json.load(open(os.path.join(HERE, 'results.json')))
X = xdata.load(); dates = X['dates']; E = X['E']
OFF = {'REAL_opt': dict(N100=1.039, p10=-2.019, t=0.532, y19=-6.220, N300=-0.444, N1000=-3.321, S100=-6.030),
       'REAL_p90': dict(N100=0.961, p10=-2.254, t=0.447, y19=-5.799, N300=-0.997, N1000=-3.109, S100=-6.262),
       'REAL_carry': dict(N100=0.857, p10=-1.555, t=0.453, y19=-6.187, N300=-0.542, N1000=-3.083, S100=-6.280)}
OFFW = dict(N100=0.86, t=0.45, p10=-2.25, y19=-6.22, N300=-1.00, N1000=-3.32, S100=-6.28)
VS = ['REAL_opt', 'REAL_p90', 'REAL_carry']
def grab(cells, v, cn):
    c = cells.get(f'E|{v}|cons_main|{cn}')
    return None if c is None else c['sum']
def table(cells, label):
    rows = []; got = {}
    for v in VS:
        g = {}
        for cn in ['N100', 'N300', 'N1000', 'S100', 'W30', 'O30']:
            s = grab(cells, v, cn)
            if s is None: continue
            g[cn] = 100 * s['exc_med']
            if cn == 'N100':
                g['p10'] = 100 * s['exc_p10']; g['t'] = s['t_med']; g['y19'] = 100 * s['exc_y19_med']; g['ann_y19'] = 100 * s['ann_y19_med']; g['ann0050_y19'] = 100 * s['ann0050_y19_med']
        got[v] = g
        for k, off in OFF[v].items():
            if k in g:
                d = g[k] - off; thr = 0.05 if k == 't' else 0.1
                rows.append(dict(version=label, variant=v, stat=k, official=off, rewrite=round(g[k], 3), diff=round(d, 3), diverge=bool(abs(d) > thr)))
    W = {}
    for k in ['N100', 't', 'p10', 'y19', 'N300', 'N1000', 'S100', 'W30', 'O30']:
        vals = {v: got[v][k] for v in VS if k in got.get(v, {})}
        if len(vals) == 3:
            vw = min(vals, key=vals.get); W[k] = dict(value=round(vals[vw], 3), variant=vw, official=OFFW.get(k), diff=(round(vals[vw] - OFFW[k], 3) if k in OFFW else None),
                                                    diverge=(bool(abs(vals[vw] - OFFW[k]) > (0.05 if k == 't' else 0.1)) if k in OFFW else None))
    if all('y19' in got.get(v, {}) for v in VS):
        vw = min(VS, key=lambda v: got[v]['y19']); W['ann_y19_of_worst'] = round(got[vw]['ann_y19'], 3); W['ann0050_y19'] = round(got[vw]['ann0050_y19'], 3)
    return rows, W, got
OUT = {}
versions = [('frozen_F5stats', R.get('RESTAT_FROZEN_F5', {}).get('cells', {})), ('fixed_F1toF5', R.get('RESULTS_FIX5', {}).get('cells', {})),
            ('gapread_diag', R.get('RESULTS_GAPREAD', {}).get('cells', {})), ('origmimic_diag', R.get('RESULTS_MIMIC', {}).get('cells', {}))]
for lab, cells in versions:
    if not cells: continue
    rows, W, got = table(cells, lab)
    OUT[lab] = dict(underlying=rows, worst=W, n_diverge_underlying=sum(r['diverge'] for r in rows), n_underlying=len(rows))
# P3：30 萬整股主欄 vs 100 萬（凍結版淨值；修正版若有）
def p3(navp, sigsrc):
    if not os.path.exists(navp): return None
    nv = pickle.load(open(navp, 'rb')); out = {}
    a21 = int(np.searchsorted(dates.values, np.datetime64('2021-01-01')))
    for v in VS:
        d = []
        for j in range(13):
            k30, k100 = (f'E|{v}|cons_main|W30', j), (f'E|{v}|cons_main|N100', j)
            if k30 not in nv or k100 not in nv: continue
            a, b = nv[k30], nv[k100]; end = int(np.where(np.isfinite(b))[0][-1])
            d.append(100 * (xstats.ann(a, a21, end, dates) - xstats.ann(b, a21, end, dates)))
        if d: out[v] = dict(diff_2021on_W30_minus_N100_med=float(np.median(d)), per_j=d)
    return out
OUT['P3_2021on'] = {'frozen': p3(xdata.SCR + 'formal_navs.pkl', None), 'fixed': p3(xdata.SCR + 'navs_FIX5.pkl', None)}
# P4 階梯（凍結版，REAL_opt，2019 起 j 中位數年化）
LAD = ['S0_close_frictionless', 'S1_open', 'S2_lim', 'S3_tax', 'S4_fee', 'S5_imp_caps', 'S6_lots_odd', 'S7_full']
C = R['RESULTS']['cells']; lad = {}
nvf = pickle.load(open(xdata.SCR + 'formal_navs.pkl', 'rb'))
f19 = int(np.searchsorted(dates.values, np.datetime64('2019-01-01')))
for s in LAD:
    c = f'A|REAL_opt|{s}'
    if c not in C: continue
    a = []; full = []
    for j in range(13):
        nv = nvf[(c, j)]; end = int(np.where(np.isfinite(nv))[0][-1])
        a.append(100 * xstats.ann(nv, f19, end, dates))
    lad[s] = dict(ann_y19_med_F5base=float(np.median(a)), ann_full_med=100 * C[c]['sum']['ann_med'], exc_full_med=100 * C[c]['sum']['exc_med'], t=C[c]['sum']['t_med'])
steps = {}
for a_, b_ in zip(LAD[:-1], LAD[1:]):
    steps[b_] = dict(d_ann_y19=lad[b_]['ann_y19_med_F5base'] - lad[a_]['ann_y19_med_F5base'], d_ann_full=lad[b_]['ann_full_med'] - lad[a_]['ann_full_med'])
OUT['P4_ladder'] = dict(levels=lad, steps=steps,
                        cost_class_y19=-(steps['S3_tax']['d_ann_y19'] + steps['S4_fee']['d_ann_y19'] + steps['S5_imp_caps']['d_ann_y19']),
                        timing_class_y19=-(steps['S1_open']['d_ann_y19'] + steps['S2_lim']['d_ann_y19']),
                        rest_y19=-(steps['S6_lots_odd']['d_ann_y19'] + steps['S7_full']['d_ann_y19']),
                        total_y19=lad['S0_close_frictionless']['ann_y19_med_F5base'] - lad['S7_full']['ann_y19_med_F5base'])
# 原引擎（K 重跑）逐 j 比對：凍結版與修正版
def vs_orig(navp, tag):
    kp = xdata.SCR.replace('g6A/', 'g6_sealed/') + 'k_orig_engine.pkl'
    if not (os.path.exists(kp) and os.path.exists(navp)): return None
    KO = pickle.load(open(kp, 'rb')); nv = pickle.load(open(navp, 'rb')); rows = []
    for (v, j, cn), s in KO['avg'].items():
        mg = 'orig' if v == 'orig' else 'cons_main'; key = (f'E|{v}|{mg}|{cn}', j)
        if key not in nv: continue
        mn = nv[key]; on = s.reindex(dates).values
        st = int(np.where(np.isfinite(mn))[0][0]); end = int(np.where(np.isfinite(mn))[0][-1]); on = on / on[st]
        yr = {}
        for y in range(2006, 2027):
            i0 = max(st, xstats.last_le(dates, f'{y-1}-12-31')); i1 = min(end, xstats.last_le(dates, f'{y}-12-31'))
            if i1 > i0: yr[y] = 100 * ((mn[i1] / mn[i0]) - (on[i1] / on[i0]))
        rows.append(dict(variant=v, cell=cn, j=j, ann_diff=100 * (xstats.ann(mn, st, end, dates) - xstats.ann(on, st, end, dates)),
                         y19_diff=100 * (xstats.ann(mn, f19, end, dates) - xstats.ann(on, f19, end, dates)), sum_abs_year=float(np.sum(np.abs(list(yr.values())))), year=yr))
    df = pd.DataFrame(rows)
    summ = df.groupby(['variant', 'cell']).agg(ann_diff_med=('ann_diff', 'median'), ann_diff_maxabs=('ann_diff', lambda x: float(np.max(np.abs(x)))),
                                               y19_diff_med=('y19_diff', 'median'), sum_abs_year_med=('sum_abs_year', 'median')).reset_index()
    yd = {}
    for (v, cn), g in df.groupby(['variant', 'cell']):
        Y = pd.DataFrame(list(g.year)); yd[f'{v}|{cn}'] = {int(k): float(x) for k, x in Y.median().items()}
    return dict(summary=summ.to_dict('records'), yearly_median_diff_pp=yd)
OUT['vs_orig_engine'] = {'frozen': vs_orig(xdata.SCR + 'formal_navs.pkl', 'frozen'), 'fixed': vs_orig(xdata.SCR + 'navs_FIX5.pkl', 'fixed')}
# Track L（名單層）各版
TL = {}
for lab, key in [('frozen', 'RESTAT_FROZEN_F5'), ('fix5_F1F2', 'RESULTS_FIX5'), ('trackl6_F1F2F6_rev', 'RESULTS_TRACKL6')]:
    cells = R.get(key, {}).get('cells', {})
    TL[lab] = {c: {k: (100 * v if k != 't_med' else v) for k, v in cells[c]['sum'].items() if k in ('exc_med', 'exc_p10', 't_med', 'exc_y19_med', 'ann_y19_med')} for c in cells if c.startswith('L|')}
OUT['trackL'] = TL
# 紙上：ia 口徑年化（年數＝窗數/12）換算
PP = {}
for k, v in R['RESULTS']['paper'].items():
    nw = v['n_win']; d0 = pd.Timestamp('2006-04-11'); d1 = pd.Timestamp('2026-10-12'); yrs = (d1 - d0).days / 365.25
    PP[k] = dict(ann_calendar=100 * v['ann'], ann_ia_months=100 * ((1 + v['ann']) ** (yrs / (nw / 12)) - 1), t=v['t'])
OUT['paper_ia'] = PP
json.dump(OUT, open(os.path.join(HERE, 'analysis.json'), 'w'), ensure_ascii=False, indent=1, default=float)
for lab in [l for l, _ in versions if l in OUT]:
    print(lab, 'diverge', OUT[lab]['n_diverge_underlying'], '/', OUT[lab]['n_underlying']); print(json.dumps(OUT[lab]['worst'], ensure_ascii=False))
print('P3', json.dumps({k: ({v: round(x['diff_2021on_W30_minus_N100_med'], 3) for v, x in (val or {}).items()}) for k, val in OUT['P3_2021on'].items()}))
print('P4', {k: round(v, 3) for k, v in OUT['P4_ladder'].items() if not isinstance(v, dict)}); print({k: round(v['d_ann_y19'], 3) for k, v in OUT['P4_ladder']['steps'].items()})
for k, v in OUT['vs_orig_engine'].items():
    if v: print('vs_orig', k); [print(' ', r) for r in v['summary']]
