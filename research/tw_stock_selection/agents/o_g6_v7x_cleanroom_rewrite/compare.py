# R4 比對（解封後；J 角色）：Track E／L 對原正式數字；以原引擎淨值（K 重跑）定位分歧
import json, pickle, sys, numpy as np, pandas as pd
import xdata, xstats
HERE = '/home/user/-/research/tw_stock_selection/agents/o_g6_v7x_cleanroom_rewrite/'
SCR = xdata.SCR.replace('g6A/', '')
R = json.load(open(HERE + 'results.json'))['RESULTS']
# 原正式數字（o18 logs/v7x_run.log，原引擎；底層 18 格）
OFF = {
 'REAL_opt':   {'N100': dict(exc=1.039, p10=-2.019, t=0.532, y19=-6.220), 'N300': dict(exc=-0.444), 'N1000': dict(exc=-3.321), 'S100': dict(exc=-6.030)},
 'REAL_p90':   {'N100': dict(exc=0.961, p10=-2.254, t=0.447, y19=-5.799), 'N300': dict(exc=-0.997), 'N1000': dict(exc=-3.109), 'S100': dict(exc=-6.262)},
 'REAL_carry': {'N100': dict(exc=0.857, p10=-1.555, t=0.453, y19=-6.187), 'N300': dict(exc=-0.542), 'N1000': dict(exc=-3.083), 'S100': dict(exc=-6.280)},
 'orig':       {'N100': dict(exc=1.610, p10=-2.346, t=0.546, y19=-6.515)},
}
OFF_WORST = dict(exc=0.86, t=0.45, p10=-2.25, y19=-6.22, N300=-1.00, N1000=-3.32, S100=-6.28, cagr_y19=23.27, cagr0050_y19=29.49)
def mine(cell):
    s = R['cells'][cell]['sum']
    return dict(exc=100 * s['exc_med'], p10=100 * s['exc_p10'], t=s['t_med'], y19=100 * s['exc_y19_med'], cagr_y19=100 * s['ann_y19_med'],
                cagr0050_y19=100 * s['ann0050_y19_med'], s1=100 * s['exc_s1_med'], s2=100 * s['exc_s2_med'], s3=100 * s['exc_s3_med'], mdd=100 * s['mdd_med'], ann=100 * s['ann_med'])
rows = []
for v, cells in OFF.items():
    mg = 'orig' if v == 'orig' else 'cons_main'
    for cn, tg in cells.items():
        m = mine(f'E|{v}|{mg}|{cn}')
        for k, x in tg.items():
            d = m[k] - x; thr = 0.05 if k == 't' else 0.1
            rows.append(dict(variant=v, cell=cn, stat=k, official=x, rewrite=round(m[k], 3), diff=round(d, 3), diverge=bool(abs(d) > thr)))
T = pd.DataFrame(rows)
# 逐格最差
W = {}
vs = ['REAL_opt', 'REAL_p90', 'REAL_carry']
mm = {v: {cn: mine(f'E|{v}|cons_main|{cn}') for cn in ['N100', 'S100', 'N300', 'N1000', 'W30', 'O30']} for v in vs}
W['exc'] = min(mm[v]['N100']['exc'] for v in vs); W['t'] = min(mm[v]['N100']['t'] for v in vs); W['p10'] = min(mm[v]['N100']['p10'] for v in vs)
W['y19'] = min(mm[v]['N100']['y19'] for v in vs)
vy = min(vs, key=lambda v: mm[v]['N100']['y19']); W['cagr_y19'] = mm[vy]['N100']['cagr_y19']; W['cagr0050_y19'] = mm[vy]['N100']['cagr0050_y19']
for cn in ['N300', 'N1000', 'S100', 'W30', 'O30']: W[cn] = min(mm[v][cn]['exc'] for v in vs)
WC = [dict(stat=k, official=OFF_WORST.get(k), rewrite=round(W[k], 3), diff=(round(W[k] - OFF_WORST[k], 3) if k in OFF_WORST else None)) for k in W]
out = {'underlying': rows, 'worst_cell': WC, 'mine_by_variant': mm,
       'trackL': {c: mine(c) for c in R['cells'] if c.startswith('L|')},
       'trackE_orig_cons': mine('E|orig|cons_main|N100') if 'E|orig|cons_main|N100' in R['cells'] else None}
# 原引擎淨值（K 重跑）逐 j 比對：用同一套 xstats 算，分開「統計碼」與「引擎」差
try:
    KO = pickle.load(open(SCR + 'g6_sealed/k_orig_engine.pkl', 'rb'))
    navs = pickle.load(open(xdata.SCR + 'formal_navs.pkl', 'rb'))
    X = xdata.load(); dates = X['dates']; E = X['E']
    perj = []
    for (v, j, cn), s in KO['avg'].items():
        mg = 'orig' if v == 'orig' else 'cons_main'
        key = (f'E|{v}|{mg}|{cn}', j)
        if key not in navs: continue
        mn = navs[key]; on = s.reindex(dates).values
        st = int(np.where(np.isfinite(mn))[0][0]); end = int(np.where(np.isfinite(mn))[0][-1])
        on = on / on[st]
        a = xstats.ann(mn, st, end, dates); b = xstats.ann(on, st, end, dates)
        b18 = xstats.last_le(dates, '2018-12-31')
        yr = {}
        for y in range(2006, 2027):
            i0 = max(st, xstats.last_le(dates, f'{y-1}-12-31')); i1 = min(end, xstats.last_le(dates, f'{y}-12-31'))
            if i1 > i0: yr[y] = 100 * ((mn[i1] / mn[i0]) - (on[i1] / on[i0]))
        perj.append(dict(variant=v, cell=cn, j=j, ann_mine=100 * a, ann_orig=100 * b, diff=100 * (a - b),
                         y19_diff=100 * (xstats.ann(mn, b18, end, dates) - xstats.ann(on, b18, end, dates)),
                         sum_abs_year=float(np.sum(np.abs(list(yr.values())))), year_diff=yr))
    out['perj_vs_orig_engine'] = perj
except FileNotFoundError as e:
    out['perj_vs_orig_engine'] = str(e)
json.dump(out, open(HERE + 'compare.json', 'w'), ensure_ascii=False, indent=1, default=float)
print(T.to_string()); print(pd.DataFrame(WC).to_string())
