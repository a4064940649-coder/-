#!/usr/bin/env python3
"""樣本外與全期報告（PREREG 必須已存在；規格不再改）。輸出 oos.json、navs_oos.parquet。"""
import sys, json, time, hashlib, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import st_core as C
T0 = time.time()
res = json.loads((C.HERE / 'results.json').read_text(encoding='utf-8'))
assert 'PREREG' in res and 'PREREG_sha256' in res, 'PREREG 不存在：不能計算樣本外'
assert hashlib.sha256(json.dumps(res['PREREG'], ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest() == res['PREREG_sha256']
import iterate as IT          # 只用它的 SPECS 與 run_spec（不會再跑迭代）
st = C.setup(); K, V, R = C.K, C.V, C.R
SELV = res['PREREG']['selected_version']; SPEC = IT.SPECS[SELV]
assert SPEC == res['PREREG']['spec']
KEYS = ['SEL', 'VAL', 'HOLD', 'POST', 'ALL', 'S1', 'S2', 'S3', 'S4', 'Y2008']
off_days, off_end = C.official_list()
WO = C.windows_from(off_days, off_end)
E1012 = off_end
JS = list(range(-8, 13))
out = {'PREREG_sha256': res['PREREG_sha256'], 'selected': SELV}
NAVS = {}

def run_single(days, end, cost, seed=0, ab=True):
    pl = C.plan_single(days, seed=seed, ab=ab)
    t0 = C.IDX.get_loc(days[0][1]); t1 = C.IDX.get_loc(end)
    nav, _, lg = C.tsim([pl], *cost, t0, t1, xrules=st['XR'], log=True)
    return nav, pl, lg, t0

def short(b, keys=KEYS):
    return {k: {q: (round(v, 4) if isinstance(v, float) else v) for q, v in b[k].items()} for k in keys if k in b}

# ---------------- 官方日曆 ----------------
navOFF, plOFF, lgOFF, t0OFF = run_single(off_days, off_end, C.NORMAL, ab=False)
EW = C.ew_nav(plOFF, t0OFF, C.IDX.get_loc(off_end))
NAVS['OFF'] = navOFF; NAVS['EW'] = EW
bOFF = C.block(navOFF, WO, KEYS, ew=EW)
out['OFF'] = dict(stats=short(bOFF), endpoint=C.endpoint(navOFF, WO, 'ALL'), turnover=C.turnover_month(lgOFF, navOFF, t0OFF))
assert abs(bOFF['ALL']['t_0050'] - 2.5486212673374764) < 1e-9
out['OFF_AB'] = {}
nav_ab = []
for s in range(5):
    nv, _, _, _ = run_single(off_days, off_end, C.NORMAL, seed=s, ab=True)
    nav_ab.append(nv)
    b = C.block(nv, WO, KEYS)
    out['OFF_AB'][s] = dict(CAGR=b['ALL']['CAGR'], t=b['ALL']['t_0050'], SEL_t=b['SEL']['t_0050'], VAL_t=b['VAL']['t_0050'], HOLD_t=b['HOLD']['t_0050'],
                            S4_CAGR=b['S4']['CAGR'], maxDD=b['ALL']['maxDD'], ex=b['ALL']['ex_mean_pp'])
NAVS['OFF_AB0'] = nav_ab[0]
print('OFF', round(bOFF['ALL']['CAGR'], 2), round(bOFF['ALL']['t_0050'], 3), 'OFF_AB', {s: round(v['t'], 2) for s, v in out['OFF_AB'].items()}, round(time.time() - T0), 's', flush=True)

# ---------------- 全期 J 掃描 ----------------
out['scan'] = {}; EXo = {}; EXc = {}
for j in JS:
    days, end = C.version_days(j)
    nav, pl, lg, t0 = run_single(days, end, C.NORMAL)
    navg, _, _, _ = run_single(days, end, C.GROSS)
    W = C.windows_from(days, end)
    b = C.block(nav, W, KEYS); bg = C.block(navg, W, ['SEL', 'POST', 'ALL'])
    ep = C.endpoint(nav, W, 'ALL')
    # 共同窗（官方日曆窗；評價點對齊，只差換股日）
    bc = C.block(nav, WO, ['SEL', 'ALL']) if E1012 <= end else C.block(nav, [w for w in WO if w[2] <= end], ['SEL', 'ALL'])
    out['scan'][f'J{j:+d}'] = dict(stats=short(b), gross={k: round(bg[k]['ex_mean_pp'], 4) for k in bg}, t_cal_ALL=ep['t_cal'], endpoint=ep,
                                   t_common_ALL=bc['ALL']['t_0050'], t_common_SEL=bc['SEL']['t_0050'], turnover=C.turnover_month(lg, nav, t0))
    EXo[j] = C.wret(nav, W) - C.wret(C.E50.reindex(nav.index), W)
    NAVS[f'J{j:+d}'] = nav
    print('J', j, round(b['ALL']['CAGR'], 2), round(b['ALL']['t_0050'], 2), 'SEL', round(b['SEL']['t_0050'], 2), 'VAL', round(b['VAL']['t_0050'], 2),
          'HOLD', round(b['HOLD']['t_0050'], 2), 'mdd', round(b['ALL']['maxDD'], 1), round(time.time() - T0), 's', flush=True)
out['J0_seeds'] = {}
for s in range(5):
    days, end = C.version_days(0)
    nv, _, _, _ = run_single(days, end, C.NORMAL, seed=s)
    b = C.block(nv, C.windows_from(days, end), ['ALL', 'SEL', 'POST'])
    out['J0_seeds'][s] = dict(CAGR=b['ALL']['CAGR'], t=b['ALL']['t_0050'], SEL_t=b['SEL']['t_0050'], POST_t=b['POST']['t_0050'])

def dist(x):
    x = np.asarray(x, float)
    return dict(min=float(x.min()), p10=float(np.percentile(x, 10)), p25=float(np.percentile(x, 25)), median=float(np.median(x)),
                p75=float(np.percentile(x, 75)), p90=float(np.percentile(x, 90)), max=float(x.max()), sd=float(x.std(ddof=1)))
def pct_of(v, x):
    x = np.asarray(x, float); return float((x < v).mean() + 0.5 * (x == v).mean())
D = {}
for key in ('ALL', 'SEL', 'VAL', 'HOLD', 'POST', 'S1', 'S2', 'S3', 'S4'):
    ts = [out['scan'][f'J{j:+d}']['stats'][key]['t_0050'] for j in JS]
    cg = [out['scan'][f'J{j:+d}']['stats'][key]['CAGR'] for j in JS]
    ex = [out['scan'][f'J{j:+d}']['stats'][key]['ex_mean_pp'] for j in JS]
    D[key] = dict(t=dist(ts), CAGR=dist(cg), ex_mean_pp=dist(ex), OFF_t=bOFF[key]['t_0050'], OFF_pctile=pct_of(bOFF[key]['t_0050'], ts),
                  OFF_AB_t_mean=float(np.mean([C.block(n, WO, [key])[key]['t_0050'] for n in nav_ab])),
                  J1_t=out['scan']['J+1']['stats'][key]['t_0050'])
    D[key]['OFF_AB_pctile'] = pct_of(D[key]['OFF_AB_t_mean'], ts)
D['ALL']['t_cal'] = dist([out['scan'][f'J{j:+d}']['t_cal_ALL'] for j in JS])
D['ALL']['t_common'] = dist([out['scan'][f'J{j:+d}']['t_common_ALL'] for j in JS])
D['ALL']['maxDD'] = dist([out['scan'][f'J{j:+d}']['stats']['ALL']['maxDD'] for j in JS])
D['ALL']['drop_last1'] = dist([out['scan'][f'J{j:+d}']['endpoint']['drop_last1'] for j in JS])
D['ALL']['drop_best10'] = dist([out['scan'][f'J{j:+d}']['endpoint']['drop_best10'] for j in JS])
out['dist'] = D
p10 = D['ALL']['t']['p10']
out['judgment_A'] = dict(p10_t_ALL=p10, flag_timing_sensitive=bool(p10 < 2.0),
                         interval_CAGR_p10_p90=[D['ALL']['CAGR']['p10'], D['ALL']['CAGR']['p90']], interval_t_p10_p90=[D['ALL']['t']['p10'], D['ALL']['t']['p90']])
# 營收新鮮度（全期與 POST）
Mo = pd.DataFrame(EXo)
def fresh(M, key):
    a = C.sel(M[[1, 2, 3, 4, 5]].mean(axis=1), key); b = C.sel(M[list(range(-8, 0))].mean(axis=1), key)
    dd = (a - b).dropna(); return dict(diff_pp=float(dd.mean() * 100), t=C.tstat(dd), n=int(len(dd)))
out['freshness_normal'] = {k: fresh(Mo, k) for k in ('SEL', 'POST', 'ALL', 'S3', 'S4')}
disp = Mo.std(axis=1, ddof=1) * 100
out['timing_dispersion_pp'] = {k: float(C.sel(disp, k).mean()) for k in ('SEL', 'VAL', 'HOLD', 'S1', 'S2', 'S3', 'S4', 'Y2008', 'ALL')}
print('A', out['judgment_A'], round(time.time() - T0), 's', flush=True)

# ---------------- 12 個分批版本 ----------------
COSTS = IT.COSTS
def run_v(spec, cost, log=False):
    return IT.run_spec(spec, cost, max_m=None, t_end=E1012, log=log)
J1 = {cn: run_v(dict(js=[1], a='a0', b='b0', c='c0'), COSTS[cn], log=(cn == 'normal')) for cn in COSTS}
NAVS['J1_common'] = J1['normal'][0]
OFFc = {cn: run_single(off_days, off_end, COSTS[cn], ab=False)[0] for cn in COSTS}
out['versions'] = {}
SELNAV = {}
for nm, spec in IT.SPECS.items():
    o = {'spec': spec}
    for cn in COSTS:
        nav, navs, lg, pls, t0 = run_v(spec, COSTS[cn], log=(cn == 'normal'))
        b = C.block(nav, WO, KEYS, ew=EW)
        o[cn] = dict(stats=short(b), paired_J1={k: C.paired(nav, J1[cn][0], WO, k) for k in ('SEL', 'VAL', 'HOLD', 'POST', 'ALL')},
                     paired_OFF={k: C.paired(nav, OFFc[cn], WO, k) for k in ('SEL', 'VAL', 'HOLD', 'POST', 'ALL')},
                     sd_change_vs_J1={k: b[k]['ex_sd_pp'] / C.block(J1[cn][0], WO, [k])[k]['ex_sd_pp'] - 1 for k in ('SEL', 'POST', 'ALL')})
        if cn == 'normal':
            o[cn]['turnover'] = C.turnover_month(lg, nav, t0)
            o[cn]['capacity_ALL'] = C.capacity(lg['trades'], nav, t0)
            o[cn]['n_frozen'] = len(lg['frozen']); o[cn]['n_exits'] = len(lg['exits'])
            if nm == SELV: SELNAV.update(nav=nav, navs=navs, lg=lg, pls=pls, t0=t0)
        if nm == SELV: NAVS[f'{nm}_{cn}'] = nav
    out['versions'][nm] = o
    n = o['normal']
    print(nm, 'ALL', round(n['stats']['ALL']['CAGR'], 2), round(n['stats']['ALL']['t_0050'], 2), 'VAL pair', round(n['paired_J1']['VAL']['mean_pp'], 3),
          'HOLD pair', round(n['paired_J1']['HOLD']['mean_pp'], 3), 'POST sdchg', round(n['sd_change_vs_J1']['POST'], 3), round(time.time() - T0), 's', flush=True)
bJ1 = C.block(J1['normal'][0], WO, KEYS, ew=EW)
out['J1_common'] = dict(stats=short(bJ1), endpoint=C.endpoint(J1['normal'][0], WO, 'ALL'), capacity_ALL=C.capacity(J1['normal'][2]['trades'], J1['normal'][0], J1['normal'][4]),
                        paired_OFF={k: C.paired(J1['normal'][0], OFFc['normal'], WO, k) for k in ('SEL', 'VAL', 'HOLD', 'POST', 'ALL')})

# ---------------- 選中版本細節 ----------------
nav = SELNAV['nav']; pls = SELNAV['pls']; t0 = SELNAV['t0']
det = {}
det['endpoint'] = {k: C.endpoint(nav, WO, k) for k in ('ALL', 'SEL', 'POST')}
det['endpoint_OFF'] = {k: C.endpoint(navOFF, WO, k) for k in ('ALL', 'SEL', 'POST')}
det['endpoint_J1'] = {k: C.endpoint(J1['normal'][0], WO, k) for k in ('ALL', 'SEL', 'POST')}
# 判定 B（挑選期）
vn = out['versions'][SELV]
pj = vn['normal']['paired_J1']['SEL']
dd = C.sel(C.wret(nav, WO) - C.wret(J1['normal'][0], WO), 'SEL').dropna()
best = dd.idxmax()
e50w = C.wret(C.E50.reindex(nav.index), WO)
exv = C.sel(C.wret(nav, WO) - e50w, 'SEL').dropna(); exb = C.sel(C.wret(J1['normal'][0], WO) - e50w, 'SEL').dropna()
CUT = dict(off_days)[pd.Period('2019-01', 'M')]
capS = C.capacity([x for x in SELNAV['lg']['trades'] if x['d'] < CUT], nav, t0)
capJ = C.capacity([x for x in J1['normal'][2]['trades'] if x['d'] < CUT], J1['normal'][0], J1['normal'][4])
capSp = C.capacity([x for x in SELNAV['lg']['trades'] if x['d'] >= CUT], nav, t0)
capJp = C.capacity([x for x in J1['normal'][2]['trades'] if x['d'] >= CUT], J1['normal'][0], J1['normal'][4])
B = dict(B1=dict(value=pj['mean_pp'], pass_=pj['mean_pp'] >= -0.05),
         B2=dict(sd_change=vn['normal']['sd_change_vs_J1']['SEL'], pass_=vn['normal']['sd_change_vs_J1']['SEL'] <= -0.10),
         B3=dict(share_sel=capS['share_gt'], share_J1=capJ['share_gt'], ratio=capS['share_gt'] / capJ['share_gt'], pass_=capS['share_gt'] <= 0.5 * capJ['share_gt']),
         B4=dict(flat=vn['stress_flat']['paired_J1']['SEL']['mean_pp'], plus03=vn['stress_plus03']['paired_J1']['SEL']['mean_pp'],
                 pass_=min(vn['stress_flat']['paired_J1']['SEL']['mean_pp'], vn['stress_plus03']['paired_J1']['SEL']['mean_pp']) >= -0.05),
         B5=dict(best_month=str(best), best_pp=float(dd.max() * 100), B1_ex_best=float(dd.drop(best).mean() * 100),
                 B2_ex_best=float(exv.drop(best).std(ddof=1) / exb.drop(best).std(ddof=1) - 1)))
B['B5']['pass_'] = B['B5']['B1_ex_best'] >= -0.05 and B['B5']['B2_ex_best'] <= -0.10
B['all_pass'] = all(B[k]['pass_'] for k in ('B1', 'B2', 'B3', 'B4', 'B5'))
det['judgment_B'] = B
# 損益兩平滑價（單邊滑價 s；手續費、稅照舊）
grid = [0.0, 0.001, 0.002, 0.003, 0.005, 0.008, 0.012]
be = {}
for lab, fn in (('SEL', lambda s: IT.run_spec(SPEC, C.rates_slip(s), max_m=None, t_end=E1012)[0]),
                ('J1', lambda s: IT.run_spec(dict(js=[1], a='a0', b='b0', c='c0'), C.rates_slip(s), max_m=None, t_end=E1012)[0]),
                ('OFF', lambda s: run_single(off_days, off_end, C.rates_slip(s), ab=False)[0])):
    rows = []
    for s in grid:
        nv = fn(s); b = C.block(nv, WO, ['ALL', 'SEL'])
        rows.append((s, b['ALL']['t_0050'], b['ALL']['ex_mean_pp'], b['SEL']['t_0050'], b['SEL']['ex_mean_pp'], b['ALL']['CAGR']))
    A = np.array(rows)
    def cross(col, target):
        y = A[:, col]
        for i in range(len(y) - 1):
            if (y[i] - target) * (y[i + 1] - target) <= 0 and y[i] != y[i + 1]:
                return float(A[i, 0] + (target - y[i]) * (A[i + 1, 0] - A[i, 0]) / (y[i + 1] - y[i]))
        return float('nan') if y[0] < target else float('inf')
    be[lab] = dict(grid=[dict(slip=r[0], t_ALL=r[1], ex_ALL=r[2], t_SEL=r[3], ex_SEL=r[4], CAGR_ALL=r[5]) for r in rows],
                   s_t23_ALL=cross(1, 2.3), s_t20_ALL=cross(1, 2.0), s_ex0_ALL=cross(2, 0.0), s_t20_SEL=cross(3, 2.0), s_ex0_SEL=cross(4, 0.0))
    print('BE', lab, {k: v for k, v in be[lab].items() if k != 'grid'}, round(time.time() - T0), 's', flush=True)
det['breakeven'] = be
# 全額手續費（無折扣）
FULLFEE = (0.001425 + 0.001, 0.001425 + 0.003 + 0.001)
for lab, nv in (('SEL', IT.run_spec(SPEC, FULLFEE, max_m=None, t_end=E1012)[0]), ('OFF', run_single(off_days, off_end, FULLFEE, ab=False)[0])):
    b = C.block(nv, WO, ['ALL']); det.setdefault('full_fee', {})[lab] = dict(t=b['ALL']['t_0050'], CAGR=b['ALL']['CAGR'])
# 容量（全期、POST）
det['capacity'] = dict(ALL=vn['normal']['capacity_ALL'], J1_ALL=out['J1_common']['capacity_ALL'], SEL=capS, J1_SEL=capJ, POST=capSp, J1_POST=capJp)
# Bonferroni
m = res['PREREG']['multiple_testing']['m_total']
tA = vn['normal']['stats']['ALL']['t_0050']
det['bonferroni'] = dict(m=m, t_ALL=tA, p_one=C.p_one(tA), p_bonf=min(1.0, C.p_one(tA) * m), paired_ALL_t=vn['normal']['paired_J1']['ALL']['t'],
                         p_bonf_paired=min(1.0, C.p_one(vn['normal']['paired_J1']['ALL']['t']) * m),
                         OFF_p_one=C.p_one(bOFF['ALL']['t_0050']), OFF_p_bonf_m10=min(1.0, C.p_one(bOFF['ALL']['t_0050']) * 10))
# 分年（錨定月年份）：月超額與對 J+1 配對差
exS = C.wret(nav, WO) - e50w; exO = C.wret(navOFF, WO) - C.wret(C.E50.reindex(navOFF.index), WO); exJ = C.wret(J1['normal'][0], WO) - e50w
pd_ = (C.wret(nav, WO) - C.wret(J1['normal'][0], WO))
det['yearly'] = {int(y): dict(sel_ex_pp=float(exS[exS.index.year == y].sum() * 100), off_ex_pp=float(exO[exO.index.year == y].sum() * 100),
                              J1_ex_pp=float(exJ[exJ.index.year == y].sum() * 100), paired_vs_J1_pp=float(pd_[pd_.index.year == y].sum() * 100))
                 for y in sorted(set(exS.index.year))}
# 分產業貢獻（扣成本前、各批 1/k）與去掉最大貢献股
def contrib(plans, t_end, wts):
    con = {}; per = []
    for pl, w in zip(plans, wts):
        c, p = V.stock_contrib([(q[0], q[1], q[2], q[3]) for q in pl], t_end)
        for k2, v in c.items(): con[k2] = con.get(k2, 0.0) + v * w
        per.append(p)
    return con, per
def cagr_ex(per, wts, excl, t_end, a=None, b=None):
    tot = 0.0; yrs = None
    for p, w in zip(per, wts):
        g = 1.0
        for d, pk, r in p:
            if a is not None and not (a <= d <= b): continue
            rr = r.copy()
            if excl is not None:
                mm = np.array([C.COLS[j] == excl for j in pk])
                if mm.any() and (~mm).any(): rr[mm] = rr[~mm].mean()
            g *= 1 + rr.mean()
        tot += w * g
    d0 = min(p[0][0] for p in per); d1 = C.IDX[t_end]
    yrs = (d1 - d0).days / 365.25
    return (tot ** (1 / yrs) - 1) * 100
t_end = C.IDX.get_loc(E1012)
k = len(pls); wts = [1.0 / k] * k
con, per = contrib(pls, t_end, wts)
conO, perO = contrib([plOFF], t_end, [1.0])
top = max(con, key=con.get); topO = max(conO, key=conO.get)
SL = V.SL
ind = {}
for c2, v in con.items():
    i2 = SL['industry'].get(c2) if isinstance(SL['industry'].get(c2), str) else '（無）'
    ind[i2] = ind.get(i2, 0.0) + v
indO = {}
for c2, v in conO.items():
    i2 = SL['industry'].get(c2) if isinstance(SL['industry'].get(c2), str) else '（無）'
    indO[i2] = indO.get(i2, 0.0) + v
totc = sum(con.values()); totO = sum(conO.values())
det['industry'] = {'selected_top8': sorted(((a2, round(b2 / totc, 3)) for a2, b2 in ind.items()), key=lambda x: -x[1])[:8],
                   'OFF_top8': sorted(((a2, round(b2 / totO, 3)) for a2, b2 in indO.items()), key=lambda x: -x[1])[:8]}
det['top_stock'] = dict(selected=dict(code=top, name=SL['name'].get(top, ''), share=con[top] / totc, gross_CAGR=cagr_ex(per, wts, None, t_end), gross_CAGR_ex_top=cagr_ex(per, wts, top, t_end)),
                        OFF=dict(code=topO, name=SL['name'].get(topO, ''), share=conO[topO] / totO, gross_CAGR=cagr_ex(perO, [1.0], None, t_end), gross_CAGR_ex_top=cagr_ex(perO, [1.0], topO, t_end)),
                        top5_selected=sorted(((c2, SL['name'].get(c2, ''), round(v * 100, 1)) for c2, v in con.items()), key=lambda x: -x[2])[:5],
                        top5_OFF=sorted(((c2, SL['name'].get(c2, ''), round(v * 100, 1)) for c2, v in conO.items()), key=lambda x: -x[2])[:5])
# 分產業：電子業 vs 非電子的配對差無法直接拆（同一組合），改報「剔除電子業股票的貢獻後」的扣成本前月均（近似）——只報貢獻占比
out['selected_detail'] = det
print('B', B, round(time.time() - T0), 's', flush=True)

# ---------------- 位置平移的 t 分布（描述性） ----------------
rot = {}
for lab, base_js, j0s, c in (('T4', [0, 5, 10, 15], range(-8, -2), 'c1'), ('T2', [0, 10], range(-8, 3), 'c0')):
    rr = []
    for j0 in j0s:
        js = [j0 + q for q in base_js]
        sp = dict(js=js, a='a0', b='b0', c=c)
        nv = IT.run_spec(sp, C.NORMAL, max_m=None, t_end=E1012)[0]
        b = C.block(nv, WO, ['ALL', 'SEL', 'POST', 'S4'])
        rr.append(dict(js=js, t_ALL=b['ALL']['t_0050'], t_SEL=b['SEL']['t_0050'], t_POST=b['POST']['t_0050'], CAGR_ALL=b['ALL']['CAGR'], ex_ALL=b['ALL']['ex_mean_pp'],
                       ex_sd_ALL=b['ALL']['ex_sd_pp'], S4_CAGR=b['S4']['CAGR']))
    rot[lab] = dict(rows=rr, t_ALL=dist([x['t_ALL'] for x in rr]), t_SEL=dist([x['t_SEL'] for x in rr]), CAGR_ALL=dist([x['CAGR_ALL'] for x in rr]))
    print('rot', lab, rot[lab]['t_ALL'], round(time.time() - T0), 's', flush=True)
# 單日版在共同窗的分布（公平比較）
rot['single_common'] = dict(t_ALL=D['ALL']['t_common'])
out['rotation'] = rot
out['V4'] = 'V4 尚未凍結（o_v4_revenue_patch_official 無 results.json／data_freeze）；待凍結後以同一 oos.py 的 J 掃描補報。'
(C.HERE / 'oos.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
pd.DataFrame(NAVS).to_parquet(C.HERE / 'navs_oos.parquet')
print('done', round(time.time() - T0), 's')
