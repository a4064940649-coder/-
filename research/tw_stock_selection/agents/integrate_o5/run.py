#!/usr/bin/env python3
"""integrate_o5 回測：照 results.json 的 PREREG 重跑 AB（AB_V4 與 J+1，各 40 seeds）、組合、統計、SPA、SEL 機械選版、判定，寫回 RESULTS（只寫一次）。
引擎全部只讀 import：o_rev_event_trigger/fx_core.py（→ o_v4_revenue_patch_official/v4lib.py → main_v2/v2_core、o_disposal_rejudge/drj_core、
o_infra_v3_rebuild/o3_core、independent_audit/ia_core），o_main_spa_breaks/spa_lib.py 與 fam_common.py。
成本、成交、漲停遞補、停牌凍結、併購出清、AB 抽樣、SPA bootstrap 全部沿用；本檔只新寫袖子混合與統計彙整。
用法：python3 -I -B run.py（正式）；python3 -I -B run.py --smoke（2 seeds、B=200，只測管線，不寫 results.json、不印數字）。"""
import sys, json, time, math, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
AG = HERE.parent
DATA = HERE / 'data'
SMOKE = '--smoke' in sys.argv
import numpy as np, pandas as pd

def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()
RJ = HERE / 'results.json'
if not SMOKE:
    RES = json.loads(RJ.read_text(encoding='utf-8'))
    assert hashlib.sha256(json.dumps(RES['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == RES['PREREG_sha256'], 'PREREG 被改過'
    assert 'RESULTS' not in RES, 'RESULTS 已存在：只跑一次'
    for f, h in RES['PREREG']['sha256'].items():
        assert sha(f) == h, f'{f} 與 PREREG 登記的 sha256 不符'
# 資料副本必須與原檔相同
COPIES = {'navs_v4.parquet': AG / 'o_v4_revenue_patch_official/navs_v4.parquet', 'navs_oos.parquet': AG / 'o_rev_event_trigger/navs_oos.parquet',
          'integrate_o4_navs.parquet': AG / 'integrate_o4/navs.parquet', 'family_excess.parquet': AG / 'o_main_spa_breaks/data/family_excess.parquet',
          'family_members.csv': AG / 'o_main_spa_breaks/data/family_members.csv', 'plan_V4.csv': AG / 'o_v4_revenue_patch_official/plan_V4.csv',
          'etf_0050.csv': Path('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv')}
for k, f in COPIES.items(): assert sha(DATA / k) == sha(f), k

sys.path.insert(0, str(AG / 'o_rev_event_trigger'))
import fx_core as F
sys.path.pop(0)
sys.path.insert(0, str(AG / 'o_main_spa_breaks'))
import fam_common as C, spa_lib as SP
sys.path.pop(0)
L, K, V = F.L, F.K, F.V
T0_ = time.time()
def log(*a):
    if SMOKE: print(f'[{time.time() - T0_:6.0f}s]', *[x for x in a if isinstance(x, str)], flush=True)
    else: print(f'[{time.time() - T0_:6.0f}s]', *a, flush=True)

st = F.setup()
XR, YFX, Y5F, DMT = st['XR'], st['YFX'], st['Y5F'], st['DMT']
NAV4 = pd.read_parquet(DATA / 'navs_v4.parquet')
NOOS = pd.read_parquet(DATA / 'navs_oos.parquet')
NI4 = pd.read_parquet(DATA / 'integrate_o4_navs.parquet')
SEEDS = range(2) if SMOKE else range(40)
SLIPS = [0.0010, 0.0020] if SMOKE else [round(0.0010 + 0.0002 * i, 4) for i in range(16)]
COSTS = ['normal', 'stress'] + [('slip', s) for s in SLIPS]
A_, B_ = '2006-04-01', '2026-10-12'
VAL = {}

# ---------------- AB_V4（主策略單獨）：40 seeds ----------------
S0 = L.V4(YFX, Y5F, DMT)
t0, t1, sd = S0.bounds()
pl0 = S0.plan(10, 'official')
EXEC = [p[0] for p in pl0]
def mean_nav(lst): return (1 + pd.concat(lst, axis=1).mean(axis=1).fillna(0)).cumprod()
abr = {str(c): [] for c in COSTS}; abseed = {str(c): [] for c in ('normal', 'stress')}; abew = []
for s in SEEDS:
    S = L.V4(YFX, Y5F, DMT, unavail=V.make_unavail('mc', s)); pl = S.plan(10, 'official')
    assert [p[0] for p in pl] == EXEC
    for c in COSTS:
        nav, _ = L.sim_plan(pl, t0, t1, c, XR); abr[str(c)].append(nav.pct_change())
        if str(c) in abseed: abseed[str(c)].append(nav)
    ew, _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, t0, t1); abew.append(ew.pct_change())
ABV = {c: mean_nav(v) for c, v in abr.items()}
EWB = mean_nav(abew)
IX = K.IDX[t0:t1 + 1]
if not SMOKE:
    for c, ref in (('normal', 'AB_V4|normal'), ('stress', 'AB_V4|stress')):
        VAL[f'AB_V4_{c}_maxrel'] = float((ABV[c] / NAV4[ref].reindex(ABV[c].index) - 1).abs().max()); assert VAL[f'AB_V4_{c}_maxrel'] <= 1e-9, VAL
    VAL['AB_V4_EW_maxrel'] = float((EWB / NAV4['AB_V4|EW'].reindex(EWB.index) - 1).abs().max()); assert VAL['AB_V4_EW_maxrel'] <= 1e-9
log('AB_V4 done', VAL)

# ---------------- J+1（o_rev_event_trigger 選定版）：40 seeds ----------------
JD, JE = F.version_days(1)
RJ1 = F.ab_navs(JD, JE, mode='day', seeds=SEEDS, costs=tuple(COSTS), keep_seed=True)
J1V = {str(c): v.reindex(IX) for c, v in RJ1['mean'].items()}
J1seed = {str(c): [v.reindex(IX) for v in RJ1['seeds'][c]] for c in ('normal', 'stress')}
assert RJ1['t0'] == t0, (RJ1['t0'], t0)
if not SMOKE:
    for c, ref in (('normal', 'J+1|AB'), ('stress', 'J+1|AB_stress')):
        a = RJ1['mean'][c]; VAL[f'J1_{c}_maxrel'] = float((a / NOOS[ref].reindex(a.index) - 1).abs().max()); assert VAL[f'J1_{c}_maxrel'] <= 1e-9, VAL
    a = RJ1['ew']; VAL['J1_EW_maxrel'] = float((a / NOOS['J+1|EW'].reindex(a.index) - 1).abs().max()); assert VAL['J1_EW_maxrel'] <= 1e-9
log('J+1 done', VAL)
J1DAYS = {m: d for m, d in JD}; OFFD = {pd.Period(d, 'M'): d for d in S0.sig_days}
VAL['signal_day_same_months'] = int(sum(1 for m in OFFD if J1DAYS.get(m) == OFFD[m])); VAL['n_months'] = len(OFFD)
VAL['J1_one_day_later_months'] = int(sum(1 for m in OFFD if m in J1DAYS and F.TCAL.get_loc(J1DAYS[m]) - F.TCAL.get_loc(OFFD[m]) == 1))

# ---------------- 袖子混合（同 integrate_o4）----------------
EXI = set(i - t0 for i in EXEC if t0 <= i <= t1)
R_E50 = K.E50.reindex(IX).pct_change().fillna(0).to_numpy()
J2330 = K.CI['2330']
R_2330 = (K.GROW[t0:t1 + 1, J2330] - 1.0).copy(); R_2330[0] = 0.0
def sleeve(navA, rB, w, cost, kindB, ix=IX, exi=EXI):
    """navA：主策略腿淨值（已扣自身成本）；rB：另一腿日報酬；w：主策略腿固定比例；cost＝(買, 賣) 股票費率。
    kindB：'etf'（賣方＝買方＋0.1%）、'stock'（同股票費率）。AB_V4 成交日 e 收盤調回比例，移轉金額扣成本。"""
    buy, sell = cost
    bB, sB = {'etf': (buy, buy + 0.001), 'stock': (buy, sell)}[kindB]
    ra = navA.reindex(ix).pct_change().fillna(0).to_numpy()
    a, b = w, 1 - w; out = np.empty(len(ix))
    for i in range(len(ix)):
        if i > 0: a *= 1 + ra[i]; b *= 1 + rB[i]
        if i in exi and i > 0:
            tot = a + b; ta = w * tot
            if a < ta: m = ta - a; c = m * (sB + buy)
            else: m = a - ta; c = m * (sell + bB)
            tot -= c; a, b = w * tot, (1 - w) * tot
        out[i] = a + b
    return pd.Series(out, index=ix)
def rets(nav): return nav.reindex(IX).pct_change().fillna(0).to_numpy()

MIXW = {'M_J1_20': 0.8, 'M_J1_50': 0.5}
def build(name, c):
    cr = L.cost_rates(c if c in ('normal', 'stress') else c)
    if name == 'AB_V4': return ABV[str(c)]
    if name == 'R_J1': return J1V[str(c)] / J1V[str(c)].iloc[0]
    if name in MIXW: return sleeve(ABV[str(c)], rets(J1V[str(c)]), MIXW[name], cr, 'stock')
    if name == 'REF_E50_30': return sleeve(ABV[str(c)], R_E50, 0.7, cr, 'etf')
    if name == 'REF_2330_30': return sleeve(ABV[str(c)], R_2330, 0.7, cr, 'stock')
    raise ValueError(name)
ORDER = ['AB_V4', 'R_J1', 'M_J1_20', 'M_J1_50', 'REF_E50_30', 'REF_2330_30']
CAND = ['R_J1', 'M_J1_20', 'M_J1_50']
NAV = {f'{n}|{c}': build(n, c) for n in ORDER for c in ('normal', 'stress')}
SLIPNAV = {(n, s): build(n, ('slip', s)) for n in ORDER for s in SLIPS}
if not SMOKE:
    for n, ref in (('REF_E50_30', 'L_E50_30'), ('REF_2330_30', 'L_2330_30')):
        for c in ('normal', 'stress'):
            a = NAV[f'{n}|{c}']; VAL[f'{n}_{c}_vs_integrate_o4_maxrel'] = float((a / NI4[f'{ref}|{c}'].reindex(a.index) - 1).abs().max())
            assert VAL[f'{n}_{c}_vs_integrate_o4_maxrel'] <= 1e-9, VAL
# 逐 seed（seed s 的 AB_V4 腿配 seed s 的 J+1 腿）
SEEDNAV = {}
for k, s in enumerate(SEEDS):
    a, j = abseed['normal'][k], J1seed['normal'][k]
    SEEDNAV[('AB_V4', s)] = a; SEEDNAV[('R_J1', s)] = j / j.iloc[0]
    for n, w in MIXW.items(): SEEDNAV[(n, s)] = sleeve(a, rets(j), w, L.cost_rates('normal'), 'stock')
log('combos built', VAL)

# ---------------- 統計 ----------------
E50 = K.E50
PER = L.PER
PK = ('S1', 'S2', 'S3', 'SEL', 'VAL', 'HOLD', 'ALL')
def wsel(s, a, b): return s[(s.index >= pd.Timestamp(a)) & (s.index <= pd.Timestamp(b))].dropna()
def xs(nav, a=A_, b=B_): return L.excess(nav, E50.reindex(nav.index), sd, a, b)
def tstat(x): return F.tstat(x)
def pd_stats(d):
    d = d.dropna(); nz = d[d.abs() > 1e-12]
    o = dict(mean_pp=float(d.mean() * 100), t=tstat(d), n=int(len(d)), n_nonzero=int(len(nz)),
             MDE80_pp=float(2.8 * d.std(ddof=1) / math.sqrt(len(d)) * 100) if len(d) > 2 else None)
    if len(nz) >= 3:
        big = d.abs().idxmax(); o['drop_maxabs_t'] = tstat(d.drop(big)); o['maxabs_month'] = str(big)[:10]; o['maxabs_pp'] = float(d[big] * 100)
    return o
def paired_win(a, b, x0, x1): return pd_stats(wsel(L.win_ret(a, sd) - L.win_ret(b.reindex(a.index), sd), x0, x1))
FK = {'S1': 'S1', 'S2': 'S2', 'S3': 'S3', 'SEL': 'SEL', 'VAL': 'VAL', 'HOLD': 'HOLD', 'ALL': 'ALL'}
def paired_cal(a, b, key):
    d = F.selp((F.cal_ret(a) - F.cal_ret(b.reindex(a.index))).dropna(), FK[key]); return pd_stats(d)
def corr(a, b, x0, x1, excess=False):
    ra, rb = wsel(L.win_ret(a, sd), x0, x1), wsel(L.win_ret(b.reindex(a.index), sd), x0, x1)
    if excess:
        e = wsel(L.win_ret(E50.reindex(a.index), sd), x0, x1); ra, rb = ra - e, rb - e
    z = pd.concat([ra, rb], axis=1).dropna(); return float(z.iloc[:, 0].corr(z.iloc[:, 1]))
def corr_cal(a, b, key):
    z = pd.concat([F.selp(F.cal_ret(a), FK[key]), F.selp(F.cal_ret(b.reindex(a.index)), FK[key])], axis=1).dropna(); return float(z.iloc[:, 0].corr(z.iloc[:, 1]))
def cal_t(nav, key='ALL'):
    c = F.selp(F.cal_ex(nav), FK[key]); return tstat(c), int(len(c))
def endpoint(nav):
    x = xs(nav); o = {'t_all': tstat(x), 'drop_last1': tstat(x.iloc[:-1])}
    xsrt = x.sort_values(ascending=False)
    for k in (1, 3, 5, 10): o[f'drop_best{k}'] = tstat(x.drop(xsrt.index[:k]))
    yr = x.groupby(x.index.year).sum(); by = int(yr.idxmax())
    o['best_year'] = by; o['drop_best_year'] = tstat(x[x.index.year != by])
    o['drop_first_year_2006'] = tstat(x[x.index.year != 2006])
    o['start_2007'] = tstat(x[x.index >= pd.Timestamp('2007-01-01')]); o['start_2008'] = tstat(x[x.index >= pd.Timestamp('2008-01-01')])
    o['to_2025_end'] = tstat(x[x.index <= pd.Timestamp('2025-12-31')])
    c = F.selp(F.cal_ex(nav), 'ALL'); o['cal_month_t'] = tstat(c); o['cal_drop_best1'] = tstat(c.drop(c.idxmax()))
    return o
def yearly(nav):
    r = nav.pct_change(); r50 = E50.reindex(nav.index).pct_change()
    msk = r.index > nav.index[0]
    return {int(y): {'s': float((1 + g.fillna(0)).prod() - 1) * 100, '0050': float((1 + r50[g.index].fillna(0)).prod() - 1) * 100}
            for y, g in r[msk].groupby(r[msk].index.year)}
def breakeven(nm):
    pts = [(s, L.seg(SLIPNAV[(nm, s)], EWB, sd, A_, B_)['t_0050']) for s in SLIPS]
    def cross(th):
        for (s1, a), (s2, b) in zip(pts, pts[1:]):
            if a >= th > b: return s1 + (s2 - s1) * (a - th) / (a - b)
        return '<0.10%' if pts[0][1] < th else '>0.40%'
    return {'slip_at_t2.3': cross(2.3), 'slip_at_t2.0': cross(2.0), 't_at_slip': {str(s): t for s, t in pts}}
def seed_dist(nm):
    ts, tc, th = [], [], []
    for s in SEEDS:
        n_ = SEEDNAV[(nm, s)]
        ts.append(L.seg(n_, EWB, sd, A_, B_)['t_0050']); tc.append(cal_t(n_)[0]); th.append(L.seg(n_, EWB, sd, *PER['HOLD'])['t_0050'])
    ts, tc, th = map(np.array, (ts, tc, th))
    return {'t_median': float(np.median(ts)), 't_min': float(ts.min()), 't_max': float(ts.max()), 't_p10': float(np.percentile(ts, 10)),
            'n_below_2.3': int((ts < 2.3).sum()), 'n_below_2.0': int((ts < 2.0).sum()), 'cal_t_median': float(np.median(tc)),
            'cal_t_min': float(tc.min()), 'HOLD_t_median': float(np.median(th)), 'n': len(ts)}

B10 = L.seg(NAV4['AB_V4_B10|normal'].dropna(), EWB, sd, A_, B_)['CAGR']
STATS = {}
for nm in ORDER:
    nav = NAV[f'{nm}|normal']; o = {'periods': {}}
    for p in PK:
        a, b = PER[p]; s_ = L.seg(nav, EWB, sd, a, b)
        x = xs(nav, a, b); s_['TE'] = float(x.std(ddof=1) * math.sqrt(12) * 100)
        s_['t_cal'], s_['n_cal'] = cal_t(nav, p)
        if nm != 'AB_V4':
            ab = NAV['AB_V4|normal']
            s_['vs_AB_V4_cal'] = paired_cal(nav, ab, p); s_['vs_AB_V4_win'] = paired_win(nav, ab, a, b)
            s_['corr_AB_V4_win'] = corr(nav, ab, a, b); s_['corr_AB_V4_excess'] = corr(nav, ab, a, b, True); s_['corr_AB_V4_cal'] = corr_cal(nav, ab, p)
        st_ = L.seg(NAV[f'{nm}|stress'], EWB, sd, a, b); s_['stress_CAGR'] = st_['CAGR']; s_['stress_t_0050'] = st_['t_0050']
        o['periods'][p] = s_
    o['endpoint'] = endpoint(nav); o['yearly'] = yearly(nav); o['breakeven'] = breakeven(nm)
    if nm in ('AB_V4',) + tuple(CAND): o['seed_dist'] = seed_dist(nm)
    al = o['periods']['ALL']
    o['live_expectation'] = al['E50_CAGR'] + 0.5 * (al['CAGR'] - al['E50_CAGR'])
    o['live_expectation_lower_edge'] = o['live_expectation'] - 0.5 * (STATS['AB_V4']['periods']['ALL']['CAGR'] - B10 if nm != 'AB_V4' else al['CAGR'] - B10)
    o['years_lose_0050'] = int(sum(1 for y, v in o['yearly'].items() if v['s'] < v['0050']))
    STATS[nm] = o
    log('stats', nm)
STATS['_AB_V4_B10_CAGR'] = B10

# ---------------- SPA（o_main_spa_breaks 的 F_long × W_long，損失＝對 0050 窗超額）----------------
SPR = json.loads((AG / 'o_main_spa_breaks/results.json').read_text(encoding='utf-8'))
SPP = SPR['PREREG']; MAIN_ID = SPP['main_id']
D = pd.read_parquet(DATA / 'family_excess.parquet'); MM = pd.read_csv(DATA / 'family_members.csv')
mem = MM[MM['cov_W_long'] & MM['F_long']].member_id.tolist()
m = C.wmask(D.index, *C.WINDOWS['W_long']); n = int(m.sum())
X0 = D.loc[m, mem].fillna(0.0).to_numpy()
e50w = C.win_ret(C.e50_nav())
def cand_ex(nav): return (C.win_ret(nav) - e50w).reindex(D.index[m]).to_numpy()
APP = {nm: cand_ex(NAV[f'{nm}|normal']) for nm in CAND}
for nm, v in APP.items(): assert np.isfinite(v).all(), nm
VAL['AB_V4_excess_vs_family_maxabs'] = float(np.abs(cand_ex(NAV['AB_V4|normal']) - X0[:, mem.index(MAIN_ID)]).max())
REFID = {'AB_V4': MAIN_ID, 'REF_E50_30': 'integrate_o4/navs.parquet::L_E50_30|normal', 'REF_2330_30': 'integrate_o4/navs.parquet::L_2330_30|normal'}
for k, mid in REFID.items():
    if mid in mem: VAL[f'{k}_excess_vs_family_maxabs'] = float(np.abs(cand_ex(NAV[f'{k}|normal']) - X0[:, mem.index(mid)]).max())
B = 200 if SMOKE else SPP['spa']['B']
SPA = {}
def grade(p): return '第 2 級（通過資料窺探校正）' if p <= 0.05 else ('第 2 級（邊緣）' if p <= 0.10 else '第 3 級（有理論支持，但未通過資料窺探校正）')
for Lb in SPP['spa']['block_lengths']:
    Wc = SP.sb_counts(n, Lb, B, SPP['spa']['seed_base'] + 1000 * 0 + Lb)
    o0 = SP.mcp_tests(X0, Wc, Lb, SPP['spa']['variance'], focus=mem.index(MAIN_ID))
    ref = SPR['RESULTS_SPA']['grid'][f'F_long|W_long|L{Lb}|x0050']
    if not SMOKE:
        VAL[f'SPA_repro_L{Lb}'] = {'p_SPA_c_focus': o0['p_SPA_c_focus'], 'ref': ref['p_SPA_c_focus']}
        assert abs(o0['p_SPA_c_focus'] - ref['p_SPA_c_focus']) < 1e-12 and o0['focus_rank'] == ref['focus_rank'], VAL
    names = list(mem) + [f'integrate_o5::{k}' for k in CAND]
    X = np.hstack([X0] + [APP[k][:, None] for k in CAND])
    Mb = SP.boot_means(Wc, X)
    for k in ORDER:
        fid = REFID.get(k, f'integrate_o5::{k}')
        if fid not in names: continue
        o = SP.mcp_tests(X, Wc, Lb, SPP['spa']['variance'], Mb=Mb, focus=names.index(fid))
        SPA[f'{k}|L{Lb}'] = {kk: o[kk] for kk in ('N', 'max_t', 'p_SPA_c', 'p_SPA_c_focus', 'p_SPA_l_focus', 'p_SPA_u_focus', 'p_RC_focus', 'focus_rank', 'focus_t',
                                                   'p_StepM_focus', 'p_single_focus')}
        SPA[f'{k}|L{Lb}']['grade'] = grade(o['p_SPA_c_focus'])
    del Mb
    log('SPA', Lb)

# ---------------- 選版（只用 SEL）與判定 ----------------
PR = None if SMOKE else RES['PREREG']
ab = STATS['AB_V4']['periods']
sel = []
for nm in CAND:
    s_ = STATS[nm]['periods']['SEL']
    ok = (s_['t_0050'] > ab['SEL']['t_0050']) and (ab['SEL']['CAGR'] - s_['CAGR'] <= 1.0)
    sel.append({'name': nm, 'SEL_t': s_['t_0050'], 'SEL_CAGR_loss_pp': ab['SEL']['CAGR'] - s_['CAGR'], 'eligible': bool(ok)})
elig = [x for x in sel if x['eligible']]
SELECTED = max(elig, key=lambda x: x['SEL_t'])['name'] if elig else 'AB_V4'
JUDGE = {}
for nm in CAND:
    P = STATS[nm]['periods']
    pa = P['ALL']['vs_AB_V4_cal']['t']; ms, mv = P['SEL']['vs_AB_V4_cal']['mean_pp'], P['VAL']['vs_AB_V4_cal']['mean_pp']
    A = pa >= 1.0 and ms > 0 and mv > 0
    Bp = (P['ALL']['t_0050'] - ab['ALL']['t_0050'] >= 0.2 and P['SEL']['t_0050'] > ab['SEL']['t_0050'] and P['VAL']['t_0050'] > ab['VAL']['t_0050']
          and ab['ALL']['CAGR'] - P['ALL']['CAGR'] <= 1.0 and P['ALL']['maxDD'] >= ab['ALL']['maxDD'])
    W = pa > -1.0 and ab['ALL']['CAGR'] - P['ALL']['CAGR'] <= 2.0 and (P['SEL']['t_0050'] > ab['SEL']['t_0050'] or P['VAL']['t_0050'] > ab['VAL']['t_0050'])
    raw = '可用' if (A or Bp) else ('待觀察' if W else '不可用')
    JUDGE[nm] = {'path_A': bool(A), 'path_B': bool(Bp), 'watch_cond': bool(W), 'raw': raw, 'final': '待觀察' if raw == '可用' else raw,
                 'SPA_L6_p_c_focus': SPA[f'{nm}|L6']['p_SPA_c_focus'], 'SPA_grade': SPA[f'{nm}|L6']['grade']}
OUT = {'validation': VAL, 'stats': STATS, 'spa': SPA, 'selection': {'table': sel, 'selected': SELECTED}, 'judgement': JUDGE,
       'run_seconds': time.time() - T0_}
def js(o):
    if isinstance(o, dict): return {str(k): js(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [js(v) for v in o]
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    return o
if SMOKE:
    Path(sys.argv[sys.argv.index('--smoke') + 1]).write_text(json.dumps(js(OUT), ensure_ascii=False))
    print('smoke ok', flush=True)
else:
    pd.DataFrame({k: v for k, v in NAV.items()}).to_parquet(HERE / 'navs.parquet')
    RES['RESULTS'] = js(OUT); RES['RESULTS_written_at'] = pd.Timestamp.now(tz='UTC').isoformat()
    RJ.write_text(json.dumps(RES, ensure_ascii=False, indent=1), encoding='utf-8')
    log('done selected', SELECTED, {k: v['final'] for k, v in JUDGE.items()})
