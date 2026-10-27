#!/usr/bin/env python3
"""樣本外：斷言 results.json 的 PREREG 存在且未被改過、RESULTS 尚未寫過；凍結版與全部對照組跑到 2026-10-12，
一般／壓力成本；判定門檻機械套用；穩健性與可行性。寫回 results.json 的 RESULTS（只寫一次）與 navs.parquet。
環境變數不使用；資料版本＝研究中心 V2 面板＋R_DEL（V3 重跑見 oos_v3.py）。"""
import sys, json, time, hashlib, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
RJ = HERE / 'results.json'
R = json.loads(RJ.read_text(encoding='utf-8'))
assert 'PREREG' in R and hashlib.sha256(json.dumps(R['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == R['PREREG_sha256'], 'PREREG 不存在或被改過'
assert 'RESULTS' not in R, 'RESULTS 已存在：只跑一次'
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()
assert sha(HERE / 'iterations.json') == R['PREREG']['iterations_sha256']
assert sha(HERE / 'tw50_members.json') == R['PREREG']['tw50_members_sha256']
LC_SHA_AT_FREEZE = R['PREREG']['lcore_sha256']
sys.path.insert(0, str(HERE))
import lcore as L
sys.path.pop(0)
import numpy as np, pandas as pd
K, V, O = L.K, L.V, L.O
T0_ = time.time()
def log(*a): print(f'[{time.time()-T0_:5.0f}s]', *a, flush=True)
lcore_changed = sha(HERE / 'lcore.py') != LC_SHA_AT_FREEZE
# PREREG 的 frozen_spec['score'] 在 freeze.py 被文字說明覆蓋（鍵名重複，文字＝0.5/0.5），代碼值改從 sha256 已鎖定的 iterations.json 讀
_IT0 = json.loads((HERE / 'iterations.json').read_text(encoding='utf-8'))
SPEC = _IT0['versions'][R['PREREG']['selection']['picked']]['spec']
assert SPEC['P'] == R['PREREG']['frozen_spec']['P'] and SPEC['n'] == R['PREREG']['frozen_spec']['n'] and SPEC['w_leg'] == R['PREREG']['frozen_spec']['w_leg']
assert R['PREREG']['frozen_spec']['score'].startswith('池內百分位：0.5×52') and SPEC['score'] == '55'
P_, N_, SC_, WM_, WL_ = SPEC['P'], SPEC['n'], SPEC['score'], SPEC['weight'], SPEC['w_leg']
t1 = L.T1; SD = L.SD
KEYS = ('SEL', 'VAL', 'VAL_A', 'HOLD', 'S1', 'S2', 'ALL')
E = L.wr(K.E50); Z = L.wr(O.TSMC_IDX) - E
mp = L.MAIN_PLAN
COSTS = {'normal': L.NORMAL, 'stress3': L.STRESS3, 'stress_old': L.STRESS_OLD}

# ---------------- 計畫 ----------------
leg = L.plan_leg(P_, N_, SC_, WM_)
leg1 = L.plan_leg(P_, 1, 'v60', 'ew')
legX = L.plan_leg(P_, N_, SC_, WM_, excl=('2330',))
ewMain, _ = K.simulate([(p[0], p[2]) for p in mp], 0, 0, L.T0, t1)
ewPool = L.pool_ew(leg, t1); ewPoolX = L.pool_ew(legX, t1)
WP = {'MAIN': L.wplan_main(mp), 'TSMC30': O.tsmc_plan(mp, 0.3), 'COMBO': L.wplan_mix(mp, leg, WL_), 'MAIN70_TOP1VALUE30': L.wplan_mix(mp, leg1, 0.3),
      'MAIN70_POOLEW30': L.wplan_pool(leg, 0.3, mp), 'L_ALONE': L.wplan_leg(leg), 'COMBO_EX2330': L.wplan_mix(mp, legX, WL_), 'L_ALONE_EX2330': L.wplan_leg(legX)}
NAV = {}; TOV = {}
for k, wp in WP.items():
    for c, cost in COSTS.items():
        NAV[(k, c)], tv = L.sim(wp, cost, t1)
        if c == 'normal': TOV[k] = round(tv, 4)
    log('sim', k, round(float(NAV[(k, 'normal')].iloc[-1]), 2))
for c, cost in COSTS.items():
    NAV[('MAIN70_0050_30', c)] = L.sleeve_mix(NAV[('MAIN', c)], 0.3, t1, cost)
# 驗證：MAIN、TSMC30 與 integrate_o2 逐日相同
ref = pd.read_parquet(L.AG / 'integrate_o2' / 'navs.parquet')
val = {}
for a, b in (('MAIN', 'MAIN_R'), ('TSMC30', 'TSMC30_R')):
    x = NAV[(a, 'normal')]; r = ref[b].reindex(x.index)
    val[f'{a}_vs_integrate_o2_{b}_maxreldiff'] = float(((x / x.iloc[0]) / (r / r.iloc[0]) - 1).abs().max())
    assert val[f'{a}_vs_integrate_o2_{b}_maxreldiff'] < 1e-9, val
log('validated', val)

# ---------------- 統計 ----------------
W = {k: L.wr(v) for k, v in NAV.items()}
WEP, WEPX = L.wr(ewPool), L.wr(ewPoolX)
xM = W[('MAIN', 'normal')] - E
def table(k, c='normal'):
    nav = NAV[(k, c)]; out = {}
    for p in KEYS:
        st = L.segstat(nav, ewMain, p)
        xc = L.win(W[(k, c)] - E, p); xm = L.win(W[('MAIN', c)] - E, p)
        st['decomp_vs_MAIN'] = L.decomp(xc, xm)
        st['paired_vs_MAIN'] = L.mstat(L.win(W[(k, c)] - W[('MAIN', c)], p))
        st['corr_ex0050_with_MAIN'] = L.corr(xc, xm)
        st['beta_ex0050_on_2330m0050'] = L.beta(xc, L.win(Z, p))
        out[p] = st
    return out
NAMES = ['MAIN', 'TSMC30', 'COMBO', 'MAIN70_0050_30', 'MAIN70_POOLEW30', 'MAIN70_TOP1VALUE30', 'L_ALONE', 'COMBO_EX2330', 'L_ALONE_EX2330']
RES = {'validation': val, 'lcore_changed_after_freeze': lcore_changed, 'periods': L.PERIODS, 'turnover': TOV, 'combos': {}}
for k in NAMES:
    RES['combos'][k] = {c: table(k, c) for c in COSTS}
    s = RES['combos'][k]['normal']
    log(k, {p: (s[p]['CAGR'], s[p]['t_0050'], s[p]['TE_ann'], s[p]['mean_ex_0050_m']) for p in ('SEL', 'VAL', 'VAL_A', 'HOLD', 'ALL')})
# L 腿
def legstats(navkey, ewp, pre=''):
    o = {}
    for p in KEYS:
        o[p] = {}
        for c in COSTS:
            o[p][f'L_vs_poolEW_{c}'] = L.mstat(L.win(W[(navkey, c)] - ewp, p))
        xl = L.win(W[(navkey, 'normal')] - E, p)
        o[p]['L_vs_0050'] = L.mstat(xl); o[p]['poolEW_vs_0050'] = L.mstat(L.win(ewp - E, p))
        o[p]['corr_Lex0050_MAINex0050'] = L.corr(xl, L.win(xM, p))
        o[p]['corr_LexPool_MAINex0050'] = L.corr(L.win(W[(navkey, 'normal')] - ewp, p), L.win(xM, p))
        o[p]['beta_Lex0050_on_2330m0050'] = L.beta(xl, L.win(Z, p))
        o[p]['beta_MAINex0050_on_2330m0050'] = L.beta(L.win(xM, p), L.win(Z, p))
        o[p]['beta_LexPool_on_2330m0050'] = L.beta(L.win(W[(navkey, 'normal')] - ewp, p), L.win(Z, p))
    return o
RES['leg'] = legstats('L_ALONE', WEP); RES['leg_ex2330'] = legstats('L_ALONE_EX2330', WEPX)
RES['leg_info'] = {'share_months_2330': round(float(np.mean([L.J2330 in set(p[1].tolist()) for p in leg])), 3),
                   'share_months_2330_VAL': round(float(np.mean([L.J2330 in set(p[1].tolist()) for p in leg if p[3] >= L.SPLIT])), 3),
                   'top1_is_2330_share_VAL': round(float(np.mean([p[1][0] == L.J2330 for p in leg1 if p[3] >= L.SPLIT])), 3)}
log('leg', {p: RES['leg'][p]['L_vs_poolEW_normal'] for p in ('SEL', 'VAL', 'HOLD')})

# ---------------- 2330 漂移歸零反事實 ----------------
G2, dinfo = L.drift_zero_G(L.T_SPLIT, t1)
CF = {'drift_bp_per_day': dinfo}
for k in ('MAIN', 'TSMC30', 'COMBO', 'MAIN70_TOP1VALUE30'):
    nv, _ = L.sim(WP[k], L.NORMAL, t1, G=G2)
    CF[k] = {p: {kk: v for kk, v in L.segstat(nv, ewMain, p).items() if kk in ('CAGR', 't_0050', 'TE_ann', 'mean_ex_0050_m', 'sd_ex_0050_m')} for p in KEYS}
RES['drift_zero_cf'] = CF
log('cf', {k: (CF[k]['SEL']['t_0050'], CF[k]['VAL']['t_0050']) for k in ('MAIN', 'TSMC30', 'COMBO')})

# ---------------- 剔除貢獻最大 5 檔（VAL） ----------------
per = L.leg_period_returns(leg, t1)
pv = [x for x in per if x[0] >= L.SPLIT]
con = {}
for d, pk, w, r, pool, rp in pv:
    ew = rp.mean()
    for j, wj, rj in zip(pk, w, r): con[L.COLS[j]] = con.get(L.COLS[j], 0.0) + wj * (rj - ew)
top5 = sorted(con, key=con.get, reverse=True)[:5]
g_before = np.array([float((w * (r - rp.mean())).sum()) for d, pk, w, r, pool, rp in pv])
g_after = np.array([float(sum(wj * ((rj if L.COLS[j] not in top5 else rp.mean()) - rp.mean()) for j, wj, rj in zip(pk, w, r))) for d, pk, w, r, pool, rp in pv])
net_val = L.win(W[('L_ALONE', 'normal')] - WEP, 'VAL')
drag = float(g_before.mean() - net_val.mean())
RES['drop_top5_VAL'] = {'top5': [(c, V.SL['name'].get(c, ''), round(con[c] * 100, 2)) for c in top5], 'sum_contrib_pp_all': round(sum(con.values()) * 100, 2),
                        'gross_mean_before_pp': round(g_before.mean() * 100, 3), 'gross_mean_after_pp': round(g_after.mean() * 100, 3),
                        'cost_drag_pp': round(drag * 100, 3), 'net_after_pp': round((g_after.mean() - drag) * 100, 3),
                        't_gross_after': round(L.tstat(g_after), 3), 'n_periods': int(len(pv))}
# 也做 SEL 的同一拆解（描述）
ps = [x for x in per if x[0] < L.SPLIT]
cs_ = {}
for d, pk, w, r, pool, rp in ps:
    for j, wj, rj in zip(pk, w, r): cs_[L.COLS[j]] = cs_.get(L.COLS[j], 0.0) + wj * (rj - rp.mean())
RES['top_contrib_SEL'] = [(c, V.SL['name'].get(c, ''), round(cs_[c] * 100, 2)) for c in sorted(cs_, key=cs_.get, reverse=True)[:8]]
RES['top_contrib_VAL'] = [(c, V.SL['name'].get(c, ''), round(con[c] * 100, 2)) for c in sorted(con, key=con.get, reverse=True)[:8]]
RES['worst_contrib_VAL'] = [(c, V.SL['name'].get(c, ''), round(con[c] * 100, 2)) for c in sorted(con, key=con.get)[:5]]

# ---------------- 判定 ----------------
C = RES['combos']; cN = lambda k, p: C[k]['normal'][p]
lg = RES['leg']; lgX = RES['leg_ex2330']
c1 = {'SEL_t': lg['SEL']['L_vs_poolEW_normal']['t'], 'VAL_mean_pp': lg['VAL']['L_vs_poolEW_normal']['mean_pp']}
c1['pass'] = c1['SEL_t'] >= 1.5 and c1['VAL_mean_pp'] > 0
c2 = {'VAL_paired_mean_pp': cN('COMBO', 'VAL')['paired_vs_MAIN']['mean_pp'], 'HOLD_CAGR': cN('COMBO', 'HOLD')['CAGR'], 'HOLD_CAGR_MAIN': cN('MAIN', 'HOLD')['CAGR'],
      'TE_VAL_A': cN('COMBO', 'VAL_A')['TE_ann'], 'TE_VAL_A_MAIN': cN('MAIN', 'VAL_A')['TE_ann'], 'TE_HOLD': cN('COMBO', 'HOLD')['TE_ann'], 'TE_HOLD_MAIN': cN('MAIN', 'HOLD')['TE_ann']}
c2['pass'] = (c2['VAL_paired_mean_pp'] >= -0.1 and c2['HOLD_CAGR'] >= c2['HOLD_CAGR_MAIN'] and c2['TE_VAL_A'] <= c2['TE_VAL_A_MAIN'] - 3 and c2['TE_HOLD'] <= c2['TE_HOLD_MAIN'] - 3)
c3 = {'SEL_t_COMBO': cN('COMBO', 'SEL')['t_0050'], 'SEL_t_TSMC30': cN('TSMC30', 'SEL')['t_0050'],
      'CF_SEL_t_COMBO': CF['COMBO']['SEL']['t_0050'], 'CF_SEL_t_TSMC30': CF['TSMC30']['SEL']['t_0050'],
      'CF_VAL_t_COMBO': CF['COMBO']['VAL']['t_0050'], 'CF_VAL_t_TSMC30': CF['TSMC30']['VAL']['t_0050'],
      'EX2330_SEL_t': lgX['SEL']['L_vs_poolEW_normal']['t'], 'EX2330_VAL_mean_pp': lgX['VAL']['L_vs_poolEW_normal']['mean_pp']}
c3['pass'] = (c3['SEL_t_COMBO'] >= c3['SEL_t_TSMC30'] and c3['CF_SEL_t_COMBO'] > c3['CF_SEL_t_TSMC30'] and c3['CF_VAL_t_COMBO'] > c3['CF_VAL_t_TSMC30']
              and c3['EX2330_SEL_t'] >= 1.5 and c3['EX2330_VAL_mean_pp'] > 0)
c4 = {'net_after_pp': RES['drop_top5_VAL']['net_after_pp']}; c4['pass'] = c4['net_after_pp'] > 0
P1_fail = c1['SEL_t'] < 1.5 or lg['SEL']['L_vs_poolEW_normal']['mean_pp'] <= 0
ex_fail = lgX['SEL']['L_vs_poolEW_normal']['mean_pp'] <= 0 or lgX['VAL']['L_vs_poolEW_normal']['mean_pp'] <= 0
if c1['pass'] and c2['pass'] and c3['pass'] and c4['pass']: verdict, why = '待觀察', '全部通過（上限待觀察）'
elif P1_fail or ex_fail: verdict, why = '不可用', 'P1 不成立或池剔除 2330 後超額 ≤ 0：L 腿只能當降低追蹤誤差的配置，和 0050 沒有差別' + ('；且 (3) 不過 → 維持 TSMC30_R 現狀' if not c3['pass'] else '')
elif not c3['pass']: verdict, why = '不可用', '(3) 不過 → 維持 TSMC30_R 現狀'
else: verdict, why = '不可用', '未全部通過'
pred = {'P1': {'SEL_mean_pp': lg['SEL']['L_vs_poolEW_normal']['mean_pp'], 'SEL_t': c1['SEL_t'], 'holds': not P1_fail},
        'P2': {'corr_SEL': lg['SEL']['corr_Lex0050_MAINex0050'], 'corr_VAL': lg['VAL']['corr_Lex0050_MAINex0050'],
               'beta_L_SEL': lg['SEL']['beta_Lex0050_on_2330m0050']['beta'], 'beta_L_VAL': lg['VAL']['beta_Lex0050_on_2330m0050']['beta'],
               'beta_MAIN_SEL': lg['SEL']['beta_MAINex0050_on_2330m0050']['beta'], 'beta_MAIN_VAL': lg['VAL']['beta_MAINex0050_on_2330m0050']['beta']},
        'P3': {p: {'TE_combo_minus_main': round(cN('COMBO', p)['TE_ann'] - cN('MAIN', p)['TE_ann'], 2), 'mean_ex_combo_minus_main_pp': round(cN('COMBO', p)['mean_ex_0050_m'] - cN('MAIN', p)['mean_ex_0050_m'], 3),
                   'decomp': cN('COMBO', p)['decomp_vs_MAIN']} for p in ('SEL', 'VAL', 'VAL_A', 'HOLD', 'ALL')},
        'P4': {'ex2330_SEL': lgX['SEL']['L_vs_poolEW_normal'], 'ex2330_VAL': lgX['VAL']['L_vs_poolEW_normal'], 'cf': {k: {p: CF[k][p]['t_0050'] for p in ('SEL', 'VAL', 'ALL')} for k in ('COMBO', 'TSMC30')}}}
pred['P2']['holds'] = (pred['P2']['corr_SEL'] <= 0.5 and pred['P2']['corr_VAL'] <= 0.5 and pred['P2']['beta_L_SEL'] >= 0 and pred['P2']['beta_L_VAL'] >= 0)
pred['P3']['holds'] = all(pred['P3'][p]['TE_combo_minus_main'] <= -3 and pred['P3'][p]['mean_ex_combo_minus_main_pp'] >= -0.15 for p in ('SEL', 'VAL'))
pred['P4']['holds'] = (not ex_fail) and all(CF['COMBO'][p]['t_0050'] > CF['TSMC30'][p]['t_0050'] for p in ('SEL', 'VAL'))
RES['judgement'] = {'(1)': c1, '(2)': c2, '(3)': c3, '(4)': c4, 'P1_fail': P1_fail, 'ex2330_fail': ex_fail, 'verdict': verdict, 'why': why, 'predictions': pred}
log('verdict', verdict, why, c1, c2['pass'], c3['pass'], c4)

# ---------------- 穩健性 ----------------
rob = {}
xLP = W[('L_ALONE', 'normal')] - WEP
rob['L_vs_pool_by_year_sum_pp'] = {int(y): round(float(g.sum()) * 100, 2) for y, g in xLP.dropna().groupby(xLP.dropna().index.year)}
xCM = W[('COMBO', 'normal')] - W[('MAIN', 'normal')]
rob['COMBO_minus_MAIN_by_year_sum_pp'] = {int(y): round(float(g.sum()) * 100, 2) for y, g in xCM.dropna().groupby(xCM.dropna().index.year)}
def yearly(nav):
    n = nav.dropna(); g = n.groupby(n.index.year).last(); prev = g.shift(1); prev.iloc[0] = n.iloc[0]; return (g / prev - 1) * 100
rob['yearly_pct'] = {k: {int(i): round(float(v), 1) for i, v in yearly(NAV[(k, 'normal')]).items()} for k in ('MAIN', 'TSMC30', 'COMBO', 'MAIN70_0050_30', 'L_ALONE')}
rob['yearly_pct']['0050'] = {int(i): round(float(v), 1) for i, v in yearly(K.E50.reindex(NAV[('MAIN', 'normal')].index)).items()}
rob['yearly_pct']['POOL_EW'] = {int(i): round(float(v), 1) for i, v in yearly(ewPool).items()}
# 權值股領漲年份（0050 贏池等權）
yE = rob['yearly_pct']['0050']; yP = rob['yearly_pct']['POOL_EW']
lead = [y for y in yE if y in yP and yE[y] > yP[y]]
for p in ('SEL', 'VAL'):
    x = L.win(xLP, p); xc = L.win(xCM, p)
    rob[f'{p}_capweight_lead_years'] = [y for y in lead if (x.index.year == y).any()]
    m = np.isin(x.index.year, lead); mc = np.isin(xc.index.year, lead)
    rob[f'{p}_L_vs_pool_lead'] = L.mstat(x[m]); rob[f'{p}_L_vs_pool_nonlead'] = L.mstat(x[~m])
    rob[f'{p}_COMBO_minus_MAIN_lead'] = L.mstat(xc[mc]); rob[f'{p}_COMBO_minus_MAIN_nonlead'] = L.mstat(xc[~mc])
# 電子 vs 非電子入選股（期報酬 − 池等權）
for p, sub in (('SEL', ps), ('VAL', pv)):
    el_, ne_ = [], []
    for d, pk, w, r, pool, rp in sub:
        m = L.ELEC[pk]
        if m.any(): el_.append((d, r[m].mean() - rp.mean()))
        if (~m).any(): ne_.append((d, r[~m].mean() - rp.mean()))
    rob[f'{p}_elec_picks_minus_pool'] = L.mstat(pd.Series(dict(el_))); rob[f'{p}_nonelec_picks_minus_pool'] = L.mstat(pd.Series(dict(ne_)))
# 外資買超月份（2021-10 起有 inst_5y；上市股外資買賣超 × 收盤，訊號窗內加總）
inst = pd.read_csv(K.X / 'inst_5y.csv.gz', dtype={'code': str}, usecols=['date', 'code', 'foreign'], parse_dates=['date'])
inst = inst[inst['code'].isin(set(L.COLS))]
fp = inst.pivot_table(index='date', columns='code', values='foreign', aggfunc='sum').reindex(columns=L.COLS)
fnt = (fp.reindex(L.IDX) * 1000 * K.C).sum(axis=1, min_count=50)
sdv = [d for d in SD if d >= pd.Timestamp('2021-10-11')]
fw = pd.Series({a: float(fnt[(fnt.index > a) & (fnt.index <= b)].sum()) for a, b in zip(sdv[:-1], sdv[1:])})
fw = fw[fw.index >= pd.Timestamp('2021-11-01')]
xx = xLP.reindex(fw.index); xc = xCM.reindex(fw.index)
rob['foreign_netbuy_months'] = {'n_buy': int((fw > 0).sum()), 'n_sell': int((fw <= 0).sum()), 'L_vs_pool_buy': L.mstat(xx[fw > 0]), 'L_vs_pool_sell': L.mstat(xx[fw <= 0]),
                                'COMBO_minus_MAIN_buy': L.mstat(xc[fw > 0]), 'COMBO_minus_MAIN_sell': L.mstat(xc[fw <= 0]),
                                'note': '同期（訊號窗）外資買賣超，事後分組，只描述；2021-11～2026-09'}
# 輔助：臺灣50 成分池
legT = L.plan_leg(P_, N_, SC_, WM_, members_fn=lambda d: L.tw50_members(d) if d >= L.TW50_START else None)
navLT, _ = L.sim(L.wplan_leg(legT), L.NORMAL, t1); ewT = L.pool_ew(legT, t1)
navCT, _ = L.sim(L.wplan_mix(mp, legT, WL_), L.NORMAL, t1)
xT = L.wr(navLT) - L.wr(ewT); xT = xT[xT.index >= pd.Timestamp('2015-04-01')]
rob['aux_TW50'] = {'pool_size_mean_VAL': round(float(np.mean([len(p[4]) for p in legT if p[3] >= L.SPLIT])), 1),
                   'L_vs_TW50poolEW_2015_2018': L.mstat(xT[xT.index < L.SPLIT]), 'L_vs_TW50poolEW_VAL': L.mstat(L.win(xT, 'VAL')),
                   'L_vs_TW50poolEW_HOLD': L.mstat(L.win(xT, 'HOLD')),
                   'COMBO_TW50': {p: {kk: L.segstat(navCT, ewMain, p)[kk] for kk in ('CAGR', 't_0050', 'TE_ann', 'mean_ex_0050_m')} for p in ('VAL', 'VAL_A', 'HOLD')}}
# 12 版全部在 VAL（揭露，不能據以選版）
IT = json.loads((HERE / 'iterations.json').read_text(encoding='utf-8'))
allv = {}
for k, v in IT['versions'].items():
    s = v['spec']; lp = L.plan_leg(s['P'], s['n'], s['score'], s['weight'])
    nvL, _ = L.sim(L.wplan_leg(lp), L.NORMAL, t1); ewp = L.wr(L.pool_ew(lp, t1))
    nvC, _ = L.sim(L.wplan_mix(mp, lp, s['w_leg']), L.NORMAL, t1)
    allv[k] = {p: {'L_vs_pool': L.mstat(L.win(L.wr(nvL) - ewp, p)), **{kk: L.segstat(nvC, ewMain, p)[kk] for kk in ('CAGR', 't_0050', 'TE_ann', 'mean_ex_0050_m')}} for p in ('SEL', 'VAL', 'HOLD')}
rob['all_versions_VAL_disclosure'] = allv
log('robust done')

# ---------------- 可行性（整張／零股）與配置表 ----------------
feas = {}
recent = [p for p in leg if p[3] >= pd.Timestamp('2024-01-01')]
recM = [p for p in mp if p[3] >= pd.Timestamp('2024-01-01')]
for cap in (300_000, 1_000_000):
    a_leg = cap * WL_ / N_; a_main = cap * (1 - WL_) / 10; a_ts = cap * 0.3
    def stat(pl, alloc, n_only=None):
        lot_ok, resid_odd, resid_lot, px = [], [], [], []
        for p in pl:
            for j in p[1]:
                c = K.Cn[p[0] - 1, j]   # 訊號日收盤
                px.append(c); lot = c * 1000
                lot_ok.append(lot <= alloc)
                sh = math.floor(alloc / c); resid_odd.append((alloc - sh * c) / alloc)
                nl = round(alloc / lot); resid_lot.append(abs(alloc - nl * lot) / alloc)
        return {'alloc_per_stock': round(alloc), 'share_whole_lot_affordable': round(float(np.mean(lot_ok)), 3), 'median_price': round(float(np.median(px)), 1),
                'odd_lot_cash_residual_mean': round(float(np.mean(resid_odd)), 4), 'whole_lot_rounding_err_mean': round(float(np.mean(resid_lot)), 3)}
    j = L.J2330
    tsm_px = [K.Cn[p[0] - 1, j] for p in recM]
    feas[str(cap)] = {'L_leg': stat(recent, a_leg), 'MAIN': stat(recM, a_main),
                      '2330_30pct': {'alloc': round(a_ts), 'lots_affordable_mean': round(float(np.mean([a_ts / (x * 1000) for x in tsm_px])), 3),
                                     'odd_lot_cash_residual_mean': round(float(np.mean([(a_ts - math.floor(a_ts / x) * x) / a_ts for x in tsm_px])), 4)},
                      'n_trades_per_month_L': round(float(np.mean([len(set(a[1].tolist()) ^ set(b[1].tolist())) for a, b in zip(recent[:-1], recent[1:])])), 2)}
RES['feasibility_2024_2026'] = feas
RES['allocation_with_disposal20'] = {
 'note': '處置股策略上限 20%（待觀察）；其餘 80% 依主策略配置選項分配。本題 L 腿未通過，表中 L 腿列只供對照。',
 '現行建議（TSMC30_R 選項）': {'處置股策略': 20, '主策略 n10（排除金融）': 56, '2330': 24},
 '只用主策略': {'處置股策略': 20, '主策略 n10（排除金融）': 80},
 '若採 L 腿（未通過，不建議）': {'處置股策略': 20, '主策略 n10（排除金融）': 56, 'L 腿（大型股前 50 選 5）': 24},
 '0050 降追蹤誤差選項': {'處置股策略': 20, '主策略 n10（排除金融）': 56, '0050': 24}}
RES['n_tests_disclosure'] = '12 個模型版本（SEL 選版一次）＋7 個對照組＋判定 4 條（8 個子條件）＋預測 P1～P4；VAL 12 版全部揭露但不參與判定'
pd.DataFrame({f'{k}|{c}': v for (k, c), v in NAV.items()}).to_parquet(HERE / 'navs.parquet')
R['RESULTS'] = RES
R['RESULTS_written_at'] = pd.Timestamp.now(tz='UTC').isoformat()
RJ.write_text(L.dumps(R), encoding='utf-8')
log('done')
