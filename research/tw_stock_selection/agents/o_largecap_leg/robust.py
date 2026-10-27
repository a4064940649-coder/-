#!/usr/bin/env python3
"""穩健性（oos.py 漏存 rob 字典，這裡用同一段程式重算後另存 results_robust.json；只描述，不參與判定）。原說明：樣本外：斷言 results.json 的 PREREG 存在且未被改過、RESULTS 尚未寫過；凍結版與全部對照組跑到 2026-10-12，
一般／壓力成本；判定門檻機械套用；穩健性與可行性。寫回 results.json 的 RESULTS（只寫一次）與 navs.parquet。
環境變數不使用；資料版本＝研究中心 V2 面板＋R_DEL（V3 重跑見 oos_v3.py）。"""
import sys, json, time, hashlib, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
RJ = HERE / 'results.json'
R = json.loads(RJ.read_text(encoding='utf-8'))
assert 'PREREG' in R and hashlib.sha256(json.dumps(R['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == R['PREREG_sha256'], 'PREREG 不存在或被改過'
assert 'RESULTS' in R, '先跑 oos.py'
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
WP = {'MAIN': L.wplan_main(mp), 'TSMC30': O.tsmc_plan(mp, 0.3), 'COMBO': L.wplan_mix(mp, leg, WL_), 'L_ALONE': L.wplan_leg(leg)}
NAV = {}; TOV = {}
for k, wp in WP.items():
    for c, cost in (('normal', L.NORMAL),):
        NAV[(k, c)], tv = L.sim(wp, cost, t1)
        if c == 'normal': TOV[k] = round(tv, 4)
    log('sim', k, round(float(NAV[(k, 'normal')].iloc[-1]), 2))
NAV[('MAIN70_0050_30', 'normal')] = L.sleeve_mix(NAV[('MAIN', 'normal')], 0.3, t1, L.NORMAL)

W = {k: L.wr(v) for k, v in NAV.items()}
WEP = L.wr(ewPool)
xLP_ = W[('L_ALONE', 'normal')] - WEP
ref = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))['RESULTS']
assert abs(L.win(xLP_, 'VAL').mean() * 100 - ref['leg']['VAL']['L_vs_poolEW_normal']['mean_pp']) < 1e-3, '與 oos.py 不一致'
per = L.leg_period_returns(leg, t1)
pv = [x for x in per if x[0] >= L.SPLIT]; ps = [x for x in per if x[0] < L.SPLIT]
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


(HERE / 'results_robust.json').write_text(L.dumps({'written_at': pd.Timestamp.now(tz='UTC').isoformat(), 'ROBUST': rob}), encoding='utf-8')
log('written')
