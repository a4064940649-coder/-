#!/usr/bin/env python3
"""V3 資料重跑（o_infra_v3_rebuild：P 下市股價格＋X 永久停牌照對價出清＋D_hi 處置全期一致＋R_DEL）。
凍結規格不變（斷言 PREREG）；只報告，不改判定。寫 results_v3.json。
主策略用 o3_core.V3（含 D_hi 漏網名額剔除），斷言等權版淨值與 o_infra_v3_rebuild/navs_P1.parquet 的 V3 欄逐日相同；
加權組合用 ocore.simulate_w 的寫法加上 o3_core.simulate 的 X 出清（simulate_wx）。L 腿的處置濾網＝近似名單＋真實名單（漏網名額剔除只定義在前 300 池）。"""
import sys, json, time, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
RJ = HERE / 'results.json'
R = json.loads(RJ.read_text(encoding='utf-8'))
assert 'PREREG' in R and hashlib.sha256(json.dumps(R['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == R['PREREG_sha256']
assert 'RESULTS' in R
V3D = HERE.parent / 'o_infra_v3_rebuild'
sys.path.insert(0, str(V3D))
import o3_core as O3
sys.path.pop(0)
import numpy as np, pandas as pd
V3R = json.loads((V3D / 'results.json').read_text(encoding='utf-8'))
for f in O3.freeze()['files']: O3.check(f)
npatch = O3.apply_patch()
H = O3.load_disposal_hist(); EXTRA = {pd.Timestamp(k): v for k, v in V3R['PREREG_params']['D_hi_extra'].items()}
O3.set_disposal('all_plus', H, EXTRA)
XR = O3.exit_rules(O3.load_events(), 'main')
sys.path.insert(0, str(HERE))
import lcore as L
sys.path.pop(0)
K, V, O = L.K, L.V, L.O
assert O3.K is K
T0_ = time.time()
def log(*a): print(f'[{time.time()-T0_:5.0f}s]', *a, flush=True)
_IT0 = json.loads((HERE / 'iterations.json').read_text(encoding='utf-8'))
SPEC = _IT0['versions'][R['PREREG']['selection']['picked']]['spec']
P_, N_, SC_, WM_, WL_ = SPEC['P'], SPEC['n'], SPEC['score'], SPEC['weight'], SPEC['w_leg']
YR = O3.load_rdel(); _, Y5, _ = V.load_revenue()
S3 = O3.V3('gapfix', YR, Y5, exfin=True)
mp = S3.plan(10, 'official')
assert [p[0] for p in mp] == L.EXEC, 'V3 換股日與 V2 不同'
t0, t1, sd = S3.bounds(); assert t0 == L.T0 and t1 == L.T1

def simulate_wx(plan, buy, sell, t0, t1, xrules=None):
    """ocore.simulate_w ＋ o3_core.simulate 的 X 出清（換股日凍結部位若已過出清日，乘對價係數轉回現金）。"""
    G = K.GROW
    reb = {p[0]: (np.asarray(p[1], int), np.asarray(p[2], float)) for p in plan}
    val = np.zeros(K.N); cash = 1.0
    nav = np.empty(t1 - t0 + 1); tov = []; nex = 0
    rel = {j: (v[0], int(np.searchsorted(K.IDX, v[1])), int(np.searchsorted(K.IDX, v[3])) if len(v) > 3 else 10 ** 9) for j, v in (xrules or {}).items()}
    for t in range(t0, t1 + 1):
        if t in reb:
            pk, w = reb[t]
            tr = K.TRADED[t] & np.isfinite(K.On[t])
            gap = np.where(tr, G[t] * K.On[t] / np.where(tr, K.Cn[t], 1.0), 1.0)
            h = val * gap
            frz = (h > 0) & ~tr
            for j in np.where(frz)[0]:
                if j in rel and rel[j][1] <= t < rel[j][2]:
                    cash += h[j] * rel[j][0]; h[j] = 0.0; frz[j] = False; nex += 1
            htr = np.where(frz, 0.0, h)
            Vt = htr.sum() + cash
            wv = np.zeros(K.N); np.add.at(wv, pk, w)
            def cost_of(T):
                tgt = wv * T
                return buy * np.clip(tgt - htr, 0, None).sum() + sell * np.clip(htr - tgt, 0, None).sum()
            lo, hi = 0.0, Vt
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if mid + cost_of(mid) > Vt: hi = mid
                else: lo = mid
            tgt = wv * lo
            tov.append(np.clip(htr - tgt, 0, None).sum() / (Vt + h[frz].sum()))
            newval = np.where(frz, h, 0.0) + tgt
            oc = np.where(tr, K.Cn[t] / np.where(tr, K.On[t], 1.0), 1.0)
            val = np.where(frz, newval, newval * oc)
            cash = 0.0
        else:
            val = val * G[t] if t > t0 else val
        nav[t - t0] = val.sum() + cash
    return pd.Series(nav, index=K.IDX[t0:t1 + 1]), {'turnover': float(np.mean(tov[1:])), 'n_exits': nex}

# 驗證：主策略等權 ＝ V3 正式淨值
ref = pd.read_parquet(V3D / 'navs_P1.parquet')
navM, lgM = simulate_wx(L.wplan_main(mp), *L.NORMAL, t0, t1, XR)
r = ref[V3R['V3_nav_column']['column']].reindex(navM.index)
dd = float(((navM / navM.iloc[0]) / (r / r.iloc[0]) - 1).abs().max())
assert dd < 1e-9, dd
ewMain, _ = O3.simulate([(p[0], p[2]) for p in mp], 0, 0, t0, t1, xrules=XR)
re = ref[V3R['V3_nav_column']['EW_column']].reindex(ewMain.index)
dde = float(((ewMain / ewMain.iloc[0]) / (re / re.iloc[0]) - 1).abs().max()); assert dde < 1e-9, dde
log('V3 validated', dd, dde, 'patched cells', npatch)

leg = L.plan_leg(P_, N_, SC_, WM_); leg1 = L.plan_leg(P_, 1, 'v60', 'ew'); legX = L.plan_leg(P_, N_, SC_, WM_, excl=('2330',))
ewPool, _ = O3.simulate([(p[0], np.asarray(p[4], int)) for p in leg], 0, 0, t0, t1, xrules=XR)
ewPoolX, _ = O3.simulate([(p[0], np.asarray(p[4], int)) for p in legX], 0, 0, t0, t1, xrules=XR)
WP = {'MAIN': L.wplan_main(mp), 'TSMC30': O.tsmc_plan(mp, 0.3), 'COMBO': L.wplan_mix(mp, leg, WL_), 'MAIN70_TOP1VALUE30': L.wplan_mix(mp, leg1, 0.3),
      'MAIN70_POOLEW30': L.wplan_pool(leg, 0.3, mp), 'L_ALONE': L.wplan_leg(leg), 'L_ALONE_EX2330': L.wplan_leg(legX)}
NAV = {}
for k, wp in WP.items():
    for c, cost in (('normal', L.NORMAL), ('stress3', L.STRESS3)):
        NAV[(k, c)], _ = simulate_wx(wp, *cost, t0, t1, XR)
for c, cost in (('normal', L.NORMAL), ('stress3', L.STRESS3)):
    NAV[('MAIN70_0050_30', c)] = L.sleeve_mix(NAV[('MAIN', c)], 0.3, t1, cost)
log('sims done')
E = L.wr(K.E50); Z = L.wr(O.TSMC_IDX) - E
W = {k: L.wr(v) for k, v in NAV.items()}; WEP, WEPX = L.wr(ewPool), L.wr(ewPoolX)
KEYS = ('SEL', 'VAL', 'VAL_A', 'HOLD', 'ALL')
out = {'validation': {'MAIN_vs_V3_nav_maxreldiff': dd, 'EW_vs_V3_EW_maxreldiff': dde, 'patched_cells': npatch}, 'combos': {}, 'leg': {}}
for k in ('MAIN', 'TSMC30', 'COMBO', 'MAIN70_0050_30', 'MAIN70_POOLEW30', 'MAIN70_TOP1VALUE30', 'L_ALONE'):
    out['combos'][k] = {}
    for c in ('normal', 'stress3'):
        d = {}
        for p in KEYS:
            st = L.segstat(NAV[(k, c)], ewMain, p)
            st['paired_vs_MAIN'] = L.mstat(L.win(W[(k, c)] - W[('MAIN', c)], p))
            st['decomp_vs_MAIN'] = L.decomp(L.win(W[(k, c)] - E, p), L.win(W[('MAIN', c)] - E, p))
            d[p] = st
        out['combos'][k][c] = d
    s = out['combos'][k]['normal']
    log(k, {p: (s[p]['CAGR'], s[p]['t_0050'], s[p]['TE_ann']) for p in KEYS})
xM = W[('MAIN', 'normal')] - E
for p in KEYS:
    out['leg'][p] = {'L_vs_poolEW_normal': L.mstat(L.win(W[('L_ALONE', 'normal')] - WEP, p)), 'L_vs_poolEW_stress3': L.mstat(L.win(W[('L_ALONE', 'stress3')] - WEP, p)),
                     'ex2330_L_vs_poolEW': L.mstat(L.win(W[('L_ALONE_EX2330', 'normal')] - WEPX, p)),
                     'corr_Lex0050_MAINex0050': L.corr(L.win(W[('L_ALONE', 'normal')] - E, p), L.win(xM, p)),
                     'beta_Lex0050_on_2330m0050': L.beta(L.win(W[('L_ALONE', 'normal')] - E, p), L.win(Z, p))['beta'],
                     'beta_MAINex0050_on_2330m0050': L.beta(L.win(xM, p), L.win(Z, p))['beta']}
G2, dinfo = L.drift_zero_G(L.T_SPLIT, t1)
# 反事實：只換報酬（X 出清不受影響：2330 沒有出清事件）
def sim_cf(wp):
    Gs = K.GROW; K.GROW = G2
    try: return simulate_wx(wp, *L.NORMAL, t0, t1, XR)[0]
    finally: K.GROW = Gs
cf = {k: {p: L.segstat(sim_cf(WP[k]), ewMain, p)['t_0050'] for p in ('SEL', 'VAL', 'ALL')} for k in ('COMBO', 'TSMC30')}
C = out['combos']; cN = lambda k, p: C[k]['normal'][p]; lg = out['leg']
j1 = lg['SEL']['L_vs_poolEW_normal']['t'] >= 1.5 and lg['VAL']['L_vs_poolEW_normal']['mean_pp'] > 0
j2 = (cN('COMBO', 'VAL')['paired_vs_MAIN']['mean_pp'] >= -0.1 and cN('COMBO', 'HOLD')['CAGR'] >= cN('MAIN', 'HOLD')['CAGR']
      and cN('COMBO', 'VAL_A')['TE_ann'] <= cN('MAIN', 'VAL_A')['TE_ann'] - 3 and cN('COMBO', 'HOLD')['TE_ann'] <= cN('MAIN', 'HOLD')['TE_ann'] - 3)
j3 = (cN('COMBO', 'SEL')['t_0050'] >= cN('TSMC30', 'SEL')['t_0050'] and cf['COMBO']['SEL'] > cf['TSMC30']['SEL'] and cf['COMBO']['VAL'] > cf['TSMC30']['VAL']
      and lg['SEL']['ex2330_L_vs_poolEW']['t'] >= 1.5 and lg['VAL']['ex2330_L_vs_poolEW']['mean_pp'] > 0)
out['drift_zero_cf'] = {'drift_bp': dinfo, **cf}
out['judgement_report_only'] = {'(1)': bool(j1), '(2)': bool(j2), '(3)': bool(j3), 'note': 'V3 只報告；(4) 未重算（V2 已不過）'}
(HERE / 'results_v3.json').write_text(L.dumps({'written_at': pd.Timestamp.now(tz='UTC').isoformat(), 'V3': out}), encoding='utf-8')
log('done', out['judgement_report_only'], {p: lg[p]['L_vs_poolEW_normal'] for p in ('SEL', 'VAL', 'HOLD')})
