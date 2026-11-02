#!/usr/bin/env python3
"""步驟 6：主策略改良候選 R2（兩段式換股）——PREREG 後一次跑完（全期，含前期與後期）。
R2_j（j＝1～12；j＝0 時兩段同日，不定義）：
  事件 A（期限日當晚）：錨定月 m 的期限日 D_m 收盤後算訊號，營收用「期限日當晚」口徑（＝st_core.unavail_for 以 D_m 的下一個交易日評估：
    只有 2020-02 延長期限與 2026 INS15 名單仍視為未公告），規格＝AB_V7_B10（V7 面板、缺營收下市股營收百分位 1.0）。
    取官方同分序前 10 名 T_A（開盤漲停且未持有者跳過）。高關注組 Hi_A＝D_m 的 el 內 V60 前 1/3。
    D_m 次一交易日開盤：持股 H 換成 X＝(H − Hi_A) ∪ (T_A ∩ Hi_A)（檔數可不等於 10；等權；X＝H 則不交易）。
  事件 B（官方 j 日）：J_j 收盤訊號、次日開盤換成 AB_V7_B10 J_j 名單（o_v7 data/plans_j7.csv，seed 0～4，原名單一字不改）。
對照：同一 j、同一 seed 的 AB_V7_B10 J_j（只有事件 B）。另列官方日 AB_V7_B10（navs_v7）與 0050。
引擎：零衝擊＝v4lib.sim_plan（一般、壓力、一般＋每邊 0.15%）；可執行＝o_v7 ecore7.Sim（第 3 題 V7X 未完成，暫用 ecore7 並加註），
  K5T（主欄）與 TICK（敏感度）、四級距 × FIX／GROW、一般與壓力（opts 與 o_v7 run_exec7 相同）。
判定（候選，事前寫死）：主欄＝100 萬 FIX、K5T、一般成本；j＝1～12 中位數：
  (1) 年化配對差（R2 − B10）> 0 且配對 t（窗＝事件 B 訊號日到下一事件 B 訊號日）≥ 1.5；(2) 2019-05 起配對差點估計 ≥ 0；
  (3) 壓力成本（K5T 100 萬 FIX）配對差 ≥ 0；(4) 零衝擊每邊再加 0.15% 時配對差 ≥ 0（＝成本兩平 ≥ 0.15%／邊）。
  全部成立 → 待觀察；任一不成立 → 不可用。"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import multiprocessing as mp
import fa_common as C, fa_build as B
C.check_prereg()
E = C.engine(); K, L, V, STC = E['K'], E['L'], E['V'], E['STC']
XR = E['XR']; IDX = K.IDX
sys.path.insert(0, str(C.V7DIR)); import ecore7 as EC; sys.path.pop(0)
D = EC.Data(L, XR)
SPEC = json.loads((C.V7DIR / 'data' / 'model_spec7.json').read_text(encoding='utf-8'))
MODELS = {k: EC.Model(SPEC, k) for k in ('K5T', 'TICK')}
NORMAL = dict(lim=True, odd0=True, cash=0.02, feemix=True, zpart=1.0, disp=True, mdelay=True)
STRESS = dict(lim=True, odd0=True, cash=0.02, feemix=False, zpart=1.0, disp=True, mdelay=True, extra=0.004, odd_stress=True)
SEEDS = [0, 1, 2, 3, 4]; JR = list(range(1, 13))
PJ = pd.read_csv(C.V7DIR / 'data' / 'plans_j7.csv', dtype={'picks': str})
PJ = PJ[(PJ.strat == 'AB_V7_B10') & PJ.seed.isin(SEEDS)]
JEND = json.loads((C.V7DIR / 'data' / 'jscan_end_days.json').read_text(encoding='utf-8'))
_, V60x = V.rolling_signals('gapfix')
TCAL = STC.TCAL
def next_td(d): return TCAL[TCAL.searchsorted(d, side='right')]

# 事件 A 的日子
AD = {}
for m in E['ANCH']:
    dA = STC.deadline(m)
    if dA is None or dA not in IDX: continue
    t = IDX.get_loc(dA)
    if not (K.GOOD[t] and t + 1 < len(IDX) and K.GOOD[t + 1]): continue
    AD[m] = (dA, t + 1)
SA = {}
for s in SEEDS:
    S = E['JV7'](E['YFX'], E['Y5F'], E['DMT'], gap=E['GAP'], pfix=1.0, unavail=(lambda d, _s=s: STC.unavail_for(SA[_s].anchor[d], next_td(d), _s)))
    S.set_anchor({v[0]: m for m, v in AD.items()}); SA[s] = S

def r2_plan(j, s):
    g = PJ[(PJ.j == j) & (PJ.seed == s)].sort_values('exec_idx')
    Bev = [(int(e), [K.CI[c] for c in pk.split(',')], pd.Timestamp(sg)) for e, pk, sg in zip(g.exec_idx, g.picks, g.signal)]
    t0 = IDX.get_loc(Bev[0][2]); de = pd.Timestamp(JEND[f'AB_V7_B10|J{j}']); t1 = IDX.get_loc(de)
    evs = [(e, 1, ('B', pk, sg)) for e, pk, sg in Bev] + [(eA, 0, ('A', m, dA)) for m, (dA, eA) in AD.items() if t0 < eA <= t1]
    evs.sort(key=lambda x: (x[0], x[1]))
    out = []; H = None; nA = 0; nA_trade = 0; sizes = []
    for e, _, ev in evs:
        if ev[0] == 'B':
            H = set(ev[1]); out.append((e, list(ev[1]), ev[2]))
        else:
            if H is None: continue
            m, dA = ev[1], ev[2]
            cs = SA[s].cross_section(dA); el = cs['el']
            if el.sum() < 50: continue
            nA += 1
            gu = K.gap_up_open(e)
            TA = B.pick(cs['sc'], el, 10, gu, H, cs['y3'])
            hi = B.tercile(V60x[cs['t']], el) == 2
            X = {x for x in H if not hi[x]} | {x for x in TA if hi[x]}
            if X and X != H:
                out.append((e, sorted(X), dA)); H = X; nA_trade += 1; sizes.append(len(X))
    return out, t0, t1, [p[2] for p in Bev] + [de], dict(nA=nA, nA_trade=nA_trade, size_mean=float(np.mean(sizes)) if sizes else None,
                                                         size_min=int(min(sizes)) if sizes else None, size_max=int(max(sizes)) if sizes else None)

def b10_plan(j, s):
    g = PJ[(PJ.j == j) & (PJ.seed == s)].sort_values('exec_idx')
    return [(int(e), [K.CI[c] for c in pk.split(',')], pd.Timestamp(sg)) for e, pk, sg in zip(g.exec_idx, g.picks, g.signal)]

def turnover(plan):
    tv = []
    for a, b in zip(plan[:-1], plan[1:]):
        A_, B_ = set(a[1]), set(b[1]); tv.append(len(A_ - B_) / max(len(A_), 1))
    return float(np.sum(tv))

CELLS = [('zero', 'normal'), ('zero', 'stress'), ('zero', 'extra15')] + \
        [('K5T', c, tier, ver) for c in ('normal', 'stress') for tier in ('30萬', '100萬', '300萬', '1000萬') for ver in ('FIX', 'GROW')] + \
        [('TICK', 'normal', '100萬', 'FIX'), ('TICK', 'normal', '1000萬', 'FIX')]

def run_cell(plan, t0, t1, cell):
    if cell[0] == 'zero':
        cost = {'normal': 'normal', 'stress': 'stress', 'extra15': ('extra', 0.0015)}[cell[1]]
        nav, _ = L.sim_plan([(p[0], np.array(p[1], int)) for p in plan], t0, t1, cost, XR)
        return nav.pct_change().to_numpy()
    model, c, tier, ver = cell
    nav, _ = EC.Sim(D, MODELS[model], NORMAL if c == 'normal' else STRESS).run(plan, t0, t1, EC.TIERS[tier], ver, 'A' if c == 'normal' else 'B')
    return nav.pct_change().to_numpy()

def job(a):
    j, s = a
    r2, t0, t1, grid, info = r2_plan(j, s)
    b10 = b10_plan(j, s)
    out = {}
    for cell in CELLS:
        out[('R2',) + cell] = run_cell(r2, t0, t1, cell)
        out[('B10',) + cell] = run_cell(b10, t0, t1, cell)
    info.update(turn_R2=turnover(r2) / (len(grid) - 1), turn_B10=turnover(b10) / (len(grid) - 1))
    return j, s, t0, t1, grid, out, info

def win(nav, grid):
    v = nav.reindex(pd.DatetimeIndex(grid)).ffill().to_numpy()
    return pd.Series(v[1:] / v[:-1] - 1, index=pd.DatetimeIndex(grid[:-1]))

def cagr(nav, a, b):
    s = nav[(nav.index >= pd.Timestamp(a)) & (nav.index <= pd.Timestamp(b))]
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    return float((s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1) * 100

if __name__ == '__main__':
    T0 = time.time()
    jobs = [(j, s) for j in JR for s in SEEDS]
    with mp.get_context('fork').Pool(4) as Pp: res = Pp.map(job, jobs, chunksize=1)
    E50 = K.E50.ffill()
    OUT = {'cells': {}, 'info': {}}
    for cell in CELLS:
        ck = '|'.join(cell)
        rows = {}
        for j in JR:
            rr = [x for x in res if x[0] == j]
            t0, t1, grid = rr[0][2], rr[0][3], rr[0][4]
            idx = IDX[t0:t1 + 1]
            navs = {}
            for st in ('R2', 'B10'):
                m = np.nanmean([x[5][(st,) + cell] for x in rr], axis=0)
                navs[st] = pd.Series((1 + np.nan_to_num(m, nan=0.0)).cumprod(), index=idx)
            wr, wb = win(navs['R2'], grid), win(navs['B10'], grid); we = win(E50.reindex(idx).ffill(), grid)
            dlt = (wr - wb).dropna(); post = dlt[dlt.index >= pd.Timestamp('2019-05-01')]; pre = dlt[dlt.index <= pd.Timestamp('2018-12-31')]
            a, b = idx[0], idx[-1]
            o = dict(CAGR_R2=cagr(navs['R2'], a, b), CAGR_B10=cagr(navs['B10'], a, b), CAGR_0050=cagr(E50.reindex(idx).ffill(), a, b),
                     CAGR_R2_post=cagr(navs['R2'], '2019-05-01', b), CAGR_B10_post=cagr(navs['B10'], '2019-05-01', b), CAGR_0050_post=cagr(E50.reindex(idx).ffill(), '2019-05-01', b),
                     diff_mean_pp=float(dlt.mean() * 100), diff_t=C.tstat(dlt.to_numpy()), diff_post_mean_pp=float(post.mean() * 100), diff_post_t=C.tstat(post.to_numpy()),
                     diff_pre_mean_pp=float(pre.mean() * 100), diff_pre_t=C.tstat(pre.to_numpy()),
                     t_R2_vs_0050=C.tstat((wr - we).dropna().to_numpy()), t_B10_vs_0050=C.tstat((wb - we).dropna().to_numpy()), n_windows=int(len(dlt)))
            o['CAGR_diff'] = o['CAGR_R2'] - o['CAGR_B10']; o['CAGR_diff_post'] = o['CAGR_R2_post'] - o['CAGR_B10_post']
            rows[j] = o
        med = {k: float(np.median([rows[j][k] for j in JR])) for k in rows[JR[0]]}
        OUT['cells'][ck] = {'by_j': rows, 'median_j': med}
        print(ck, {k: round(med[k], 3) for k in ('CAGR_diff', 'diff_t', 'diff_post_mean_pp', 'CAGR_R2', 'CAGR_B10')}, flush=True)
    for j in JR:
        ii = [x[6] for x in res if x[0] == j]
        OUT['info'][j] = {k: float(np.mean([z[k] for z in ii if z[k] is not None])) for k in ii[0]}
    # 官方日 AB_V7_B10（零衝擊一般成本）參考
    NV = pd.read_parquet(C.V7DIR / 'navs_v7.parquet')
    if 'AB_V7_B10|normal' in NV:
        nv = NV['AB_V7_B10|normal'].dropna()
        OUT['official_AB_V7_B10_zero_normal'] = dict(CAGR=cagr(nv, nv.index[0], nv.index[-1]), CAGR_post=cagr(nv, '2019-05-01', nv.index[-1]))
    M = OUT['cells']
    main = M['K5T|normal|100萬|FIX']['median_j']; strs = M['K5T|stress|100萬|FIX']['median_j']; x15 = M['zero|extra15']['median_j']
    cond = {'(1) 配對差>0 且 t≥1.5': bool(main['CAGR_diff'] > 0 and main['diff_t'] >= 1.5),
            '(2) 2019-05 起點估計≥0': bool(main['diff_post_mean_pp'] >= 0),
            '(3) 壓力成本配對差≥0': bool(strs['diff_mean_pp'] >= 0),
            '(4) 每邊+0.15% 配對差≥0': bool(x15['diff_mean_pp'] >= 0)}
    OUT['candidate_conditions'] = cond
    OUT['verdict'] = '待觀察' if all(cond.values()) else '不可用'
    OUT['note'] = '可執行引擎暫用 o_v7 ecore7（第 3 題 V7X 尚未完成）；seeds 0～4；R2 只定義 j＝1～12'
    OUT['runtime_s'] = time.time() - T0
    C.save_key('RESULTS_R2', OUT)
    print(json.dumps(C.conv({'cond': cond, 'verdict': OUT['verdict'], 'info_j1': OUT['info'][1]}), ensure_ascii=False))
