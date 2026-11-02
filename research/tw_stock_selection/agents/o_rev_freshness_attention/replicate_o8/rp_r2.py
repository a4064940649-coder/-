#!/usr/bin/env python3
"""獨立複製 R2：事件 A（期限日盤後、高關注組換成當晚口徑前 10 名的交集）＋事件 B（J_j 換成 plans_j7 AB_V7_B10 名單）。
對照＝同 j、同 seed 的 AB_V7_B10。ecore7 K5T 100 萬 FIX（一般／壓力）、零衝擊 ideal（一般、每邊再加 0.15%）。"""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
import multiprocessing as mp
HERE = Path(__file__).resolve().parent
AG = HERE.parent.parent
EX = AG / 'o_v7_listing_exec_engine'
sys.path.insert(0, str(EX))
import v7env, ecore7 as E, v7stats as ST
sys.path.pop(0)
L = v7env.load('v7'); K, V = L.K, L.V
XR = L.setup_px()
YFX, Y5F = L.load_rfix(); DMT = L.disp_mask_fn('trade'); GAP = L.gap_dict()
sys.path.insert(0, str(AG / 'o_stagger_tranche')); import st_core as STC; sys.path.pop(0)
IDX, COLS, N = K.IDX, K.COLS, K.N
T0 = time.time()
def log(*a): print(f'[{time.time()-T0:6.0f}s]', *a, flush=True)
D = E.Data(L, XR)
SPEC = json.loads((EX / 'data' / 'model_spec7.json').read_text(encoding='utf-8'))
K5T = E.Model(SPEC, 'K5T')
NORMAL = dict(lim=True, odd0=True, cash=0.02, feemix=True, zpart=1.0, disp=True, mdelay=True)
STRESS = dict(lim=True, odd0=True, cash=0.02, feemix=False, zpart=1.0, disp=True, mdelay=True, extra=0.004, odd_stress=True)
PJ = pd.read_csv(EX / 'data' / 'plans_j7.csv', dtype={'picks': str})
PJ = PJ[(PJ.strat == 'AB_V7_B10') & (PJ.seed < 5) & (PJ.j >= 1)]
JEND = json.loads((EX / 'data' / 'jscan_end_days.json').read_text(encoding='utf-8'))

class JB(L.V4):
    def y3(self, d, back=0):
        m = self.anchor[d] - back
        Y = self.Y5 if d >= V.SWITCH else self.YL
        return np.where(V.FINMASK, np.nan, K.yoy3_for_month(Y, m))
_TA = {}
def event_a(m, seed):
    """回傳 (exec_idx, signal, T_A set, hi mask) 或 None。"""
    key = (str(m), seed)
    if key in _TA: return _TA[key]
    Dm = STC.deadline(m); out = None
    if Dm is not None and Dm in IDX:
        t = IDX.get_loc(Dm)
        if K.GOOD[t] and t + 1 < len(IDX) and K.GOOD[t + 1]:
            nxt = IDX[t + 1]
            S = JB(YFX, Y5F, DMT, gap=GAP, pfix=1.0, unavail=(lambda d, _m=m, _n=nxt, _s=seed: STC.unavail_for(_m, _n, _s)))
            S.anchor = {Dm: m}
            cs = S.cross_section(Dm); el = cs['el']
            if el.sum() >= 50:
                o = V.V2.order(cs, 'official')[:int(el.sum())]
                TA = set(int(x) for x in o[:10])
                pv = pd.Series(np.where(el, S.V60[t], np.nan)).rank(pct=True).to_numpy()
                hi = set(int(x) for x in np.where(el & (pv > 2 / 3))[0])
                out = (t + 1, Dm, TA, hi)
    _TA[key] = out
    return out

PLANS = {}; BOUND = {}; GRID = {}; NEV = {}
for (j, s), g in PJ.groupby(['j', 'seed']):
    base = [(int(e), [K.CI[c] for c in pk.split(',')], pd.Timestamp(sg)) for e, pk, sg in zip(g.exec_idx, g.picks, g.signal)]
    r2 = []; H = None; nev = 0
    for e, pk, sg in base:
        m = pd.Period(sg, 'M')
        if H is not None:
            ea = event_a(m, s)
            if ea is not None and ea[0] < e and ea[0] > r2[-1][0]:
                tgt = (H - ea[3]) | (ea[2] & ea[3])
                if tgt != H and len(tgt) > 0:
                    r2.append((ea[0], sorted(tgt), ea[1])); nev += 1
        r2.append((e, pk, sg)); H = set(pk)
    de = pd.Timestamp(JEND[f'AB_V7_B10|J{j}'])
    BOUND[j] = (int(IDX.get_loc(base[0][2])), int(IDX.get_loc(de)))
    GRID[j] = [b[2] for b in base] + [de]
    PLANS[('R2', j, s)] = r2; PLANS[('B10', j, s)] = base; NEV[(j, s)] = (nev, len(base))
log('plans built', len(PLANS))

def job(a):
    key, mode = a
    t0, t1 = BOUND[key[1]]
    pl = PLANS[key]
    if mode == 'k5t_n':
        nav, _ = E.Sim(D, K5T, NORMAL).run(pl, t0, t1, E.TIERS['100萬'], 'FIX', 'A')
    elif mode == 'k5t_s':
        nav, _ = E.Sim(D, K5T, STRESS).run(pl, t0, t1, E.TIERS['100萬'], 'FIX', 'B')
    else:
        b, s_ = L.cost_rates('normal') if mode == 'z_n' else L.cost_rates(('extra', 0.0015))
        nav, _ = E.ideal(D, [(p[0], p[1]) for p in pl], t0, t1, b, s_)
    return key, mode, nav.pct_change()

if __name__ == '__main__':
    jobs = [(k, m) for k in PLANS for m in ('z_n', 'z_x', 'k5t_n', 'k5t_s')]
    with mp.get_context('fork').Pool(4) as P:
        out = P.map(job, jobs, chunksize=4)
    log('sims done')
    R = {}
    by = {}
    for key, mode, r in out: by.setdefault((key[0], key[1], mode), []).append(r)
    E50 = K.E50
    res = {}
    for mode in ('z_n', 'z_x', 'k5t_n', 'k5t_s'):
        rows = []
        for j in range(1, 13):
            nr = ST.mean_nav(by[('R2', j, mode)]); nb = ST.mean_nav(by[('B10', j, mode)])
            sd = GRID[j]; de = sd[-1]
            cr = ST.cagr_mdd(nr, ST.A0, min(de, pd.Timestamp(ST.B0)))[0]; cb = ST.cagr_mdd(nb, ST.A0, min(de, pd.Timestamp(ST.B0)))[0]
            d = ST.xs(nr, nb, sd); d19 = ST.xs(nr, nb, sd, '2019-05-01', ST.B0)
            rows.append(dict(j=j, R2=cr, B10=cb, diff=cr - cb, t_pair=ST.tstat(d), mean_pp=float(d.mean() * 100),
                             mean19_pp=float(d19.mean() * 100), t19=ST.tstat(d19),
                             R2_19=ST.cagr_mdd(nr, '2019-05-01', min(de, pd.Timestamp(ST.B0)))[0], B10_19=ST.cagr_mdd(nb, '2019-05-01', min(de, pd.Timestamp(ST.B0)))[0],
                             E50_19=ST.cagr_mdd(E50.reindex(nr.index), '2019-05-01', min(de, pd.Timestamp(ST.B0)))[0],
                             t50_R2=ST.tstat(ST.xs(nr, E50.reindex(nr.index), sd)), t50_B10=ST.tstat(ST.xs(nb, E50.reindex(nb.index), sd))))
        df = pd.DataFrame(rows)
        res[mode] = dict(median=df.median().round(3).to_dict(), byj=df.round(3).to_dict('records'))
        print(mode, df.median().round(3).to_dict())
        print(df[['j', 'R2', 'B10', 'diff', 'mean_pp', 't_pair']].round(3).to_string())
    res['events'] = {f'J{j}s{s}': v for (j, s), v in NEV.items()}
    json.dump(res, open(HERE / 'rp_r2.json', 'w'), ensure_ascii=False, indent=1, default=float)
    log('done')
