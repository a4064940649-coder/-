#!/usr/bin/env python3
"""S4b-2 V7X 可執行重算（PREREG＋ADDENDUM 之後，只跑一次）：cz_v7x.CSim（只讀 import）× K5T × GROW × cons_main，R4＝orig_cfg＋f1,f2,f3=0.5,f4（同題 1）。
名單＝s11a 產生的 plans_strict.csv（嚴格＝寬鬆，斷言計數＝題 1）與 plans_posthoc.csv。
格：嚴格 × {100 萬一般（重現題 1）、30 萬一般（V7X 標準：2020-10-26 前盤後零股，f3 上限）、30 萬一般整股限制（2020-10-26 前不允許零股）、30 萬壓力}；
posthoc：若名單有任何變動 → 6 正式格（100 萬一般／壓力、300 萬、1,000 萬一般）全跑；否則 Δ＝0 為構造結果。
python3 -I -B s11b_v7x_run.py"""
import sys, json, hashlib, time, math
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent
SCR = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/o_disp_pkl18')
RJ = HERE / 'results.json'
sys.path.insert(0, str(AG / 'o_engine_delist_consideration')); import cz_v7x as CV, cz_engine as CE; sys.path.pop(0)
D = CV.D
sys.path.insert(0, str(CV.V7X)); import x_stats as S; import x_exec as XE; sys.path.pop(0)
VR = json.loads((CV.V7X / 'results.json').read_text())
BASE = dict(VR['PREREG']['orig_cfg']); BASE.update(dict(f1=True, f2=True, f3=0.5, f4=True))
R2 = json.loads((AG / 'o_engine_delist_consideration/results.json').read_text(encoding='utf-8'))
dd = R2['PREREG']['events_v2']['summary']['d']['d_used']
E2 = pd.read_csv(AG / 'o_engine_delist_consideration/events_v2.csv', dtype={'code': str, 'successor': str})
ST = CE.build_settlement(E2, 'main', d=dd, TR=D.TR, IDX=D.IDX, Cn=D.C, CI=D.CI, On=D.O)
I_ODD = XE.I_ODD
class WholeLotPre(CV.CSim):
    """2020-10-26（盤中零股開放）前不允許任何零股成交（整股限制）。"""
    def oddcap(self, t, j):
        if t < I_ODD: return 0
        return super().oddcap(t, j)
TIERS = {'30萬': 3e5, '100萬': 1e6, '300萬': 3e6, '1000萬': 1e7}
def to_plan(g): return sorted([(int(e), np.array([D.CI[c] for c in pk.split(',')], int), D.TPOS[pd.Timestamp(sg)]) for e, pk, sg in zip(g.exec_idx, g.picks, g.signal)], key=lambda x: x[0])
def job(a):
    dv, v, j, s, pl, cells = a
    out = {}
    for (c, t, mode) in cells:
        cfg = dict(tier=TIERS[t], mode='GROW', model='K5T', cost=c); cfg.update(BASE)
        Cls = WholeLotPre if mode == 'wholelot' else CV.CSim
        sm = Cls(pl, dict(cfg), settle=ST, cmode='consideration'); out[(c, t, mode)] = sm.run()
    return dv, v, j, s, out
if __name__ == '__main__':
    R = json.loads(RJ.read_text(encoding='utf-8'))
    for k in ('PREREG', 'PREREG_ADDENDUM'): assert hashlib.sha256(json.dumps(R[k], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == R[k + '_sha256'], k
    assert 'RESULTS_V7X' not in R, '只跑一次'
    PJ = json.loads((HERE / 's11a_v7x_plans.json').read_text(encoding='utf-8'))
    assert all(PJ['checks']['strict_counts_eq_T1'].values()) and PJ['checks']['strict_ends_eq_T1']
    for dv in ('strict', 'posthoc'): assert hashlib.sha256(Path(PJ['out'][dv]['file']).read_bytes()).hexdigest() == PJ['out'][dv]['sha256']
    t0 = time.time()
    CELLS_S = [('normal', '100萬', 'std'), ('normal', '30萬', 'std'), ('normal', '30萬', 'wholelot'), ('stress', '30萬', 'std')]
    CELLS_P = [('normal', '100萬', 'std'), ('stress', '100萬', 'std'), ('normal', '300萬', 'std'), ('normal', '1000萬', 'std')]
    jobs = []
    for dv, cells in (('strict', CELLS_S), ('posthoc', CELLS_P)):
        if dv == 'posthoc' and PJ['checks']['posthoc_plans_changed'] == 0: continue
        PL = pd.read_csv(PJ['out'][dv]['file'], dtype={'picks': str, 'inserted': str})
        for (v, j, s), g in PL.groupby(['variant', 'j', 'seed']): jobs.append((dv, v, int(j), int(s), to_plan(g), cells))
    navs = {}
    with Pool(4) as P:
        for n, (dv, v, j, s, out) in enumerate(P.imap_unordered(job, jobs, chunksize=1)):
            for ck, nv in out.items(): navs.setdefault((dv, v, j) + ck, []).append((s, nv))
            if n % 100 == 0: print(n, len(jobs), round(time.time() - t0), flush=True)
    cols = {'|'.join(map(str, k)): S.avg_nav([nv for s, nv in sorted(L, key=lambda x: x[0])]) for k, L in navs.items()}
    df = pd.DataFrame(cols).sort_index(axis=1); df.to_parquet(SCR / 'navs_v7x_pkl18.parquet')
    def segx(nav, j, a, b):
        e = S.end_day('AB_V7_B10', j); nav = nav.loc[:e].dropna()
        x = nav[(nav.index >= a) & (nav.index <= b)]
        if len(x) < 20: return np.nan
        prev = nav[nav.index < x.index[0]]; s0 = prev.index[-1] if len(prev) else x.index[0]
        return S.cagr(nav, s0, x.index[-1]) - S.cagr(S.E50, s0, x.index[-1])
    TAB = {}
    keys = sorted({(k[0], k[1]) + k[3:] for k in navs})
    for (dv, v, c, t, mode) in keys:
        nv = [df[f'{dv}|{v}|{j}|{c}|{t}|{mode}'] for j in range(13)]
        T = S.jsumm([S.stats(nv[j], 'AB_V7_B10', j) for j in range(13)])
        for nm, (a, b) in {'S0612': ('2006-01-01', '2012-12-31'), 'S1318': ('2013-01-01', '2018-12-31'), 'S19on': ('2019-01-01', '2026-12-31')}.items():
            xs = np.array([segx(nv[j], j, a, b) for j in range(13)]); T[f'exc_{nm}_med'] = float(np.nanmedian(xs))
        TAB[f'{dv}|{v}|cons_main|{c}|{t}|{mode}'] = T
    T1 = json.loads((AG / 'o_delist_rev_backfill_allpit/results.json').read_text(encoding='utf-8'))['RESULTS_V7X']['table']
    F = ('exc_med', 'exc_p10', 't_0050_med', 'exc_Y19on_med', 'exc_S1318_med', 'exc_S0612_med', 'exc_S19on_med', 'maxDD_med', 'CAGR_Y19on_med')
    rep = {}
    for v in ('REAL_opt', 'REAL_p90', 'REAL_carry'):
        a = TAB[f'strict|{v}|cons_main|normal|100萬|std']; b = T1[f'{v}|cons_main|normal|100萬']
        rep[v] = max(abs(a[f] - b[f]) for f in F if f in b)
    R = json.loads(RJ.read_text(encoding='utf-8')); assert 'RESULTS_V7X' not in R
    R['RESULTS_V7X'] = dict(utc=pd.Timestamp.now('UTC').isoformat(), runtime_s=round(time.time() - t0, 1), reproduce_T1_100m_normal_maxabs=rep,
                            posthoc_plans_changed=PJ['checks']['posthoc_plans_changed'], table=TAB, navs_file=str(SCR / 'navs_v7x_pkl18.parquet'))
    RJ.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    for k, T in TAB.items(): print(k, {f: round(T[f], 3) for f in F if f in T})
    print('reproduce', rep, round(time.time() - t0))
