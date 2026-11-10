#!/usr/bin/env python3
"""integrate_o19 V7X 可執行整合（PREREG 之後只跑一次）：cz_v7x.CSim（只讀 import）× K5T × GROW × cons_main，R4＝orig_cfg＋f1,f2,f3=0.5,f4（同題 1／審查題）。
名單＝gen_v7x_plans.py 的 plans_{strict,posthoc,I3a,I3b,I3c}.csv。strict（＝現行正式）780 組 × 7 格全跑；其他版本只跑名單有變動的組，其餘沿用 strict 淨值（決定性引擎）。
格：100 萬一般／壓力、300 萬一般、1,000 萬一般、30 萬一般（標準）、30 萬一般整股限制、30 萬壓力。
另：各版本對主策略（strict 同放回版同 j）的訊號日窗月差與 t、窗報酬相關、日報酬相關；I5 固定比例混合 80/20、50/50（第二條腿 strict、I3a），每個 j 的訊號日再平衡。
python3 -I -B run_v7x.py"""
import sys, json, hashlib, time, math
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent
SCR = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/integrate_o19')
RJ = HERE / 'results.json'
sys.path.insert(0, str(AG / 'o_engine_delist_consideration')); import cz_v7x as CV, cz_engine as CE; sys.path.pop(0)
D = CV.D
sys.path.insert(0, str(CV.V7X)); import x_stats as S; import x_exec as XE; sys.path.pop(0)
VR = json.loads((CV.V7X / 'results.json').read_text())
BASE = dict(VR['PREREG']['orig_cfg']); BASE.update(dict(f1=True, f2=True, f3=0.5, f4=True))
R2 = json.loads((AG / 'o_engine_delist_consideration/results.json').read_text(encoding='utf-8'))
dd_ = R2['PREREG']['events_v2']['summary']['d']['d_used']
E2 = pd.read_csv(AG / 'o_engine_delist_consideration/events_v2.csv', dtype={'code': str, 'successor': str})
ST = CE.build_settlement(E2, 'main', d=dd_, TR=D.TR, IDX=D.IDX, Cn=D.C, CI=D.CI, On=D.O)
I_ODD = XE.I_ODD
class WholeLotPre(CV.CSim):
    def oddcap(self, t, j):
        if t < I_ODD: return 0
        return super().oddcap(t, j)
TIERS = {'30萬': 3e5, '100萬': 1e6, '300萬': 3e6, '1000萬': 1e7}
CELLS = [('normal', '100萬', 'std'), ('stress', '100萬', 'std'), ('normal', '300萬', 'std'), ('normal', '1000萬', 'std'),
         ('normal', '30萬', 'std'), ('normal', '30萬', 'wholelot'), ('stress', '30萬', 'std')]
VARS = ['REAL_opt', 'REAL_p90', 'REAL_carry']; DVS = ['strict', 'posthoc', 'I3a', 'I3b', 'I3c']
def to_plan(g): return sorted([(int(e), np.array([D.CI[c] for c in pk.split(',')], int), D.TPOS[pd.Timestamp(sg)]) for e, pk, sg in zip(g.exec_idx, g.picks, g.signal)], key=lambda x: x[0])
def job(a):
    dv, v, j, s, pl = a
    out = {}
    for (c, t, mode) in CELLS:
        cfg = dict(tier=TIERS[t], mode='GROW', model='K5T', cost=c); cfg.update(BASE)
        Cls = WholeLotPre if mode == 'wholelot' else CV.CSim
        out[(c, t, mode)] = Cls(pl, dict(cfg), settle=ST, cmode='consideration').run()
    return dv, v, j, s, out
def segbounds(nav, a, b):
    x = nav[(nav.index >= a) & (nav.index <= b)]
    prev = nav[nav.index < x.index[0]]; s0 = prev.index[-1] if len(prev) else x.index[0]
    return s0, x.index[-1]
PER = {'ALL': ('1900-01-01', '2100-01-01'), 'S0612': ('2006-01-01', '2012-12-31'), 'S1318': ('2013-01-01', '2018-12-31'),
       'S19on': ('2019-01-01', '2026-12-31'), 'S24on': ('2024-01-01', '2026-12-31'), 'SEL': ('1900-01-01', '2019-04-30'), 'VAL': ('2019-05-01', '2026-12-31')}
def winret(nav, j):
    e = S.end_day('AB_V7_B10', j); sd = S.SIGD[('AB_V7_B10', j)]; sd = sd[sd < e].append(pd.DatetimeIndex([e]))
    r = nav.reindex(sd); rb = S.E50.reindex(sd)
    return pd.DataFrame({'r': (r.shift(-1) / r - 1).iloc[:-1], 'r50': (rb.shift(-1) / rb - 1).iloc[:-1]})
def pstats(nav, j):
    """各期間：年化、0050 年化、超額、最大回撤、對 0050 t（窗起點在期間內）。"""
    e = S.end_day('AB_V7_B10', j); nav = nav.loc[:e].dropna(); w = winret(nav, j); o = {}
    for k, (a, b) in PER.items():
        s0, s1 = segbounds(nav, a, b); seg = nav.loc[s0:s1]
        ww = w[(w.index >= a) & (w.index <= b)]
        o[k] = dict(cagr=S.cagr(nav, s0, s1), cagr50=S.cagr(S.E50, s0, s1), exc=S.cagr(nav, s0, s1) - S.cagr(S.E50, s0, s1),
                    mdd=float((seg / seg.cummax() - 1).min() * 100), t=S.tstat(ww.r - ww.r50))
    return o
def vsmain(nav, ref, j):
    e = S.end_day('AB_V7_B10', j); a = winret(nav.loc[:e].dropna(), j); b = winret(ref.loc[:e].dropna(), j)
    J = a.join(b, rsuffix='_m', how='inner').dropna(); o = {}
    for k, (x, y) in PER.items():
        q = J[(J.index >= x) & (J.index <= y)]; d = q.r - q.r_m
        o[k] = dict(diff_pp=float(d.mean() * 100), t=(S.tstat(d) if d.std() > 1e-15 else None),
                    corr=(float(np.corrcoef(q.r, q.r_m)[0, 1]) if q.r.std() > 0 and q.r_m.std() > 0 else None))
    dr = pd.concat([nav.loc[:e].pct_change(), ref.loc[:e].pct_change()], axis=1).dropna()
    o['daily_corr'] = float(dr.corr().iloc[0, 1]) if dr.iloc[:, 0].std() > 0 else None
    return o
def blend(nm, nl, j, w):
    e = S.end_day('AB_V7_B10', j); sd = S.SIGD[('AB_V7_B10', j)]; nm = nm.loc[:e].dropna(); nl = nl.reindex(nm.index).ffill()
    pts = [nm.index[0]] + [d for d in sd if nm.index[0] < d < e] + [e]
    out = []; v = 1.0
    for a, b in zip(pts[:-1], pts[1:]):
        seg = nm.loc[a:b]; sl = nl.loc[a:b]
        val = v * (w * seg / seg.iloc[0] + (1 - w) * sl / sl.iloc[0])
        out.append(val if not out else val.iloc[1:]); v = float(val.iloc[-1])
    return pd.concat(out)
def med(L, f):
    v = [x for x in L if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else None
def summ(per_j):
    """per_j：13 個 dict（巢狀） → 中位數與 p10。"""
    o = {}
    for k in per_j[0]:
        if isinstance(per_j[0][k], dict):
            o[k] = {f: dict(med=med([p[k][f] for p in per_j], f), p10=(float(np.percentile([p[k][f] for p in per_j if p[k][f] is not None], 10)) if all(p[k][f] is not None for p in per_j) else None)) for f in per_j[0][k]}
        else:
            o[k] = dict(med=med([p[k] for p in per_j], k))
    return o
if __name__ == '__main__':
    R = json.loads(RJ.read_text(encoding='utf-8'))
    assert hashlib.sha256(json.dumps(R['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == R['PREREG_sha256']
    assert 'RESULTS_PAPER' in R and 'RESULTS_V7X' not in R, '只跑一次'
    PJ = json.loads((HERE / 'v7x_plans_check.json').read_text(encoding='utf-8'))
    assert all(PJ['checks']['strict_counts_eq_T1'].values()) and PJ['checks']['strict_ends_eq_T1']
    for dv in DVS: assert hashlib.sha256(Path(PJ['out'][dv]['file']).read_bytes()).hexdigest() == PJ['out'][dv]['sha256']
    t0 = time.time()
    PL = {dv: pd.read_csv(PJ['out'][dv]['file'], dtype={'picks': str, 'inserted': str}) for dv in DVS}
    G = {dv: {k: g for k, g in PL[dv].groupby(['variant', 'j', 'seed'])} for dv in DVS}
    jobs, changed = [], {}
    for dv in DVS:
        ch = []
        for k, g in G[dv].items():
            if dv == 'strict' or not (len(g) == len(G['strict'][k]) and (g.picks.values == G['strict'][k].picks.values).all() and (g.exec_idx.values == G['strict'][k].exec_idx.values).all()):
                jobs.append((dv, k[0], int(k[1]), int(k[2]), to_plan(g)));
                if dv != 'strict': ch.append(k)
        changed[dv] = len(ch)
    print('jobs', len(jobs), changed, flush=True)
    NV = {}
    with Pool(4) as P:
        for n, (dv, v, j, s, out) in enumerate(P.imap_unordered(job, jobs, chunksize=1)):
            for ck, nv in out.items(): NV[(dv, v, j, s) + ck] = nv
            if n % 100 == 0: print(n, len(jobs), round(time.time() - t0), flush=True)
    AV = {}
    for dv in DVS:
        for v in VARS:
            for j in range(13):
                for ck in CELLS:
                    L = [NV.get((dv, v, j, s) + ck, NV[('strict', v, j, s) + ck]) for s in range(20)]
                    AV[(dv, v, j) + ck] = S.avg_nav(L)
    pd.DataFrame({'|'.join(map(str, k)): x for k, x in AV.items()}).to_parquet(SCR / 'navs_v7x_o19.parquet')
    TAB, CMP, BL = {}, {}, {}
    for dv in DVS:
        for v in VARS:
            for ck in CELLS:
                nv = [AV[(dv, v, j) + ck] for j in range(13)]; ref = [AV[('strict', v, j) + ck] for j in range(13)]
                T = S.jsumm([S.stats(nv[j], 'AB_V7_B10', j) for j in range(13)])
                T['periods'] = summ([pstats(nv[j], j) for j in range(13)])
                TAB['|'.join((dv, v) + ck)] = T
                CMP['|'.join((dv, v) + ck)] = summ([vsmain(nv[j], ref[j], j) for j in range(13)])
                if dv == 'strict' or dv == 'I3a':
                    for w in (0.8, 0.5):
                        bn = [blend(ref[j], nv[j], j, w) for j in range(13)]
                        BL['|'.join((v,) + ck + (f'leg={dv}', f'w={w}'))] = dict(stats=S.jsumm([S.stats(bn[j], 'AB_V7_B10', j) for j in range(13)]),
                                                                              periods=summ([pstats(bn[j], j) for j in range(13)]), vs_main=summ([vsmain(bn[j], ref[j], j) for j in range(13)]))
    # 重現：strict 100 萬一般 vs 題 1／審查題
    T1 = json.loads((AG / 'o_delist_rev_backfill_allpit/results.json').read_text(encoding='utf-8'))['RESULTS_V7X']['table']
    AUV = json.loads((AG / 'o_disposal_list_pit_audit_pkl18/results.json').read_text(encoding='utf-8'))['RESULTS_V7X']['table']
    F = ('exc_med', 'exc_p10', 't_0050_med', 'exc_Y19on_med', 'maxDD_med', 'CAGR_Y19on_med', 'CAGR_med')
    rep = {}
    for v in VARS:
        for (c, t, m) in CELLS:
            a = TAB[f'strict|{v}|{c}|{t}|{m}']
            b = T1.get(f'{v}|cons_main|{c}|{t}') if m == 'std' else None
            if b is not None: rep[f'T1|{v}|{c}|{t}|{m}'] = max(abs(a[f] - b[f]) for f in F if f in b)
            b2 = AUV.get(f'strict|{v}|cons_main|{c}|{t}|{m}') or AUV.get(f'posthoc|{v}|cons_main|{c}|{t}|{m}')
            if b2 is not None and (f'strict|{v}|cons_main|{c}|{t}|{m}' in AUV):
                rep[f'AUDIT|{v}|{c}|{t}|{m}'] = max(abs(a[f] - b2[f]) for f in F if f in b2)
            bp = AUV.get(f'posthoc|{v}|cons_main|{c}|{t}|{m}')
            if bp is not None: rep[f'AUDIT_posthoc|{v}|{c}|{t}|{m}'] = max(abs(TAB[f'posthoc|{v}|{c}|{t}|{m}'][f] - bp[f]) for f in F if f in bp)
    R = json.loads(RJ.read_text(encoding='utf-8')); assert 'RESULTS_V7X' not in R
    R['RESULTS_V7X'] = dict(utc=pd.Timestamp.now('UTC').isoformat(), runtime_s=round(time.time() - t0, 1), n_jobs=len(jobs), plans_changed=changed,
                            reproduce_maxabs=rep, table=TAB, vs_main=CMP, blends=BL, navs_file=str(SCR / 'navs_v7x_o19.parquet'))
    RJ.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    for k, T in TAB.items(): print(k, {f: round(T[f], 3) for f in ('exc_med', 'exc_p10', 't_0050_med', 'CAGR_Y19on_med', 'exc_Y19on_med')})
    print('reproduce', rep, round(time.time() - t0))
