#!/usr/bin/env python3
"""凍結後：V7X 可執行引擎（K5T、f1～f4 正式設定、0.25% 不交易帶）跑選定版、預設版與合成基準 B，FORMAL 2014-01～2026-10，13 個 j。
級距：30 萬 FIX、100 萬 FIX、100 萬 GROW、300 萬 GROW、1,000 萬 GROW；一般成本與壓力成本（V7X stress：每邊再加 0.5% 等）。
對 0050 含息（同窗）報 j 中位數與 p10；2019 起與 0050 並列年化。先斷言 PREREG。"""
import sys, json, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lclib as X
import os
DRY = os.environ.get('LC_DRY') == '1'     # 凍結前的程式除錯：只用挑選期、輸出到 scratch，不碰 results.json
if DRY:
    _it = json.loads((HERE / 'iterations.json').read_text(encoding='utf-8'))
    PRE = dict(chosen=dict(cfg=_it['chosen']['cfg'], params=_it['versions'][_it['chosen']['version']]['params']), default=dict(cfg='COMP|k5|A30|M'),
               placebo_persistence=json.loads((HERE / 'explore.json').read_text(encoding='utf-8'))['persistence_jmed'], code_sha256={}, data_sha256={})
    RES = dict(PREREG=PRE)
    RESF = X.SCR / f'dry_{Path(__file__).stem}.json'
    if RESF.exists(): RES.update(json.loads(RESF.read_text(encoding='utf-8')))
    for f in ('dry_s6_oos.json',):
        if (X.SCR / f).exists(): RES.update({k: v for k, v in json.loads((X.SCR / f).read_text(encoding='utf-8')).items() if k == 'OOS'})
else:
    RES = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
    PRE = RES['PREREG']
    RESF = HERE / 'results.json'
    assert X.canon(PRE) == RES['PREREG_sha256'], 'PREREG 雜湊不符'
    for f, h in PRE['code_sha256'].items(): assert X.sha(HERE / f) == h, f'{f} 與 PREREG 不符'
import lcrun as R, lcexec as E
assert list(E.D.COLS) == list(X.COLS)
SEL = PRE['chosen']['cfg']; DEF = PRE['default']['cfg']; P = PRE['chosen']['params']
CELLS = [('30萬', 'FIX'), ('100萬', 'FIX'), ('100萬', 'GROW'), ('300萬', 'GROW'), ('1000萬', 'GROW')]
def cfgt(nm):
    a, k, A, f = nm.split('|'); return (a, int(k[1:]), int(A[1:]) / 100, f)
R.init()
SCHED = {}
for j in range(13):
    B = R.build(j, '2014-01', '2016-12' if DRY else '2026-10', '2017-01' if DRY else '2026-11', P=P, grid=[cfgt(SEL), cfgt(DEF)])
    days = B['days']
    SCHED[j] = dict(days=days, e50=X.bench_win(days),
                    B=[(d,) + B['Bplan'][d] for d in days[:-1]],
                    **{nm: [(d,) + B['plans'][nm][d] for d in days[:-1] if B['plans'][nm][d] is not None] for nm in (SEL, DEF)})
def job(a):
    j, nm, tier, mode, cost = a
    S = SCHED[j]
    nav, acc = E.run_weighted(S[nm], tier, mode, cost, t1=E.D.TPOS[S['days'][-1]])
    r = E.win_returns(nav, S['days'])
    return a, r, nav, {k: float(v) for k, v in acc.items() if isinstance(v, (int, float))}
if __name__ == '__main__':
    t0 = time.time()
    jobs = [(j, nm, t, m, c) for j in range(13) for nm in (SEL, DEF, 'B') for (t, m) in CELLS for c in ('normal', 'stress')]
    with Pool(4) as p: out = p.map(job, jobs, chunksize=4)
    print('exec runs', len(out), round(time.time() - t0), flush=True)
    W = {a: (r, nav, acc) for a, r, nav, acc in out}
    def msk(idx, a, b='2026-12-31'):
        if DRY: a, b = '2014-01-01', '2016-12-31'
        return (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))
    T = {}
    for nm in (SEL, DEF, 'B'):
        for (t, m) in CELLS:
            for c in ('normal', 'stress'):
                row = {}
                for per, a, b in (('VAL', '2017-01-01', '2026-12-31'), ('FORMAL', '2014-01-01', '2026-12-31'), ('Y19', '2019-01-01', '2026-12-31'),
                                  ('SEG2', '2014-01-01', '2018-12-31')):
                    st = []; stB = []
                    for j in range(13):
                        r = W[(j, nm, t, m, c)][0]; e = SCHED[j]['e50']; x = (r - e)[msk(r.index, a, b)]
                        s = X.stats(x); s['cagr'] = X.cagr(r[msk(r.index, a, b)]); s['cagr0050'] = X.cagr(e[msk(e.index, a, b)])
                        st.append(s)
                        if nm != 'B':
                            rb = W[(j, 'B', t, m, c)][0]; stB.append(X.stats((r - rb)[msk(r.index, a, b)]))
                    row[per] = dict(vs0050_mean_jmed=float(np.median([s['mean_pp'] for s in st])), vs0050_mean_p10=float(np.percentile([s['mean_pp'] for s in st], 10)),
                                    vs0050_t_jmed=float(np.median([s['t'] for s in st])), vs0050_t_p10=float(np.percentile([s['t'] for s in st], 10)),
                                    vs0050_IR_jmed=float(np.median([s['IR'] for s in st])), TE_jmed=float(np.median([s['TE'] for s in st])),
                                    cagr_jmed=float(np.median([s['cagr'] for s in st])), cagr0050=float(np.median([s['cagr0050'] for s in st])))
                    if stB:
                        row[per].update(vsB_mean_jmed=float(np.median([s['mean_pp'] for s in stB])), vsB_mean_p10=float(np.percentile([s['mean_pp'] for s in stB], 10)),
                                        vsB_t_jmed=float(np.median([s['t'] for s in stB])), vsB_IR_jmed=float(np.median([s['IR'] for s in stB])))
                accs = [W[(j, nm, t, m, c)][2] for j in range(13)]
                tier = E.TIERS[t]; n = len(SCHED[0]['days']) - 1
                row['cost_pp_per_month_jmed'] = float(np.median([(a['tax'] + a['fee'] + a['impact'] + a['odd_spread'] + a.get('disp_cost', 0)) / tier / n * 100 for a in accs]))
                row['orders_per_month_jmed'] = float(np.median([a['n_orders'] / n for a in accs]))
                rs = [W[(j, nm, t, m, c)][1].pct_change() for j in range(13)]
                nv = (1 + pd.concat(rs, axis=1).mean(axis=1).fillna(0)).cumprod()
                v19 = nv[nv.index >= '2019-01-01']; row['mdd_from2019'] = X.mdd_nav(v19); row['mdd_from2014'] = X.mdd_nav(nv)
                T[f'{nm}|{t}|{m}|{c}'] = row
    e = X.L.E50; e19 = e[e.index >= '2019-01-01']
    RES = json.loads(RESF.read_text(encoding='utf-8')) if RESF.exists() else {}
    RES['EXEC'] = dict(cells=T, mdd_0050_from2019=X.mdd_nav(e19), band=0.0025, note='V7X K5T（x_exec.py 原碼）＋權重子類 lcexec.WSim；stress＝V7X cost=stress')
    RESF.write_text(X.dumps(RES), encoding='utf-8')
    for k, v in T.items():
        if k.startswith(SEL) or k.startswith('B|100萬|GROW'):
            print(k, {p: {a: round(b, 3) for a, b in v[p].items() if a in ('vs0050_mean_jmed', 'vs0050_mean_p10', 'vs0050_t_jmed', 'vsB_mean_jmed', 'vsB_t_jmed', 'cagr_jmed', 'cagr0050')} for p in ('VAL', 'Y19')},
                  'cost', round(v['cost_pp_per_month_jmed'], 3))
    print('time', round(time.time() - t0))
