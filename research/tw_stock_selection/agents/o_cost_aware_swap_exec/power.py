#!/usr/bin/env python3
"""檢定力分析（PREREG 之後、任何變體報酬之前）：挑選期、100 萬一般、seed 0、REAL_opt、13 j。
安慰劑 PR20（新進名額 20% 隨機改成第 11～20 名的下一名新候選）與 PS90（再平衡單與零股尾數以 90% 機率隨機不下；
90% 取自 o20 成交日誌 2006～2014 100 萬 seed 0：續抱再平衡單名目 < F* 的比例 0.906、零股尾數 0.762）各 200 次。
寫入 results.json['PREREG_POWER']（含 sha256），逐次結果存 data/power_placebo.csv.gz。
python3 -I -B power.py"""
import sys, json, time, math
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); import cas_lib as CL; sys.path.pop(0)
RES = HERE / 'results.json'
R0 = json.loads(RES.read_text(encoding='utf-8')); assert CL.canon(R0['PREREG']) == R0['PREREG_sha256']; assert 'PREREG_POWER' not in R0
SEL_END = '2014-12-31'; NREP = 200
_G = {}
def init():
    E = CL.engine(); _G['E'] = E; D = E['D']
    C = CL.load_cands('REAL_opt', 0)
    _G['P'] = {j: CL.to_plan(g, D) for j, g in C.groupby('j')}
    _G['t1'] = CL.t_end_idx(D, SEL_END)
def job(a):
    kind, rep, j = a
    E = _G['E']; pl, cands = _G['P'][j]
    cfg = CL.cfg_for(E, '100萬', 'normal')
    opt = {} if kind == 'BASE' else ({'pr': 0.2} if kind == 'PR20' else {'ps': 0.9})
    sim = E['CASim'](pl, dict(cfg), settle=E['ST'], cands=cands, opt=opt, rng_seed=rep * 1000 + j)
    nav = sim.run(t1=_G['t1'])
    return kind, rep, j, nav, sim.st
if __name__ == '__main__':
    t0 = time.time()
    jobs = [('BASE', 0, j) for j in range(13)] + [(k, r, j) for k in ('PR20', 'PS90') for r in range(NREP) for j in range(13)]
    NV = {}; ST = {}
    sys.path.insert(0, str(HERE)); import run_lib as RL; sys.path.pop(0)
    RL.job = job
    if True:
        for n, (k, r, j, nav, st) in enumerate(RL._iter_results(jobs, 2, init)):
            NV[(k, r, j)] = nav; ST[(k, r, j)] = st
            if n % 500 == 0: print(n, len(jobs), round(time.time() - t0), flush=True)
    base = {j: NV[('BASE', 0, j)] for j in range(13)}
    bm = {j: CL.monthly(base[j]) for j in range(13)}
    rows = []
    for k in ('PR20', 'PS90'):
        for r in range(NREP):
            dm = pd.concat([CL.monthly(NV[(k, r, j)]) - bm[j] for j in range(13)], axis=1)
            dm = dm[dm.index >= pd.Period('2006-05', 'M')]
            avg = dm.mean(axis=1).dropna()
            cd = [CL.cagr(NV[(k, r, j)], NV[(k, r, j)].index[0], SEL_END) - CL.cagr(base[j], base[j].index[0], SEL_END) for j in range(13)]
            rows.append(dict(kind=k, rep=r, sd_avg_m=float(avg.std(ddof=1)), mean_avg_m=float(avg.mean()), n_m=int(len(avg)),
                             sd_single_m=float(np.nanmean(dm.std(ddof=1).values)), jmed_cagr_diff=float(np.median(cd)),
                             n_pr=int(sum(ST[(k, r, j)]['n_pr'] for j in range(13))), n_new=int(sum(ST[(k, r, j)]['n_new'] for j in range(13)))))
    T = pd.DataFrame(rows); T.to_csv(HERE / 'data/power_placebo.csv.gz', index=False, compression='gzip')
    Z = 1.959964 + 0.841621
    out = dict(written_utc=pd.Timestamp.now('UTC').isoformat(), runtime_s=round(time.time() - t0, 1), n_rep=NREP, design=__doc__)
    nseg = dict(selection=104, validation=72, holdout=70)
    for k in ('PR20', 'PS90'):
        t = T[T.kind == k]
        sd = float(t.sd_avg_m.median()); sd1 = float(t.sd_single_m.median()); n = int(t.n_m.median())
        o = dict(sigma_d_avg13_monthly_pp=sd * 100, sigma_d_single_j_monthly_pp=sd1 * 100, n_months_sel=n,
                 placebo_jmed_cagr_diff_mean=float(t.jmed_cagr_diff.mean()), placebo_jmed_cagr_diff_sd=float(t.jmed_cagr_diff.std(ddof=1)),
                 placebo_jmed_cagr_diff_p05_p95=[float(t.jmed_cagr_diff.quantile(.05)), float(t.jmed_cagr_diff.quantile(.95))],
                 MDE_from_placebo_distribution_pp=float(Z * t.jmed_cagr_diff.std(ddof=1)), replaced_share=float(t.n_pr.sum() / max(t.n_new.sum(), 1)))
        for seg, nm in nseg.items():
            o[f'MDE_avg13_{seg}_pp'] = float(Z * sd / math.sqrt(nm) * 12 * 100)
            o[f'MDE_neff_{seg}_pp'] = float(Z * sd1 / math.sqrt(13 * nm / 4) * 12 * 100)
        out[k] = o
    out['MDE_primary_pp'] = dict(seg=('取 PR20／PS90 兩者較大的 avg13 口徑'),
                                 selection=max(out['PR20']['MDE_avg13_selection_pp'], out['PS90']['MDE_avg13_selection_pp']),
                                 validation=max(out['PR20']['MDE_avg13_validation_pp'], out['PS90']['MDE_avg13_validation_pp']),
                                 holdout=max(out['PR20']['MDE_avg13_holdout_pp'], out['PS90']['MDE_avg13_holdout_pp']))
    out['rule'] = ('選擇規則 (a) 的「≥ MDE 的一半」使用 MDE_primary_pp.selection。若 MDE 大於 P1～P7 預期效果（0.3～0.5pp），'
                   '結果只能是「改善」或「無資訊」，不能當成否定證據。')
    r = json.loads(RES.read_text(encoding='utf-8'))
    r['PREREG_POWER'] = out; r['PREREG_POWER_sha256'] = CL.canon(out)
    RES.write_text(CL.dumps(r), encoding='utf-8')
    print(json.dumps({k: out[k] for k in ('PR20', 'PS90', 'MDE_primary_pp')}, ensure_ascii=False, indent=1))
