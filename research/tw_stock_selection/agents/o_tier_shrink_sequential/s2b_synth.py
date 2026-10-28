#!/usr/bin/env python3
"""步驟 1b：合成驗證（PREREG_DESIGN step1_validation.synthetic）。
真實族群殘差（每成員去均值）以平穩 bootstrap L6 同列重抽＋已知真值 α_k＝a_c＋v_k；焦點＝第 114 群結構內合成 α̂ 最大的成員。
對 V1～V7 計算焦點事後均值偏誤與 80% 涵蓋；另報同群隨機成員與任一群隨機非 medoid 成員。
輸出：data/synth_runs.parquet（逐次）、results.json 的 STEP1_SYNTH（彙整）。
用法：python3 -I -B s2b_synth.py
"""
import sys, json, math, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import tlib as T, hmethods as H

FOCUS_C = 114
SCEN = {'P_primary': dict(g='real', sw=0.10, wdist='normal', R=200),
        'S_T2': dict(g='T2', sw=0.10, wdist='normal', R=100),
        'S_w0': dict(g='real', sw=0.0, wdist='normal', R=100),
        'S_w25': dict(g='real', sw=0.25, wdist='normal', R=100),
        'S_t3': dict(g='real', sw=0.10, wdist='t3', R=100)}
_G = {}

def _init():
    X, lab, med = H.load_real()
    E = X.values - X.values.mean(axis=0, keepdims=True)
    fam0 = H.Fam(X.values, lab, med, need_g3=False)
    _G.update(E=E, lab=lab, med=med, g_real=fam0.g2['g'], th=fam0.th)

def draw_a(kind, n, rng, tail=False):
    th = _G['th']
    if kind == 'real':
        g = _G['g_real'].copy()
        if tail:
            c = np.cumsum(g); g = np.where(c >= 0.9, g, 0.0); g /= g.sum()
        return th[rng.choice(len(th), size=n, p=g)] + rng.uniform(-0.01, 0.01, n)
    # T2 混合
    if tail:
        out = []
        while len(out) < n:
            z = draw_a('T2', 4000, rng)
            q = np.quantile(z, 0.9); out += list(z[z >= q][:n - len(out)])
        return np.array(out[:n])
    comp = rng.random(n) < 0.75
    return np.where(comp, rng.normal(-0.6, 0.7, n), rng.normal(0.4, 0.25, n))

def one(args):
    scen, r = args
    sp = SCEN[scen]
    seed = 1000003 * (list(SCEN).index(scen) + 1) + r
    rng = np.random.default_rng(seed)
    E, lab, med = _G['E'], _G['lab'], _G['med']
    labv = np.asarray(lab, int)
    n, K = E.shape
    cl = sorted(med)
    a = dict(zip(cl, draw_a(sp['g'], len(cl), rng)))
    a[FOCUS_C] = float(draw_a(sp['g'], 1, rng, tail=True)[0])
    if sp['wdist'] == 'normal':
        v = rng.normal(0, sp['sw'], K)
    else:
        v = rng.standard_t(3, K) * sp['sw'] / math.sqrt(3.0)
    alpha = np.array([a[c] for c in labv]) + v
    idx = T.sb_index(n, n, 6, 1, seed)[0]
    Xs = E[idx] + alpha[None, :]
    fam = H.Fam(Xs, lab, med, need_g3=True)
    ah = Xs.mean(axis=0)
    ii = np.where(labv == FOCUS_C)[0]
    f_sel = int(ii[np.argmax(ah[ii])])
    f_rnd = int(rng.choice(ii[ii != med[FOCUS_C]]))
    multi = [c for c in cl if (labv == c).sum() >= 2]
    c_any = int(rng.choice(multi)); jj = np.where(labv == c_any)[0]; jj = jj[jj != med[c_any]]
    f_any = int(rng.choice(jj))
    bq_abs = fam.maxbias(FOCUS_C, 'abs', seed=seed)
    bq_diff = fam.maxbias(FOCUS_C, 'diff', seed=seed + 1)
    rows = []
    for tag, k, c in (('sel', f_sel, FOCUS_C), ('rand114', f_rnd, FOCUS_C), ('randany', f_any, c_any)):
        for ver in H.VERSIONS:
            bq = None
            if c == FOCUS_C: bq = bq_abs if ver == 'V3_G2MAX' else (bq_diff if ver == 'V4_HNMAX' else None)
            d = fam.posterior(ver, Xs[:, k], c, is_medoid=(k == med[c]), bq=bq, seed=seed)
            s = T.dens_summary(d, fam.th)
            rows.append(dict(scen=scen, r=r, tag=tag, ver=ver, truth=float(alpha[k]), ahat=float(ah[k]), mean=s['mean'],
                             q10=s['q10'], q90=s['q90'], p_pos=s['p_pos'], sigma_w_hat=fam.sigma_w,
                             bias_abs_mean=float(bq_abs.mean()), bias_diff_mean=float(bq_diff.mean())))
    return rows

def main():
    T.check_block('PREREG_DESIGN', files=False)
    t0 = time.time()
    jobs = [(s, r) for s, sp in SCEN.items() for r in range(sp['R'])]
    rows = []
    with ProcessPoolExecutor(max_workers=4, initializer=_init) as ex:
        for i, rr in enumerate(ex.map(one, jobs, chunksize=2)):
            rows += rr
            if i % 100 == 0: print(f'[{time.time() - t0:5.0f}s] {i}/{len(jobs)}', flush=True)
    D = pd.DataFrame(rows)
    D.to_parquet(T.DATA / 'synth_runs.parquet')
    D['err'] = D['mean'] - D['truth']; D['cov'] = (D.truth >= D.q10) & (D.truth <= D.q90)
    D['raw_err'] = D['ahat'] - D['truth']
    S = D.groupby(['scen', 'tag', 'ver']).agg(bias=('err', 'mean'), bias_se=('err', lambda e: e.std(ddof=1) / math.sqrt(len(e))),
                                              rmse=('err', lambda e: float(np.sqrt((e ** 2).mean()))), coverage80=('cov', 'mean'),
                                              raw_bias=('raw_err', 'mean'), n=('err', 'size'), sigma_w_hat=('sigma_w_hat', 'mean'),
                                              maxbias_abs=('bias_abs_mean', 'mean')).reset_index()
    S['gate_pass'] = (S.bias.abs() <= 0.05) & S.coverage80.between(0.75, 0.85)
    out = {'summary': S.to_dict('records'), 'runtime_s': time.time() - t0, 'utc': T.now_utc(),
           'gate_primary_sel': {v: bool(S[(S.scen == 'P_primary') & (S.tag == 'sel') & (S.ver == v)].gate_pass.iloc[0]) for v in H.VERSIONS}}
    T.save_key('STEP1_SYNTH', out, need=['PREREG_DESIGN'])
    pd.set_option('display.width', 200)
    print(S[S.tag == 'sel'].round(3).to_string())
    print(S[S.tag != 'sel'][S.scen == 'P_primary'].round(3).to_string())
    print(out['gate_primary_sel'])

if __name__ == '__main__':
    main()
