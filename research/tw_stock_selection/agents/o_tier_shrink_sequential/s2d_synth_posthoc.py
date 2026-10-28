#!/usr/bin/env python3
"""步驟 1d（事後診斷，不改選版）：登記的合成主情境把焦點群真值限定在 g_true 右尾 10%，
這會讓任何「以資料為條件才不偏」的貝氏事後在「以真值為條件」時必然往下偏（同群隨機成員 V1／V2 偏誤都約 −0.14 即為證據）。
本檔用兩個沒有這個問題的情境重跑：
  D_uncond：焦點群真值 a_114~g_true（不限右尾），焦點＝群內 α̂ 最大者；
  D_select：真值全部 iid；先在 m_c ≥ 30 的 10 個大群中挑 medoid α̂ 最大的群，再挑群內 α̂ 最大的成員（兩層都以資料挑選）。
其餘與 s2b_synth.py 相同。輸出 results.json 的 STEP1_SYNTH_POSTHOC。
用法：python3 -I -B s2d_synth_posthoc.py
"""
import sys, json, math, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import tlib as T, hmethods as H, s2b_synth as S2

SC = {'D_uncond': 200, 'D_select': 200}

def one(args):
    scen, r = args
    seed = 7000003 * (list(SC).index(scen) + 1) + r
    rng = np.random.default_rng(seed)
    E, lab, med = S2._G['E'], S2._G['lab'], S2._G['med']
    labv = np.asarray(lab, int); n, K = E.shape
    cl = sorted(med)
    a = dict(zip(cl, S2.draw_a('real', len(cl), rng)))
    v = rng.normal(0, 0.10, K)
    alpha = np.array([a[c] for c in labv]) + v
    idx = T.sb_index(n, n, 6, 1, seed)[0]
    Xs = E[idx] + alpha[None, :]
    fam = H.Fam(Xs, lab, med, need_g3=True)
    ah = Xs.mean(axis=0)
    if scen == 'D_uncond':
        fc = S2.FOCUS_C
    else:
        big = [c for c in cl if (labv == c).sum() >= 30]
        fc = max(big, key=lambda c: ah[med[c]])
    ii = np.where(labv == fc)[0]
    k = int(ii[np.argmax(ah[ii])])
    rows = []
    for ver in H.VERSIONS:
        d = fam.posterior(ver, Xs[:, k], fc, is_medoid=(k == med[fc]), seed=seed)
        s = T.dens_summary(d, fam.th)
        rows.append(dict(scen=scen, r=r, ver=ver, cluster=int(fc), truth=float(alpha[k]), ahat=float(ah[k]), mean=s['mean'], q10=s['q10'], q90=s['q90']))
    return rows

def main():
    T.check_block('PREREG_DESIGN', files=False)
    jobs = [(s, r) for s, R in SC.items() for r in range(R)]
    rows = []; t0 = time.time()
    with ProcessPoolExecutor(max_workers=4, initializer=S2._init) as ex:
        for rr in ex.map(one, jobs, chunksize=2): rows += rr
    D = pd.DataFrame(rows)
    D['err'] = D['mean'] - D.truth; D['cov'] = (D.truth >= D.q10) & (D.truth <= D.q90); D['raw_err'] = D.ahat - D.truth
    Sm = D.groupby(['scen', 'ver']).agg(bias=('err', 'mean'), bias_se=('err', lambda e: e.std(ddof=1) / math.sqrt(len(e))), coverage80=('cov', 'mean'),
                                        raw_bias=('raw_err', 'mean'), rmse=('err', lambda e: float(np.sqrt((e ** 2).mean()))), n=('err', 'size')).reset_index()
    Sm['would_pass_gate'] = (Sm.bias.abs() <= 0.05) & Sm.coverage80.between(0.75, 0.85)
    out = {'summary': Sm.to_dict('records'), 'note': '事後診斷（看過 STEP1_SYNTH 後才設計），不改變登記的選版規則與結果。',
           'focus_cluster_counts': {f'{a}|{b}': int(n) for (a, b), n in D[D.ver == 'V1_o6'].groupby('scen').cluster.value_counts().items()}, 'runtime_s': time.time() - t0, 'utc': T.now_utc()}
    T.save_key('STEP1_SYNTH_POSTHOC', out, need=['PREREG_DESIGN'])
    pd.set_option('display.width', 200); print(Sm.round(3).to_string())

if __name__ == '__main__':
    main()
