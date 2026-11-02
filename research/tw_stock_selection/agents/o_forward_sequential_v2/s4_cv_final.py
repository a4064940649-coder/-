#!/usr/bin/env python3
"""步驟 4（樣本外，PREREG 之後）：真實資料 CV 選版。
(a) CV_FINAL：全 245 窗，時間切半 122／123 兩方向＋滾動起點 {84,120,157,192}×48 窗 → V*、s、k（前瞻先驗用）。
(b) CV_PRE2024：只用錨定月 < 2024-01 的窗，時間切半＋滾動起點 {84,120,157,192} 中 T0＋48 ≤ n 者 → 保留期偽前瞻（2024-01 起）的先驗。
挑選期的 CV_SEL（s2b）＝驗證期偽前瞻（2019-05 起）的先驗。
用法：python3 -I -B s4_cv_final.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import fslib as F, cvlib as CV
from s2b_cv_sel import run, evaluate


def main():
    F.check_prereg('PREREG')
    X, lab, med, anc = F.family()
    allm = np.ones(len(anc), bool)
    D, dens, rt, sp = run(len(anc), True, (84, 120, 157, 192))
    D.to_parquet(F.DATA / 'cv_final.parquet'); np.save(F.DATA / 'cv_final_dens.npy', dens)
    out = evaluate(D, dens, allm, 'CV_FINAL'); out.update(splits=sp, runtime_s=rt, n_runs=len(D), utc=F.now_utc())
    F.save_key('CV_FINAL', out, need=['PREREG'])
    print('FINAL', pd.DataFrame(out['summary_sel']).round(3).to_string())
    print({k: out[k] for k in ('selected', 'tie_candidates', 's', 's_by_scheme', 'k', 'cov80_at_s_k1', 'cov80_at_s_k')})
    pre = F.per_mask(anc, None, pd.Period('2023-12', 'M'))
    n = int(pre.sum())
    D2, dens2, rt2, sp2 = run(n, True, tuple(o for o in (84, 120, 157, 192) if o + 48 <= n))
    D2.to_parquet(F.DATA / 'cv_pre2024.parquet'); np.save(F.DATA / 'cv_pre2024_dens.npy', dens2)
    out2 = evaluate(D2, dens2, pre, 'CV_PRE2024'); out2.update(splits=sp2, runtime_s=rt2, n_runs=len(D2), n_windows=n, utc=F.now_utc())
    F.save_key('CV_PRE2024', out2, need=['PREREG'])
    print('PRE2024', {k: out2[k] for k in ('selected', 'tie_candidates', 's', 's_by_scheme', 'k', 'cov80_at_s_k1', 'cov80_at_s_k')})


if __name__ == '__main__':
    main()
