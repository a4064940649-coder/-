#!/usr/bin/env python3
"""登記前煙霧測試（G8P-2a）：只做管線核對，P＝20,000，不輸出任何版本的誤升級或檢定力。
 S1 產生器：邊際 sd≈σ、AR acf1≈ρ、MA acf1≈θ/(1+θ²)、D3／D5 去均值標準化
 S2 B* 一致性：upgrade(b) ⇔ B*_comb ≥ b（隨機 b，各版本）
 S3 逐路徑子集：up_v2a ⊆ up_v2、fail_v2 ⊆ fail_v2a（各版本、各 DGP）
 S4 向量化 vs 純量參考實作（refimpl.combined_state），每 DGP 40 條路徑 × 4 版本
用法：python3 -I -B smoke/smoke0.py"""
import sys, json, math
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import numpy as np
import g8alib as L
import refimpl as R

P = 20_000
SEED = 2026110799
rep = dict(P=P, seed=SEED, S1={}, S2={}, S3={}, S4={})
PAR = dict(sigma_design=L.SIGMA_D, c=L.C_INFL, c_d=L.CD_V2, b_v2=L.B_V2, b_a=2.2)
bad = 0
for k, dgp in enumerate(L.DGPS + ['ARN10', 'MAT3']):
    rng = np.random.default_rng([SEED, k])
    X = L.gen(dgp, 0.0, 8.0, P, 24, rng)
    sd = float(X.std()); a1 = float(np.mean([np.corrcoef(X[:, t], X[:, t + 1])[0, 1] for t in range(23)]))
    exp = L.AR_RHO.get(dgp, None)
    if dgp in L.MA_THETA: th = L.MA_THETA[dgp]; exp = th / (1 + th * th)
    ok1 = abs(sd - 8.0) < 0.12 and (exp is None or abs(a1 - exp) < 0.02)
    rep['S1'][dgp] = dict(sd=round(sd, 3), acf1=round(a1, 4), expected_acf1=exp, ok=bool(ok1))
    bad += not ok1
    Z = L.all_stats(X + 0.25)
    for ver in ('A1', 'A2', 'A3'):
        B = L.bstar_combined(Z, ver)
        for bb in (1.95, 2.1, 2.3):
            o = L.combined(Z, ver, bb)
            mm = int(np.sum(o['up'] != (B >= bb)))
            rep['S2'][f'{dgp}|{ver}|{bb}'] = mm; bad += mm != 0
    for ver in L.VERSIONS:
        o = L.combined(Z, ver, 2.2)
        rep['S3'][f'{dgp}|{ver}'] = dict(viol_up=o['viol_up'], viol_fail=o['viol_fail']); bad += (o['viol_up'] + o['viol_fail']) != 0
        # S4：純量參考
        idx = np.concatenate([np.where(o['up'])[0][:10], np.where(o['fail'])[0][:15], np.where(~o['up'] & ~o['fail'])[0][:15]])
        mis = 0
        for i in idx:
            items = [('valid', float(v)) for v in (X[i] + 0.25)]
            st, s2, sn, lit = R.combined_state(items, dict(PAR), ver)
            vs = 'S1_EXCEPTION_EVIDENCE' if o['up'][i] else ('FORWARD_FAILED' if o['fail'][i] else 'FORWARD_VALIDATING_NO_REJECTION_24')
            mis += (st != vs)
        rep['S4'][f'{dgp}|{ver}'] = dict(n=int(len(idx)), mismatch=int(mis)); bad += mis
rep['n_bad'] = int(bad)
rep['note'] = '本檔不計算、不輸出任何版本的誤升級或檢定力（只輸出不一致計數與產生器矩）'
print(json.dumps(rep, ensure_ascii=False))
