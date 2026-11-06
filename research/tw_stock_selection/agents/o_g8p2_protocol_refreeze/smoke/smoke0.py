#!/usr/bin/env python3
"""登記前煙霧測試（P ≤ 20,000；PREREG_DESIGN 之前執行，程式、輸出與 sha256 一併保存在 smoke/）。
只檢查管線：(1) g8lib2 的 G8P1 實作與 G8P-1 g8lib.run_rule＋oc 合併程序逐路徑一致；(2) bstar 與 outcomes 等價；
(3) V2 的 t→z 轉換單調與對稱；(4) 計時。不印任何 V1～V4 的誤升級或檢定力數字（只印一致性與 G8P-1 既有公開數字）。
用法：python3 -I -B smoke/smoke0.py > smoke/smoke0.out"""
import sys, time, math, json
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import numpy as np
import g8lib2 as L
sys.path.insert(0, str(L.G8P1_DIR))
import g8lib as G1
sys.path.pop(0)

t0 = time.time()
P = 20_000
rng = np.random.default_rng(1)
out = {}
# (1) G8P1 原樣
X = L.gen('D1', 0.0, L.SIGMA_D, P, 24, rng)
par = dict(sigma=L.SIGMA_D, c=L.C_INFL, c_d=L.CD_G8P1, b=L.B_G8P1, up_looks=[12, 24], dn_every=6)
u, d = G1.run_rule('R-FIX', X, par, Hdec=24, refit=True)
g8 = X[:, :12].sum(axis=1) > 0
uu = np.where(g8, u, 0)
ref_up = (uu > 0) & (uu <= 24)
Z = L.zstats(X, dict(stat='FIX_REFIT'))
o = L.outcomes(Z, L.B_G8P1, L.CD_G8P1)
out['g8p1_up_mismatch'] = int((o['up'] != ref_up).sum())
out['g8p1_dn_mismatch'] = int((o['dn_seq'] != ((d > 0) & (d <= 24))).sum())
out['g8p1_up_rate_sigma_d_P20000'] = float(o['up'].mean())
# (2) bstar ⇔ outcomes（V1、V2、V3 與任意 b、c_d 組；只比較布林，不印比率）
mm = 0
for ver in (dict(stat='MAXSD'), dict(stat='HAC_T'), dict(stat='MAXSD', delta=0.2), dict(stat='FIX_REFIT')):
    Xv = L.gen('D2', 0.3, 7.5, P, 24, rng)
    Zv = L.zstats(Xv, ver)
    for b, cd in ((1.6, 0.9), (1.9, 1.3), (2.2, 0.5)):
        B = L.bstar(Zv, cd); ov = L.outcomes(Zv, b, cd)
        mm += int(((B >= b) != ov['up']).sum())
out['bstar_vs_outcomes_mismatch'] = mm
# (3) t→z
t = np.array([-50, -3, -1, 0, 1, 3, 50.0])
z = L.t_to_z(t, 11)
out['t_to_z_monotone'] = bool(np.all(np.diff(z) > 0)); out['t_to_z_symmetric'] = bool(np.allclose(z, -z[::-1]))
out['t_to_z_at_0'] = float(z[3])
# (4) 計時：D3 生成 P＝20,000
t1 = time.time(); _ = L.gen('D3', 0.0, 8.0, P, 24, rng); out['sec_gen_D3_20000'] = round(time.time() - t1, 3)
out['sec_total'] = round(time.time() - t0, 2)
print(json.dumps(out, ensure_ascii=False, indent=1))
