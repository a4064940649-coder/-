#!/usr/bin/env python3
"""事後對帳診斷（PREREG 之後；不改凍結規格，只用來追分歧原因）。
R0 全期在 30 萬格超過 0.10pp。逐日追到兩個來源：
 (i) 零股量上限在 V60_{t−1} 缺值時的處理：V7X 依登錄解讀當 0（零股永遠不能成交，例：6670 在 2025-04 起 V60 缺值、118 股零股賣不掉，連帶 1316 買單因現金不足一路順延）；
     原版很可能是「缺值＝不設上限」（NaN 比較為假）。本診斷以子類別改成缺值不設上限，只跑原設定 30 萬／100 萬 × FIX／GROW × K5T／TICK（一般成本）。
 (ii) 處置期賣出遇收盤跌停（收盤−一跳低於跌停價，例：2108 2021-04-13）。只記錄，不改。"""
import sys, json, math
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import x_data as D, x_exec as X, x_stats as S, x_run as XR

R = json.loads((HERE / 'results.json').read_text()); BASE = R['PREREG']['orig_cfg']


class DiagSim(X.Sim):
    def oddcap(self, t, j):
        if t >= X.I_ODD and not np.isfinite(D.V60p[t, j]): return None
        return X.Sim.oddcap(self, t, j)


CL = [(m, 'normal', t, v) for m in ('K5T', 'TICK') for t in ('30萬', '100萬') for v in ('FIX', 'GROW')]


def job(key_pl):
    key, pl = key_pl; out = {}
    for (m, c, t, v) in CL:
        cfg = dict(tier=XR.TIERS[t], mode=v, model=m, cost=c); cfg.update(BASE)
        out[(m, c, t, v)] = DiagSim(pl, cfg).run()
    return key, out


if __name__ == '__main__':
    P = D.plans('AB_V7_B10', js=list(range(13)), seeds=list(range(20)))
    navs = {}
    with Pool(4) as pool:
        for key, out in pool.imap_unordered(job, sorted(P.items(), key=lambda kv: kv[0])):
            for ck, n in out.items(): navs.setdefault((key[0],) + ck, []).append(n)
    df = pd.DataFrame({f'B10|J{j}|{m}|{c}|{t}|{v}': S.avg_nav(L) for (j, m, c, t, v), L in navs.items()}).sort_index(axis=1)
    df.to_parquet(HERE / 'out' / 'navs_diag_nanv60.parquet')
    NX = pd.read_parquet(D.O7 / 'navs_exec7.parquet'); R0 = pd.read_parquet(HERE / 'out' / 'navs_oos_R0.parquet')
    res = {}
    for (m, c, t, v) in CL:
        k = f'{m}|{c}|{t}|{v}'
        med = lambda F: float(np.median([S.stats(F[f'B10|J{j}|{k}'], 'AB_V7_B10', j)['CAGR'] for j in range(13)]))
        res[k] = dict(orig=med(NX), v7x_R0=med(R0), diag_nan_nocap=med(df))
        res[k]['diff_R0'] = res[k]['v7x_R0'] - res[k]['orig']; res[k]['diff_diag'] = res[k]['diag_nan_nocap'] - res[k]['orig']
    (HERE / 'out' / 'diag_nanv60.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))
