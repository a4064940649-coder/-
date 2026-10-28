#!/usr/bin/env python3
"""引擎移植檢查（PREREG 前；不涉及本題校準模型的任何策略結果）：
  (1) impact='O5'、四項摩擦全關：逐日重現 o5 navs_main.parquet 的主格（seed 0～3；30萬 GROW A、300萬 FIX A、1000萬 GROW A）。
  (2) impact='O5'、四項摩擦全開（＝o5 成本審查 ALL）：40 seed 平均 t 對照 o5 成本審查報告值（30萬 GROW 1.94、30萬 FIX 2.06、100萬 FIX 2.05、300萬 FIX 1.78）。
輸出 data/check_engine.json。用法：python3 -I -B check_engine.py
"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
import ecore6 as E
SPEC = json.load(open(E.DATA / 'model_calib.json'))
O5N = HERE.parent / 'o_exec_capacity_tiers' / 'navs_main.parquet'
_W = {}
def _init(): _W['D'] = E.Data()
def job(a):
    tier, ver, seed, fr = a
    D = _W['D']; M = E.Model(SPEC, impact='O5')
    nav, acc, _ = E.Sim(D, M, E.FRICTIONS if fr else {}).run(D.plans[('AB', seed)], D.t0, D.t1, E.TIERS[tier], ver, 'A')
    return a, nav.to_numpy()
if __name__ == '__main__':
    T0 = time.time(); out = {}
    ref = pd.read_parquet(O5N)
    jobs1 = [(ti, ve, s, False) for ti, ve in (('30萬', 'GROW'), ('300萬', 'FIX'), ('1000萬', 'GROW')) for s in range(4)]
    jobs2 = [(ti, ve, s, True) for ti, ve in (('30萬', 'GROW'), ('30萬', 'FIX'), ('100萬', 'FIX'), ('300萬', 'FIX')) for s in range(40)]
    with ProcessPoolExecutor(4, initializer=_init) as ex: R = dict(ex.map(job, jobs1 + jobs2, chunksize=2))
    D = E.Data(); idx = D.IDX[D.t0:D.t1 + 1]
    out['repro_o5_main'] = {f'{a[0]}|{a[1]}|s{a[2]:02d}': float(np.max(np.abs(R[a] / ref[f'{a[0]}|{a[1]}|A|s{a[2]:02d}'].to_numpy() - 1))) for a in jobs1}
    E50 = D.E50.reindex(idx)
    for ti, ve in (('30萬', 'GROW'), ('30萬', 'FIX'), ('100萬', 'FIX'), ('300萬', 'FIX')):
        Rm = pd.DataFrame({s: pd.Series(R[(ti, ve, s, True)], index=idx).pct_change() for s in range(40)}).mean(axis=1).fillna(0)
        nav = (1 + Rm).cumprod()
        out[f'ALL_frictions|{ti}|{ve}'] = dict(CAGR=E.cagr_mdd(nav, E.A_, E.B_)[0], t=E.tstat(E.excess(nav, E50, D.sig_days)))
    out['o5_review_reported_t'] = {'30萬|GROW': 1.94, '30萬|FIX': 2.06, '100萬|FIX': 2.05, '300萬|FIX': 1.78}
    out['runtime_s'] = time.time() - T0
    json.dump(out, open(E.DATA / 'check_engine.json', 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))
