#!/usr/bin/env python3
"""包裝器自我檢查（不算任何變體報酬）：
(1) CASim 全部開關關閉時，淨值與 cz_v7x.CSim 逐位相同（9 個模擬：3 放回 × j∈{0,6,12}；100 萬一般、30 萬一般、100 萬壓力各抽）。
(2) 名單層重建（官方序列＋開盤漲停跳過）與官方 picks 逐列相同（base_mismatch＝0）。
(3) 單次模擬耗時（到 2014-12-31 與全期）。
python3 -I -B selftest.py"""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); import cas_lib as CL; sys.path.pop(0)
E = CL.engine(); D = E['D']
C = CL.load_cands()
G = {k: g for k, g in C.groupby(['variant', 'j', 'seed'])}
out = []
t14 = CL.t_end_idx(D, '2014-12-31')
for (v, j, cell) in [('REAL_opt', 0, ('normal', '100萬')), ('REAL_p90', 6, ('normal', '100萬')), ('REAL_carry', 12, ('normal', '100萬')),
                     ('REAL_opt', 6, ('normal', '30萬')), ('REAL_p90', 12, ('normal', '30萬')), ('REAL_carry', 0, ('normal', '30萬')),
                     ('REAL_opt', 12, ('stress', '100萬')), ('REAL_p90', 0, ('stress', '100萬')), ('REAL_carry', 6, ('normal', '1000萬'))]:
    pl, cands = CL.to_plan(G[(v, j, 0)], D)
    cfg = CL.cfg_for(E, cell[1], cell[0])
    t0 = time.time(); ref = E['CV'].CSim(pl, dict(cfg), settle=E['ST'], cmode='consideration').run(); tr = time.time() - t0
    t0 = time.time(); sim = E['CASim'](pl, dict(cfg), settle=E['ST'], cands=cands, track=True); nav = sim.run(); tm = time.time() - t0
    t0 = time.time(); nav14 = E['CASim'](pl, dict(cfg), settle=E['ST'], cands=cands).run(t1=t14); t14s = time.time() - t0
    d = float(np.nanmax(np.abs(ref.values - nav.values))); d14 = float(np.nanmax(np.abs(ref.values[:len(nav14)] - nav14.values)))
    out.append(dict(v=v, j=j, cell=cell, maxabs=d, maxabs_t14=d14, base_mismatch=sim.st.get('base_mismatch', -1), n_reb=len(pl),
                    sec_ref=round(tr, 2), sec_cas=round(tm, 2), sec_cas_t14=round(t14s, 2)))
    print(out[-1], flush=True)
assert all(o['maxabs'] == 0.0 and o['maxabs_t14'] == 0.0 and o['base_mismatch'] == 0 for o in out), out
(HERE / 'selftest.json').write_text(CL.dumps(dict(utc=pd.Timestamp.now('UTC').isoformat(), engine_sha=CL.check_engine(), rows=out)), encoding='utf-8')
print('OK')
