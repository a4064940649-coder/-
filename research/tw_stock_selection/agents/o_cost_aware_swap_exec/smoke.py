#!/usr/bin/env python3
"""程式煙霧測試（只檢查各開關能跑、計數合理；不輸出淨值）：j=0、REAL_opt、seed 0、100 萬一般與成本全關、30 萬，到 2007-12-31。"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); import cas_lib as CL; sys.path.pop(0)
E = CL.engine(); D = E['D']
C = CL.load_cands('REAL_opt', 0); pl, cd = CL.to_plan(C[C.j == 0], D)
t1 = CL.t_end_idx(D, '2007-12-31'); out = {}
for nm, o in [('CA80', {'ca': 0.8}), ('MT', {'mt': 1.0}), ('NR', {'nr': True}), ('DC5C', {'dc': 5}), ('DC5E', {'dc': 5, 'dcmode': 'E'}), ('A20', {'a20': True}),
              ('CB', {'ca': 0.9, 'mt': 1.0, 'nr': True, 'dc': 5, 'dcmode': 'E'})]:
    for tier, sw in (('100萬', None), ('100萬', CL.OFFC), ('30萬', None)):
        s = E['CASim'](pl, CL.cfg_for(E, tier, 'normal'), settle=E['ST'], cands=cd, opt=o, sw=sw, track=True); nav = s.run(t1=t1)
        assert np.isfinite(nav.values).all() and (nav.values > 0).all()
        out[f'{nm}|{tier}|{"off" if sw else "on"}'] = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in s.st.items()}
for k, v in out.items(): print(k, v)
