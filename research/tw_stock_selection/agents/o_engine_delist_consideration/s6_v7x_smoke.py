#!/usr/bin/env python3
"""V7X 煙霧測試（凍結前；不算 consideration 的登記數字）：
(a) CSim(cmode='orig') 與 x_exec.Sim 逐日差＝0（J0 seed0、J7 seed3 × K5T 一般／壓力 × 100 萬 GROW，R4 設定）；
(b) 合成單一股票名單的 consideration 結算對帳（現金 3474、換股＋現金 2847、股份轉換 2311）：結算值與公開對價差 < 0.5%。
python3 -I -B s6_v7x_smoke.py → smoke_v7x.json"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cz_v7x as CV
import cz_engine as CE
D, X = CV.D, CV.X
R = json.loads((CV.V7X / 'results.json').read_text())
BASE = dict(R['PREREG']['orig_cfg']); BASE.update(dict(f1=True, f2=True, f3=0.5, f4=True))
E = pd.read_csv(HERE / 'events_v2.csv', dtype={'code': str, 'successor': str})
ST = CE.build_settlement(E, 'main', TR=D.TR, IDX=D.IDX, Cn=D.C, CI=D.CI, On=D.O)
P = D.plans('AB_V7_B10', js=[0, 7], seeds=[0, 3])
out = dict(a={}, b={})
for key in [(0, 0), (7, 3)]:
    pl = P[key]
    for cost in ('normal', 'stress'):
        cfg = dict(tier=1e6, mode='GROW', model='K5T', cost=cost); cfg.update(BASE)
        a = X.Sim(pl, cfg).run(); b = CV.CSim(pl, cfg, cmode='orig').run()
        out['a'][f'J{key[0]}s{key[1]}|{cost}'] = float(np.max(np.abs(a.to_numpy() - b.to_numpy())))
assert max(out['a'].values()) == 0.0, out['a']
# (b) 合成名單：在 lt 前最後一個訊號日買入單一股票，之後換成 2330
sig = pd.DatetimeIndex(pd.read_csv(D.O7 / 'plan_V7_B10.csv').signal)
for code, typ in (('3474', 'cash'), ('2847', 'mixed'), ('2311', 'conversion')):
    r = E.set_index('code').loc[code]; lt = pd.Timestamp(r.last_trade)
    s0 = sig[sig < lt][-1]; later = sig[sig > lt][:3]
    pl = [(D.TPOS[s0] + 1, np.array([D.CI[code]]), D.TPOS[s0])] + [(D.TPOS[s] + 1, np.array([D.CI['2330']]), D.TPOS[s]) for s in later]
    cfg = dict(tier=1e7, mode='GROW', model='K5T', cost='normal'); cfg.update(BASE)
    sm = CV.CSim(pl, cfg, settle=ST, cmode='consideration'); nav = sm.run(t1=D.TPOS[later[-1]] + 5)
    ev = [e for e in sm.log if e['code'] == code]; assert len(ev) == 1, (code, sm.log)
    ev = ev[0]; lc = float(r.last_close); sh = ev['value_lastclose'] / lc
    res = dict(type=ev['type'], parts=ev['parts'], shares=sh)
    if typ in ('cash', 'mixed'):
        got = [p for p in ev['parts'] if p[0] == 'cash'][0][2] / sh; res['cash_err'] = got / float(r.cash) - 1
        assert abs(res['cash_err']) < 0.005, res
    if typ in ('mixed', 'conversion'):
        sp = [p for p in ev['parts'] if p[0] == 'stock'][0]; sj = D.CI[sp[2]]; ts = D.TPOS[pd.Timestamp(sp[1])]
        res['succ_px_err'] = sp[3] / (sh * float(r.ratio)) / D.C[ts, sj] - 1
        assert abs(res['succ_px_err']) < 0.005, res
    res['n_settled'] = sm.acc['n_settled']; res['n_fallback'] = sm.acc['n_fallback']
    out['b'][code] = res
(HERE / 'smoke_v7x.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps(out, ensure_ascii=False, default=str)[:2500])
