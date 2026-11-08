#!/usr/bin/env python3
"""面板 B（V7 面板＋V4 營收含 o2 補值）的 2013 年後存活者缺口 U_opt 計數。獨立行程（v7env 會換掉 ia_core 的面板路徑）。
--smoke：只檢查 V7 正式名單重現 plan_V7.csv（不算任何放回槽位）。正式：斷言 SPEC 已凍結，輸出 v7surv.json。
python3 -I -B s4b_v7surv.py [--smoke]"""
import sys, json, time, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
AG = HERE.parent
SMOKE = '--smoke' in sys.argv
if not SMOKE:
    R = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((HERE / 'SPEC.json').read_bytes()).hexdigest() == R['PREREG']['sha256'], 'SPEC 未凍結或被改'
sys.path.insert(0, str(AG / 'o_v7_listing_exec_engine'))
import v7env
L = v7env.load('v7')
sys.path.pop(0)
import numpy as np, pandas as pd
K, V = L.K, L.V
sys.path.insert(0, str(HERE))
import pclib as P          # 在 V7 行程中，pclib 的 K 就是 V7 面板（同一個 ia_core 模組）
assert P.K is K
T0 = time.time()
fz4 = L.check_freeze(); XR = L.setup_px()
YFX, Y5F = L.load_rfix(); DMT = L.disp_mask_fn('trade')
COLS = list(K.COLS); N = len(COLS); IDX = K.IDX

class V4U(L.V4):
    def __init__(self, mode='base', lo='2013-01-01', hi='2026-12-31'):
        self.mode = mode; self.lo, self.hi = pd.Timestamp(lo), pd.Timestamp(hi); self.aff = {}; self.causes = {}
        super().__init__(YFX, Y5F, DMT)
    def cross_section(self, d):
        if d in self._cs: return self._cs[d]
        cs = super().cross_section(d)
        if self.mode == 'base' or not (self.lo <= d <= self.hi):
            self.aff[d] = []; return cs
        ok = cs['ok_other']; y3 = cs['y3']
        A = np.zeros(N, bool); cz = {}
        for j in np.where(ok & ~np.isfinite(y3))[0]:
            k = P.cause(COLS[j], d); cz[COLS[j]] = k
            if k in P.GOV: A[j] = True
        self.causes[d] = cz; self.aff[d] = [COLS[j] for j in np.where(A)[0]]
        el = cs['el'] | A; n = el.sum(); s52 = cs['s52']
        s52p = pd.Series(np.where(el, s52, np.nan)).rank(pct=True).to_numpy()
        norm = el & ~A
        rp = pd.Series(np.where(norm, y3, np.nan)).rank().to_numpy() / n
        rp[A] = 1.0
        sc = 0.5 * s52p + 0.5 * rp; sc[~el] = np.nan
        ytie = np.where(el, np.where(A, np.inf, y3), np.nan)
        cs2 = dict(cs, el=el, sc=sc, y3=ytie, A=A)
        self._cs[d] = cs2
        return cs2

def lists(S):
    pl = S.plan(10, 'official'); out = {}
    for e, pk, el, d in pl:
        S.cross_section(d)
        out[str(d.date())] = dict(exec=str(IDX[e].date()), picks=[COLS[j] for j in pk], aff=list(S.aff.get(d, [])), causes=dict(S.causes.get(d, {})))
    return out

B = lists(V4U('base'))
pv = pd.read_csv(AG / 'o_v7_listing_exec_engine/plan_V7.csv', dtype=str)
ref = {s: p.split(',') for s, p in zip(pv.signal, pv.picks)}
same = all(B.get(s, {}).get('picks') == ref[s] for s in ref) and set(B) == set(ref)
diffs = [s for s in ref if B.get(s, {}).get('picks') != ref[s]][:10]
print('V7 base reproduces plan_V7.csv:', same, 'n', len(B), len(ref), diffs, round(time.time() - T0), flush=True)
assert same, 'V7 正式名單未重現'
if SMOKE: sys.exit(0)

U = lists(V4U('opt'))
rows = []
for s, v in U.items():
    if s < '2013-01-01': continue
    ins = [c for c in v['picks'] if c in set(v['aff'])]
    rows.append(dict(sig=s, n_aff=len(v['aff']), inserted=ins, causes={c: v['causes'][c] for c in ins},
                     changed=len(set(B[s]['picks']) - set(v['picks']))))
def seg(a, b):
    x = [r for r in rows if a <= r['sig'] <= b]
    n = sum(len(r['inserted']) for r in x); slots = 10 * len(x)
    by = {}
    for r in x:
        for c in r['inserted']: by[c] = by.get(c, 0) + 1
    top = max(by, key=by.get) if by else None
    return dict(U_opt=n, slots=slots, share=n / slots if slots else None, changed=sum(r['changed'] for r in x),
                n_days=sum(1 for r in x if r['inserted']), n_issuers=len(by), top_issuer=top, top_n=by.get(top, 0),
                share_ex_top=(n - by.get(top, 0)) / slots if slots else None, by_issuer=dict(sorted(by.items(), key=lambda z: -z[1])),
                by_cause={k: sum(1 for r in x for c, kk in r['causes'].items() if kk == k) for k in P.GOV})
out = {'utc': pd.Timestamp.now('UTC').isoformat(), 'panel': 'B：V7 面板＋V4 營收（R_FIX/Y5_FIX）＋官方處置 trade＋排除 KY/DR、FIN2', 'smoke_plan_V7_reproduced': same,
       'n_signal_days_2013_on': len(rows), 'last_signal': max(r['sig'] for r in rows),
       'seg': {'2013_2026': seg('2013-01-01', '2026-12-31'), '2013_2018': seg('2013-01-01', '2018-12-31'), '2019_2026': seg('2019-01-01', '2026-12-31'),
               '2014_2018': seg('2014-01-01', '2018-12-31')},
       'rows': rows, 'no_returns': '本程式未計算任何報酬', 'runtime_s': round(time.time() - T0, 1)}
(HERE / 'v7surv.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ('by_issuer',)} for k, v in out['seg'].items()}, ensure_ascii=False, indent=1, default=str))
