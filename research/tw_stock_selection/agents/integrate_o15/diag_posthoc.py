#!/usr/bin/env python3
"""integrate_o15 事後診斷（PREREG 之外、看過 RESULTS 之後才寫；不改 DECISION）：
  1) freeze 引擎下各 U 變體「下市後永久凍結」持股占 NAV 的比例（死錢拖累）；
  2) 敏感度：freeze=False（換股日無成交者以最後收盤出清＝舊程式作法，近似股份轉換／現金收購的實際兌現）。
結果寫入 results.json 的 POSTHOC_DIAG（只追加，不動 PREREG／RESULTS／DECISION）。python3 -I -B diag_posthoc.py"""
import sys, json, math, hashlib
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent; ER = AG / 'o_rev_trunc_2009_erratum'
RJ = HERE / 'results.json'
R = json.loads(RJ.read_text(encoding='utf-8')); assert 'RESULTS' in R and 'POSTHOC_DIAG' not in R
sys.path.insert(0, str(ER)); import erlib as EL; sys.path.pop(0)
V, K, COLS = EL.V, EL.K, EL.COLS
sys.path.insert(0, str(AG / 'o_forward_sequential_v2')); import fslib as F; sys.path.pop(0)
E = F.e50()
VARS = {'C0': ('base', 'SURV'), 'U_opt': ('opt', 'SURV'), 'U_p90': ('p90', 'SURV'), 'U_carry': ('carry', 'SURV'), 'U_opt_ALL': ('opt', 'ALL')}
bb, ss = K.rates(0.001)
out = {}; WIN = {}
for k, (mode, which) in VARS.items():
    S = EL.VU(mode, which); pl = S.plan(10, 'official'); t0, t1, sd = S.bounds(); SD = pd.DatetimeIndex(sd)
    plan = [(p[0], p[1]) for p in pl]
    nav, lg = K.simulate(plan, bb, ss, t0, t1)
    fr = lg['frozen']
    fz = {d: w for d, c, w in fr}
    def at(y): 
        ks = [d for d in fz if d <= y]; return (fz[ks[-1]] if ks else 0.0)
    codes = sorted({c for _, cs, _ in fr for c in cs})
    perm = sorted({c for d, cs, _ in fr if d >= '2014-01-01' for c in cs})
    out[k] = dict(n_frozen_events=len(fr), frozen_codes=codes, frozen_codes_still_after_2014=perm,
                  frozen_w_at={y: at(y) for y in ('2009-12-31', '2011-12-31', '2013-12-31', '2018-12-31', '2026-09-30')})
    nav2, _ = K.simulate(plan, bb, ss, t0, t1, freeze=False)
    pts = list(SD[SD < V.END1012]) + [pd.Timestamp(V.END1012)]
    rows = [(pd.Period(a, 'M'), F._at(nav2, b) / F._at(nav2, a) - 1, F._at(E, b) / F._at(E, a) - 1) for a, b in zip(pts[:-1], pts[1:])]
    D = pd.DataFrame(rows, columns=['per', 'r', 'r50']).set_index('per'); WIN[k] = D[~D.index.duplicated(keep='last')].dropna()
    print(k, out[k]['frozen_w_at'], perm, flush=True)
def ann(r): r = np.asarray(r); return float(np.prod(1 + r) ** (12 / len(r)) - 1) * 100
def tt(x): x = np.asarray(x); return float(x.mean() / x.std(ddof=1) * math.sqrt(len(x)))
def mdd(r): n = np.cumprod(1 + np.asarray(r)); return float((n / np.maximum.accumulate(n) - 1).min()) * 100
PER = {'ALL': (None, None), 'SEG1': ('2006-01', '2012-12'), 'SEG2': ('2013-01', '2018-12'), 'SEG3': ('2019-01', None), 'Y0613': ('2006-01', '2013-12'), 'Y14ON': ('2014-01', None)}
NF = {}
for k in VARS:
    W = WIN['C0'].join(WIN[k][['r']], rsuffix='_k', how='inner'); o = {}
    for p, (a, b) in PER.items():
        m = np.ones(len(W), bool)
        if a: m &= W.index >= pd.Period(a, 'M')
        if b: m &= W.index <= pd.Period(b, 'M')
        x, y, z = W['r_k'].to_numpy()[m], W['r50'].to_numpy()[m], W['r'].to_numpy()[m]
        o[p] = dict(n=int(m.sum()), ann=ann(x), gex=ann(x) - ann(y), mdd=mdd(x), t_vs_0050=tt(x - y), ann_diff_vs_C0=ann(x) - ann(z),
                    diff_vs_C0_pp=float(np.mean(x - z) * 100), t_diff_vs_C0=(tt(x - z) if np.std(x - z) > 0 else None),
                    corr_C0=float(np.corrcoef(x, z)[0, 1]), n_win_diff=int(np.sum(np.abs(x - z) > 1e-12)))
    NF[k] = o
R['POSTHOC_DIAG'] = dict(note='事後診斷，PREREG 之外，不改 DECISION。freeze 引擎把下市當時仍持有、之後再無成交的股票永久凍結在最後市值（零報酬死錢），使 U 變體 2014 年後名單與 C0 相同卻持續落後；nofreeze＝換股日以最後收盤出清。',
                         frozen=out, nofreeze_normal=NF, utc=pd.Timestamp.now('UTC').isoformat(),
                         script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
RJ.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
for k in VARS: print(k, {p: (round(v['ann'], 2), round(v['gex'], 2), round(v['t_vs_0050'], 3), round(v['ann_diff_vs_C0'], 2), v['t_diff_vs_C0'] and round(v['t_diff_vs_C0'], 2), round(v['corr_C0'], 4), v['n_win_diff'], round(v['mdd'], 1)) for p, v in NF[k].items()})
