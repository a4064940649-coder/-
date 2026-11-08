#!/usr/bin/env python3
"""integrate_o17 事後描述（未事前登記，不得推翻 results.json 的 DECISION）。在正式執行之後跑一次，輸出 posthoc.json：
  (1) C_DSP 異動槽位：移出股（官方處置期涵蓋）與移入股的「訊號日收盤→下一訊號日收盤」紙上報酬，以及處置清單列的旗標；
  (2) U13 放回槽位按發行人：放回股與被擠出股的下一期紙上報酬；
  (3) U13 的放回強度敏感度：pclib.VU 'p90'（放回股營收百分位 0.9）與 'carry'（沿用最後一個已知營收）× cons_main 一般成本（題 1 的描述口徑）；
  (4) A_UB 與 A_WORST 在 2019 年起的差異來源（2024-06 起有快照時戳）。
python3 -I -B posthoc.py"""
import sys, json, math, hashlib
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
AG = HERE.parent
RA = AG / 'o_rev_avail_ds_pit_count'; CZ = AG / 'o_engine_delist_consideration'
R = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
assert 'RESULTS' in R and not (HERE / 'posthoc.json').exists()
sys.path.insert(0, str(RA)); import pclib as P; sys.path.pop(0)
sys.path.insert(0, str(CZ)); import cz_engine as CE; sys.path.pop(0)
sys.path.insert(0, str(AG / 'o_forward_sequential_v2')); import fslib as F; sys.path.pop(0)
V, K = P.V, P.K; COLS = P.COLS; IDX = P.IDX
EV = pd.read_csv(CZ / 'events_v2.csv', dtype={'code': str, 'successor': str})
STM = CE.build_settlement(EV, 'main', d=0.057)
DU = pd.read_csv(RA / 'raw' / 'disposal_official_universe.csv', dtype={'code': str}, parse_dates=['start', 'end'])
_orig = K.disposal_mask
def dm_off(d):
    d = pd.Timestamp(d); m = _orig(d).copy()
    if d <= pd.Timestamp('2021-09-23'):
        e = IDX[IDX.get_loc(d) + 1]
        for c in DU[((DU.start <= d) & (DU.end >= d)) | ((DU.start <= e) & (DU.end >= e))].code:
            if c in P.CI: m[P.CI[c]] = True
    return m
S0 = V.V2('gapfix', P.YL, P.Y5, exfin=True); pl0 = S0.plan(10, 'official')
K.disposal_mask = dm_off
SD_ = V.V2('gapfix', P.YL, P.Y5, exfin=True); plD = SD_.plan(10, 'official')
K.disposal_mask = _orig
SD = pd.DatetimeIndex(S0.bounds()[2]); END = pd.Timestamp(V.END1012)
ADJ = K.ADJ.ffill()
def fwd(d, codes):
    i = SD.get_loc(d); b = SD[i + 1] if i + 1 < len(SD) else END
    a0 = ADJ.loc[:d].iloc[-1]; b0 = ADJ.loc[:b].iloc[-1]
    return {c: float(b0[c] / a0[c] - 1) for c in codes if np.isfinite(a0[c]) and np.isfinite(b0[c])}
PK0 = {p[3]: [COLS[j] for j in p[1]] for p in pl0}; PKD = {p[3]: [COLS[j] for j in p[1]] for p in plD}
rows = []
for d in PK0:
    o, i = sorted(set(PK0[d]) - set(PKD[d])), sorted(set(PKD[d]) - set(PK0[d]))
    if o or i:
        fo, fi = fwd(d, o), fwd(d, i)
        flags = {c: DU[(DU.code == c) & (DU.start <= d + pd.Timedelta(days=5)) & (DU.end >= d)][['start', 'end', 'overlap_proxy', 'second30', 'take_R1']].astype(str).to_dict('records') for c in o}
        rows.append(dict(sig=str(d.date()), out=fo, in_=fi, diff_pp=(np.mean(list(fi.values())) - np.mean(list(fo.values()))) * 100 if fo and fi else None, out_flags=flags))
OUT = {'C_DSP_rows': rows, 'C_DSP_sum_in_minus_out_pp': float(np.nansum([r['diff_pp'] for r in rows if r['diff_pp'] is not None])),
       'C_DSP_sum_2019on_pp': float(np.nansum([r['diff_pp'] for r in rows if r['diff_pp'] is not None and r['sig'] >= '2019'])),
       'C_DSP_mean_out_ret_pp': float(np.mean([v for r in rows for v in r['out'].values()]) * 100)}
# (2) U13 放回按發行人
SU = P.VU('opt', lo='2013-01-01', hi='2026-12-31', which='GOV'); plU = SU.plan(10, 'official')
PKU = {p[3]: [COLS[j] for j in p[1]] for p in plU}
byi = {}
for d in PKU:
    A_ = set(SU.aff.get(d, [])); ins = [c for c in PKU[d] if c in A_]
    if not ins: continue
    out_ = sorted(set(PK0[d]) - set(PKU[d]))
    fi, fo = fwd(d, ins), fwd(d, out_)
    for c in ins:
        byi.setdefault(c, []).append(dict(sig=str(d.date()), r=fi.get(c), out_mean=float(np.mean(list(fo.values()))) if fo else None))
OUT['U13_by_issuer'] = {c: dict(n=len(v), mean_r_pp=float(np.nanmean([x['r'] for x in v if x['r'] is not None]) * 100) if any(x['r'] is not None for x in v) else None,
                               mean_out_pp=float(np.nanmean([x['out_mean'] for x in v if x['out_mean'] is not None]) * 100) if any(x['out_mean'] is not None for x in v) else None,
                               first=v[0]['sig'], last=v[-1]['sig'])
                        for c, v in sorted(byi.items(), key=lambda kv: -len(kv[1]))}
# (3) p90／carry × cons_main 一般
E50 = F.e50()
def win(nav):
    pts = list(SD[SD < END]) + [END]
    W = pd.DataFrame([(pd.Period(a, 'M'), F._at(nav, b) / F._at(nav, a) - 1, F._at(E50, b) / F._at(E50, a) - 1) for a, b in zip(pts[:-1], pts[1:])], columns=['per', 'r', 'r50']).set_index('per')
    return W[~W.index.duplicated(keep='last')].dropna()
def ann(r): return float(np.prod(1 + np.asarray(r)) ** (12 / len(r)) - 1) * 100
def tt(x): x = np.asarray(x); return float(x.mean() / x.std(ddof=1) * math.sqrt(len(x)))
bb, ss = K.rates(0.001)
def summ(pl, t0, t1, eng):
    plan = [(p[0], p[1]) for p in pl]
    nav, _ = CE.simulate(plan, bb, ss, t0, t1, mode='consideration', settle=STM) if eng == 'cons_main' else CE.simulate(plan, bb, ss, t0, t1, mode=eng)
    W = win(nav); o = {}
    for k, (a, b) in {'ALL': (None, None), 'SEG1': ('2006-01', '2012-12'), 'SEG2': ('2013-01', '2018-12'), 'SEG3': ('2019-01', None)}.items():
        m = np.ones(len(W), bool)
        if a: m &= W.index >= pd.Period(a, 'M')
        if b: m &= W.index <= pd.Period(b, 'M')
        o[k] = dict(ann=ann(W.r[m]), gex=ann(W.r[m]) - ann(W.r50[m]), t=tt(W.r[m] - W.r50[m]))
    return o
t0, t1, _ = S0.bounds()
OUT['U13_intensity_cons_main_normal'] = {}
for md in ('opt', 'p90', 'carry'):
    S = P.VU(md, lo='2013-01-01', hi='2026-12-31', which='GOV'); pl = S.plan(10, 'official')
    n_ins = sum(1 for p in pl for j in p[1] if COLS[j] in set(S.aff.get(p[3], [])))
    OUT['U13_intensity_cons_main_normal'][md] = dict(n_ins=n_ins, stats=summ(pl, t0, t1, 'cons_main'))
OUT['C0_cons_main_normal'] = summ(pl0, t0, t1, 'cons_main')
# (4) A_UB 與 A_WORST 名單差（2019 起）
d1 = R['RESULTS']['desc']['A_nold_on_DS']
OUT['A_UB_vs_WORST_lagged_counts_2019on'] = {d: (d1['A_UB'][d], d1['A_WORST'][d]) for d in d1['A_UB'] if d >= '2019'}
OUT['label'] = '事後描述（未事前登記，不得推翻判定）'; OUT['utc'] = pd.Timestamp.now('UTC').isoformat()
(HERE / 'posthoc.json').write_text(json.dumps(OUT, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps({k: v for k, v in OUT.items() if k not in ('C_DSP_rows', 'U13_by_issuer')}, ensure_ascii=False, indent=1, default=str))
print(json.dumps(OUT['C_DSP_rows'], ensure_ascii=False, default=str)[:3000])
print(json.dumps(dict(list(OUT['U13_by_issuer'].items())[:12]), ensure_ascii=False))
