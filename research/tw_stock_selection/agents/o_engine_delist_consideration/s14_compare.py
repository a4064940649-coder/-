#!/usr/bin/env python3
"""獨立複製比對：事件表逐筆、C0 逐日淨值、U_opt／V7X 判定欄。寫入 results.json 的 REPLICATION（只寫一次）。python3 -I -B s14_compare.py"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); import cz_lib as Z
RJ = HERE / 'results.json'; R = json.loads(RJ.read_text(encoding='utf-8'))
assert 'REPLICATION' not in R
RP = HERE / 'replicate'
# 事件表
a = pd.read_csv(RP / 'rp_events.csv', dtype={'code': str, 'successor': str}); b = pd.read_csv(HERE / 'events_v2.csv', dtype={'code': str, 'successor': str})
m = a.merge(b, on='code', suffixes=('_r', '_m'))
def eq(x, y): return (pd.isna(x) and pd.isna(y)) or (not pd.isna(x) and not pd.isna(y) and abs(float(x) - float(y)) < 1e-9)
diff = []
for r in m.itertuples():
    bad = [f for f in ('cash', 'ratio', 'pref_ratio', 'pref_value') if not eq(getattr(r, f + '_r'), getattr(r, f + '_m'))]
    if r.type_r != r.type_m or bad or r.last_trade_r != r.last_trade_m:
        diff.append(dict(code=r.code, type_r=r.type_r, type_m=r.type_m, fields=bad, held=bool(r.held)))
adj = json.loads((RP / 'adjudication.json').read_text(encoding='utf-8'))
ev = dict(n_common=len(m), n_only_researcher=int((~b.code.isin(a.code)).sum()), n_only_replicator=int((~a.code.isin(b.code)).sum()), diffs=diff, adjudication=adj['rows'])
# C0 逐日淨值
N1 = pd.read_parquet(Z.WORK / 'navs_ia.parquet'); N2 = pd.read_parquet(Z.WORK / 'rp_navs_ia_asis.parquet')
c0 = {eg: float(np.max(np.abs(N2[f'C0|{eg}'].to_numpy() - N1[f'C0|normal|{eg}'].to_numpy()))) for eg in ('freeze', 'cons_main', 'cons_lo', 'cons_hi', 'cons_c50')}
c0['rp_cons_vs_rp_freeze'] = float(np.max(np.abs(N2['C0|cons_main'].to_numpy() - N2['C0|freeze'].to_numpy())))
# U_opt 判定欄
S = R['RESULTS_IA']['stats']; ua = {}
for tag in ('asis', 'aligned'):
    P = json.loads((RP / f'rp_ia_{tag}.json').read_text(encoding='utf-8')); mx = {}
    for k, v in P.items():
        var, eg = k.split('|'); s = S[f'{var}|normal|{eg}']
        dd = [abs(v[p]['ann'] - s[p]['ann']) for p in ('ALL',)] + [abs(v[p]['gex'] - s[p]['gex']) for p in ('SEG1', 'SEG2', 'SEG3')]
        dt = [abs(v['ALL']['t'] - s['ALL']['t']), abs(v['t_drop_best3'] - s['ALL']['t_drop_best3'])]
        mx[k] = dict(max_ann_pp=max(dd), max_t=max(dt))
    ua[tag] = mx
# V7X
T = R['RESULTS_V7X']['table']; F = ['CAGR_med', 'exc_med', 't_0050_med', 'exc_p10', 'CAGR_Y19on_med', 'exc_Y19on_med', 't_Y19on_med', 'exc_Y19on_p10']
va = {}
for tag in ('asis', 'aligned'):
    p = RP / f'rp_v7x_{tag}.json'
    if not p.exists(): va[tag] = 'missing'; continue
    P = json.loads(p.read_text(encoding='utf-8')); mx = {}
    for k, v in P.items():
        if k not in T: continue
        a_ = max(abs(v[f] - T[k][f]) for f in F if not f.startswith('t_')); t_ = max(abs(v[f] - T[k][f]) for f in F if f.startswith('t_'))
        mx[k] = dict(max_ann_pp=a_, max_t=t_)
    va[tag] = mx
def ok_block(mx, engines):
    sel = {k: v for k, v in mx.items() if any(k.endswith('|' + e) or k.endswith(e) for e in engines)}
    return dict(n=len(sel), max_ann_pp=max(v['max_ann_pp'] for v in sel.values()) if sel else None, max_t=max(v['max_t'] for v in sel.values()) if sel else None,
                pass_=all(v['max_ann_pp'] <= 0.01 and v['max_t'] <= 0.005 for v in sel.values()) if sel else None)
summ = dict(U_opt_asis_main_c50=ok_block(ua['asis'], ['freeze', 'cons_main', 'cons_c50']), U_opt_aligned_all=ok_block(ua['aligned'], ['freeze', 'cons_main', 'cons_lo', 'cons_hi', 'cons_c50']),
            U_opt_asis_lo_hi=ok_block(ua['asis'], ['cons_lo', 'cons_hi']))
if isinstance(va.get('asis'), dict):
    summ['V7X_asis_orig'] = ok_block(va['asis'], ['orig'])
    summ['V7X_asis_main_c50'] = ok_block(va['asis'], ['cons_main', 'cons_c50'])
    summ['V7X_asis_lo_hi'] = ok_block(va['asis'], ['cons_lo', 'cons_hi'])
if isinstance(va.get('aligned'), dict):
    summ['V7X_aligned_lo_hi'] = ok_block(va['aligned'], ['cons_lo', 'cons_hi'])
R['REPLICATION'] = dict(utc=pd.Timestamp.now('UTC').isoformat(), events=ev, C0_daily_maxabs=c0, U_opt=ua, V7X=va, summary=summ,
                        n_distinct_events_engine=None)
RJ.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps(dict(events=dict(n_common=ev['n_common'], diffs=diff), C0=c0, summary=summ), ensure_ascii=False, indent=1, default=str))
