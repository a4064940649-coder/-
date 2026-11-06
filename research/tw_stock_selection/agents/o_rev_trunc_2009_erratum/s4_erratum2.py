#!/usr/bin/env python3
"""第 4 步：斷言 ERRATUM2_SPEC 已凍結且雜湊相符，然後計算 U_opt（判定）、U_p90、U_carry、ALL（描述）。
只輸出名單與計數（ERRATUM2.json、results.json 的 ERRATUM2）。不計算任何報酬、淨值或 t 值。用法：python3 -I -B s4_erratum2.py"""
import sys, json, time, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SP = HERE / 'ERRATUM2_SPEC.json'
assert SP.exists(), 'ERRATUM2_SPEC 未凍結'
import hashlib
RES = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
assert hashlib.sha256(SP.read_bytes()).hexdigest() == RES['PREREG']['sha256'], 'SPEC 雜湊不符'
SPEC = json.loads(SP.read_text(encoding='utf-8'))
import erlib as E
for k, p in (('v2_core_sha256', E.AG / 'main_v2/v2_core.py'), ('ia_core_sha256', E.AG / 'independent_audit/ia_core.py'), ('erlib_sha256', HERE / 'erlib.py'),
             ('tw_revenue_sha256', E.K.LONG), ('erratum_json_sha256', E.AG / 'o_g8_protocol_fin_erratum/ERRATUM.json')):
    assert E.sha(p) == SPEC['base'][k], k
import numpy as np, pandas as pd
T0 = time.time()
L0 = json.loads((E.AG / 'o_g8_protocol_fin_erratum/ERRATUM.json').read_text(encoding='utf-8'))['lists_C0']

base = E.lists(E.VU('base'))
assert sorted(base) == sorted(L0) and len(base) == 93
for d in base: assert base[d]['picks'] == L0[d], d
print('baseline = lists_C0 OK', round(time.time() - T0), flush=True)

RUNS = {}
for nm, mode, which in (('U_opt', 'opt', 'SURV'), ('U_p90', 'p90', 'SURV'), ('U_carry', 'carry', 'SURV'), ('U_opt_ALL', 'opt', 'ALL')):
    RUNS[nm] = E.lists(E.VU(mode, which)); print(nm, round(time.time() - T0), flush=True)

days = sorted(base)
D1 = [d for d in days if d < '2013-01-01']
def cnt(R):
    rows = []
    for d in days:
        A = set(R[d]['aff']); new = R[d]['picks']; old = base[d]['picks']
        ins = [c for c in new if c in A]
        rows.append(dict(sig=d, n_aff=len(A), inserted=ins, out=sorted(set(old) - set(new)), in_=sorted(set(new) - set(old))))
    u1 = sum(len(r['inserted']) for r in rows if r['sig'] < '2013-01-01'); u2 = sum(len(r['inserted']) for r in rows)
    c1 = sum(len(r['out']) for r in rows if r['sig'] < '2013-01-01'); c2 = sum(len(r['out']) for r in rows)
    return dict(U_2006_2012=u1, share_2006_2012=u1 / (10 * len(D1)), U_2006_2013=u2, share_2006_2013=u2 / (10 * len(days)),
                changed_2006_2012=c1, changed_2006_2013=c2, governing_share=max(u1 / (10 * len(D1)), u2 / (10 * len(days))),
                by_year={y: sum(len(r['inserted']) for r in rows if r['sig'][:4] == y) for y in sorted({d[:4] for d in days})},
                n_days_with_insert=sum(1 for r in rows if r['inserted']), rows=[r for r in rows if r['inserted'] or r['out']])
C = {k: cnt(v) for k, v in RUNS.items()}
assert len(D1) == 81

# 子樣本（U_opt）：原因、52 週百分位、受影響股
Ropt = RUNS['U_opt']
ins_rows = []
for d in days:
    A = set(Ropt[d]['aff'])
    el = Ropt[d]['el']; s52 = Ropt[d]['s52']
    p52 = pd.Series(np.where(el, s52, np.nan)).rank(pct=True).to_numpy()
    m = pd.Period(d, 'M')
    for c in Ropt[d]['picks']:
        if c in A:
            j = E.CI[c]
            ins_rows.append(dict(sig=d, code=c, name=E.NAME.get(c, ''), cause=E.cause(c, m), p52=float(p52[j]),
                                 rank_in_list=Ropt[d]['picks'].index(c) + 1, delist=E.LM.get(c, {}).get('delist_date', ''),
                                 last_trade=str(E.LT[c].date()), last_rev=str(E.LASTREV.get(c, ''))))
IR = pd.DataFrame(ins_rows)
# 所有受影響股月的 p52（分母，描述用）
aff_p52 = []
for d in days:
    el = Ropt[d]['el']; s52 = Ropt[d]['s52']
    p52 = pd.Series(np.where(el, s52, np.nan)).rank(pct=True).to_numpy()
    for c in Ropt[d]['aff']: aff_p52.append(dict(sig=d, code=c, p52=float(p52[E.CI[c]]), ins=c in Ropt[d]['picks']))
AP = pd.DataFrame(aff_p52)
sub = {}
if len(IR):
    sub['by_cause'] = IR.groupby('cause').size().to_dict()
    sub['by_cause_2006_2012'] = IR[IR.sig < '2013-01-01'].groupby('cause').size().to_dict()
    sub['p52_ge_0p8'] = dict(aff_stockmonths=int((AP.p52 >= 0.8).sum()), inserted=int(((AP.p52 >= 0.8) & AP.ins).sum()),
                             rate=float(AP[AP.p52 >= 0.8].ins.mean()) if (AP.p52 >= 0.8).any() else None)
    sub['p52_lt_0p8'] = dict(aff_stockmonths=int((AP.p52 < 0.8).sum()), inserted=int(((AP.p52 < 0.8) & AP.ins).sum()),
                             rate=float(AP[AP.p52 < 0.8].ins.mean()) if (AP.p52 < 0.8).any() else None)
    sub['seg_2006_2009'] = int((IR.sig < '2010-01-01').sum()); sub['seg_2010_2012'] = int(((IR.sig >= '2010-01-01') & (IR.sig < '2013-01-01')).sum())
    sub['seg_2013'] = int((IR.sig >= '2013-01-01').sum())
    sub['share_2006_2009_slots'] = sub['seg_2006_2009'] / (10 * sum(1 for d in days if d < '2010-01-01'))
    sub['share_2010_2012_slots'] = sub['seg_2010_2012'] / (10 * sum(1 for d in days if '2010-01-01' <= d < '2013-01-01'))
    sub['fin_inserted'] = int(sum(1 for c in IR.code if E.V.FINMASK[E.CI[c]]))
    sub['by_stock'] = IR.groupby(['code', 'name', 'cause', 'delist']).size().reset_index().rename(columns={0: 'n'}).sort_values('n', ascending=False).to_dict(orient='records')
    sub['aff_stockmonths_total'] = int(len(AP)); sub['aff_stockmonths_2006_2012'] = int((AP.sig < '2013-01-01').sum())
# 被擠出的原名單股
outs = {}
for r in C['U_opt']['rows']:
    for c in r['out']: outs[c] = outs.get(c, 0) + 1
outs = sorted(((c, E.NAME.get(c, ''), n) for c, n in outs.items()), key=lambda x: -x[2])
# 5854 / 6012
def k_of(R, c, a='2006', b='2014'):
    return [d for d in days if c in R[d]['picks'] and a <= d < b]
s5854 = {nm: k_of(R, '5854') for nm, R in [('base', base)] + list(RUNS.items())}
s6012 = {nm: k_of(R, '6012') for nm, R in [('base', base)] + list(RUNS.items())}
# 19 檔
L19 = [c for c, r in E.LM.items() if r['delist_date'][:4] in ('2010', '2011')]
s19 = {nm: {c: len(k_of(R, c)) for c in L19 if k_of(R, c)} for nm, R in [('base', base)] + list(RUNS.items())}

g = C['U_opt']['governing_share']
judg = dict(governing_share=g, threshold=0.05, exceeds=bool(g > 0.05),
            action=(SPEC['rule']['gt_5pct'] if g > 0.05 else SPEC['rule']['le_5pct']),
            G4_seg1=('未證實' if g > 0.05 else '維持（加註上界）'), G5_seg1=('存活者偏誤未排除' if g > 0.05 else '加註'))
slot_list = sorted((r['sig'], c) for r in C['U_opt']['rows'] for c in r['inserted'])
out = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), spec_sha256=RES['PREREG']['sha256'], no_returns='本程式未計算任何報酬、淨值或 t 值',
           n_days=len(days), n_days_2006_2012=len(D1), slots_2006_2012=10 * len(D1), slots_2006_2013=10 * len(days),
           archive_values_used=0, judgement=judg,
           counts={k: {kk: vv for kk, vv in v.items() if kk != 'rows'} for k, v in C.items()},
           U_opt_slot_list=slot_list, U_opt_rows=C['U_opt']['rows'], U_opt_inserted=ins_rows, subsamples=sub,
           displaced_original=outs, s5854=s5854, s6012=s6012, L19_top10=s19,
           descriptive_rows={k: C[k]['rows'] for k in ('U_p90', 'U_carry', 'U_opt_ALL')}, runtime_s=time.time() - T0)
(HERE / 'ERRATUM2.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
RES['ERRATUM2'] = dict(utc=out['utc'], judgement=judg, counts=out['counts'], slot_list_sha256=hashlib.sha256(json.dumps(slot_list).encode()).hexdigest())
(HERE / 'results.json').write_text(json.dumps(RES, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps(out['counts'], ensure_ascii=False, indent=0))
print(json.dumps(judg, ensure_ascii=False)); print(json.dumps({k: v for k, v in sub.items() if k != 'by_stock'}, ensure_ascii=False))
print(pd.DataFrame(sub.get('by_stock', [])).to_string()); print(outs[:15]); print('5854', s5854); print('6012', s6012); print('L19', s19)
