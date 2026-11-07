#!/usr/bin/env python3
"""第 4 步（凍結後一次）：斷言 PREREG（ERRATUM_SPEC_FM）存在且雜湊、程式與引擎 sha 都未變；
以 C0／F1／F2／F3 遮罩重跑 C0 型與 U_opt 名單（只產名單），計算 8 個判定比例 M 與描述口徑。
不計算任何報酬、淨值或 t 值。輸出 FM_COUNTS.json、results.json 的 COUNTS。用法：python3 -I -B s4_count.py"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import freg as R
SPEC, REG = R.check_reg('ERRATUM_SPEC_FM')
RES = R.load_res()
assert RES.get('PREREG', {}).get('sha256') == REG['sha256'], 'PREREG 未登記或雜湊不符'
for k, h in SPEC['files']['code'].items():
    if k != 's3_freeze.py': assert R.sha(HERE / k) == h, f'{k} 已變'
AG = R.AG
for k, p in (('v2_core', AG / 'main_v2/v2_core.py'), ('ia_core', AG / 'independent_audit/ia_core.py'), ('erlib', AG / 'o_rev_trunc_2009_erratum/erlib.py')):
    assert R.sha(p) == SPEC['files']['engine'][k], k
import fclass as F
for k, h in SPEC['files']['sources'].items():
    if k.startswith('o13/'): assert R.sha(F.O13 / k[4:]) == h, k
import flib as L
import numpy as np, pandas as pd
E, V, K = L.E, L.V, L.K
T0 = time.time()
CL = F.Classifier(SPEC['classifiers']['keywords'], SPEC['classifiers']['f2_manual_exclusion'], SPEC['classifiers']['name_exclusion'])
C0 = L.C0MASK
F2V = np.array([CL.f2(c) for c in L.COLS])
_m = {}
def mF1(d):
    k = ('F1', d)
    if k not in _m: _m[k] = np.array([CL.f1_value(CL.f1(c, d)[0], C0[j]) for j, c in enumerate(L.COLS)])
    return _m[k]
MASKS = {'C0': lambda d: C0, 'F1': mF1, 'F2': lambda d: F2V, 'F3': lambda d: mF1(d) | F2V}
# 分類指紋核對
days_all = [str(d.date()) for d in L.VM(lambda d: C0).sig_days if E.LO <= d <= E.HI]
rows = []
for d in days_all:
    a1 = mF1(pd.Timestamp(d))
    for j, c in enumerate(L.COLS):
        if a1[j] != C0[j] or F2V[j] != C0[j]: rows.append([d, c, int(C0[j]), int(a1[j]), int(F2V[j]), CL.f1(c, d)[0]])
import hashlib
assert hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode()).hexdigest() == SPEC['classifiers']['fingerprint']['sha256'], '分類指紋不符'

# ---------------- 名單 ----------------
RUN = {}
for nm in ('C0', 'F1', 'F2', 'F3'):
    for mode in ('base', 'opt'):
        RUN[(nm, mode)] = L.run(MASKS[nm], mode)
    print(nm, round(time.time() - T0), flush=True)
days = sorted(RUN[('C0', 'base')]); D1 = [d for d in days if d < '2013-01-01']
assert days == days_all and len(days) == 93 and len(D1) == 81
L0 = json.loads((AG / 'o_g8_protocol_fin_erratum/ERRATUM.json').read_text(encoding='utf-8'))['lists_C0']
assert all(RUN[('C0', 'base')][d]['picks'] == L0[d] for d in days), 'C0 原樣重跑不一致'
ER2 = json.loads((AG / 'o_rev_trunc_2009_erratum/ERRATUM2.json').read_text(encoding='utf-8'))
assert sorted(map(list, L.inserted(RUN[('C0', 'opt')]))) == sorted(map(list, ER2['U_opt_slot_list'])), 'U_opt 原樣重跑不一致'
validity = dict(C0_rerun_identical=True, U_opt_rerun_identical=True, n_U_opt=len(L.inserted(RUN[('C0', 'opt')])))

# 5 日獨立重算（F1、F2 遮罩）
S52, V60 = V.rolling_signals('gapfix')
def indep(d, mask, held):
    D = pd.Timestamp(d); t = K.IDX.get_loc(D); e = t + 1
    top = L.top300(D)
    y = K.yoy3_for_month(E.YL, pd.Period(D, 'M')); y = np.where(mask, np.nan, y)
    s = S52[t]
    el = top & np.isfinite(y) & np.isfinite(s) & ~K.disposal_mask(D) & np.isfinite(K.On[e]) & K.TRADED[e]
    idx = np.where(el)[0]
    r1 = pd.Series(s[idx]).rank(pct=True).to_numpy(); r2 = pd.Series(y[idx]).rank(pct=True).to_numpy()
    sc = np.round(0.5 * r1 + 0.5 * r2, 10)
    srt = sorted(range(len(idx)), key=lambda i: (-sc[i], -y[idx[i]], idx[i]))
    gu = K.gap_up_open(e); pk = []
    for i in srt:
        j = idx[i]
        if gu[j] and L.COLS[j] not in held: continue
        pk.append(L.COLS[j])
        if len(pk) == 10: break
    return pk
samp = {}
for nm in ('F1', 'F2'):
    Rb = RUN[(nm, 'base')]
    for d in SPEC['pipeline']['sample_days']:
        i = days.index(d); held = set(Rb[days[i - 1]]['picks']) if i > 0 else set()
        pk = indep(d, MASKS[nm](pd.Timestamp(d)), held)
        samp[f'{nm}:{d}'] = pk == Rb[d]['picks']
assert all(samp.values()), samp
validity['sample5_indep_F1_F2'] = samp

# ---------------- 判定比例 ----------------
def ratio_block(A, B):
    tot, rows = L.sym_slots(A, B)
    t1 = sum(r['slots'] for r in rows if r['sig'] < '2013-01-01')
    return dict(slots_930=tot, share_930=tot / 930, slots_810=t1, share_810=t1 / 810, rows=rows,
                by_year={y: sum(r['slots'] for r in rows if r['sig'][:4] == y) for y in sorted({d[:4] for d in days})})
LISTN = {'base': 'C0', 'opt': 'U_opt'}
BL = {}
for nm in ('F1', 'F2', 'F3'):
    for mode in ('base', 'opt'):
        BL[f'{LISTN[mode]}|{nm}'] = ratio_block(RUN[('C0', mode)], RUN[(nm, mode)])
gov = {f'{k}|{w}': BL[k][f'share_{w}'] for k in ('C0|F1', 'C0|F2', 'U_opt|F1', 'U_opt|F2') for w in ('930', '810')}
M = max(gov.values()); Mk = max(gov, key=gov.get)
judg = dict(ratios=gov, M=M, M_at=Mk, threshold=0.05, exceeds=bool(M > 0.05),
            action=SPEC['judgement']['gt_5pct'] if M > 0.05 else SPEC['judgement']['le_5pct'],
            G5_seg1=('再加一條「金融遮罩非 PIT，未排除」' if M > 0.05 else '不新增項目，只加註'))
print('M', M, Mk, flush=True)

# ---------------- 遮罩差異代號、直接污染、股池 ----------------
def diffcodes(nm):
    out = set()
    for d in days:
        m = MASKS[nm](pd.Timestamp(d)); out |= {L.COLS[j] for j in np.where(m != C0)[0]}
    return sorted(out)
DC = {nm: diffcodes(nm) for nm in ('F1', 'F2', 'F3')}
def missed_del_fin(nm):
    return sorted(c for c in DC[nm] if (not C0[L.CI[c]]) and F.is_delisted(c) and any(MASKS[nm](pd.Timestamp(d))[L.CI[c]] for d in days))
MDF = {nm: missed_del_fin(nm) for nm in ('F1', 'F2', 'F3')}
POOL = {d: L.top300(pd.Timestamp(d)) for d in days}
def pool_months(nm):
    out = {}
    for d in days:
        m = MASKS[nm](pd.Timestamp(d)); el = RUN[('C0', 'base')][d]['el']
        for c in MDF[nm]:
            j = L.CI[c]
            if POOL[d][j] and m[j] and not C0[j]:
                o = out.setdefault(c, dict(pool=0, el=0, by_year={}))
                o['pool'] += 1; o['el'] += int(el[j]); o['by_year'][d[:4]] = o['by_year'].get(d[:4], 0) + 1
    return out
PM = {nm: pool_months(nm) for nm in ('F1', 'F2', 'F3')}
def direct(nm):
    o = {}
    for mode in ('base', 'opt'):
        Rr = RUN[('C0', mode)]
        o[LISTN[mode]] = sorted([d, c, Rr[d]['picks'].index(c) + 1] for d in days for c in Rr[d]['picks'] if c in MDF[nm])
    return o
DIR = {nm: direct(nm) for nm in ('F1', 'F2', 'F3')}
# UNKNOWN
UNK = {}
for d in days:
    for j in np.where(POOL[d])[0]:
        c = L.COLS[j]; s, _ = CL.f1(c, d)
        if s == 'UNKNOWN':
            k = 'UNKNOWN_EVID' if CL.evidence(c) else 'UNKNOWN_NOEVID'
            u = UNK.setdefault(k, {}).setdefault(c, dict(pool=0, C0_slots=0, U_opt_slots=0))
            u['pool'] += 1
            u['C0_slots'] += int(c in RUN[('C0', 'base')][d]['picks']); u['U_opt_slots'] += int(c in RUN[('C0', 'opt')][d]['picks'])
UNKS = {k: dict(n_codes=len(v), n_pool_months=sum(x['pool'] for x in v.values()), C0_slots=sum(x['C0_slots'] for x in v.values()),
                U_opt_slots=sum(x['U_opt_slots'] for x in v.values()), codes=v) for k, v in UNK.items()}

# ---------------- 描述：單股、去最大貢獻股、拆解 ----------------
def mask_with(nm, codes):
    S = set(codes)
    idx = np.array([c in S for c in L.COLS])
    return lambda d: np.where(idx, MASKS[nm](d), C0)
single = {}
for nm in ('F1', 'F2'):
    for c in DC[nm]:
        for mode in ('base', 'opt'):
            Rr = L.run(mask_with(nm, [c]), mode)
            b = ratio_block(RUN[('C0', mode)], Rr)
            single[f'{LISTN[mode]}|{nm}|{c}'] = dict(slots_930=b['slots_930'], slots_810=b['slots_810'], rows=b['rows'])
exto = {}
for nm in ('F1', 'F2'):
    for mode in ('base', 'opt'):
        k = f'{LISTN[mode]}|{nm}'
        cand = sorted(DC[nm], key=lambda c: (-single[f'{k}|{c}']['slots_930'], c))
        top = cand[0]
        rest = [c for c in DC[nm] if c != top]
        Rr = L.run(mask_with(nm, rest), mode) if rest else RUN[('C0', mode)]
        b = ratio_block(RUN[('C0', mode)], Rr)
        exto[k] = dict(top_code=top, top_single_slots_930=single[f'{k}|{top}']['slots_930'], ex_top_share_930=b['share_930'], ex_top_share_810=b['share_810'],
                       ex_top_slots_930=b['slots_930'], ex_top_slots_810=b['slots_810'])
decomp = {}
for nm in ('F1', 'F2'):
    grp = dict(delisted_fin_added=[c for c in DC[nm] if c in MDF[nm]], other_reclass=[c for c in DC[nm] if c not in MDF[nm]])
    for g, cs in grp.items():
        for mode in ('base', 'opt'):
            if not cs:
                decomp[f'{LISTN[mode]}|{nm}|{g}'] = dict(codes=[], slots_930=0, slots_810=0); continue
            b = ratio_block(RUN[('C0', mode)], L.run(mask_with(nm, cs), mode))
            decomp[f'{LISTN[mode]}|{nm}|{g}'] = dict(codes=cs, slots_930=b['slots_930'], slots_810=b['slots_810'], share_930=b['share_930'], share_810=b['share_810'])
print('desc done', round(time.time() - T0), flush=True)

# ---------------- 可成交性（直接污染槽位，只描述） ----------------
def exe_rows(nm):
    out = []
    for lst, mode in (('C0', 'base'), ('U_opt', 'opt')):
        Rr = RUN[('C0', mode)]
        for d, c, rk in DIR[nm][lst]:
            j = L.CI[c]; e = Rr[d]['e']; i = days.index(d)
            nxt = Rr[days[i + 1]]['e'] if i + 1 < len(days) else e + 21
            w = np.where(K.TRADED[:, j])[0]; lt = w[-1]
            out.append(dict(list=lst, sig=d, code=c, rank=rk, entry=str(K.IDX[e].date()), gap_up_entry=bool(K.gap_up_open(e)[j]),
                            traded_whole_hold=bool(K.TRADED[e:nxt + 1, j].all()), last_trade=str(K.IDX[lt].date()),
                            tdays_signal_to_last_trade=int(lt - K.IDX.get_loc(pd.Timestamp(d))), delisted_in_hold=bool(lt < nxt),
                            v60_mn=round(float(V60[K.IDX.get_loc(pd.Timestamp(d)), j]) / 1e6, 1)))
    return out
EXE = {nm: exe_rows(nm) for nm in ('F1', 'F2')}

# ---------------- 預測 ----------------
pm_tot = sum(v['pool'] for v in PM['F1'].values())
known = {'5854', '6004', '6012'}
pred = {}
pred['P1'] = dict(n_missed_delisted_fin_F1=len(MDF['F1']), codes_F1=MDF['F1'], n_F2=len(MDF['F2']), codes_F2=MDF['F2'],
                  known3_share_of_pool_months_F1=(sum(v['pool'] for c, v in PM['F1'].items() if c in known) / pm_tot if pm_tot else None),
                  known3_share_of_pool_months_F2=sum(v['pool'] for c, v in PM['F2'].items() if c in known) / max(1, sum(v['pool'] for v in PM['F2'].values())),
                  holds_F1=bool(3 <= len(MDF['F1']) <= 10), holds_F2=bool(3 <= len(MDF['F2']) <= 10))
dC0 = DIR['F2']['C0']
pred['P2'] = dict(direct_C0_F1=len(DIR['F1']['C0']), direct_C0_F2=len(dC0), direct_C0_F2_share=len(dC0) / 930, includes_6012_20080411=any(d == '2008-04-11' and c == '6012' for d, c, _ in dC0),
                  n5854_in_C0=sum(1 for d in days if '5854' in RUN[('C0', 'base')][d]['picks']),
                  holds=bool(1 <= len(dC0) <= 4 and any(d == '2008-04-11' and c == '6012' for d, c, _ in dC0)))
pred['P3'] = dict(C0_F1_930=BL['C0|F1']['slots_930'], C0_F2_930=BL['C0|F2']['slots_930'], C0_max_share_930=max(gov['C0|F1|930'], gov['C0|F2|930']),
                  C0_max_share_810=max(gov['C0|F1|810'], gov['C0|F2|810']), U_opt_max_share=max(gov['U_opt|F1|930'], gov['U_opt|F2|930'], gov['U_opt|F1|810'], gov['U_opt|F2|810']),
                  holds=bool(max(gov['C0|F1|930'], gov['C0|F2|930']) <= 0.02 and max(gov['C0|F1|810'], gov['C0|F2|810']) <= 0.02 and
                             max(gov['U_opt|F1|930'], gov['U_opt|F2|930'], gov['U_opt|F1|810'], gov['U_opt|F2|810']) <= 0.03 and M <= 0.05))
tot4 = sum(BL[k]['slots_930'] for k in ('C0|F1', 'C0|F2', 'U_opt|F1', 'U_opt|F2'))
late = sum(BL[k]['by_year'].get(y, 0) for k in ('C0|F1', 'C0|F2', 'U_opt|F1', 'U_opt|F2') for y in ('2010', '2011', '2012', '2013'))
early = sum(BL[k]['by_year'].get(y, 0) for k in ('C0|F1', 'C0|F2', 'U_opt|F1', 'U_opt|F2') for y in ('2007', '2008', '2009'))
pred['P4'] = dict(total_slots_4combos=tot4, slots_2010_2013=late, share_2010_2013=(late / tot4 if tot4 else None), slots_2007_2009=early,
                  holds=bool(tot4 > 0 and late / tot4 <= 0.30))

out = dict(utc=R.now_utc(), spec_sha256=REG['sha256'], no_returns='本程式未計算任何報酬、淨值或 t 值', validity=validity, judgement=judg,
           blocks={k: {kk: vv for kk, vv in v.items()} for k, v in BL.items()}, diff_codes=DC, missed_delisted_fin=MDF, pool_months=PM, direct=DIR,
           unknown=UNKS, single_code=single, ex_top=exto, decomposition=decomp, executability=EXE, predictions=pred,
           lists={f'{nm}|{LISTN[mode]}': {d: RUN[(nm, mode)][d]['picks'] for d in days} for nm in ('C0', 'F1', 'F2', 'F3') for mode in ('base', 'opt')},
           runtime_s=time.time() - T0)
(HERE / 'FM_COUNTS.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
slot_list = {k: [[r['sig'], r['out'], r['in_']] for r in v['rows']] for k, v in BL.items()}
RES = R.load_res()
RES['COUNTS'] = dict(utc=out['utc'], judgement=judg, slot_list_sha256=hashlib.sha256(json.dumps(slot_list, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
                     fm_counts_sha256=R.sha(HERE / 'FM_COUNTS.json'))
R.save_res(RES)
print(json.dumps(judg, ensure_ascii=False, indent=0))
for k, v in BL.items(): print(k, v['slots_930'], v['slots_810'], v['by_year'], v['rows'])
print(json.dumps(dict(MDF=MDF, PM=PM, DIR=DIR, ex_top=exto, decomp=decomp, pred=pred), ensure_ascii=False, default=str)[:6000])
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'codes'} for k, v in UNKS.items()}, ensure_ascii=False))
for nm in EXE:
    for r in EXE[nm]: print(nm, r)
