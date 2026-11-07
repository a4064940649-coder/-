#!/usr/bin/env python3
"""o16 獨立複製（FINMASK 下市金融股 PIT 勘誤計數）。只依 ERRATUM_SPEC_FM.json 文字自寫；
不 import 研究員本題程式、不 import erlib。只計數名單槽位與股月，不計算任何報酬、淨值或 t 值。
執行：python3 -I -B rep_o16.py"""
import sys, json, hashlib, time
from collections import Counter
from pathlib import Path
HERE = Path('/home/user/-/research/tw_stock_selection/agents/o_finmask_delisted_fin_erratum/replicate_o16')
sys.path.insert(0, str(HERE))
t0 = time.time()
import numpy as np, pandas as pd
from scipy.stats import beta
import rpipe as R, rcls
V, K, COLS, S = R.V, R.K, R.COLS, R.S
AG = rcls.AG
DAYS = R.DAYS
D12 = [d for d in DAYS if d.year <= 2012]
assert len(D12) == 81
ds = lambda d: str(d.date())

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
ENG = {'v2_core': (AG / 'main_v2/v2_core.py', '30741b58276f8278057648b1e7302c7d335d593546a5587970a3f22021e96fc3'),
       'ia_core': (AG / 'independent_audit/ia_core.py', '0e4a3265743af32593f57d50fa445eaafaff891de78e47a3fb589abfcaacb208'),
       'ERRATUM2': (AG / 'o_rev_trunc_2009_erratum/ERRATUM2.json', '560019883d46d1a23b6f8db6633ad29bef08e819482776a1bdfc7c8cf6168ce3'),
       'o14': (AG / 'o_g8_protocol_fin_erratum/ERRATUM.json', '9f1603c34fe21bcaedb10928aa6022617515bea758f03a059a2e4d6095801d5a'),
       'SPEC_FM': (AG / 'o_finmask_delisted_fin_erratum/ERRATUM_SPEC_FM.json', '9f2d0622bb1754c1d13e1411760fd8fad9cc5f0fac40fc9281a688ce992b077d')}
for k, (p, h) in ENG.items(): assert sha(p) == h, k
out = {'no_returns': '本程式未計算任何報酬、淨值或 t 值', 'files_sha_ok': True}

# ---------------- 分類器 ----------------
C0 = R.C0
F2 = np.array([rcls.f2_fin(c, S) for c in COLS])
ST = {d: [rcls.f1_status(c, d, S) for c in COLS] for d in DAYS}
def mval(s, c0, unk=None):
    if s in ('FIN', 'FIN_Q'): return True
    if s in ('NONFIN', 'NONFIN_Q'): return False
    if unk is not None: return unk(s, c0)
    return bool(c0)
M1 = {d: np.array([mval(s, C0[j]) for j, s in enumerate(ST[d])]) for d in DAYS}
F1FIN = {d: np.array([s in ('FIN', 'FIN_Q') for s in ST[d]]) for d in DAYS}
mask = {'C0': lambda d: C0, 'F1': lambda d: M1[d], 'F2': lambda d: F2, 'F3': lambda d: M1[d] | F2}
rows = []
for d in DAYS:
    for j, c in enumerate(COLS):
        if M1[d][j] != C0[j] or F2[j] != C0[j]:
            s = ST[d][j]
            rows.append([ds(d), c, int(C0[j]), int(M1[d][j]), int(F2[j]), 'UNKNOWN' if s.startswith('UNKNOWN') else s])
fp = hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode()).hexdigest()
out['fingerprint'] = {'n_rows': len(rows), 'sha256': fp, 'match': fp == '1a2a0b3c517c91f2829e5dc962cdce437c5a63161a6949d34df9d036e89c2f09',
                      'codes_F1_diff': sorted({r[1] for r in rows if r[3] != r[2]}), 'codes_F2_diff': sorted({r[1] for r in rows if r[4] != r[2]})}

# ---------------- 有效性 ----------------
o14 = json.load(open(ENG['o14'][0]))['lists_C0']
e2 = json.load(open(ENG['ERRATUM2'][0]))
LC = {}; AC = {}
LC['C0'], _ = R.run(mask['C0'])
out['valid_C0'] = {'n_days': len(LC['C0']), 'n_slots': sum(len(v) for v in LC['C0'].values()),
                   'order_diff_days': sum(LC['C0'][d] != o14[ds(d)] for d in DAYS)}
LU = {}; AU = {}
LU['C0'], AU['C0'] = R.run(mask['C0'], mode='U')
def uslots(L, A, days=DAYS): return sorted((ds(d), c) for d in days for c in L[d] if c in A[d])
us = uslots(LU['C0'], AU['C0'])
out['valid_U'] = {'n': len(us), 'identical_to_ER2': us == sorted(tuple(x) for x in e2['U_opt_slot_list']),
                  'n_2006_2012': len(uslots(LU['C0'], AU['C0'], D12))}
# 另一寫法：V2 引擎 + FINMASK 模組屬性注入（C0 模式，全部 93 日；含 5 個抽樣日）
def v2_lists(mfn):
    v = V.V2('gapfix', R.YL, R.Y5, exfin=True, unavail=None)
    v.sig_days = list(DAYS)
    res = {}; held = set(); orig = V.FINMASK
    try:
        for d in DAYS:
            V.FINMASK = mfn(d); v._cs = {}
            cs = v.cross_section(d); el = cs['el']
            gu = K.gap_up_open(cs['e']); cand = v.order(cs, 'official')[:int(el.sum())]
            pick = []
            for j in cand:
                if gu[j] and j not in held: continue
                pick.append(j)
                if len(pick) == 10: break
            held = set(pick); res[d] = [COLS[j] for j in pick]
    finally:
        V.FINMASK = orig
    return res
for f in ('F1', 'F2', 'F3'):
    LC[f], _ = R.run(mask[f]); LU[f], AU[f] = R.run(mask[f], mode='U')
xchk = {}
for f in ('C0', 'F1', 'F2'):
    W = v2_lists(mask[f])
    xchk[f] = {'all93_identical': all(W[d] == LC[f][d] for d in DAYS),
               'sample5_identical': all(W[pd.Timestamp(x)] == LC[f][pd.Timestamp(x)] for x in ['2006-04-11', '2008-04-11', '2009-11-11', '2011-07-11', '2013-12-11'])}
out['crosscheck_V2_engine'] = xchk

# ---------------- (c) 總異動 ----------------
def changes(A, B, days=DAYS):
    det = []
    for d in days:
        a, b = set(A[d]), set(B[d])
        if a != b: det.append({'d': ds(d), 'out': sorted(a - b), 'in': sorted(b - a), 'n': len(a ^ b) // 2})
    return det
def nchg(A, B, days=DAYS): return sum(x['n'] for x in changes(A, B, days))
ratios = {}; detail = {}
for X, L in (('C0', LC), ('U_opt', LU)):
    for f in ('F1', 'F2', 'F3'):
        n930 = nchg(L['C0'], L[f]); n810 = nchg(L['C0'], L[f], D12)
        ratios[f'{X}x{f}'] = {'n930': n930, 'r930': n930 / 930, 'n810': n810, 'r810': n810 / 810}
        detail[f'{X}x{f}'] = changes(L['C0'], L[f])
judg = {k: v for k, v in ratios.items() if not k.endswith('F3')}
Mx = max(max(v['r930'], v['r810']) for v in judg.values())
out['ratios'] = ratios; out['change_detail'] = detail
out['M'] = Mx; out['M_le_5pct'] = Mx <= 0.05
by = Counter()
for k in judg:
    for x in detail[k]: by[x['d'][:4]] += x['n']
out['by_year_4combos_930'] = dict(sorted(by.items()))
tot = sum(by.values())
out['share_2010_2013'] = sum(v for y, v in by.items() if y >= '2010') / tot
out['n_2007_2009'] = sum(v for y, v in by.items() if '2007' <= y <= '2009')

# ---------------- (a) 股池、(b) 直接污染 ----------------
DEL = {c: rcls.is_delisted(c, S) for c in COLS}
def finF(f, d):
    return F1FIN[d] if f == 'F1' else F2
pool = {}
for f in ('F1', 'F2'):
    cnt = Counter(); cel = Counter()
    for d in DAYS:
        b = R.base(d); el0 = b['pre'] & np.isfinite(np.where(C0, np.nan, b['y3']))
        sel = b['top'] & ~C0 & finF(f, d) & np.array([DEL[c] for c in COLS])
        for j in np.where(sel)[0]:
            cnt[COLS[j]] += 1; cel[COLS[j]] += int(el0[j])
    pool[f] = {c: [cnt[c], cel[c]] for c in sorted(cnt)}
# 28xx 下市股（C0 已遮）
c28 = Counter()
for d in DAYS:
    b = R.base(d)
    for j in np.where(b['top'])[0]:
        c = COLS[j]
        if c[:2] == '28' and DEL[c]: c28[c] += 1
pool['delisted_28xx_C0masked'] = {'n_codes': len(c28), 'n_months': sum(c28.values()), 'codes': dict(sorted(c28.items())),
                                  'all_C0_masked': all(C0[R.CI[c]] for c in c28)}
out['pool'] = pool
direct = {}
for f in ('F1', 'F2'):
    codes = set(pool[f])
    direct[f] = {'C0': [(ds(d), c, LC['C0'][d].index(c) + 1) for d in DAYS for c in LC['C0'][d] if c in codes],
                 'U_opt': [(ds(d), c) for d in DAYS for c in LU['C0'][d] if c in codes]}
    direct[f]['U_opt_by_code'] = dict(Counter(c for _, c in direct[f]['U_opt']))
out['direct'] = direct
out['n5854_in_C0'] = sum('5854' in LC['C0'][d] for d in DAYS)

# ---------------- 描述：去最大貢獻股、拆解 ----------------
def code_mask(f, codes):
    """只把 codes 套用 F，其餘維持 C0"""
    idx = np.array([R.CI[c] for c in codes], int)
    def m(d):
        x = C0.copy(); full = mask[f](d)
        if len(idx): x[idx] = full[idx]
        return x
    return m
diffc = {'F1': out['fingerprint']['codes_F1_diff'], 'F2': out['fingerprint']['codes_F2_diff']}
def runX(X, mfn):
    return R.run(mfn)[0] if X == 'C0' else R.run(mfn, mode='U')[0]
extop = {}; decomp = {}
for X, L in (('C0', LC), ('U_opt', LU)):
    for f in ('F1', 'F2'):
        single = {c: nchg(L['C0'], runX(X, code_mask(f, [c]))) for c in diffc[f]}
        top = sorted(single, key=lambda c: (-single[c], c))[0]
        rest = [c for c in diffc[f] if c != top]
        n = nchg(L['C0'], runX(X, code_mask(f, rest)))
        extop[f'{X}x{f}'] = {'single': single, 'top': top, 'n930_ex_top': n, 'r930': n / 930}
        dl = [c for c in diffc[f] if c in pool[f]]; ot = [c for c in diffc[f] if c not in pool[f]]
        decomp[f'{X}x{f}'] = {'delisted_codes': dl, 'n_delisted': nchg(L['C0'], runX(X, code_mask(f, dl))),
                              'other_codes': ot, 'n_other': nchg(L['C0'], runX(X, code_mask(f, ot)))}
out['ex_top'] = extop; out['decomp'] = decomp

# ---------------- UNKNOWN ----------------
unk = {}
for kind in ('UNKNOWN_EVID', 'UNKNOWN_NOEVID'):
    cnt = Counter(); sC = 0; sU = 0
    for d in DAYS:
        b = R.base(d)
        for j in np.where(b['top'])[0]:
            if ST[d][j] == kind: cnt[COLS[j]] += 1
        sC += sum(ST[d][R.CI[c]] == kind for c in LC['C0'][d])
        sU += sum(ST[d][R.CI[c]] == kind for c in LU['C0'][d])
    unk[kind] = {'n_codes_in_pool': len(cnt), 'pool_months': sum(cnt.values()), 'codes': dict(sorted(cnt.items())), 'slots_C0': sC, 'slots_U_opt': sU}
unk['mask_eq_C0'] = True
# 審查的雙向處理
alt = {'evid_fin': lambda s, c0: True if s == 'UNKNOWN_EVID' else bool(c0), 'all_nonfin': lambda s, c0: False}
for k, fn in alt.items():
    Mk = {d: np.array([mval(s, C0[j], fn) for j, s in enumerate(ST[d])]) for d in DAYS}
    mk = lambda d, Mk=Mk: Mk[d]
    unk['F1_' + k] = {'C0': nchg(LC['C0'], R.run(mk)[0]), 'U_opt': nchg(LU['C0'], R.run(mk, mode='U')[0]),
                      'U_opt_810': nchg(LU['C0'], R.run(mk, mode='U')[0], D12)}
# 差異說明用變體：快照 TSE 產業別空白者視為「該快照沒有分類」（規格字面「其他 → NONFIN」時我把空白當 NONFIN）
blank = set()
for f in ['goristock_14393dd2_grs_stock_no.csv', 'goristock_d9d8662c_grs_stock_no.csv', 'goristock_235da71f_grs_old_stock_no.csv',
          'goristock_c3c718ab_grs_stock_no.csv', 'grs_893c8772_grs_stock_no.csv']:
    _, mm = rcls._snap_goristock(rcls.SNAP / f); blank |= {(f, c) for c, (n, k) in mm.items() if k is None}
S2 = dict(S); S2['snaps'] = [(dt, {c: v for c, v in mm.items() if (f, c) not in blank}, f) for dt, mm, f in S['snaps']]
cntv = Counter(); fpv = 0
for d in DAYS:
    b = R.base(d)
    for j in np.where(b['top'])[0]:
        if rcls.f1_status(COLS[j], d, S2) == 'UNKNOWN_NOEVID': cntv[COLS[j]] += 1
    for j, c in enumerate(COLS):
        if c in ('2841', '5854', '5871', '6004', '6012'):
            fpv += mval(rcls.f1_status(c, d, S2), C0[j]) != M1[d][j]
unk['variant_blank_tse_as_missing'] = {'UNKNOWN_NOEVID_codes': len(cntv), 'pool_months': sum(cntv.values()),
                                       'extra_codes': sorted(set(cntv) - set(unk['UNKNOWN_NOEVID']['codes'])), 'mask_diff_vs_main': fpv}
out['unknown'] = unk
out['binom_14_810_upper95'] = float(beta.ppf(0.95, 14 + 1, 810 - 14))

# ---------------- 文字更正 E1～E6 ----------------
er = {}
def ucount(mfn, **kw):
    L, A = R.run(mfn, mode='U', **kw)
    return len(uslots(L, A, D12)), len(uslots(L, A)), (L, A)
a, b, _ = ucount(mask['C0'], gov=('TRUNC',)); er['E1_TRUNC'] = [a, b]
a, b, _ = ucount(mask['C0'], gov=('TRUNC',), drop={'5854'}); er['E1_TRUNC_ex5854'] = [a, b]
a, b, _ = ucount(mask['C0'], gov=('TRUNC',), drop={'5854', '2473'}); er['E1_TRUNC_ex5854_2473'] = [a, b]
for f in ('F1', 'F2'):
    a, b, _ = ucount(mask[f], gov=('TRUNC',)); er[f'E1_TRUNC_{f}'] = [a, b]
# E2
lm = S['lm']
affc = sorted({c for d in DAYS for c in AU['C0'][d]})
listed = [c for c in affc if (not isinstance(lm['delist_date'].get(c) if c in lm.index else None, str)) and c in S['sl26'].index]
ins = Counter(c for _, c in us)
er['E2_listed_affected'] = {c: ins.get(c, 0) for c in listed}
rel = {c: 'LISTED_GAP' for c in listed}
a, b, _ = ucount(mask['C0'], relabel=rel); er['E2_relabel_all'] = [a, b]
a, b, _ = ucount(mask['C0'], relabel={'2915': 'LISTED_GAP'}); er['E2_relabel_2915_only'] = [a, b]
if '3454' in lm.index:
    er['E2_3454_lm'] = {k: lm.at['3454', k] for k in ('listed_date', 'delist_date', 'last_listed_day')}
    j = R.CI['3454']; w = np.where(K.TRADED[:, j])[0]
    er['E2_3454_panel_last'] = ds(K.IDX[w[-1]])
    er['E2_3454_last10_close'] = [round(float(x), 2) for x in K.Cn[w[-10:], j]]
    er['E2_3454_rev_last'] = str(R.LAST.get('3454'))
# E3
dsp = K.DSP
er['E3_disposal5y_min_start'] = str(dsp['start'].min().date())
er['E3_disposal5y_hits_2006_2013'] = int(sum(K.disposal_mask(d)[R.base(d)['top']].sum() for d in DAYS))
offt = pd.read_csv(AG / 'disposal_reconcile/official_trades_twse.csv', dtype={'code': str})
er['E3_official_trades_twse_min_ann'] = str(offt['ann_date'].min())
hist = pd.read_csv(AG / 'o_disposal_rejudge/disposal_hist_rejudge.csv', dtype={'code': str}, parse_dates=['start', 'end'])
er['E3_hist_min_start'] = str(hist['start'].min().date())
er['E3_C0_slots_in_hist_disposal'] = [(ds(d), c) for d in DAYS for c in LC['C0'][d]
                                      if ((hist['code'] == c) & (hist['start'] <= d) & (hist['end'] >= d)).any()]
# E4
dS = [d for d in DAYS if V.deadline_case(d) == 'D=S']
er['E4'] = {'n_days': len(dS), 'n_days_2006_2012': sum(d.year <= 2012 for d in dS), 'C0_slots': sum(len(LC['C0'][d]) for d in dS),
            'U_opt_put_back_slots': sum(1 for d in dS for c in LU['C0'][d] if c in AU['C0'][d])}
# E6
fin_pfx = {c for c in COLS if c[:2] in ('28', '58', '60')}
a, b, _ = ucount(mask['C0'], drop=fin_pfx); er['E6_drop_28_58_60'] = [a, b]
for f in ('F1', 'F2'):
    a, b, _ = ucount(mask[f]); er[f'E6_U_opt_put_back_{f}'] = [a, b]
out['errata'] = er
out['runtime_s'] = time.time() - t0
json.dump(out, open(HERE / 'rep_o16_results.json', 'w'), ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: out[k] for k in ('fingerprint', 'valid_C0', 'valid_U', 'crosscheck_V2_engine', 'M')}, ensure_ascii=False, default=str))
for k, v in ratios.items(): print(k, v['n930'], round(v['r930'] * 100, 2), v['n810'], round(v['r810'] * 100, 2))
print('runtime', out['runtime_s'])
