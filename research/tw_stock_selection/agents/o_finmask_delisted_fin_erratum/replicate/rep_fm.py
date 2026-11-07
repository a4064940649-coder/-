#!/usr/bin/env python3
"""G6 複製（從頭重寫計數）：只讀 ERRATUM_SPEC_FM.json 與原始資料檔；不 import 本題的 freg／fclass／flib，也不 import erlib。
只用共用引擎原語：ia_core（面板陣列、換股日曆、開盤漲停、處置遮罩、營收年增率）與 v2_core.rolling_signals('gapfix')、v2_core.load_revenue、v2_core.FINMASK（C0 定義本身）。
分類器、遮罩、股池、合格集合、百分位、排序、遞補、U_opt 放回、原因分類、計數全部自己寫。用法：python3 -I -B replicate/rep_fm.py"""
import sys, csv, json, re, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
TOP = HERE.parent
AG = TOP.parent
SPEC = json.loads((TOP / 'ERRATUM_SPEC_FM.json').read_text(encoding='utf-8'))
sys.path.insert(0, str(AG / 'main_v2'))
import numpy as np, pandas as pd
import v2_core as V
sys.path.pop(0)
K = V.K
IDX, COLS = K.IDX, list(K.COLS); N = len(COLS); CI = {c: j for j, c in enumerate(COLS)}
O13 = AG / 'o_pit_shares_industry_2006_2013/raw'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

# ---------- 分類來源（依 spec） ----------
def four(x): return isinstance(x, str) and len(x) == 4 and x.isdigit()
snaps = []
for rel, eff in SPEC['classifiers']['snapshot_effective_dates'].items():
    fin, nm = {}, {}
    with open(O13 / rel, encoding='utf-8', errors='replace') as fh:
        rd = csv.reader(fh); next(rd)
        for r in rd:
            r = [x.strip().strip('"').strip('﻿') for x in r]
            if len(r) >= 3 and four(r[0]) and re.fullmatch(r'\d+', r[2] or ''):
                fin[r[0]] = int(r[2]) == SPEC['classifiers']['tse_fin_code']; nm[r[0]] = r[1]
    snaps.append((pd.Timestamp(eff), fin, nm))
snaps.sort(key=lambda x: x[0])
FL = set(SPEC['classifiers']['fin_labels'])
later = []
for rel, eff in SPEC['classifiers']['later_sources_dates'].items():
    T = pd.read_csv(O13 / rel, dtype=str, on_bad_lines='skip')
    cc, nn, ii = ('code', 'name', 'industry') if 'code' in T.columns else ('代號', '名稱', '產業別')
    fin = {str(a).strip(): (str(b).strip() in FL) for a, b in zip(T[cc], T[ii]) if four(str(a).strip()) and isinstance(b, str)}
    nm = {str(a).strip(): n for a, n in zip(T[cc], T[nn]) if four(str(a).strip())}
    later.append((pd.Timestamp(eff), fin, nm))
SLp = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv')
SLT = pd.read_csv(SLp, dtype=str).drop_duplicates('code')
later.append((pd.Timestamp('2026-10-01'), {a: (b in FL) for a, b in zip(SLT['code'], SLT['industry']) if four(a) and isinstance(b, str)},
              {a: n for a, n in zip(SLT['code'], SLT['name']) if four(a)}))
SLSET = set(SLT['code'])
names = {}
def addn(c, n):
    if four(c) and isinstance(n, str) and n.strip(): names.setdefault(c, set()).add(n.strip())
for _, _, nm in snaps + later:
    for c, n in nm.items(): addn(c, n)
REVP = Path('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv')
RV = pd.read_csv(REVP, dtype={'code': str, 'market': str})
for c, n in RV[['code', 'name']].drop_duplicates().itertuples(index=False): addn(c, n)
for r in csv.DictReader(open(AG / 'o_lc_sector_rotation_pit/raw/delist_official.csv', encoding='utf-8')): addn(r['code'], r['name'])
LMr = {r['code']: r for r in csv.DictReader(open(AG / 'o_v7_listing_exec_engine/listing_master.csv', encoding='utf-8'))}
KW = SPEC['classifiers']['keywords']; EX2 = set(SPEC['classifiers']['f2_manual_exclusion']); NEX = set(SPEC['classifiers']['name_exclusion'])
def kwhit(c): return any(any(k in n for k in KW) for n in names.get(c, ()) if n not in NEX)
def evid(c): return c[:2] in ('28', '58', '60') or kwhit(c)
def f2(c): return evid(c) and c not in EX2
def f1(c, d):
    b = [s for s in snaps if s[0] <= d and c in s[1]]
    if b: return 'FIN' if b[-1][1][c] else 'NONFIN'
    a = sorted([s for s in snaps + later if s[0] > d and c in s[1]], key=lambda s: s[0])
    if not a: return 'UNKNOWN'
    fn = a[0][1][c]; ev = evid(c)
    return 'FIN_Q' if (fn and ev) else ('NONFIN_Q' if (not fn and not ev) else 'UNKNOWN')
C0 = V.FINMASK.copy()
def delisted(c): return (c not in SLSET) or bool(LMr.get(c, {}).get('delist_date', ''))

# ---------- 換股日 ----------
YL = V.load_revenue()[0]
cnt = YL.notna().sum(axis=1); lmL = cnt[cnt >= 0.8 * cnt.median()].index.max()
days = []
for d in K.rebalance_days('2005-01-01'):
    m = pd.Period(d, 'M')
    if (m - 1) > lmL or d >= V.SWITCH: continue
    if np.isfinite(K.yoy3_for_month(YL, m)).sum() > 100: days.append(d)
days = [d for d in days if pd.Timestamp('2006-01-01') <= d <= pd.Timestamp('2013-12-31')]
assert len(days) == 93, len(days)
DS = [str(d.date()) for d in days]
f1st = {d: {c: f1(c, d) for c in COLS} for d in days}
def maskv(nm, d):
    if nm == 'C0': return C0
    if nm == 'F2': return np.array([f2(c) for c in COLS])
    s = f1st[d]
    m1 = np.array([True if s[c] in ('FIN', 'FIN_Q') else (False if s[c] in ('NONFIN', 'NONFIN_Q') else bool(C0[j])) for j, c in enumerate(COLS)])
    return m1 if nm == 'F1' else (m1 | np.array([f2(c) for c in COLS]))
# 指紋
fp = []
for d, ds in zip(days, DS):
    a1 = maskv('F1', d); a2 = maskv('F2', d)
    for j, c in enumerate(COLS):
        if a1[j] != C0[j] or a2[j] != C0[j]: fp.append([ds, c, int(C0[j]), int(a1[j]), int(a2[j]), f1st[d][c]])
fp_ok = hashlib.sha256(json.dumps(fp, ensure_ascii=False).encode()).hexdigest() == SPEC['classifiers']['fingerprint']['sha256']

# ---------- 營收水準與原因（ER2 定義重寫） ----------
RV['p'] = pd.PeriodIndex(pd.to_datetime(RV['month']), freq='M')
RV['pri'] = (RV['market'] != 'sii').astype(int)
LVd = RV.sort_values(['p', 'code', 'pri']).drop_duplicates(['p', 'code']).pivot(index='p', columns='code', values='revenue_k_twd')
LVd = LVd.reindex(pd.period_range(LVd.index.min(), LVd.index.max(), freq='M'))
LR = {c: LVd[c].dropna().index.max() for c in LVd.columns}; FR = {c: LVd[c].dropna().index.min() for c in LVd.columns}
LTm = {}
for j, c in enumerate(COLS):
    w = np.where(K.TRADED[:, j])[0]; LTm[c] = pd.Period(IDX[w[-1]], 'M') if len(w) else None
def foreign(c): return c[:2] == '91' or any(('KY' in n.upper()) or n.endswith('DR') or n.startswith('F-') for n in names.get(c, ()))
def mcause(c, x):
    if x > LR[c]:
        if pd.Period('2009-12', 'M') <= LTm[c] <= pd.Period('2012-12', 'M'): return 'TRUNC'
        if x >= pd.Period('2013-01', 'M') and LR[c] >= pd.Period('2012-12', 'M'): return 'IFRS13'
        return 'PRE12'
    return 'NEWLIST' if x < FR[c] else 'HOLE'
ORD = ['TRUNC', 'IFRS13', 'PRE12', 'HOLE', 'NEWLIST', 'BASE_LE0']
def cause(c, m):
    if foreign(c): return 'FOREIGN'
    if c not in LVd.columns: return 'NEVER_DOM'
    ks = set()
    for k in (1, 2, 3, 13, 14, 15):
        x = m - k; v = LVd[c].get(x, np.nan)
        if not np.isfinite(v): ks.add(mcause(c, x))
        elif v <= 0 and k >= 13: ks.add('BASE_LE0')
    return next((k for k in ORD if k in ks), 'UNKNOWN')
SURV = set(SPEC['pipeline']['U_opt']['affected_set']['governing_causes'])

# ---------- 名單（自寫排序與遞補） ----------
S52, V60 = V.rolling_signals('gapfix')
def pool(t):
    v = V60[t]; px = K.Cn[t]
    ok = np.isfinite(v) & (v >= 1e7) & np.isfinite(px) & (px >= 10)
    o = np.argsort(-np.where(ok, v, -np.inf), kind='stable')[:min(300, ok.sum())]
    p = np.zeros(N, bool); p[o] = True; return p & ok
def lists(nm, opt):
    held = set(); res = {}
    for d, ds in zip(days, DS):
        t = IDX.get_loc(d); e = t + 1; m = pd.Period(d, 'M')
        mk = maskv(nm, d); top = pool(t)
        y = K.yoy3_for_month(YL, m); y = np.where(mk, np.nan, y)
        s = S52[t]
        base = top & np.isfinite(s) & ~K.disposal_mask(d) & np.isfinite(K.On[e]) & K.TRADED[e]
        el = base & np.isfinite(y)
        A = np.zeros(N, bool)
        if opt:
            for j in np.where(base & ~mk & ~np.isfinite(y))[0]:
                if cause(COLS[j], m) in SURV: A[j] = True
        E2 = el | A; n = E2.sum()
        ii = np.where(E2)[0]
        sp = pd.Series(s[ii]).rank(pct=True).to_numpy()
        if opt:
            nn = ii[~A[ii]]
            rr = pd.Series(y[nn]).rank().to_numpy() / n
            rv = np.ones(len(ii)); rv[~A[ii]] = rr
            yt = np.where(A[ii], np.inf, y[ii])
        else:
            rv = pd.Series(y[ii]).rank(pct=True).to_numpy(); yt = y[ii]
        sc = np.round(0.5 * sp + 0.5 * rv, 10)
        order = sorted(range(len(ii)), key=lambda k: (-sc[k], -yt[k], ii[k]))
        gu = K.gap_up_open(e); pk = []
        for k in order:
            j = ii[k]
            if gu[j] and j not in held: continue
            pk.append(j)
            if len(pk) == 10: break
        held = set(pk)
        res[ds] = dict(picks=[COLS[j] for j in pk], A={COLS[j] for j in np.where(A)[0]}, top=top, el=el)
    return res
RUN = {(nm, o): lists(nm, o) for nm in ('C0', 'F1', 'F2', 'F3') for o in (False, True)}
L0 = json.loads((AG / 'o_g8_protocol_fin_erratum/ERRATUM.json').read_text(encoding='utf-8'))['lists_C0']
c0_ok = all(RUN[('C0', False)][d]['picks'] == L0[d] for d in DS)
ER2 = json.loads((AG / 'o_rev_trunc_2009_erratum/ERRATUM2.json').read_text(encoding='utf-8'))
ins = sorted([d, c] for d in DS for c in RUN[('C0', True)][d]['picks'] if c in RUN[('C0', True)][d]['A'])
uopt_ok = ins == sorted(map(list, ER2['U_opt_slot_list']))
def blk(a, b):
    rows = []
    for d in DS:
        x, y = set(a[d]['picks']), set(b[d]['picks'])
        if x != y: rows.append([d, sorted(x - y), sorted(y - x), len(x ^ y) / 2])
    s9 = sum(r[3] for r in rows); s8 = sum(r[3] for r in rows if r[0] < '2013')
    return dict(s930=s9, s810=s8, share_930=s9 / 930, share_810=s8 / 810, rows=rows)
B = {f"{'U_opt' if o else 'C0'}|{nm}": blk(RUN[('C0', o)], RUN[(nm, o)]) for nm in ('F1', 'F2', 'F3') for o in (False, True)}
gov = {f'{k}|{w}': B[k][f'share_{w}'] for k in ('C0|F1', 'C0|F2', 'U_opt|F1', 'U_opt|F2') for w in ('930', '810')}
M = max(gov.values())
# 下市金融股（C0 未遮）與直接污染、股池股月
mdf = {}
for nm in ('F1', 'F2'):
    s = set()
    for d in days:
        mk = maskv(nm, d); s |= {COLS[j] for j in np.where(mk & ~C0)[0] if delisted(COLS[j])}
    mdf[nm] = sorted(s)
direct = {nm: {lst: sorted([d, c] for d in DS for c in RUN[('C0', o)][d]['picks'] if c in mdf[nm]) for lst, o in (('C0', False), ('U_opt', True))} for nm in mdf}
pm = {nm: {c: sum(1 for d, dd in zip(DS, days) if RUN[('C0', False)][d]['top'][CI[c]] and maskv(nm, dd)[CI[c]]) for c in mdf[nm]} for nm in mdf}
out = dict(fingerprint_ok=fp_ok, n_fp_rows=len(fp), C0_identical=c0_ok, U_opt_identical_ER2=uopt_ok, n_U_opt=len(ins), ratios=gov, M=M,
           blocks=B, missed_delisted_fin=mdf, direct=direct, pool_months=pm)
# 與研究員比對
FM = json.loads((TOP / 'FM_COUNTS.json').read_text(encoding='utf-8'))
cmp = {}
for k in B:
    mine = [[r[0], r[1], r[2]] for r in B[k]['rows']]
    theirs = [[r['sig'], r['out'], r['in_']] for r in FM['blocks'][k]['rows']]
    cmp[k] = dict(same=(mine == theirs), mine_n=B[k]['s930'], theirs_n=FM['blocks'][k]['slots_930'])
cmp['ratios_same'] = all(abs(gov[k] - FM['judgement']['ratios'][k]) < 1e-12 for k in gov)
cmp['direct_same'] = all(sorted(map(list, [[d, c] for d, c, _ in FM['direct'][nm][lst]])) == direct[nm][lst] for nm in mdf for lst in ('C0', 'U_opt'))
cmp['pool_months_same'] = all(pm[nm][c] == FM['pool_months'][nm][c]['pool'] for nm in mdf for c in mdf[nm])
out['compare'] = cmp
(HERE / 'rep_fm_results.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps(dict(fp=fp_ok, c0=c0_ok, uopt=uopt_ok, M=M, gov=gov, mdf=mdf, pm=pm, cmp=cmp), ensure_ascii=False, indent=0))
