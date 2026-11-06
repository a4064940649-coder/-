#!/usr/bin/env python3
"""複製：只依 ERRATUM2_SPEC.json 的文字重寫原因分類、百分位、排序、遞補與計數。
不 import erlib／s4；只從 v2_core 取股池素材（V2 物件的 S52、V60、sig_days、FINMASK）與 ia_core 的盤面陣列、處置、開盤漲停判斷。
營收年增率由 tw_revenue.csv 水準自行計算。用法：python3 -I -B replicate/r_erratum2.py"""
import sys, json, csv, hashlib, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
TOP = HERE.parent
AG = TOP.parent
SPEC = json.loads((TOP / 'ERRATUM2_SPEC.json').read_text(encoding='utf-8'))
sys.path.insert(0, str(AG / 'main_v2'))
import numpy as np, pandas as pd
import v2_core as V
sys.path.pop(0)
K = V.K
T0 = time.time()
COLS = list(K.COLS); N = len(COLS); IX = {c: i for i, c in enumerate(COLS)}

# ---- 營收（自算）----
R = pd.read_csv('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv', dtype={'code': str})
R['m'] = pd.to_datetime(R['month']).dt.to_period('M')
R = R.assign(o=(R.market != 'sii').astype(int)).sort_values(['m', 'code', 'o']).drop_duplicates(['m', 'code'])
LEV = R.pivot(index='m', columns='code', values='revenue_k_twd')
LEV = LEV.reindex(pd.period_range(LEV.index[0], LEV.index[-1], freq='M'))
B = LEV.shift(12)
YOY = ((LEV / B - 1) * 100).where(B > 0).reindex(columns=COLS)
YOYa = YOY.to_numpy(); MI = {p: i for i, p in enumerate(YOY.index)}
def y3_of(m):
    ii = [MI.get(m - k) for k in (1, 2, 3)]
    if None in ii: return np.full(N, np.nan)
    Z = YOYa[ii]
    return np.where(np.isnan(Z).any(0), np.nan, Z.mean(0))
nonnull = {c: LEV[c].dropna().index for c in LEV.columns}
FIRST = {c: s.min() for c, s in nonnull.items() if len(s)}; LAST = {c: s.max() for c, s in nonnull.items() if len(s)}

# ---- 名稱（外國股判斷）----
names = {}
for c, n in zip(R.code, R.name): names.setdefault(c, set()).add(str(n))
for r in csv.DictReader(open(K.X.parent / 'stock_list.csv', encoding='utf-8')): names.setdefault(r['code'], set()).add(r['name'])
for f in ('grs_568e6685_tools_twse_list.csv', 'grs_893c8772_grs_stock_no.csv', 'goristock_c3c718ab_grs_stock_no.csv'):
    for row in csv.reader(open(AG / 'o_pit_shares_industry_2006_2013/raw/industry_snap' / f, encoding='utf-8', errors='replace')):
        row = [x.strip().strip('"') for x in row]
        if len(row) > 1 and len(row[0]) == 4 and row[0].isdigit(): names.setdefault(row[0], set()).add(row[1])
def foreign(c):
    return c.startswith('91') or any('KY' in n.upper() or n.endswith('DR') or n.startswith('F-') for n in names.get(c, ()))
lastm = {}
for c in COLS:
    w = np.flatnonzero(K.TRADED[:, IX[c]])
    if len(w): lastm[c] = pd.Period(K.IDX[w[-1]], 'M')

def why(c, m):
    if foreign(c): return 'FOREIGN'
    if c not in FIRST: return 'NEVER_DOM'
    tags = []
    for k in (1, 2, 3, 13, 14, 15):
        x = m - k
        v = LEV[c].get(x, np.nan)
        if np.isnan(v):
            if x > LAST[c]:
                if pd.Period('2009-12', 'M') <= lastm[c] <= pd.Period('2012-12', 'M'): tags.append('TRUNC')
                elif x >= pd.Period('2013-01', 'M') and LAST[c] >= pd.Period('2012-12', 'M'): tags.append('IFRS13')
                else: tags.append('PRE12')
            elif x < FIRST[c]: tags.append('NEWLIST')
            else: tags.append('HOLE')
        elif v <= 0 and k >= 13: tags.append('BASE_LE0')
    for t in ('TRUNC', 'IFRS13', 'PRE12', 'HOLE', 'NEWLIST', 'BASE_LE0'):
        if t in tags: return t
    return 'UNKNOWN'
GOV = set(SPEC['affected_set']['governing_causes'])

def avgrank(x, mask):
    """mask 內的平均名次（1 起算），其他 NaN。"""
    out = np.full(len(x), np.nan)
    idx = np.flatnonzero(mask)
    if not len(idx): return out
    s = pd.Series(x[idx]); out[idx] = s.rank(method='average').to_numpy()
    return out

S = V.V2('gapfix', V.load_revenue()[0], V.load_revenue()[1], exfin=True)   # 只取股池素材
LO, HI = '2006-01-01', '2013-12-31'

def build(opt):
    held = set(); res = {}
    for d in S.sig_days:
        if str(d.date()) > HI: break
        t = K.IDX.get_loc(d); e = t + 1; m = pd.Period(d, 'M')
        v60 = S.V60[t]; px = K.Cn[t]
        ok = np.isfinite(v60) & (v60 >= 1e7) & np.isfinite(px) & (px >= 10)
        rk = np.argsort(-np.where(ok, v60, -np.inf), kind='stable')[:min(300, int(ok.sum()))]
        pool = np.zeros(N, bool); pool[rk] = True; pool &= ok
        s52 = S.S52[t]
        y = y3_of(m) if d < V.SWITCH else S.y3(d)
        y = np.where(V.FINMASK, np.nan, y)
        base = pool & np.isfinite(s52) & ~K.disposal_mask(d) & np.isfinite(K.On[e]) & K.TRADED[e]
        el = base & np.isfinite(y)
        A = np.zeros(N, bool)
        ds = str(d.date())
        if opt and LO <= ds <= HI:
            for j in np.flatnonzero(base & ~V.FINMASK & ~np.isfinite(y)):
                if why(COLS[j], m) in GOV: A[j] = True
        E2 = el | A
        n = E2.sum()
        if n < 50:
            continue
        p52 = avgrank(s52, E2) / n
        prev = avgrank(y, el) / n
        prev[A] = 1.0
        score = 0.5 * p52 + 0.5 * prev
        ytb = np.where(A, np.inf, y)
        cand = [j for j in np.flatnonzero(E2)]
        cand.sort(key=lambda j: (-round(score[j], 10), -ytb[j], j))
        gu = K.gap_up_open(e)
        pick = []
        for j in cand:
            if gu[j] and j not in held: continue
            pick.append(j)
            if len(pick) == 10: break
        held = set(pick)
        if LO <= ds <= HI:
            res[ds] = dict(picks=[COLS[j] for j in pick], A={COLS[j] for j in np.flatnonzero(A)})
    return res

b = build(False)
L0 = json.loads((AG / 'o_g8_protocol_fin_erratum/ERRATUM.json').read_text(encoding='utf-8'))['lists_C0']
mism = [d for d in L0 if b.get(d, {}).get('picks') != L0[d]]
print('baseline mismatches', len(mism), mism[:5], flush=True)
o = build(True)
slots = sorted((d, c) for d in o for c in o[d]['picks'] if c in o[d]['A'])
u12 = sum(1 for d, c in slots if d < '2013-01-01'); u13 = len(slots)
nd12 = sum(1 for d in o if d < '2013-01-01')
mine = dict(n_days=len(o), n_days_2006_2012=nd12, U_2006_2012=u12, U_2006_2013=u13, share_2006_2012=u12 / (10 * nd12), share_2006_2013=u13 / (10 * len(o)),
            governing=max(u12 / (10 * nd12), u13 / (10 * len(o))), baseline_mismatch=len(mism),
            k5854=[d for d in o if '5854' in o[d]['picks']], k6012=[d for d in o if '6012' in o[d]['picks']])
ref = json.loads((TOP / 'ERRATUM2.json').read_text(encoding='utf-8'))
refslots = sorted(tuple(x) for x in ref['U_opt_slot_list'])
mine['slot_list_identical'] = (slots == refslots)
mine['only_mine'] = sorted(set(slots) - set(refslots)); mine['only_ref'] = sorted(set(refslots) - set(slots))
mine['runtime_s'] = time.time() - T0
(HERE / 'replica.json').write_text(json.dumps(mine, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps({k: v for k, v in mine.items() if k not in ('only_mine', 'only_ref')}, ensure_ascii=False), len(mine['only_mine']), len(mine['only_ref']))
