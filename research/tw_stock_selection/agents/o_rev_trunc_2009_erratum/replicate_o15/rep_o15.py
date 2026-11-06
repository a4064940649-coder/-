#!/usr/bin/env python3
"""o15 獨立複製（只依 ERRATUM2_SPEC.json）：受影響集合原因分類、U_opt／U_p90／U_carry／ALL 排序與計數。
只 import main_v2/v2_core（股池素材、sig_days、S52、V60、plan 的成交規則）。不讀研究員的 erlib／s4。"""
import sys, json, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
AG = HERE.parent.parent
sys.path.insert(0, str(AG / 'main_v2'))
import numpy as np, pandas as pd
import v2_core as V2C
sys.path.pop(0)
K = V2C.K
COLS, N, IDX = V2C.COLS, V2C.N, V2C.IDX
CI = {c: j for j, c in enumerate(COLS)}

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
SPEC = HERE.parent / 'ERRATUM2_SPEC.json'
assert sha(SPEC) == '722f7d56eec3674980445d179996c8d3f04226d91fe502b7a0f14260c846126e'
spec = json.loads(SPEC.read_text(encoding='utf-8'))
assert sha(AG / 'main_v2/v2_core.py') == spec['base']['v2_core_sha256']
assert sha(K.LONG) == spec['base']['tw_revenue_sha256']
ERJ = AG / 'o_g8_protocol_fin_erratum/ERRATUM.json'
assert sha(ERJ) == spec['base']['erratum_json_sha256']

# ---------- 基準 ----------
YL, Y5, _ = V2C.load_revenue()
V = V2C.V2('gapfix', YL, Y5, exfin=True)
base_plan = V.plan(10, tie='official')
lists_C0 = json.loads(ERJ.read_text(encoding='utf-8'))['lists_C0']
B = {}
for (e, pk, elx, d) in base_plan:
    if pd.Timestamp('2006-01-01') <= d <= pd.Timestamp('2013-12-31'):
        B[d] = [COLS[j] for j in pk]
assert len(B) == 93, len(B)
nmis = 0
for d, l in B.items():
    ref = lists_C0[str(d.date())]
    if list(ref) != l: nmis += 1
assert nmis == 0 and set(lists_C0) == {str(d.date()) for d in B}
DAYS = sorted(B)

# ---------- 營收水準（自己讀） ----------
raw = pd.read_csv(K.LONG, dtype={'code': str, 'market': str})
raw['p'] = pd.PeriodIndex(pd.to_datetime(raw['month']), freq='M')
names = {}
for c, nm in zip(raw['code'], raw['name']):
    names.setdefault(c, set()).add(str(nm))
for c, nm in V2C.SL['name'].items():
    names.setdefault(c, set()).add(str(nm))
import re
_iso = (AG / 'o_v7_listing_exec_engine/raw/isin/mtw/twseStock/log.txt').read_text(encoding='utf-8', errors='ignore')
for c, nm in re.findall(r'(\d{4,6}) ([^ ,\[\]()]+) \(', _iso):   # ISIN 名單（listing_master 來源之一）的名稱
    names.setdefault(c, set()).add(nm)
raw['pri'] = (raw['market'] != 'sii').astype(int)
r1 = raw.sort_values(['p', 'code', 'pri']).drop_duplicates(['p', 'code'])
LV = r1.pivot(index='p', columns='code', values='revenue_k_twd')
LV = LV.reindex(pd.period_range(LV.index.min(), LV.index.max(), freq='M'))
LVn = LV.notna()
FIRST = {c: LV.index[LVn[c].to_numpy()].min() for c in LV.columns if LVn[c].any()}
LAST = {c: LV.index[LVn[c].to_numpy()].max() for c in LV.columns if LVn[c].any()}

def foreign(c):
    if c[:2] == '91': return True
    for nm in names.get(c, ()):
        if 'KY' in nm or 'DR' in nm or nm.startswith('F-'): return True
    return False

tr = K.TRADED
lastidx = np.array([np.where(tr[:, j])[0].max() if tr[:, j].any() else -1 for j in range(N)])
LASTTRADE_M = {COLS[j]: pd.Period(IDX[lastidx[j]], 'M') for j in range(N) if lastidx[j] >= 0}
P0912, P1212, P1301 = pd.Period('2009-12', 'M'), pd.Period('2012-12', 'M'), pd.Period('2013-01', 'M')
PRI = ['TRUNC', 'IFRS13', 'PRE12', 'HOLE', 'NEWLIST', 'BASE_LE0']

def lev(c, x):
    if c not in LV.columns or x not in LV.index: return np.nan
    return LV.at[x, c]

def cause(c, m):
    if foreign(c): return 'FOREIGN'
    if c not in FIRST: return 'NEVER_DOM'
    six = [m - 1, m - 2, m - 3, m - 13, m - 14, m - 15]
    fm, lm, ltm = FIRST[c], LAST[c], LASTTRADE_M.get(c)
    cs = set()
    for x in six:
        v = lev(c, x)
        if not np.isfinite(v):
            if x > lm:
                if ltm is not None and P0912 <= ltm <= P1212: cs.add('TRUNC')
                elif x >= P1301 and lm >= P1212: cs.add('IFRS13')
                else: cs.add('PRE12')
            elif x < fm: cs.add('NEWLIST')
            else: cs.add('HOLE')
    for k in range(3):
        b = lev(c, m - 13 - k); a = lev(c, m - 1 - k)
        if np.isfinite(a) and np.isfinite(b) and b <= 0: cs.add('BASE_LE0')
    for p in PRI:
        if p in cs: return p
    return 'UNEXPLAINED'

GOV = {'TRUNC', 'IFRS13', 'PRE12', 'HOLE', 'NEVER_DOM'}

def pool(d):
    t = IDX.get_loc(d); e = t + 1
    v60 = V.V60[t]; px = K.Cn[t]
    base = np.isfinite(v60) & (v60 >= 1e7) & np.isfinite(px) & (px >= 10)
    vv = np.where(base, v60, -np.inf)
    o = np.argsort(-vv, kind='stable')
    top = np.zeros(N, bool); top[o[:min(300, base.sum())]] = True; top &= base
    P = top & np.isfinite(V.S52[t]) & ~K.disposal_mask(d) & np.isfinite(K.On[e]) & K.TRADED[e] & ~V2C.FINMASK
    return t, e, P

def avg_rank(vals):  # average rank 1..n
    return pd.Series(vals).rank(method='average').to_numpy()

def pick10(sc, y3key, elx, e, held):
    sc = np.round(np.where(elx, sc, -9.0), 10)
    yk = np.where(elx, y3key, -1e18)
    idx = np.where(elx)[0]
    order = sorted(idx, key=lambda j: (-sc[j], -yk[j], j))
    gu = K.gap_up_open(e)
    out = []
    for j in order:
        if gu[j] and j not in held: continue
        out.append(j)
        if len(out) == 10: break
    return out

def run(which):
    """which: OPT / P90 / CARRY / ALL / BASE（自寫官方排序，應重現基準）"""
    held = set(); res = {}
    for d in DAYS:
        t, e, P = pool(d)
        m = pd.Period(d, 'M')
        y3 = K.yoy3_for_month(YL, m)
        y3 = np.where(V2C.FINMASK, np.nan, y3)
        el = P & np.isfinite(y3)
        mis = np.where(P & ~np.isfinite(y3))[0]
        causes = {j: cause(COLS[j], m) for j in mis}
        if which in ('OPT', 'P90', 'CARRY'): A = [j for j in mis if causes[j] in GOV]
        elif which == 'ALL': A = list(mis)
        else: A = []
        s52 = V.S52[t]
        if which == 'CARRY':
            yy = y3.copy(); Ain = []
            for j in A:
                for k in range(1, 25):
                    v = K.yoy3_for_month(YL, m - k)[j]
                    if np.isfinite(v): yy[j] = v; Ain.append(j); break
            elx = el.copy(); elx[Ain] = True
            idx = np.where(elx)[0]
            p52 = np.full(N, np.nan); p52[idx] = avg_rank(s52[idx]) / len(idx)
            pr = np.full(N, np.nan); pr[idx] = avg_rank(yy[idx]) / len(idx)
            sc = 0.5 * p52 + 0.5 * pr; yk = yy; Aset = set(Ain)
        else:
            Aset = set(A)
            elx = el.copy(); elx[A] = True
            idx = np.where(elx)[0]; n = len(idx)
            p52 = np.full(N, np.nan); p52[idx] = avg_rank(s52[idx]) / n
            non = np.where(el)[0]
            pr = np.full(N, np.nan); pr[non] = avg_rank(y3[non]) / n
            yk = y3.copy()
            if which in ('OPT', 'ALL'):
                pr[A] = 1.0; yk[A] = np.inf
            elif which == 'P90':
                pr[A] = 0.9; yk[A] = np.nanpercentile(y3[non], 90)
            sc = 0.5 * p52 + 0.5 * pr
        pk = pick10(sc, yk, elx, e, held)
        held = set(pk)
        res[d] = dict(pick=[COLS[j] for j in pk], A={COLS[j]: causes[j] for j in Aset} if which != 'CARRY' else {COLS[j]: causes[j] for j in Aset},
                      miss={COLS[j]: causes[j] for j in mis}, p52={COLS[j]: float(p52[j]) for j in Aset})
    return res

out = {}
RB = run('BASE')
nb = sum(RB[d]['pick'] != B[d] for d in DAYS)
out['self_base_mismatch_days'] = int(nb)

def summarize(R, tag):
    s = {}
    D1 = [d for d in DAYS if d.year <= 2012]
    def U(ds): return sum(len(set(R[d]['pick']) & set(R[d]['A'])) for d in ds)
    def CH(ds): return sum(len(set(B[d]) - set(R[d]['pick'])) for d in ds)
    s['n_D1'] = len(D1); s['U_0612'] = U(D1); s['U_0613'] = U(DAYS)
    s['share_0612'] = s['U_0612'] / (10 * len(D1)); s['share_0613'] = s['U_0613'] / 930
    s['changed_0612'] = CH(D1); s['changed_0613'] = CH(DAYS)
    s['days_with_A_in'] = sum(1 for d in DAYS if set(R[d]['pick']) & set(R[d]['A']))
    s['by_year'] = {y: U([d for d in DAYS if d.year == y]) for y in range(2006, 2014)}
    bc = {}; bc12 = {}; bs = {}
    for d in DAYS:
        for c in set(R[d]['pick']) & set(R[d]['A']):
            k = R[d]['A'][c]; bc[k] = bc.get(k, 0) + 1
            if d.year <= 2012: bc12[k] = bc12.get(k, 0) + 1
            bs[c] = bs.get(c, 0) + 1
    s['by_cause'] = bc; s['by_cause_0612'] = bc12
    s['by_stock'] = dict(sorted(bs.items(), key=lambda x: -x[1]))
    s['slots'] = sorted((str(d.date()), c) for d in DAYS for c in set(R[d]['pick']) & set(R[d]['A']))
    hi = [0, 0]; lo = [0, 0]
    for d in DAYS:
        for c, p in R[d]['p52'].items():
            tgt = hi if p >= 0.8 else lo
            tgt[0] += 1; tgt[1] += c in R[d]['pick']
    s['p52_ge08'] = hi; s['p52_lt08'] = lo
    for c in ('5854', '6012'):
        s['n_' + c] = sum(c in R[d]['pick'] for d in DAYS)
        s['d_' + c] = [str(d.date()) for d in DAYS if c in R[d]['pick']]
    s['fin_in'] = sum(V2C.FINMASK[CI[c]] for d in DAYS for c in R[d]['pick'])
    return s

ROPT = run('OPT')
out['OPT'] = summarize(ROPT, 'OPT')
for w in ('P90', 'CARRY', 'ALL'):
    s = summarize(run(w), w)
    out[w] = {k: s[k] for k in ('U_0612', 'U_0613', 'share_0612', 'share_0613', 'changed_0612', 'changed_0613', 'n_5854', 'n_6012', 'd_5854')}
out['BASE_n_5854'] = sum('5854' in B[d] for d in DAYS)
out['BASE_n_6012'] = sum('6012' in B[d] for d in DAYS); out['BASE_d_6012'] = [str(d.date()) for d in DAYS if '6012' in B[d]]
# 19 檔
L19 = ['2350', '2391', '2403', '2452', '2526', '3009', '3063', '3534', '5854', '6012', '1716']
out['opt_19'] = {c: sum(c in ROPT[d]['pick'] for d in DAYS) for c in L19}
out['base_19'] = {c: sum(c in B[d] for d in DAYS) for c in L19}
# 池內缺值股月
cm = {}; cm13 = {}
for d in DAYS:
    for c, k in ROPT[d]['miss'].items():
        tgt = cm if d.year <= 2012 else cm13
        tgt[k] = tgt.get(k, 0) + 1
out['miss_0612'] = cm; out['miss_2013'] = cm13
out['miss_0612_total'] = sum(cm.values())
out['surv_0612'] = sum(v for k, v in cm.items() if k in GOV)
out['surv_0613'] = out['surv_0612'] + sum(v for k, v in cm13.items() if k in GOV)
def rng_count(a, b, gov_only):
    tot = 0; per = []
    for d in DAYS:
        if pd.Timestamp(a) <= d <= pd.Timestamp(b):
            x = sum(1 for c, k in ROPT[d]['miss'].items() if (k in GOV if gov_only else k != 'FOREIGN'))
            tot += x; per.append(x)
    return tot, (np.mean(per) if per else None), (min(per) if per else None), (max(per) if per else None), len(per)
out['dom_miss_0912_1312'] = rng_count('2009-12-01', '2013-12-31', False)
out['surv_0912_1312'] = rng_count('2009-12-01', '2013-12-31', True)
out['surv_0912_1212'] = rng_count('2009-12-01', '2012-12-31', True)
out['surv_0604_0911'] = rng_count('2006-04-01', '2009-11-30', True)
# P1 截斷形狀
dom = [c for c in COLS if not foreign(c)]
P1 = [c for c in dom if c in LASTTRADE_M and P0912 <= LASTTRADE_M[c] <= P1212]
out['P1_n'] = len(P1)
out['P1_le_0911'] = sum(1 for c in P1 if c in LAST and LAST[c] <= pd.Period('2009-11', 'M'))
out['P1_norev'] = [c for c in P1 if c not in LAST]
lastm = pd.Series({c: LAST[c] for c in LAST})
out['revend_2010_01_2011_11'] = {str(p): int((lastm == p).sum()) for p in pd.period_range('2010-01', '2011-11', freq='M')}
out['revend_2012_12'] = int((lastm == P1212).sum())
P2 = [c for c in dom if c in LASTTRADE_M and pd.Period('2013-01', 'M') <= LASTTRADE_M[c] <= pd.Period('2014-12', 'M')]
out['P2_n'] = len(P2)
out['P2_affected'] = sum(1 for c in P2 if c in LAST and LAST[c] <= pd.Period('2009-11', 'M'))
# C2 金融 5854 事後排除
out['OPT_ex5854'] = [out['OPT']['U_0612'] - sum('5854' in ROPT[d]['pick'] and '5854' in ROPT[d]['A'] for d in DAYS if d.year <= 2012),
                     out['OPT']['U_0613'] - sum('5854' in ROPT[d]['pick'] and '5854' in ROPT[d]['A'] for d in DAYS)]
(HERE / 'rep_o15_results.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
o = out['OPT']
print('self_base_mismatch', nb)
print('OPT', o['U_0612'], o['share_0612'], o['U_0613'], o['share_0613'], 'changed', o['changed_0612'], o['changed_0613'], 'days', o['days_with_A_in'])
print('by_year', o['by_year']); print('cause', o['by_cause'], o['by_cause_0612'])
print('stock', list(o['by_stock'].items())[:15]); print('p52', o['p52_ge08'], o['p52_lt08'], 'fin', o['fin_in'])
for w in ('P90', 'CARRY', 'ALL'): print(w, out[w])
print('5854/6012 OPT', o['n_5854'], o['d_5854'], o['n_6012'], o['d_6012'], 'BASE', out['BASE_n_5854'], out['BASE_n_6012'], out['BASE_d_6012'])
print('19', out['opt_19'], out['base_19']); print('ex5854', out['OPT_ex5854'])
for k in ('miss_0612', 'miss_2013', 'miss_0612_total', 'surv_0612', 'surv_0613', 'dom_miss_0912_1312', 'surv_0912_1312', 'surv_0912_1212', 'surv_0604_0911',
          'P1_n', 'P1_le_0911', 'P1_norev', 'revend_2012_12', 'P2_n', 'P2_affected'):
    print(k, out[k])
print('revend 2010-01..2011-11 nonzero', {k: v for k, v in out['revend_2010_01_2011_11'].items() if v})
# 補充：只算面板（上市）欄位
colset = set(COLS)
out['revend_2012_12_panel'] = int(sum(1 for c in LAST if c in colset and LAST[c] == P1212))
out['revend_2012_12_panel_dom'] = int(sum(1 for c in LAST if c in colset and LAST[c] == P1212 and not foreign(c)))
out['revend_2010_01_2011_11_panel'] = {str(p): int(sum(1 for c in LAST if c in colset and LAST[c] == p)) for p in pd.period_range('2010-01', '2011-11', freq='M')}
P2all = [c for c in COLS if c in LASTTRADE_M and pd.Period('2013-01', 'M') <= LASTTRADE_M[c] <= pd.Period('2014-12', 'M')]
out['P2_n_incl_foreign'] = len(P2all)
out['P2_affected_incl_foreign'] = sum(1 for c in P2all if c in LAST and LAST[c] <= pd.Period('2009-11', 'M'))
(HERE / 'rep_o15_results.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print('panel revend', out['revend_2012_12_panel'], out['revend_2012_12_panel_dom'], sum(out['revend_2010_01_2011_11_panel'].values()), 'P2', out['P2_n_incl_foreign'], out['P2_affected_incl_foreign'])
