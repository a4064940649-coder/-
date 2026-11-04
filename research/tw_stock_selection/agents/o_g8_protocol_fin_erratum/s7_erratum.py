#!/usr/bin/env python3
"""題 B 執行：斷言 ERRATUM_SPEC 已登記（且 v2_core／ia_core／快照 sha 未變）；重建 2006-01～2013-12 主策略名單（C0）與 C1／C1B／C2、2841 放回的名單，
只輸出名單與計數（ERRATUM.json、results.json 的 ERRATUM）。不計算任何報酬、淨值或 t 值。用法：python3 -I -B s7_erratum.py"""
import sys, json, csv, re, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import g8lib as G
SPEC = G.check_reg('ERRATUM_SPEC', files=False)
AG = G.AG
assert G.sha(AG / 'main_v2/v2_core.py') == SPEC['base']['v2_core_sha256'], 'v2_core 已變'
assert G.sha(AG / 'independent_audit/ia_core.py') == SPEC['base']['ia_core_sha256'], 'ia_core 已變'
O13 = AG / 'o_pit_shares_industry_2006_2013'
for rel, m in SPEC['snapshots'].items():
    assert G.sha(O13 / 'raw' / rel) == m['sha256'], rel
sys.path.insert(0, str(AG / 'main_v2'))
import numpy as np, pandas as pd
import v2_core as V
sys.path.pop(0)
K = V.K
T0 = time.time()
COLS = list(K.COLS); CI = {c: j for j, c in enumerate(COLS)}; N = len(COLS)
TSE_FIN = 17

# ---------------- 快照（C1／C1B 用：只收 2013 以前 6 份；und 寬鬆口徑另加 o13 center 快照） ----------------
def load_snap(rel):
    p = O13 / 'raw' / rel; mp, nm = {}, {}
    rows = list(csv.reader(open(p, encoding='utf-8', errors='replace')))
    for r in rows[1:]:
        r = [x.strip().strip('"') for x in r]
        if len(r) >= 3 and r[0].isdigit() and len(r[0]) == 4:
            if 'twse_list' in rel:
                if len(r) >= 4 and r[2].isdigit(): mp[r[0]] = int(r[2]) == TSE_FIN; nm[r[0]] = r[1]
            else:
                try: mp[r[0]] = int(r[2]) == TSE_FIN; nm[r[0]] = r[1]
                except ValueError: pass
    return mp, nm

SN = []
for rel, m in SPEC['snapshots'].items():
    mp, nm = load_snap(rel)
    SN.append(dict(date=pd.Timestamp(m['effective_date']), rel=rel, fin=mp, name=nm))
SN.sort(key=lambda s: s['date'])
FIRST = SN[0]['date']

def center_snaps():
    out = []
    for rel, d in (('StockList_20150522.csv', '2015-05-22'), ('StockList_20160401.csv', '2016-04-01'), ('StockList_20170706.csv', '2017-07-06')):
        T = pd.read_csv(O13 / 'raw/center' / rel, dtype=str, on_bad_lines='skip')
        out.append(dict(date=pd.Timestamp(d), rel='center/' + rel, fin={str(a).strip(): str(b).strip() in ('金融保險業', '金融保險') for a, b in zip(T['代號'], T['產業別'])
                                                                       if str(a).strip().isdigit() and len(str(a).strip()) == 4 and isinstance(b, str)}))
    T = pd.read_csv(O13 / 'raw/center/isin_20230116_listed.csv', dtype=str)
    out.append(dict(date=pd.Timestamp('2023-01-16'), rel='center/isin_20230116_listed.csv',
                    fin={str(a): str(b).strip() in ('金融保險業', '金融保險') for a, b in zip(T['code'], T['industry']) if isinstance(b, str)}))
    return out
SN_ALL = sorted(SN + center_snaps(), key=lambda s: s['date'])

# ---------------- 名稱（C2） ----------------
NAMES = {}
for s in SN:
    for c, n in s['name'].items(): NAMES.setdefault(c, set()).add(n)
for r in csv.DictReader(open('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv', encoding='utf-8')):
    if r['code'].isdigit() and len(r['code']) == 4: NAMES.setdefault(r['code'], set()).add(r['name'])
KW = re.compile('金控|金融控股|銀行|商銀|保險|產險|人壽|壽險|證券|票券|期貨|(銀|金|證|保|壽)$')
EXCL = {'2049', '2947', '3049', '4576', '7819', '9917', '9925'}
C2SET = {c for c in COLS if c[:2] in ('28', '58', '60')} | {c for c in COLS if c in NAMES and c not in EXCL and any(KW.search(n) for n in NAMES[c])}
C2MASK = np.array([c in C2SET for c in COLS])
C0MASK = V.FINMASK.copy()

def c1_mask(d, back=False):
    """C1：生效日 ≤ d 的最新快照；無 → C0。back＝True（C1B）：d 早於最早快照時用最早快照。"""
    pre = [s for s in SN if s['date'] <= d]
    s = pre[-1] if pre else (SN[0] if back else None)
    m = C0MASK.copy()
    if s is None: return m, 0
    nd = 0
    for c, f in s['fin'].items():
        j = CI.get(c)
        if j is not None:
            m[j] = f; nd += 1
    return m, nd

# ---------------- 主策略重建 ----------------
YL, Y5, _ = V.load_revenue()

class VM(V.V2):
    def __init__(self, maskfn, **kw):
        self.maskfn = maskfn
        super().__init__('gapfix', YL, Y5, exfin=True, **kw)
    def y3(self, d, back=0):
        Y = self.Y5 if d >= V.SWITCH else self.YL
        y = K.yoy3_for_month(Y, pd.Period(d, 'M') - back)
        return np.where(self.maskfn(d), np.nan, y)

LO, HI = pd.Timestamp('2006-01-01'), pd.Timestamp('2013-12-31')
def run(maskfn):
    S = VM(maskfn)
    pl = S.plan(10, 'official')
    out = {}
    for e, pk, el, d in pl:
        if LO <= d <= HI:
            cs = S.cross_section(d)
            od = S.order(cs, 'official')[:int(cs['el'].sum())]
            out[d] = dict(picks=[COLS[j] for j in pk], order=[COLS[j] for j in od], n_el=int(cs['el'].sum()))
    return S, out

S0, R0 = run(lambda d: C0MASK)
print('C0 done', len(R0), round(time.time() - T0), flush=True)
_, R1 = run(lambda d: c1_mask(d)[0])
_, R1B = run(lambda d: c1_mask(d, back=True)[0])
_, R2 = run(lambda d: C2MASK)
M2841 = C0MASK.copy(); M2841[CI['2841']] = False
_, R2841 = run(lambda d: M2841)
print('all runs', round(time.time() - T0), flush=True)

days = sorted(R0)
def slots(Rk):
    per = {}; tot = 0; enter = {}; leave = {}
    for d in days:
        a, b = set(R0[d]['picks']), set(Rk[d]['picks'])
        ch = len(a - b); tot += ch
        if ch:
            per[str(d.date())] = dict(out=sorted(a - b), in_=sorted(b - a))
            for c in a - b: leave[c] = leave.get(c, 0) + 1
            for c in b - a: enter[c] = enter.get(c, 0) + 1
    n_sl = 10 * len(days)
    by_year = {}
    for k, v in per.items(): by_year[k[:4]] = by_year.get(k[:4], 0) + len(v['out'])
    return dict(n_changed=tot, n_slots=n_sl, share=tot / n_sl, n_days_changed=len(per), by_year=by_year, enter=enter, leave=leave, detail=per)

def track(code, R):
    rows = []
    for d in days:
        o = R[d]['order']
        rk = o.index(code) + 1 if code in o else None
        rows.append(dict(signal=str(d.date()), in_pool=rk is not None, rank=rk, in_top10=code in R[d]['picks'], n_el=R[d]['n_el']))
    return rows

t5854 = [r for r in track('5854', R0) if r['signal'] <= '2011-11-30']
inpool = [r for r in t5854 if r['in_pool']]
top = [r for r in t5854 if r['in_top10']]
ranks = [r['rank'] for r in inpool]
s5854 = dict(n_months=len(t5854), n_in_pool=len(inpool), rank_median=float(np.median(ranks)) if ranks else None,
             rank_p10=float(np.percentile(ranks, 10)) if ranks else None, rank_min=min(ranks) if ranks else None,
             k_top10=len(top), top10_dates=[(r['signal'], r['rank']) for r in top],
             top10_by_year={y: sum(1 for r in top if r['signal'][:4] == y) for y in sorted({r['signal'][:4] for r in top})},
             rank_le20=sum(1 for x in ranks if x <= 20), rank_le30=sum(1 for x in ranks if x <= 30),
             pool_by_year={y: sum(1 for r in inpool if r['signal'][:4] == y) for y in sorted({r['signal'][:4] for r in inpool})},
             c0_flag=bool(C0MASK[CI['5854']]), c1_dated_fin={str(s['date'].date()): s['fin'].get('5854') for s in SN})
t2841 = track('2841', R2841)
s2841 = dict(c0_flag=bool(C0MASK[CI['2841']]), c2_flag=bool(C2MASK[CI['2841']]),
             put_back_k_top10=sum(r['in_top10'] for r in t2841), put_back_top10_dates=[(r['signal'], r['rank']) for r in t2841 if r['in_top10']],
             put_back_in_pool=sum(r['in_pool'] for r in t2841),
             put_back_rank_median=float(np.median([r['rank'] for r in t2841 if r['in_pool']])) if any(r['in_pool'] for r in t2841) else None,
             c1_top10=[d for d in [str(x.date()) for x in days] if '2841' in R1[pd.Timestamp(d)]['picks']],
             c1b_top10=[d for d in [str(x.date()) for x in days] if '2841' in R1B[pd.Timestamp(d)]['picks']],
             c1_dated_fin={str(s['date'].date()): s['fin'].get('2841') for s in SN})

# ---------------- und（o13 D2 口徑，前 300 流動性池） ----------------
def pool300(S, d):
    t = K.IDX.get_loc(d); v60 = S.V60[t]; px = K.Cn[t]
    base = np.isfinite(v60) & (v60 >= 1e7) & np.isfinite(px) & (px >= 10)
    o = np.argsort(-np.where(base, v60, -np.inf), kind='stable')[:min(300, int(base.sum()))]
    return [j for j in o if base[j]]

und_l, und_s = {}, {}; n_pm = 0; n_und_l = 0; n_und_s = 0
for d in days:
    for j in pool300(S0, d):
        c = COLS[j]; n_pm += 1
        before = [s for s in SN_ALL if c in s['fin'] and s['date'] <= d]
        after = [s for s in SN_ALL if c in s['fin'] and s['date'] > d and (s['date'] - d).days <= 731]
        lenient_ok = bool(before and (d - before[-1]['date']).days <= 731) or bool(after)
        strict_ok = any(c in s['fin'] for s in SN if s['date'] <= d)
        if not lenient_ok:
            n_und_l += 1
            if C0MASK[j]: und_l[c] = und_l.get(c, 0) + 1
        if not strict_ok:
            n_und_s += 1
            if C0MASK[j]: und_s[c] = und_s.get(c, 0) + 1
def lst(dct):
    return [dict(code=c, name='/'.join(sorted(NAMES.get(c, {'（無名稱）'}))), n_stock_months=n) for c, n in sorted(dct.items(), key=lambda x: (-x[1], x[0]))]
und = dict(n_pool_stock_months=n_pm, lenient=dict(n_undeterminable=n_und_l, n_fin_c0=sum(und_l.values()), stocks=lst(und_l)),
           strict=dict(n_undeterminable=n_und_s, n_fin_c0=sum(und_s.values()), stocks=lst(und_s)),
           o13_reference=dict(n_pool=28548, n_undeterminable=12728, n_fin_main=861),
           note='o13 用 V7 面板（含下市股補價）與 strict 滾動窗、main_rebalance_days；本題用主策略 v2 gapfix 的實際股池，股月數可能略有不同')

sl = {'C1': slots(R1), 'C1B': slots(R1B), 'C2': slots(R2), 'put_back_2841': slots(R2841)}
gov = max(sl['C1']['share'], sl['C2']['share'])
judg = dict(R6_annotate=bool(s5854['k_top10'] >= 3), R8_G4_seg1_unproven=bool(gov > 0.05), governing_share=gov,
            P6_k_in_0_6=bool(0 <= s5854['k_top10'] <= 6), P7_slots_le_2pct=bool(gov <= 0.02), P7_2841_le5=bool(s2841['put_back_k_top10'] <= 5))
annot = None
if judg['R6_annotate']:
    annot = (f"勘誤（o14，{G.now_utc()[:10]}）：主策略 2006～2011 段的金融排除用的是現行（2026）產業分類，不是當時可得的資訊（PIT 產業快照最早 2011-07-28）。"
             f"5854 合庫（當時屬金融保險業、現行旗標未排除）在 2006-01～2011-11 共 {s5854['k_top10']} 個月進入 10 檔名單（名次見 o_g8_protocol_fin_erratum/ERRATUM.json）。"
             "2006～2011 段含非 PIT 金融股；本勘誤只計數名單，沒有重算報酬。")
out = dict(utc=G.now_utc(), spec_sha256=G.load_res()['ERRATUM_SPEC_sha256'], n_signal_days=len(days), first_signal=str(days[0].date()), last_signal=str(days[-1].date()),
           s5854=s5854, s5854_monthly=t5854, s2841=s2841, slots={k: {kk: vv for kk, vv in v.items()} for k, v in sl.items()},
           undeterminable=und, judgement=judg, annotation_text=annot,
           C2_extra_vs_C0=sorted(c for c in C2SET if not C0MASK[CI[c]]), C0_extra_vs_C2=sorted(c for c in COLS if C0MASK[CI[c]] and c not in C2SET),
           lists_C0={str(d.date()): R0[d]['picks'] for d in days}, runtime_s=time.time() - T0,
           no_returns='本程式未計算任何報酬、淨值或 t 值')
(G.HERE / 'ERRATUM.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=G._js), encoding='utf-8')
G.save_key('ERRATUM', {k: v for k, v in out.items() if k not in ('s5854_monthly', 'lists_C0')}, need=['ERRATUM_SPEC'])
print(json.dumps(dict(s5854={k: v for k, v in s5854.items()}, s2841=s2841, slots={k: (v['n_changed'], v['n_slots'], round(v['share'], 4), v['by_year'], v['enter'], v['leave']) for k, v in sl.items()},
                      und={k: (v['n_undeterminable'], v['n_fin_c0'], v['stocks'][:15]) for k, v in und.items() if k in ('lenient', 'strict')}, n_pool=n_pm, judg=judg), ensure_ascii=False, default=G._js))
