#!/usr/bin/env python3
"""獨立複製（不 import 研究員的 pclib／avail／s*.py）：只讀 SPEC.json 的文字規則、raw/ 的時間戳與處置檔，
只 import main_v2/v2_core（官方引擎，取 C0 名單、面板陣列、滾動訊號）。從頭重寫：訊號日分類、可得性分類、R1 比例、
U_opt（2013～2026）、處置計數。最後逐筆對照研究員的 data/slot_avail.csv 與 results.json。
cache 只放 scratchpad/o17_revavail/rep_cache（題目結束刪除）。python3 -I -B replicate/rep_count.py"""
import sys, json, csv
from pathlib import Path
REP = Path(__file__).resolve().parent
TOP = REP.parent
AG = TOP.parent
RAW = TOP / 'raw'
CACHE = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/o17_revavail/rep_cache'); CACHE.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(AG / 'main_v2'))
import numpy as np, pandas as pd
import v2_core as V
sys.path.pop(0)
K = V.K
COLS = list(K.COLS); N = len(COLS); IDX = K.IDX

# ---------- C0 ----------
YL, Y5, Y5F = V.load_revenue()
def c0(latest):
    S = V.V2('gapfix', YL, Y5F if latest else Y5, latest=latest, exfin=True)
    out = []
    for e, pk, el, d in S.plan(10, 'official'):
        cs = S.cross_section(d)
        out.append((d, IDX[e], [COLS[j] for j in pk], [COLS[j] for j in np.where(cs['el'])[0]]))
    return out
C0 = c0(False); C0L = c0(True)
formal_days = {d for d, _, _, _ in C0}

# ---------- 訊號日分類（自寫，與 deadline_case 同語意） ----------
tx = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date'])['date']
CAL = pd.DatetimeIndex(sorted(set(tx) | set(IDX)))
def dcase(d):
    rm = (d.to_period('M') - 1).strftime('%Y-%m')
    if rm in ('2019-01', '2023-03'): return 'D>S'
    if rm == '2020-01': return 'D=S'
    D = CAL[CAL >= pd.Timestamp(d.year, d.month, 10)][0]
    return 'D<S' if D < d else ('D=S' if D == d else 'D>S')

# ---------- 時間戳證據 ----------
def tpe(s): return pd.to_datetime(s, utc=True, errors='coerce').dt.tz_convert('Asia/Taipei').dt.tz_localize(None)
fs = pd.read_csv(RAW / 'ts_tasi_upstream_firstseen.csv.gz', dtype=str)
fs['a'] = tpe(fs['T']); fs['b'] = tpe(fs['T_prev'])
SNAP = {}
for src, c, y, a, b in zip(fs.source, fs.code, fs.ym, fs.a, fs.b): SNAP.setdefault((c, y), []).append((src, a, b))
an = pd.read_csv(RAW / 'ts_mops_ann_date.csv', dtype=str)
ANN = {(c, y): pd.Timestamp(a) for c, y, a in zip(an.code, an.ym, an.ann_date)}
nw = pd.read_csv(RAW / 'ts_news_mentions.csv.gz', dtype={'code': str, 'ym': str})
tw = pd.read_csv('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv', dtype={'code': str}); tw['ym'] = tw.month.str[:7]
tw = tw.drop_duplicates(['code', 'ym'])
r5 = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras/rev_5y.csv.gz', dtype={'code': str})
TRU = {}
for c, y, v in zip(tw.code, tw.ym, tw.revenue_k_twd): TRU.setdefault((c, y), []).append(v)
for c, y, v in zip(r5.code, r5.ym, r5.revenue): TRU.setdefault((c, y), []).append(v)
def strict_ok(c, y, v):
    if not (isinstance(v, float) and np.isfinite(v) and v > 0): return False
    s = str(int(round(v))); tz = len(s) - len(s.rstrip('0'))
    return any(np.isfinite(t) and abs(v - t) <= 0.5 * 10 ** tz + 1 for t in TRU.get((c, y), []))
NEWS = {}
for c, y, v, ts, ht in zip(nw.code, nw.ym, nw.rev_k, nw.ts, nw.has_time.astype(str)):
    if strict_ok(c, y, float(v)): NEWS.setdefault((c, y), []).append((pd.Timestamp(ts), ht == 'True'))

def klass(S, c, y):
    close = S + pd.Timedelta(hours=13, minutes=30); mid = S + pd.Timedelta(days=1)
    ev = []
    if (c, y) in ANN:
        a = ANN[(c, y)]; ev.append('A' if a < S else ('U' if a == S else 'X'))
    for src, a, b in SNAP.get((c, y), []):
        if pd.isna(a): continue
        if a <= close: ev.append('A')
        elif pd.notna(b) and b.normalize() > S: ev.append('X')
        elif pd.isna(b) and a > mid + pd.Timedelta(days=3): ev.append('N')
        else: ev.append('U')
    for ts, ht in NEWS.get((c, y), []):
        if ht: ev.append('A' if ts <= close else ('U' if ts < mid else 'N'))
        else:
            d0 = ts.normalize(); ev.append('A' if d0 < S else ('U' if d0 == S else 'N'))
    if 'A' in ev and 'X' in ev: return 'UNDET'
    if 'A' in ev: return 'AVAIL'
    if 'X' in ev: return 'UNAVAIL'
    if 'U' in ev: return 'UNDET'
    return 'NONE'

rows = []
for d, e, pk, pool in C0L:
    cs = dcase(d)
    if cs == 'D<S': continue
    y = (d.to_period('M') - 1).strftime('%Y-%m')
    for c in pk: rows.append((d.strftime('%Y-%m-%d'), c, cs, klass(d, c, y), d in formal_days))
RS = pd.DataFrame(rows, columns=['sig', 'code', 'case', 'state', 'formal'])
F = RS[RS.formal]
def up(x): return int((x.state != 'AVAIL').sum())
def lo(x): return int((x.state == 'UNAVAIL').sum())
nslots = {k: 10 * sum(1 for d, _, _, _ in C0 if a <= d.strftime('%Y-%m-%d') <= b) for k, (a, b) in
          {'0613': ('2006-01-01', '2013-12-31'), 'ALL': ('2006-01-01', '2026-12-31'), '19': ('2019-01-01', '2026-12-31')}.items()}
r1 = {'r1a': (up(F[F.sig <= '2013-12-31']) / nslots['0613'], lo(F[F.sig <= '2013-12-31']) / nslots['0613']),
      'r1b': (up(F) / nslots['ALL'], lo(F) / nslots['ALL']),
      'r1c': (up(F[F.sig >= '2019-01-01']) / nslots['19'], lo(F[F.sig >= '2019-01-01']) / nslots['19'])}

# ---------- R2：U_opt 2013～2026（自寫原因與排序） ----------
lv = pd.read_csv('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv', dtype={'code': str, 'market': str})
lv['p'] = pd.PeriodIndex(pd.to_datetime(lv.month), freq='M'); lv['o'] = (lv.market != 'sii').astype(int)
lv = lv.sort_values(['p', 'code', 'o']).drop_duplicates(['p', 'code'])
LVL = {(c, p): v for c, p, v in zip(lv.code, lv.p, lv.revenue_k_twd)}
LAST = lv.dropna(subset=['revenue_k_twd']).groupby('code').p.max().to_dict(); FIRST = lv.dropna(subset=['revenue_k_twd']).groupby('code').p.min().to_dict()
r5['p'] = pd.PeriodIndex(r5.ym, freq='M'); r5 = r5.drop_duplicates(['p', 'code'], keep='last')
Y5V = {(c, p): v for c, p, v in zip(r5.code, r5.p, r5.yoy)}
L5 = r5.dropna(subset=['yoy']).groupby('code').p.max().to_dict(); F5 = r5.dropna(subset=['yoy']).groupby('code').p.min().to_dict()
NM = {}
for c, n in zip(lv.code, lv.name): NM.setdefault(c, n)
for r in csv.DictReader(open('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv', encoding='utf-8')): NM.setdefault(r['code'], r['name'])
for f in ('grs_568e6685_tools_twse_list.csv', 'grs_893c8772_grs_stock_no.csv', 'goristock_c3c718ab_grs_stock_no.csv'):
    p = AG / 'o_pit_shares_industry_2006_2013/raw/industry_snap' / f
    for r in csv.reader(open(p, encoding='utf-8', errors='replace')):
        r = [x.strip().strip('"') for x in r]
        if len(r) >= 2 and r[0].isdigit() and len(r[0]) == 4: NM.setdefault(r[0] + '#', set()).add(r[1])
def foreign(c):
    ns = {NM.get(c, '') or ''} | NM.get(c + '#', set())
    return c.startswith('91') or any('KY' in n.upper() or n.endswith('DR') or n.startswith('F-') for n in ns)
LTM = {}
for j, c in enumerate(COLS):
    w = np.where(K.TRADED[:, j])[0]; LTM[c] = IDX[w[-1]].to_period('M') if len(w) else None
GOVR = {'TRUNC', 'IFRS13', 'POST13', 'POST13_R5', 'PRE12', 'HOLE', 'NEVER_DOM'}
def why(c, d):
    if foreign(c): return 'FOREIGN'
    m = d.to_period('M')
    if d < pd.Timestamp('2021-03-01'):
        if c not in LAST: return 'NEVER_DOM'
        got = set()
        for k in (1, 2, 3, 13, 14, 15):
            x = m - k; v = LVL.get((c, x), np.nan)
            if v is None or not np.isfinite(v):
                if x > LAST[c]:
                    if LTM[c] is not None and pd.Period('2009-12', 'M') <= LTM[c] <= pd.Period('2012-12', 'M'): got.add('TRUNC')
                    elif x >= pd.Period('2013-01', 'M') and LAST[c] >= pd.Period('2012-12', 'M'): got.add('IFRS13' if d.year <= 2013 else 'POST13')
                    else: got.add('PRE12')
                elif x < FIRST[c]: got.add('NEWLIST')
                else: got.add('HOLE')
            elif v <= 0 and k >= 13: got.add('BASE_LE0')
    else:
        if c not in L5: return 'POST13_R5' if c in LAST else 'NEVER_DOM'
        got = set()
        for k in (1, 2, 3):
            x = m - k; v = Y5V.get((c, x), np.nan)
            if v is None or not np.isfinite(v): got.add('POST13_R5' if x > L5[c] else ('NEWLIST' if x < F5[c] else 'HOLE'))
    for k in ('TRUNC', 'IFRS13', 'POST13', 'POST13_R5', 'PRE12', 'HOLE', 'NEWLIST', 'BASE_LE0'):
        if k in got: return k
    return 'UNKNOWN'
S52, V60 = V.rolling_signals('gapfix')
FM = V.FINMASK
def y3(d):
    Y = Y5 if d >= V.SWITCH else YL
    y = K.yoy3_for_month(Y, d.to_period('M'))
    return np.where(FM, np.nan, y)
held = set(); ins = []; base_pk = {d: pk for d, _, pk, _ in C0}
for d, e_day, pk0, _ in C0:
    t = IDX.get_loc(d); e = t + 1
    v = V60[t]; px = K.Cn[t]
    ok = np.isfinite(v) & (v >= 1e7) & np.isfinite(px) & (px >= 10)
    order = np.argsort(-np.where(ok, v, -np.inf), kind='stable')
    top = np.zeros(N, bool); top[order[:min(300, ok.sum())]] = True; top &= ok
    s52 = S52[t]; yy = y3(d)
    base = top & np.isfinite(s52) & ~K.disposal_mask(d) & np.isfinite(K.On[e]) & K.TRADED[e]
    el = base & np.isfinite(yy)
    if d >= pd.Timestamp('2013-01-01'):
        Aff = np.array([base[j] and not FM[j] and not np.isfinite(yy[j]) and why(COLS[j], d) in GOVR for j in range(N)])
    else:
        Aff = np.zeros(N, bool)
    el2 = el | Aff; n = el2.sum()
    p52 = pd.Series(np.where(el2, s52, np.nan)).rank(pct=True).to_numpy()
    pr = pd.Series(np.where(el2 & ~Aff, yy, np.nan)).rank().to_numpy() / n
    pr[Aff] = 1.0
    sc = 0.5 * p52 + 0.5 * pr; sc[~el2] = np.nan
    yt = np.where(el2, np.where(Aff, np.inf, yy), np.nan)
    sc_r = np.round(np.nan_to_num(sc, nan=-9), 10)
    od = np.lexsort((np.arange(N), -np.nan_to_num(yt, nan=-1e18), -sc_r))[:int(el2.sum())]
    gu = K.gap_up_open(e); pick = []
    if el2.sum() < 50: continue
    for j in od:
        if gu[j] and j not in held: continue
        pick.append(j)
        if len(pick) == 10: break
    held = set(pick)
    if d < pd.Timestamp('2013-01-01'): assert [COLS[j] for j in pick] == base_pk[d], d
    if d >= pd.Timestamp('2013-01-01'):
        for j in pick:
            if Aff[j]: ins.append((d.strftime('%Y-%m-%d'), COLS[j]))
def cntu(a, b): return sum(1 for s, _ in ins if a <= s <= b)
r2 = {'2013_2026': cntu('2013-01-01', '2026-12-31') / 1640, '2013_2018': cntu('2013-01-01', '2018-12-31') / 720, '2019_2026': cntu('2019-01-01', '2026-12-31') / 920}

# ---------- R3 ----------
du = pd.read_csv(RAW / 'disposal_official_universe.csv', dtype={'code': str}, parse_dates=['start', 'end'])
G = {c: g for c, g in du.groupby('code')}
def hit(c, d):
    g = G.get(c); return g is not None and bool(((g.start <= d) & (g.end >= d)).any())
dsl = []
for d, e, pk, _ in C0:
    for c in pk: dsl.append((d.strftime('%Y-%m-%d'), c, hit(c, d), hit(c, e)))
DS = pd.DataFrame(dsl, columns=['sig', 'code', 's', 't'])
def dsh(a, b, col):
    x = DS[(DS.sig >= a) & (DS.sig <= b)]; return int(x[col].sum()) / len(x)
r3 = max(dsh('2006-01-01', '2013-12-31', 's'), dsh('2006-01-01', '2013-12-31', 't'), dsh('2014-01-01', '2021-09-23', 's'), dsh('2014-01-01', '2021-09-23', 't'))

# ---------- 對照 ----------
mine = pd.read_csv(TOP / 'data/slot_avail.csv', dtype={'code': str, 'sig': str})
mm = mine.set_index(['sig', 'code'])['state'].to_dict(); rr = RS.set_index(['sig', 'code'])['state'].to_dict()
slot_mismatch = [(k, mm.get(k), rr.get(k)) for k in sorted(set(mm) | set(rr)) if mm.get(k) != rr.get(k)]
E = json.loads((TOP / 'results.json').read_text(encoding='utf-8'))['ERRATUM']
ra = E['R1']['ratios']
cmpr = {'r1a_upper': (r1['r1a'][0], ra['r1a_2006_2013']['upper']['share']), 'r1a_lower': (r1['r1a'][1], ra['r1a_2006_2013']['lower']['share']),
        'r1b_upper': (r1['r1b'][0], ra['r1b_ALL']['upper']['share']), 'r1b_lower': (r1['r1b'][1], ra['r1b_ALL']['lower']['share']),
        'r1c_upper': (r1['r1c'][0], ra['r1c_2019_on']['upper']['share']), 'r1c_lower': (r1['r1c'][1], ra['r1c_2019_on']['lower']['share']),
        'R2_2013_2026_panelA': (r2['2013_2026'], E['R2']['panelA_C0']['2013_2026']['share']), 'R2_2013_2018_panelA': (r2['2013_2018'], E['R2']['panelA_C0']['2013_2018']['share']),
        'R2_2019_2026_panelA': (r2['2019_2026'], E['R2']['panelA_C0']['2019_2026']['share']), 'R3_governing': (r3, E['R3']['judgement']['governing_share'])}
out = {'utc': pd.Timestamp.now('UTC').isoformat(), 'independent_of': '未 import pclib／avail／s*.py；只 import main_v2/v2_core',
       'compare': {k: {'rep': a, 'researcher': b, 'equal': abs(a - b) < 1e-12} for k, (a, b) in cmpr.items()},
       'n_slots_classified_rep': int(len(RS)), 'n_slots_researcher': int(len(mine)), 'slot_mismatch_n': len(slot_mismatch), 'slot_mismatch_first': slot_mismatch[:20],
       'U_opt_insert_list_n': len(ins),
       'all_equal': all(abs(a - b) < 1e-12 for a, b in cmpr.values()) and not slot_mismatch and len(RS) == len(mine)}
(REP / 'rep_result.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
pd.DataFrame(ins, columns=['sig', 'code']).to_csv(REP / 'rep_uopt_slots_2013_2026.csv', index=False)
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
