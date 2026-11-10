#!/usr/bin/env python3
"""反方審查（成本與可成交）o19：只讀研究員檔案，不寫入研究員檔案。輸出只印到 stdout。
檢查：
 R1 紙上管線 o18lib 處置遮罩在 2021-09-23 之後只看「訊號日 d 在處置」（disposal_5y, d-only），不看成交日 e；
    官方清單 start＝e（d 盤後公告）的股票在紙上仍可被選入 → 計數＋以 PIT 正確遮罩（d5y date<=d 且涵蓋 e）重建紙上名單與報酬。
 R2 ELD 盤後證據逐列核對（名稱鄰近誤配）。
 R3 disposal_5y 停更：2026-10 訊號官方處置未進 d5y。
python3 -I -B review_exec_o19.py
"""
import sys, json, hashlib
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent
T1 = AG / 'o_delist_rev_backfill_allpit'
sys.path.insert(0, str(T1)); import o18lib as O; sys.path.pop(0)
X = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras/')
D5 = pd.read_csv(X / 'disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['date', 'start', 'end'])
U = pd.read_csv(HERE / 'data/universe_announce.csv', dtype={'code': str}, parse_dates=['announce', 'start', 'end'])

# ---------- R1 ----------
_base_dm = O.dm_official
_fc = {}
def dm_fixed(d):
    d = pd.Timestamp(d)
    if d in _fc: return _fc[d]
    m = _base_dm(d).copy()
    if d > O.DSP_HI:
        t = O.IDX.get_loc(d); e = O.IDX[t + 1] if t + 1 < len(O.IDX) else d
        x = D5[(D5.date <= d) & (D5.start <= e) & (D5.end >= e)]
        for c in x.code:
            if c in O.CI: m[O.CI[c]] = True
        x = U[(U.announce <= d) & (U.start <= e) & (U.end >= e)]
        for c in x.code:
            if c in O.CI: m[O.CI[c]] = True
    _fc[d] = m; return m

VARS = {'C0': ('base', None, None, False, None), 'ALLPIT_REAL_opt': ('opt', '2006-01-01', 'UB', True, 'B1'),
        'ALLPIT_REAL_p90': ('p90', '2006-01-01', 'UB', True, 'B1'), 'ALLPIT_REAL_carry': ('carry', '2006-01-01', 'UB', True, 'B1')}
R1 = json.loads((T1 / 'results.json').read_text(encoding='utf-8'))['RESULTS_PAPER']
def lists_sha(b):
    PK = {p[3]: [O.COLS[j] for j in p[1]] for p in b['plan']}
    return hashlib.sha256(json.dumps({str(d.date()): v for d, v in PK.items()}, sort_keys=True).encode()).hexdigest(), PK
B = {}; PK = {}
for tag, fn in (('base', _base_dm), ('fixed', dm_fixed)):
    O.dm_official = fn
    for k, kind in VARS.items():
        if tag == 'fixed' and k == 'C0': continue
        b = O.build(kind); B[(tag, k)] = b; s, PK[(tag, k)] = lists_sha(b)
        if tag == 'base': assert s == R1['lists_sha'][k], ('base 名單與官方不同', k)
O.dm_official = _base_dm
print('R1 base lists sha == official: OK')

# 官方 start=e（公告=d）且 d>2021-09-23 的股票，在紙上名單中的次數
R = pd.read_csv(HERE / 'data/signal_trade_cover_rows.csv', dtype={'code': str}, parse_dates=['signal', 'trade', 'announce', 'start', 'end'])
E = R[R.covers_e & ~R.covers_d & (R.signal > O.DSP_HI)]
print('R1 post-2021-09-23 official rows start=e:', len(E))
for k in ('ALLPIT_REAL_opt', 'ALLPIT_REAL_p90', 'ALLPIT_REAL_carry'):
    pk = PK[('base', k)]
    hits = [(str(r.signal.date()), r.code) for r in E.itertuples() if r.code in pk.get(r.signal, [])]
    chg = [(str(d.date()), sorted(set(pk[d]) - set(PK[('fixed', k)].get(d, []))), sorted(set(PK[('fixed', k)].get(d, [])) - set(pk[d])))
           for d in pk if set(pk[d]) != set(PK[('fixed', k)].get(d, []))]
    print(' ', k, 'picked e-start disposal stocks (base):', hits, '| slots changed by PIT mask:', chg)

SD0 = B[('base', 'C0')]['sd']
NAV = {}
for (tag, k), b in B.items():
    for c in ('normal', 'stress'):
        if k == 'C0' and tag != 'base': continue
        if k != 'C0' and tag == 'fixed' and PK[('fixed', k)] == PK[('base', k)]: continue
        NAV[(tag, k, c)] = O.sim(b, c, 'freeze' if k == 'C0' else 'cons_main')[0]
W = {key: O.win(nav, SD0) for key, nav in NAV.items()}
for key in sorted(W):
    tag, k, c = key
    if k == 'C0': continue
    s = O.stats_pair(W[key], W[('base', 'C0', c)])
    print('R1', key, 'ALL ann %.3f gex %.3f t %.3f | SEG3 gex %.3f' % (s['ALL']['ann'], s['ALL']['gex'], s['ALL']['t_vs_0050'], s['SEG3']['gex']))

# ---------- R2 ----------
TJ = json.loads((HERE / 's13_timing.json').read_text(encoding='utf-8'))
rj = pd.read_csv(HERE / 'raw/lineage/disposal_hist_rejudge.csv', dtype={'code': str})
nm = {c: n.replace(' ', '') for c, n in zip(rj.code, rj.name.fillna(''))}
for r in TJ['rows']:
    print('R2', r['code'], nm.get(r['code']), r['post_time'], 'ann', r['announce'], '| name in snip:', nm.get(r['code'], '?') in r['snip'].replace(' ', ''),
          '| code in snip:', r['code'] in r['snip'], '|', r['snip'][:80])

# ---------- R3 ----------
print('R3 d5y max start', D5.start.max().date(), '| official rows start>2026-10-06:',
      U[U.start > '2026-10-06'][['code', 'announce', 'start']].astype(str).values.tolist())

# ---------- R4：最後一個訊號月（2026-10）名單是否含官方處置中（d5y 未收）之股票 ----------
for k in ('ALLPIT_REAL_opt', 'ALLPIT_REAL_p90', 'ALLPIT_REAL_carry'):
    pk = PK[('base', k)]
    for d in sorted(pk):
        if d < pd.Timestamp('2026-09-01'): continue
        t = O.IDX.get_loc(d); e = O.IDX[t + 1] if t + 1 < len(O.IDX) else d
        x = U[(U.announce <= d) & (((U.start <= d) & (U.end >= d)) | ((U.start <= e) & (U.end >= e)))]
        print('R4', k, d.date(), 'picks', pk[d], '| official-in-disposal on d/e:', sorted(set(x.code)), '| overlap:', sorted(set(x.code) & set(pk[d])))
