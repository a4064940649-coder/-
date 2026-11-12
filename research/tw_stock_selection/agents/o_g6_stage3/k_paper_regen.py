#!/usr/bin/env python3
"""K 保管員（G6 第三階段）：用原程式（只讀 import o18lib）重產紙上 ALLPIT_REAL_{opt,p90,carry} 名單，遮罩 M1（現行正式）與 M2（全期統一），只產名單、不算報酬。
遮罩 m1／m2 的定義逐字抄自 integrate_o20/run_paper.py（I20P:20-42；該檔不能原樣重跑，因為它斷言 RESULTS_PAPER 尚未寫入並會寫 results.json）。
核對：(1) 名單 sha 等於 integrate_o20/results.json RESULTS_PAPER.lists_sha 與 o_g6_allpit_list_trace_stage2/out/k_paper_stats.json；
      (2) 逐列等於 o_g6_allpit_list_trace_stage2/out/k_paper_lists.csv.gz 的 M1／M2 列。
輸出：sealed/paper_lists_M1.csv.gz、sealed/paper_lists_M2.csv.gz（欄 mask,variant,signal,exec,picks,inserted）、k_paper_regen_check.json。
python3 -I -B k_paper_regen.py"""
import sys, json, hashlib, gzip, io, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent
T0 = time.time()
T1 = AG / 'o_delist_rev_backfill_allpit'; S2 = AG / 'o_g6_allpit_list_trace_stage2'
sys.path.insert(0, str(T1)); import o18lib as O; sys.path.pop(0)
X = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras/')
D5 = pd.read_csv(X / 'disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['date', 'start', 'end'])
U = pd.read_csv(AG / 'o_disposal_list_pit_audit_pkl18/data/universe_announce.csv', dtype={'code': str}, parse_dates=['announce', 'start', 'end'])
base = O.dm_official
def nxt(d):
    t = O.IDX.get_loc(d); return O.IDX[t + 1] if t + 1 < len(O.IDX) else d
c1, c2 = {}, {}
def m1(d):
    d = pd.Timestamp(d)
    if d in c1: return c1[d]
    m = base(d).copy()
    if d > O.DSP_HI:
        e = nxt(d)
        for x in (D5[(D5.date <= d) & (D5.start <= e) & (D5.end >= e)].code, U[(U.announce <= d) & (U.start <= e) & (U.end >= e)].code):
            for c in x:
                if c in O.CI: m[O.CI[c]] = True
    c1[d] = m; return m
def m2(d):
    d = pd.Timestamp(d)
    if d in c2: return c2[d]
    e = nxt(d); m = np.zeros(len(O.COLS), bool)
    x = U[(U.announce <= d) & (U.start <= e) & (U.end >= e)]
    assert all((a <= d) and (s <= e <= en) for a, s, en in zip(x.announce, x.start, x.end))
    for c in x.code:
        if c in O.CI: m[O.CI[c]] = True
    c2[d] = m; return m
VARS = {'ALLPIT_REAL_opt': ('opt', '2006-01-01', 'UB', True, 'B1'), 'ALLPIT_REAL_p90': ('p90', '2006-01-01', 'UB', True, 'B1'),
        'ALLPIT_REAL_carry': ('carry', '2006-01-01', 'UB', True, 'B1')}
R20 = json.loads((AG / 'integrate_o20/results.json').read_text(encoding='utf-8'))['RESULTS_PAPER']['lists_sha']
KS = json.loads((S2 / 'out/k_paper_stats.json').read_text(encoding='utf-8'))
OFF = pd.read_csv(S2 / 'out/k_paper_lists.csv.gz', dtype=str)
chk = {}; rows = {'M1': [], 'M2': []}
for tag, fn in (('M1', m1), ('M2', m2)):
    O.dm_official = fn
    for k, kind in VARS.items():
        b = O.build(kind)
        P = {p[3]: [O.COLS[j] for j in p[1]] for p in b['plan']}
        sha = hashlib.sha256(json.dumps({str(d.date()): v for d, v in P.items()}, sort_keys=True).encode()).hexdigest()
        off = OFF[(OFF['mask'] == tag) & (OFF.variant == k)]
        mine = pd.DataFrame([dict(signal=str(d.date()), picks=','.join(v)) for d, v in P.items()])
        same = (len(off) == len(mine)) and (off.signal.values == mine.signal.values).all() and (off.picks.values == mine.picks.values).all()
        chk[f'{tag}|{k}'] = dict(lists_sha=sha, eq_integrate_o20_lists_sha=(sha == R20[f'{tag}|{k}']), eq_k_paper_stats_lists_sha=(sha == KS[f'{tag}|{k}|lists_sha']),
                                 rows_eq_k_paper_lists=bool(same), n_signal=len(mine))
        for e, pk, el, d in b['plan']:
            ed = O.IDX[int(e)] if not isinstance(e, pd.Timestamp) else e
            aff = set(b['aff'].get(d, []))
            pkc = [O.COLS[j] for j in pk]
            rows[tag].append(dict(mask=tag, variant=k, signal=str(d.date()), exec=str(pd.Timestamp(ed).date()), picks=','.join(pkc), inserted=','.join(c for c in pkc if c in aff)))
O.dm_official = base
assert all(v['eq_integrate_o20_lists_sha'] and v['eq_k_paper_stats_lists_sha'] and v['rows_eq_k_paper_lists'] for v in chk.values()), chk
(HERE / 'sealed').mkdir(exist_ok=True)
out = {}
for tag in ('M1', 'M2'):
    D = pd.DataFrame(rows[tag], columns=['mask', 'variant', 'signal', 'exec', 'picks', 'inserted'])
    f = HERE / 'sealed' / f'paper_lists_{tag}.csv.gz'
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb', mtime=0) as g: g.write(D.to_csv(index=False).encode())
    f.write_bytes(buf.getvalue())
    out[tag] = dict(file=str(f.relative_to(HERE)), rows=len(D), sha256=hashlib.sha256(f.read_bytes()).hexdigest())
res = dict(utc=pd.Timestamp.now('UTC').isoformat(), runtime_s=round(time.time() - T0, 1), checks=chk, out=out,
           src_sha256={p: hashlib.sha256((AG / p).read_bytes()).hexdigest() for p in ('o_delist_rev_backfill_allpit/o18lib.py', 'integrate_o20/run_paper.py', 'o_g6_allpit_list_trace_stage2/out/k_paper_lists.csv.gz')})
(HERE / 'k_paper_regen_check.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(res, ensure_ascii=False, indent=1))
