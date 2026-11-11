# -*- coding: utf-8 -*-
"""K 角色（目標產生器，非獨立複製）：以 o18 紙上管線（只讀 import o18lib）重產 ALLPIT_REAL_{opt,p90,carry} 紙上名單與窗報酬，三種處置遮罩：
 M0 原管線（≤2021-09-23 d 或 e 涵蓋＋ia_core 原遮罩；之後 disposal_5y 只看 d）＝o18 正式；
 M1 審查員修正（M0 ∪ 2021-09-23 後「公告 ≤ d 且涵蓋 e」）＝22.06% 來源；
 M2 全期統一（只用官方表「公告 ≤ d 且涵蓋 e」，不與 ia_core 原遮罩聯集）＝PREREG (d)。
名單 CSV.gz 與窗報酬 npz 存本題 out/；本檔不得被 A／B／C 程式 import。python3 -I -B k_paper.py"""
import sys, json, hashlib
from pathlib import Path
import numpy as np, pandas as pd
AG = Path('/home/user/-/research/tw_stock_selection/agents'); OUT = AG / 'o_g6_allpit_list_trace_stage2/out'
T1 = AG / 'o_delist_rev_backfill_allpit'
sys.path.insert(0, str(T1)); import o18lib as O; sys.path.pop(0)
X = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras/')
D5 = pd.read_csv(X / 'disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['date', 'start', 'end'])
U = pd.read_csv(AG / 'o_disposal_list_pit_audit_pkl18/data/universe_announce.csv', dtype={'code': str}, parse_dates=['announce', 'start', 'end'])
base = O.dm_official
def nxt(d):
    t = O.IDX.get_loc(d); return O.IDX[t + 1] if t + 1 < len(O.IDX) else d
c1 = {}
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
c2 = {}
def m2(d):
    d = pd.Timestamp(d)
    if d in c2: return c2[d]
    e = nxt(d); m = np.zeros(len(O.COLS), bool)
    x = U[(U.announce <= d) & (U.start <= e) & (U.end >= e)]
    for c in x.code:
        if c in O.CI: m[O.CI[c]] = True
    # 斷言：每一列都是「公告 ≤ d 且涵蓋 e」
    assert all((a <= d) and (s <= e <= en) for a, s, en in zip(x.announce, x.start, x.end))
    c2[d] = m; return m
VARS = {'C0': ('base', None, None, False, None), 'ALLPIT_REAL_opt': ('opt', '2006-01-01', 'UB', True, 'B1'),
        'ALLPIT_REAL_p90': ('p90', '2006-01-01', 'UB', True, 'B1'), 'ALLPIT_REAL_carry': ('carry', '2006-01-01', 'UB', True, 'B1')}
R1 = json.loads((T1 / 'results.json').read_text(encoding='utf-8'))['RESULTS_PAPER']
rows = []; W = {}; res = {}
SD0 = None
for tag, fn in (('M0', base), ('M1', m1), ('M2', m2)):
    O.dm_official = fn
    for k, kind in VARS.items():
        if k == 'C0' and tag != 'M0': continue
        b = O.build(kind)
        PK = {p[3]: [O.COLS[j] for j in p[1]] for p in b['plan']}
        sha = hashlib.sha256(json.dumps({str(d.date()): v for d, v in PK.items()}, sort_keys=True).encode()).hexdigest()
        if tag == 'M0': assert sha == R1['lists_sha'][k], ('M0 名單與 o18 官方 sha 不同', k)
        if k == 'C0': SD0 = b['sd']
        for d, v in PK.items(): rows.append(dict(mask=tag, variant=k, signal=str(d.date()), picks=','.join(v)))
        for c in ('normal', 'stress'):
            nav = O.sim(b, c, 'freeze' if k == 'C0' else 'cons_main')[0]
            W[(tag, k, c)] = O.win(nav, SD0)
        res[f'{tag}|{k}|lists_sha'] = sha
O.dm_official = base
for key, w in W.items():
    if key[1] == 'C0': continue
    s = O.stats_pair(w, W[('M0', 'C0', key[2])])
    res['|'.join(key)] = {seg: {kk: s[seg][kk] for kk in ('ann', 'gex', 't_vs_0050') if kk in s[seg]} for seg in s if isinstance(s[seg], dict) and 'ann' in s[seg]}
pd.DataFrame(rows).to_csv(OUT / 'k_paper_lists.csv.gz', index=False)
Z = {}
for k, w in W.items():
    kk = '|'.join(k); Z[kk + '|r'] = w['r'].to_numpy(float); Z[kk + '|r50'] = w['r50'].to_numpy(float); Z[kk + '|per'] = np.array([str(x) for x in w.index])
np.savez_compressed(OUT / 'k_paper_windows.npz', **Z)
json.dump(res, open(OUT / 'k_paper_stats.json', 'w'), ensure_ascii=False, indent=1, default=float)
for k in sorted(res):
    if '|lists_sha' in k: continue
    print(k, 'ALL', {kk: round(v, 3) for kk, v in res[k].get('ALL', {}).items()})
