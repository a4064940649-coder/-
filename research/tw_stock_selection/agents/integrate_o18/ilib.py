#!/usr/bin/env python3
"""integrate_o18 共用接線（只讀 import；不改任何其他資料夾）。
  紙上：o_delist_rev_backfill_allpit/o18lib（題 1：ALLPIT_REAL 名單、cons_* 引擎、win()/stats_pair()）
        ＋ o_delist_terms_dispmask_fix/cz3（題 2：consideration_v3 結算規則）、dispmask_official（replace 模式，只作敏感度）。
  變體名單＝o18lib.build(kind)；kind＝(fill, lo, rev, dsp, patch)，與題 1 PREREG 相同。
  DSPR（處置 replace 模式）：build 時暫時把 o18lib.dm_official 換成 dispmask_official.make_mask(..., 'replace')。
held_events(plan, ST)：名單層面的「持有跨越下市」事件（不需報酬）：持有區間 [e, 下一個 e) 內含規則 lt 的 (code, lt)。"""
import sys, hashlib, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent
T1 = AG / 'o_delist_rev_backfill_allpit'; T2 = AG / 'o_delist_terms_dispmask_fix'
SCR = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/integrate_o18')
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()

sys.path.insert(0, str(T1)); import o18lib as O; sys.path.pop(0)
sys.path.insert(0, str(T2)); import cz3, dispmask_official as DO; sys.path.pop(0)
assert cz3.CE is O.CE, 'cz_engine 模組不一致'
P, V, K, CE, COLS, IDX = O.P, O.V, O.K, O.CE, O.COLS, O.IDX
E2, E3 = cz3.load_E()
_kw = dict(TR=K.TRADED, IDX=K.IDX, Cn=K.Cn, CI=K.CI, On=K.On)
ST = dict(O.ST)                                    # cons_main / cons_lo / cons_hi（題 1 口徑：events_v2、d＝0.057）
ST.update({'v3_main': cz3.build(E2, E3, 'main', **_kw), 'v3_lo': cz3.build(E2, E3, 'lo', **_kw), 'v3_hi': cz3.build(E2, E3, 'hi', **_kw)})
COST = O.COST

def build(kind, excl=(), dsp_mode='union'):
    kind = tuple(None if x == 'None' else x for x in kind)
    if dsp_mode == 'union': return O.build(kind, excl=excl)
    assert dsp_mode == 'replace' and kind[3] is True
    old = O.dm_official
    O.dm_official = DO.make_mask(O._orig_dm, IDX, K.CI, O.N, 'replace')
    try: return O.build(kind, excl=excl)
    finally: O.dm_official = old

def sim(b, cost, eng):
    pl = [(p[0], p[1]) for p in b['plan']]; bb, ss = COST[cost]
    if eng in ('freeze', 'nofreeze', 'dzero'): return CE.simulate(pl, bb, ss, b['t0'], b['t1'], mode=eng)
    return CE.simulate(pl, bb, ss, b['t0'], b['t1'], mode='consideration', settle=ST[eng])

def lists_sha(b):
    PK = {p[3]: [COLS[j] for j in p[1]] for p in b['plan']}
    return hashlib.sha256(json.dumps({str(d.date()): v for d, v in PK.items()}, sort_keys=True).encode()).hexdigest()

def held_events(plan, rules, end_idx):
    """plan＝[(e, picks, ...)]（依 e 排序）；rules＝{j: [rule]}；回傳 [(code, lt_date, rule_type, e_date)]。"""
    pl = sorted([(int(p[0]), np.asarray(p[1], int)) for p in plan], key=lambda x: x[0])
    ex = [e for e, _ in pl] + [end_idx]
    out = []
    for i, (e, pk) in enumerate(pl):
        e2 = ex[i + 1]
        for j in pk:
            for r in rules.get(int(j), []):
                if e <= r['lt'] < e2: out.append((r['code'], str(K.IDX[r['lt']].date()), r['type'], str(K.IDX[e].date())))
    return out

def held_nodelist_rule(plan, end_idx):
    """持有區間內最後成交（之後再無成交）但沒有任何結算規則的代碼（引擎會凍結；描述）。"""
    TR = K.TRADED; last = {j: (int(np.flatnonzero(TR[:, j])[-1]) if TR[:, j].any() else -1) for j in range(O.N)}
    allr = set(ST['v3_main']) | set(ST['cons_main'])
    pl = sorted([(int(p[0]), np.asarray(p[1], int)) for p in plan], key=lambda x: x[0]); ex = [e for e, _ in pl] + [end_idx]
    out = []
    for i, (e, pk) in enumerate(pl):
        for j in pk:
            if e <= last[int(j)] < ex[i + 1] and last[int(j)] < end_idx - 5 and int(j) not in allr:
                out.append((COLS[int(j)], str(K.IDX[last[int(j)]].date()), str(K.IDX[e].date())))
    return out

def v3_status(code):
    r = E3[E3.code == code]
    return None if not len(r) else f"{r.text_status.iloc[0]}/{r.settle.iloc[0]}"
