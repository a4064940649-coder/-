#!/usr/bin/env python3
"""描述性附表（不判定、不產生訊號、不是排除規則）：C0 與 V7X（官方日 plan_V7_B10.csv、j 名單 seed 0）名單中，
「訊號日後 120 個交易日內終止交易、且為現金／換股／轉換／混合（含未核實條款）」的鎖價標的槽位：
槽位紙上報酬（訊號日收盤→下一訊號日收盤；期內下市者依 events_v2 main 規則結算）與同期同名單其他檔平均的差。
日後若要變成規則，必須重新事前登記，且依 S1 只能用前瞻資料判定。python3 -I -B s12_desc.py → desc_locked.json"""
import sys, json, pickle
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cz_lib as Z
E = pd.read_csv(HERE / 'events_v2.csv', dtype={'code': str, 'successor': str}).set_index('code')
GR = np.where(Z.TRU, 1 + np.nan_to_num(Z.RU, nan=0.0), 1.0)
CUM = np.cumprod(GR, axis=0)


def slot_ret(c, a, b):
    j = Z.CI.get(c)
    if j is None: return np.nan, False
    if c in E.index and isinstance(E.loc[c, 'last_trade'], str) and E.loc[c, 'last_trade']:
        lt = Z.idx_of(E.loc[c, 'last_trade'])
        if a <= lt < b:
            r = E.loc[c]; lc = r.last_close; g = CUM[lt, j] / CUM[a, j]
            sj = Z.CI.get(r.successor) if isinstance(r.successor, str) and r.successor else None
            s = Z.idx_of(r.succ_first_trade) if isinstance(r.succ_first_trade, str) and r.succ_first_trade else None
            t = r.type
            if t == 'cash': v = r.cash
            elif t in ('mixed', 'conversion', 'swap', 'swap_ur') and sj is not None and s is not None and s <= b:
                ratio = r.ratio if np.isfinite(r.ratio) else lc / Z.CU[lt, sj]
                v = ratio * Z.CU[s, sj] * CUM[b, sj] / CUM[s, sj]
                v += (r.cash if np.isfinite(r.cash) else 0) + ((r.pref_ratio * r.pref_value) if np.isfinite(r.pref_ratio) else 0)
            else: v = lc
            return g * v / lc - 1, True
    return CUM[b, j] / CUM[a, j] - 1, False


def table(lists, name):
    sd = sorted(lists); rows = []
    for i, s in enumerate(sd[:-1]):
        a, b = Z.idx_of(s), Z.idx_of(sd[i + 1])
        rr = {c: slot_ret(c, a, b) for c in lists[s]}
        for c in lists[s]:
            if c not in E.index or not isinstance(E.loc[c, 'last_trade'], str) or not E.loc[c, 'last_trade']: continue
            lt = Z.idx_of(E.loc[c, 'last_trade'])
            if not (a <= lt <= a + 120): continue
            if E.loc[c, 'type'] not in ('cash', 'mixed', 'conversion', 'swap', 'swap_ur', 'acq_ur'): continue
            oth = [rr[x][0] for x in lists[s] if x != c and np.isfinite(rr[x][0])]
            rows.append(dict(signal=s, code=c, type=E.loc[c, 'type'], verified=E.loc[c, 'verified'], delisted_in_window=bool(rr[c][1]),
                             slot_ret=float(rr[c][0]), others_mean=float(np.mean(oth)), diff=float(rr[c][0] - np.mean(oth))))
    D = pd.DataFrame(rows)
    return dict(name=name, n_slots=len(D), n_events=int(D.code.nunique()) if len(D) else 0,
                mean_diff_pp=float(D['diff'].mean() * 100) if len(D) else None, median_diff_pp=float(D['diff'].median() * 100) if len(D) else None,
                mean_slot_ret_pp=float(D.slot_ret.mean() * 100) if len(D) else None, rows=D.to_dict('records'))


P = pickle.loads((Z.WORK / 'plans_ia.pkl').read_bytes())
C0 = {p[2]: [Z.COLS[0]] for p in []}
import importlib
sys.path.insert(0, str(Z.AG / 'independent_audit')); import ia_core as K; sys.path.pop(0)
C0 = {pd.Timestamp(K.IDX[p[0] - 1]).strftime('%Y-%m-%d') if False else p[2]: [K.COLS[j] for j in p[1]] for p in P['C0']['plan']}
v7o = pd.read_csv(Z.AG / 'o_v7_listing_exec_engine' / 'plan_V7_B10.csv', dtype={'picks': str})
V7O = {s: p.split(',') for s, p in zip(v7o.signal, v7o.picks)}
pj = pd.read_csv(Z.AG / 'o_v7_listing_exec_engine' / 'data' / 'plans_j7.csv', dtype={'picks': str})
pj = pj[(pj.strat == 'AB_V7_B10') & (pj.seed == 0)]
J = {int(j): {s: p.split(',') for s, p in zip(g.signal, g.picks)} for j, g in pj.groupby('j')}
out = dict(note='描述性；不是排除規則；日後要變規則須重新事前登記且只能用前瞻資料判定（S1）', C0=table(C0, 'C0'), V7X_official=table(V7O, 'V7_B10 官方日'),
           V7X_j_seed0={j: {k: v for k, v in table(L, f'J{j}').items() if k != 'rows'} for j, L in J.items()})
(HERE / 'desc_locked.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print({k: {kk: vv for kk, vv in v.items() if kk != 'rows'} for k, v in out.items() if k in ('C0', 'V7X_official')})
print(pd.DataFrame(out['V7X_official']['rows']).to_string() if out['V7X_official']['rows'] else '')
print(pd.DataFrame(out['C0']['rows']).to_string() if out['C0']['rows'] else '')
