"""紙上名單：官方 DU vs 自建 v2（年檔重建）vs posthoc（+2344 2020-12-10～12-23），比較逐期選股名單。只讀 import o18lib。"""
import sys, json, hashlib, time
import numpy as np, pandas as pd
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_delist_rev_backfill_allpit')
import o18lib as O
OUT = '/home/user/-/research/tw_stock_selection/agents/o_disposal_list_pit_audit_pkl18/replicate_o19/'
C = pd.read_csv(OUT + 'r1_universe.csv', dtype={'code': str}, parse_dates=['start', 'end'])
mine = C[['code', 'start', 'end']].sort_values(['code', 'start']).reset_index(drop=True)
off = O.DU[['code', 'start', 'end']].sort_values(['code', 'start']).reset_index(drop=True)
same_set = mine.equals(off)
post = pd.concat([mine, pd.DataFrame([dict(code='2344', start=pd.Timestamp('2020-12-10'), end=pd.Timestamp('2020-12-23'))])], ignore_index=True)
OFF_DU = O.DU
def plan_sig(kind, du):
    O.DU = du; O._dc.clear()
    b = O.build(kind)
    O.DU = OFF_DU; O._dc.clear()
    rows = [(str(d)[:10], sorted(O.COLS[k] for k in pk)) for e, pk, el, d in b['plan']]
    return rows, hashlib.sha256(json.dumps(rows).encode()).hexdigest()
kinds = {'C0': ('base', '2006-01-01', None, True, None)}
for f in ('opt', 'p90', 'carry'):
    kinds['ALLPIT_REAL_' + f] = (f, '2006-01-01', 'UB', True, 'B1')
res = dict(v2_equals_official_DU=bool(same_set), n_mine=len(mine), n_off=len(off))
for nm, k in kinds.items():
    t = time.time()
    ro, so = plan_sig(k, OFF_DU)
    rm, sm = plan_sig(k, mine)
    rp, sp = plan_sig(k, post)
    chg = sum(len(set(a[1]) ^ set(b[1])) // 2 for a, b in zip(ro, rp))
    in2344 = [d for d, pk in ro if '2344' in pk and '2020-12' <= d < '2021-01']
    res[nm] = dict(n_periods=len(ro), sha_off=so[:16], sha_v2=sm[:16], sha_post=sp[:16], v2_same=so == sm,
                   post_slot_changes=chg, picked_2344_dec2020=in2344, sec=round(time.time() - t, 1))
    print(nm, res[nm], flush=True)
json.dump(res, open(OUT + 'r5_paper_lists.json', 'w'), indent=1)
# 對照組：DU 清空時名單必須改變（證明替換路徑有效）
r0, s0 = plan_sig(kinds['C0'], OFF_DU.iloc[0:0])
ro, so = plan_sig(kinds['C0'], OFF_DU)
res['sanity_emptyDU_C0_changes'] = sum(len(set(a[1]) ^ set(b[1])) // 2 for a, b in zip(ro, r0))
print('sanity', res['sanity_emptyDU_C0_changes'])
json.dump(res, open(OUT + 'r5_paper_lists.json', 'w'), indent=1)
