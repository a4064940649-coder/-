"""獨立複製 o19 S5：census perm>0（集合 A）對 exposure「完整名單」（集合 B，依定義重建），逐組歸類差集。"""
import json, collections
AG = '/home/user/-/research/tw_stock_selection/agents/'
OUT = AG + 'o_disposal_list_pit_audit_pkl18/replicate_o19/'
C = json.load(open(AG + 'o_engine_delist_consideration/census.json'))
X = json.load(open(AG + 'o_engine_delist_consideration/exposure.json'))
PR = C['per_result']
A = {k for k, v in PR.items() if (v.get('perm_slots_U') or 0) > 0 or (v.get('perm_slots_IA') or 0) > 0}
BAD = ('/frozen', '/rows', 'slot', 'U_opt_inserted', 'displaced')
def rule1(k): return any(b in k for b in BAD)
def rule2(k): return PR[k]['n_lists'] < 24
B = {k for k in A if not rule1(k) and not rule2(k)}
XB = set(X['per_result'].keys())
D = sorted(A - B)
cls = collections.Counter()
rows = []
for k in D:
    hit = [b for b in BAD if b in k]
    r2 = rule2(k)
    path = k.split('|')[0]
    rows.append(dict(key=k, rule1=hit, rule2_lt24=r2, n_lists=PR[k]['n_lists']))
    cls[(tuple(hit), r2)] += 1
# 依路徑資料夾歸組
grp = collections.Counter('/'.join(r['key'].split('|')[0].split('/')[:2]) + ' :: ' + ','.join(r['rule1']) for r in rows)
# 凍結占比 ≥ 5%
def share(v): return v.get('frozen_share_max_U') or 0
Af = {k for k in A if share(PR[k]) >= 0.05}
Bf = {k for k in B if share(PR[k]) >= 0.05}
res = dict(nA=len(A), nB=len(B), n_exposure=len(XB), B_equals_exposure_keys=B == XB, nDiff=len(D),
           all_rule1=all(r['rule1'] for r in rows), n_rule2_also=sum(r['rule2_lt24'] for r in rows),
           n_rule2_only=sum(r['rule2_lt24'] and not r['rule1'] for r in rows),
           by_group=dict(grp), nAf=len(Af), nBf=len(Bf), n_flag_exposure=X['n_flag_ge5'],
           f_diff_same_18=(Af - Bf) == set(D) & Af, f_diff=len(Af - Bf), rows=rows)
json.dump(res, open(OUT + 'r6_pkl18.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in res.items() if k != 'rows'}, ensure_ascii=False, indent=1))
for r in rows: print(r)
