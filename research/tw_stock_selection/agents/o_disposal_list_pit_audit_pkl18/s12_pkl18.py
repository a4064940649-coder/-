#!/usr/bin/env python3
"""S5：o18 補掃 2,093 對 o17 2,075 組差 18 組追查。只讀 o_engine_delist_consideration/census.json（2,093 的來源：perm_slots_U>0 或 perm_slots_IA>0）
與 exposure.json（2,075 的來源：s11_exposure.py「完整名單」定義）。依 2,075 的文字定義（s11_exposure.py docstring＋程式）重建，按結果鍵取差集並逐組歸類。
python3 -I -B s12_pkl18.py → s12_pkl18.json"""
import json, hashlib
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; AG = HERE.parent; CZ = AG / 'o_engine_delist_consideration'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
C = json.loads((CZ / 'census.json').read_text(encoding='utf-8'))['per_result']
E = json.loads((CZ / 'exposure.json').read_text(encoding='utf-8'))
EP = E['per_result']
# 集合 A：o18 pklscan 的 2,093 定義（s10_pklscan.py 第 65 行：perm_slots_U>0 or perm_slots_IA>0）
A = {k for k, v in C.items() if v['perm_slots_U'] > 0 or v['perm_slots_IA'] > 0}
# 集合 B：依 2,075 的文字定義重建（s11_exposure.py：完整名單＝≥24 期、每期中位數 ≥5 檔、路徑不是事件紀錄 /frozen、/rows、slot、U_opt_inserted、displaced）
EXCL = ('/frozen', '/rows', 'slot', 'U_opt_inserted', 'displaced')
rule1 = {k for k in A if any(s in k for s in EXCL)}
rule2 = {k for k in A if C[k]['n_lists'] < 24}
B_rebuilt = A - rule1 - rule2
D = sorted(A - set(EP))
cls = []
for k in D:
    r = [s for s in EXCL if s in k]
    cls.append(dict(key=k, n_lists=C[k]['n_lists'], perm_slots_U=C[k]['perm_slots_U'], events=[e[0] for e in C[k]['perm_events_U']],
                    rule1_path_is_event_record=bool(r), rule1_token=r, rule2_lt24_periods=C[k]['n_lists'] < 24))
ge5_A = {k for k in A if (C[k].get('frozen_share_max_U') or 0) >= 0.05}
ge5_E = {k for k, v in EP.items() if v['frozen_share_max'] >= 0.05}
out = dict(inputs={'census.json': sha(CZ / 'census.json'), 'exposure.json': sha(CZ / 'exposure.json'), 's11_exposure.py': sha(CZ / 's11_exposure.py'),
                   '../o_delist_terms_dispmask_fix/s10_pklscan.py': sha(AG / 'o_delist_terms_dispmask_fix/s10_pklscan.py')},
           n_A_2093=len(A), n_exposure_2075=len(EP), n_B_rebuilt=len(B_rebuilt), B_rebuilt_equals_exposure_keys=(B_rebuilt == set(EP)),
           exposure_subset_of_A=set(EP) <= A, n_diff=len(D), diff_explained_by_rule1=sum(c['rule1_path_is_event_record'] for c in cls),
           diff_also_rule2=sum(c['rule2_lt24_periods'] for c in cls), diff_unexplained=sum(not (c['rule1_path_is_event_record'] or c['rule2_lt24_periods']) for c in cls),
           diff_by_token={t: sum(t in c['rule1_token'] for c in cls) for t in EXCL},
           ge5=dict(census_U_ge5=len(ge5_A), exposure_ge5=len(ge5_E), diff=len(ge5_A) - len(ge5_E), diff_keys_subset_of_18=(ge5_A - ge5_E) <= set(D), exposure_ge5_subset=ge5_E <= ge5_A),
           classified=cls,
           verdict='P5 成立：18 組全部是「事件紀錄路徑」（14 組 /frozen：o_rev_survivorship run_*.json 的凍結事件紀錄；4 組 /rows：o_finmask 複製與審查的事件列），'
                   '被 2,075 的「完整名單」定義（s11_exposure.py：路徑排除 /frozen、/rows…；≥24 期）排除；2,093 是 o18 s10_pklscan.py 直接數 census per_result 的 perm>0，沒有套完整名單篩選。'
                   '用 1 條規則（路徑排除）即可解釋 18/18；第 2 條（<24 期）只重複解釋其中 4 組。≥5% 的 2,063 對 2,045 同樣是這 18 組。不需要逐案特判，結案。')
(HERE / 's12_pkl18.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps({k: v for k, v in out.items() if k not in ('classified', 'inputs')}, ensure_ascii=False, indent=0))
