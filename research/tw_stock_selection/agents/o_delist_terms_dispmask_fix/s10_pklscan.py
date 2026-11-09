#!/usr/bin/env python3
"""步驟 7：補掃 o17 普查跳過的 26 個大型 pkl（一次只載入一個，用完立即釋放，不另存副本）。
用 o_engine_delist_consideration/s1_census.py 的 walk／holdings／frozen_share（只讀 import）找名單；
沒有名單、只有淨值的檔案，記錄結構與可得的執行層計數（V7X run_R*：acc n_mdelay＝A16 併購延遲撥款次數＝持有到下市的部位）。
V7X run_R0～R4c 用的是 AB_V7_B10 名單（plans_j7），其名單層凍結資產占比取 o17 census 的同名單結果。python3 -I -B s10_pklscan.py"""
import sys, json, pickle, gc, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent; CZ = AG / 'o_engine_delist_consideration'
sys.path.insert(0, str(CZ)); import s1_census as C1; sys.path.pop(0)
Z = C1.Z
CEN = json.loads((CZ / 'census.json').read_text(encoding='utf-8'))
FILES = CEN['files_skipped_large']; assert len(FILES) == 26
def struct(o, depth=0):
    if isinstance(o, dict):
        ks = list(o)[:3]; return {'type': 'dict', 'n': len(o), 'keys': [str(k)[:60] for k in ks], 'first': struct(o[ks[0]], depth + 1) if ks and depth < 2 else None}
    if isinstance(o, (list, tuple)): return {'type': type(o).__name__, 'n': len(o)}
    return {'type': type(o).__name__}
OUT = {}
for f in FILES:
    p = AG / f; t0 = time.time()
    sys.path.insert(0, str(p.parent.parent)); sys.path.insert(0, str(p.parent))
    try:
        with open(p, 'rb') as fh: o = pickle.load(fh)
        err = None
    except Exception as e:
        o = None; err = repr(e)[:200]
    sys.path.pop(0); sys.path.pop(0)
    rec = dict(size_mb=round(p.stat().st_size / 2 ** 20, 1), load_error=err)
    if o is not None:
        rec['structure'] = struct(o)
        res = {}
        try: C1.walk(o, f, res)
        except Exception as e: rec['walk_error'] = repr(e)[:200]
        rec['n_list_results'] = len(res)
        aff = 0; perm_ev = set(); fmax = 0.0
        for name, Rr in res.items():
            H = C1.holdings(Rr); hit = False
            for e0, e1, cc, s in H:
                for c in cc:
                    j = Z.CI.get(c)
                    if j is None: continue
                    if not Z.TRU[e1, j] and Z.last_trade_before(j, e0, Z.TRU) < e1:
                        hit = True; perm_ev.add((c, s))
            if hit:
                aff += 1; fmax = max(fmax, C1.frozen_share(H, None)[0])
        rec.update(n_results_with_perm=aff, perm_events=sorted({c for c, _ in perm_ev}), frozen_share_max=round(fmax, 4) if aff else None)
        if isinstance(o, dict) and 'acc' in o:
            md = [v.get('n_mdelay', 0) for v in o['acc'].values() if isinstance(v, dict)]
            rec['exec_acc'] = dict(n_cells=len(md), n_cells_mdelay_pos=int(sum(x > 0 for x in md)), mdelay_mean=float(np.mean(md)) if md else None, mdelay_max=float(max(md)) if md else None)
        if isinstance(o, dict) and all(isinstance(v, dict) and 'nav' in v for v in list(o.values())[:3]):
            rec['note'] = 'o5 容量分級輸出：只有 nav／comp／cashf／st，沒有持股或名單，無法判定是否持有下市股（需回到其名單來源）'
        if isinstance(o, dict) and 'navs' in o: rec['note'] = 'V7X 重寫版複製輸出：只有 navs／stats／acc；名單＝AB_V7_B10 plans_j7（名單層普查見 linked）'
    del o; gc.collect()
    rec['sec'] = round(time.time() - t0, 1)
    OUT[f] = rec; print(f, {k: v for k, v in rec.items() if k != 'structure'}, flush=True)
# V7X run_R0～R4c 對應名單（o17 census 同名單結果）
pr = CEN['per_result']
lk = {k: v for k, v in pr.items() if k.startswith('o_v7_listing_exec_engine/data/plans_j7.csv|AB_V7_B10|')}
lk_seed20 = {k: v for k, v in lk.items() if int(k.split('|')[-1]) < 20}
linked = dict(n=len(lk_seed20), n_perm=sum(1 for v in lk_seed20.values() if v['perm_slots_U'] > 0),
              frozen_share_max=max([v.get('frozen_share_max_U', 0) for v in lk_seed20.values()] or [0]),
              n_frozen_ge5=sum(1 for v in lk_seed20.values() if v.get('frozen_share_max_U', 0) >= 0.05),
              perm_events=sorted({e[0] for v in lk_seed20.values() for e in v['perm_events_U']}))
n_old = sum(1 for v in pr.values() if v['perm_slots_U'] > 0 or v['perm_slots_IA'] > 0)
new_lists = sum(OUT[f].get('n_results_with_perm', 0) for f in FILES)
summary = dict(n_files=26, n_loaded=sum(OUT[f]['load_error'] is None for f in FILES), n_with_lists=sum(OUT[f].get('n_list_results', 0) > 0 for f in FILES),
               n_with_perm_list_level=sum(OUT[f].get('n_results_with_perm', 0) > 0 for f in FILES),
               n_exec_mdelay_pos=sum(1 for f in FILES if OUT[f].get('exec_acc', {}).get('n_cells_mdelay_pos', 0) > 0),
               v7x_runs_linked_lists=linked, census_o17_perm_results=n_old, census_o17_frozen_ge5=sum(1 for v in pr.values() if (v['perm_slots_U'] > 0 or v['perm_slots_IA'] > 0) and v.get('frozen_share_max_U', 0) >= 0.05),
               census_total_after=n_old + new_lists, census_files_error_o17=CEN['files_error'])
(HERE / 'pklscan.json').write_text(json.dumps(dict(utc=pd.Timestamp.now('UTC').isoformat(), files=OUT, summary=summary), ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=1))
