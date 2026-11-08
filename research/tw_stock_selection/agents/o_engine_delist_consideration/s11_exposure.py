#!/usr/bin/env python3
"""普查補充（只描述，不看策略報酬）：每個「完整名單」結果的下市凍結資產占比（只計永久停止交易的部位，U 或 IA 面板任一判永久者），
freeze 語意、等權進場、不計成本的簡化模擬；報最大值、日期；≥ 5% 者列為「結論可能受下市凍結影響、待重算」。
完整名單＝至少 24 期、每期中位數 ≥ 5 檔，且路徑不是事件紀錄（/frozen、/rows、slot 清單）。
python3 -I -B s11_exposure.py → exposure.json"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cz_lib as Z
import s1_census as C1
res, flog = C1.scan()
rows = pd.read_parquet(Z.WORK / 'census_rows.parquet')
perm = rows[rows.perm_U | rows.perm_IA]
pk = {(r.result, r.code, r.exec) for r in perm.itertuples()}
out = {}; flag = []
PG = {k: list(g.itertuples()) for k, g in perm.groupby('result')}
for name, R in res.items():
    if any(s in name for s in ('/frozen', '/rows', 'slot', 'U_opt_inserted', 'displaced')): continue
    H = C1.holdings(R)
    if len(H) < 24 or np.median([len(h[2]) for h in H]) < 5: continue
    pr = PG.get(name, [])
    if not pr: continue
    permset = {(Z.CI[r.code], Z.idx_of(r.exec)): Z.idx_of(r.last_U or r.last_IA) for r in pr if r.code in Z.CI}
    # 簡化模擬：凍結只限永久部位
    val = {}; tot = 1.0; best = (0.0, None); reb = {h[0]: h for h in H}
    dead = {}
    for t in range(H[0][0], H[-1][1] + 1):
        if t in reb:
            fz = {j: v for j, v in val.items() if j in dead and t > dead[j]}
            dead = {j: x for j, x in dead.items() if j in fz}
            free = tot - sum(fz.values())
            cc = [Z.CI[c] for c in reb[t][2] if c in Z.CI]
            val = dict(fz)
            for j in cc:
                val[j] = val.get(j, 0.0) + free / len(cc)
                if (j, t) in permset: dead[j] = permset[(j, t)]
            sh = sum(fz.values()) / tot if tot > 0 else 0
            if sh > best[0]: best = (sh, str(Z.IDX[t].date()))
        else:
            for j in list(val):
                if Z.TRU[t, j] and (j not in dead or t <= dead[j]):
                    r = Z.RU[t, j]; val[j] *= (1 + (r if np.isfinite(r) else 0.0))
        tot = sum(val.values()) if val else tot
    ev = sorted({(r.code, r.last_U or r.last_IA) for r in pr})
    out[name] = dict(n_lists=len(H), perm_slots=len(pr), events=ev, n_events=len(ev), frozen_share_max=round(best[0], 4), date=best[1])
    if best[0] >= 0.05: flag.append(name)
byfile = {}
for k in flag:
    f = k.split('|')[0]
    f = '/'.join(f.split('/')[:2]) if f.count('/') >= 1 else f
    byfile.setdefault(f, []).append(k)
J = dict(utc=pd.Timestamp.now('UTC').isoformat(), no_strategy_returns=True, n_complete_results_with_perm=len(out), n_flag_ge5=len(flag),
         flagged_by_file={f: dict(n=len(v), max_share=max(out[x]['frozen_share_max'] for x in v),
                                  events=sorted({tuple(e) for x in v for e in out[x]['events']})) for f, v in byfile.items()},
         per_result=out)
(HERE / 'exposure.json').write_text(json.dumps(J, ensure_ascii=False, indent=0, default=str), encoding='utf-8')
print(len(out), len(flag)); print(json.dumps(J['flagged_by_file'], ensure_ascii=False, default=str)[:4000])
