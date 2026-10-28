#!/usr/bin/env python3
"""驗證：(1) planner 原版參數 ＝ V2.plan（每期名單集合）；(2) sim(band=None) ＝ o3_core.simulate 逐日；
(3) 重現主策略 V3（官方名單 E1＋P＋X＋R_DEL）跨循環 26.95%、t 2.549（o_disposal_rejudge/v3_results.json R_DEL|E1）。
注意：本檔會算出全期淨值，但只印出／寫入「是否與既有已公開正式數字相同」的檢查結果與挑選期數字，不使用 2019+ 的任何結果做決策。"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import hb_core as H
T0 = time.time()
st = H.setup(); K, V, R = H.K, H.V, H.R
S = H.make_S()
P = H.Prep(S)
print('prep', round(time.time() - T0), 's', len(P.days), 'days', flush=True)
pl0 = S.plan(10, 'official')
pl1 = H.planner(P, 10)
assert len(pl0) == len(pl1)
same = all(a[0] == b[0] and set(a[1].tolist()) == set(b[1].tolist()) for a, b in zip(pl0, pl1))
assert same, 'planner 與 V2.plan 不同'
t0, t1, sd = S.bounds()
nO, lO = H.O3.simulate([(p[0], p[1]) for p in pl0], *H.NORMAL, t0, t1, xrules=st['XR'])
nM, lM = H.sim([(p[0], p[1]) for p in pl1], *H.NORMAL, t0, t1, xrules=st['XR'])
md = float(np.max(np.abs(nO.to_numpy() / nM.to_numpy() - 1)))
assert md < 1e-12, md
cg, mdd = R.cagr_mdd(nM, H.SEL_A, H.HOLD_B)
tt = R.tvs(nM, H.E50.reindex(nM.index), sd, H.SEL_A, H.HOLD_B)['t']
ref = json.loads((H.AG / 'o_disposal_rejudge' / 'v3_results.json').read_text(encoding='utf-8'))['runs']['R_DEL|E1']['stats']['ALL']
ok = abs(cg - ref['CAGR']) < 1e-6 and abs(tt - ref['vs0050']['t']) < 1e-6
print('V3 reproduce ALL', cg, tt, 'ref', ref['CAGR'], ref['vs0050']['t'], 'ok', ok, flush=True)
# band=0 應與 band=None 相同（全部調回）——只檢查到挑選期終點
out = dict(planner_equals_V2plan=bool(same), sim_equals_o3=md, V3_ALL_CAGR=cg, V3_ALL_t=tt, V3_ref=dict(CAGR=ref['CAGR'], t=ref['vs0050']['t']),
           V3_reproduced=bool(ok), n_patched=st['npatch'], n_exit_rules=len(st['XR']), turnover_full=lM['turnover'], n_days=len(P.days),
           data_sha={f: H.sha(H.AG / 'o_infra_v3_rebuild' / f) for f in ('delist_prices.parquet', 'merger_consideration.csv')} |
                    {'disposal_hist_rejudge.csv': H.sha(H.AG / 'o_disposal_rejudge' / 'disposal_hist_rejudge.csv')})
(H.HERE / 'validate.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
print('done', round(time.time() - T0), 's')
