#!/usr/bin/env python3
"""驗證（不產生任何 2019+ 決策資訊；只檢查與既有正式數字是否逐位相同）：
(1) plan_single(官方日曆, ab=False) 每期名單 ＝ V2.plan（V3 正式物件）；
(2) tsim(k=1) ＝ o3_core.simulate 逐日（max 相對差 < 1e-12）；重現 V3 正式 ALL 26.9459%、t 2.5486；
(3) tsim(k=2, c0) ＝ 兩個獨立 o3_core.simulate 的平均（逐日）；
(4) 官方日在錨定日曆中的 j 分布。"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import st_core as C
T0 = time.time()
st = C.setup(); K, V, R, O3 = C.K, C.V, C.R, C.O3
S = C.official_S()
pl0 = S.plan(10, 'official')
days, end = C.official_list()
pl1 = C.plan_single(days, ab=False)
assert len(pl0) == len(pl1), (len(pl0), len(pl1))
same = all(a[0] == b[0] and set(a[1].tolist()) == set(b[1].tolist()) for a, b in zip(pl0, pl1))
assert same, 'plan_single 與 V2.plan 不同'
t0, t1, sd = S.bounds()
assert C.IDX[t1] == end
nO, lO = O3.simulate([(p[0], p[1]) for p in pl0], *C.NORMAL, t0, t1, xrules=st['XR'])
nM, navs, lg = C.tsim([pl1], *C.NORMAL, t0, t1, xrules=st['XR'])
md = float(np.max(np.abs(nO.to_numpy() / nM.to_numpy() - 1)))
assert md < 1e-12, md
cg, mdd = R.cagr_mdd(nM, R.SEL_A, R.HOLD_B)
tt = R.tvs(nM, K.E50.reindex(nM.index), sd, R.SEL_A, R.HOLD_B)['t']
ref = json.loads((C.AG / 'o_disposal_rejudge' / 'v3_results.json').read_text(encoding='utf-8'))['runs']['R_DEL|E1']['stats']['ALL']
ok = abs(cg - ref['CAGR']) < 1e-6 and abs(tt - ref['vs0050']['t']) < 1e-6
print('V3 reproduce', cg, tt, ok, flush=True)
assert ok
# 我方 window 統計與 drj_core 統計一致（ALL）
W = C.windows_from(days, end)
b = C.block(nM, W, ['ALL'])['ALL']
assert abs(b['t_0050'] - tt) < 1e-9, (b['t_0050'], tt)
assert abs(b['CAGR'] - cg) < 1e-6, (b['CAGR'], cg)
# (3) k=2 c0 ＝ 獨立平均（用 J+1 與 J+11 的 AB 版，只檢查淨值相等，不印績效）
dA, eA = C.version_days(1); dB, eB = C.version_days(11)
pa = C.plan_single(dA); pb = C.plan_single(dB)
pls = C.plan_tranches([1, 11], 'a0', 'b0')
sa = all(set(x[1].tolist()) == set(y[1].tolist()) and x[0] == y[0] for x, y in zip(pa, pls[0])) and len(pa) == len(pls[0])
sb = all(set(x[1].tolist()) == set(y[1].tolist()) and x[0] == y[0] for x, y in zip(pb, pls[1])) and len(pb) == len(pls[1])
assert sa and sb, 'a0b0 分批計畫應等於各自的單日計畫'
tt0 = min(pa[0][0], pb[0][0]) - 1; tt1 = C.IDX.get_loc(C.IDX[-1])
n1, _ = O3.simulate([(p[0], p[1]) for p in pa], *C.NORMAL, tt0, tt1, xrules=st['XR'])
n2, _ = O3.simulate([(p[0], p[1]) for p in pb], *C.NORMAL, tt0, tt1, xrules=st['XR'])
nT, _, _ = C.tsim(pls, *C.NORMAL, tt0, tt1, xrules=st['XR'])
md2 = float(np.max(np.abs((0.5 * n1 + 0.5 * n2).to_numpy() / nT.to_numpy() - 1)))
assert md2 < 1e-12, md2
# (4) 官方日的 j
A = C.anchors()
jj = []
for m, d in days:
    D = C.deadline(m)
    jj.append(int(C.TCAL.get_loc(d) - C.TCAL.get_loc(D)))
jc = pd.Series(jj).value_counts().sort_index().to_dict()
sel_jc = pd.Series([j for (m, d), j in zip(days, jj) if m <= pd.Period('2018-12', 'M')]).value_counts().sort_index().to_dict()
out = dict(planner_equals_V2plan=bool(same), tsim_equals_o3=md, tsim_k2_equals_avg=md2, V3_ALL_CAGR=cg, V3_ALL_t=tt,
           V3_ref=dict(CAGR=ref['CAGR'], t=ref['vs0050']['t']), V3_reproduced=bool(ok), n_patched=st['npatch'], n_exit_rules=len(st['XR']),
           n_anchor_months=len(A), first_anchor=str(A[0]), last_anchor=str(A[-1]), official_j_counts_all={int(k): int(v) for k, v in jc.items()},
           official_j_counts_SEL={int(k): int(v) for k, v in sel_jc.items()},
           data_sha={f: C.sha(C.AG / 'o_infra_v3_rebuild' / f) for f in ('delist_prices.parquet', 'merger_consideration.csv')} |
                    {'disposal_hist_rejudge.csv': C.sha(C.AG / 'o_disposal_rejudge' / 'disposal_hist_rejudge.csv'),
                     'value.parquet': C.sha(Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel/value.parquet'))})
(C.HERE / 'validate.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
print(json.dumps(out, ensure_ascii=False, default=float)[:1500])
print('done', round(time.time() - T0), 's')
