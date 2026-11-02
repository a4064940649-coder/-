#!/usr/bin/env python3
"""第 1 步重現閘門（PREREG 後）：用 o7 原輸入（shares_est、o7 面板、官方日）照 review_stat_o7.py 的 G 段逐行重算
「實際 − 隨機平均（50 次，seed 5000+）」2019-01～2026-10-12 的月序列 t（原報 4.02）。|t − 4.02| ≤ 0.10 才往下做。
同時用 cmlib.simulate_w（原引擎）與 pcore.sim_fast（快速引擎）各算一次，核對兩引擎逐窗一致；再用 1,000 次隨機（seed0 起）報同口徑 t。
python3 -I -B s1_gate.py"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import pcore as P
PRE = json.loads((P.DATA / 'PREREG.json').read_text(encoding='utf-8'))
for f, h in PRE['code_sha256'].items():
    if f in ('s1_gate.py', 'pcore.py'): assert P.sha(HERE / f) == h, f
T = P.setup('o7'); L = P.L
FROZEN = P.FROZEN
END = pd.Timestamp('2026-10-12')
G = P.GX
DAYS = [d for d in L.official_days('2006-04-01', END) if d < END] + [END]
DAYS19 = [d for d in DAYS if d >= pd.Timestamp('2019-01-01')]
t0 = L.IDX.get_loc(DAYS19[0]); t1 = L.IDX.get_loc(END)
b_, s_ = L.rates('normal')
def plan_with(score_fn):
    plan = {}; held = set()
    for d in DAYS19:
        cs = T.cross_section(d); c2 = dict(cs); c2['sc'] = score_fn(cs)
        js, ws, _, _ = T.target(c2, FROZEN, held); plan[cs['e']] = (js, ws); held = set(js.tolist())
    return plan
def nav_orig(plan): return L.simulate_w(plan, b_, s_, t0, t1, G=G)[0]
def ret_fast(plan):
    q = set(L.IDX.get_loc(d) for d in DAYS19)
    nd = P.sim_fast(plan, b_, s_, t0, t1, q)
    return pd.Series(P.win_returns(nd, DAYS19[:-1], END), index=pd.DatetimeIndex(DAYS19[:-1]))
T0 = time.time()
pA = plan_with(lambda cs: cs['sc'])
nA = nav_orig(pA); rA = L.win_ret(nA, DAYS19); rA_f = ret_fast(pA)
rr = []; rr_f = []
for sd in range(50):
    rg = np.random.default_rng(5000 + sd); pl = plan_with(lambda cs: np.where(cs['elig'], rg.random(L.N), np.nan))
    rr.append(L.win_ret(nav_orig(pl), DAYS19)); rr_f.append(ret_fast(pl))
rRand = pd.concat(rr, axis=1).mean(axis=1); rRand_f = pd.concat(rr_f, axis=1).mean(axis=1)
m = (rA.index >= '2019-01-01') & (rA.index <= END)
d1 = (rA - rRand)[m]; d1f = (rA_f - rRand_f)[m]
t_orig = L.tstat(d1); t_fast = L.tstat(d1f)
eng_maxdiff = float(max(np.abs(rA.to_numpy() - rA_f.to_numpy()).max(), max(np.abs(a.to_numpy() - b.to_numpy()).max() for a, b in zip(rr, rr_f))))
# 同口徑、1,000 次隨機（seed0 起）
rr2 = []
for sd in range(PRE['n_random']):
    rg = np.random.default_rng(PRE['seed0'] + sd); rr2.append(ret_fast(plan_with(lambda cs: np.where(cs['elig'], rg.random(L.N), np.nan))))
R2 = pd.concat(rr2, axis=1)
d2 = (rA_f - R2.mean(axis=1))[m]
perm_p = float(np.mean((R2[m].mean(axis=0) >= rA_f[m].mean()).to_numpy()))
out = dict(target=4.02, tol=0.10, t_orig_engine_50=t_orig, t_fast_engine_50=t_fast, mean_pp_50=float(d1.mean() * 100), n=int(m.sum()),
           engine_max_abs_diff_window_return=eng_maxdiff, pass_gate=bool(abs(t_orig - 4.02) <= 0.10),
           t_fast_1000=L.tstat(d2), mean_pp_1000=float(d2.mean() * 100), perm_p_1000=perm_p, runtime_s=time.time() - T0)
print(json.dumps(out, ensure_ascii=False, indent=1))
R = json.loads((HERE / 'results.json').read_text(encoding='utf-8')) if (HERE / 'results.json').exists() else {}
R['gate_reproduce'] = out
(HERE / 'results.json').write_text(P.dumps(R), encoding='utf-8')
