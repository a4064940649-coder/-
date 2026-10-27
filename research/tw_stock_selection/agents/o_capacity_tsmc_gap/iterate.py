#!/usr/bin/env python3
"""迭代建模：只用挑選期（日頻引擎只模擬到 2019-04-11 收盤；2019-04 換股不執行）。
版本（全部 n10、排除金融、v2 規格）：it1 U300 w0（原版）、it2 U51_300、it3 U101_300、it4 U100（對照，不能被選）、
依事前規則選股池後，it5～it7＝選定股池 ＋ 台積電 w＝10%、20%、30%。輸出 iterations.json。"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import ocore as O
K = O.K
T0 = time.time()
R = json.loads((O.HERE / 'results.json').read_text(encoding='utf-8'))
assert R.get('PLAN_sha256') and 'PREREG' not in R, '要在 plan 之後、freeze 之前執行'
tS = O.IDX.get_loc(O.SPLIT)

def run(univ, w=0.0, n=10, cost=O.NORMAL):
    lo, hi = O.UNIV[univ]
    S = O.V2R(lo, hi)
    pl = S.plan(n)
    t0, _, sd = S.bounds()
    ew, _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, t0, tS)
    nav, lg = O.simulate_w(O.tsmc_plan(pl, w), *cost, t0, tS)
    sdS = [d for d in sd if d <= O.SPLIT]
    P = O.periods(sdS)
    st = {k: O.segstat(nav, ew, sdS, *P[k]) for k in ('SEL',)}
    for lab, a, b in (('SEL_2006_2013', '2006-04-01', '2013-12-31'), ('SEL_2014_2019Q1', '2014-01-01', P['SEL'][1])):
        st[lab] = O.segstat(nav, ew, sdS, a, b, a if lab.endswith('2013') else '2014-01-01', b if lab.endswith('2013') else str(O.SPLIT.date()))
    return {'nav': nav, 'ew': ew, 'sd': sdS, 'stats': st, 'turnover': lg['turnover'], 'pool_mean': float(np.mean([len(p[2]) for p in pl if p[3] < O.SPLIT]))}

out = {'note': '只用挑選期：訊號窗起點 < 2019-04-11、日頻到 2019-04-11 收盤。全部 n10、排除金融、v2 規格、一般成本（另列壓力成本）。', 'versions': {}}
base = run('U300')
W0 = O.window_returns(base['nav'], base['sd']); E0 = O.window_returns(base['ew'], base['sd'])
a_sel, b_sel = O.periods(base['sd'])['SEL'][:2]
def rec(name, r, extra=None):
    Wv = O.window_returns(r['nav'], r['sd']); Ev = O.window_returns(r['ew'], r['sd'])
    d = {'stats': {k: {kk: vv for kk, vv in v.items() if kk != 'yearly'} for k, v in r['stats'].items()},
         'yearly': r['stats']['SEL']['yearly'], 'turnover': r['turnover'], 'pool_mean': r['pool_mean'],
         'paired_vs_U300': O.paired_t(Wv, W0, a_sel, b_sel),
         'paired_excess_vs_own_EW_minus_U300_excess': O.paired_t(Wv - Ev, W0 - E0, a_sel, b_sel)}
    if extra: d.update(extra)
    out['versions'][name] = d
    s = d['stats']['SEL']
    print(f'[{time.time()-T0:5.0f}s] {name}: CAGR {s["CAGR"]} DD {s["maxDD"]} t0050 {s["t_0050"]} tEW {s["t_EW"]} TE {s["TE_ann"]} paired {d["paired_vs_U300"]}', flush=True)

rec('it1_U300_w0', base)
for i, u in ((2, 'U51_300'), (3, 'U101_300'), (4, 'U100')):
    rec(f'it{i}_{u}_w0', run(u))

# ---- 事前規則：股池 ----
cand = {u: out['versions'][f'it{i}_{u}_w0']['paired_vs_U300']['t'] for i, u in ((2, 'U51_300'), (3, 'U101_300'))}
ok = {u: t for u, t in cand.items() if t >= 1.5}
chosen_u = max(ok, key=ok.get) if ok else 'U300'
out['pool_rule'] = {'candidates_paired_t': cand, 'threshold': 1.5, 'chosen': chosen_u}
print('pool chosen', chosen_u, cand, flush=True)

# ---- 台積電權重（在選定股池上）----
b0 = base if chosen_u == 'U300' else run(chosen_u)
for i, w in ((5, 0.10), (6, 0.20), (7, 0.30)):
    r = run(chosen_u, w)
    rec(f'it{i}_{chosen_u}_w{int(w*100)}', r, {'CAGR_loss_vs_w0_pp': round(b0['stats']['SEL']['CAGR'] - r['stats']['SEL']['CAGR'], 2)})
ws = {0.0: (b0['stats']['SEL']['t_0050'], 0.0)}
for i, w in ((5, 0.10), (6, 0.20), (7, 0.30)):
    v = out['versions'][f'it{i}_{chosen_u}_w{int(w*100)}']
    ws[w] = (v['stats']['SEL']['t_0050'], v['CAGR_loss_vs_w0_pp'])
elig = {w: t for w, (t, loss) in ws.items() if loss <= 1.0}
chosen_w = max(elig, key=elig.get) if elig else 0.0
out['tsmc_rule'] = {'t_and_loss': {str(k): v for k, v in ws.items()}, 'max_loss_pp': 1.0, 'chosen_w': chosen_w}
print('w chosen', chosen_w, ws, flush=True)

# ---- 壓力成本（同版本，只報不選）----
out['stress'] = {}
for u in ('U300', 'U51_300', 'U101_300', 'U100'):
    r = run(u, 0.0, cost=O.STRESS); out['stress'][u] = {k: v for k, v in r['stats']['SEL'].items() if k != 'yearly'}
for w in (0.10, 0.20, 0.30):
    r = run(chosen_u, w, cost=O.STRESS); out['stress'][f'{chosen_u}_w{int(w*100)}'] = {k: v for k, v in r['stats']['SEL'].items() if k != 'yearly'}

# ---- 多重檢定：挑選期 max-t bootstrap（股池 2 候選的配對差；w 3 候選的對 0050 超額）----
rng = np.random.default_rng(0)
def maxt(series_list, B=5000):
    M = pd.concat(series_list, axis=1).dropna().to_numpy()
    T = len(M); tobs = M.mean(0) / (M.std(0, ddof=1) / np.sqrt(T))
    Mc = M - M.mean(0)
    mx = np.empty(B)
    for b in range(B):
        ii = rng.integers(0, T, T); x = Mc[ii]
        mx[b] = np.max(x.mean(0) / (x.std(0, ddof=1) / np.sqrt(T)))
    return {'t_obs': [round(float(v), 3) for v in tobs], 'p_maxt': round(float((mx >= tobs.max()).mean()), 4), 'q95_max_t': round(float(np.quantile(mx, 0.95)), 3), 'T': int(T)}
sel = lambda s: s[(s.index >= pd.Timestamp(a_sel)) & (s.index <= pd.Timestamp(b_sel))]
diffs = [sel(O.window_returns(run(u)['nav'], base['sd']) - W0) for u in ('U51_300', 'U101_300')]
out['maxt_pool'] = maxt(diffs)
E50w = O.window_returns(K.E50, base['sd'])
exs = [sel(O.window_returns(run(chosen_u, w)['nav'], base['sd']) - E50w) for w in (0.0, 0.10, 0.20, 0.30)]
out['maxt_tsmc_vs_0050'] = maxt(exs)
out['n_versions'] = len(out['versions'])
(O.HERE / 'iterations.json').write_text(O.dumps(out), encoding='utf-8')
print('done', time.time() - T0, out['maxt_pool'], out['maxt_tsmc_vs_0050'])
