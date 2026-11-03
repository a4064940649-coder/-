#!/usr/bin/env python3
"""迭代（只用挑選期 SEL）：v1 原規格；v2 緩衝帶 n+2；v3 上限溢出分給同產業；v4 上限溢出分給骨架 B。
每版跑 IND 24＋CTRL 24（一般成本、零成本），寫 iterations.json；選定規則照研究計畫三、；Q7(b) 安慰劑（扣成本前）百分位；
S2：用本題的追蹤誤差、周轉與成本模擬 IR 0.3／0.5／0.66 時 G1～G7 的通過機率。python3 -I -B s2_iterate.py"""
import sys, json, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import lc_lib as X, lc_run as Rn, lc_placebo as PL

PER = 'SEL'
MODE = sys.argv[1] if len(sys.argv) > 1 else 'COMP'; X.set_mode(MODE)
VERS = {'v1': dict(buf=1, overflow='prop'), 'v2': dict(buf=2, overflow='prop'), 'v3': dict(buf=1, overflow='sector'), 'v4': dict(buf=1, overflow='skeleton')}
it = {'mode': MODE, 'mode_params': X.mode_params(), 'rules': '允許改：緩衝帶（n+1→n+2）、上限溢出處理；不改分數成分、宇宙、產業排名；不看 2017 年以後。', 'versions': {}}
B = {j: Rn.run(dict(kind='B', carrier='PURE'), j, PER) for j in Rn.JS}
ALL = {}
for v, opt in VERS.items():
    tab = {}
    for g in Rn.GRID:
        sp = dict(g, **opt); nm = Rn.name(sp)
        xs, xb, xg, to = {}, {}, {}, []
        for j in Rn.JS:
            o = Rn.run(sp, j, PER); oz = Rn.run(sp, j, PER, 'zero')
            xs[j] = o['x']; xb[j] = o['r'] - B[j]['r']; xg[j] = oz['x']; to.append(o['to'].mean())
        a = Rn.summ_j(xs); bb = Rn.summ_j(xb); gg = Rn.summ_j(xg)
        tab[Rn.name(g)] = dict(IR_jmed=a['IR_jmed'], t_jmed=a['t_jmed'], mean_pp_jmed=a['mean_pp_jmed'], TE_jmed=a['TE_jmed'], t_p10=a['t_p10'],
                               vsB_t_jmed=bb['t_jmed'], vsB_mean_pp_jmed=bb['mean_pp_jmed'], vsB_TE_jmed=bb['TE_jmed'], gross_t_jmed=gg['t_jmed'], turnover=float(np.mean(to)))
        ALL[(v, Rn.name(g))] = (sp, tab[Rn.name(g)])
    it['versions'][v] = dict(change=opt, table=tab)
    ind = {k: r for k, r in tab.items() if k.startswith('IND')}
    best = max(ind, key=lambda k: ind[k]['IR_jmed'])
    it['versions'][v]['best_IND'] = (best, round(ind[best]['IR_jmed'], 3))
    print(v, 'best IND', best, round(ind[best]['IR_jmed'], 3), flush=True)

# ---------- 選定 ----------
def neighbors(g):
    out = []
    for dn in (-1, 1):
        if g['n'] + dn in (1, 2, 3): out.append(dict(g, n=g['n'] + dn))
    out.append(dict(g, wt='eq' if g['wt'] == 'cap' else 'cap')); out.append(dict(g, cap=0.25 if g['cap'] == 0.15 else 0.15))
    return out
cands = []
for (v, nm), (sp, r) in ALL.items():
    if not nm.startswith('IND'): continue
    g = {k: sp[k] for k in ('kind', 'n', 'wt', 'cap', 'carrier')}
    nb = [ALL[(v, Rn.name(h))][1]['IR_jmed'] for h in neighbors(g)]
    best = r['IR_jmed']
    ok_nb = best > 0 and all(x > 0 and x >= 0.5 * best for x in nb)
    cands.append(dict(version=v, name=nm, IR=best, neighbors=nb, neighbor_ok=bool(ok_nb), spec=sp))
cands.sort(key=lambda c: -c['IR'])
it['candidates_top10'] = [{k: c[k] for k in ('version', 'name', 'IR', 'neighbors', 'neighbor_ok')} for c in cands[:10]]

# Q7(b)：前 3 名候選＋事前預設版，扣成本前百分位（j 中位數）
def q7b(sp, P=1000):
    cal = PL.calibrate(sp, PER)
    pc = []
    for j in Rn.JS:
        o = Rn.run(sp, j, PER, 'zero'); real = o['x'].mean()
        pb = PL.run(sp, j, PER, 'b', P=P, rho_sec=cal['rho_sec'], rho_stk=cal['rho_stk'])
        sims = (pb['gross'] - pb['b'][None, :]).mean(1)
        pc.append(float((sims < real).mean()))
    return dict(calib=cal, pct_jmed=float(np.median(pc)), pct_per_j=pc)
it['Q7b_SEL'] = {}
tested = []
for c in cands[:3] + [dict(name=Rn.name(Rn.DEFAULT), version='v1', spec=dict(Rn.DEFAULT, buf=1, overflow='prop'))]:
    key = f"{c['version']}:{c['name']}"
    if key in it['Q7b_SEL']: continue
    r = q7b(c['spec']); it['Q7b_SEL'][key] = r; tested.append((c, r)); print('Q7b', key, round(r['pct_jmed'], 3), r['calib']['rho_stk'], r['calib']['rho_sec'], flush=True)
sel = None
for c, r in tested[:3]:
    if c.get('neighbor_ok') and r['pct_jmed'] >= 0.90: sel = c; break
if sel is None:
    it['selection'] = dict(rule_result='沒有候選同時滿足「鄰域同號且 ≥ 最佳 50%」與「Q7(b) 扣成本前 ≥ p90」→ 選定版＝事前預設版（v1：SAT、n=2、市值加權、上限 15%）',
                           selected=f"v1:{Rn.name(Rn.DEFAULT)}", selected_spec=dict(Rn.DEFAULT, buf=1, overflow='prop'), best_by_IR=f"{cands[0]['version']}:{cands[0]['name']}")
else:
    it['selection'] = dict(rule_result='條件成立', selected=f"{sel['version']}:{sel['name']}", selected_spec=sel['spec'])
print('selection', it['selection'], flush=True)

# ---------- S2：通過機率模擬 ----------
ss = it['selection']['selected_spec']; nm_sel = Rn.name(ss)
tab_sel = ALL[(it['selection']['selected'].split(':')[0], Rn.name({k: ss[k] for k in ('kind', 'n', 'wt', 'cap', 'carrier')}))][1]
TE = tab_sel['TE_jmed'] / 100
def s2(IR, TE, nsim=4000, seed=1):
    rng = np.random.default_rng(seed)
    nm_all, n19 = 247, 94          # 2006-04～2026-10、2019-01 起
    mu = IR * TE / 12; sd = TE / math.sqrt(12)
    rho_j = 0.59                    # 各 j 之間相關（o9 C 候選中位數）
    pas = dict(G1=0, G2=0, G3a=0, G3b=0, G4=0, all=0, all_noDSR=0)
    for s in range(nsim):
        z0 = rng.standard_normal(nm_all)
        Z = math.sqrt(rho_j) * z0[None, :] + math.sqrt(1 - rho_j) * rng.standard_normal((13, nm_all))
        R = mu + sd * Z
        tj = R.mean(1) / (R.std(1, ddof=1) / math.sqrt(nm_all)); x = R[np.argsort(tj)[6]]   # j 中位數路徑
        g1 = np.percentile(R.mean(1), 10) > 0 and np.median(R.mean(1)) > 0
        t_all = X.tstat(x); t19 = X.tstat(x[-n19:])
        g2 = t_all >= 3 and t19 >= 2
        g3a = t_all >= 3.37                       # 全族群 SPA 的 Tb_q95（o11）
        g3b = x.mean() / x.std(ddof=1) >= 0.4727  # 全族群 DSR 的 SR0（月）
        seg = [x[:81], x[81:153], x[153:]]
        yrs = x[:240].reshape(20, 12).sum(1); keep = np.ones(247, bool)
        for y in np.argsort(-yrs)[:3]: keep[y * 12:(y + 1) * 12] = False
        g4 = all(sg.mean() > 0 for sg in seg) and X.tstat(x[keep]) >= 2
        pas['G1'] += g1; pas['G2'] += g2; pas['G3a'] += g3a; pas['G3b'] += g3b; pas['G4'] += g4
        pas['all'] += (g1 and g2 and g3a and g3b and g4); pas['all_noDSR'] += (g1 and g2 and g3a and g4)
    return {k: v / nsim for k, v in pas.items()}
it['S2'] = dict(spec=nm_sel, TE_used=TE, turnover=tab_sel['turnover'], note='IR 為扣成本後可執行尺度；G5（PIT 只到 2014）與 G6、G7、G8 不在模擬內（G5 全期段落本題已知不成立）',
                by_IR={str(ir): s2(ir, TE) for ir in (0.3, 0.5, 0.66)})
print('S2', it['S2'], flush=True)
(HERE / f'iterations_{MODE}.json').write_text(X.dumps(it), encoding='utf-8')
