#!/usr/bin/env python3
"""R86 補充（看過原報告的輸出檔欄名之後才加的口徑，FINDINGS 會註明）：
 A. 三情境的 4 個位移平均（降低單一路徑取整雜訊）。
 B. 單一路徑雜訊量：一般情境把本金在 98～102 萬之間微調 9 次，看「前→後」的差會晃多少。
 C. 「零股制度回推」口徑：把盤中零股上路日（cslib.ODD_DATE）往前推到回測起點以前（小帳戶一開始就能買零股），再加每邊 0.3%／0.6%。
 D. 權息合併事件拆法的敏感度（當 0／用殖利率估／全部當現金）對第 1～3 項的影響。
用法：python3 -I -B run_r86_extra.py → r86_extra.json"""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r86lib as L
T0 = time.time()
P, U, S, avA = L.load()
c = L.stlib.cagr
D19 = P.IDX[P.IDX < pd.Timestamp('2019-01-01')][-1]; DE = P.IDX[-1]
c19 = lambda nav: c(nav, D19, DE) * 100
SG = {s: L.s10lib.make_signal_s10(P, U, dict(S=S, av=avA), shift=s) for s in range(4)}
DVS = {m: L.dividends(P, m) for m in ('est', 'zero', 'all')}; DV = DVS['est']
PR, THR, DLY = 0.0211, 20000.0, 20
SC = {'小資': dict(cap=5e5, tau=0.05, disc=0.6), '一般': dict(cap=1e6, tau=0.12, disc=0.28), '高所得': dict(cap=5e6, tau=0.28, disc=0.28)}
R = {}
# A
A = {}
for name, a in SC.items():
    full = dict(tau=a['tau'], prem=PR, thr=THR, delay=DLY)
    bf = [c19(L.sim(P, SG[s], U, DV, cap=a['cap'])[0]) for s in range(4)]
    ba = c19(L.bench(P, DV, cap=a['cap'], **full)); bb = c19(L.bench(P, DV, cap=a['cap']))
    A[name] = dict(before=bf, e50_before=bb, e50_after=ba)
    for m in DVS:
        af = [c19(L.sim(P, SG[s], U, DVS[m], cap=a['cap'], disc=a['disc'], **full)[0]) for s in range(4)]
        A[name][f'after_{m}'] = af
    A[name]['mean4'] = dict(before=float(np.mean(bf)), after=float(np.mean(A[name]['after_est'])), lead_before=float(np.mean(bf) - bb), lead_after=float(np.mean(A[name]['after_est']) - ba))
R['A_scen_mean4'] = A
# B
a = SC['一般']; full = dict(tau=a['tau'], prem=PR, thr=THR, delay=DLY); d = []
for cap in np.linspace(0.98e6, 1.02e6, 9):
    d.append(c19(L.sim(P, SG[0], U, DV, cap=cap, disc=a['disc'], **full)[0]) - c19(L.sim(P, SG[0], U, DV, cap=cap)[0]))
R['B_noise_general'] = dict(caps=list(np.linspace(0.98e6, 1.02e6, 9)), drag=d, mean=float(np.mean(d)), sd=float(np.std(d, ddof=1)), min=float(min(d)), max=float(max(d)))
# C
odd0 = L.cslib.ODD_DATE
L.cslib.ODD_DATE = pd.Timestamp('2015-01-01')
C = {}
for cap in (3e5, 5e5, 1e6):
    k = f'{int(cap/1e4)}萬'
    b_ = [c19(L.sim(P, SG[s], U, DV, cap=cap)[0]) for s in range(4)]
    C[k] = dict(base=b_, base_mean4=float(np.mean(b_)))
    for lab, cost in (('all+0.3%', dict(cmult=1.0, xslip=0.006, oddx=0.0)), ('all+0.6%', dict(cmult=1.0, xslip=0.009, oddx=0.0)), ('oddpart+0.3%', dict(cmult=1.0, xslip=0.003, oddx=0.003))):
        v = [c19(L.sim(P, SG[s], U, DV, cap=cap, cost=cost)[0]) for s in range(4)]
        C[k][lab] = dict(mean4=float(np.mean(v)), diff=float(np.mean(v) - np.mean(b_)))
    if cap <= 5e5:
        full = dict(tau=0.05, prem=PR, thr=THR, delay=DLY)
        f_ = [c19(L.sim(P, SG[s], U, DV, cap=cap, disc=0.6, **full)[0]) for s in range(4)]
        ba = c19(L.bench(P, DV, cap=cap, **full))
        C[k]['friction6'] = dict(by_shift=f_, mean4=float(np.mean(f_)), e50_after=ba)
        for lab, x in (('friction6_all+0.3%', 0.006), ('friction6_all+0.6%', 0.009)):
            v = [c19(L.sim(P, SG[s], U, DV, cap=cap, disc=0.6, cost=dict(cmult=1.0, xslip=x, oddx=0.0), **full)[0]) for s in range(4)]
            C[k][lab] = dict(by_shift=v, mean4=float(np.mean(v)), lead=float(np.mean(v) - ba), diff_vs_friction6=float(np.mean(v) - np.mean(f_)))
L.cslib.ODD_DATE = odd0
R['C_oddlot_backdated'] = C
# D
Dm = {}
caps = (6e5, 8e5, 1e6, 1.5e6, 2e6); PATHS = [(cp, s) for s in range(4) for cp in caps]
base = {k: c19(L.sim(P, SG[k[1]], U, DV, cap=k[0])[0]) for k in PATHS}
for m in DVS:
    Dm[m] = dict(tax_paths20={t: float(np.mean([c19(L.sim(P, SG[k[1]], U, DVS[m], cap=k[0], tau=t)[0]) - base[k] for k in PATHS])) for t in (0.05, 0.12, 0.20, 0.28)},
                 delay_paths20=float(np.mean([c19(L.sim(P, SG[k[1]], U, DVS[m], cap=k[0], delay=DLY)[0]) - base[k] for k in PATHS])),
                 prem_paths20=float(np.mean([c19(L.sim(P, SG[k[1]], U, DVS[m], cap=k[0], prem=PR, thr=THR)[0]) - base[k] for k in PATHS])))
    rec = []; L.sim(P, SG[0], U, DVS[m], rec=rec); q = pd.DataFrame(rec, columns=['t', 'j', 'D', 'navp', 'tx', 'pm']); q['year'] = P.IDX[q['t'].to_numpy()].year; q = q[q['year'] >= 2019]; q['y'] = q['D'] / q['navp']
    Dm[m]['div_pct'] = dict(total=float(q['y'].sum() / 8 * 100), stock=float(q.loc[q['j'] >= 0, 'y'].sum() / 8 * 100), e50=float(q.loc[q['j'] < 0, 'y'].sum() / 8 * 100))
    for size in (1e6, 3e6, 1e7):
        amt = q['y'] * size; Dm[m][f'prem_fixed_{int(size)}'] = float(np.where(amt >= THR, amt * PR, 0.0).sum() / 8)
R['D_mixed_sensitivity'] = Dm
R['secs'] = round(time.time() - T0)
json.dump(R, open(HERE / 'r86_extra.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False))
