"""2026-09 完成窗：MAIN_OFFICIAL、13 個 J 窗、JMED、MIX_C70；上限 10%／20% 比較。"""
import sys, json
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_forward_infra_live/replicate_o9')
import numpy as np, pandas as pd
import rlib as L, r_engine as RE

ASOF = pd.Timestamp('2026-11-02')
Dm, M = RE.build_D(ASOF); X = RE.load_engine(Dm)
data = M['data']
lists = pd.read_pickle(L.WORK / 'lists.pkl')
sch = {('OFF', '202609'): L.official_day(2026, 9), ('OFF', '202610'): L.official_day(2026, 10)}
for k, (y, m) in [('202609', (2026, 9)), ('202610', (2026, 10))]:
    for j, d in enumerate(L.j_days(y, m)): sch[(f'J{j}', k)] = d


def run(line, cfgx=None, cap=None):
    sd0, sd1 = sch[(line, '202609')], sch[(line, '202610')]
    order = pd.DataFrame(index=lists[(line, '202609')])
    ex = L.MP.execute(data, order, sd0, 10, set())
    pk = np.array([Dm.CI[c] for c in ex['picks']])
    i0, i1, ie = Dm.TPOS[sd0], Dm.TPOS[sd1], Dm.TPOS[ex['next_day']]
    cfg = dict(RE.BASE_CFG, **(cfgx or {}))
    old = X.CAP
    if cap is not None: X.CAP = cap
    nav, acc = X.run_one([(ie, pk, i0)], cfg) if False else (None, None)
    s = X.Sim([(ie, pk, i0)], cfg); nav = s.run(t1=i1)
    X.CAP = old
    r = nav.iloc[-1] - 1
    b = Dm.E50[i1] / Dm.E50[i0] - 1
    return dict(sd0=str(sd0.date()), sd1=str(sd1.date()), picks=ex['picks'], skipped=[c for c, _ in ex['skipped']],
                ret=100 * r, b0050=100 * b, excess=100 * (r - b), n_cap_hit=s.acc['n_cap_hit'])


res = {}
for line in ['OFF'] + [f'J{j}' for j in range(13)]:
    a = run(line); s = run(line, dict(cost='stress')); c10 = run(line, cap=0.10)
    res[line] = dict(a, excess_stress=s['excess'], excess_cap10=c10['excess'], cap10_minus_cap20=c10['excess'] - a['excess'])
    print(line, a['sd0'], a['sd1'], f"ret {a['ret']:+.2f} 0050 {a['b0050']:+.2f} ex {a['excess']:+.2f} stress {s['excess']:+.2f} cap10-20 {c10['excess'] - a['excess']:.2e}", a['skipped'])
jx = [res[f'J{j}']['excess'] for j in range(13)]; jxs = [res[f'J{j}']['excess_stress'] for j in range(13)]
summ = dict(MAIN_OFFICIAL=res['OFF']['excess'], MAIN_OFFICIAL_stress=res['OFF']['excess_stress'],
            JMED=float(np.median(jx)), JMED_stress=float(np.median(jxs)), J_min=min(jx), J_max=max(jx),
            MIX_C70_nocost=0.3 * res['OFF']['excess'])
print(json.dumps(summ, indent=1))
(L.HERE / 'out_windows.json').write_text(json.dumps(dict(lines=res, summary=summ), ensure_ascii=False, indent=1))
