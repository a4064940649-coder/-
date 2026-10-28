import sys, pathlib, json, math; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.argv = ['x', 'anchor', 'next']
import ind_jscan as JS
import ind_tr as TRN, ind_detail as DT
C = TRN.C; import numpy as np, pandas as pd
H = pathlib.Path(__file__).resolve().parent
out = {}
def P(k, v): out[k] = v; print(k, v, flush=True)
# ---------- SEL 日曆月 t 口徑 ----------
dn = pd.read_parquet(H / 'ind_detail_navs.parquet')
for nm in ('V08', 'OFF'):
    cm = C.calm(dn[nm].dropna())
    P(f'calSEL_{nm}', {lab: C.tst(cm[(cm.index <= pd.Period(b, 'M'))]) for lab, b in (('<=2018-12', '2018-12'), ('<=2019-01', '2019-01'), ('<=2019-04', '2019-04'))})
# ---------- 容量 ----------
def cap(pls, lg):
    pmap = {}
    for pl in pls:
        for q in pl: pmap[q['t']] = q['p']
    rows = []
    for (i, te, ts, nb, frac) in lg['buys']:
        p = pmap[ts]
        for j in nb: rows.append((p, 1e7 * frac / C.V60[ts, j]))
    df = pd.DataFrame(rows, columns=['p', 'r'])
    o = {}
    for lab, m in (('SEL', df.p <= pd.Period('2018-12', 'M')), ('ALL', df.p == df.p)):
        x = df.r[m]; o[lab] = dict(n=int(len(x)), gt1=float((x > 0.01).mean() * 100), med=float(x.median() * 100))
    return o
for nm, js, c1 in (('J+1', [1], 0), ('V01', [1, 11], 0), ('V02', [1, 8, 15], 0), ('V03', [1, 6, 11, 16], 0), ('V08', [1, 6, 11, 16], 1)):
    pls = TRN.plans_for(js); t0 = min(pl[0]['t'] for pl in pls)
    nav, lg = C.tsim(pls, *C.NORMAL, t0, C.TEND, c1=bool(c1)); P(f'cap_{nm}', cap(pls, lg))
# ---------- B5 ----------
J1 = dn['J+1'].dropna(); V8 = dn['V08'].dropna()
w1, w8 = C.wret(J1, C.OFFSD), C.wret(V8, C.OFFSD); e1, e8 = C.exw(J1, C.OFFSD), C.exw(V8, C.OFFSD)
m = DT.mask(w1.index, None, '2018-12'); d = (w8 - w1)[m]; best = d.idxmax()
keep = [x for x in d.index if x != best]
P('B5', dict(best=str(DT.AM[best]), B1=float(d[keep].mean() * 100), B2=float((e8[keep].std(ddof=1) / e1[keep].std(ddof=1) - 1) * 100)))
# ---------- 損益兩平、全額手續費 ----------
grid = [0, 0.001, 0.002, 0.003, 0.005, 0.008, 0.012, 0.015]
def be(fn):
    ts, ms = [], []
    for s in grid:
        nav = fn(C.slip(s)); ex = C.exw(nav, C.OFFSD); ts.append(C.tst(ex)); ms.append(float(ex.mean()))
    def cross(y, lv):
        for a in range(len(grid) - 1):
            if (y[a] - lv) * (y[a + 1] - lv) <= 0 and y[a] != y[a + 1]:
                return (grid[a] + (lv - y[a]) * (grid[a + 1] - grid[a]) / (y[a + 1] - y[a])) * 100
        return None
    return dict(t=[round(x, 3) for x in ts], s23=cross(ts, 2.3), s20=cross(ts, 2.0), s0=cross(ms, 0.0))
P('BE_OFF', be(lambda cs: DT.off_run(cs)[0])); P('BE_J1', be(lambda cs: TRN.run([1], costs=cs)[0])); P('BE_V08', be(lambda cs: TRN.run([1, 6, 11, 16], 0, 0, 1, costs=cs)[0]))
FULLFEE = (0.001425 + 0.001, 0.001425 + 0.003 + 0.001)
n0 = DT.off_run(FULLFEE)[0]; n8 = TRN.run([1, 6, 11, 16], 0, 0, 1, costs=FULLFEE)[0]
P('fullfee', dict(OFF_t=C.tst(C.exw(n0, C.OFFSD)), OFF_cagr=C.cagr(n0)[0], V08_t=C.tst(C.exw(n8, C.OFFSD))))
# ---------- 位置平移 ----------
sh = {}
for j0 in range(-8, -2):
    nav, _ = TRN.run([j0, j0 + 5, j0 + 10, j0 + 15], 0, 0, 1); sh[f'T4c1_{j0}'] = C.tst(C.exw(nav, C.OFFSD))
for c1 in (0, 1):
    for j0 in range(-8, 3):
        nav, _ = TRN.run([j0, j0 + 10], 0, 0, c1); sh[f'T2c{c1}_{j0}'] = C.tst(C.exw(nav, C.OFFSD))
P('shift', sh)
for pre in ('T4c1_', 'T2c0_', 'T2c1_'):
    v = np.array([x for k, x in sh.items() if k.startswith(pre)]); P(f'shift_{pre}', dict(min=v.min(), max=v.max(), med=float(np.median(v)), sd=float(v.std(ddof=1)), sd0=float(v.std())))
json.dump(out, open(H / 'ind_misc.json', 'w'), indent=1, default=float)
