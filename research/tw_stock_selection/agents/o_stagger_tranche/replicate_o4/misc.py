import sys, pathlib, json; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from tranche import *
out = {}
DN = pd.read_parquet(pathlib.Path(__file__).parent / 'detail_navs.parquet')
# drop 2021（錨定年）
for k in ('OFF', 'J1', 'V08'):
    x = win(DN[k].dropna()); out[f'drop2021_{k}'] = C.tstat(x[x.index.year != 2021])
# EW 變體
off = [(m, C.OFFICIAL[m]) for m in C.ANCHORS]
plO = C.make_plan(off, 10, None)
ewO_k = C.K.simulate([(e, np.where(C._CS[(m, s, None)]['el'])[0]) for (e, pk, m, s) in plO], 0, 0, T0, C.TEND)[0]
out['OFF_vsEW_Ksim'] = C.tstat(C.excess(DN['OFF'].dropna(), SD, ewO_k))
out['V08_vs_OFFpoolEW'] = C.tstat(C.excess(DN['V08'].dropna(), SD, ewO_k))
pJ = plans_for([1])[0]
ewJ = C.K.simulate([(e, np.where([v for kk, v in C._CS.items() if kk[0] == m and kk[1] == s][0]['el'])[0]) for (e, pk, m, s) in pJ], 0, 0, T0, C.TEND)[0]
out['V08_vs_J1poolEW'] = C.tstat(C.excess(DN['V08'].dropna(), SD, ewJ))
# 自己窗、21 版拿掉最好 10 月的 t 中位數
NV = pd.read_parquet(pathlib.Path(__file__).parent / 'jscan_navs_anchor_next.parquet')
db = []
for j in range(-8, 13):
    nav = NV[f'J{j:+d}'].dropna()
    days = [(m, C.jday(m, j)) for m in C.ANCHORS]; wd = [s for m, s in days if s in C.POS]
    x = C.excess(nav, wd + [nav.index[-1]])
    db.append(C.tstat(x.drop(x.sort_values(ascending=False).index[:10])))
out['drop_best10_median_21'] = float(np.median(db))
# J0 seeds
r0 = []
for sd in range(5):
    days = [(m, C.jday(m, 0)) for m in C.ANCHORS]; pl = C.make_plan(days, 10, C.AB(sd))
    e_ = C.jday(pd.Period('2026-10', 'M'), 0)
    nav, _ = C.tsim([pl], *C.NORMAL, C.POS[pl[0][3]], C.POS[e_])
    r0.append(C.tstat(C.excess(nav, [p[3] for p in pl] + [e_])))
out['J0_seeds_t'] = r0
# 全額手續費
FULL = (0.001425 + 0.001, 0.001425 + 0.003 + 0.001)
navF = C.tsim([plO], *FULL, T0, C.TEND)[0]
out['fullfee_OFF'] = (C.tstat(win(navF)), C.cagr_mdd(navF, navF.index[0], C.END)[0])
nav8F = run([1, 6, 11, 16], 'a0', 'b0', 'c1', cost=FULL)[0]
out['fullfee_V08'] = (C.tstat(win(nav8F)), C.cagr_mdd(nav8F, nav8F.index[0], C.END)[0])
# 損益兩平
grid = np.round(np.arange(0, 1.501, 0.1), 3)
def be(fn):
    ts, ms = [], []
    for g in grid:
        cst = (C.K.FEE + g / 100, C.K.FEE + 0.003 + g / 100)
        x = win(fn(cst)); ts.append(C.tstat(x)); ms.append(float(x.mean()))
    def cross(v, lvl):
        for i in range(len(v) - 1):
            if v[i] >= lvl > v[i + 1]: return float(grid[i] + (grid[i + 1] - grid[i]) * (v[i] - lvl) / (v[i] - v[i + 1]))
        return None
    return dict(t23=cross(ts, 2.3), t20=cross(ts, 2.0), ex0=cross(ms, 0.0), t_at=dict(zip(grid.tolist(), ts)))
out['be_OFF'] = be(lambda cst: C.tsim([plO], *cst, T0, C.TEND)[0])
out['be_J1'] = be(lambda cst: run([1], cost=cst)[0])
out['be_V08'] = be(lambda cst: run([1, 6, 11, 16], 'a0', 'b0', 'c1', cost=cst)[0])
# 位置平移
rot = {}
for lab, base, c in (('T4c1', [1, 6, 11, 16], 'c1'), ('T2c1', [1, 11], 'c1'), ('T2c0', [1, 11], 'c0')):
    rr = {}
    for d in range(-1, 11 if lab.startswith('T2') else 6):
        js = [x + d for x in base]
        if min(js) < 0: continue
        nav = run(js, 'a0', 'b0', c)[0]; rr[d] = C.tstat(win(nav))
    rot[lab] = rr
out['rotation'] = rot
json.dump(out, open(pathlib.Path(__file__).parent / 'misc.json', 'w'), indent=1, default=float)
for k, v in out.items(): print(k, v)
