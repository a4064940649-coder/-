import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from tranche import *
T = pd.read_parquet(pathlib.Path(__file__).resolve().parents[1] / 'navs_oos.parquet')
def cmp(name, nav):
    t = T[name].dropna(); t = t.reindex(nav.index)
    r = (nav.pct_change() - t.pct_change()).abs()
    print(name, 'max abs daily ret diff', float(r.max()), 'n>1e-10', int((r > 1e-10).sum()), 'their range', T[name].first_valid_index().date(), T[name].last_valid_index().date())
off = [(m, C.OFFICIAL[m]) for m in C.ANCHORS]
pl = C.make_plan(off, 10, None); nav, _ = C.tsim([pl], *C.NORMAL, T0, C.TEND); cmp('OFF', nav)
pl = C.make_plan(off, 10, C.AB(0)); nav, _ = C.tsim([pl], *C.NORMAL, T0, C.TEND); cmp('OFF_AB0', nav)
nav, _, _ = run([1]); cmp('J1_common', nav)
for nm, cst in (('V08_normal', C.NORMAL), ('V08_gross', C.ZERO), ('V08_stress_flat', C.FLAT), ('V08_stress_plus03', C.P03)):
    nav, _, _ = run([1, 6, 11, 16], 'a0', 'b0', 'c1', cost=cst); cmp(nm, nav)
