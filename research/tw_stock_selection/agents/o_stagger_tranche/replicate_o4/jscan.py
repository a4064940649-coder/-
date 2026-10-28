import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import rcore as C, numpy as np, pandas as pd, json
OFF = [C.OFFICIAL[m] for m in C.ANCHORS] + [C.END]
def per(m):
    return 'SEL' if m <= pd.Period('2018-12','M') else ('VAL' if m <= pd.Period('2023-12','M') else 'HOLD')
res = {}; navs = {}
for j in range(-8, 13):
    days = [(m, C.jday(m, j)) for m in C.ANCHORS]
    ab = C.AB(0)
    pl = C.make_plan(days, 10, ab)
    t0 = C.POS[pl[0][3]]; t1 = C.TEND
    nav, lg = C.tsim([pl], *C.NORMAL, t0, t1)
    sd = [p[3] for p in pl] + [C.END]
    ex = C.excess(nav, sd)
    am = {p[3]: p[2] for p in pl}
    P = pd.Series([per(am[d]) for d in ex.index], index=ex.index)
    cg, mdd = C.cagr_mdd(nav, nav.index[0], C.END)
    # common window
    exo = C.excess(nav, [d for d in OFF if d >= nav.index[0]])
    r = dict(CAGR=cg, mdd=mdd, t=C.tstat(ex), n=len(ex), t_SEL=C.tstat(ex[P=='SEL']), t_HOLD=C.tstat(ex[P=='HOLD']), t_VAL=C.tstat(ex[P=='VAL']),
             t_cal=C.tstat(C.cal_t(nav)), t_common=C.tstat(exo), turn=lg['turnover'], nplan=len(pl))
    res[j] = r; navs[f'J{j:+d}'] = nav
    print(j, {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
json.dump(res, open(pathlib.Path(__file__).parent / 'jscan.json', 'w'), indent=1)
pd.DataFrame(navs).to_parquet(pathlib.Path(__file__).parent / 'jscan_navs.parquet')
