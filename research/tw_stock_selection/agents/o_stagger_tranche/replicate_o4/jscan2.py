import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import rcore as C, numpy as np, pandas as pd, json
OFF = [C.OFFICIAL[m] for m in C.ANCHORS] + [C.END]
mode = sys.argv[1] if len(sys.argv) > 1 else 'month'; C.REVRULE['mode'] = mode
endmode = sys.argv[2] if len(sys.argv) > 2 else 'END'
def per(m): return 'SEL' if m <= pd.Period('2018-12','M') else ('VAL' if m <= pd.Period('2023-12','M') else 'HOLD')
res = {}; navs = {}
for j in range(-8, 13):
    days = [(m, C.jday(m, j)) for m in C.ANCHORS]
    pl = C.make_plan(days, 10, C.AB(0))
    wd = [(m, s) for m, s in days if s in C.POS]          # 自己的窗：所有在面板內的 J 日
    if endmode == 'END': e_ = C.END
    else:
        nx = C.jday(pd.Period('2026-10', 'M'), j); e_ = nx if (nx is not None and nx in C.POS) else C.IDX[-1]
    t0 = C.POS[wd[0][1]]; t1 = C.POS[e_]
    nav, lg = C.tsim([pl], *C.NORMAL, t0, t1)
    sd = [s for _, s in wd] + [e_]
    ex = C.excess(nav, sd)
    am = {s: m for m, s in wd}
    P = np.array([per(am[d]) for d in ex.index])
    cg, mdd = C.cagr_mdd(nav, nav.index[0], e_)
    exo = C.excess(nav, [d for d in OFF if d >= nav.index[0]])
    r = dict(CAGR=cg, mdd=mdd, t=C.tstat(ex), n=len(ex), nS=int((P=='SEL').sum()), nH=int((P=='HOLD').sum()), t_SEL=C.tstat(ex[P=='SEL']), t_HOLD=C.tstat(ex[P=='HOLD']), t_VAL=C.tstat(ex[P=='VAL']),
             t_cal=C.tstat(C.cal_t(nav)), t_common=C.tstat(exo), turn=lg['turnover'], nplan=len(pl),
             ex_mean=float(ex.mean()*100), ex_sd=float(ex.std()*100))
    res[j] = r; navs[f'J{j:+d}'] = nav
    print(j, {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
json.dump(res, open(pathlib.Path(__file__).parent / f'jscan_{mode}_{endmode}.json', 'w'), indent=1)
pd.DataFrame(navs).to_parquet(pathlib.Path(__file__).parent / f'jscan_navs_{mode}_{endmode}.parquet')
