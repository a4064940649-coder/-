import sys, pathlib, json; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ind_core as C, numpy as np, pandas as pd
H = pathlib.Path(__file__).resolve().parent
rev, endm = sys.argv[1], sys.argv[2]
js = [int(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else list(range(-8, 13))
C.OPT['rev'] = rev
def run_j(j, ab, costs=C.NORMAL, w=0.5):
    days = [(p, C.J(p, j)) for p in C.ANCH]
    pl = C.plan(days, ab, w=w)
    if endm == 'END': e_ = C.END
    else:
        x = C.J(pd.Period('2026-10', 'M'), j); e_ = x if (x is not None and x in C.POS) else C.IDX[-1]
    wd = [(p, s) for p, s in days if s is not None and s in C.POS]     # 自己的窗：所有在面板內的訊號日（不換股＝續抱）
    t0 = C.POS[wd[0][1]]; t1 = C.POS[e_]
    nav, lg = C.tsim([pl], *costs, t0, t1)
    sd = [s for _, s in wd] + [e_]
    run_j.am = {s: p for p, s in wd}
    return nav, lg, pl, sd, e_
if __name__ == '__main__':
    res = {}; navs = {}
    for j in js:
        nav, lg, pl, sd, e_ = run_j(j, C.AB(0))
        ex = C.exw(nav, sd); am = run_j.am
        P = np.array([C.per(am[d]) for d in ex.index])
        cg, mdd = C.cagr(nav)
        r = dict(CAGR=cg, mdd=mdd, t=C.tst(ex), n=len(ex), t_SEL=C.tst(ex[P == 'SEL']), t_VAL=C.tst(ex[P == 'VAL']), t_HOLD=C.tst(ex[P == 'HOLD']),
                 t_cal=C.tst(C.calm(nav)), t_common=C.tst(C.exw(nav, [d for d in C.OFFSD if d >= nav.index[0]])), turn=lg['turnover'], nplan=len(pl))
        res[j] = r; navs[f'J{j:+d}'] = nav
        print(j, {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
    json.dump(res, open(H / f'ind_jscan_{rev}_{endm}.json', 'w'), indent=1)
    pd.DataFrame(navs).to_parquet(H / f'ind_jscan_{rev}_{endm}.parquet')
