import sys, pathlib, json; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from tranche import *
"""每個 J 版本、三種口徑（一般、零成本；權重 0.5／價格腿 1.0／營收腿 0.0），存 nav。"""
navs = {}
for w in (0.5, 1.0, 0.0):
    C.W52['w'] = w
    for cname, cst in (('normal', C.NORMAL), ('gross', C.ZERO)):
        if w != 0.5 and cname == 'gross': continue
        for j in range(-8, 13):
            days = [(m, C.jday(m, j)) for m in C.ANCHORS]
            pl = C.make_plan(days, 10, C.AB(0))
            wd = [(m, s) for m, s in days if s in C.POS]
            nav, _ = C.tsim([pl], *cst, C.POS[wd[0][1]], C.TEND)
            navs[f'w{w}|{cname}|J{j:+d}'] = nav
        print(w, cname, flush=True)
pd.DataFrame(navs).to_parquet(pathlib.Path(__file__).parent / 'scan_all_navs.parquet')
