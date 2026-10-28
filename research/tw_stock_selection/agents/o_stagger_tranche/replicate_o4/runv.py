import sys, pathlib, json, time; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from tranche import *
VERS = {'V01': ([1, 11], 'a0', 'b0', 'c0'), 'V02': ([1, 8, 15], 'a0', 'b0', 'c0'), 'V03': ([1, 6, 11, 16], 'a0', 'b0', 'c0'),
        'V04': ([1, 11], 'a1', 'b0', 'c0'), 'V05': ([1, 8, 15], 'a1', 'b0', 'c0'), 'V06': ([1, 6, 11, 16], 'a1', 'b0', 'c0'),
        'V07': ([1, 11], 'a0', 'b0', 'c1'), 'V08': ([1, 6, 11, 16], 'a0', 'b0', 'c1'), 'V09': ([1, 11], 'a0', 'b1', 'c0'),
        'V10': ([1, 6, 11, 16], 'a0', 'b1', 'c0'), 'V11': ([1, 6, 11, 16], 'a0', 'b1', 'c1'), 'V12': ([1, 3, 5], 'a0', 'b0', 'c0')}
sel = sys.argv[1:] or list(VERS)
navJ1, _, _ = run([1])
xJ1 = win(navJ1); sJ1 = selstats(xJ1)
print('J1 common SEL sd', sJ1.std() * 100, 'ALL t', C.tstat(xJ1))
out = {}
for v in sel:
    t_ = time.time()
    js, a, b, c = VERS[v]
    nav, lg, pls = run(js, a, b, c)
    x = win(nav); s = selstats(x)
    d = (C.winret(nav, SD) - C.winret(navJ1, SD)); dS = selstats(d)
    out[v] = dict(paired_SEL=float(dS.mean() * 100), sd_SEL=float(s.std() * 100), sd_chg=float(s.std() / sJ1.std() - 1) * 100, t_ALL=C.tstat(x), CAGR=C.cagr_mdd(nav, nav.index[0], C.END)[0])
    print(v, js, a, b, c, {k: round(val, 4) for k, val in out[v].items()}, round(time.time() - t_, 1), flush=True)
    nav.to_frame(v).to_parquet(pathlib.Path(__file__).parent / f'nav_{v}.parquet')
json.dump(out, open(pathlib.Path(__file__).parent / ('versions_' + '_'.join(sel) + '.json'), 'w'), indent=1)
