import sys, pathlib, json; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from tranche import *
out = {}
for lab, base, c, rng in (('T4c1', [-8, -3, 2, 7], 'c1', range(6)), ('T4c0', [-8, -3, 2, 7], 'c0', range(6)), ('T2c1', [-8, 2], 'c1', range(11)), ('T2c0', [-8, 2], 'c0', range(11))):
    rr = {}
    for d in rng:
        js = [x + d for x in base]; nav = run(js, 'a0', 'b0', c)[0]; rr[str(js)] = C.tstat(win(nav))
    v = np.array(list(rr.values()))
    out[lab] = dict(rows=rr, min=v.min(), max=v.max(), median=float(np.median(v)), sd=float(v.std(ddof=1)))
    print(lab, {k: round(x, 3) for k, x in rr.items()}, round(v.min(), 3), round(v.max(), 3), round(float(np.median(v)), 3), round(float(v.std(ddof=1)), 3), flush=True)
json.dump(out, open(pathlib.Path(__file__).parent / 'rot.json', 'w'), indent=1, default=float)
