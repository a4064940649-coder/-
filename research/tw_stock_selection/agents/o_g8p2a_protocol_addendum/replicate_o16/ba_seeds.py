import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[0])))
import numpy as np, rlib as R
from multiprocessing import Pool
def job(a):
    base, d, si, ssi = a
    return R.bstar(R.paths(base, d, si, ssi, 2, 0, 200000))
if __name__ == '__main__':
    out = {}
    with Pool(4) as p:
        for base in (2026110791, 2026110792, 2026110793):
            q = p.map(job, [(base, d, si, ssi) for d in R.MAIN for si in range(9) for ssi in range(5)], chunksize=4)
            out[base] = max(R.BV2, max(q))
    v = np.array(list(out.values()))
    out['mean'] = float(v.mean()); out['sd'] = float(v.std(ddof=1))
    print(out)
    json.dump({str(k): v for k, v in out.items()}, open(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'rep_ba_seeds.json'), 'w'))
