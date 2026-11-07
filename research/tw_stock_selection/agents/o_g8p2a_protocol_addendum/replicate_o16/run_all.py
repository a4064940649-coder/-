import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[0])))
import numpy as np
import rlib as R
from multiprocessing import Pool

def jobA(a):
    d, si, ssi = a
    x = R.paths(2026110701, d, si, ssi, 2, 0, 200000)
    return (f"{d}|{R.SLAB[si]}|{R.SGRID[ssi]}", R.bstar(x))

def jobE3(a):
    d, si = a
    x = R.paths(2026110700, d, si, 4, 2, 0, 200000)
    return (f"{d}|{R.SLAB[si]}", R.evaluate(x)['up_v2'])

def jobB(a):
    kind, d, si, ssi, ai, mode = a
    x = R.paths(2026110702, d, si, ssi, ai, mode, 100000)
    r = R.evaluate(x, R.BA)
    r = {k: float(v) for k, v in r.items()}
    return (kind, f"{d}|{R.SLAB[si]}|{'DW' if ssi==5 else R.SGRID[ssi]}|a{R.ALPHAS[ai]}|m{mode}", r)

if __name__ == '__main__':
    t0 = time.time()
    out = {}
    with Pool(4) as pool:
        A = pool.map(jobA, [(d, si, ssi) for d in R.MAIN for si in range(9) for ssi in range(5)], chunksize=4)
        out['setA_q'] = dict(A)
        print('A done', time.time() - t0, flush=True)
        E3 = pool.map(jobE3, [(d, si) for d in ['ARN057','ARN08','ARN10','AR057','AR08','AR10','MA3','MAT3','D5','D3','D1'] for si in (5, 7)])
        out['E3'] = dict(E3)
        jobs = []
        for d in R.MAIN:
            for si in range(9):
                for ssi in range(5):
                    jobs.append(('main', d, si, ssi, 2, 0))
                    jobs.append(('strict', d, si, ssi, 2, 1))
                for ssi in (0, 5):
                    for ai in (0, 1, 2):
                        jobs.append(('down', d, si, ssi, ai, 0))
                    for ai in (3, 4, 5):
                        jobs.append(('power', d, si, ssi, ai, 0))
        for d in R.SENS:
            for si in range(9):
                for ssi in range(5):
                    jobs.append(('sens', d, si, ssi, 2, 0))
        print('n setB cells', len(jobs), flush=True)
        B = pool.map(jobB, jobs, chunksize=4)
        out['setB'] = [list(b) for b in B]
    out['runtime_s'] = time.time() - t0
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'rep_raw.json'), 'w'))
    print('done', time.time() - t0)
