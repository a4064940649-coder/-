import sys, os, pickle, time, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, rsim
from multiprocessing import Pool
TIERS = [3e5, 1e6, 3e6, 1e7]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')

def job(a):
    name, seed, tier, mode, kw = a
    r = rsim.simulate(seed, tier, mode, **kw)
    return (name, seed, tier, mode), r

VARIANTS = {
    'A': dict(fee='A'), 'B': dict(fee='B'), 'C': dict(fee='C'),
    'STRESS': dict(fee='B', kappa=0.30958472420094185, odd_stress=True),
    'd1': dict(fee='D1', odd_time=False, odd_spread=False, impact=False, cap=False),
    'd2': dict(fee='D1', odd_spread=False, impact=False, cap=False),
    'd3': dict(fee='D1', impact=False, cap=False),
    'd4': dict(fee='A', impact=False, cap=False),
    'd5': dict(fee='A', cap=False),
    'KLO': dict(kappa=0.30958472420094185), 'KHI': dict(kappa=1.038273615635179),
    'OAHALF': dict(oa_half=True), 'ODDSTRESS': dict(odd_stress=True), 'KEEP': dict(keep=True),
}
MODES = {'A': ('GROW', 'FIX'), 'B': ('GROW', 'FIX'), 'C': ('GROW', 'FIX'), 'STRESS': ('GROW', 'FIX')}

if __name__ == '__main__':
    names = sys.argv[1].split(',')
    seeds = list(range(40))
    jobs = []
    for nm in names:
        for mode in MODES.get(nm, ('GROW', 'FIX') if nm.startswith('d') else ('GROW',)):
            for tier in TIERS:
                for s in seeds:
                    jobs.append((nm, s, tier, mode, VARIANTS[nm]))
                if nm == 'A':
                    jobs.append((nm, -1, tier, mode, VARIANTS[nm]))
    os.makedirs(OUT, exist_ok=True)
    t = time.time(); res = {}
    with Pool(4, initializer=rsim.load) as p:
        for key, r in p.imap_unordered(job, jobs, chunksize=4):
            res[key] = r
    print('done', len(res), time.time() - t)
    for nm in names:
        sub = {k: v for k, v in res.items() if k[0] == nm}
        pickle.dump(sub, open(os.path.join(OUT, nm + '.pkl'), 'wb'))
