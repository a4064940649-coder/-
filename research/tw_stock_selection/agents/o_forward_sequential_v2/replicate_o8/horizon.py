import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import oc
sig = 6.1414
rng = np.random.default_rng(20261102); N = 200000
e0 = rng.standard_normal((N, 120)); e1 = rng.standard_normal((N, 120))
for Hc in (36, 42, 48, 60, 72, 84, 96, 120):
    n = np.arange(1, Hc + 1); looks = np.arange(12, Hc + 1, 6) - 1
    z0 = np.cumsum(e0[:, :Hc] * sig, 1) / (sig * np.sqrt(n)); zm = np.cumsum(e1[:, :Hc] * sig - 1, 1) / (sig * np.sqrt(n))
    cu = np.quantile((z0[:, looks] / np.sqrt(Hc / (looks + 1))).max(1), 0.96); cd = np.quantile((-zm[:, looks]).max(1), 0.45)
    res = {}
    for dgp in ('D1', 'D2', 'D3'):
        for a in (-1, 0):
            r = np.random.default_rng([20261105, round((a + 2) * 100), ord(dgp[-1]), Hc])
            x = oc.gen(dgp, a, 20000, Hc, r)
            tu, td = oc.run_rule(x, 'R2', cu=cu, cd=cd, H_c=Hc)
            res[(dgp, a)] = ((tu <= Hc).mean(), (td <= Hc).mean())
    pw = min(res[(d, -1)][1] for d in ('D1', 'D2', 'D3')); fu = max(res[(d, 0)][0] for d in ('D1', 'D2', 'D3')); fd = max(res[(d, 0)][1] for d in ('D1', 'D2', 'D3'))
    print(Hc, 'cu %.3f cd %.3f' % (cu, cd), 'minPower %.3f maxFalseUp %.3f maxFalseDown %.3f' % (pw, fu, fd), 'pass' if (pw >= .5 and fu <= .05 and fd <= .3) else 'fail', flush=True)
