import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import oc
def cal(sig, tu, N=200000):
    rng = np.random.default_rng(20261102); e0 = rng.standard_normal((N, 120))[:, :36]; e1 = rng.standard_normal((N, 120))[:, :36]
    n = np.arange(1, 37); z0 = np.cumsum(e0, 1) / np.sqrt(n); zm = np.cumsum(e1 * sig - 1, 1) / (sig * np.sqrt(n))
    l2 = np.arange(12, 37, 6) - 1; l4 = np.array([12, 24, 36]) - 1
    return dict(c_u=np.quantile((z0[:, l2] / np.sqrt(36 / (l2 + 1))).max(1), 1 - tu), c_d=np.quantile((-zm[:, l2]).max(1), .45),
                c_u4=np.quantile(z0[:, l4].max(1), 1 - tu), c_d4=np.quantile((-zm[:, l4]).max(1), .45))
for lab, sig, tu in (('target_up 3%', 6.1414, .03), ('sigma max(HAC,iid)=6.5097 + 3%', 6.509652, .03)):
    c = cal(sig, tu); print(lab, {k: round(v, 4) for k, v in c.items()})
    for r in ('R2', 'R4'):
        o = {}
        for dgp in ('D1', 'D2', 'D3'):
            for a in (-1, 0, 1):
                rng = np.random.default_rng([20261105, round((a + 2) * 100), ord(dgp[-1])])
                x = oc.gen(dgp, a, 40000, 36, rng)  # DGP noise scale stays sigma_design 6.1414 (D1/D3)
                tu_, td_ = oc.run_rule(x, r, sig=sig, cal=c)
                o[(dgp, a)] = ((tu_ <= 36).mean(), (td_ <= 36).mean())
        print('  ', r, 'power', [round(o[(d, -1)][1], 3) for d in ('D1', 'D2', 'D3')], 'falseUp', [round(o[(d, 0)][0], 3) for d in ('D1', 'D2', 'D3')],
              'falseDown', [round(o[(d, 0)][1], 3) for d in ('D1', 'D2', 'D3')], 'up@+1', [round(o[(d, 1)][0], 3) for d in ('D1', 'D2', 'D3')], flush=True)
