# Diagnostic only: run MY R1 implementation with the researcher's published prior grid (data output, not code)
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import oc
z = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'forward_prior_main.npz'))
g = z['theta']; p = z['prior']; mu = (g * p).sum()
print('their prior mean %.3f sd %.3f P>0 %.3f P>.25 %.3f' % (mu, np.sqrt(((g - mu) ** 2 * p).sum()), p[g > 0].sum(), p[g > 0.25].sum()))
df = oc.oc_table(['R1'], (g, np.log(np.maximum(p, 1e-300))), alphas=(-1, 0, 1))
d = df[df.H == 36]
print(d[d.alpha == -1][['dgp', 'down']].to_string(), d[d.alpha == 0][['dgp', 'down', 'up']].to_string(), df[df.H == 60].up.max())
