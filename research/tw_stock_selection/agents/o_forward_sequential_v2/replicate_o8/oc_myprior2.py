import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import oc, hier
raw = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'my_prior.npz'))['raw']
w = hier.transform(raw, 0.308, 1.2)
df = oc.oc_table(['R1'], (hier.GRID, np.log(np.maximum(w, 1e-300))), alphas=(-1, 0, 1))
d = df[df.H == 36]
print('R1 own s,k prior: power', d[d.alpha == -1].down.round(3).tolist(), 'falseDown', d[d.alpha == 0].down.round(3).tolist(), 'maxUp60', df[df.H == 60].up.max())
