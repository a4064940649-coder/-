# Large-sample check of R2 D1 at 36 months: simulate the 5 look-points exactly (Gaussian increments), 4M paths.
import numpy as np, json
C = json.load(open('calib.json'))['6.1414']; sig = 6.1414
looks = np.array([12, 18, 24, 30, 36]); rng = np.random.default_rng(7); N = 4_000_000
for a in (0, -1):
    inc = np.diff(np.concatenate([[0], looks]))
    S = np.cumsum(rng.standard_normal((N, 5)) * np.sqrt(inc) * sig + a * inc, 1)
    z = S / (sig * np.sqrt(looks))
    up = z >= C['c_u'] * np.sqrt(36 / looks); dn = z <= -C['c_d']
    fu = np.where(up.any(1), up.argmax(1), 99); fd = np.where(dn.any(1), dn.argmax(1), 99)
    print('alpha', a, 'up', (fu < fd).mean(), 'down', (fd < fu).mean())
