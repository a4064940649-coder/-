import numpy as np, json
SIG = 6.1414  # spec value; also run with own 6.13956
def calib(sig, N=200000, H=36, seed=20261102):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((N, 120))[:, :H]
    e1 = rng.standard_normal((N, 120))[:, :H]  # second draw for alpha=-1 (matches researcher to 4 dp)
    out = {}
    n = np.arange(1, H + 1)
    S0 = np.cumsum(e * sig, 1); Sm = np.cumsum(e1 * sig - 1.0, 1)
    z0 = S0 / (sig * np.sqrt(n)); zm = Sm / (sig * np.sqrt(n))
    looks = np.array([12, 18, 24, 30, 36]) - 1
    # R2 up: max over looks of z/sqrt(36/n) >= c
    stat = (z0[:, looks] / np.sqrt(36 / (looks + 1))).max(1)
    out['c_u'] = np.quantile(stat, 0.96)
    stat = (-zm[:, looks]).max(1); out['c_d'] = np.quantile(stat, 0.45)
    l4 = np.array([12, 24, 36]) - 1
    out['c_u4'] = np.quantile(z0[:, l4].max(1), 0.96)
    out['c_d4'] = np.quantile((-zm[:, l4]).max(1), 0.45)
    # CUSUM
    x = e1 * sig - 1.0; C = np.zeros(N); mn = np.full(N, np.inf)
    for t in range(H):
        C = np.minimum(0, C + (x[:, t] + 0.5) / sig)
        if t >= 11: mn = np.minimum(mn, C)
    out['h'] = np.quantile(-mn, 0.45)
    return out
if __name__ == '__main__':
    r = {str(s): calib(s) for s in (6.1414, 6.13956)}
    print(json.dumps(r, indent=1)); json.dump(r, open('calib.json', 'w'), indent=1)
