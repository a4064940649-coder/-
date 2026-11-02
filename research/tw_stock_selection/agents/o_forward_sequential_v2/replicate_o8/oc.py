import numpy as np, pandas as pd, json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SIG = 6.1414
C = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'calib.json')))['6.1414']

def resid():
    med = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'decision_line.csv'), index_col=0).iloc[:, 0]
    med.index = pd.PeriodIndex(med.index, freq='M')
    r = med[med.index >= pd.Period('2019-05', 'M')].values
    return r - r.mean()

def gen(dgp, alpha, N, H, rng, sig=SIG):
    if dgp == 'D1':
        e = rng.standard_normal((N, H)) * sig
    elif dgp == 'D2':
        r = resid(); T = len(r); nb = -(-H // 6)
        st = rng.integers(0, T, (N, nb))
        idx = (st[:, :, None] + np.arange(6)[None, None, :]) % T
        e = r[idx.reshape(N, -1)[:, :H]]
    elif dgp == 'D3':
        t = rng.standard_t(4, (N, H)) / np.sqrt(2)
        ph, hl, lh = 0.347, 0.06, 0.0319
        s = rng.random(N) < ph; u = rng.random((N, H)); M = np.empty((N, H))
        for k in range(H):
            if k > 0: s = np.where(s, u[:, k] >= hl, u[:, k] < lh)
            M[:, k] = np.where(s, 1.4, 0.7)
        e = t * M * sig
    return alpha + e

def run_rule(x, rule, sig=SIG, prior=None, cal=C, cu=None, cd=None, H_c=36):
    """returns (t_up, t_down) arrays: month of first action (inf if none); absorbing."""
    N, H = x.shape; n = np.arange(1, H + 1); S = np.cumsum(x, 1); z = S / (sig * np.sqrt(n))
    tu = np.full(N, np.inf); td = np.full(N, np.inf); done = np.zeros(N, bool)
    cu_ = cal['c_u'] if cu is None else cu; cd_ = cal['c_d'] if cd is None else cd
    if rule in ('R1', 'R3'):
        th, lp = prior  # grid, log prior
    C_ = np.zeros(N)
    for t in range(1, H + 1):
        up = np.zeros(N, bool); dn = np.zeros(N, bool)
        zt = z[:, t - 1]
        if rule == 'R2' and t >= 12 and (t - 12) % 6 == 0:
            up = zt >= cu_ * np.sqrt(H_c / t); dn = zt <= -cd_
        elif rule == 'R4' and t in (12, 24, 36, 48, 60):
            up = zt >= cal['c_u4']; dn = zt <= -cal['c_d4']
        if rule == 'R3':
            C_ = np.minimum(0, C_ + (x[:, t - 1] + 0.5) / sig)
            if t >= 12: dn = C_ <= -cal['h']
        if rule in ('R1', 'R3') and t >= 12 and (t - 12) % 6 == 0:
            m = S[:, t - 1] / t; v = sig ** 2 / t
            ll = lp[None, :] - 0.5 * (m[:, None] - th[None, :]) ** 2 / v
            ll -= ll.max(1, keepdims=True); w = np.exp(ll); w /= w.sum(1, keepdims=True)
            pu = w[:, th > 0.25].sum(1); pd_ = w[:, th < 0].sum(1)
            up = pu >= 0.975
            if rule == 'R1': dn = pd_ >= 0.80
        dn = dn & ~done & ~up; up = up & ~done
        tu[up] = t; td[dn] = t; done |= up | dn
    return tu, td

def oc_table(rules, prior=None, N=40000, H=60, alphas=(-1, -0.5, 0, 0.5, 1)):
    out = []
    for dgp in ('D1', 'D2', 'D3'):
        for a in alphas:
            rng = np.random.default_rng([20261105, round((a + 2) * 100), ord(dgp[-1])])
            x = gen(dgp, a, N, H, rng)
            for r in rules:
                tu, td = run_rule(x, r, prior=prior)
                for hh in (36, 60):
                    out.append(dict(dgp=dgp, alpha=a, rule=r, H=hh, up=(tu <= hh).mean(), down=(td <= hh).mean()))
    return pd.DataFrame(out)

if __name__ == '__main__':
    rules = sys.argv[1].split(',') if len(sys.argv) > 1 else ['R2', 'R4']
    prior = None
    if os.path.exists('my_prior.npz'):
        z = np.load('my_prior.npz'); prior = (z['th'], np.log(np.maximum(z['p'], 1e-300)))
    df = oc_table(rules, prior)
    df.to_csv('oc_' + '_'.join(rules) + '.csv', index=False)
    d36 = df[df.H == 36]
    for r in rules:
        q = d36[d36.rule == r]
        pw = q[q.alpha == -1].set_index('dgp').down; fu = q[q.alpha == 0].set_index('dgp').up
        fd = q[q.alpha == 0].set_index('dgp').down; u1 = q[q.alpha == 1].set_index('dgp').up
        print(r, 'power', pw.round(3).tolist(), 'falseUp', fu.round(3).tolist(), 'falseDown', fd.round(3).tolist(), 'up@+1', u1.round(3).tolist())
        q60 = df[(df.H == 60) & (df.rule == r)]; print('   max up within 60 any alpha:', q60.up.max().round(3))
