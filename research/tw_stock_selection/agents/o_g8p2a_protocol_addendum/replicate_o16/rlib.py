# Independent replication library for G8P-2a (o16). Written from the frozen spec only.
import json, hashlib, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.join(HERE, '..', 'G8_PROTOCOL_v2a.json')
SHA_PROTO = '13972450a2b21938fe892b4e8659c89c8ade56372ce3a2fff678300642ad9d33'

raw = open(PROTO, 'rb').read()
assert hashlib.sha256(raw).hexdigest() == SHA_PROTO
P_ = json.loads(raw)

SIGD = 6.141373712658076
C = 1.0596378779547428
CD = 1.2098465850539828
BV2 = 1.9337576513495873
BA = 2.2796305424941274
DW = 0.527
SIGMAS = [SIGD, 6.51, 7.0, 7.5, 8.0, 8.4, 9.0, 9.5, 10.4]
SLAB = ["6.14", "6.51", "7.0", "7.5", "8.0", "8.4", "9.0", "9.5", "10.4"]
SGRID = [0.0, 0.2, 0.4, 0.577, 0.78]  # s index 0..4 ; index 5 = DW
ALPHAS = [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
DGP_ID = dict(D1=1, D2=2, D3=3, AR03=11, AR057=12, AR08=13, AR10=14, MA1=21, MA2=22, MA3=23, D5=31,
              ARN057=41, ARN08=42, ARN10=43, MAT3=44)
MAIN = ['D1', 'D2', 'D3', 'AR03', 'AR057', 'AR08', 'AR10', 'MA1', 'MA2', 'MA3', 'D5']
SENS = ['ARN057', 'ARN08', 'ARN10', 'MAT3']


def stdz(a):
    a = np.asarray(a, float)
    a = a - a.mean()
    return a / a.std(ddof=1)


R3 = stdz(P_['data']['resid_post2019'])
R5 = stdz(P_['data']['resid_sel_D5'])
T = 24


def t_std(rng, df, size):
    return rng.standard_t(df, size=size) / np.sqrt(df / (df - 2.0))


def cbb(rng, r, n):
    N = len(r)
    idx = np.empty((n, T), np.int64)
    idx[:, 0] = rng.integers(0, N, n)
    for t in range(1, T):
        new = rng.random(n) < 1.0 / 6.0
        nxt = (idx[:, t - 1] + 1) % N
        idx[:, t] = np.where(new, rng.integers(0, N, n), nxt)
    return r[idx]


def gen_e(rng, dgp, n):
    """unit-variance innovations e (n x 24)"""
    if dgp == 'D1':
        return rng.standard_normal((n, T))
    if dgp == 'D2':
        return rng.standard_t(4, size=(n, T)) / np.sqrt(2.0)
    if dgp == 'D3':
        return cbb(rng, R3, n)
    if dgp == 'D5':
        return cbb(rng, R5, n)
    if dgp.startswith('AR'):
        normal = dgp.startswith('ARN')
        rho = {'03': 0.03, '057': 0.057, '08': 0.08, '10': 0.10}[dgp[3:] if normal else dgp[2:]]
        u = rng.standard_normal((n, T)) if normal else t_std(rng, 5, (n, T))
        e = np.empty_like(u)
        e[:, 0] = u[:, 0]
        a = np.sqrt(1 - rho * rho)
        for t in range(1, T):
            e[:, t] = rho * e[:, t - 1] + a * u[:, t]
        return e
    if dgp.startswith('MA'):
        th = {'MA1': 0.1, 'MA2': 0.2, 'MA3': 0.3, 'MAT3': 0.3}[dgp]
        u = t_std(rng, 5, (n, T + 1)) if dgp == 'MAT3' else rng.standard_normal((n, T + 1))
        return (u[:, 1:] + th * u[:, :-1]) / np.sqrt(1 + th * th)
    raise ValueError(dgp)


def cell_rng(base, dgp, si, ssi, ai, mode):
    return np.random.default_rng(np.random.SeedSequence([base, DGP_ID[dgp], si, ssi, ai, mode]))


def paths(base, dgp, si, ssi, ai, mode, P, chunk=50000):
    rng = cell_rng(base, dgp, si, ssi, ai, mode)
    sig = SIGMAS[si]
    s = DW if ssi == 5 else SGRID[ssi]
    al = ALPHAS[ai]
    mu = al + s - (DW if mode == 0 else 0.0)
    out = []
    left = P
    while left > 0:
        n = min(chunk, left)
        out.append(mu + sig * gen_e(rng, dgp, n))
        left -= n
    return np.concatenate(out)


def stats(x):
    """returns dict of S_n, s_n at n=12,18,24"""
    d = {}
    for n in (12, 18, 24):
        xs = x[:, :n]
        d[n] = (xs.sum(1), xs.std(1, ddof=1))
    return d


FAIL, NOREJ, UP = 0, 1, 2


def seq(st, kind, b, ignore_literal=False):
    """kind 'v2': z=S/(max(c sd, s) sqrt n); 'A1': z=S/(c max(sd,s) sqrt n).
    returns final state array (FAIL/NOREJ/UP), plus info: upgrade month, z dict, dn flag (sequential down)"""
    z = {}
    for n, (S, sd) in st.items():
        den = np.maximum(C * SIGD, sd) if kind == 'v2' else C * np.maximum(SIGD, sd)
        z[n] = S / (den * np.sqrt(n))
    S12 = st[12][0]
    P = len(S12)
    state = np.full(P, NOREJ)
    done = np.zeros(P, bool)
    dn = np.zeros(P, bool)
    lit_fail = ~(S12 > 0)
    if not ignore_literal:
        state[lit_fail] = FAIL; done |= lit_fail
    m = ~done & (z[12] <= -CD); state[m] = FAIL; dn |= m; done |= m
    m = ~done & (z[12] >= b * np.sqrt(2)); state[m] = UP; done |= m
    m = ~done & (z[18] <= -CD); state[m] = FAIL; dn |= m; done |= m
    m = ~done & (z[24] <= -CD); state[m] = FAIL; dn |= m; done |= m
    m = ~done & (z[24] >= b); state[m] = UP; done |= m
    return state, z, dn


def evaluate(x, ba=BA):
    st = stats(x)
    s2, z2, dn2 = seq(st, 'v2', BV2)
    sa, za, dna = seq(st, 'A1', ba)
    comb = np.minimum(s2, sa)
    # A1b0 diagnostic: A1 denominator, v2 b, combined with v2
    sb, _, dnb = seq(st, 'A1', BV2)
    combb = np.minimum(s2, sb)
    # sequential-down variants (no literal gate)
    s2n, _, dn2n = seq(st, 'v2', BV2, ignore_literal=True)
    san, _, dnan = seq(st, 'A1', ba, ignore_literal=True)
    return dict(
        up_comb=(comb == UP).mean(), up_v2=(s2 == UP).mean(), up_A1b0=(combb == UP).mean(),
        fail_comb=(comb == FAIL).mean(), fail_v2=(s2 == FAIL).mean(),
        dn_comb=(dn2 | dna).mean(), dn_v2=dn2.mean(),
        dn_comb_nolit=(dn2n | dnan).mean(), dn_v2_nolit=dn2n.mean(),
        viol_up=int(((comb == UP) & (s2 != UP)).sum()), viol_fail=int(((s2 == FAIL) & (comb != FAIL)).sum()),
        lit_pass=(st[12][0] > 0).mean(),
    )


def bstar(x):
    st = stats(x)
    s2, z2, _ = seq(st, 'v2', BV2)
    z = {}
    for n, (S, sd) in st.items():
        z[n] = S / (C * np.maximum(SIGD, sd) * np.sqrt(n))
    ok = (s2 == UP) & (st[12][0] > 0) & (z[12] > -CD)
    b24 = np.where(z[18] > -CD, z[24], -np.inf)
    B = np.maximum(z[12] / np.sqrt(2), b24)
    B = np.where(ok, B, -1e9)
    return float(np.quantile(B, 0.9525))
