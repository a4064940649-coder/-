# o15 獨立複製：G8P-2 序貫檢定的蒙地卡羅函式庫（只依凍結規格自寫，未讀研究員程式）
import json, hashlib, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.join(HERE, '..', 'G8_PROTOCOL_v2.json')
PROTO_V1 = os.path.join(HERE, '..', '..', 'o_g8_protocol_fin_erratum', 'G8_PROTOCOL.json')

SIG_D = 6.141373712658076
C = 1.0596378779547428
DW = 0.527
SIGMAS = [SIG_D, 6.51, 7.0, 7.5, 8.0, 8.4, 9.0, 9.5, 10.4]
SLAB = ['6.14', '6.51', '7.0', '7.5', '8.0', '8.4', '9.0', '9.5', '10.4']
SGRID = [0.0, 0.2, 0.4, 0.577, 0.78]
# 協定凍結參數（OC 比對時使用）
B_P = 1.9337576513495873
CD_P = 1.2098465850539828
# G8P-1 原樣
B1 = 1.6787025976148637
CD1 = 0.8946228207485066
REFIT = 1.5


def load_resid():
    d = json.load(open(PROTO))
    r = d['data']['resid_post2019']
    assert hashlib.sha256(json.dumps(r).encode()).hexdigest().startswith('057770ae')
    return np.array(r, float)


RESID = load_resid()
RESID_X = np.delete(RESID, 24)          # 2019-05 起第 25 個月＝2021-05（+26.14）
RESID_X = RESID_X - RESID_X.mean()


def gen(dgp, sigma, mu, P, rng, H=24):
    """回傳 P×H 的月決策線路徑。"""
    if dgp == 'D1':
        return mu + sigma * rng.standard_normal((P, H))
    if dgp == 'D2':
        return mu + sigma * rng.standard_t(4, (P, H)) / np.sqrt(2.0)
    if dgp in ('D3', 'D3x'):
        r = RESID if dgp == 'D3' else RESID_X
        r = r * (sigma / r.std(ddof=1))
        N = len(r)
        idx = np.empty((P, H), np.int64)
        cur = rng.integers(0, N, P)
        idx[:, 0] = cur
        for t in range(1, H):
            new = rng.random(P) < 1.0 / 6.0
            cur = np.where(new, rng.integers(0, N, P), (cur + 1) % N)
            idx[:, t] = cur
        return mu + r[idx]
    if dgp == 'D4':
        pHL, pLH = 0.06, 0.0319
        piH = pLH / (pLH + pHL)
        st = rng.random(P) < piH
        m = np.empty((P, H))
        for t in range(H):
            if t > 0:
                u = rng.random(P)
                st = np.where(st, u >= pHL, u < pLH)
            m[:, t] = np.where(st, 1.4, 0.7)
        return mu + sigma * m * rng.standard_t(4, (P, H)) / np.sqrt(2.0)
    raise ValueError(dgp)


def prefix_stats(X):
    S = np.cumsum(X, 1)
    Q = np.cumsum(X * X, 1)
    return S, Q


def zstat(S, Q, n, delta=0.0, stat='MAXSD'):
    Sn = S[:, n - 1]
    Qn = Q[:, n - 1]
    # (x-δ) 的樣本 sd 與 x 相同
    var = (Qn - Sn * Sn / n) / (n - 1)
    sd = np.sqrt(np.maximum(var, 0.0))
    if stat == 'MAXSD':
        sig = np.maximum(C * SIG_D, sd)
    else:
        sig = np.full_like(sd, C * SIG_D)
    return (Sn - delta * n) / (sig * np.sqrt(n)), sd


def eval_v(X, b, cd, delta=0.0, upgrade=True):
    """G8P-2 V1（delta=0）／V3（delta=0.2）／V4（upgrade=False）。
    回傳 dict：up（24 個月內 S1 證據，含字面閘）、fail、dn（序貫下修，不含字面閘，含升級吸收）、
    dn_raw（min z ≤ −c_d，不含任何閘）、lit、up12、Bstar。"""
    S, Q = prefix_stats(X)
    z12, _ = zstat(S, Q, 12, delta)
    z18, _ = zstat(S, Q, 18, delta)
    z24, _ = zstat(S, Q, 24, delta)
    lit = S[:, 11] > 0
    # 狀態機（含字面閘）
    fail12 = (~lit) | (z12 <= -cd)
    up12 = (~fail12) & (z12 >= b * np.sqrt(2)) & upgrade
    alive = ~(fail12 | up12)
    fail18 = alive & (z18 <= -cd)
    alive = alive & ~fail18
    fail24 = alive & (z24 <= -cd)
    up24 = alive & ~fail24 & (z24 >= b) & upgrade
    up = up12 | up24
    fail = fail12 | fail18 | fail24
    # 序貫下修（不含字面閘；升級吸收）
    d12 = z12 <= -cd
    u12 = (~d12) & (z12 >= b * np.sqrt(2)) & upgrade
    a = ~(d12 | u12)
    d18 = a & (z18 <= -cd)
    a = a & ~d18
    d24 = a & (z24 <= -cd)
    dn = d12 | d18 | d24
    dn_raw = np.minimum(np.minimum(z12, z18), z24) <= -cd
    # B*（b 校準用）
    Bs = np.where(z18 <= -cd, -np.inf, z24)
    Bs = np.maximum(z12 / np.sqrt(2), Bs)
    Bs = np.where(fail12, -np.inf, Bs)
    return dict(up=up, fail=fail, dn=dn, dn_raw=dn_raw, lit=lit, up12=up12, Bstar=Bs,
                minz=np.minimum(np.minimum(z12, z18), z24))


def eval_g8p1(X):
    """G8P-1 原樣：σ_eff＝c·σ_d；n≥12 檢視點樣本 sd > 1.5σ_d → 自該月起 σ_eff＝max(cσ_d, sd)，只觸發一次；
    升級 z_n ≥ b√(24/n)（n＝12、24）；下修 n＝12、18、24 z ≤ −0.8946；字面 G8 閘。"""
    S, Q = prefix_stats(X)
    P = X.shape[0]
    sig = np.full(P, C * SIG_D)
    refit_done = np.zeros(P, bool)
    state = np.zeros(P, np.int8)   # 0 進行中 1 升級 2 失敗
    for n in (12, 18, 24):
        Sn = S[:, n - 1]
        var = (Q[:, n - 1] - Sn * Sn / n) / (n - 1)
        sd = np.sqrt(np.maximum(var, 0))
        trig = (~refit_done) & (sd > REFIT * SIG_D)
        sig = np.where(trig, np.maximum(C * SIG_D, sd), sig)
        refit_done |= trig
        z = Sn / (sig * np.sqrt(n))
        act = state == 0
        if n == 12:
            lf = act & ~(S[:, 11] > 0)
            state[lf] = 2
            act = state == 0
        dn = act & (z <= -CD1)
        state[dn] = 2
        act = state == 0
        if n in (12, 24):
            u = act & (z >= B1 * np.sqrt(24 / n))
            state[u] = 1
    return dict(up=state == 1, fail=state == 2)


def cell_rng(base, *k):
    return np.random.default_rng(np.random.SeedSequence([base] + [int(v) for v in k]))
