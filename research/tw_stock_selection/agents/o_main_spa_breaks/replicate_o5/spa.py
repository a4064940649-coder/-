"""SPA／RC／StepM／有效試驗數／DSR（獨立實作）。"""
import sys; sys.path.insert(0, '.')
from rlib import *
from scipy import stats as st

def omega_all(D, L):
    return pr_omega2(D, L)

def boot_means(D, idx):
    n = D.shape[0]; B = idx.shape[0]
    Cn = np.zeros((B, n), np.float32)
    rows = np.repeat(np.arange(B), n)
    np.add.at(Cn, (rows, idx.ravel()), 1.0)
    return (Cn @ D.astype(np.float32)) / n          # (B, K)

def spa(D, focus, L, idx, stepm=True, unstd=True):
    D = np.asarray(D, float); n, K = D.shape
    db = D.mean(0); om = np.sqrt(omega_all(D, L))
    Db = boot_means(D, idx).astype(np.float64)
    sq = math.sqrt(n)
    t = sq * db / om
    T = max(0.0, t.max()); tm = t[focus]
    thr = -np.sqrt(om ** 2 / n * 2 * math.log(math.log(n)))
    gl = np.maximum(db, 0); gc = db * (db >= thr); gu = db
    out = dict(K=K, t_main=float(tm), t_max=float(t.max()), argmax=int(t.argmax()), rank=int((t > tm).sum() + 1),
               dbar_main=float(db[focus]))
    for nm, g in (('l', gl), ('c', gc), ('u', gu)):
        Ts = np.maximum(0, (sq * (Db - g) / om).max(1))
        out['p_SPA_' + nm] = float((Ts >= T).mean())
        out['p_SPA_' + nm + '_focus'] = float((Ts >= max(tm, 0)).mean())
    Vs = sq * (Db - db).max(1)
    out['p_RC'] = float((Vs >= sq * db.max()).mean()); out['p_RC_focus'] = float((Vs >= sq * db[focus]).mean())
    if unstd:
        for nm, g in (('l', gl), ('c', gc), ('u', gu)):
            Ts = np.maximum(0, (sq * (Db - g)).max(1))
            out['p_SPAunstd_' + nm] = float((Ts >= max(0, sq * db.max())).mean())
    if stepm:
        Z = sq * (Db - db) / om
        order = np.argsort(-t); r = out['rank']
        Zs = Z[:, order]
        suf = np.maximum.accumulate(Zs[:, ::-1], axis=1)[:, ::-1]   # max over ranks >= i
        ts = t[order]
        padj = np.array([(suf[:, i] >= ts[i]).mean() for i in range(r)])
        out['p_StepM_focus'] = float(np.maximum.accumulate(padj)[-1])
        out['p_StepM_top'] = float(padj[0])
    return out

def eff_tests(D):
    """Nyholt 與 Li-Ji：相關矩陣特徵值，經 n×n Gram 求非零特徵值，零特徵值補足到 N。"""
    D = np.asarray(D, float); n, N = D.shape
    Z = (D - D.mean(0)) / D.std(0, ddof=1)
    Gm = Z @ Z.T / (n - 1)
    lam = np.clip(np.linalg.eigvalsh(Gm), 0, None)
    lam = np.concatenate([lam, np.zeros(N - len(lam))]) if N > n else np.linalg.eigvalsh(np.corrcoef(D.T))
    lam = np.clip(lam, 0, None)
    ny = 1 + (N - 1) * (1 - np.var(lam, ddof=1) / N)
    lj = float(np.sum((lam >= 1).astype(float) + (lam - np.floor(lam))))
    return dict(N=N, nyholt=float(ny), liji=lj)

def dsr(x, srs, N):
    """Deflated Sharpe（Bailey-López de Prado 2014）。x: 焦點超額窗序列；srs: 族群各成員 SR；N: 有效試驗數。"""
    x = np.asarray(x, float); n = len(x)
    sr = x.mean() / x.std(ddof=1)
    g = 0.5772156649
    v = np.var(srs, ddof=1)
    sr0 = math.sqrt(v) * ((1 - g) * st.norm.ppf(1 - 1 / N) + g * st.norm.ppf(1 - 1 / (N * math.e)))
    sk = st.skew(x); ku = st.kurtosis(x, fisher=False)
    z = (sr - sr0) * math.sqrt(n - 1) / math.sqrt(1 - sk * sr + (ku - 1) / 4 * sr ** 2)
    return dict(sr=float(sr), sr0=float(sr0), dsr=float(st.norm.cdf(z)), p=float(1 - st.norm.cdf(z)))
