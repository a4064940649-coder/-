#!/usr/bin/env python3
"""o_tier_shrink_sequential 共用函式庫（只讀 raw/ 與 data/）。

內容
- 格點、窗報酬、0050（與 o_main_spa_breaks/fam_common 相同定義；格點＝AB_V4 訊號日＋2026-10-12）。
- o6 經驗貝氏工具：HAC、誤差拆解、G1／G2（Efron log-spline）／G3（NPMLE）、單成員事後分布、平穩 bootstrap、配置模擬。
  這些函式逐字複製自 agents/o_shrink_core_satellite/sc_lib.py（raw/o6shrink/sc_lib.py，sha256 見 raw/manifest.json），
  只把格點與懲罰係數改成參數，方便做 c0／格點敏感度；預設值與 o6 凍結值相同（[−8,4]、0.02、df 5、c0 0.1、grid 節點）。
- 本題新增：群內挑選偏誤 bootstrap、階層事後分布（群間 G2 × 群內常態）、焦點格事後密度的組合（j 隨機效果、成本平移）、
  e-process（混合 SPRT）、PREREG 工具。
單位：超額一律 pp／窗（1.0＝1%）。
"""
import json, hashlib, math, datetime
from pathlib import Path
import numpy as np, pandas as pd
from scipy import optimize, stats
from scipy.signal import fftconvolve

HERE = Path(__file__).resolve().parent
RAW, DATA = HERE / 'raw', HERE / 'data'
RES_F = HERE / 'results.json'
END_DAY = pd.Timestamp('2026-10-12')
MAIN_ID = 'o_v4_revenue_patch_official/navs_v4.parquet::AB_V4|normal'
B10_ID = 'o_v4_revenue_patch_official/navs_v4.parquet::AB_V4_B10|normal'
SEL = ('2006-04-01', '2019-04-30')
VAL = ('2019-05-01', '2023-12-31')
HOLD = ('2024-01-01', '2026-10-12')

# ---------------------------------------------------------------- 雜湊與 results.json
def sha(f):
    return hashlib.sha256(Path(f).read_bytes()).hexdigest()

def canon_sha(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, default=_js).encode()).hexdigest()

def _js(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (pd.Timestamp,)): return str(o.date())
    if isinstance(o, (np.bool_,)): return bool(o)
    return str(o)

def now_utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def load_res(p=RES_F):
    return json.loads(Path(p).read_text(encoding='utf-8')) if Path(p).exists() else {}

def save_key(key, obj, p=RES_F, need=None):
    res = load_res(p)
    for k in (need or []):
        assert k in res, f'{k} 不存在：順序錯誤'
    res[key] = obj
    Path(p).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=_js), encoding='utf-8')

def check_block(key, hashes_key='hashes', files=True):
    """斷言 results.json[key] 存在、內容雜湊與登記相同；files＝True 時再斷言登記檔案 sha256 與現在相同
    （設計登記 PREREG_DESIGN／AMEND 的檔案雜湊只是當時的快照紀錄，之後程式會依登記內容實作，故以 files＝False 檢查）。"""
    res = load_res()
    assert key in res and key + '_sha256' in res, f'{key} 不存在'
    assert canon_sha(res[key]) == res[key + '_sha256'], f'{key} 被改過'
    for rel, h in (res[key].get(hashes_key, {}).items() if files else []):
        p = Path(rel) if rel.startswith('/') else HERE / rel
        assert sha(p) == h, f'{rel} 與 {key} 登記的 sha256 不符'
    return res

# ---------------------------------------------------------------- 格點與序列
def grid():
    p = pd.read_csv(RAW / 'spa' / 'plan_V4.csv')
    return pd.DatetimeIndex(list(pd.to_datetime(p['signal'])) + [END_DAY])

GRID = grid()
WIN = GRID[:-1]

def sample(nav, g=GRID):
    s = nav.dropna()
    if len(s) == 0: return pd.Series(np.nan, index=g)
    s = s[~s.index.duplicated(keep='last')].sort_index()
    v = s.reindex(s.index.union(g)).ffill().reindex(g)
    v[g < s.index[0]] = np.nan
    v[g > s.index[-1] + pd.Timedelta(days=7)] = np.nan
    return v

def win_ret(nav, g=GRID):
    v = sample(nav, g).to_numpy()
    return pd.Series(v[1:] / v[:-1] - 1, index=g[:-1])

def e50_nav():
    return pd.read_csv(RAW / 'navs' / 'etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()

def wmask(idx, a, b):
    return (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))

def load_family():
    D = pd.read_parquet(RAW / 'spa' / 'family_excess.parquet')
    MM = pd.read_csv(RAW / 'spa' / 'family_members.csv')
    fl = MM[MM.F_long].reset_index(drop=True)
    X = D[fl.member_id.tolist()].fillna(0.0) * 100.0
    return X, fl

def load_clusters():
    return pd.read_csv(RAW / 'o6shrink' / 'clusters_k161.csv')

# ---------------------------------------------------------------- 統計（複製自 sc_lib）
def hac_var_mean(x, L=6):
    x = np.asarray(x, float)
    one = x.ndim == 1
    if one: x = x[:, None]
    n = x.shape[0]; e = x - x.mean(axis=0)
    s = (e * e).sum(axis=0) / n
    for i in range(1, L + 1):
        s = s + 2 * (1 - i / (L + 1)) * (e[i:] * e[:-i]).sum(axis=0) / n
    v = np.maximum(s, 1e-12) / n
    return v[0] if one else v

def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(x.mean() / x.std(ddof=1) * math.sqrt(len(x)))

def li_ji(X):
    lam = np.abs(np.linalg.eigvalsh(np.corrcoef(np.asarray(X, float), rowvar=False)))
    return float(((lam >= 1).astype(float) + (lam - np.floor(lam))).sum())

def decompose(Xp, focus=None, L=6, mode='decomp'):
    A = np.asarray(Xp, float)
    xhat = A.mean(axis=0)
    full = hac_var_mean(A, L)
    fx, fs2 = {}, {}
    if mode == 'naive':
        s2 = full; vP = 0.0
        for k, v in (focus or {}).items():
            v = np.asarray(v, float); fx[k] = float(v.mean()); fs2[k] = float(hac_var_mean(v, L))
        m = None
    else:
        m = A.mean(axis=1)
        s2 = hac_var_mean(A - m[:, None], L)
        vP = float(hac_var_mean(m, L))
        for k, v in (focus or {}).items():
            v = np.asarray(v, float); fx[k] = float(v.mean()); fs2[k] = float(hac_var_mean(v - m, L))
    keff = float(np.mean(full) / vP) if vP > 0 else float('inf')
    return {'xhat': xhat, 's2': np.asarray(s2, float), 'vP': vP, 'focus_xhat': fx, 'focus_s2': fs2, 'Keff': keff,
            'full_s2': np.asarray(full, float), 'mean_series': m}

# ---------------------------------------------------------------- 先驗（複製自 sc_lib；格點、c0 參數化）
class Grid:
    def __init__(self, lo=-8.0, hi=4.0, step=0.02):
        self.lo, self.hi, self.step = lo, hi, step
        self.th = np.round(np.arange(lo, hi + step / 2, step), 10)

G_DEFAULT = Grid()

def ns_basis(x, df=5, data=None):
    x = np.asarray(x, float)
    K = df + 1
    if data is None:
        kn = np.quantile(x, np.linspace(0, 1, K))
    else:
        kn = np.r_[x.min(), np.quantile(np.asarray(data, float), np.linspace(0, 1, K)[1:-1]), x.max()]
    def d(k):
        a = np.maximum(x - kn[k], 0) ** 3 - np.maximum(x - kn[K - 1], 0) ** 3
        return a / (kn[K - 1] - kn[k])
    cols = [x]
    for k in range(K - 2):
        cols.append(d(k) - d(K - 2))
    Q = np.column_stack(cols)
    Q = Q - Q.mean(axis=0)
    U, S, Vt = np.linalg.svd(Q, full_matrices=False)
    return U / np.sqrt((U * U).sum(axis=0))

def lik_matrix(xhat, s2, th):
    s = np.sqrt(s2)[:, None]
    return np.exp(-0.5 * ((xhat[:, None] - th[None, :]) / s) ** 2) / s

def fit_g1(xhat, s2):
    xhat = np.asarray(xhat, float); s2 = np.asarray(s2, float)
    def negl(lt):
        t2 = math.exp(lt); w = 1 / (t2 + s2); mu = (w * xhat).sum() / w.sum()
        return 0.5 * (np.log(t2 + s2).sum() + math.log(w.sum()) + (w * (xhat - mu) ** 2).sum())
    r = optimize.minimize_scalar(negl, bounds=(math.log(1e-8), math.log(100.0)), method='bounded', options={'xatol': 1e-8})
    t2 = math.exp(r.x)
    if t2 < 1e-6: t2 = 0.0
    w = 1 / (t2 + s2); mu = float((w * xhat).sum() / w.sum())
    return {'mu': mu, 'tau2': float(t2), 'var_mu': float(1 / w.sum())}

def fit_g2(xhat, s2, df=5, c0=0.1, G=G_DEFAULT, knots='grid'):
    th = G.th
    Q = ns_basis(th, df, data=(np.asarray(xhat, float) if knots == 'data' else None))
    P = lik_matrix(np.asarray(xhat, float), np.asarray(s2, float), th)
    def fg(b):
        a = Q @ b; a = a - a.max(); g = np.exp(a); g = g / g.sum()
        f = P @ g + 1e-300
        ll = np.log(f).sum()
        nb = math.sqrt((b * b).sum()) + 1e-12
        Wk = (P * g[None, :]) / f[:, None]
        wj = Wk.sum(axis=0)
        Qbar = g @ Q
        grad_ll = wj @ Q - wj.sum() * Qbar
        return -(ll - c0 * nb), -(grad_ll - c0 * b / nb)
    r = optimize.minimize(fg, np.zeros(Q.shape[1]), jac=True, method='BFGS', options={'gtol': 1e-7, 'maxiter': 5000})
    a = Q @ r.x; a = a - a.max(); g = np.exp(a); g = g / g.sum()
    return {'th': th, 'g': g, 'beta': r.x, 'loglik': float(np.log(P @ g + 1e-300).sum()), 'converged': bool(r.success), 'G': G}

def fit_g3(xhat, s2, G=G_DEFAULT, iters=6000, tol=1e-10):
    th = G.th
    P = lik_matrix(np.asarray(xhat, float), np.asarray(s2, float), th)
    g = np.full(len(th), 1.0 / len(th)); llo = -np.inf
    K = P.shape[0]
    for it in range(iters):
        f = P @ g + 1e-300
        g = g * (P.T @ (1 / f)) / K
        if it % 200 == 0:
            ll = np.log(f).sum()
            if abs(ll - llo) < tol * max(1, abs(ll)): break
            llo = ll
    g = g / g.sum()
    return {'th': th, 'g': g, 'loglik': float(np.log(P @ g + 1e-300).sum()), 'iters': it + 1, 'G': G}

def gauss_kernel(sd, step):
    kx = np.arange(-int(6 * sd / step) - 1, int(6 * sd / step) + 2) * step
    k = stats.norm.pdf(kx, 0, sd)
    return k / k.sum()

def convolve_dens(dens, sd, step):
    if sd <= 0: return dens
    k = gauss_kernel(sd, step)
    full = np.convolve(dens, k, mode='full')          # 核長可能超過格點（大 sd）：取 full 再置中裁切（核長為奇數、對稱）
    off = (len(k) - 1) // 2
    d = full[off:off + len(dens)]
    d = np.maximum(d, 0)
    return d / d.sum()

def posterior_g(g, th, x, s2, vP, step):
    """sc_lib.posterior 的 G2／G3 分支（格點離散先驗）：β 的事後 × N(0, vP) 卷積。回傳格點密度。"""
    lp = np.log(g + 1e-300) - 0.5 * (x - th) ** 2 / s2
    pb = np.exp(lp - lp.max()); pb = pb / pb.sum()
    return convolve_dens(pb, math.sqrt(vP), step) if vP > 0 else pb

def dens_summary(dens, th):
    dens = np.asarray(dens, float); dens = dens / dens.sum()
    mean = float((th * dens).sum()); sd = float(math.sqrt(max(((th - mean) ** 2 * dens).sum(), 0)))
    cdf = np.cumsum(dens)
    q = lambda p: float(np.interp(p, cdf, th))
    p_pos = float(dens[th > 0].sum() + 0.5 * dens[th == 0].sum())
    return {'mean': mean, 'sd': sd, 'p_pos': p_pos, 'q05': q(0.05), 'q10': q(0.10), 'q50': q(0.5), 'q90': q(0.90), 'q95': q(0.95)}

def shift_dens(dens, th, delta):
    """把格點密度平移 delta（pp），線性內插；超出格點的質量堆在端點（只用於小幅平移）。"""
    out = np.interp(th - delta, th, dens, left=0.0, right=0.0)
    s = out.sum()
    return out / s if s > 0 else dens

def sample_dens(dens, th, n, rng, step):
    j = rng.choice(len(th), size=n, p=dens / dens.sum())
    return th[j] + rng.uniform(-step / 2, step / 2, n)

# ---------------------------------------------------------------- bootstrap 與配置模擬（複製自 sc_lib）
def sb_index(n, H, L, B, seed):
    rng = np.random.default_rng(seed)
    idx = np.empty((B, H), np.int32)
    idx[:, 0] = rng.integers(0, n, B)
    p = 1.0 / L
    for t in range(1, H):
        new = rng.random(B) < p
        idx[:, t] = np.where(new, rng.integers(0, n, B), (idx[:, t - 1] + 1) % n)
    return idx

FEE = 0.001425 * 0.6
C_M_SELL, C_M_BUY = FEE + 0.001 + 0.003, FEE + 0.001
C_E_SELL, C_E_BUY = FEE + 0.001 + 0.0005, FEE + 0.0005

def simulate(r50p, ep, alpha, x, rebal='M'):
    B, H = r50p.shape
    rm = np.maximum(r50p + alpha[:, None] + ep, -0.99)
    wm = np.full(B, x); we = np.full(B, 1 - x)
    Wp = np.ones(B); W50 = np.ones(B)
    peak = np.ones(B); mdd = np.zeros(B)
    for t in range(H):
        wm = wm * (1 + rm[:, t]); we = we * (1 + r50p[:, t])
        Wp = wm + we
        if (rebal == 'M') or (rebal == 'A' and (t + 1) % 12 == 0):
            if 0 < x < 1:
                tgt = x * Wp
                d = wm - tgt
                cost = np.where(d > 0, d * (C_M_SELL + C_E_BUY), -d * (C_E_SELL + C_M_BUY))
                Wp = Wp - cost
                wm = x * Wp; we = (1 - x) * Wp
        W50 = W50 * (1 + r50p[:, t])
        peak = np.maximum(peak, Wp); mdd = np.minimum(mdd, Wp / peak - 1)
    return {'Wp': Wp, 'W50': W50, 'mdd': mdd, 'logR': np.log(Wp / W50), 'logW': np.log(Wp)}

# ---------------------------------------------------------------- 本題新增：群內挑選偏誤
def max_noise_boot(Xc, B=2000, L=6, seed=0, center='member'):
    """群內最大值挑選偏誤：Xc＝群內成員的窗超額（n×m，pp）。每個成員先扣掉自己的樣本平均（虛無：每個成員的真 α 就是自己的、
    重抽只帶來雜訊），以平穩 bootstrap（平均區塊 L）同列重抽（保留成員間相關），每次取 m 個成員重抽平均的最大值。
    回傳偏誤樣本（長度 B，pp）：max_k ē*_k。期望值＝從 m 個相關變體挑最大所帶來的平均高估。"""
    A = np.asarray(Xc, float)
    n, m = A.shape
    E = A - A.mean(axis=0, keepdims=True)
    idx = sb_index(n, n, L, B, seed)
    out = np.empty(B)
    for b in range(B):
        out[b] = E[idx[b]].mean(axis=0).max()
    return out

def max_noise_boot_diff(Xc, ref, B=2000, L=6, seed=0):
    """同上，但對群內代表（medoid）的差序列：d_k＝x_k − x_ref。用於群內偏離的挑選偏誤。"""
    A = np.asarray(Xc, float) - np.asarray(ref, float)[:, None]
    return max_noise_boot(A, B, L, seed)

# ---------------------------------------------------------------- 本題新增：e-process（混合 SPRT，常態、已知 σ）
def log_lr_grid(xs, th, theta0, sigma):
    """Σ_t log[φ(x_t; θ, σ)/φ(x_t; θ0, σ)]，對每個 θ（格點）。xs：前瞻月超額（pp）。"""
    xs = np.asarray(xs, float)
    n = len(xs); S = xs.sum()
    return ((th - theta0) * S - 0.5 * n * (th ** 2 - theta0 ** 2)) / sigma ** 2

def e_mixture(xs, th, w, theta0, sigma):
    """混合 e 值：∫ LR(θ vs θ0) dπ(θ)，π 為格點權重 w（已正規化，只放在對立假設區）。"""
    if len(xs) == 0: return 1.0
    ll = log_lr_grid(xs, th, theta0, sigma)
    m = ll.max()
    return float(np.exp(m) * (w * np.exp(ll - m)).sum())

def update_post(prior, th, xs, sigma):
    """以常態概似（已知 σ）更新格點先驗。"""
    if len(xs) == 0: return prior / prior.sum()
    xs = np.asarray(xs, float)
    ll = -0.5 * ((xs[:, None] - th[None, :]) ** 2).sum(axis=0) / sigma ** 2
    p = np.log(prior + 1e-300) + ll
    p = np.exp(p - p.max())
    return p / p.sum()
