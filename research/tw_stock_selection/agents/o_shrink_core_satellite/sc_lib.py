#!/usr/bin/env python3
"""o_shrink_core_satellite 共用函式庫：族群讀取、HAC 標準誤、群集化、經驗貝氏先驗（G1 常態 REML、G2 Efron log-spline、
G3 NPMLE）、焦點事後分布、核心衛星配置模擬、PREREG 雜湊工具。

單位約定：族群與焦點的月（換股窗）超額一律以 pp／窗 表示（1.0 = 1%）。
窗格點沿用 o_main_spa_breaks/fam_common.GRID（AB_V4 訊號日，245 窗，2006-04-11～2026-10-12；只讀 import）。
"""
import sys, json, hashlib, math
from pathlib import Path
import numpy as np, pandas as pd
from scipy import optimize, stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

HERE = Path(__file__).resolve().parent
AG = HERE.parent
DATA = HERE / 'data'
SPA = AG / 'o_main_spa_breaks'
sys.path.insert(0, str(SPA))
import fam_common as FC      # noqa: E402（只讀 import）
import spa_lib as SL         # noqa: E402（只讀 import）
sys.path.pop(0)

MAIN_ID = 'o_v4_revenue_patch_official/navs_v4.parquet::AB_V4|normal'
B10_ID = 'o_v4_revenue_patch_official/navs_v4.parquet::AB_V4_B10|normal'
W_LONG = ('2006-04-01', '2026-10-12')
SEL = ('2006-04-01', '2019-04-30')
VAL = ('2019-05-01', '2023-12-31')
HOLD = ('2024-01-01', '2026-10-12')
RES_F = HERE / 'results.json'

# ---------------------------------------------------------------- 雜湊與 PREREG
def sha(f):
    return hashlib.sha256(Path(f).read_bytes()).hexdigest()

def canon_sha(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

def load_res():
    return json.loads(RES_F.read_text(encoding='utf-8')) if RES_F.exists() else {}

def save_key(key, obj, need_prereg=True):
    res = load_res()
    if need_prereg:
        assert 'PREREG' in res, 'PREREG 不存在'
    res[key] = obj
    RES_F.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=_js), encoding='utf-8')

def _js(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (pd.Timestamp,)): return str(o.date())
    return str(o)

def check_prereg():
    """正式計算前的斷言：PREREG 存在、內容未被改、所有登記檔案的 sha256 與登記相同。"""
    res = load_res()
    assert 'PREREG' in res and 'PREREG_sha256' in res, 'PREREG 不存在：先跑 prereg.py'
    assert canon_sha(res['PREREG']) == res['PREREG_sha256'], 'PREREG 被改過'
    for rel, h in res['PREREG']['hashes'].items():
        p = Path(rel) if rel.startswith('/') else HERE / rel
        assert sha(p) == h, f'{rel} 與 PREREG 登記的 sha256 不符'
    return res

# ---------------------------------------------------------------- 資料
def wmask(idx, a, b):
    return (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))

def load_family():
    """回傳 (X, members)：X＝F_long 成員對 0050 的窗超額（pp，缺值以 0 填補，與 SPA 題同規則）；members＝family_members 子表。"""
    D = pd.read_parquet(SPA / 'data' / 'family_excess.parquet')
    MM = pd.read_csv(SPA / 'data' / 'family_members.csv')
    fl = MM[MM.F_long].reset_index(drop=True)
    X = D[fl.member_id.tolist()].fillna(0.0) * 100.0
    return X, fl

def load_focus(path=None):
    """焦點窗報酬檔（data/focus_windows.parquet；總監整合時可換成 V6／B10R／校準後可執行版）。回傳 DataFrame（簡單報酬，小數）。"""
    p = Path(path) if path else DATA / 'focus_windows.parquet'
    F = pd.read_parquet(p)
    assert 'r0050' in F.columns
    return F

# ---------------------------------------------------------------- 統計
def hac_var_mean(x, L=6):
    """Newey-West（Bartlett，落後 L）下樣本平均數的變異數。x：1D 或 n×K。"""
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
    x = np.asarray(x, float)
    return float(x.mean() / x.std(ddof=1) * math.sqrt(len(x)))

def li_ji(X):
    lam = np.linalg.eigvalsh(np.corrcoef(np.asarray(X, float), rowvar=False))
    lam = np.abs(lam)
    return float(((lam >= 1).astype(float) + (lam - np.floor(lam))).sum())

def cluster_medoids(X, k, method='average'):
    """相關係數群集化：距離 1−ρ，average linkage，切成 k 群；每群取 medoid（與群內其他成員平均相關最高者）。
    回傳 (labels(1..k), medoid_cols(list), corr)。"""
    C = np.corrcoef(np.asarray(X, float), rowvar=False)
    C = np.nan_to_num(C, nan=0.0)
    Dm = np.clip(1 - C, 0, 2); np.fill_diagonal(Dm, 0.0)
    Dm = (Dm + Dm.T) / 2
    Z = linkage(squareform(Dm, checks=False), method=method)
    lab = fcluster(Z, t=k, criterion='maxclust')
    cols = list(X.columns)
    med = []
    for c in sorted(set(lab)):
        ii = np.where(lab == c)[0]
        if len(ii) == 1: med.append(cols[ii[0]]); continue
        sub = C[np.ix_(ii, ii)]
        med.append(cols[ii[np.argmax(sub.mean(axis=1))]])
    return lab, med, C

# ---------------------------------------------------------------- 誤差分解（相關調整）
def decompose(Xp, focus=None, L=6, mode='decomp'):
    """Xp：先驗族群 n×K（pp）；focus：dict name→1D 序列（pp）。
    mode='naive'：各成員誤差變異＝自身 HAC 變異，v_P＝0（假設誤差獨立）。
    mode='decomp'：誤差拆成族群共同部分 ē_P（族群平均的誤差）與個別部分 u_k＝ē_k−ē_P；
        去卷積用 u_k 的 HAC 變異，共同部分的變異 v_P＝HAC Var(族群平均序列的平均)，事後再加回（平移不確定，均值不變）。
    回傳 dict：xhat(K)、s2(K)（去卷積用）、vP、focus_xhat、focus_s2、Keff（＝平均個別變異÷v_P 的等價有效數）。"""
    A = np.asarray(Xp, float)
    xhat = A.mean(axis=0)
    full = hac_var_mean(A, L)
    fx, fs2 = {}, {}
    if mode == 'naive':
        s2 = full; vP = 0.0
        for k, v in (focus or {}).items():
            v = np.asarray(v, float); fx[k] = float(v.mean()); fs2[k] = float(hac_var_mean(v, L))
    else:
        m = A.mean(axis=1)
        s2 = hac_var_mean(A - m[:, None], L)
        vP = float(hac_var_mean(m, L))
        for k, v in (focus or {}).items():
            v = np.asarray(v, float); fx[k] = float(v.mean()); fs2[k] = float(hac_var_mean(v - m, L))
    keff = float(np.mean(full) / vP) if vP > 0 else float('inf')
    return {'xhat': xhat, 's2': np.asarray(s2, float), 'vP': vP, 'focus_xhat': fx, 'focus_s2': fs2, 'Keff': keff,
            'full_s2': np.asarray(full, float)}

# ---------------------------------------------------------------- 先驗估計
GRID_LO, GRID_HI, GRID_STEP = -8.0, 4.0, 0.02   # 原登記 [−6,6]；W_long 族群 α̂ 最小 −6.04 → 凍結前改 [−8,4]（寬度不變），v3 以新格點重驗
def theta_grid():
    return np.round(np.arange(GRID_LO, GRID_HI + GRID_STEP / 2, GRID_STEP), 10)

G2_C0 = 0.1         # G2 懲罰係數（v1／v2＝1，deconvolveR 預設；選定 v3＝0.1，凍結）
G2_KNOTS = 'grid'   # 'grid'＝內部節點取格點等分位（deconvolveR 作法，v1／v2）；'data'＝內部節點取 α̂ 的等分位（v3）

def ns_basis(x, df=5, data=None):
    """自然三次樣條基底（ESL 5.2.1 截斷冪表示），df 欄、不含截距；邊界節點為 x（格點）的端點；
    內部節點：data 為 None 時取 x 的等分位，否則取 data（α̂）的等分位。之後中心化並以 SVD 正交化、各欄單位長度（同 deconvolveR）。"""
    x = np.asarray(x, float)
    K = df + 1                       # 總節點數（含兩端）→ 基底 1, x, N3..N_K 共 K 個；去掉截距剩 df 個
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
    Q = U / np.sqrt((U * U).sum(axis=0))
    return Q

def lik_matrix(xhat, s2, th):
    s = np.sqrt(s2)[:, None]
    return np.exp(-0.5 * ((xhat[:, None] - th[None, :]) / s) ** 2) / s

def fit_g1(xhat, s2):
    """常態先驗 N(μ, τ²)，REML。回傳 dict(mu, tau2, var_mu)。"""
    xhat = np.asarray(xhat, float); s2 = np.asarray(s2, float)
    def negl(lt):
        t2 = math.exp(lt); w = 1 / (t2 + s2); mu = (w * xhat).sum() / w.sum()
        return 0.5 * (np.log(t2 + s2).sum() + math.log(w.sum()) + (w * (xhat - mu) ** 2).sum())
    r = optimize.minimize_scalar(negl, bounds=(math.log(1e-8), math.log(100.0)), method='bounded', options={'xatol': 1e-8})
    t2 = math.exp(r.x)
    if t2 < 1e-6: t2 = 0.0
    w = 1 / (t2 + s2); mu = float((w * xhat).sum() / w.sum())
    return {'mu': mu, 'tau2': float(t2), 'var_mu': float(1 / w.sum())}

def fit_g2(xhat, s2, df=5, c0=None, th=None, knots=None):
    """Efron（2016）log-spline g-modeling：g(θ_j)∝exp(Q_j β)，最大化 Σlog f_k − c0‖β‖。回傳 dict(th, g, beta, loglik)。"""
    th = theta_grid() if th is None else th
    knots = G2_KNOTS if knots is None else knots
    c0 = G2_C0 if c0 is None else c0
    Q = ns_basis(th, df, data=(np.asarray(xhat, float) if knots == 'data' else None))
    P = lik_matrix(np.asarray(xhat, float), np.asarray(s2, float), th)
    def fg(b):
        a = Q @ b; a = a - a.max(); g = np.exp(a); g = g / g.sum()
        f = P @ g + 1e-300
        ll = np.log(f).sum()
        nb = math.sqrt((b * b).sum()) + 1e-12
        # 梯度：Σ_k Σ_j (P_kj g_j / f_k)(Q_j − Q̄)
        Wk = (P * g[None, :]) / f[:, None]          # K×m
        wj = Wk.sum(axis=0)                           # m
        Qbar = g @ Q
        grad_ll = wj @ Q - wj.sum() * Qbar
        return -(ll - c0 * nb), -(grad_ll - c0 * b / nb)
    b0 = np.zeros(Q.shape[1])
    r = optimize.minimize(fg, b0, jac=True, method='BFGS', options={'gtol': 1e-7, 'maxiter': 5000})
    a = Q @ r.x; a = a - a.max(); g = np.exp(a); g = g / g.sum()
    return {'th': th, 'g': g, 'beta': r.x, 'loglik': float(np.log(P @ g + 1e-300).sum()), 'converged': bool(r.success)}

def fit_g3(xhat, s2, th=None, iters=6000, tol=1e-10):
    """非參數 NPMLE（Kiefer-Wolfowitz；Koenker-Mizera 的格點版，以 EM 求解）。"""
    th = theta_grid() if th is None else th
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
    return {'th': th, 'g': g, 'loglik': float(np.log(P @ g + 1e-300).sum()), 'iters': it + 1}

def fit_prior(model, xhat, s2):
    if model == 'G1': return fit_g1(xhat, s2)
    if model == 'G2': return fit_g2(xhat, s2)
    if model == 'G3': return fit_g3(xhat, s2)
    raise ValueError(model)

# ---------------------------------------------------------------- 事後分布
def posterior(prior, model, x, s2, vP, vmu_extra=0.0):
    """單一成員的事後分布（α＝β−共同誤差；β 的事後分布由先驗與個別誤差決定，再與 N(0, vP) 卷積）。
    回傳 dict(mean, sd, p_pos, q10, q90, q05, q95, grid, dens)（grid 上的離散密度，供抽樣）。"""
    th = theta_grid()
    if model == 'G1':
        mu, t2, vm = prior['mu'], prior['tau2'], prior['var_mu']
        B = t2 / (t2 + s2) if (t2 + s2) > 0 else 0.0
        m = mu + B * (x - mu)
        v = t2 * s2 / (t2 + s2) + (1 - B) ** 2 * vm + vP + vmu_extra
        dens = stats.norm.pdf(th, m, math.sqrt(v)); dens = dens / dens.sum()
    else:
        g = prior['g']
        lp = np.log(g + 1e-300) - 0.5 * (x - th) ** 2 / s2
        pb = np.exp(lp - lp.max()); pb = pb / pb.sum()
        if vP + vmu_extra > 0:
            sd = math.sqrt(vP + vmu_extra)
            # 與 N(0, sd²) 卷積（格點上）
            ker_x = np.arange(-int(6 * sd / GRID_STEP) - 1, int(6 * sd / GRID_STEP) + 2) * GRID_STEP
            ker = stats.norm.pdf(ker_x, 0, sd); ker = ker / ker.sum()
            dens = np.convolve(pb, ker, mode='same')
        else:
            dens = pb
        dens = dens / dens.sum()
    mean = float((th * dens).sum()); sd = float(math.sqrt(max(((th - mean) ** 2 * dens).sum(), 0)))
    cdf = np.cumsum(dens)
    q = lambda p: float(np.interp(p, cdf, th))
    p_pos = float(dens[th > 0].sum() + 0.5 * dens[th == 0].sum())
    return {'mean': mean, 'sd': sd, 'p_pos': p_pos, 'q05': q(0.05), 'q10': q(0.10), 'q50': q(0.5), 'q90': q(0.90), 'q95': q(0.95),
            'dens': dens}

def post_all(prior, model, xhat, s2, vP):
    """族群每個成員的事後平均與 80% 區間（向量化；驗證用）。"""
    th = theta_grid()
    xhat = np.asarray(xhat, float); s2 = np.asarray(s2, float)
    if model == 'G1':
        mu, t2, vm = prior['mu'], prior['tau2'], prior['var_mu']
        B = t2 / (t2 + s2)
        m = mu + B * (xhat - mu)
        v = t2 * s2 / (t2 + s2) + (1 - B) ** 2 * vm + vP
        z = stats.norm.ppf(0.9)
        return m, m - z * np.sqrt(v), m + z * np.sqrt(v)
    g = prior['g']
    lp = np.log(g + 1e-300)[None, :] - 0.5 * (xhat[:, None] - th[None, :]) ** 2 / s2[:, None]
    pb = np.exp(lp - lp.max(axis=1, keepdims=True)); pb = pb / pb.sum(axis=1, keepdims=True)
    if vP > 0:
        sd = math.sqrt(vP)
        ker_x = np.arange(-int(6 * sd / GRID_STEP) - 1, int(6 * sd / GRID_STEP) + 2) * GRID_STEP
        ker = stats.norm.pdf(ker_x, 0, sd); ker = ker / ker.sum()
        from scipy.signal import fftconvolve
        pb = fftconvolve(pb, ker[None, :], mode='same', axes=1)
        pb = np.maximum(pb, 0); pb = pb / pb.sum(axis=1, keepdims=True)
    m = pb @ th
    cdf = np.cumsum(pb, axis=1)
    lo = th[np.minimum((cdf < 0.10).sum(axis=1), len(th) - 1)]
    hi = th[np.minimum((cdf < 0.90).sum(axis=1), len(th) - 1)]
    return m, lo, hi

def sample_post(dens, n, rng):
    th = theta_grid()
    j = rng.choice(len(th), size=n, p=dens)
    return th[j] + rng.uniform(-GRID_STEP / 2, GRID_STEP / 2, n)

# ---------------------------------------------------------------- 平穩區塊 bootstrap 索引
def sb_index(n, H, L, B, seed):
    """平穩 bootstrap（幾何區塊，平均長度 L，環狀接續），回傳 B×H 的列索引。"""
    rng = np.random.default_rng(seed)
    idx = np.empty((B, H), np.int32)
    idx[:, 0] = rng.integers(0, n, B)
    p = 1.0 / L
    for t in range(1, H):
        new = rng.random(B) < p
        idx[:, t] = np.where(new, rng.integers(0, n, B), (idx[:, t - 1] + 1) % n)
    return idx

# ---------------------------------------------------------------- 配置模擬
# 再平衡成本（單邊，小數）：主策略賣出＝手續費 0.1425%×0.6＋滑價 0.1%＋證交稅 0.3%；買進＝手續費＋滑價；
# 0050 賣出＝手續費＋ETF 證交稅 0.1%＋滑價 0.05%；買進＝手續費＋滑價 0.05%。
FEE = 0.001425 * 0.6
C_M_SELL, C_M_BUY = FEE + 0.001 + 0.003, FEE + 0.001
C_E_SELL, C_E_BUY = FEE + 0.001 + 0.0005, FEE + 0.0005

def simulate(r50p, ep, alpha, x, rebal='M', want_paths=False):
    """r50p：B×H 的 0050 窗報酬路徑；ep：B×H 的主策略超額殘差路徑（已去均值）；alpha：長度 B 的每路徑 α（小數／窗）。
    主策略窗報酬 r_m＝r50＋α＋e。x：主策略權重。rebal：'M' 每窗再平衡、'A' 每 12 窗。
    回傳 dict：Wp、W50（期末財富）、mdd（組合最大回撤）、rel_mdd（財富比最大落後）、roll5_lose（任一 60 窗滾動輸 0050）、
    under60（財富比連續 ≥60 窗 < 1）、logR（期末 log 財富比）。"""
    B, H = r50p.shape
    rm = r50p + alpha[:, None] + ep
    rm = np.maximum(rm, -0.99)
    wm = np.full(B, x); we = np.full(B, 1 - x)     # 以金額計（起始 1）
    Wp = np.ones(B); W50 = np.ones(B)
    peak = np.ones(B); mdd = np.zeros(B)
    Rpeak = np.ones(B); rmdd = np.zeros(B)
    Rhist = np.empty((B, H + 1)); Rhist[:, 0] = 1.0
    under = np.zeros(B, np.int32); under_max = np.zeros(B, np.int32)
    for t in range(H):
        wm = wm * (1 + rm[:, t]); we = we * (1 + r50p[:, t])
        Wp = wm + we
        if (rebal == 'M') or (rebal == 'A' and (t + 1) % 12 == 0):
            if 0 < x < 1:
                tgt = x * Wp
                d = wm - tgt                          # >0：賣主策略、買 0050
                cost = np.where(d > 0, d * (C_M_SELL + C_E_BUY), -d * (C_E_SELL + C_M_BUY))
                Wp = Wp - cost
                wm = x * Wp; we = (1 - x) * Wp
        W50 = W50 * (1 + r50p[:, t])
        peak = np.maximum(peak, Wp); mdd = np.minimum(mdd, Wp / peak - 1)
        R = Wp / W50
        Rhist[:, t + 1] = R
        Rpeak = np.maximum(Rpeak, R); rmdd = np.minimum(rmdd, R / Rpeak - 1)
        under = np.where(R < 1, under + 1, 0); under_max = np.maximum(under_max, under)
    roll = (Rhist[:, 60:] / Rhist[:, :-60]) < 1 if H >= 60 else np.zeros((B, 1), bool)
    out = {'Wp': Wp, 'W50': W50, 'mdd': mdd, 'rel_mdd': rmdd, 'roll5_lose': roll.any(axis=1),
           'under60': under_max >= 60, 'logR': np.log(Wp / W50)}
    return out

def summarize_sim(o, H):
    R = o['Wp'] / o['W50']
    return {'P_win': float((R > 1).mean()),
            'ratio_p10': float(np.percentile(R, 10)), 'ratio_p50': float(np.percentile(R, 50)), 'ratio_p90': float(np.percentile(R, 90)),
            'P_ratio_lt_0.75': float((R < 0.75).mean()), 'P_ratio_lt_0.5': float((R < 0.5).mean()),
            'mdd_p50': float(np.percentile(o['mdd'], 50)), 'mdd_p90_severe': float(np.percentile(o['mdd'], 10)),
            'rel_mdd_p50': float(np.percentile(o['rel_mdd'], 50)), 'rel_mdd_p90_severe': float(np.percentile(o['rel_mdd'], 10)),
            'P_roll5y_lose': float(o['roll5_lose'].mean()), 'P_under_ge60w': float(o['under60'].mean()),
            'Elog_ratio_per_year': float(o['logR'].mean() / (H / 12.0)),
            'cagr_p50_pp': float((np.percentile(o['Wp'], 50) ** (12.0 / H) - 1) * 100)}
