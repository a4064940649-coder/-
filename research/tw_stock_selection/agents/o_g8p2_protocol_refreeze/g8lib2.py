#!/usr/bin/env python3
"""o_g8p2_protocol_refreeze 共用函式庫（G8P-2：主策略 G8 前瞻序貫協定修正並重凍結）。

不產生任何選股訊號、不讀任何前瞻報酬。只讀：
  - G8P-1 凍結檔 o_g8_protocol_fin_erratum/G8_PROTOCOL.json（sha256 799f8b84…113fa 斷言）：σ_d、c、2019-05～2026-09 去均值殘差。
  - 歷史可執行線（探索與 in-sample 規則示範）：o_exec_engine_rewrite_v7x/out/navs_oos_R4.parquet（V7X K5T GROW 100 萬，一般／壓力）
    與 o_forward_sequential_v2/fslib.py（只 import 窗、錨定月與 cell_jmatrix）。

符號（全部 pp／月）：
  x_n：第 n 個有效錨定月的決策線月超額。決策線＝逐 J 窗 w：min(excess_pp_w − adj_w, excess_stress_pp_w)，再取有效 J 窗中位數（adj＝成本校準調整）。
  S_n＝Σ_{t≤n} x_t；s_n＝前 n 個有效月樣本 sd（ddof＝1，去樣本均值）。
  σ_d＝6.141373712658076（G8P-1／o9 設計 sd，HAC L6）；c＝1.0596378779547428（G8P-1 相依膨脹係數）。
版本（統計量；全部「升級檢視 n∈{12,24}、OBF 形 z_n ≥ b·√(24/n)；下修檢視 n∈{12,18,24}（之後每 6 個月）z_n ≤ −c_d；先觸發者吸收」）：
  G8P1：z_n＝S_n／(σ_eff√n)，σ_eff＝c·σ_d；檢視點 s_n＞1.5σ_d 時一次性改成 max(c·σ_d, s_n)（G8P-1 R-FIX 原樣，決策線＝一般成本線）。
  V1  ：z_n＝S_n／(max(c·σ_d, s_n)·√n)，每個檢視點重算（無空窗）。
  V2  ：t_n＝S_n／(√n·ŝ_HAC6,n)（Bartlett L＝6、去均值、γ_k 除以 n），z̃_n＝Φ⁻¹(F_{t,n−1}(t_n))。
  V3  ：V1，但序貫統計量用 x_n − δ（δ＝0.20）；字面 G8 仍用 x_n。
  V4  ：只有字面 G8＋V1 下修模組（無升級路徑）。
字面 G8：S_12(x) > 0；未過＝前瞻失敗（吸收，升級路徑關閉）。
"""
import sys, json, math, hashlib, datetime
from pathlib import Path
import numpy as np
from scipy import stats as st

HERE = Path(__file__).resolve().parent
AG = HERE.parent
RES_F = HERE / 'results.json'
SCR = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/o15_g8p2')
SCR.mkdir(parents=True, exist_ok=True)
G8P1_DIR = AG / 'o_g8_protocol_fin_erratum'
G8P1_SHA = '799f8b845b69f4df6f18a6e9cc901aa804fcd0d5dee393ef2bd97f7449b113fa'

SIGMA_D = 6.141373712658076
C_INFL = 1.0596378779547428
B_G8P1 = 1.6787025976148637
CD_G8P1 = 0.8946228207485066
REG_M = np.array([0.7, 1.4]); REG_P_HL, REG_P_LH = 0.06, 0.0319
SIG_GRID = [SIGMA_D, 6.51, 7.0, 7.5, 8.0, 8.4, 9.0, 9.5, 10.4]       # 第一格＝σ_d（標示 6.14）
SIG_LABEL = ['6.14', '6.51', '7.0', '7.5', '8.0', '8.4', '9.0', '9.5', '10.4']
S_GRID = [0.0, 0.2, 0.4, 0.577, 0.78]
DGP_PRIMARY = ['D1', 'D2', 'D3']
DELTA_V3 = 0.20
LOOK_UP = (12, 24)
LOOK_DN = (12, 18, 24)


# ---------------------------------------------------------------- 雜項
def _js(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, np.bool_): return bool(o)
    return str(o)


def canon_sha(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, default=_js).encode()).hexdigest()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def load_res():
    return json.loads(RES_F.read_text(encoding='utf-8')) if RES_F.exists() else {}


def save_key(key, obj, need=(), overwrite=True):
    res = load_res()
    for k in need: assert k in res, f'{k} 不存在：執行順序錯誤'
    if not overwrite: assert key not in res, f'{key} 已存在，不可覆寫'
    res[key] = obj
    RES_F.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=_js), encoding='utf-8')


def register(key, obj):
    res = load_res()
    assert key not in res, f'{key} 已登記，不可覆寫'
    obj = json.loads(json.dumps(obj, ensure_ascii=False, default=_js))
    res[key] = obj; res[key + '_sha256'] = canon_sha(obj); res[key + '_utc'] = now_utc()
    RES_F.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=_js), encoding='utf-8')
    return res[key + '_sha256']


def check_reg(key, files=True):
    res = load_res()
    assert key in res and key + '_sha256' in res, f'{key} 不存在：必須先登記'
    assert canon_sha(res[key]) == res[key + '_sha256'], f'{key} 被改過'
    if files:
        for rel, h in res[key].get('file_sha256', {}).items():
            assert sha(HERE / rel) == h, f'{rel} 與 {key} 登記的 sha256 不符'
    return res[key]


def g8p1_protocol():
    p = G8P1_DIR / 'G8_PROTOCOL.json'
    assert sha(p) == G8P1_SHA, 'G8P-1 協定 sha 不符'
    P = json.loads(p.read_text(encoding='utf-8'))
    assert abs(P['params']['sigma_design'] - SIGMA_D) < 1e-12 and abs(P['params']['c'] - C_INFL) < 1e-12
    assert abs(P['params']['b'] - B_G8P1) < 1e-12 and abs(P['params']['c_d'] - CD_G8P1) < 1e-12
    return P


def resid(drop_max=False):
    e = np.array(g8p1_protocol()['data']['resid_post2019'], float)
    if drop_max: e = np.delete(e, int(np.argmax(e)))           # 2021-05（+26.14）
    return e - e.mean()


# ---------------------------------------------------------------- DGP
def gen(dgp, mu, sigma, P, H, rng, block=6):
    """回傳 P×H：mu＋σ·（單位變異雜訊）。D3／D3x：區塊 bootstrap 殘差等比縮放到 σ（以 ddof＝1 sd）。"""
    if dgp == 'D1':
        return mu + sigma * rng.standard_normal((P, H))
    if dgp == 'D2':
        return mu + sigma * rng.standard_t(4, size=(P, H)) / math.sqrt(2.0)
    if dgp in ('D3', 'D3x'):
        e = resid(drop_max=(dgp == 'D3x')); e = e / e.std(ddof=1); n = len(e)
        idx = np.empty((P, H), np.int64)
        idx[:, 0] = rng.integers(0, n, P)
        for t in range(1, H):
            jump = rng.random(P) < 1.0 / block
            idx[:, t] = np.where(jump, rng.integers(0, n, P), (idx[:, t - 1] + 1) % n)
        return mu + sigma * e[idx]
    if dgp == 'D4':
        ph = REG_P_LH / (REG_P_LH + REG_P_HL)
        s = (rng.random(P) < ph).astype(int)
        out = np.empty((P, H))
        tt = rng.standard_t(4, size=(P, H)) / math.sqrt(2.0)
        for t in range(H):
            out[:, t] = mu + sigma * REG_M[s] * tt[:, t]
            u = rng.random(P)
            s = np.where(s == 1, np.where(u < REG_P_HL, 0, 1), np.where(u < REG_P_LH, 1, 0))
        return out
    raise ValueError(dgp)


DGP_ID = {'D1': 1, 'D2': 2, 'D3': 3, 'D3x': 4, 'D4': 5}


def cell_rng(base, dgp, sig_i, s_i, a_i, mode=0):
    """每格獨立 seed：SeedSequence([base, DGP_ID, σ 格序, s 格序（99＝無）, α 序, 口徑 mode])。"""
    return np.random.default_rng(np.random.SeedSequence([int(base), DGP_ID[dgp], int(sig_i), int(s_i), int(a_i), int(mode)]))


# ---------------------------------------------------------------- 統計量
def _sd(S, S2, n):
    m = S / n
    return np.sqrt(np.maximum((S2 - n * m * m) / (n - 1), 0.0))


def hac_lrv(X, n, L=6):
    """X 前 n 欄的 Bartlett L 長期變異（去均值，γ_k 除以 n）。"""
    Y = X[:, :n] - X[:, :n].mean(axis=1, keepdims=True)
    v = (Y * Y).sum(axis=1) / n
    for k in range(1, min(L, n - 1) + 1):
        v = v + 2.0 * (1.0 - k / (L + 1.0)) * (Y[:, k:] * Y[:, :-k]).sum(axis=1) / n
    return np.maximum(v, 1e-12)


def t_to_z(t, df):
    """z̃＝Φ⁻¹(F_t(t; df))，用尾端函數保精度。"""
    t = np.asarray(t, float)
    a = st.norm.isf(st.t.sf(np.abs(t), df))
    return np.where(t >= 0, a, -a)


def zstats(X, ver):
    """回傳 dict：lit（S12>0，用 x 本身）、z{n}（n∈12,18,24，版本統計量，normal 尺度）、sd{n}、S{n}。"""
    stat = ver['stat']; delta = ver.get('delta', 0.0)
    S = np.cumsum(X, axis=1); S2 = np.cumsum(X * X, axis=1)
    Xd = X - delta
    Sd = S - delta * np.arange(1, X.shape[1] + 1)[None, :]
    Sd2 = np.cumsum(Xd * Xd, axis=1)
    out = dict(lit=S[:, 11] > 0, S12=S[:, 11])
    cs = C_INFL * SIGMA_D
    if stat == 'FIX_REFIT':
        sig = np.full(X.shape[0], cs); trig = np.zeros(X.shape[0], bool)
        for n in LOOK_DN:
            s_hat = _sd(S[:, n - 1], S2[:, n - 1], n)
            new = (s_hat > 1.5 * SIGMA_D) & ~trig
            sig = np.where(new, np.maximum(cs, s_hat), sig); trig |= new
            out[f'z{n}'] = S[:, n - 1] / (sig * math.sqrt(n)); out[f'sd{n}'] = s_hat
    elif stat in ('MAXSD', 'LIT_ONLY'):
        for n in LOOK_DN:
            s_hat = _sd(Sd[:, n - 1], Sd2[:, n - 1], n)
            out[f'z{n}'] = Sd[:, n - 1] / (np.maximum(cs, s_hat) * math.sqrt(n)); out[f'sd{n}'] = s_hat
    elif stat == 'HAC_T':
        for n in LOOK_DN:
            lrv = hac_lrv(Xd, n, 6)
            t = Sd[:, n - 1] / (np.sqrt(lrv) * math.sqrt(n))
            out[f'z{n}'] = t_to_z(t, n - 1); out[f'sd{n}'] = _sd(Sd[:, n - 1], Sd2[:, n - 1], n)
    else:
        raise ValueError(stat)
    return out


NEG = -1e9


def bstar(Z, c_d, upgrade=True):
    """每條路徑「剛好升級所需的 b」：B*＝max(z12/√2, z24（若 18 未下修）)，字面 G8 未過或 12 下修 → −∞。
    升級 ⇔ B* ≥ b（判定器語意：12 先字面閘、再下修、再升級；18 只下修；24 先下修再升級；升級後吸收）。"""
    if not upgrade:
        return np.full(len(Z['lit']), NEG)
    d12 = Z['z12'] <= -c_d; d18 = Z['z18'] <= -c_d
    m12 = Z['z12'] / math.sqrt(2.0)
    m24 = np.where(d18, NEG, Z['z24'])
    B = np.maximum(m12, m24)
    return np.where(Z['lit'] & ~d12, B, NEG)


def outcomes(Z, b, c_d, upgrade=True):
    """24 個月內：up（S1 證據，含字面閘）、up12（第 12 月即觸發）、dn_seq（序貫下修，不含字面閘；升級後吸收）、
    fail（字面未過或序貫下修）、lit。"""
    lit = Z['lit']
    m12 = Z['z12'] / math.sqrt(2.0)
    d12 = Z['z12'] <= -c_d; d18 = Z['z18'] <= -c_d; d24 = Z['z24'] <= -c_d
    if upgrade:
        u12 = (m12 >= b) & ~d12
        u24 = ~u12 & ~d12 & ~d18 & (Z['z24'] >= b)
    else:
        u12 = np.zeros_like(lit); u24 = np.zeros_like(lit)
    dn_seq = d12 | (~u12 & d18) | (~u12 & ~d18 & d24 & ~u24)
    up = lit & (u12 | u24)
    up12 = lit & u12
    fail = ~lit | (lit & ~up & dn_seq)
    return dict(up=up, up12=up12, dn_seq=dn_seq, fail=fail, lit=lit)


def summarize(o, P):
    r = {k: float(np.mean(v)) for k, v in o.items()}
    r['se_up'] = math.sqrt(max(r['up'] * (1 - r['up']), 1e-12) / P)
    r['P'] = int(P)
    return r


def calib_cd(ver, P, seed, target=0.55, s=0.0, alpha=-1.0, dw=None):
    """c_d：D1、σ_d、s、α_net 下，P(min_{n∈12,18,24} z_n ≤ −c_d)＝target（不含升級吸收與字面閘，同 G8P-1）。"""
    rng = np.random.default_rng(seed)
    mu = alpha + s - dw
    X = gen('D1', mu, SIGMA_D, P, 24, rng)
    Z = zstats(X, ver)
    zmin = np.minimum(np.minimum(Z['z12'], Z['z18']), Z['z24'])
    return float(-np.quantile(zmin, target))


# ---------------------------------------------------------------- 歷史決策線（V7X GROW 100 萬，一般／壓力；逐 J 窗）
def hist_windows():
    """回傳 (Mn, Ms)：錨定月 × j 的窗超額（pp），V7X B10 K5T 100 萬 GROW，一般／壓力。"""
    import pandas as pd
    sys.path.insert(0, str(AG / 'o_forward_sequential_v2'))
    import fslib as F
    sys.path.pop(0)
    E = F.e50(); Dm = F.deadlines()
    cols = [f'B10|J{j}|K5T|{c}|100萬|GROW' for j in range(13) for c in ('normal', 'stress')]
    X = __import__('pandas').read_parquet(AG / 'o_exec_engine_rewrite_v7x/out/navs_oos_R4.parquet', columns=cols)
    out = {}
    for c in ('normal', 'stress'):
        M = {}
        for j in range(13):
            W = F.window_excess(X[f'B10|J{j}|K5T|{c}|100萬|GROW'], F.j_signals(j, 'AB_V7_B10'), F.j_end(j, 'AB_V7_B10'), E, Dm)
            W = W[~W.index.duplicated(keep='last')].dropna(subset=['r', 'r50'])
            W = W[[x is not None for x in W.index]]
            W.index = pd.PeriodIndex(W.index, freq='M')
            M[j] = W.ex
        out[c] = pd.DataFrame(M)
    return out['normal'], out['stress'], F


def hist_windows_v7fix():
    """V7 navs_exec7（o8 fslib.cell_jmatrix，100 萬 FIX）一般／壓力：交叉核對用。"""
    import pandas as pd
    sys.path.insert(0, str(AG / 'o_forward_sequential_v2'))
    import fslib as F
    sys.path.pop(0)
    Mn, _, _ = F.cell_jmatrix('100萬', 'FIX', cost='normal'); Ms, _, _ = F.cell_jmatrix('100萬', 'FIX', cost='stress')
    for M in (Mn, Ms): M.index = pd.PeriodIndex(M.index, freq='M')
    return Mn, Ms


def jmed_line(Mn, Ms=None, min_j=7, worse=True):
    """決策線：逐窗 min(一般, 壓力)（worse＝True）或一般成本，再取有效 J 窗中位數（≥ min_j）。"""
    M = Mn if (Ms is None or not worse) else np.minimum(Mn, Ms.reindex_like(Mn))
    ok = M.notna().sum(axis=1) >= min_j
    return M.median(axis=1, skipna=True).where(ok).dropna()
