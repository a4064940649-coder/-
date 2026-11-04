#!/usr/bin/env python3
"""o_g8_protocol_fin_erratum 共用函式庫（題 A：主策略 G8 前瞻序貫檢定協定）。

只讀 import：o_forward_sequential_v2/fslib.py（o8；歷史決策線，用來取殘差形狀與設計 σ，不校準成本）。
本檔不讀、不寫 live_tracking/main_forward 與 o_forward_infra_live 的任何檔案（前瞻檔只由 s1_explore.py 讀欄位定義）。

符號：x_t＝第 t 個「有效」錨定月的決策線月超額（pp，JMED 對 0050）；S_n＝Σ_{t≤n} x_t；σ_d＝設計 sd（o9 凍結 6.1414pp）。
DGP：
  D1 iid 常態 N(α, σ_d²)
  D2 t(4) 厚尾：α＋σ_d·t4/√2（單位變異）
  D3 定態區塊 bootstrap（Politis-Romano，平均區塊 L＝6 個月、環狀）：2019-05～2026-09 決策線殘差（去均值、不縮放）＋α
  敏感度：D3_2024（只用 2024-01 起殘差）、D3_L3／D3_L12（區塊 3／12）、D4_o8（o8 的 t(4)×波動狀態）
規則（全部先觸發者吸收；升級只在 24 個月期滿時生效，但事件月照記）：
  共同下修模組 DN：z_n＝S_n／(σ_eff√n)，n∈{12,18,24}（之後每 6 個月）z_n ≤ −c_d 即下修
  R-FIX：升級檢視 {12,24}，OBF 形界限 z_n ≥ b·√(24/n)，b 在單位布朗運動下使 P(穿越)＝0.05
  R-OBF：升級檢視 {6,12,18,24}，同上（4 次檢視）
  R-E ：季為單位的區塊混合 e-process（季和 Q_i／(σ_d√3)；混合＝o8 校準先驗 θ>0 部分），E_k ≥ 20 即升級（k＝1..8 季）
  R-E+：R-E 但季和除以 c·σ_d√3
  σ_eff＝c·σ_d（R-FIX、R-OBF、R-E+ 與 DN 一律用 c；R-E 升級用 σ_d、DN 仍用 c）
  refit（協定 (b)）：檢視點 n ≥ 12 時，若前 n 個月樣本 sd ＞ 1.5σ_d → 自此 σ_eff＝max(σ_eff, 樣本 sd)（兩側同時；只會放寬下修、收緊升級）
"""
import sys, json, math, hashlib, datetime
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
AG = HERE.parent
RES_F = HERE / 'results.json'
SCR = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/o14_g8')
SCR.mkdir(parents=True, exist_ok=True)

SIGMA_D = 6.141373712658076          # o9 PREREG_FORWARD.power_warning.sigma_design_pp（＝o8 sigma_post2019，HAC L6）
REG_M = np.array([0.7, 1.4]); REG_P_HL, REG_P_LH = 0.06, 0.0319


# ---------------------------------------------------------------- results.json
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
    res[key] = obj; res[key + '_sha256'] = canon_sha(obj)
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


# ---------------------------------------------------------------- DGP
def gen(dgp, alpha, P, H, rng, resid=None, sigma=SIGMA_D, block=6):
    if dgp == 'D1':
        return alpha + sigma * rng.standard_normal((P, H))
    if dgp == 'D2':
        return alpha + sigma * rng.standard_t(4, size=(P, H)) / math.sqrt(2.0)
    if dgp.startswith('D3'):
        e = np.asarray(resid, float); e = e - e.mean(); n = len(e)
        idx = np.empty((P, H), np.int64)
        idx[:, 0] = rng.integers(0, n, P)
        pnew = 1.0 / block
        for t in range(1, H):
            jump = rng.random(P) < pnew
            idx[:, t] = np.where(jump, rng.integers(0, n, P), (idx[:, t - 1] + 1) % n)
        return alpha + e[idx]
    if dgp == 'D4_o8':
        ph = REG_P_LH / (REG_P_LH + REG_P_HL)
        s = (rng.random(P) < ph).astype(int)
        out = np.empty((P, H))
        tt = rng.standard_t(4, size=(P, H)) / math.sqrt(2.0)
        for t in range(H):
            out[:, t] = alpha + sigma * REG_M[s] * tt[:, t]
            u = rng.random(P)
            s = np.where(s == 1, np.where(u < REG_P_HL, 0, 1), np.where(u < REG_P_LH, 1, 0))
        return out
    raise ValueError(dgp)


# ---------------------------------------------------------------- 界限
def obf_b(looks, Hmax=24, alpha=0.05, P=2_000_000, seed=20261105):
    """單位布朗運動（月步長、unit 變異）下 OBF 形界限 z_n ≥ b·√(Hmax/n) 在 looks 穿越機率＝alpha 的 b（模擬，固定 seed）。"""
    rng = np.random.default_rng(seed)
    looks = sorted(looks); out = []
    # 以獨立增量模擬 S 在 looks 的值（只需要 looks 處）
    prev = 0; S = np.zeros(P)
    M = np.full(P, -np.inf)
    for n in looks:
        S = S + rng.standard_normal(P) * math.sqrt(n - prev); prev = n
        M = np.maximum(M, S / math.sqrt(n) / math.sqrt(Hmax / n))
    return float(np.quantile(M, 1 - alpha))


def dn_cd(c, target_pow, looks=(12, 18, 24), P=400_000, seed=20261106, sigma=SIGMA_D):
    """下修界限 c_d：D1（σ_d）、α＝−1、統計量 S_n/(cσ_d√n)，使 P(24 個月內 min_n z_n ≤ −c_d)＝target_pow。"""
    rng = np.random.default_rng(seed)
    H = max(looks)
    X = -1.0 + sigma * rng.standard_normal((P, H))
    S = np.cumsum(X, axis=1)
    z = np.stack([S[:, n - 1] / (c * sigma * math.sqrt(n)) for n in looks], axis=1)
    return float(-np.quantile(z.min(axis=1), target_pow))


# ---------------------------------------------------------------- e-process
def emix_logE(X, theta, w, scale):
    """季和 e-process：X P×H（H 為 3 的倍數），theta 月漂移格點（>0），w 權重（和 1），
    scale＝季和的 sd（純量，或長度 P 的逐路徑陣列；＝c·σ·√3）。回傳 P×(H/3) 的 log E_k。"""
    P_, H = X.shape; K = H // 3
    sc = np.broadcast_to(np.asarray(scale, float), (P_,))
    Q = X[:, :3 * K].reshape(P_, K, 3).sum(axis=2) / sc[:, None]    # 標準化季和
    C = np.cumsum(Q, axis=1)                                         # P×K
    mu = 3.0 * np.asarray(theta)[None, :] / sc[:, None]              # P×G：每季標準化漂移
    lw = np.log(np.asarray(w))[None, :]
    out = np.empty((P_, K))
    for k in range(K):
        ll = lw + mu * C[:, k][:, None] - 0.5 * (k + 1) * mu ** 2
        m = ll.max(axis=1, keepdims=True)
        out[:, k] = (m[:, 0] + np.log(np.exp(ll - m).sum(axis=1)))
    return out


# ---------------------------------------------------------------- 規則
def run_rule(rule, X, par, Hdec=24, refit=True):
    """回傳 (up_month, down_month)：首次事件月（無＝0），先觸發者吸收。
    par：c、c_d、b（R-FIX/R-OBF）、e_thr、theta、w（R-E/R-E+）、dn_looks_every（6）。"""
    P_, H = X.shape
    sd = par['sigma']; c = par['c']
    S = np.cumsum(X, axis=1)
    S2 = np.cumsum(X ** 2, axis=1)
    n_ = np.arange(1, H + 1)
    # refit：檢視點（12、18、24…）樣本 sd > 1.5σ_d → 自該檢視點起 σ_eff＝max(cσ_d, s_hat)
    sig_eff = np.full((P_, H), c * sd)
    trig = np.zeros(P_, bool); trig_m = np.zeros(P_, int)
    if refit:
        looks_r = [n for n in range(12, H + 1, 6)]
        trig = np.zeros(P_, bool); cur = np.full(P_, c * sd); trig_m = np.zeros(P_, int)
        for n in range(1, H + 1):
            if n in looks_r:
                m = S[:, n - 1] / n
                v = (S2[:, n - 1] - n * m ** 2) / (n - 1)
                s_hat = np.sqrt(np.maximum(v, 0))
                new = s_hat > 1.5 * sd
                cur = np.where(new & ~trig, np.maximum(c * sd, s_hat), cur)
                trig_m = np.where(new & ~trig, n, trig_m)
                trig |= new
            sig_eff[:, n - 1] = cur
    z = S / (sig_eff * np.sqrt(n_)[None, :])
    up = np.zeros((P_, H), bool); dn = np.zeros((P_, H), bool)
    dn_looks = [n for n in range(12, H + 1, par.get('dn_every', 6))]
    for n in dn_looks:
        dn[:, n - 1] = z[:, n - 1] <= -par['c_d']
    if rule in ('R-FIX', 'R-OBF'):
        looks = par['up_looks']
        for n in looks:
            if n <= H:
                up[:, n - 1] = z[:, n - 1] >= par['b'] * math.sqrt(Hdec / n)
    elif rule in ('R-E', 'R-E+'):
        cc = 1.0 if rule == 'R-E' else c
        Hq = (min(H, Hdec) // 3) * 3
        lE = emix_logE(X[:, :Hq], par['theta'], par['w'], cc * sd * math.sqrt(3))
        # refit 只能收緊升級：觸發路徑自觸發月起改用 σ_alt＝max(cc·σ_d, s_hat) 計算 E，並取兩者較小
        if refit and trig.any():
            s_alt = np.where(trig, np.maximum(cc * sd, sig_eff[:, -1]), cc * sd)
            lE2 = emix_logE(X[:, :Hq], par['theta'], par['w'], s_alt * math.sqrt(3))
            qend = 3 * np.arange(1, lE.shape[1] + 1)
            after = trig_m[:, None] <= qend[None, :]
            lE = np.where(after, np.minimum(lE, lE2), lE)
        thr = math.log(par['e_thr'])
        for k in range(lE.shape[1]):
            up[:, 3 * (k + 1) - 1] = lE[:, k] >= thr
    else:
        raise ValueError(rule)
    up[:, Hdec:] = False                                              # 升級只在 24 個月內判定（本協定）
    tu = np.where(up.any(axis=1), up.argmax(axis=1) + 1, 0)
    td = np.where(dn.any(axis=1), dn.argmax(axis=1) + 1, 0)
    u = np.where((tu > 0) & ((td == 0) | (tu < td)), tu, 0)
    d = np.where((td > 0) & ((tu == 0) | (td < tu)), td, 0)
    return u, d


def oc(u, d, X=None, horizons=(12, 18, 24)):
    P_ = len(u)
    r = {}
    for h in horizons:
        pu = float(np.mean((u > 0) & (u <= h))); pd_ = float(np.mean((d > 0) & (d <= h)))
        r[int(h)] = dict(up=pu, down=pd_, se_up=math.sqrt(max(pu * (1 - pu), 1e-12) / P_), se_down=math.sqrt(max(pd_ * (1 - pd_), 1e-12) / P_))
    if X is not None and X.shape[1] >= 12:
        g8 = X[:, :12].sum(axis=1) > 0
        r['literal_g8_pass'] = float(g8.mean())
        # 合併程序（協定 (a)）：第 12 個有效月字面 G8 未過 → 前瞻失敗（吸收、升級路徑關閉）；
        # 字面 G8 通過時，事件＝序貫規則本身（升級與下修互斥、先觸發者吸收）
        uu = np.where(g8, u, 0)
        dd = np.where(g8, d, 12)
        r['combined'] = {int(h): dict(up=float(np.mean((uu > 0) & (uu <= h))), fail=float(np.mean((dd > 0) & (dd <= h)))) for h in horizons}
    return r


def mc_se(p, P):
    return math.sqrt(max(p * (1 - p), 1e-12) / P)
