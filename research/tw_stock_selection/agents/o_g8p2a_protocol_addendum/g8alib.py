#!/usr/bin/env python3
"""o_g8p2a_protocol_addendum 共用函式庫（G8P-2a：G8P-2 只能更嚴的補遺）。

不產生任何選股訊號、不讀任何前瞻報酬。只讀：
  - G8P-2 凍結檔 o_g8p2_protocol_refreeze/G8_PROTOCOL_v2.json（sha256 dcab4dbd…b993 斷言）：σ_d、c、b、c_d、Δ_w、resid_post2019。
  - 歷史可執行線（只在探索取形狀／ACF、在 PREREG 後做 in-sample 規則示範）：
    o_exec_engine_rewrite_v7x/out/navs_oos_R4.parquet（V7X K5T GROW 100 萬，一般／壓力）＋ o_forward_sequential_v2/fslib.py（窗與錨定月）。

符號（pp／月）：x_n 第 n 個有效錨定月的決策線；S_n＝Σx；s_n＝前 n 個有效月樣本 sd（ddof＝1）。
v2（G8P-2 V1）：z_n＝S_n／(max(c·σ_d, s_n)·√n)，b＝1.93376，c_d＝1.20985。
v2a 候選（全部「v2a 決策＝v2 決策 ∧ 新條件」；失敗＝v2 失敗 ∨ 新規則失敗；字面 G8 不變）：
  A1：z^A_n＝S_n／(c·max(σ_d, s_n)·√n)，升級界限 b_a（重校），下修 c_d 沿用。
  A2：A1，且升級時 σ_checked＝max(s_n, 滾動 12, 滾動 24 sd) ≤ 10.4（所在格 OC 最大誤升級 ≤ 0.05），否則不升級。
  A3：z^H_n＝S_n／(max(c·σ_d, ŝ_HAC6,n)·√n)（Newey-West Bartlett L＝6、去均值、γ_k 除以 n）。
  A4：沒有升級路徑（v2 的 V4＋A1 下修聯集）。
"""
import sys, json, math, hashlib, datetime
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
AG = HERE.parent
RES_F = HERE / 'results.json'
SCR = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/o16_g8p2a')
V2_DIR = AG / 'o_g8p2_protocol_refreeze'
V2_PROTO_SHA = 'dcab4dbd255fb11e0ad8f4508397ae109cae9a47090eee17ab2ec7952981b993'
V2_MANIFEST_SHA = '23f45ce8ee8040a1fa8e721f76241d377f36e573e32a882d9e63c64aa4ed5c30'
V2_EVAL_SHA = '1e4c0a16ab627e9418fef688bda4e27dfbb0540a53c6eacb954c7e6cfe22c38f'
G8P1_SHA = '799f8b845b69f4df6f18a6e9cc901aa804fcd0d5dee393ef2bd97f7449b113fa'

SIGMA_D = 6.141373712658076
C_INFL = 1.0596378779547428
B_V2 = 1.9337576513495873
CD_V2 = 1.2098465850539828
DW = 0.527
SIG_GRID = [SIGMA_D, 6.51, 7.0, 7.5, 8.0, 8.4, 9.0, 9.5, 10.4]
SIG_LABEL = ['6.14', '6.51', '7.0', '7.5', '8.0', '8.4', '9.0', '9.5', '10.4']
S_GRID = [0.0, 0.2, 0.4, 0.577, 0.78]
ALPHAS = [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0]          # α 序 0..5（α＝0 → 2，與 v2 b_rule 相同）
DGPS = ['D1', 'D2', 'D3', 'AR03', 'AR057', 'AR08', 'AR10', 'MA1', 'MA2', 'MA3', 'D5']
DGP_ID = {'D1': 1, 'D2': 2, 'D3': 3, 'D3x': 4, 'D4': 5, 'AR03': 11, 'AR057': 12, 'AR08': 13, 'AR10': 14,
          'MA1': 21, 'MA2': 22, 'MA3': 23, 'D5': 31, 'ARN057': 41, 'ARN08': 42, 'ARN10': 43, 'MAT3': 44}
AR_RHO = {'AR03': 0.03, 'AR057': 0.057, 'AR08': 0.08, 'AR10': 0.10, 'ARN057': 0.057, 'ARN08': 0.08, 'ARN10': 0.10}
MA_THETA = {'MA1': 0.1, 'MA2': 0.2, 'MA3': 0.3, 'MAT3': 0.3}
T_DF = 5                                             # AR 創新：t5 標準化到單位變異；MA 創新：常態（較保守）。ARN＝常態 AR、MAT＝t5 MA，只作敏感度
SIG_MAX = 10.4
VERSIONS = ['A1', 'A2', 'A3', 'A4']
NEG = -1e9


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


def save_key(key, obj, need=()):
    res = load_res()
    for k in need: assert k in res, f'{k} 不存在：執行順序錯誤'
    res[key] = json.loads(json.dumps(obj, ensure_ascii=False, default=_js))
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


def v2_protocol():
    p = V2_DIR / 'G8_PROTOCOL_v2.json'
    assert sha(p) == V2_PROTO_SHA, 'G8P-2 協定 sha 不符'
    P = json.loads(p.read_text(encoding='utf-8'))
    pr = P['params']
    assert abs(pr['sigma_design'] - SIGMA_D) < 1e-12 and abs(pr['c'] - C_INFL) < 1e-12
    assert abs(pr['b'] - B_V2) < 1e-12 and abs(pr['c_d'] - CD_V2) < 1e-12 and abs(pr['delta_w'] - DW) < 1e-12
    return P


_RES = {}
def resid_post2019():
    if 'post' not in _RES:
        e = np.array(v2_protocol()['data']['resid_post2019'], float); e = e - e.mean()
        _RES['post'] = e / e.std(ddof=1)
    return _RES['post']


def resid_sel():
    """D5：挑選期 2006-04～2019-04 V7X GROW100 較差成本 JMED 線去均值、標準化（只用形狀）。
    優先讀本資料夾 d5_resid_sel.json（探索時寫出、sha 由 PREREG_DESIGN 鎖定）。"""
    if 'sel' not in _RES:
        f = HERE / 'd5_resid_sel.json'
        d = json.loads(f.read_text(encoding='utf-8'))
        e = np.array(d['z'], float)
        assert abs(e.mean()) < 1e-9 and abs(e.std(ddof=1) - 1) < 1e-9
        _RES['sel'] = e
    return _RES['sel']


# ---------------------------------------------------------------- DGP
def _tstd(rng, shape):
    return rng.standard_t(T_DF, size=shape) / math.sqrt(T_DF / (T_DF - 2.0))


def _boot(e, P, H, rng, block=6):
    n = len(e)
    idx = np.empty((P, H), np.int64)
    idx[:, 0] = rng.integers(0, n, P)
    for t in range(1, H):
        jump = rng.random(P) < 1.0 / block
        idx[:, t] = np.where(jump, rng.integers(0, n, P), (idx[:, t - 1] + 1) % n)
    return e[idx]


def gen(dgp, mu, sigma, P, H, rng):
    """P×H：mu＋σ·（邊際單位變異雜訊）。"""
    if dgp == 'D1':
        return mu + sigma * rng.standard_normal((P, H))
    if dgp == 'D2':
        return mu + sigma * rng.standard_t(4, size=(P, H)) / math.sqrt(2.0)
    if dgp == 'D3':
        return mu + sigma * _boot(resid_post2019(), P, H, rng)
    if dgp == 'D5':
        return mu + sigma * _boot(resid_sel(), P, H, rng)
    if dgp in AR_RHO:
        rho = AR_RHO[dgp]
        u = rng.standard_normal((P, H)) if dgp.startswith('ARN') else _tstd(rng, (P, H))
        e = np.empty((P, H)); e[:, 0] = u[:, 0]                     # 平穩起點（邊際變異 1）
        a = math.sqrt(1 - rho * rho)
        for t in range(1, H): e[:, t] = rho * e[:, t - 1] + a * u[:, t]
        return mu + sigma * e
    if dgp in MA_THETA:
        th = MA_THETA[dgp]
        u = _tstd(rng, (P, H + 1)) if dgp.startswith('MAT') else rng.standard_normal((P, H + 1))
        return mu + sigma * (u[:, 1:] + th * u[:, :-1]) / math.sqrt(1 + th * th)
    raise ValueError(dgp)


def cell_rng(base, dgp, sig_i, s_i, a_i, mode=0):
    """SeedSequence([base, DGP_ID, σ 序, s 序（5＝Δ_w）, α 序, 口徑 mode（0 主、1 嚴格）])；同一格各版本與 v2 共用路徑（CRN）。"""
    return np.random.default_rng(np.random.SeedSequence([int(base), DGP_ID[dgp], int(sig_i), int(s_i), int(a_i), int(mode)]))


# ---------------------------------------------------------------- 統計量
def _sd(S, S2, n):
    m = S / n
    return np.sqrt(np.maximum((S2 - n * m * m) / (n - 1), 0.0))


def hac_sd(X, n, L=6):
    """X 前 n 欄的 Newey-West Bartlett L 長期 sd（去均值，γ_k 除以 n）。"""
    Y = X[:, :n] - X[:, :n].mean(axis=1, keepdims=True)
    v = (Y * Y).sum(axis=1) / n
    for k in range(1, min(L, n - 1) + 1):
        v = v + 2.0 * (1.0 - k / (L + 1.0)) * (Y[:, k:] * Y[:, :-k]).sum(axis=1) / n
    return np.sqrt(np.maximum(v, 1e-12))


def all_stats(X):
    """回傳 dict：lit、S{n}、sd{n}、z_v2{n}、z_A{n}、z_H{n}（n＝12,18,24）、chk12、chk24（A2 σ 檢查）。"""
    S = np.cumsum(X, axis=1); S2 = np.cumsum(X * X, axis=1)
    cs = C_INFL * SIGMA_D
    o = dict(lit=S[:, 11] > 0)
    for n in (12, 18, 24):
        s = _sd(S[:, n - 1], S2[:, n - 1], n); rn = math.sqrt(n)
        o[f'sd{n}'] = s
        o[f'v2{n}'] = S[:, n - 1] / (np.maximum(cs, s) * rn)
        o[f'A{n}'] = S[:, n - 1] / (C_INFL * np.maximum(SIGMA_D, s) * rn)
        o[f'H{n}'] = S[:, n - 1] / (np.maximum(cs, hac_sd(X, n, 6)) * rn)
    r12 = X[:, 12:24].std(axis=1, ddof=1)
    o['chk12'] = o['sd12'] <= SIG_MAX + 1e-12
    o['chk24'] = np.maximum(o['sd24'], r12) <= SIG_MAX + 1e-12
    return o


def outcomes_rule(lit, z12, z18, z24, b, cd, cond12=None, cond24=None, upgrade=True):
    """單一規則的 24 個月結果（同 G8P-2 判定器語意）：12 先字面、再下修、再升級；18 只下修；24 先下修再升級；先觸發者吸收。"""
    d12 = z12 <= -cd; d18 = z18 <= -cd; d24 = z24 <= -cd
    if upgrade:
        c12 = True if cond12 is None else cond12
        c24 = True if cond24 is None else cond24
        u12 = (z12 / math.sqrt(2.0) >= b) & ~d12 & c12
        u24 = ~u12 & ~d12 & ~d18 & (z24 >= b) & c24
    else:
        u12 = np.zeros_like(lit); u24 = np.zeros_like(lit)
    dn_seq = d12 | (~u12 & d18) | (~u12 & ~d18 & d24 & ~u24)
    up = lit & (u12 | u24)
    fail = ~lit | (lit & ~up & dn_seq)
    return dict(up=up, up12=lit & u12, dn_seq=dn_seq, fail=fail)


def bstar_rule(lit, z12, z18, z24, cd, cond12=None, cond24=None):
    """單一規則「剛好升級所需的 b」：升級(b) ⇔ B* ≥ b。"""
    d12 = z12 <= -cd; d18 = z18 <= -cd
    m12 = z12 / math.sqrt(2.0)
    if cond12 is not None: m12 = np.where(cond12, m12, NEG)
    m24 = np.where(d18, NEG, z24)
    if cond24 is not None: m24 = np.where(cond24, m24, NEG)
    return np.where(lit & ~d12, np.maximum(m12, m24), NEG)


def version_parts(Z, ver):
    """版本的新規則統計量與 σ 條件。"""
    if ver in ('A1', 'A4'): return Z['A12'], Z['A18'], Z['A24'], None, None
    if ver == 'A2': return Z['A12'], Z['A18'], Z['A24'], Z['chk12'], Z['chk24']
    if ver == 'A3': return Z['H12'], Z['H18'], Z['H24'], None, None
    raise ValueError(ver)


def combined(Z, ver, b_a, b_v2=B_V2, cd=CD_V2):
    """v2a 版本 ver 的結果：up＝up_v2 ∧ up_new；fail＝fail_v2 ∨ fail_new。另回傳 v2 結果與逐路徑子集檢查。"""
    lit = Z['lit']
    o2 = outcomes_rule(lit, Z['v212'], Z['v218'], Z['v224'], b_v2, cd)
    z12, z18, z24, c12, c24 = version_parts(Z, ver)
    on = outcomes_rule(lit, z12, z18, z24, b_a, cd, c12, c24, upgrade=(ver != 'A4'))
    if ver == 'A4':
        o2u = outcomes_rule(lit, Z['v212'], Z['v218'], Z['v224'], b_v2, cd, upgrade=False)
        up = np.zeros_like(lit); fail = o2u['fail'] | on['fail']
        up12 = up
    else:
        up = o2['up'] & on['up']; fail = o2['fail'] | on['fail']
        up12 = on['up12'] & o2['up']
    assert not np.any(up & fail), '升級與失敗同時成立：實作錯誤'
    dn = (o2u['dn_seq'] if ver == 'A4' else o2['dn_seq']) | on['dn_seq']
    return dict(up=up, up12=up12, fail=fail, dn_seq=dn, lit=lit, v2_dn=o2['dn_seq'],
                v2_up=o2['up'], v2_fail=o2['fail'],
                viol_up=int(np.sum(up & ~o2['up'])), viol_fail=int(np.sum(o2['fail'] & ~fail)))


def bstar_combined(Z, ver, b_v2=B_V2, cd=CD_V2):
    """校準用：B*_comb＝where(up_v2, B*_new, −∞)。升級_v2a(b_a) ⇔ B*_comb ≥ b_a。"""
    lit = Z['lit']
    o2 = outcomes_rule(lit, Z['v212'], Z['v218'], Z['v224'], b_v2, cd)
    z12, z18, z24, c12, c24 = version_parts(Z, ver)
    return np.where(o2['up'], bstar_rule(lit, z12, z18, z24, cd, c12, c24), NEG)


def summ(o, P):
    r = {k: float(np.mean(v)) for k, v in o.items() if isinstance(v, np.ndarray)}
    r['se_up'] = math.sqrt(max(r['up'] * (1 - r['up']), 1e-12) / P)
    r['P'] = int(P)
    for k in ('viol_up', 'viol_fail'):
        if k in o: r[k] = int(o[k])
    return r


# ---------------------------------------------------------------- 歷史決策線（V7X GROW 100 萬，一般／壓力；逐 J 窗）
def hist_windows():
    """回傳 (Mn, Ms, F, starts)：錨定月 × j 的窗超額（pp），V7X B10 K5T 100 萬 GROW，一般／壓力；starts＝各窗起訖日。"""
    import pandas as pd
    sys.path.insert(0, str(AG / 'o_forward_sequential_v2'))
    import fslib as F
    sys.path.pop(0)
    E = F.e50(); Dm = F.deadlines()
    cols = [f'B10|J{j}|K5T|{c}|100萬|GROW' for j in range(13) for c in ('normal', 'stress')]
    X = pd.read_parquet(AG / 'o_exec_engine_rewrite_v7x/out/navs_oos_R4.parquet', columns=cols)
    out = {}; win = {}
    for c in ('normal', 'stress'):
        M = {}
        for j in range(13):
            W = F.window_excess(X[f'B10|J{j}|K5T|{c}|100萬|GROW'], F.j_signals(j, 'AB_V7_B10'), F.j_end(j, 'AB_V7_B10'), E, Dm)
            W = W[~W.index.duplicated(keep='last')].dropna(subset=['r', 'r50'])
            W = W[[x is not None for x in W.index]]
            W.index = pd.PeriodIndex(W.index, freq='M')
            M[j] = W.ex
            if c == 'normal': win[j] = W[['start', 'end']]
        out[c] = pd.DataFrame(M)
    return out['normal'], out['stress'], F, win


def jmed_line(Mn, Ms, min_j=7):
    M = np.minimum(Mn, Ms.reindex_like(Mn))
    ok = M.notna().sum(axis=1) >= min_j
    return M.median(axis=1, skipna=True).where(ok).dropna()
