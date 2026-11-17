#!/usr/bin/env python3
"""S53 共用程式庫：波動調整部位家族（A 個股層 1/σ60、B 組合層衛星波動目標），選股、出場、股池一律不動。
鏈：s51lib → s47lib → s20lib → s17lib → s10lib（全部不改，只讀）；正式開關 cslib.EXRIGHT_FIX、DEADLINE_FIX、INS15_MASK 全開（s47lib 匯入時設定）。
引擎副本 simulate_s53：取 s51lib 已替換好的原始碼（＝s10lib.simulate_s10 ＋ S47 的 ndiv/idle ＋ S51 的 n_sized 計數器），再做 9 處字串替換（6 組功能）後 exec（原檔不動）：
  (1) 函式名 simulate_s53，新參數 wmult=None, vt=None；
  (2) A：買進目標 target＝u × wmult[k][j]（k＝訊號列 t−1；wmult 沒有的股票＝1）；
  (3) 新陣列 sret（衛星持股日報酬）、satser（每個換股執行日用的衛星目標）；
  (4) 每日收盤估值時記錄 sret[t]＝Σ(前收市值 × R[t]) ÷ Σ 前收市值（只算隔夜續抱的持股；當天新買與當天賣出不算；沒有持股＝NaN）；
  (5) B：換股執行日 t 在拉回之前，用 sret[t−win .. t−1]（到前一日收盤）算年化已實現波動 σ̂（有效天數 ≥ min_obs，否則用 base），
      sat_w＝clip(base × target ÷ σ̂, lo, hi)，band＝vt['band']，再走原本的 rebalance（|衛星÷淨值 − sat_w| > band 才把底倉簿記設回 (1−sat_w)×淨值，衛星過多時按比例賣股）；
  (6) st 加回傳 sret、satser。
  wmult=None、vt=None、ndiv=10 時與 s10lib.simulate_s10(sizing='W10') 逐位相同（驗收 A3）。
中間檔放 scratchpad/s53/cache。規格見 SPEC.json、PREREG.json。一律 python3 -I -B。"""
import sys, hashlib
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s53')
CACHE = TMP / 'cache'
sys.path.insert(0, str(AG / 's51_position_size_validation'))
import s51lib as L51                                       # noqa: E402
from s51lib import s20lib, s17lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402,F401
for _m, _a in ((s20lib, 'S20'), (s17lib, 'S17'), (s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'), (s7lib, 'S7'), (s6lib, 'S6'), (s5lib, 'S5'),
               (s4lib, 'S4'), (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP')):
    setattr(_m, _a, CACHE)
assert cslib.EXRIGHT_FIX is True and cslib.DEADLINE_FIX is True and cslib.INS15_MASK is True, '正式開關必須全開'

CUR = 'CUR'
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
SHIFTS = (0, 1, 2, 3)
ACC = dict(since2019=0.4875617247721724, full=0.3439657562340761, maxdd=-0.32119049020243395, n_closed=381)
DROP = L51.DROP
SIG_WIN, SIG_MINOBS = 60, 40                               # A：個股 σ60（訊號列 k 用 R 列 k−60..k−1，有成交日 ≥ 40）
W_FLOOR = 0.5                                              # A：倍數下限（÷20 衛星）
VT_BASE, VT_LO, VT_HI = 0.7, 0.5, 0.9                      # B：基準衛星、上下限

# 家族（事前寫死；目標波動 target 由 prep.py 用現行格算出後寫進 PREREG）
CELLS = {
    'CUR':     dict(kind='cur'),
    'A_D6':    dict(kind='A', cap_div=6),
    'A_D8':    dict(kind='A', cap_div=8),
    'B60':     dict(kind='B', win=60, band=0.10),
    'B120':    dict(kind='B', win=120, band=0.10),
    'B60_R':   dict(kind='B', win=60, band=0.0),
    'B120_R':  dict(kind='B', win=120, band=0.0),
}
FAMILY = [k for k in CELLS if k != 'CUR']
REF_CELL = 'REF_S70_D8'                                     # 參考（S47／S51 的「單純加碼」），只描述、不進判定

_SRC0 = L51._SRC
_REPL = [
    ("def simulate_s51(", "def simulate_s53("),
    ("requal=None, ndiv=None, idle='e50'):", "requal=None, ndiv=None, idle='e50', wmult=None, vt=None):"),
    ("            target = u\n", "            target = u * (wmult.get(k, {}).get(j, 1.0) if wmult is not None else 1.0)\n"),
    ("    nav = np.full(T, np.nan); nav[t0] = capital\n",
     "    nav = np.full(T, np.nan); nav[t0] = capital\n    sret = np.full(T, np.nan); satser = np.full(T, np.nan)\n"),
    ("""            for j, p in pos.items():
                if p['new']:""",
     """            _sv0 = 0.0; _sv1 = 0.0
            for j, p in pos.items():
                if p['new']:"""),
    ("""                else:
                    p['v'] *= (1 + R[t, j])
                    p['g'] *= (1 + R[t, j])""",
     """                else:
                    _sv0 += p['v']; _sv1 += p['v'] * R[t, j]
                    p['v'] *= (1 + R[t, j])
                    p['g'] *= (1 + R[t, j])"""),
    ("""        tot = 0.0
        is_sig = bool(sigrows[t])""",
     """            sret[t] = _sv1 / _sv0 if _sv0 > 0 else np.nan
        tot = 0.0
        is_sig = bool(sigrows[t])"""),
    ("""        if sat_w < 1.0 and t in exec_days:
            rebalance(t)""",
     """        if vt is not None and t in exec_days:
            _w = sret[max(t - vt['win'], t0 + 1):t]
            _w = _w[np.isfinite(_w)]
            if len(_w) >= vt['min_obs']:
                _sh = float(np.std(_w, ddof=1) * np.sqrt(252.0))
                sat_w = float(min(max(vt['base'] * vt['target'] / _sh, vt['lo']), vt['hi'])) if _sh > 0 else vt['hi']
            else:
                sat_w = vt['base']
            band = vt['band']
            satser[t] = sat_w
        if sat_w < 1.0 and t in exec_days:
            rebalance(t)"""),
    ("    return navs, lg, st\n",
     "    st['sret'] = pd.Series(sret[t0:T], index=P.IDX[t0:T]); st['satser'] = pd.Series(satser[t0:T], index=P.IDX[t0:T])\n    return navs, lg, st\n"),
]
_SRC = _SRC0
for _a, _b in _REPL:
    assert _SRC.count(_a) == 1, f'替換點不唯一或不存在：{_a[:50]}'
    _SRC = _SRC.replace(_a, _b)
_G = dict(s10lib.__dict__)
exec(compile(_SRC, '<s53 simulate_s53>', 'exec'), _G)
simulate_s53 = _G['simulate_s53']
SRC_SHA = dict(L51.SRC_SHA, simulate_s53=hashlib.sha256(_SRC.encode()).hexdigest())

base, slim, capped_rate = L51.base, L51.slim, L51.capped_rate


def load():
    CACHE.mkdir(parents=True, exist_ok=True)
    P = s20lib.load()
    assert P.exright_fix is True
    return P


def sigma60(P):
    """個股日報酬波動（未年化）：列 k 用 R 列 k−60..k−1 的有成交日（P.has），有效 ≥ 40 天，否則 NaN。到訊號日前一日為止。"""
    has = P.has.astype(np.float64)
    X = np.where(P.has, P.R, 0.0)
    z = np.zeros((1, P.N))
    c0 = np.vstack([z, np.cumsum(has, 0)]); c1 = np.vstack([z, np.cumsum(X, 0)]); c2 = np.vstack([z, np.cumsum(X * X, 0)])
    del X, has
    out = np.full((P.T, P.N), np.nan, dtype=np.float32)
    for k in range(SIG_WIN + 1, P.T):
        a, b = k - SIG_WIN, k                              # cumsum 前綴差：列 a..b−1＝k−60..k−1
        n = c0[b] - c0[a]
        s1 = c1[b] - c1[a]; s2 = c2[b] - c2[a]
        with np.errstate(invalid='ignore', divide='ignore'):
            v = (s2 - s1 * s1 / n) / (n - 1)
            sd = np.sqrt(np.maximum(v, 0.0))
        sd[n < SIG_MINOBS] = np.nan
        out[k] = sd
    return out


def make_wmult(S2, U, SIG, cap_div):
    """A：每個訊號列 k，名單 L＝sig[k] & U[k]（F1 名單），σ_ref＝名單 σ60 中位數，m_j＝clip(σ_ref ÷ σ_j, 0.5, 10 ÷ cap_div)；σ 缺值 m＝1。"""
    sig = S2['sig']; hi = 10.0 / cap_div; W = {}
    for k in np.flatnonzero(sig.any(1)):
        L = np.flatnonzero(sig[k] & U[k])
        if not len(L):
            continue
        s = SIG[k, L].astype(np.float64)
        ok = np.isfinite(s) & (s > 0)
        if ok.sum() < 2:
            W[int(k)] = {int(j): 1.0 for j in L}; continue
        ref = float(np.median(s[ok]))
        W[int(k)] = {int(j): (float(min(max(ref / sj, W_FLOOR), hi)) if (np.isfinite(sj) and sj > 0) else 1.0) for j, sj in zip(L, s)}
    return W


def sim(P, S, U, cell, cap=1e6, cost='strict', SIG=None, targets=None, orig=False, cap_pct=0.05):
    """cell：CELLS 的鍵或 REF_S70_D8。CUR／orig=True → 原引擎 simulate_s10 只用於驗收；其餘一律 simulate_s53。"""
    S2, ekw = s10lib.variant_S(S, 'E0')
    sat = 0.7
    kw = dict(cost=COSTS[cost], capital=cap, cap_pct=cap_pct, sat_w=sat, sizing='W10', lock=False, **ekw)
    if orig:
        return s10lib.simulate_s10(P, S2, U, **kw)
    if cell == REF_CELL:
        return simulate_s53(P, S2, U, ndiv=8, idle='e50', **kw)
    c = CELLS[cell]
    wm = vt = None
    if c['kind'] == 'A':
        wm = make_wmult(S2, U, SIG, c['cap_div'])
    elif c['kind'] == 'B':
        vt = dict(win=c['win'], min_obs=c['win'] // 3, base=VT_BASE, lo=VT_LO, hi=VT_HI, band=c['band'], target=targets[str(c['win'])])
    return simulate_s53(P, S2, U, ndiv=10, idle='e50', wmult=wm, vt=vt, **kw)


def roll_vol(sret, win):
    """B 的估計式套在一條 sret 上：每一天 t 用 t−win..t−1 有效值（≥ win//3）算年化 σ；回傳 Series。"""
    a = sret.to_numpy(); out = np.full(len(a), np.nan)
    for t in range(1, len(a)):
        w = a[max(t - win, 1):t]; w = w[np.isfinite(w)]
        if len(w) >= win // 3:
            out[t] = np.std(w, ddof=1) * np.sqrt(252.0)
    return pd.Series(out, index=sret.index)
