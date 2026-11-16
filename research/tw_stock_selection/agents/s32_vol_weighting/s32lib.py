#!/usr/bin/env python3
"""S32 共用程式庫：衛星每檔金額改「依波動度調整」（波動大的買少、波動小的買多）。
在 s20lib（不改；其下 s17lib→s10lib：面板、創新板、SURGE（STD，σ>0）、F1、E0、引擎）之上，本檔只新增：
- 開除權息修正：cslib.EXRIGHT_FIX = True（必須在 load() 之前；現行驗收 2019 起 48.99%、全期 34.54%、回撤 −32.11%、已平倉 381 筆）。
- simulate_w()：s10lib.simulate_s10 的「原始碼最小修補版」（執行時讀 s10lib 原始碼、做 3 處字串替換後編譯；共用引擎檔案不動）：
    (1) 每檔目標金額 target＝u × WM.get((訊號列 k, 股票 j), 1.0)，WM＝S['wmul']（沒給＝空字典＝×1.0，與原引擎逐位相同，驗收 A1）；
    (2) 每日記錄「最大單檔市值 ÷ 淨值」與持股集中度 Σ(個股市值 ÷ 個股總市值)²（只讀狀態，不影響交易）。
- features()：訊號列 k（含當天收盤）往前 60／250 個交易日的日報酬波動 σ（P.R，含息還原日報酬；有值 ≥ 40／150 天才算）
  與對 0050 的 β60（P.E50 日報酬；同一窗、成對有值）。
- weights()：照規則把當月名單（make_signal_s10 的 F1，含已持有）轉成倍數 m（相對等額 u＝衛星 ÷ 10），預算中性（名單內 Σm＝檔數），
  上下限 [0.5, 1.5]（＝單檔目標 5%～15% 衛星），夾住後把差額按比例分給沒碰到上下限的股票，反覆到穩定。
中間檔放 scratchpad/s32/。規格見 PREREG.json。一律 python3 -I -B。"""
import sys, inspect
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
S32 = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s32')
sys.path.insert(0, str(AG / 's20_persistent_growth'))
import s20lib                                              # noqa: E402
from s20lib import s17lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402,F401
s20lib.S20 = S32; s17lib.S17 = S32; s10lib.S10 = S32; s9lib.S9 = S32; s8lib.S8 = S32; s7lib.S7 = S32; s6lib.S6 = S32; s5lib.S5 = S32; s4lib.S4 = S32
s3lib.S3 = S32; s2lib.S2 = S32; stlib.S1 = S32; cslib.TMP = S32
cslib.EXRIGHT_FIX = True

TRAIN_LAST_MONTH = s20lib.TRAIN_LAST_MONTH                 # 營收月份 ≤ 2021-11（訊號日 2016-02～2021-12）
VALID_FIRST_MONTH = s20lib.VALID_FIRST_MONTH               # 營收月份 ≥ 2021-12（訊號日 2022-01 起）
SEGMENTS = s17lib.SEGMENTS
fresh_from = s20lib.fresh_from
ACC = dict(since2019=0.4899, full=0.3454, maxdd=-0.3211, n_closed=381)
LO, HI = 0.5, 1.5                                          # 單檔目標 ＝ 衛星 × 10% × m → 5%～15% 衛星
BETA_FLOOR = 0.25

RULES = [
    dict(id='IV60', q='inv_sig60',
         desc='反波動：倍數 ∝ 1／σ60（訊號日含當天往前 60 個交易日日報酬標準差），名單內預算中性（Σm＝檔數），夾在 [0.5, 1.5]（單檔 5%～15% 衛星）'),
    dict(id='IVAR60', q='inv_var60',
         desc='反變異數（最小變異數、相關＝0 時的解）：倍數 ∝ 1／σ60²，預算中性，夾在 [0.5, 1.5]'),
    dict(id='IB60', q='inv_beta60',
         desc='反 β（Betting Against Beta 精神）：倍數 ∝ 1／max(β60, 0.25)，β60＝同窗對 0050 日報酬迴歸斜率；預算中性，夾在 [0.5, 1.5]'),
    dict(id='HV3', q='hv3',
         desc='等額但名單內 σ60 最高的 ceil(n／3) 檔減半（0.5），其餘等比例放大使 Σm＝檔數（上限 1.5）；n＝1 時不變'),
]
CONTROL = dict(id='HV3_CASH', q='hv3_cash',
               desc='對照（不參與選擇與判定）：σ60 最高 ceil(n／3) 檔減半、其餘不變、不重新分配（省下的錢照引擎放 0050）＝降低衛星曝險，用來分離「降曝險」與「重新分配」')


def load():
    S32.mkdir(parents=True, exist_ok=True)
    P = s20lib.load()
    assert P.exright_fix is True, '除權息修正沒有開'
    return P


def surge(P):
    F = s17lib.rev_family(P, defs=['STD'])
    return dict(S=F['S']['STD'], av=F['av'])


# ---------------------------------------------------------------- 引擎最小修補（不改 s10lib 檔案）
_src = inspect.getsource(s10lib.simulate_s10)
_REPL = [
    ("def simulate_s10(", "def simulate_w("),
    ("    sigrows = S['sigrows']; RQ = S.get('RQ', {})\n",
     "    sigrows = S['sigrows']; RQ = S.get('RQ', {})\n    WM = S.get('wmul') or {}\n"),
    ("            target = u\n", "            target = u * WM.get((k, int(j)), 1.0)\n"),
    ("    stockfrac = np.full(T, np.nan); bser = np.full(T, np.nan); bser[t0] = B\n",
     "    stockfrac = np.full(T, np.nan); bser = np.full(T, np.nan); bser[t0] = B\n    maxw = np.full(T, np.nan); hhi = np.full(T, np.nan); nposd = np.zeros(T, np.int16)\n"),
    ("        stockfrac[t] = tot / nav[t]; bser[t] = B\n",
     "        stockfrac[t] = tot / nav[t]; bser[t] = B\n"
     "        if pos:\n"
     "            _vv = [p['v'] for p in pos.values()]\n"
     "            maxw[t] = max(_vv) / nav[t]; nposd[t] = len(_vv)\n"
     "            hhi[t] = sum((x / tot) ** 2 for x in _vv) if tot > 0 else np.nan\n"
     "        else:\n"
     "            maxw[t] = 0.0\n"),
    ("    st['stockfrac'] = pd.Series(stockfrac[t0:T], index=P.IDX[t0:T]);",
     "    st['maxw'] = pd.Series(maxw[t0:T], index=P.IDX[t0:T]); st['hhi'] = pd.Series(hhi[t0:T], index=P.IDX[t0:T]); st['nposd'] = pd.Series(nposd[t0:T], index=P.IDX[t0:T])\n"
     "    st['stockfrac'] = pd.Series(stockfrac[t0:T], index=P.IDX[t0:T]);"),
]
for a, b in _REPL:
    assert _src.count(a) == 1, ('引擎原始碼與預期不符，修補失敗', a)
    _src = _src.replace(a, b)
_ns = dict(vars(s10lib))
exec(compile(_src, '<s32 simulate_w（s10lib.simulate_s10 修補版）>', 'exec'), _ns)
simulate_w = _ns['simulate_w']
PATCH = [dict(old=a, new=b) for a, b in _REPL]


# ---------------------------------------------------------------- 特徵與權重
def features(P, k, js):
    """訊號列 k、股票索引 js：回傳 dict(sig60, sig250, beta60)（長度 len(js)；不足天數為 NaN）。"""
    js = np.asarray(js, int)
    r50 = P.E50[1:] / P.E50[:-1] - 1.0                     # r50[t-1]＝第 t 日 0050 報酬
    out = {}
    for n, mn, nm in ((60, 40, 'sig60'), (250, 150, 'sig250')):
        a = P.R[max(k - n + 1, 1):k + 1, js]
        cnt = np.isfinite(a).sum(0)
        with np.errstate(invalid='ignore'):
            s = np.nanstd(a, axis=0, ddof=1)
        out[nm] = np.where(cnt >= mn, s, np.nan)
    a = P.R[k - 59:k + 1, js]; m = r50[k - 60:k][:, None] * np.ones((1, len(js)))
    ok = np.isfinite(a) & np.isfinite(m)
    b = np.full(len(js), np.nan)
    for i in range(len(js)):
        o = ok[:, i]
        if o.sum() >= 40:
            x = m[o, i]; y = a[o, i]
            vx = x.var(ddof=1)
            if vx > 0:
                b[i] = np.cov(x, y, ddof=1)[0, 1] / vx
    out['beta60'] = b
    return out


def _neutral(q, lo=LO, hi=HI):
    """q：正的原始分數（長度 n）。回傳 m，Σm＝n，夾在 [lo, hi]（夾住後把差額按 q 比例分給未夾的）。"""
    n = len(q)
    if n <= 1:
        return np.ones(n)
    m = n * q / q.sum()
    fixed = np.zeros(n, bool)
    for _ in range(100):
        over = (~fixed) & (m > hi + 1e-12); under = (~fixed) & (m < lo - 1e-12)
        if not over.any() and not under.any():
            break
        m[over] = hi; m[under] = lo; fixed |= over | under
        rest = n - m[fixed].sum(); free = ~fixed
        if not free.any():
            break
        m[free] = rest * q[free] / q[free].sum()
    return m


def mult_for(rule_q, F):
    """F＝features 的結果（名單內）。回傳倍數 m 與使用的原始分數。缺值用名單內有值者的中位數（全缺 → 等額）。"""
    s = F['sig60']; n = len(s)
    if rule_q in ('hv3', 'hv3_cash'):
        m = np.ones(n)
        if n >= 2:
            sv = np.where(np.isfinite(s), s, np.nanmedian(s) if np.isfinite(s).any() else 0.0)
            h = int(np.ceil(n / 3))
            idx = np.argsort(-sv, kind='stable')[:h]
            m[idx] = 0.5
            if rule_q == 'hv3':
                m = _neutral(m)
        return m
    if rule_q == 'inv_sig60':
        q = 1.0 / s
    elif rule_q == 'inv_var60':
        q = 1.0 / s ** 2
    elif rule_q == 'inv_beta60':
        q = 1.0 / np.maximum(F['beta60'], BETA_FLOOR)
    else:
        raise ValueError(rule_q)
    ok = np.isfinite(q) & (q > 0)
    if not ok.any():
        return np.ones(n)
    q = np.where(ok, q, np.median(q[ok]))
    return _neutral(q)


def build_wmul(P, S, rule):
    """S＝make_signal_s10 的結果。回傳 S1（多 S1['wmul']＝{(k, j): m}）與每月紀錄 log（month, k, codes, m, sig60, beta60）。rule＝None → 等額（空字典）。"""
    S1 = dict(S)
    if rule is None:
        S1['wmul'] = {}
        return S1, []
    cidx = {str(c): i for i, c in enumerate(P.COLS)}
    W = {}; log = []
    for x in S['info']:
        k = x['k']; js = [cidx[c] for c in x['targets']]
        if not js:
            continue
        F = features(P, k, js)
        m = mult_for(rule['q'], F)
        for j, v in zip(js, m):
            W[(k, j)] = float(v)
        log.append(dict(month=x['month'], k=int(k), codes=x['targets'], m=[float(v) for v in m],
                        sig60=[float(v) for v in F['sig60']], beta60=[float(v) for v in F['beta60']]))
    S1['wmul'] = W
    return S1, log


def capm_beta(nav, bench, a, b):
    """日報酬對 0050 日報酬的 OLS β 與年化 α（a～b 窗）。"""
    x = bench.loc[a:b].pct_change().dropna(); y = nav.loc[a:b].pct_change().reindex(x.index)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok].to_numpy(); y = y[ok].to_numpy()
    bb = np.cov(x, y, ddof=1)[0, 1] / x.var(ddof=1)
    al = (y.mean() - bb * x.mean()) * 252
    return dict(beta=float(bb), alpha_ann=float(al))


def calmar(nav, a, b):
    c = stlib.cagr(nav, a, b); d = stlib.maxdd(nav, a, b)
    return float(c / abs(d)) if d < 0 else float('nan')
