#!/usr/bin/env python3
"""S97 外掛：開盤鎖漲跌停的執行情境。不改任何既有函式庫。
做法同 s47lib：讀 s10lib.simulate_s10 的原始碼，做「每段剛好出現一次」的文字取代，產生 simulate_s97（在 s10lib 的名稱空間 exec）。
lim=None（預設）時每個取代點都走回原式，與 simulate_s10 逐日相同（驗收 PLUG_OFF）。
lim＝dict(crit='A'|'B'|'NONE', buy='i'|'ii'|'iii', LU=漲停價陣列, LD=跌停價陣列, LOW=最低價, HIGH=最高價)。定義見 PREREG.json。
事件紀錄在 st['lim_ev']：(t, j, 訊號列, 種類)。"""
import inspect
import numpy as np, pandas as pd

LIMIT_CHANGE = pd.Timestamp('2015-06-01')


def limit_prices(P, tick, hist7=True):
    """漲跌停價：hist7＝True → 2015-06-01 以前 7%、之後 10%；False → 一律 10%（＝cslib.Panel 的 LIMUP／LIMDN，逐位相同）。"""
    pct = np.where(np.asarray(P.IDX < LIMIT_CHANGE), 0.07, 0.10)[:, None] if hist7 else 0.10
    up = P.REF * (1 + pct); t = tick(up); LU = np.floor(up / t + 1e-9) * t
    dn = P.REF * (1 - pct); t = tick(dn); LD = np.ceil(dn / t - 1e-9) * t
    return LU, LD


def lock_buy(o, lu, lo, crit):
    """開盤鎖漲停而買不到？（純量或陣列）"""
    with np.errstate(invalid='ignore'):
        a = o >= lu - 1e-9
        if crit == 'A':
            return a
        if crit == 'NONE':
            return np.zeros_like(a, dtype=bool) if isinstance(a, np.ndarray) else False
        return a & ~(np.isfinite(lo) & (lo < lu - 1e-9))


def lock_sell(o, ld, hi, crit):
    with np.errstate(invalid='ignore'):
        a = o <= ld + 1e-9
        if crit == 'A':
            return a
        if crit == 'NONE':
            return np.zeros_like(a, dtype=bool) if isinstance(a, np.ndarray) else False
        return a & ~(np.isfinite(hi) & (hi > ld + 1e-9))


_EDITS = [
    ("def simulate_s10(", "def simulate_s97("),
    ("exec_mode='open', trail=None, trail_daily=True, requal=None):",
     "exec_mode='open', trail=None, trail_daily=True, requal=None, lim=None):"),
    # 事件紀錄、判定函式、收盤買進
    ("    def px_of(t, k):\n", '''    lim_ev = []; st['lim_ev'] = lim_ev; pend = {}; pend_close = []
    _crit = lim['crit'] if lim else None; _buy = lim['buy'] if lim else 'i'

    def buy_locked(t, j, o):
        if lim is None:
            return o >= P.LIMUP[t, j] - 1e-9
        return bool(_s97.lock_buy(o, lim['LU'][t, j], lim['LOW'][t, j], _crit))

    def sell_locked(t, j, o):
        if lim is None:
            return o <= P.LIMDN[t, j] + 1e-9
        return bool(_s97.lock_sell(o, lim['LD'][t, j], lim['HIGH'][t, j], _crit))

    def close_buys(t, items, pc):
        """處理 (iii)：執行日收盤價買進（持股、0050 已照收盤估值）。逐行沿用 buys() 的尺寸與成本公式，價格換成原始收盤。"""
        nonlocal cash
        free = slots - len(pos)
        e50_liq = max(v50 * (1 - 0.5 * cm * etf_tick(pc) / pc) * (1 - fee - TAX_ETF * cm) - (minfee if v50 > 0 else 0.0), 0.0)
        cash_s = cash
        eq = cash + v50 + sum(p['v'] for p in pos.values())
        u = max(eq - B, 0.0) / 10.0
        for j, ks in items:
            target = u
            c = CR[t, j]
            if free <= 0 or j in pos or not (np.isfinite(c) and c > 0) or c >= lim['LU'][t, j] - 1e-9:
                lim_ev.append((int(t), int(j), int(ks), 'close_fail_limup' if (np.isfinite(c) and c >= lim['LU'][t, j] - 1e-9) else 'close_fail_other')); continue
            sl = 0.5 * cm * float(tick(c)) / c + xs
            px = c * (1 + sl)
            avail = max(cash_s, 0.0) + e50_liq
            budget = min(target, avail)
            sh = int(budget / (px * (1 + fee) + (c * ox if t >= odd_t else 0.0)))
            if t < odd_t:
                sh = (sh // 1000) * 1000
            if sh > 0 and cap_pct and np.isfinite(AMT[t, j]):
                mx = int(cap_pct * AMT[t, j] / c)
                if t < odd_t:
                    mx = (mx // 1000) * 1000
                sh = min(sh, mx)
            f = _fee(sh, px, fee, minfee) if sh > 0 else 0.0; oc = (sh % 1000) * c * ox
            while sh > 0 and sh * px + f + oc > avail + 1e-6:
                sh -= 1000 if t < odd_t else 1
                f = _fee(sh, px, fee, minfee) if sh > 0 else 0.0; oc = (sh % 1000) * c * ox
            if sh <= 0 or (min_frac > 0 and sh * c < min_frac * target):
                lim_ev.append((int(t), int(j), int(ks), 'close_fail_other')); continue
            cst = sh * px + f + oc
            if cash_s < cst:
                e50_liq -= (cst - max(cash_s, 0.0))
            cash -= cst; cash_s -= cst
            st['buy_val'] += sh * c; st['fee'] += f; st['slip'] += sh * c * sl + oc; st['n_buy'] += 1
            pos[j] = dict(v=sh * c, sh=sh, sh0=sh, e=t, sig=ks, cost=cst, flag=False, why='', proc=0.0, new=False, g=1.0, gpk=1.0, nm=0)
            lim_ev.append((int(t), int(j), int(ks), 'close_fill'))
            free -= 1

    def px_of(t, k):
'''),
    # 換股日減碼：鎖跌停跳過
    ("                if not (np.isfinite(o) and o > 0) or o <= P.LIMDN[t, k] + 1e-9:\n",
     "                if not (np.isfinite(o) and o > 0) or sell_locked(t, k, o):\n"),
    # buys：可指定訊號列（重試用）
    ("    def buys(t, cand, po):\n        nonlocal cash\n", "    def buys(t, cand, po, ks=None, retry=False):\n        nonlocal cash\n"),
    ("        rk = rank[t - 1, cand]\n", "        rk = rank[(t - 1) if ks is None else ks, cand]\n"),
    ("        k = t - 1\n        div =", "        k = (t - 1) if ks is None else ks\n        div ="),
    ("            if o >= P.LIMUP[t, j] - 1e-9:\n                st['blk_limup'] += 1; continue\n",
     """            if buy_locked(t, j, o):
                st['blk_limup'] += 1
                lim_ev.append((int(t), int(j), int(k), 'retry_fail_locked' if retry else 'buy_locked'))
                if not retry and _buy == 'ii' and t + 1 < T:
                    pend.setdefault(t + 1, []).append((j, k))
                elif not retry and _buy == 'iii':
                    pend_close.append((j, k))
                continue
"""),
    ("            pos[j] = dict(v=sh * o, sh=sh, sh0=sh, e=t, sig=t - 1, cost=cst,",
     "            if retry:\n                lim_ev.append((int(t), int(j), int(k), 'retry_fill'))\n            pos[j] = dict(v=sh * o, sh=sh, sh0=sh, e=t, sig=k, cost=cst,"),
    # 賣出：鎖跌停順延
    ("            if o <= P.LIMDN[t, j] + 1e-9:\n                st['sell_defer_limdn'] += 1; continue\n",
     "            if sell_locked(t, j, o):\n                st['sell_defer_limdn'] += 1; lim_ev.append((int(t), int(j), int(p['sig']), 'sell_locked')); continue\n"),
    # 主迴圈：重試買進（ii）
    ("        if cand:\n            buys(t, cand, po_o)\n        e50_net(po_o)\n",
     """        if cand:
            buys(t, cand, po_o)
        if t in pend:
            for j_, ks_ in pend.pop(t):
                if j_ in pos or j_ in sold_today:
                    lim_ev.append((int(t), int(j_), int(ks_), 'retry_fail_other')); continue
                n0_ = len(pos)
                buys(t, [j_], po_o, ks=ks_, retry=True)
                if len(pos) == n0_ and not (lim_ev and lim_ev[-1][:2] == (int(t), int(j_)) and lim_ev[-1][3] == 'retry_fail_locked'):
                    lim_ev.append((int(t), int(j_), int(ks_), 'retry_fail_other'))
        e50_net(po_o)
"""),
    # 收盤買進（iii）：持股與 0050 已走到收盤之後、算淨值之前
    ("        tot = 0.0\n        is_sig = bool(sigrows[t])\n",
     """        if pend_close:
            pc_ = P.E50C_raw[t] if np.isfinite(P.E50C_raw[t]) and P.E50C_raw[t] > 0 else 100.0
            close_buys(t, list(pend_close), pc_); pend_close.clear()
            e50_net(pc_)
        tot = 0.0
        is_sig = bool(sigrows[t])
"""),
]


def build(s10lib):
    """回傳 simulate_s97（在 s10lib 名稱空間；原函式不動）與產生的原始碼。"""
    import sys
    src = inspect.getsource(s10lib.simulate_s10)
    for a, b in _EDITS:
        assert src.count(a) == 1, f'取代點必須剛好一次：{a[:60]!r} 出現 {src.count(a)} 次'
        src = src.replace(a, b)
    ns = dict(vars(s10lib)); ns['_s97'] = sys.modules[__name__]
    exec(compile(src, '<simulate_s97>', 'exec'), ns)
    return ns['simulate_s97'], src
