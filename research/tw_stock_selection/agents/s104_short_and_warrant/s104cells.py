#!/usr/bin/env python3
"""S104 三題的格子定義與單路徑執行（K1／K2／K3）。只依賴 s104lib 的快取陣列。"""
import math
import numpy as np, pandas as pd
import s104lib as L

# ---------------------------------------------------------------- 格子（每題 ≤ 8；看到任何報酬前寫死）
K1_CELLS = {}
for ex in ('rebal', 'ma60'):
    for un in ('all', 'large150'):
        for fl in ('none', 'ret20neg'):
            K1_CELLS[f'K1|{ex}|{un}|{fl}'] = dict(exit=ex, univ=un, filt=fl)
K2_CELLS = {}
for thr in (2e7, 1e8):
    for w in (1, 5):
        for su in ('small', 'large150'):
            K2_CELLS[f'K2|v{int(thr/1e6)}M|slope{w}|short_{su}'] = dict(thr=thr, w=w, short_univ=su)
K3_CELLS = {f'K3|T{ten}|f{int(f*100)}': dict(tenor=ten, f=f) for ten in (90, 120) for f in (0.10, 0.20)}
K3_BASE = dict(ivm=1.35, spread=0.03)
K3_SENS = {f'iv{ivm}|spr{int(sp*100)}': dict(ivm=ivm, spread=sp) for ivm in (1.2, 1.35, 1.5) for sp in (0.02, 0.03, 0.04)}
BORROWS = (0.01, 0.02, 0.03)
K1_W = (0.3, 0.5)


def top_mc(d, t, n=150):
    u = np.flatnonzero(d.U[t] & np.isfinite(d.MC[t]))
    o = u[np.argsort(-d.MC[t, u], kind='stable')][:n]
    m = np.zeros(d.N, bool); m[o] = True
    return m


# ---------------------------------------------------------------- K1 營收負驚奇放空腿
def k1_run(d, cell, si, borrow=0.02, cost='strict', side=-1):
    c = K1_CELLS[cell] if isinstance(cell, str) else cell
    rows = {}
    for mi in range(d.KK.shape[1]):
        k = int(d.KK[si, mi])
        if k < 0 or k <= d.TA or k >= d.TE:
            continue
        s = d.SG[si, mi].astype(float)
        ok = d.U[k] & np.isfinite(s) & ~d.FMA60[k]
        if c['univ'] == 'large150':
            ok &= top_mc(d, k)
        if c['filt'] == 'ret20neg':
            ok &= d.RET20[k] < 0
        idx = np.flatnonzero(ok)
        rows[k] = idx[np.argsort(s[idx], kind='stable')][:10]
    keep = {k: set(int(j) for j in v) for k, v in rows.items()}

    def cand(t):
        return rows.get(t)

    def exitf(t, j, e):
        if t in rows:
            return 'rebal'
        if c['exit'] == 'ma60' and d.FMA60[t, j]:
            return 'ma60'
        return None
    nav, tr = L.sleeve(d, side, cand, exitf, d.TA, d.TE, slots=10, borrow=borrow, cost=cost, hold_to=keep)
    return nav, tr, dict(n_rebal=len(rows), avg_names=float(np.mean([len(v) for v in rows.values()])))


# ---------------------------------------------------------------- K2 小哥式雙向（布林 20,2 ＋ 20MA 斜率）
class K2Pre:
    """預先算好指標（還原收盤＝含息累積指數 G）。"""
    def __init__(self, d):
        G = pd.DataFrame(np.where(d.U | np.isfinite(d.O), d.G, np.nan))
        mid = G.rolling(20, min_periods=20).mean()
        sd = G.rolling(20, min_periods=20).std(ddof=0)
        self.G = G.to_numpy(); self.mid = mid.to_numpy(); sdv = sd.to_numpy()
        self.up = self.mid + 2 * sdv; self.lo = self.mid - 2 * sdv
        del sdv
        mcU = np.where(d.U & np.isfinite(d.MC), d.MC, np.nan)
        with np.errstate(invalid='ignore'):
            self.med = np.nanmedian(mcU, axis=1)
        self.small = d.U & np.isfinite(d.MC) & (d.MC < self.med[:, None])
        self.large = np.zeros_like(d.U)
        for t in range(d.T):
            if d.U[t].any():
                self.large[t] = top_mc(d, t)

    def slope(self, w):
        s = np.full_like(self.mid, np.nan)
        s[w:] = self.mid[w:] - self.mid[:-w]
        return s


def k2_run(d, pre, cell, off, borrow=0.02, cost='strict'):
    c = K2_CELLS[cell] if isinstance(cell, str) else cell
    sl = pre.slope(c['w'])
    with np.errstate(invalid='ignore'):
        liq = d.AMT20 >= c['thr']
        upx = (sl > 0) & (pre.G > pre.up)
        dnx = (sl < 0) & (pre.G < pre.lo)
        candL = d.U & pre.small & liq & upx
        su = pre.small if c['short_univ'] == 'small' else pre.large
        candS = d.U & su & liq & dnx
        exL = ~((sl > 0) & (pre.G >= pre.mid))          # 斜率翻負（或持平）或收盤跌回中軌下 → 出場；無值也出場
        exS = ~((sl < 0) & (pre.G <= pre.mid))
    G, up, lo = pre.G, pre.up, pre.lo

    def mk(cand, key):
        def f(t):
            idx = np.flatnonzero(cand[t])
            if not len(idx):
                return None
            return idx[np.argsort(-key(t, idx), kind='stable')]
        return f
    fL = mk(candL, lambda t, i: G[t, i] / up[t, i])
    fS = mk(candS, lambda t, i: lo[t, i] / G[t, i])
    t0 = d.TA + off
    navL, trL = L.sleeve(d, +1, fL, lambda t, j, e: 'x' if exL[t, j] else None, t0, d.TE, slots=10, cost=cost)
    navS, trS = L.sleeve(d, -1, fS, lambda t, j, e: 'x' if exS[t, j] else None, t0, d.TE, slots=10, borrow=borrow, cost=cost)
    del candL, candS, exL, exS, liq, upx, dnx
    return navL, trL, navS, trS


# ---------------------------------------------------------------- K3 合成權證
def k3_sleeve(d, trades, tenor, ivm, spread, t_start, t_end, slots=10):
    """trades：現行策略某路徑的逐筆（j, ent_t, exit_t）。每筆新買進同日開盤買一檔合成認購權證（價外 10%、tenor 日曆天），
    權利金＝前一日收盤套筒淨值 ÷ slots（同時最多 slots 檔；現金不足則用剩餘現金，< 一半不買）。
    隱含波動＝進場前 60 個交易日含息日報酬年化波動 × ivm（限 0.2～2.0）。買價＝BS 中價 ×(1＋價差/2)＋手續費；
    賣價＝BS 中價 ×(1−價差/2)−手續費−證交稅 0.1%；現行策略賣出該股當天開盤賣權證；到期前未賣則到期日以內含價值現金結算（不扣價差）。
    每日淨值用 BS 中價估值（持有中略偏樂觀）。回傳 nav、逐筆 DataFrame。"""
    O, CF, R, REF = d.O, d.CF, d.R, d.REF
    days = (d.IDX - d.IDX[0]).days.to_numpy()
    tr = trades[(trades['ent_t'] > t_start) & (trades['ent_t'] <= t_end)].sort_values(['ent_t', 'j'])
    byday = {}
    for row in tr.itertuples():
        byday.setdefault(int(row.ent_t), []).append((int(row.j), int(row.exit_t), row.why))
    cash = 1.0; pos = []; nav = np.full(d.T, np.nan); nav[t_start] = 1.0; log = []
    lr = np.log1p(np.where(np.isfinite(R), R, 0.0))
    for t in range(t_start + 1, t_end + 1):
        # 出場（到期或現行策略賣出日）
        still = []
        for p in pos:
            k = t - p['e']
            if t >= p['xt']:
                val = p['u'] * p['mark'][k]
                if p['kind'] == 'sell':
                    val *= (1 - spread / 2) * (1 - L.FEE - L.WTAX)
                cash += val; log.append((p['j'], p['e'], t, val / p['c'] - 1, p['kind']))
            else:
                still.append(p)
        pos = still
        # 進場
        if t in byday:
            eq = nav[t - 1]
            for j, x, why in byday[t]:
                if len(pos) >= slots:
                    break
                o = O[t, j]
                if not (np.isfinite(o) and o > 0):
                    continue
                a = max(0, t - 60)
                hv = lr[a:t, j]
                hv = hv[np.isfinite(hv)]
                sig = float(np.std(hv, ddof=1) * math.sqrt(252)) if len(hv) >= 20 else 0.5
                iv = min(max(sig * ivm, 0.2), 2.0)
                # 到期日＝第一個日曆天數 ≥ tenor 的交易日
                exp_t = int(np.searchsorted(days, days[t] + tenor, side='left'))
                xr = min(x if why != 'open_end' else t_end, exp_t, t_end)
                kind = 'sell' if (xr == x and why != 'open_end' and xr < exp_t) else ('expiry' if xr == exp_t else 'open_end')
                xr = max(xr, t)
                # 標的相對價格路徑（進場開盤＝1）：收盤值；出場日若為賣出用開盤值
                Sp = np.empty(xr - t + 1)
                Sp[0] = CF[t, j] / o
                rr = np.where(np.isfinite(R[t + 1:xr + 1, j]), R[t + 1:xr + 1, j], 0.0)
                Sp[1:] = Sp[0] * np.cumprod(1 + rr)
                if kind == 'sell' and xr > t and np.isfinite(O[xr, j]) and O[xr, j] > 0 and np.isfinite(REF[xr, j]) and REF[xr, j] > 0:
                    Sp[-1] = Sp[-2] * O[xr, j] / REF[xr, j]
                Tm = np.maximum(tenor - (days[t:xr + 1] - days[t]), 0) / 365.0
                if kind == 'expiry':
                    Tm[-1] = 0.0
                mark = L.bs_call(Sp, 1.10, Tm, iv)
                mid0 = float(L.bs_call(np.array([1.0]), 1.10, np.array([tenor / 365.0]), iv)[0])
                price = mid0 * (1 + spread / 2) * (1 + L.FEE)
                c = min(eq / slots, cash)
                if c < 0.5 * eq / slots or price <= 0:
                    continue
                cash -= c
                pos.append(dict(j=j, e=t, xt=xr, kind=kind, u=c / price, c=c, mark=mark, iv=iv, mid0=mid0))
        tot = cash + sum(p['u'] * p['mark'][t - p['e']] for p in pos)
        nav[t] = tot
    for p in pos:
        log.append((p['j'], p['e'], t_end, p['u'] * p['mark'][t_end - p['e']] / p['c'] - 1, 'open_end'))
    return nav, pd.DataFrame(log, columns=['j', 'ent_t', 'exit_t', 'ret', 'why'])


def blend_rebal(navA, navB, f, rebal, t_start, t_end):
    """(1−f)×A＋f×B，只在 rebal（交易日列號集合）收盤再平衡。"""
    A = np.asarray(navA, float); B = np.asarray(navB, float)
    out = np.full(len(A), np.nan); out[t_start] = 1.0
    wa, wb = (1 - f) / A[t_start], f / B[t_start]
    for t in range(t_start + 1, t_end + 1):
        out[t] = wa * A[t] + wb * B[t]
        if t in rebal:
            wa, wb = (1 - f) * out[t] / A[t], f * out[t] / B[t]
    return out
