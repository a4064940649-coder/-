#!/usr/bin/env python3
"""處置腿（10 槽）日頻模擬：沿用 SOP 處置股策略（處置第一天收盤買、最後一天收盤賣；10 槽；單筆 ≤ 當日成交值 5%；
加權指數前一日收盤 < 200 日均線不買；進場日收盤漲幅 ≥ 9% 不買；已持有不加碼；出場日收盤跌幅 ≤ −9% 延到第一個非跌停日；
流動性 V60 ≥ 1,000 萬、股價 ≥ 10；未用資金放 0050）。事件表＝disposal_5y（2021-09 起，上市＋上櫃）。
成本：一般來回 0.8%（買 0.25%、賣 0.55%，同 disposal_reconcile 口徑）；壓力來回 2%。0050 進出另計 0.15%（賣出）。
上櫃價格洞（2023-01～09 多數上櫃股無價格）內的事件無法成交 → 剔除並計數。"""
import numpy as np, pandas as pd
import otlib as L
D = L.D

_ds = pd.read_csv(L.EXT / 'extras' / 'disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['date', 'start', 'end'])
_ds = _ds[_ds.code.isin(D.CI)].sort_values(['start', 'code']).drop_duplicates(['code', 'start'])
_tx = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date']).set_index('date')['close']
BULL = (_tx >= _tx.rolling(200).mean()).reindex(D.IDX).ffill().shift(1).fillna(False).to_numpy().astype(bool)
AIf = pd.DataFrame(np.where(D.TR, D.AI, np.nan)).ffill().to_numpy()


def events(markets=('TWSE', 'TPEX'), first_only=False):
    rows = []; prev_end = {}
    for c, s, e in zip(_ds.code, _ds.start, _ds.end):
        j = D.CI[c]
        i0 = int(np.searchsorted(D.DATES, np.datetime64(s))); i1 = int(np.searchsorted(D.DATES, np.datetime64(e), side='right')) - 1
        pe = prev_end.get(c); second = pe is not None and 0 <= i0 - pe <= 30
        prev_end[c] = max(i1, pe or -1)
        if i0 >= len(D.IDX) or i1 <= i0: continue
        mk = 'TPEX' if D.MK[i0, j] else 'TWSE'
        if mk not in markets or (first_only and second): continue
        rows.append(dict(code=c, j=j, i0=i0, i1=i1, mkt=mk, second=second))
    return pd.DataFrame(rows)


def sleeve(ev, a, b, cost=(0.0025, 0.0055), etf_cost=0.0015, cap_val=0.05, equity0=2e5):
    """回傳（日淨值, 交易明細）。a、b：模擬起訖日（Timestamp）。equity0：處置部位資金（2026 幣值，用於 5% 成交值上限）。"""
    ta = int(np.searchsorted(D.IDX, a)); tb = int(np.searchsorted(D.IDX, b, side='right')) - 1
    ev = ev[(ev.i0 >= ta) & (ev.i0 <= tb)]
    byday = {}
    for r in ev.itertuples(): byday.setdefault(r.i0, []).append(r)
    cash50 = equity0 * D.cpi_scale(D.IDX[ta].year)   # 以 0050 計值的閒置資金
    pos = {}   # j -> [value, exit_idx, code, mkt, entry_value]
    nav = np.empty(tb - ta + 1); trades = []; skipped = {'no_price': 0, 'bear': 0, 'limitup': 0, 'illiquid': 0, 'full': 0, 'held': 0}
    for t in range(ta, tb + 1):
        if t > ta:
            r50 = D.E50[t] / D.E50[t - 1]
            cash50 *= r50
            for j, p in pos.items():
                g = AIf[t, j] / AIf[t - 1, j]
                p[0] *= g if np.isfinite(g) else 1.0
        # 出場（收盤）
        for j in list(pos):
            p = pos[j]
            if t >= p[1]:
                if np.isfinite(D.R[t, j]) and D.R[t, j] <= -0.09 and D.TR[t, j]:
                    continue                 # 跌停延後
                if not D.TR[t, j]: continue
                v = p[0] * (1 - cost[1]); cash50 += v
                trades.append(dict(code=p[2], mkt=p[3], entry=str(D.IDX[p[5]].date()), exit=str(D.IDX[t].date()), ret=v / p[4] - 1, second=p[6]))
                del pos[j]
        # 進場（收盤）
        for r in byday.get(t, []):
            j = r.j
            if j in pos: skipped['held'] += 1; continue
            if not (D.TR[t, j] and np.isfinite(D.C[t, j])): skipped['no_price'] += 1; continue
            if not BULL[t]: skipped['bear'] += 1; continue
            if np.isfinite(D.R[t, j]) and D.R[t, j] >= 0.09: skipped['limitup'] += 1; continue
            v60 = D.V60p[t, j]
            if not (np.isfinite(v60) and v60 >= 1e7 and D.CF[t - 1, j] >= 10): skipped['illiquid'] += 1; continue
            if len(pos) >= 10: skipped['full'] += 1; continue
            eq = cash50 + sum(p[0] for p in pos.values())
            amt = min(eq / 10, cap_val * np.nan_to_num(D.V[t, j]), cash50 / (1 + etf_cost))
            if amt <= 0: continue
            cash50 -= amt * (1 + etf_cost)
            val = amt * (1 - cost[0])
            pos[j] = [val, r.i1, r.code, r.mkt, amt, t, r.second]
        nav[t - ta] = cash50 + sum(p[0] for p in pos.values())
    return pd.Series(nav / nav[0], index=D.IDX[ta:tb + 1]), pd.DataFrame(trades), skipped
