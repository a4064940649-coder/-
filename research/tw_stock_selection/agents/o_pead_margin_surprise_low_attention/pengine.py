#!/usr/bin/env python3
"""日曆時間「槽位」組合引擎（本題自寫；成本與成交規則照 V7X 規格書的 K5T 與四項修正精神簡化）。
- 每個事件一個槽，S 槽等額；空槽與閒置資金放 0050（買賣 0050 付手續費，賣出付 0.1% 證交稅）。
- 進場：事件的實際進場列（已處理開盤漲停順延與處置放棄）開盤買；同日多個候選依 SUE⊥ 由大到小，槽滿不買。
- 整股：張數＝floor(目標金額 /(1000×開盤))，參與上限 floor(0.2×OA_pred/(1000×前收))，OA_pred＝0.03×V60_{t−1}；
  衝擊 K5T＝max(0.001＋0.5·σ60·√(Q/OA_actual), 一跳/開盤)，OA_actual＝0.03×當日成交值。
- 零股（剩餘金額）：2020-10-26 起盤中零股，開盤價×(1＋c)，每日上限 zcap×V60/價（zcap＝0.000193）；
  之前為盤後零股，以收盤×(1＋c) 成交，上限 0.5×zcap×V60/收盤；c 依價格桶（V7 odd 表，名次 >150 欄）。
- 手續費 0.1425%×0.28（2015 起），每筆最低 20 元；賣出證交稅 0.3%。
- 壓力：每邊再加 0.4%，手續費 6 折，零股價差×2。
- 出場：到期日開盤賣；開盤 ≤ 跌停或無開盤 → 順延；資料結束仍未賣 → 以最後收盤計值。
- GROW：槽目標＝前一日淨值÷S；FIX：槽目標＝起始資金÷S（不隨淨值成長），其餘放 0050。"""
import numpy as np, pandas as pd

ODD_PRICE_EDGES = np.array([50.0, 100.0, 300.0])
ODD_BUY = np.array([0.0028308577519022826, 0.00390625, 0.0052701013748091, 0.004546362737566589])   # 名次 >150 欄
ODD_SELL = np.array([0.0, -0.0012345848365546974, -0.001776198934280604, -0.001953125])
ODD_SWITCH = np.datetime64('2020-10-26')
ZCAP = 0.0001928525621986513
OA_SHARE = 0.03


def tick(p):
    return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)


class Market:
    def __init__(self, P, e50):
        self.IDX = pd.DatetimeIndex(P['IDX']); self.T = len(self.IDX)
        self.O, self.C, self.V, self.R = P['O'], P['C'], P['V'], P['R']
        self.G = np.where(np.isfinite(self.R), 1 + self.R, 1.0)
        self.V60, self.SIG, self.LIM = P['V60'], P['SIG'], P['LIMUP']
        self.CF = pd.DataFrame(self.C).ffill().to_numpy()
        with np.errstate(invalid='ignore', divide='ignore'):
            ref = self.C / self.G
        ref = np.where(np.isfinite(ref), ref, self.CF)
        self.LIMDN = np.ceil(ref * 0.9 / tick(ref) - 1e-9) * tick(ref)
        self.e50 = np.asarray(e50, float)
        self.r50 = np.r_[0.0, self.e50[1:] / self.e50[:-1] - 1]
        self.r50 = np.where(np.isfinite(self.r50), self.r50, 0.0)
        self.d64 = self.IDX.values


def run(M, ev_rows, S=20, H=60, K=1e6, mode='GROW', cost='normal', start=None, end=None):
    """ev_rows: DataFrame 需含 col（面板欄）、e（實際進場列）、score（同日排序用）。回傳每日淨值 Series 與交易統計。"""
    T = M.T
    t0 = M.IDX.searchsorted(pd.Timestamp(start)) if start else 0
    t1 = M.IDX.searchsorted(pd.Timestamp(end), side='right') - 1 if end else T - 1
    fee_rate = 0.001425 * (0.6 if cost == 'stress' else 0.28)
    extra = 0.004 if cost == 'stress' else 0.0
    oddmul = 2.0 if cost == 'stress' else 1.0
    by_day = {}
    ev = ev_rows[(ev_rows['e'] >= t0) & (ev_rows['e'] <= t1)].sort_values(['e', 'score'], ascending=[True, False])
    for e, col, sc in zip(ev['e'].to_numpy(), ev['col'].to_numpy(), ev['score'].to_numpy()):
        by_day.setdefault(int(e), []).append(int(col))
    B = K                     # 0050 部位市值
    pos = []                  # dict(col, val, due, e)
    nav = np.full(T, np.nan)
    stats = dict(buys=0, sells=0, fee=0.0, tax=0.0, imp=0.0, part=[], filled_ratio=[], skipped_full=0, skipped_cash=0, odd_share=[])

    def fee(x):
        return max(20.0, fee_rate * x) if x > 0 else 0.0

    prev_nav = K
    for t in range(t0, t1 + 1):
        # 1. 0050 收盤漂移（前收→今收；交易在開盤，以前收近似）
        g50 = 1 + M.r50[t] if t > t0 else 1.0
        # 2. 到期賣出（開盤）
        keep = []
        cash = 0.0
        for p in pos:
            j = p['col']
            if t >= p['due']:
                o = M.O[t, j]; c = M.C[t, j]
                can = np.isfinite(o) and o > 0 and o > M.LIMDN[t, j] + 1e-9
                if can:
                    gx = o * M.G[t, j] / c if (np.isfinite(c) and c > 0) else o / M.CF[t - 1, j]
                    gross = p['val'] * gx
                    q_oa = OA_SHARE * M.V[t, j] if np.isfinite(M.V[t, j]) else np.nan
                    sig = M.SIG[t - 1, j] if np.isfinite(M.SIG[t - 1, j]) else 0.03
                    imp = max(0.001 + 0.5 * sig * np.sqrt(gross / q_oa), tick(o) / o) if (np.isfinite(q_oa) and q_oa > 0) else 0.01
                    imp = min(imp, 0.1) + extra
                    f = fee(gross); tx = 0.003 * gross
                    net = gross * (1 - imp) - f - tx
                    stats['sells'] += 1; stats['fee'] += f; stats['tax'] += tx; stats['imp'] += gross * imp
                    cash += net
                    continue
                if t == t1:
                    pass
            keep.append(p)
        pos = keep
        # 3. 進場（開盤）
        B = B * 1.0  # 0050 價值在今日收盤才漂移
        cands = by_day.get(t, [])
        held = {p['col'] for p in pos}
        for j in cands:
            if len(pos) >= S:
                stats['skipped_full'] += 1
                continue
            if j in held:
                continue
            o = M.O[t, j]
            if not (np.isfinite(o) and o > 0):
                continue
            target = (prev_nav / S) if mode == 'GROW' else (K / S)
            avail = B + cash
            if avail <= target * 0.2:
                stats['skipped_cash'] += 1
                continue
            target = min(target, avail * 0.995)
            pc = M.CF[t - 1, j]
            v60 = M.V60[t - 1, j] if np.isfinite(M.V60[t - 1, j]) else 0.0
            cap_lots = int(np.floor(0.2 * OA_SHARE * v60 / (1000 * pc))) if pc > 0 else 0
            lots = int(min(np.floor(target / (1000 * o)), cap_lots))
            amt_w = lots * 1000 * o
            # 零股
            rem = target - amt_w
            odd_amt = 0.0; odd_close = False
            if rem > 0:
                if M.d64[t] >= ODD_SWITCH:
                    cap = ZCAP * v60
                    px = o
                else:
                    cap = 0.5 * ZCAP * v60
                    px = M.C[t, j] if np.isfinite(M.C[t, j]) else np.nan
                    odd_close = True
                if np.isfinite(px) and px > 0:
                    sh = int(min(np.floor(rem / px), np.floor(cap / px)))
                    odd_amt = sh * px
            if amt_w + odd_amt <= 0:
                stats['filled_ratio'].append(0.0)
                continue
            q_oa = OA_SHARE * M.V[t, j] if np.isfinite(M.V[t, j]) else np.nan
            sig = M.SIG[t - 1, j] if np.isfinite(M.SIG[t - 1, j]) else 0.03
            imp = (max(0.001 + 0.5 * sig * np.sqrt(amt_w / q_oa), tick(o) / o) if (np.isfinite(q_oa) and q_oa > 0) else 0.01) if amt_w > 0 else 0.0
            imp = min(imp, 0.1) + (extra if amt_w > 0 else 0)
            odd_px = M.C[t, j] if odd_close else o
            bi = int(np.searchsorted(ODD_PRICE_EDGES, odd_px, side='right')) if (odd_amt > 0 and np.isfinite(odd_px)) else 0
            c_odd = ODD_BUY[bi] * oddmul + (extra if odd_amt > 0 else 0)
            f = fee(amt_w) + fee(odd_amt)
            spend = amt_w * (1 + imp) + odd_amt * (1 + c_odd) + f
            # 資金：先用今日賣出現金，再賣 0050
            use_cash = min(cash, spend); cash -= use_cash
            need = spend - use_cash
            if need > 0:
                f50 = max(20.0, fee_rate * need) + 0.001 * need
                B -= need + f50
                stats['fee'] += f50
            # 部位市值：整股從開盤到收盤；零股（盤後）從收盤起算
            c = M.C[t, j]
            val_close = amt_w * (c / o if np.isfinite(c) else 1.0) + odd_amt
            pos.append(dict(col=j, val=val_close, due=t + H, e=t, odd_close=odd_close))
            held.add(j)
            stats['buys'] += 1; stats['fee'] += f; stats['imp'] += amt_w * imp + odd_amt * c_odd
            stats['part'].append(amt_w / q_oa if (np.isfinite(q_oa) and q_oa > 0) else np.nan)
            stats['filled_ratio'].append((amt_w + odd_amt) / target)
            stats['odd_share'].append(odd_amt / (amt_w + odd_amt))
        # 剩餘現金買回 0050
        if cash > 0:
            f50 = max(20.0, fee_rate * cash)
            B += cash - f50; stats['fee'] += f50; cash = 0.0
        # 4. 收盤計值
        B *= g50
        for p in pos:
            if p['e'] != t:
                p['val'] *= M.G[t, p['col']]
        nav[t] = B + sum(p['val'] for p in pos)
        prev_nav = nav[t]
    s = pd.Series(nav[t0:t1 + 1], index=M.IDX[t0:t1 + 1])
    return s, stats


def monthly(s):
    return s.resample('ME').last().pct_change().dropna()


def perf(nav, e50s, start=None, end=None):
    """nav、e50s：日頻 Series（同索引）。回傳年化、對 0050 月超額 t、最大回撤等。"""
    if start is not None:
        nav = nav[nav.index >= pd.Timestamp(start)]; e50s = e50s[e50s.index >= pd.Timestamp(start)]
    if end is not None:
        nav = nav[nav.index <= pd.Timestamp(end)]; e50s = e50s[e50s.index <= pd.Timestamp(end)]
    if len(nav) < 30:
        return None
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1
    b = (e50s.iloc[-1] / e50s.iloc[0]) ** (1 / yrs) - 1
    mn = monthly(nav); mb = monthly(e50s).reindex(mn.index)
    ex = (mn - mb).dropna()
    t = ex.mean() / (ex.std(ddof=1) / np.sqrt(len(ex))) if len(ex) > 2 and ex.std() > 0 else np.nan
    dd = (nav / nav.cummax() - 1).min()
    return dict(cagr=float(cagr), b50=float(b), ex_ann=float(cagr - b), t_m=float(t), mex=float(ex.mean()), n_m=int(len(ex)),
                mdd=float(dd), sd_m=float(ex.std(ddof=1)))
