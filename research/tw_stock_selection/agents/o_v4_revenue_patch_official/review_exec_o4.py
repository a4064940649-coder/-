#!/usr/bin/env python3
"""反方審查（成本與可成交）：只讀研究員的程式與資料，不寫入任何研究員檔案；結果只印 stdout。
用法：python3 -I -B review_exec_o4.py
檢查：
  [0] 重現 V4|normal 與 plan_V4.csv
  [1] 周轉與成本拖累（每月新買檔數、年化單邊周轉、成本 bp）
  [2] 容量：入選股 60 日均成交值、持股/ADV 參與率（30 萬、300 萬、1,000 萬）、整張價格 > 每檔部位（需零股）
  [3] 開盤漲停被擋（需 9:00 看盤補位）、開盤跌停鎖死賣不掉
  [4] 情境淨值：(a) 成交日收盤成交（零股 2020-10 前只能盤後收盤價；或不看開盤）
                (b) 參與率衝擊滑價 0.1% + 0.5·σ·sqrt(Q/ADV)（30 萬／300 萬／1,000 萬，Q＝資金/10 固定實質）
                (c) 開盤跌停鎖死者延到下一個非鎖死交易日開盤賣
  每個情境都報 V4（非 AB）與 AB（mc 40 seeds）的年化、對 0050 t、拿掉最好 1 個月 t。
"""
import sys, json, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import v4lib as L
K, V, R, O3 = L.K, L.V, L.R, L.O3
IDX, COLS, N = K.IDX, K.COLS, K.N
A_, B_ = '2006-04-01', '2026-10-12'

XR = L.setup_px()
YFX, Y5F = L.load_rfix()
DMT = L.disp_mask_fn('trade')
NV = pd.read_parquet(HERE / 'navs_v4.parquet')

def mk(**kw): return L.V4(YFX, Y5F, DMT, **kw)
S = mk(); t0, t1, sd = S.bounds(); pl = S.plan(10, 'official')
nav0, lg0 = L.sim_plan(pl, t0, t1, 'normal', XR)
ew0, _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, t0, t1)
d0 = float((nav0 / NV['V4|normal'].reindex(nav0.index) - 1).abs().max())
pc = pd.read_csv(HERE / 'plan_V4.csv', dtype=str)
same_plan = all(','.join(COLS[j] for j in p[1]) == pc.picks.iloc[i] for i, p in enumerate(pl))
print('[0] 重現 V4|normal 最大相對差', d0, '名單相同', same_plan, 'blocked(開盤漲停補位)', S.nblk)

def stat(nav, ew=ew0):
    s = L.seg(nav, ew, sd, A_, B_)
    x = L.excess(nav, K.E50.reindex(nav.index), sd, A_, B_)
    xs = x.sort_values(ascending=False)
    s4 = L.seg(nav, ew, sd, *L.PER['S4'])
    return dict(CAGR=round(s['CAGR'], 2), t=round(s['t_0050'], 3), t_drop1=round(L.tstat(x.drop(xs.index[:1])), 3),
                maxDD=round(s['maxDD'], 1), S4=round(s4['CAGR'], 1))

# ---------------- [1] 周轉 ----------------
held = set(); nnew = []; nsell = []
for p in pl:
    cur = set(p[1].tolist()); nnew.append(len(cur - held)); nsell.append(len(held - cur)); held = cur
yrs = (IDX[t1] - IDX[t0]).days / 365.25
tov_m = lg0['turnover']; reb_per_yr = len(pl) / yrs
rt_cost = 2 * K.FEE + 0.003 + 2 * 0.001
print('[1] 每月新買檔數 平均 %.2f／10；單邊周轉/月 %.3f；年換股次數 %.1f；年化單邊周轉 %.2f 倍；一般成本拖累約 %.2f%%/年（來回 %.3f%%）'
      % (np.mean(nnew[1:]), tov_m, reb_per_yr, tov_m * reb_per_yr, tov_m * reb_per_yr * rt_cost * 100, rt_cost * 100))
print('    每 +0.05%% 單邊滑價 ≈ −%.2f%%/年' % (tov_m * reb_per_yr * 2 * 0.0005 * 100))

# ---------------- [2] 容量 ----------------
_, V60 = V.rolling_signals('gapfix')
Rl = pd.DataFrame(K.Rn, index=IDX, columns=COLS)
SIG = Rl.rolling(60, min_periods=40).std().to_numpy()
rows = []
prev = set()
for p in pl:
    e, pk, _, d = p; t = IDX.get_loc(d)
    for j in pk:
        rows.append(dict(d=d, code=COLS[j], adv=V60[t, j], sig=SIG[t, j], opn=K.On[e, j], new=j not in prev))
    prev = set(pk.tolist())
P = pd.DataFrame(rows)
P['yr'] = P.d.dt.year
print('[2] 入選股 60 日均成交值（百萬元）分位：', {q: round(P.adv.quantile(q) / 1e6, 1) for q in (0.05, 0.1, 0.25, 0.5)},
      '；最低', round(P.adv.min() / 1e6, 1))
print('    分期中位數（百萬）', {k: round(g.adv.median() / 1e6, 0) for k, g in P.groupby(pd.cut(P.yr, [2005, 2013, 2020, 2023, 2026]))})
print('    入選股 60 日日報酬標準差 中位數 %.2f%%' % (np.nanmedian(P.sig) * 100))
for C in (3e5, 3e6, 1e7):
    q = C / 10; part = q / P.adv
    lot = (P.opn * 1000 > q).mean()
    print('    資金 %5.0f 萬：每檔 %.0f 萬；參與率中位 %.3f%%、P90 %.3f%%、最大 %.2f%%；> 1%% ADV 的股月 %.1f%%；整張價格 > 部位（必須零股）%.1f%% 股月'
          % (C / 1e4, q / 1e4, part.median() * 100, part.quantile(0.9) * 100, part.max() * 100, (part > 0.01).mean() * 100, lot * 100))
pre = P[P.d < pd.Timestamp('2020-10-26')]
print('    2020-10-26（盤中零股）前、30 萬需零股的股月比例 %.1f%%（當時零股只能盤後以收盤價撮合）' % ((pre.opn * 1000 > 3e4).mean() * 100))

# ---------------- [3] 開盤跌停鎖死的賣單 ----------------
def ld_price(ref, lim):
    raw = ref * (1 - lim); tk = K.tick(raw)
    return np.ceil(raw / tk - 1e-9) * tk
locked = []
prev = set()
for p in pl:
    e = p[0]; cur = set(p[1].tolist())
    lim = 0.07 if IDX[e] < pd.Timestamp('2015-06-01') else 0.10
    ref = K.ref_price(e)
    for j in prev - cur:
        o, h = K.On[e, j], K.Hn[e, j]
        if np.isfinite(o) and np.isfinite(ref[j]):
            ld = ld_price(ref[j], lim)
            if o <= ld + 1e-6:
                locked.append(dict(e=IDX[e], code=COLS[j], open_at_ld=True, locked=bool(np.isfinite(h) and h <= ld + 1e-6), j=j, t=e))
    prev = cur
LK = pd.DataFrame(locked)
nl = int(LK.locked.sum()) if len(LK) else 0
print('[3] 賣出日開盤即跌停：%d 筆，其中全日鎖死（最高＝跌停）%d 筆；開盤漲停被擋而改買下一名：%d 次' % (len(LK), nl, S.nblk))
if nl: print('    鎖死清單', LK[LK.locked][['e', 'code']].astype(str).values.tolist())

# ---------------- 自訂引擎：逐檔費率、鎖死延賣 ----------------
def simulate_vec(plan, rates_fn, xrules=XR, delay_locked=False):
    """O3.simulate 的複製，費率改為 rates_fn(t)→(buy_vec, sell_vec)。delay_locked：開盤跌停鎖死的待賣部位延到下一個非鎖死日開盤賣（期間持有）。"""
    G = K.GROW
    reb = {p[0]: p[1] for p in plan}
    rel = {j: (v[0], int(np.searchsorted(IDX, v[1])), int(np.searchsorted(IDX, v[3])) if len(v) > 3 else 10 ** 9) for j, v in xrules.items()}
    val = np.zeros(N); cash = 1.0; nav = np.empty(t1 - t0 + 1)
    pend = np.zeros(N, bool)          # 延賣部位
    def is_locked(t):
        lim = 0.07 if IDX[t] < pd.Timestamp('2015-06-01') else 0.10
        ref = K.ref_price(t); ld = ld_price(ref, lim)
        return np.isfinite(K.On[t]) & np.isfinite(ref) & (K.On[t] <= ld + 1e-6) & np.isfinite(K.Hn[t]) & (K.Hn[t] <= ld + 1e-6)
    for t in range(t0, t1 + 1):
        if t in reb or (delay_locked and pend.any() and t > t0):
            tr = K.TRADED[t] & np.isfinite(K.On[t])
            gap = np.where(tr, G[t] * K.On[t] / np.where(tr, K.Cn[t], 1.0), 1.0)
            h = val * gap
            frz = (h > 0) & ~tr
            for j in np.where(frz)[0]:
                if j in rel and rel[j][1] <= t < rel[j][2]:
                    cash += h[j] * rel[j][0]; h[j] = 0.0; frz[j] = False
            buy, sell = rates_fn(t)
            if t in reb:
                pk = reb[t]
                if delay_locked:
                    lk = is_locked(t) & (h > 0) & ~frz
                    lk[pk] = False
                    frz = frz | lk
                    pend = lk.copy()
                Vfz = h[frz].sum(); htr = np.where(frz, 0.0, h); Vt = htr.sum() + cash; k = len(pk)
                def cost_of(T):
                    tgt = np.zeros(N); tgt[pk] = T
                    return (buy * np.clip(tgt - htr, 0, None)).sum() + (sell * np.clip(htr - tgt, 0, None)).sum()
                lo, hi = 0.0, Vt / k
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    if mid * k + cost_of(mid) > Vt: hi = mid
                    else: lo = mid
                newval = np.where(frz, h, 0.0); newval[pk] += lo
                oc = np.where(tr, K.Cn[t] / np.where(tr, K.On[t], 1.0), 1.0)
                val = np.where(frz, newval, newval * oc); cash = 0.0
            else:   # 只處理延賣：非鎖死者開盤賣出 → 現金，下個換股日再投入（保守：期間現金）
                lk = is_locked(t)
                go = pend & tr & ~lk & (h > 0)
                cash += float((h[go] * (1 - sell[go])).sum())
                h[go] = 0.0; pend &= ~go
                oc = np.where(tr, K.Cn[t] / np.where(tr, K.On[t], 1.0), 1.0)
                val = np.where(tr, h * oc, h)
        else:
            val = val * G[t] if t > t0 else val
        nav[t - t0] = val.sum() + cash
    return pd.Series(nav, index=IDX[t0:t1 + 1])

FEEB, FEES = K.rates(0.001)
def flat_rates(t): return np.full(N, FEEB), np.full(N, FEES)
chk = simulate_vec(pl, flat_rates)
print('[4] 自訂引擎 vs V4|normal 最大相對差', float((chk / nav0 - 1).abs().max()))

def impact_rates(C, eta=0.5):
    q = C / 10
    def f(t):
        # 用前一交易日（≤t−1）的 V60 與 σ：換股成交日 t＝訊號日 +1
        a = V60[t - 1]; s = SIG[t - 1]
        imp = eta * np.nan_to_num(s, nan=0.03) * np.sqrt(q / np.where(np.isfinite(a) & (a > 0), a, 1e7))
        return K.FEE + 0.001 + imp, K.FEE + 0.003 + 0.001 + imp
    return f

def close_exec(fn):
    """把 K.On 換成 K.Cn：成交日以收盤價成交（gap_up 名單沿用原 plan）。"""
    On0 = K.On
    try:
        K.On = K.Cn.copy()
        return fn()
    finally:
        K.On = On0

res = {}
res['V4 一般成本（研究員）'] = stat(nav0)
res['V4 收盤成交'] = stat(close_exec(lambda: L.sim_plan(pl, t0, t1, 'normal', XR)[0]))
res['V4 跌停鎖死延賣'] = stat(simulate_vec(pl, flat_rates, delay_locked=True))
for C in (3e5, 3e6, 1e7, 3e7):
    res[f'V4 衝擊滑價 資金{int(C / 1e4)}萬'] = stat(simulate_vec(pl, impact_rates(C)))
for s_ in (0.0015, 0.002, 0.0025):
    res[f'V4 單邊滑價{s_ * 100:.2f}%'] = stat(L.sim_plan(pl, t0, t1, ('slip', s_), XR)[0])
for k, v in res.items(): print('   ', k, v)

# ---------------- AB（mc 40 seeds）各情境 ----------------
NSEED = 40
acc = {k: [] for k in ('AB 一般', 'AB 收盤成交', 'AB 衝擊 300萬', 'AB 衝擊 1000萬', 'AB 單邊0.15%', 'AB 單邊0.20%', 'AB 收盤+衝擊1000萬')}
ewr = []
for seed in range(NSEED):
    Sx = mk(unavail=V.make_unavail('mc', seed)); p2 = Sx.plan(10, 'official')
    acc['AB 一般'].append(L.sim_plan(p2, t0, t1, 'normal', XR)[0].pct_change())
    acc['AB 收盤成交'].append(close_exec(lambda: L.sim_plan(p2, t0, t1, 'normal', XR)[0]).pct_change())
    acc['AB 衝擊 300萬'].append(simulate_vec(p2, impact_rates(3e6)).pct_change())
    acc['AB 衝擊 1000萬'].append(simulate_vec(p2, impact_rates(1e7)).pct_change())
    acc['AB 單邊0.15%'].append(L.sim_plan(p2, t0, t1, ('slip', 0.0015), XR)[0].pct_change())
    acc['AB 單邊0.20%'].append(L.sim_plan(p2, t0, t1, ('slip', 0.002), XR)[0].pct_change())
    acc['AB 收盤+衝擊1000萬'].append(close_exec(lambda: simulate_vec(p2, impact_rates(1e7))).pct_change())
    ewr.append(K.simulate([(p[0], p[2]) for p in p2], 0, 0, t0, t1)[0].pct_change())
mn = lambda Lr: (1 + pd.concat(Lr, axis=1).mean(axis=1).fillna(0)).cumprod()
ewab = mn(ewr)
ab0 = mn(acc['AB 一般'])
print('    AB 一般 vs navs_v4 AB_V4|normal 最大相對差', float((ab0 / NV['AB_V4|normal'].reindex(ab0.index) - 1).abs().max()))
for k, Lr in acc.items(): print('   ', k, stat(mn(Lr), ewab))
