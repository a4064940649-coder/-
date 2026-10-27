#!/usr/bin/env python3
"""反方審查（成本與可成交）：只讀 import ocore，不寫任何研究員檔案；結果只印到 stdout。
A. w0／w30 的周轉、成本拖累（零成本 vs 一般 vs 壓力），台積電腿每月調整的交易大小
B. 加滑價情境：+0.2%、+0.4%、+0.6% 每邊，HOLD 對 0050、ALL t
C. 最低手續費 20 元（整股券商常見）：資金 30 萬、100 萬時的額外拖累
D. 零股需求：每檔分配金額 < 1 張價格的比例；2020-10-26 盤中零股開放前的比例
E. 訊號層級「各組前 1/5 減組內等權」扣成本後（周轉 × 來回成本）
F. HOLD 期間成交日開盤價來源（src）"""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import ocore as O
K = O.K
S = O.V2R(1, 300)
pl = S.plan(10); t0, t1, sd = S.bounds(); P = O.periods(sd)
ew, _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, t0, t1)
KEYS = ('SEL', 'VAL', 'HOLD', 'ALL')
def st(nav): return {k: O.segstat(nav, ew, sd, *P[k]) for k in KEYS}
def show(lab, s): print(lab, {k: (s[k]['CAGR'], s[k]['t_0050'], s[k]['E50_CAGR']) for k in KEYS}, flush=True)

# ---- A. 周轉與成本拖累 ----
print('== A ==')
res = {}
for w in (0.0, 0.3):
    for cl, c in (('zero', (0.0, 0.0)), ('normal', O.NORMAL), ('stress', O.STRESS)):
        nv, lg = O.simulate_w(O.tsmc_plan(pl, w), *c, t0, t1)
        res[(w, cl)] = st(nv)
        print(f'w{w} {cl} turnover/月 {lg["turnover"]:.3f}', {k: (res[(w, cl)][k]['CAGR'], res[(w, cl)][k]['t_0050']) for k in KEYS}, flush=True)
    for k in KEYS:
        print(f'  w{w} {k} 成本拖累 一般 {res[(w,"zero")][k]["CAGR"]-res[(w,"normal")][k]["CAGR"]:.2f}pp 壓力 {res[(w,"zero")][k]["CAGR"]-res[(w,"stress")][k]["CAGR"]:.2f}pp')

# 交易明細引擎（simulate_w 的複本，只多記錄每次換股的每檔買賣金額／當時總值）
def simulate_trades(plan, buy, sell, t0, t1):
    reb = {p[0]: (np.asarray(p[1], int), np.asarray(p[2], float)) for p in plan}
    val = np.zeros(N); cash = 1.0; trades = []
    TR, On, Cn, G = K.TRADED, K.On, K.Cn, K.GROW
    for t in range(t0, t1 + 1):
        if t in reb:
            pk, w = reb[t]
            tr = TR[t] & np.isfinite(On[t])
            gap = np.where(tr, G[t] * On[t] / np.where(tr, Cn[t], 1.0), 1.0)
            h = val * gap; frz = (h > 0) & ~tr
            htr = np.where(frz, 0.0, h); Vt = htr.sum() + cash
            wv = np.zeros(N); np.add.at(wv, pk, w)
            def cost_of(T):
                tgt = wv * T
                return buy * np.clip(tgt - htr, 0, None).sum() + sell * np.clip(htr - tgt, 0, None).sum()
            lo, hi = 0.0, Vt
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if mid + cost_of(mid) > Vt: hi = mid
                else: lo = mid
            tgt = wv * lo
            dlt = tgt - htr
            trades.append((t, dlt / (Vt + h[frz].sum()), On[t].copy()))
            newval = np.where(frz, h, 0.0) + tgt
            oc = np.where(tr, Cn[t] / np.where(tr, On[t], 1.0), 1.0)
            val = np.where(frz, newval, newval * oc); cash = 0.0
        else:
            val = val * G[t] if t > t0 else val
    return trades
N = O.N
tw = {w: simulate_trades(O.tsmc_plan(pl, w), *O.NORMAL, t0, t1) for w in (0.0, 0.3)}
j = O.J2330
x = pd.Series([abs(d[j]) for t, d, _ in tw[0.3]], index=[O.IDX[t] for t, _, _ in tw[0.3]])
print('台積電腿每次換股調整金額（占總值）中位數 %.2f%% 平均 %.2f%% p90 %.2f%%' % (x.median() * 100, x.mean() * 100, x.quantile(0.9) * 100))

# ---- C. 最低手續費 20 元 ----
print('== C. 最低手續費 20 元（資金固定為 C，每月）==')
for C in (3e5, 1e6, 1e7):
    for w in (0.0, 0.3):
        extra = []; ntr = []
        for t, d, _ in tw[w]:
            amt = np.abs(d[np.abs(d) > 1e-6]) * C
            fee_norm = 0.000855 * amt
            extra.append(np.clip(20 - fee_norm, 0, None).sum() / C); ntr.append(len(amt))
        e = pd.Series(extra, index=[O.IDX[t] for t, _, _ in tw[w]])
        print(f'C={C/1e4:.0f}萬 w{w}: 每月交易筆數 {np.mean(ntr):.1f}，最低手續費額外拖累 每月 {e.mean()*100:.3f}% ≈ 年 {e.mean()*1200:.2f}pp；HOLD 年 {e[e.index>="2024-01-01"].mean()*1200:.2f}pp')

# ---- B. 加滑價情境 ----
print('== B. 每邊額外滑價 ==')
for extra in (0.002, 0.004, 0.006):
    b, s = O.NORMAL[0] + extra, O.NORMAL[1] + extra
    for w in (0.0, 0.3):
        nv, _ = O.simulate_w(O.tsmc_plan(pl, w), b, s, t0, t1)
        show(f'+{extra*100:.1f}% w{w}', st(nv))

# ---- D. 零股需求 ----
print('== D. 零股需求 ==')
On = K.On
for C in (3e5, 1e6, 1e7):
    for w in (0.0, 0.3):
        alloc = C * (1 - w) / 10
        r = []
        for p in pl:
            t = p[0]; px = On[t, np.asarray(p[1], int)]
            r.append((O.IDX[t], float(np.mean(px * 1000 > alloc)), float(np.mean(px * 1000 > 2 * alloc))))
        df = pd.DataFrame(r, columns=['d', 'need_odd', 'lot_gt_2x']).set_index('d')
        pre = df[df.index < '2020-10-26']; post = df[df.index >= '2020-10-26']
        print(f'C={C/1e4:.0f}萬 w{w} 每檔 {alloc/1e4:.1f} 萬：需零股比例 全期 {df.need_odd.mean():.2f}（2020-10-26 前 {pre.need_odd.mean():.2f}，後 {post.need_odd.mean():.2f}，HOLD {df[df.index>="2024-01-01"].need_odd.mean():.2f}）；整股四捨五入誤差>50% 的比例 {df.lot_gt_2x.mean():.2f}')
tlast = t1
print('2330 最後收盤', K.Cn[tlast, j], O.IDX[tlast].date(), '；30 萬×30% 可買股數', int(9e4 // K.Cn[tlast, j]))
print('換股日在 2020-10-26 前的比例', np.mean([O.IDX[p[0]] < pd.Timestamp('2020-10-26') for p in pl]).round(3))

# ---- E. 訊號層級扣成本 ----
print('== E. 各組前 1/5 減組內等權：扣成本 ==')
ex = [p[0] for p in pl]; sig = [p[3] for p in pl]
def oo(t, t2, idx):
    g = K.Cn[t, idx] / K.On[t, idx] * np.prod(K.GROW[t + 1:t2 + 1, idx], axis=0)
    ok = K.TRADED[t2, idx] & np.isfinite(K.On[t2, idx])
    return np.where(ok, g * K.On[t2, idx] / np.where(ok, K.Cn[t2, idx], 1.0), np.nan) - 1
prev = {b: set() for b in range(4)}; rows = []
for i, d in enumerate(sig[:-1]):
    cs = S.cross_section(d); el = np.where(cs['el'])[0]; bk = O.bucket_of(cs['rank'][el])
    for b in range(4):
        m = el[bk == b]
        if len(m) < 15: prev[b] = set(); continue
        scb = cs['sc'][m]; topq = m[scb >= np.quantile(scb, 0.8)]
        g = np.nanmean(oo(ex[i], ex[i + 1], topq)) - np.nanmean(oo(ex[i], ex[i + 1], m))
        tov = 1 - len(set(topq.tolist()) & prev[b]) / len(topq) if prev[b] else 1.0
        rows.append((d, b, g, tov)); prev[b] = set(topq.tolist())
E = pd.DataFrame(rows, columns=['d', 'b', 'g', 'tov'])
rt_n = O.NORMAL[0] + O.NORMAL[1]; rt_s = 0.01
for lab, lo, hi in (('SEL', '2006-01-01', '2019-04-10'), ('VAL', '2019-04-11', '2026-12-31'), ('HOLD', '2024-01-01', '2026-12-31')):
    for b in range(4):
        x = E[(E.b == b) & (E.d >= lo) & (E.d <= hi)]
        net_n = x.g - rt_n * x.tov; net_s = x.g - rt_s * x.tov
        print(f'{lab} {O.BNAME[b]}: gross {x.g.mean()*100:.2f}pp (t {K.tstat(x.g):.2f}) 周轉 {x.tov.mean():.2f} 一般成本後 {net_n.mean()*100:.2f}pp (t {K.tstat(net_n):.2f}) 壓力後 {net_s.mean()*100:.2f}pp (t {K.tstat(net_s):.2f})')

# ---- F. HOLD 期間開盤價來源 ----
print('== F. 成交日入選股的 src 分布 ==')
SRC = K.SRC.to_numpy()
for lab, lo, hi in (('VAL_A', '2019-04-11', '2023-12-31'), ('HOLD', '2024-01-01', '2026-12-31')):
    v = []
    for p in pl:
        dd = O.IDX[p[0]]
        if not (pd.Timestamp(lo) <= dd <= pd.Timestamp(hi)): continue
        v += list(SRC[p[0], np.asarray(p[1], int)])
    print(lab, pd.Series(v).value_counts(normalize=True).round(3).to_dict())
