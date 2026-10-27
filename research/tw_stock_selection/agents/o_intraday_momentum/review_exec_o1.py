#!/usr/bin/env python3
"""反方審查（成本與可成交）o_intraday_momentum：P4b（買進改隔日收盤、賣出仍開盤）。只讀，不寫研究員的檔案，只印出結果。
1. 淨值層級的「收盤漲停買不到」：研究員的淨值假設收盤漲停照樣以收盤價成交；改成隔一日開盤補買（隔日開盤又漲停→隔日收盤）。
2. 收盤集合競價額外滑價（只加在收盤買進）：+0.1%、+0.2%、+0.3%、+0.5%，求損益兩平。
3. 收盤漲停改為「替補下一名」（收盤時才知道漲停）的規則。
4. 容量：新進股換股日成交值；1,000 萬／10 檔＝每檔 100 萬對當日成交值與收盤競價量的比例（5 分 K 推估收盤競價占比，粗估上限）。
5. 驗證期配對差集中度：逐年、剔除最好月份。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import olib as L
K, V = L.K, L.V
IDX, COLS, N = L.IDX, L.COLS, L.N
YL, Y5, _ = V.load_revenue()
S = V.V2('gapfix', YL, Y5, exfin=True)
pl = S.plan(10, 'official')
t0, t1, sd = S.bounds()
pp = [(p[0], p[1]) for p in pl]
nons3 = {p[0] for p in pl if not (L.SRC3[0] <= IDX[p[0]] <= L.SRC3[1])}
G = K.GROW

def lim_of(t): return 0.07 if IDX[t] < pd.Timestamp('2015-06-01') else 0.10
def luc(t):
    return np.isfinite(K.Cn[t]) & (K.Cn[t] >= K.limit_up_price(K.ref_price(t), lim_of(t)) - 1e-6)

def sim(plan, buy, sell, t0, t1, mode='p4b', extra=0.0, apply_days=None):
    """mode: open＝原版；p4b＝研究員版本（收盤漲停照樣成交）；p4b_lu＝收盤漲停改隔一日開盤（再漲停→隔日收盤）。
    extra：收盤買進額外滑價（乘在買進價）。其餘邏輯同 olib.simulate_exec。"""
    reb = {p[0]: p[1] for p in plan}
    val = np.zeros(N); cash = 1.0
    nav = np.empty(t1 - t0 + 1)
    for t in range(t0, t1 + 1):
        if t in reb:
            pk = reb[t]
            tr = K.TRADED[t] & np.isfinite(K.On[t])
            Ct = np.where(tr, K.Cn[t], 1.0); Ot = np.where(tr, K.On[t], 1.0)
            gap = np.where(tr, G[t] * Ot / Ct, 1.0)
            h = val * gap
            frz = (h > 0) & ~tr
            htr = np.where(frz, 0.0, h)
            oc = np.where(tr, Ct / Ot, 1.0)
            use = (mode != 'open') and (apply_days is None or t in apply_days)
            fs = np.ones(N)
            if use:
                fb = oc * (1 + extra)
                if mode == 'p4b_lu' and t + 1 < len(IDX):
                    lu = luc(t) & tr
                    on1 = G[t + 1] * K.On[t + 1] / K.Cn[t + 1]
                    gu1 = K.gap_up_open(t + 1)
                    alt = np.where(gu1, oc * G[t + 1], oc * on1)    # 隔日開盤；再漲停 → 隔日收盤
                    ok1 = np.isfinite(alt) & K.TRADED[t + 1]
                    fb = np.where(lu & ok1, alt * (1 + extra), fb)
            else:
                fb = np.ones(N)
            k = len(pk)
            def resid(T):
                tgt = np.zeros(N); tgt[pk] = T
                sold = np.clip(htr - tgt, 0, None); bought = np.clip(tgt - htr, 0, None)
                return (sold * fs).sum() * (1 - sell) + cash - (bought * fb).sum() * (1 + buy)
            lo, hi = 0.0, (htr.sum() + cash) * 2 / k
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if resid(mid) < 0: hi = mid
                else: lo = mid
            T = lo
            tgt = np.zeros(N); tgt[pk] = T
            keep = np.minimum(htr, tgt); bought = np.clip(tgt - htr, 0, None)
            val = np.where(frz, h, keep * oc + bought * oc)
            cash = resid(T)
        else:
            val = val * G[t] if t > t0 else val
        nav[t - t0] = val.sum() + cash
    return pd.Series(nav, index=IDX[t0:t1 + 1])

def plan_close_sub(n=10):
    """收盤時才判斷：開盤漲停不再剔除；收盤漲停（買不到）的新進股剔除、由下一名替補（替補也以收盤買）。"""
    out = []; held = set()
    for d in S.sig_days:
        cs = S.cross_section(d); el = cs['el']
        if el.sum() < 50: continue
        e = cs['e']; lu = luc(e)
        cand = S.order(cs, 'official')[:int(el.sum())]
        pick = []
        for j in cand:
            if lu[j] and j not in held: continue
            pick.append(j)
            if len(pick) == n: break
        out.append((e, np.array(pick, int)))
        held = set(pick)
    return out

SEG = {'VAL': ('2019-05-01', '2026-10-12', '2019-05-01'), 'ALL': ('2006-04-01', '2026-10-12', '2006-04-01'),
       'SEL': ('2006-04-01', '2019-04-30', '2006-04-01'), 'S3': ('2021-03-01', '2026-10-12', '2021-03-12')}
ew, _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, t0, t1)
print('=== 1/2/3 淨值（排除 src3 換股日；一般／壓力成本）===')
pcs = plan_close_sub()
store = {}
for cn, (bb, ss) in (('normal', K.rates(0.001)), ('stress', K.rates('stress'))):
    base = sim(pp, bb, ss, t0, t1, 'open')
    vars_ = [('p4b', 'p4b', 0.0, pp), ('p4b_lu', 'p4b_lu', 0.0, pp)]
    if cn == 'normal':
        vars_ += [(f'p4b_lu+{x}', 'p4b_lu', x, pp) for x in (0.001, 0.002, 0.003, 0.005)]
        vars_ += [('close_sub', 'p4b', 0.0, pcs), ('close_sub+0.002', 'p4b', 0.002, pcs)]
    for nm, md, ex, plan in vars_:
        nav = sim(plan, bb, ss, t0, t1, md, ex, apply_days=nons3 if plan is pp else {p[0] for p in plan if not (L.SRC3[0] <= IDX[p[0]] <= L.SRC3[1])})
        store[(cn, nm)] = nav
        row = []
        for k, (a, b, adl) in SEG.items():
            s0 = V.seg_stats(base, ew, a, b, sd, ad=adl); s1 = V.seg_stats(nav, ew, a, b, sd, ad=adl)
            pv, _ = L.paired(nav, base, sd, a, b)
            row.append(f"{k}: {s0['CAGR']:.1f}->{s1['CAGR']:.1f} t0050 {s0['t_0050']:.2f}->{s1['t_0050']:.2f} pair {pv['mean_pct']:+.3f} t{pv['t']:.2f}/nw{pv['t_nw6']:.2f}")
        print(cn, nm, ' | '.join(row))

print('=== 5. 驗證期配對差集中度（一般成本，p4b_lu）===')
base = sim(pp, *K.rates(0.001), t0, t1, 'open')
for nm in ('p4b', 'p4b_lu'):
    _, d = L.paired(store[('normal', nm)], base, sd, '2019-05-01', '2026-10-12')
    d = d[d != 0]   # src3 月份差為 0
    print(nm, 'nonzero months', len(d), 'mean', round(d.mean() * 100, 3), 't', round(L.tstat(d.to_numpy()), 2))
    print('  by year sum%', (d.groupby(d.index.year).sum() * 100).round(2).to_dict())
    s = d.sort_values(ascending=False)
    for k in (1, 3, 5):
        x = s.iloc[k:]; print(f'  drop best {k}', round(x.mean() * 100, 3), round(L.tstat(x.to_numpy()), 2))
    x = d[d.index.year != 2021]; print('  ex 2021', round(x.mean() * 100, 3), round(L.tstat(x.to_numpy()), 2))
    x = d[d.index >= pd.Timestamp('2022-01-01')]; print('  2022+', len(x), round(x.mean() * 100, 3), round(L.tstat(x.to_numpy()), 2))

print('=== 4. 容量：新進股換股日 e 的成交值（新台幣）===')
Vv = K.V.to_numpy()
rows = []; prev = None
for (e, pk, pool, d) in pl:
    cur = set(pk.tolist())
    if prev is not None:
        for j in cur - prev:
            rows.append(dict(d=d, e=IDX[e], code=COLS[j], val=Vv[e, j], v20=np.nanmedian(Vv[max(0, e - 20):e, j]), lu=bool(luc(e)[j])))
    prev = cur
R = pd.DataFrame(rows)
R = R[R.d >= pd.Timestamp('2006-04-01')]
for per, g in (('SEL', R[R.d <= L.SEL_LAST_SIG]), ('VAL', R[R.d > L.SEL_LAST_SIG])):
    q = g.val.quantile([.05, .1, .25, .5]) / 1e6
    print(per, 'n', len(g), 'day-e value 百萬 p5/p10/p25/p50', q.round(1).tolist(),
          '| 100萬/當日成交值 p50/p90', round(float((1e6 / g.val).median() * 100), 2), round(float((1e6 / g.val).quantile(.9) * 100), 2), '%')
# 5 分 K 推估收盤競價（13:25～13:30）占比上限：1 − 5分K量×官方收盤 / 官方成交值（2025-09-24～2026-04-01，src4）
TS = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/tw_stocker_daily')
SRC = K.SRC
sub = R[(R.e >= pd.Timestamp('2025-09-24')) & (R.e <= pd.Timestamp('2026-04-01'))]
sh = []
for _, r in sub.iterrows():
    f = TS / f"{r.code}.csv"
    if not f.exists(): continue
    t = pd.read_csv(f, index_col=0, parse_dates=True)
    if r.e not in t.index: continue
    j = K.CI[r.code]; ie = IDX.get_loc(r.e)
    sh.append(1 - t.loc[r.e, 'volume'] * K.Cn[ie, j] / Vv[ie, j])
sh = np.array(sh)
if len(sh):
    print('收盤競價＋遺漏量占比（上限）n', len(sh), 'p25/p50/p75', np.round(np.quantile(sh, [.25, .5, .75]) * 100, 1).tolist(), '%')
    print('  → 100 萬／（當日成交值×占比中位數）中位', round(float(np.median(1e6 / (sub.val.to_numpy()[:len(sh)] * max(np.median(sh), 1e-3))) * 100), 1), '%')
