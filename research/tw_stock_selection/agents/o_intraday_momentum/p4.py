#!/usr/bin/env python3
"""P4 成交面：主策略（main_v2 正式、排除金融 n10）換股日的新進股／賣出股，在「隔日開盤」以外改用「隔日收盤」或「(開盤＋收盤)/2 近似 TWAP」成交的每筆改善。
用法：python3 -I -B p4.py sel   （挑選期，lock_selection）
      python3 -I -B p4.py oos   （驗證期，必須先有 PREREG）
改善定義（正＝比開盤成交好）：買進 close：O/C−1；TWAP：2O/(O+C)−1。賣出 close：C/O−1；TWAP：(O+C)/(2O)−1。
每筆＝買進與賣出合併後的單邊交易平均；t 用月份分群（每個換股日先平均）。
收盤漲停（買）／收盤跌停（賣）的交易另外標記：收盤成交在實務上買不到／賣不掉，敏感度版本剔除。"""
import sys, json, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import olib as L
MODE = sys.argv[1]
assert MODE in ('sel', 'oos')
if MODE == 'sel':
    L.lock_selection()
else:
    PR, SH, PT = L.assert_prereg()
K, V = L.K, L.V
IDX, COLS, N = L.IDX, L.COLS, L.N

YL, Y5, _ = V.load_revenue()
S = V.V2('gapfix', YL, Y5, exfin=True)
if MODE == 'sel':
    S.sig_days = [d for d in S.sig_days if d <= L.SEL_LAST_SIG]; S.end_day = L.CUT_DAY
    a, b = pd.Timestamp('2006-04-01'), L.SEL_LAST_SIG
else:
    a, b = L.VAL_FIRST, V.END1012
pl = S.plan(10, 'official')

def limit_down_close(t):
    lim = 0.07 if IDX[t] < pd.Timestamp('2015-06-01') else 0.10
    ref = K.ref_price(t)
    raw = ref * (1 - lim); tk = K.tick(raw)
    ld = np.ceil(raw / tk - 1e-9) * tk
    return np.isfinite(K.Cn[t]) & (K.Cn[t] <= ld + 1e-6)
def limit_up_close(t):
    lim = 0.07 if IDX[t] < pd.Timestamp('2015-06-01') else 0.10
    lu = K.limit_up_price(K.ref_price(t), lim)
    return np.isfinite(K.Cn[t]) & (K.Cn[t] >= lu - 1e-6)

rows = []
prev = None
for (e, pk, pool, d) in pl:
    cur = set(pk.tolist())
    if prev is not None and a <= d <= b:
        O = K.On[e]; C = K.Cn[e]
        luc = limit_up_close(e); ldc = limit_down_close(e)
        src3 = L.SRC3[0] <= IDX[e] <= L.SRC3[1]
        for j in cur - prev:
            rows.append(dict(d=d, side='buy', code=COLS[j], oc=C[j] / O[j] - 1, imp_close=O[j] / C[j] - 1, imp_twap=2 * O[j] / (O[j] + C[j]) - 1,
                             lim_close=bool(luc[j]), src3=src3))
        for j in prev - cur:
            if not (np.isfinite(O[j]) and np.isfinite(C[j]) and K.TRADED[e, j]): continue   # 凍結（停牌）不能賣
            rows.append(dict(d=d, side='sell', code=COLS[j], oc=C[j] / O[j] - 1, imp_close=C[j] / O[j] - 1, imp_twap=(O[j] + C[j]) / (2 * O[j]) - 1,
                             lim_close=bool(ldc[j]), src3=src3))
    prev = cur
R = pd.DataFrame(rows)
def agg(df):
    if len(df) == 0: return None
    m = df.groupby('d')
    o = {'n_trades': int(len(df)), 'n_months': int(m.ngroups)}
    for c in ('oc', 'imp_close', 'imp_twap'):
        x = m[c].mean().to_numpy()
        o[c + '_mean_pct'] = float(df[c].mean() * 100); o[c + '_t_month'] = L.tstat(x); o[c + '_median_pct'] = float(df[c].median() * 100)
    o['share_limit_at_close'] = float(df['lim_close'].mean())
    return o
out = {'mode': MODE, 'period': [str(a.date()), str(b.date())]}
for nm, f in (('all', lambda r: r), ('ex_src3', lambda r: r[~r.src3]), ('ex_src3_ex_limitclose', lambda r: r[~r.src3 & ~r.lim_close]), ('only_src3', lambda r: r[r.src3])):
    rr = f(R)
    out[nm] = {'pooled': agg(rr), 'buy': agg(rr[rr.side == 'buy']), 'sell': agg(rr[rr.side == 'sell'])}
# 逐年
yy = {}
rr = R[~R.src3]
for y, g in rr.groupby(rr.d.dt.year):
    yy[int(y)] = {s: (round(float(g[g.side == s]['oc'].mean() * 100), 3) if (g.side == s).any() else None) for s in ('buy', 'sell')}
    yy[int(y)]['pooled_imp_close'] = round(float(g['imp_close'].mean() * 100), 3); yy[int(y)]['pooled_imp_twap'] = round(float(g['imp_twap'].mean() * 100), 3)
out['yearly_ex_src3'] = yy
# ---------------- 淨值版本（一般／壓力成本）----------------
t0, t1, sd = S.bounds()
if MODE == 'sel': L.assert_locked_ok(t1)
pp = [(p[0], p[1]) for p in pl]
nonsrc3 = {p[0] for p in pl if not (L.SRC3[0] <= IDX[p[0]] <= L.SRC3[1])}
VARS = [('open', 'open'), ('close', 'open'), ('twap', 'open'), ('close', 'close'), ('twap', 'twap'), ('open', 'close')]
navs = {}
out['nav'] = {}
for cn, (bb, ss) in (('normal', K.rates(0.001)), ('stress', K.rates('stress'))):
    base = L.simulate_exec(pp, bb, ss, t0, t1)
    for bp, spx in VARS:
        for ad_nm, ad in (('all', None), ('ex_src3', nonsrc3)):
            if MODE == 'sel' and ad_nm == 'ex_src3': continue
            nav = L.simulate_exec(pp, bb, ss, t0, t1, bp, spx, apply_days=ad)
            st = V.seg_stats(nav, base, str(a.date()), str(b.date()), sd)
            pv, _ = L.paired(nav, base, sd, a, b)
            out['nav'][f'{cn}|buy_{bp}|sell_{spx}|{ad_nm}'] = {'CAGR': st['CAGR'], 'maxDD': st['maxDD'], 't_0050': st['t_0050'], 'paired_vs_open': pv}
            print(cn, bp, spx, ad_nm, round(st['CAGR'], 2), round(st['t_0050'], 2), 'paired', round(pv['mean_pct'], 3), round(pv['t'], 2))
fn = L.HERE / f'p4_{MODE}.json'
out['time'] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
fn.write_text(L.jdump(out), encoding='utf-8')
for k in ('all', 'ex_src3', 'ex_src3_ex_limitclose'):
    for s in ('pooled', 'buy', 'sell'):
        o = out[k][s]
        if o: print(k, s, o['n_trades'], 'oc', round(o['oc_mean_pct'], 3), round(o['oc_t_month'], 2), 'close', round(o['imp_close_mean_pct'], 3), round(o['imp_close_t_month'], 2),
                    'twap', round(o['imp_twap_mean_pct'], 3), round(o['imp_twap_t_month'], 2), 'limc', round(o['share_limit_at_close'], 3))
