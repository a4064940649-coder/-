#!/usr/bin/env python3
"""反方審查（成本與可成交）— o_rev_event_trigger 的 J+1 選定版。唯讀：不寫任何檔案，只輸出到 stdout。
執行：python3 -I -B review_exec_o5.py
檢查：
  1) 賣出腿開盤跌停（tick 價）仍假設以開盤價賣出 → 延到第一個非跌停開盤賣出的損益
  2) 買進腿開盤接近漲停（≥ 漲停價 − 2 tick，但未被擋）／被擋遞補次數（需 9:00 盯盤）
  3) 成交日在處置期間的買賣腿（含持有中進入處置、需預收款券／分盤撮合）
  4) 成交價改為收盤（30 萬帳戶 2020-10-26 前零股只能盤後；大帳戶分散成交的代理）、改為 (開+收)/2
  5) 1,000 萬帳戶：每檔 100 萬占成交日全日成交值的比例分布
AB 用 10 個 seed（研究員正式版 40 個），先重現 10 seed 的基準再比較，差值才有意義。"""
import sys, json, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import fx_core as F
L, K, V = F.L, F.K, F.V
IDX = K.IDX
st = F.setup()
days, end = F.version_days(1)
W = F.windows([d for _, d in days], end)
SEEDS = range(10)

def tick(p): return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)
def lim_of(t): return 0.07 if IDX[t] < pd.Timestamp('2015-06-01') else 0.10
def ref_px(t):
    c = K.Cn[t]; r = K.Rn[t]; ref = c / (1 + r)
    return np.where(K.Fn[t] == 1, K.PREVC[t], ref)
def ld_price(t):
    raw = ref_px(t) * (1 - lim_of(t)); tk = tick(raw)
    return np.ceil(raw / tk - 1e-9) * tk
def lu_price(t):
    raw = ref_px(t) * (1 + lim_of(t)); tk = tick(raw)
    return np.floor(raw / tk + 1e-9) * tk

# ---------- 基準（10 seed）與名單 ----------
plans = []; rets = []
for s in SEEDS:
    S, pl, a, b, sd = F.build(days, end, F.make_unavail('day', s))
    plans.append((pl, S.nblk)); t0, t1 = a, b
    nav, lg = F.sim(pl, a, b, 'normal'); rets.append(nav.pct_change())
mean_nav = lambda X: (1 + pd.concat(X, axis=1).mean(axis=1).fillna(0)).cumprod()
base = mean_nav(rets)
bb = F.block(base, W, ('ALL', 'POST', 'HOLD'))
print('== 基準 J+1 AB 10 seeds：', {k: (round(v['CAGR'], 2), round(v['t_0050'], 3), round(v['t_0050_cal'], 3)) for k, v in bb.items()})
print('   開盤漲停被擋遞補（每 seed 次數）：', [p[1] for p in plans], ' 換股月數', len(plans[0][0]))

# ---------- 逐腿檢查 ----------
L.disposal_table(); D = L._D
OADJ = lambda t, j: K.MYADJ.to_numpy()[t, j] * K.On[t, j] / K.Cn[t, j]
MY = K.MYADJ.to_numpy()
Vv = K.V.to_numpy()
rows = []
for si, (pl, _) in enumerate(plans):
    prev = set()
    for e, pk, _, d in pl:
        cur = set(int(j) for j in pk)
        ed = np.datetime64(IDX[e]); sdd = np.datetime64(d)
        ld = ld_price(e); lu = lu_price(e); tk = tick(lu)
        for side, js in (('sell', prev - cur), ('buy', cur - prev)):
            for j in js:
                o = K.On[e, j]
                if not np.isfinite(o): continue
                dsel = (D['j'] == j) & (D['st'] <= ed) & (D['en'] >= ed)
                r = dict(seed=si, sig=d, e=e, j=j, side=side, ldo=bool(o <= ld[j] + 1e-6), near_lu=bool(o >= lu[j] - 2 * tk[j] - 1e-6),
                         disp=bool(dsel.any()), disp_ann_after_sig=bool((dsel & (D['ann'] > sdd)).any()),
                         part=1e6 / Vv[e, j] if np.isfinite(Vv[e, j]) and Vv[e, j] > 0 else np.nan)
                if side == 'sell' and r['ldo']:
                    k = e + 1
                    while k < len(IDX) and (not np.isfinite(K.On[k, j]) or K.On[k, j] <= ld_price(k)[j] + 1e-6): k += 1
                    r['delay_days'] = k - e
                    r['delay_ret'] = (MY[k, j] * K.On[k, j] / K.Cn[k, j]) / (MY[e, j] * o / K.Cn[e, j]) - 1 if k < len(IDX) else np.nan
                rows.append(r)
        prev = cur
R = pd.DataFrame(rows)
nm = len(plans[0][0]) * len(plans)
for side in ('sell', 'buy'):
    x = R[R.side == side]
    print(f'\n== {side} 腿：共 {len(x)}（每月 {len(x) / nm:.2f}）')
    print('   開盤跌停：', int(x.ldo.sum()), f'({x.ldo.mean() * 100:.2f}%)', ' 開盤 ≥ 漲停−2tick：', int(x.near_lu.sum()), f'({x.near_lu.mean() * 100:.2f}%)')
    print('   成交日在處置期間：', int(x.disp.sum()), f'({x.disp.mean() * 100:.2f}%)', ' 其中處置公告晚於訊號日：', int(x.disp_ann_after_sig.sum()))
    p = x.part.dropna()
    print('   1,000 萬帳戶（每檔 100 萬）占成交日全日成交值：中位', f'{p.median() * 100:.2f}%', ' >1%', f'{(p > .01).mean() * 100:.1f}%',
          ' >2%', f'{(p > .02).mean() * 100:.1f}%', ' >5%', f'{(p > .05).mean() * 100:.1f}%', ' >10%', f'{(p > .10).mean() * 100:.1f}%')
    for per, a_, b_ in (('2006-2018', '2006', '2018-12-31'), ('2019+', '2019', '2026-12-31')):
        y = x[(x.sig >= pd.Timestamp(a_)) & (x.sig <= pd.Timestamp(b_))].part.dropna()
        print(f'     {per}：中位 {y.median() * 100:.2f}%  >5% {(y > .05).mean() * 100:.1f}%')
ls = R[(R.side == 'sell') & R.ldo]
if len(ls):
    print('\n== 開盤跌停賣不掉 → 延到第一個非跌停開盤：平均延', round(ls.delay_days.mean(), 2), '日；延後報酬 平均', f'{ls.delay_ret.mean() * 100:.2f}%',
          ' 中位', f'{ls.delay_ret.median() * 100:.2f}%', ' 最差', f'{ls.delay_ret.min() * 100:.2f}%')
    print('   折算組合（每腿 1/10 權重）每月', f'{ls.delay_ret.sum() / 10 / nm * 100:.4f} pp')
    print(ls.assign(code=[K.COLS[j] for j in ls.j], sig=ls.sig.dt.date)[['seed', 'sig', 'code', 'delay_days', 'delay_ret']].query('seed==0').to_string())

# ---------- 成交價改成收盤／(開+收)/2 ----------
On0 = K.On
def run_px(mode, until=None):
    On_mod = On0.copy()
    for pl, _ in plans:
        for e, *_ in pl:
            if until is not None and IDX[e] >= until: continue
            if mode == 'close': On_mod[e] = np.where(np.isfinite(K.Cn[e]), K.Cn[e], On0[e])
            elif mode == 'mid': On_mod[e] = np.where(np.isfinite(K.Cn[e]) & np.isfinite(On0[e]), 0.5 * (On0[e] + K.Cn[e]), On0[e])
    K.On = On_mod
    try:
        rr = [F.sim(pl, t0, t1, 'normal')[0].pct_change() for pl, _ in plans]
    finally:
        K.On = On0
    return mean_nav(rr)
for lab, mode, until in (('全期收盤成交', 'close', None), ('2020-10-26 前收盤成交（30 萬零股）', 'close', pd.Timestamp('2020-10-26')), ('全期 (開+收)/2', 'mid', None)):
    nv = run_px(mode, until)
    b = F.block(nv, W, ('ALL', 'POST', 'HOLD'))
    print(f'\n== {lab}：', {k: (round(v['CAGR'], 2), round(v['t_0050'], 3), round(v['t_0050_cal'], 3)) for k, v in b.items()})
    d = F.paired_cal(nv, base, 'ALL'); print('   對基準日曆月配對差', round(d['mean_pp'], 3), 'pp/月 t', round(d['t'], 2))
