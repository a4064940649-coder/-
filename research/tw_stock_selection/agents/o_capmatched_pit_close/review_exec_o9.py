#!/usr/bin/env python3
"""反方審查 o9（成本與可成交角度）。不修改研究員檔案；只 import s6_exec（其 __main__ 不執行）重用 run_exec。
python3 -I -B review_exec_o9.py [a|b]  → 結果寫到環境變數 O9EXEC（預設 scratch），並印 stdout。
檢查：
 E1 30 萬級距（研究員只測 100 萬以上）：K5T 與 K5T+0.5%，GROW／FIX；每月委託筆數、最低手續費占比、持股檔數。
 E2 K5T 對 0050 分段：2019-05～2022-12（o7 挑選期）與 2023～2026。
 E3 開盤跌停賣不掉（引擎假設照開盤價賣出）：要賣的衛星在開盤即跌停的次數。
 E4 每月需下單筆數（是否需要盯盤）與 2020-10-26 前零股以收盤價成交的占比。
"""
import sys, os, json, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
CFG = sys.argv[1] if len(sys.argv) > 1 else 'a'
sys.argv = ['s6_exec.py', CFG]
import numpy as np, pandas as pd
import s6_exec as X
P, L = X.P, X.L
OUTF = Path(os.environ.get('O9EXEC', '/tmp/review_exec_o9.json'))

# --- run_exec 的包裝：額外記錄最低手續費次數、委託筆數、持股檔數、開盤跌停賣出 ---
_orig_max = max
def run_exec_rec(days, plans, tier, mode, extra=0.0):
    r, acc = X.run_exec(days, plans, tier, mode, extra)
    return r, acc

def limit_down_open(e):
    lim = 0.07 if L.IDX[e] < pd.Timestamp('2015-06-01') else 0.10
    with np.errstate(invalid='ignore', divide='ignore'):
        ref = L.PREVC[e]
        raw = ref * (1 - lim); tk = L.tick(raw)
        ld = np.ceil(raw / tk - 1e-9) * tk
    return np.isfinite(L.On[e]) & np.isfinite(ld) & (L.On[e] <= ld + 1e-6)

def minfee_stats(days, plans, tier):
    """近似：以目標權重差 × 淨值估每筆委託金額（不含成本），統計低於 50,000 元的委託（最低手續費 20 元綁定）。"""
    ti = [L.IDX.get_loc(d) for d in days]
    prev = None; n_ord = []; n_small = []; n_hold = []; tiny_fee_pp = []
    for k in range(len(ti) - 1):
        d = days[k]; idx, w = plans[d]
        wv = np.zeros(L.N); np.add.at(wv, idx, w)
        n_hold.append(int((wv > 0).sum()))
        if prev is not None:
            dw = np.abs(wv - prev); dw[dw < X.BAND] = 0.0
            val = dw * tier
            o = val[val > 0]
            n_ord.append(len(o)); n_small.append(int((o < 50000).sum()))
            fee = np.maximum(20.0, X.RATE * o).sum(); tiny_fee_pp.append(100 * (fee - X.RATE * o.sum()) / tier)
        prev = wv
    return dict(orders_per_month_med=float(np.median(n_ord)), orders_lt50k_share=float(np.sum(n_small) / max(np.sum(n_ord), 1)),
                holdings_med=float(np.median(n_hold)), minfee_excess_pp_per_month=float(np.mean(tiny_fee_pp)))

def sell_limit_down(days, plans):
    ti = [L.IDX.get_loc(d) for d in days]; prev = None; n_sell = 0; n_ld = 0; ex = []
    for k in range(len(ti) - 1):
        d = days[k]; idx, w = plans[d]; e = ti[k] + 1
        wv = np.zeros(L.N); np.add.at(wv, idx, w)
        if prev is not None:
            sell = (prev - wv) > X.BAND
            ld = limit_down_open(e)
            n_sell += int(sell.sum()); hit = sell & ld; n_ld += int(hit.sum())
            for j in np.where(hit)[0]: ex.append((str(L.IDX[e].date()), L.COLS[j]))
        prev = wv
    return n_sell, n_ld, ex[:10]

RES = {}
TIERS = [('30萬', 3e5, 'GROW'), ('30萬', 3e5, 'FIX'), ('100萬', 1e6, 'GROW')]
for tn, tier, mode in TIERS:
    for ex in (0.0, 0.005):
        rows = []
        for k, (pl, days, b) in X.PLANS.items():
            if k == 'OFF': continue
            r, acc = X.run_exec(days, pl, tier, mode, ex)
            x = r - b
            dd = pd.DatetimeIndex(days[:-1])
            m1 = dd < pd.Timestamp('2023-01-01')
            rows.append(dict(j=k, mean=x.mean() * 100, t=P.tstat(x), m_19_22=x[m1].mean() * 100, t_19_22=P.tstat(x[m1]),
                             m_23_26=x[~m1].mean() * 100, t_23_26=P.tstat(x[~m1]),
                             cost_pp=100 * (acc['fee'] + acc['tax'] + acc['impact'] + acc['odd'] + acc['flow_cost']) / (tier * len(x)),
                             fee_pp=100 * acc['fee'] / (tier * len(x)), orders_pm=acc['orders'] / len(x),
                             pre_odd_share=acc['pre_odd_val'] / max(acc['odd_val'] + acc['lot_val'], 1)))
        df = pd.DataFrame(rows)
        key = f'{tn}|{mode}|' + ('K5T' if ex == 0 else 'K5T+0.5%')
        RES[key] = dict(mean_pp_jmed=float(df['mean'].median()), mean_pp_p10=float(df['mean'].quantile(0.1)),
                        t_jmed=float(df['t'].median()), t_p10=float(df['t'].quantile(0.1)),
                        m_19_22_jmed=float(df.m_19_22.median()), t_19_22_jmed=float(df.t_19_22.median()),
                        m_23_26_jmed=float(df.m_23_26.median()), t_23_26_jmed=float(df.t_23_26.median()),
                        cost_pp_jmed=float(df.cost_pp.median()), fee_pp_jmed=float(df.fee_pp.median()),
                        orders_pm_jmed=float(df.orders_pm.median()), pre_odd_share=float(df.pre_odd_share.median()))
        print(CFG, key, {a: round(v, 3) for a, v in RES[key].items()}, flush=True)

for tier in (3e5, 1e6):
    st = [minfee_stats(days, pl, tier) for k, (pl, days, b) in X.PLANS.items() if k != 'OFF']
    RES[f'minfee_{int(tier)}'] = {a: float(np.median([s[a] for s in st])) for a in st[0]}
    print('minfee', tier, RES[f'minfee_{int(tier)}'], flush=True)

ns = nl = 0; exs = []
for k, (pl, days, b) in X.PLANS.items():
    if k == 'OFF': continue
    a, c, e = sell_limit_down(days, pl); ns += a; nl += c; exs += e
RES['sell_limit_down'] = dict(n_sell=ns, n_limit_down_open=nl, examples=exs[:10])
print('sell_limit_down', RES['sell_limit_down'], flush=True)
OUTF.write_text(json.dumps(RES, ensure_ascii=False, indent=1, default=float))
