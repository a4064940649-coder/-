#!/usr/bin/env python3
"""反方審查（成本與可成交）o18：只讀研究員檔案與上游面板，不寫任何檔案。python3 -I -B review_exec_o18.py
1. V7X：opt／p90／carry 逐格取最不利，檢查「opt＝不利端」在可執行口徑是否成立。
2. 紙上：重建 C0、ALLPIT_REAL_{opt,p90,carry} 名單，cons_main 引擎下跑成本網格，量周轉、成本拖累、損益兩平滑價。
3. 可成交：開盤跌停賣不掉、整張門檻（30 萬每檔 3 萬）、部位占 60 日均量比例（100 萬／1,000 萬）。"""
import sys, json, math
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
R = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
TB = R['RESULTS_V7X']['table']
print('== 1. V7X 逐格最不利（cons_main）')
for c in ('normal', 'stress'):
    for cap in ('100萬', '300萬', '1000萬'):
        row = []
        for f in ('exc_med', 'exc_p10', 't_0050_med', 'exc_Y19on_med', 'exc_HOLD_med', 'exc_S1318_med'):
            xs = {v: TB[f'REAL_{v}|cons_main|{c}|{cap}'][f] for v in ('opt', 'p90', 'carry')}
            w = min(xs, key=xs.get)
            row.append(f"{f}: opt {xs['opt']:+.2f} worst {xs[w]:+.2f}({w})")
        print(f'  {c} {cap}: ' + ' | '.join(row))

sys.path.insert(0, str(HERE)); import o18lib as O; sys.path.pop(0)
P, V, K, COLS, CE = O.P, O.V, O.K, O.COLS, O.CE
PR = R['PREREG']
VARS = {k: tuple(None if x == 'None' else x for x in PR['paper']['variants'][k]['kind']) for k in ('C0', 'ALLPIT_REAL_opt', 'ALLPIT_REAL_p90', 'ALLPIT_REAL_carry')}
B = {k: O.build(kind) for k, kind in VARS.items()}
SD0 = B['C0']['sd']
FEE, TAX = 0.001425 * 0.6, 0.003
def run(k, slip_b, slip_s, eng='cons_main'):
    b = B[k]; pl = [(p[0], p[1]) for p in b['plan']]
    bb, ss = FEE + slip_b, FEE + TAX + slip_s
    if eng == 'freeze': return CE.simulate(pl, bb, ss, b['t0'], b['t1'], mode='freeze')
    return CE.simulate(pl, bb, ss, b['t0'], b['t1'], mode='consideration', settle=O.ST[eng])
def summ(nav):
    W = O.win(nav, SD0)
    ex = W.r - W.r50
    return O.ann(W.r), O.ann(W.r) - O.ann(W.r50), O.tt(ex), W
print('\n== 2. 紙上成本網格（每邊滑價；一般＝0.1%，研究員壓力＝買 0.5%／賣 0.5% 全含）')
REF = {}
for k in B:
    nav, lg = run(k, 0.001, 0.001, 'freeze' if k == 'C0' else 'cons_main')
    a, ex, t, W = summ(nav); REF[k] = W
    print(f"  {k}: 一般 ann {a:.2f} 超額 {ex:+.2f} t {t:.3f}  單邊周轉/月 {lg['turnover']:.3f}  年化周轉 {lg['turnover']*12:.2f}x")
    rt = (FEE + 0.001) + (FEE + TAX + 0.001)
    print(f"     一般成本年拖累 ≈ {lg['turnover']*12*rt*100:.2f}pp（來回 {rt*100:.3f}%）；其中證交稅 ≈ {lg['turnover']*12*TAX*100:.2f}pp")
for k in ('ALLPIT_REAL_opt', 'ALLPIT_REAL_p90', 'ALLPIT_REAL_carry'):
    out = []
    for s in (0.002, 0.003, 0.005, 0.0075, 0.01):
        nav, _ = run(k, s, s); a, ex, t, W = summ(nav)
        s2 = W[(W.index >= pd.Period('2013-01', 'M')) & (W.index <= pd.Period('2018-12', 'M'))]
        s3 = W[W.index >= pd.Period('2019-01', 'M')]
        out.append(f"{s*100:.2f}%: ann {a:.2f} ex {ex:+.2f} t {t:.2f} 13-18 {O.ann(s2.r)-O.ann(s2.r50):+.2f} 19on {O.ann(s3.r)-O.ann(s3.r50):+.2f}")
    print(f'  {k} 單邊滑價 ' + ' | '.join(out))
# 損益兩平（超額＝0、t＝1.0）
for k in ('ALLPIT_REAL_opt',):
    lo, hi = 0.0, 0.05
    for target, nm in ((0.0, '超額=0'), (1.0, 't=1.0'), (1.96, 't=1.96')):
        a_, b_ = 0.0, 0.05
        for _ in range(18):
            m = 0.5 * (a_ + b_); nav, _ = run(k, m, m); _, ex, t, _ = summ(nav)
            val = ex if nm == '超額=0' else t
            if val > target: a_ = m
            else: b_ = m
        print(f'  {k} 損益兩平單邊滑價（{nm}）≈ {a_*100:.3f}%')

print('\n== 3. 可成交性（ALLPIT_REAL_opt 名單；紙上引擎在 t+1 開盤成交）')
Cn, On, IDX = K.Cn, K.On, K.IDX
PREVC = pd.DataFrame(Cn, index=IDX).ffill().shift(1).to_numpy()
def ld_price(ref, lim):
    raw = ref * (1 - lim); tk = np.select([raw < 10, raw < 50, raw < 100, raw < 500, raw < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)
    return np.ceil(raw / tk - 1e-9) * tk
for k in ('C0', 'ALLPIT_REAL_opt'):
    pl = B[k]['plan']; held = set(); nsell = nld = 0; lot30 = lot100 = npick = 0; part = {'100萬': [], '1000萬': []}
    for e, pk, el, d in pl:
        lim = 0.07 if IDX[e] < pd.Timestamp('2015-06-01') else 0.10
        sells = held - set(pk.tolist())
        for j in sells:
            if not np.isfinite(On[e, j]): continue
            ref = K.ref_price(e)[j] if hasattr(K, 'ref_price') else PREVC[e, j]
            nsell += 1
            if np.isfinite(ref) and On[e, j] <= ld_price(ref, lim) + 1e-6: nld += 1
        t = IDX.get_loc(d)
        for j in pk:
            npick += 1
            px = Cn[t, j]
            if px * 1000 > 30e4 / 10: lot30 += 1
            if px * 1000 > 100e4 / 10: lot100 += 1
        held = set(pk.tolist())
    print(f'  {k}: 賣出 {nsell} 筆，開盤即跌停 {nld}（{nld/max(nsell,1)*100:.2f}%）；入選 {npick} 檔次，一張 > 3 萬（30 萬資金）{lot30/npick*100:.1f}%、一張 > 10 萬（100 萬）{lot100/npick*100:.1f}%')
# 部位占 60 日均成交值
V60 = None
try:
    _, V60 = K.rolling_signals('strict')
except Exception as ex_:
    print('  V60 取得失敗', ex_)
if V60 is not None:
    for k in ('C0', 'ALLPIT_REAL_opt'):
        r1, r10 = [], []
        for e, pk, el, d in B[k]['plan']:
            t = IDX.get_loc(d)
            for j in pk:
                v = V60[t, j]
                if np.isfinite(v) and v > 0: r1.append(1e5 / v); r10.append(1e6 / v)
        r1, r10 = np.array(r1), np.array(r10)
        print(f'  {k}: 每檔部位／60 日均成交值  100 萬：中位 {np.median(r1)*100:.2f}% p90 {np.quantile(r1,.9)*100:.2f}% max {r1.max()*100:.1f}%；1,000 萬：中位 {np.median(r10)*100:.2f}% p90 {np.quantile(r10,.9)*100:.2f}% >5% 占 {np.mean(r10>0.05)*100:.1f}%')
# 放回（opt 假設）股的流動性
bm = B['ALLPIT_REAL_opt']['base_missing']; ins = []
for e, pk, el, d in B['ALLPIT_REAL_opt']['plan']:
    for j in pk:
        if COLS[j] in set(bm.get(d, [])):
            ins.append((d, COLS[j], V60[IDX.get_loc(d), j] if V60 is not None else np.nan))
I = pd.DataFrame(ins, columns=['d', 'c', 'v60'])
print(f'  放回槽 {len(I)}：60 日均成交值中位 {I.v60.median()/1e8:.2f} 億；1,000 萬每檔 100 萬占比中位 {np.median(1e6/I.v60)*100:.2f}%')
# 訊號日期限狀態
dc = pd.Series([V.deadline_case(pd.Timestamp(d)) for d in SD0]).value_counts().to_dict()
print('  訊號日營收期限狀態', dc)
