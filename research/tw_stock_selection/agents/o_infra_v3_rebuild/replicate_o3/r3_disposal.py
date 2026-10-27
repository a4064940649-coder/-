#!/usr/bin/env python3
"""自己重建處置近似規則 Q1_bear_c1_3consec，比對 disposal_hist.csv，並算 2021-09-24～2023（挑選）、2024～2026（驗證）的精確率／召回率（股票日）。"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
P = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
O3 = Path('/home/user/-/research/tw_stock_selection/agents/o_infra_v3_rebuild')
HERE = Path(__file__).resolve().parent
ADJ = pd.read_parquet(P / 'adj.parquet'); C = pd.read_parquet(P / 'close_raw.parquet')
tx = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date']).set_index('date')['close']
variant = sys.argv[1] if len(sys.argv) > 1 else 'cal'
IDX = ADJ.index
A = ADJ.where(C.notna())
SH = 5 if variant.endswith('5') else 6
if variant.startswith('cal'):      # 面板交易日
    r6 = A / A.shift(SH) - 1
else:                      # 個股自己的有值筆數
    r6 = A.apply(lambda s: s / s.dropna().shift(SH).reindex(s.index) - 1)
T = tx.reindex(IDX.union(tx.index)).ffill().reindex(IDX)
t6 = T / T.shift(SH) - 1
if 'tx' in variant:
    Tn = T.to_numpy(); t6m = np.full(A.shape, np.nan); An = A.to_numpy()
    for j in range(A.shape[1]):
        ii = np.where(np.isfinite(An[:, j]))[0]
        if len(ii) > SH: t6m[ii[SH:], j] = Tn[ii[SH:]] / Tn[ii[:-SH]] - 1
    t6x = pd.DataFrame(t6m, index=IDX, columns=A.columns)
    cond = (r6 > 0.32) & ((r6 - t6x) >= 0.20)
else:
    cond = (r6 > 0.32) & (r6.sub(t6, axis=0) >= 0.20)
cn = cond.to_numpy(); TR = C.notna().to_numpy()
ev = []
for j, c in enumerate(C.columns):
    run = 0; busy_until = -1
    for t in range(len(IDX)):
        if not TR[t, j]: continue
        run = run + 1 if cn[t, j] else 0
        if t <= busy_until: continue
        if run >= 3:
            # 處置期 t+1～t+10 個交易日（面板日曆）
            s, e = t + 1, min(t + 10, len(IDX) - 1)
            if s >= len(IDX): continue
            ev.append((c, IDX[t], IDX[s], IDX[e])); busy_until = e
E = pd.DataFrame(ev, columns=['code', 'trigger', 'start', 'end'])
H = pd.read_csv(O3 / 'disposal_hist.csv', dtype={'code': str}, parse_dates=['trigger', 'start', 'end'])
Epre = E[E['trigger'] <= '2021-09-23']
k = lambda d: set(zip(d['code'], d['trigger'].dt.strftime('%Y-%m-%d'), d['start'].dt.strftime('%Y-%m-%d'), d['end'].dt.strftime('%Y-%m-%d')))
a, b = k(Epre), k(H)
out = {'variant': variant, 'mine_n_pre': len(Epre), 'hist_n': len(H), 'same': len(a & b), 'only_mine': sorted(a - b)[:10], 'only_hist': sorted(b - a)[:10]}
# 精確率／召回率（股票日，面板代碼、有收盤的日子）
D5 = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras/disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['date', 'start', 'end'])
CI = {c: j for j, c in enumerate(C.columns)}
def daymask(df):
    M = np.zeros((len(IDX), len(C.columns)), bool)
    for r in df.itertuples():
        if r.code not in CI: continue
        M[(IDX >= r.start) & (IDX <= r.end), CI[r.code]] = True
    return M & TR
Ma, Mr = daymask(E), daymask(D5)
for nm, (lo, hi) in {'select': ('2021-09-24', '2023-12-31'), 'validate': ('2024-01-01', '2026-10-22')}.items():
    w = (IDX >= lo) & (IDX <= hi)
    tp = int((Ma[w] & Mr[w]).sum()); fp = int((Ma[w] & ~Mr[w]).sum()); fn = int((~Ma[w] & Mr[w]).sum())
    out[nm] = {'tp': tp, 'fp': fp, 'fn': fn, 'precision': tp / (tp + fp), 'recall': tp / (tp + fn)}
print(json.dumps(out, ensure_ascii=False, default=str, indent=1))
(HERE / f'my_disposal_{variant}.json').write_text(json.dumps(out, ensure_ascii=False, default=str, indent=1))
