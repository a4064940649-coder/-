#!/usr/bin/env python3
"""獨立抽查：用迴圈 Wilder RSI(3)、ROC(200) 重算 2 個訊號日的 A 版前 10 名，對照 trades_A 的買進。"""
import numpy as np, pandas as pd
P = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel/'
adj = pd.read_parquet(P + 'adj.parquet'); V = pd.read_parquet(P + 'value.parquet'); C = pd.read_parquet(P + 'close_raw.parquet')
good = adj.notna().sum(axis=1) >= 300
adj = adj[good]; V = V[good]; C = C[good]
v60 = V.rolling(60).mean()
tr = pd.read_csv(HERE := __file__.replace('check.py', 'trades_A.csv.gz'), dtype={'code': str})
def wilder(x, n=3):
    x = x.dropna(); d = x.diff().dropna(); g = d.clip(lower=0); l = (-d).clip(lower=0)
    ag = g.iloc[0]; al = l.iloc[0]
    for k in range(1, len(d)):
        ag = ag + (g.iloc[k] - ag) / n; al = al + (l.iloc[k] - al) / n
    return 100 - 100 / (1 + ag / al) if al > 0 else 100.0
for sig in ('2013-03-29', '2021-01-08', '2024-06-28'):
    d = pd.Timestamp(sig); i = adj.index.get_loc(d)
    ok = (v60.loc[d] >= 1e7) & (C.loc[d] >= 10) & adj.loc[d].notna() & ~adj.columns.str.startswith('91')
    top300 = v60.loc[d][ok].sort_values(ascending=False).index[:300]
    roc = (adj.iloc[i] / adj.ffill(limit=5).iloc[i - 200] - 1)[top300].dropna().sort_values(ascending=False)
    top10 = list(roc.index[:10])
    rsi = {c: round(wilder(adj[c].iloc[max(0, i - 120):i + 1]), 1) for c in top10}
    nxt = adj.index[i + 1].strftime('%Y-%m-%d')
    bought = sorted(tr[tr.e == nxt].code.tolist())
    print(sig, 'top10', [(c, round(roc[c], 2), rsi[c]) for c in top10])
    print('   RSI<50 in top10:', sorted(c for c in top10 if rsi[c] < 50), ' engine bought on', nxt, bought)
