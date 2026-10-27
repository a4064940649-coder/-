#!/usr/bin/env python3
"""P4b（買進改隔日收盤）的穩健性：收盤漲停改隔一日開盤遞補、剔除單筆最大、截尾、逐年、分產業、分流動性。凍結後執行。輸出 p4_robust.json。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import olib as L
PR, SH, PT = L.assert_prereg()
K, V = L.K, L.V
IDX, COLS, N = L.IDX, L.COLS, L.N
YL, Y5, _ = V.load_revenue()
S = V.V2('gapfix', YL, Y5, exfin=True)
pl = S.plan(10, 'official')
S52, V60 = V.rolling_signals('gapfix')
def luc(t):
    lim = 0.07 if IDX[t] < pd.Timestamp('2015-06-01') else 0.10
    return np.isfinite(K.Cn[t]) & (K.Cn[t] >= K.limit_up_price(K.ref_price(t), lim) - 1e-6)
rows = []; prev = None
for (e, pk, pool, d) in pl:
    cur = set(pk.tolist())
    if prev is not None:
        _, rk = L.base_pool(IDX.get_loc(d), S52, V60)
        lu = luc(e); s3 = L.SRC3[0] <= IDX[e] <= L.SRC3[1]
        for j in cur - prev:
            O, C = K.On[e, j], K.Cn[e, j]
            imp = O / C - 1
            # 收盤漲停買不到 → 隔一個交易日開盤買（該日開盤相對 e 開盤的價格＝C/O × 隔夜）
            real = imp
            if lu[j] and e + 1 < len(IDX) and np.isfinite(K.On[e + 1, j]):
                on = K.GROW[e + 1, j] * K.On[e + 1, j] / K.Cn[e + 1, j]
                real = 1 / ((C / O) * on) - 1
            rows.append(dict(d=d, code=COLS[j], imp=imp, real=real, lu=bool(lu[j]), src3=s3, liq=rk[j],
                             ind=V.SL['industry'].get(COLS[j], '?') if isinstance(V.SL['industry'].get(COLS[j], '?'), str) else '?'))
    prev = cur
R = pd.DataFrame(rows)
R['per'] = np.where(R.d <= L.SEL_LAST_SIG, 'selection', 'validation')
R = R[R.d <= V.END1012]
out = {}
for per, g0 in R.groupby('per'):
    g = g0[~g0.src3]
    o = {}
    for c in ('imp', 'real'):
        m = g.groupby('d')[c].mean()
        o[c] = {'mean_pct': float(g[c].mean() * 100), 't_month': L.tstat(m.to_numpy()), 'median_pct': float(g[c].median() * 100), 'n': int(len(g))}
    top = g['real'].idxmax()
    gg = g.drop(index=top)
    o['real_ex_top1'] = {'mean_pct': float(gg['real'].mean() * 100), 't_month': L.tstat(gg.groupby('d')['real'].mean().to_numpy()), 'dropped': g.loc[top, 'code'], 'dropped_imp_pct': float(g.loc[top, 'real'] * 100)}
    lo, hi = g['real'].quantile([.01, .99])
    o['real_trim1pct'] = float(g['real'].clip(lo, hi).mean() * 100)
    o['by_year'] = {int(y): round(float(x.mean() * 100), 3) for y, x in g.groupby(g.d.dt.year)['real']}
    o['years_positive'] = int(sum(v > 0 for v in o['by_year'].values())); o['years'] = len(o['by_year'])
    ind = g.groupby('ind')['real'].agg(['mean', 'count'])
    o['by_industry_top'] = {k: [round(float(r['mean'] * 100), 3), int(r['count'])] for k, r in ind.sort_values('count', ascending=False).head(8).iterrows()}
    o['ex_electronics'] = float(g[~g.ind.str.contains('電子|半導體|電腦|光電|通信|資訊', na=False)]['real'].mean() * 100)
    o['by_liq'] = {'top100': float(g[g.liq <= 100]['real'].mean() * 100), 'r101_300': float(g[g.liq > 100]['real'].mean() * 100)}
    o['share_limitup_close'] = float(g['lu'].mean())
    if per == 'validation':
        s3 = g0[g0.src3]
        o['src3_period_only'] = {'mean_pct': float(s3['imp'].mean() * 100), 'n': int(len(s3))}
    out[per] = o
(L.HERE / 'p4_robust.json').write_text(L.jdump(out), encoding='utf-8')
import json; print(json.dumps(L.r4(out), ensure_ascii=False, indent=1))
