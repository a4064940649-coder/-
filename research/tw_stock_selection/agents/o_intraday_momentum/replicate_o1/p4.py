import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import json
import numpy as np, pandas as pd
import rlib as R
S = R.main_strategy(); pl = R.plan(10); t0, t1, sd = S.bounds()
IDX = R.IDX; Cn, On = R.Cn, R.On
S3a, S3b = pd.Timestamp('2024-02-15'), pd.Timestamp('2025-09-23')
rows = []
held = set()
for i, (e, pk, pool, d) in enumerate(pl):
    pk = set(int(j) for j in pk)
    new = pk - held; ex = held - pk
    t = e
    for j in new:
        o, c = On[t, j], Cn[t, j]
        nx = t + 1
        rows.append(dict(sig=d, t=t, side='buy', j=j, oc=c / o - 1, imp_c=o / c - 1, imp_tw=o / ((o + c) / 2) - 1,
                         lu=bool(R.close_limit_up(t)[j]), imp_next=(o / On[nx, j] * 1 - 1) if np.isfinite(On[nx, j]) else np.nan,
                         first=(i == 0)))
    for j in ex:
        o, c = On[t, j], Cn[t, j]
        if not (np.isfinite(o) and np.isfinite(c)): continue
        rows.append(dict(sig=d, t=t, side='sell', j=j, oc=c / o - 1, imp_c=c / o - 1, imp_tw=((o + c) / 2) / o - 1, lu=False, imp_next=np.nan, first=False))
    held = pk
df = pd.DataFrame(rows)
df['date'] = IDX[df['t']]
df['src3'] = (df['date'] >= S3a) & (df['date'] <= S3b)
# 隔一日開盤買的改善要把隔日 C→O 跳空的含息報酬納入：買在 t 收盤 vs t+1 開盤
G = R.G
def nextopen_imp(r):
    t, j = r.t, r.j; nx = t + 1
    o, c = On[t, j], Cn[t, j]
    if not np.isfinite(On[nx, j]): return np.nan
    # 開盤買到 t+1 開盤的財富：c/o × G[nx]×O[nx]/C[nx]；改成 t+1 開盤買：1
    w_open = c / o * G[nx, j] * On[nx, j] / Cn[nx, j]
    return 1 / w_open - 1   # 正＝改到隔日開盤買比 t 開盤買好
df['imp_nextopen'] = [nextopen_imp(r) for r in df.itertuples()]
def summ(x, col):
    m = x.groupby('sig')[col].mean()
    return {'mean_pct': round(float(x[col].mean()) * 100, 3), 'n': int(x[col].notna().sum()), 'month_t': round(R.tstat(m), 2), 'n_m': int(m.notna().sum())}
out = {}
P = {'SEL': (pd.Timestamp('2006-04-01'), pd.Timestamp('2019-04-30')), 'VAL': (pd.Timestamp('2019-05-01'), pd.Timestamp('2026-10-12'))}
for k, (a, b) in P.items():
    x = df[(df.sig >= a) & (df.sig <= b)]
    vs = {'all': x}
    if k == 'VAL': vs = {'ex_src3': x[~x.src3], 'src3_only': x[x.src3], 'all': x}
    for vn, y in vs.items():
        o = {}
        for nofirst in (False, True):
            z = y[~y['first']] if nofirst else y
            tag = 'nofirst' if nofirst else 'withfirst'
            bu, se = z[z.side == 'buy'], z[z.side == 'sell']
            o[tag] = {'n_buy': len(bu), 'n_sell': len(se),
                      'merged_close': summ(z, 'imp_c'), 'merged_twap': summ(z, 'imp_tw'),
                      'buy_oc': summ(bu, 'oc'), 'buy_close': summ(bu, 'imp_c'), 'buy_twap': summ(bu, 'imp_tw'),
                      'sell_oc': summ(se, 'oc'), 'sell_close': summ(se, 'imp_c'),
                      'buy_close_ex_LU': summ(bu[~bu.lu], 'imp_c'), 'n_buy_LU': int(bu.lu.sum())}
            # 收盤漲停改隔一日開盤買：該筆改善＝隔日開盤買相對 t 開盤
            bb = bu.copy(); bb['imp_mix'] = np.where(bb.lu, bb['imp_nextopen'], bb['imp_c'])
            o[tag]['buy_close_LUnext'] = summ(bb, 'imp_mix')
            if len(bb) > 1:
                drop = bb['imp_mix'].idxmax(); o[tag]['buy_close_LUnext_dropmax'] = summ(bb.drop(drop), 'imp_mix')
            yr = bb.groupby(bb.date.dt.year)['imp_mix'].mean()
            o[tag]['yearly_buy_LUnext_pos'] = f'{int((yr > 0).sum())}/{len(yr)}'
            yr2 = bu.groupby(bu.date.dt.year)['imp_c'].mean()
            o[tag]['yearly_buy_close_pos'] = f'{int((yr2 > 0).sum())}/{len(yr2)}'
        out[f'{k}|{vn}'] = o
print(json.dumps(out, indent=1, ensure_ascii=False))
json.dump(out, open(R.HERE / 'p4.json', 'w'), ensure_ascii=False, indent=1)
