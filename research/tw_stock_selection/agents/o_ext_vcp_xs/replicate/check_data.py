# -*- coding: utf-8 -*-
"""獨立來源抽查研究員補的最低價／成交股數，以及面板在 Yahoo 段的開高收。python3 -I -B check_data.py"""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import rep_lib as L
HERE = os.path.dirname(os.path.abspath(__file__)) + '/'
SCR = L.S + 'rep_vcp/'
D = L.Data()
of = pd.read_parquet(SCR + 'official_long.parquet')
of['date'] = pd.to_datetime(of['date'])
print(of.market.value_counts().to_dict(), of.valid_bar.value_counts(dropna=False).to_dict())
res = {'official_codes': int(of.code.nunique()), 'official_rows': int(len(of))}
W = {}
for c in ('open', 'high', 'low', 'close', 'volume', 'amount', 'valid_bar'):
    W[c] = of.pivot(index='date', columns='code', values=c)
alld = W['close'].index
gapdays = alld[(alld > '2023-12-29') & (alld < '2024-02-15')]
res['gap_trading_days_in_official'] = [str(x)[:10] for x in gapdays]
A = {c: W[c].reindex(index=D.dates, columns=D.codes).values.astype(float) for c in W}
tick = np.vectorize(L.tick_of, otypes=[float])
def cmp(mask, name):
    m = mask & ~np.isnan(D.C) & ~np.isnan(A['close']) & (A['volume'] > 0)
    n = int(m.sum())
    if n == 0: return dict(n=0)
    o = dict(n=n)
    o['close_same'] = float((np.abs(D.C[m] - A['close'][m]) < 1e-6).mean())
    o['open_same'] = float((np.abs(D.O[m] - A['open'][m]) < 1e-6).mean())
    o['high_same'] = float((np.abs(D.H[m] - A['high'][m]) < 1e-6).mean())
    o['high_lower_than_official'] = float((D.H[m] < A['high'][m] - 1e-6).mean())
    dl = D.L[m] - A['low'][m]
    o['low_same'] = float((np.abs(dl) < 1e-6).mean())
    o['low_higher'] = float((dl > 1e-6).mean()); o['low_lower'] = float((dl < -1e-6).mean())
    o['low_within_1tick'] = float((np.abs(dl) <= tick(A['low'][m]) + 1e-6).mean())
    o['low_abs_err_pct_p99'] = float(np.quantile(np.abs(dl) / A['low'][m], .99))
    r = D.V[m] / A['volume'][m]
    o['vol_same_1share'] = float((np.abs(D.V[m] - A['volume'][m]) <= 1.0).mean())
    o['vol_same_0.1pct'] = float((np.abs(r - 1) <= 0.001).mean())
    o['vol_ratio_med'] = float(np.median(r)); o['vol_ratio_p05'] = float(np.quantile(r, .05)); o['vol_ratio_p95'] = float(np.quantile(r, .95))
    o['vol_ratio_p01'] = float(np.quantile(r, .01)); o['vol_ratio_p99'] = float(np.quantile(r, .99))
    o['vol_err_gt5pct'] = float((np.abs(r - 1) > 0.05).mean()); o['vol_err_gt20pct'] = float((np.abs(r - 1) > 0.20).mean())
    rv = D.VAL[m] / A['amount'][m]
    o['value_ratio_med'] = float(np.median(rv)); o['value_err_gt5pct'] = float((np.abs(rv - 1) > 0.05).mean())
    o['value_ratio_p05'] = float(np.quantile(rv, .05)); o['value_ratio_p95'] = float(np.quantile(rv, .95))
    print(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items()}))
    return o
dt = D.dates.values[:, None] * np.ones((1, D.N), dtype='timedelta64[ns]') if False else None
d2 = np.broadcast_to(D.dates.values[:, None], D.C.shape)
res['src1_2015_2019'] = cmp((D.SRC == 1) & (d2 >= np.datetime64('2015-01-01')), 'src1 Asoul 2015+')
res['src2_k66_2015_2023'] = cmp((D.SRC == 2), 'src2 k66 2015-2023')
res['src3_yahoo'] = cmp((D.SRC == 3), 'src3 Yahoo 2024-02-15~2025-09-23')
res['src3_yahoo_2024'] = cmp((D.SRC == 3) & (d2 < np.datetime64('2025-01-01')), 'src3 Yahoo 2024')
res['src3_yahoo_2025'] = cmp((D.SRC == 3) & (d2 >= np.datetime64('2025-01-01')), 'src3 Yahoo 2025')
res['src4_official'] = cmp((D.SRC == 4), 'src4 2025-09-24+')
res['day_20250924'] = cmp((d2 == np.datetime64('2025-09-24')), '2025-09-24 補值日')
# 面板有 K 棒但官方沒有 / 官方有但面板沒有（2015 起，官方有該檔者）
have_code = ~np.all(np.isnan(A['close']), axis=0)
post = d2 >= np.datetime64('2015-01-05')
offbar = ~np.isnan(A['close']) & (A['volume'] > 0)
res['panel_bar_not_in_official'] = int((~np.isnan(D.C) & ~offbar & post & have_code[None, :]).sum())
res['official_bar_not_in_panel'] = int((np.isnan(D.C) & offbar & post).sum())
res['codes_without_official'] = int((~have_code).sum())
# 斷點股：面板最後一根在 2019-05-16 的，官方後面還有沒有
last = np.array([D.dates[np.nonzero(D.has[:, j])[0][-1]] for j in range(D.N)])
brk = np.nonzero(last == np.datetime64('2019-05-16'))[0]
info = []
for j in brk:
    c = D.codes[j]
    if c in W['close'].columns:
        s = W['close'][c].dropna()
        s2 = s[s.index > '2019-05-16']
        info.append((c, int(len(s2)), str(s2.index[-1])[:10] if len(s2) else None))
    else:
        info.append((c, None, None))
res['breakpoint_codes'] = info
print('breakpoint codes', len(brk), info)
# 以 tsec_all（Asoul 來源，到 2019-05-16）抽查 k66 段（src2，2010～2019-05）的最低價與股數
def roc(s):
    y, m, d = s.split('/')
    return pd.Timestamp(int(y) + 1911, int(m), int(d))
rng = np.random.RandomState(7)
samp = rng.choice(D.codes, 150, replace=False)
tot = dict(n=0, low_same=0, vol_same=0, high_same=0, close_same=0)
pos = {c: i for i, c in enumerate(D.codes)}
di = {d: i for i, d in enumerate(D.dates)}
for c in samp:
    p = L.S + 'tsec_all/%s.csv' % c
    if not os.path.exists(p): continue
    t = pd.read_csv(p, header=None, names=['d', 'vol', 'val', 'o', 'h', 'l', 'c', 'chg', 'n'], dtype=str)
    t['d'] = [roc(x) for x in t['d']]
    for col in ('vol', 'h', 'l', 'c'):
        t[col] = pd.to_numeric(t[col].str.replace(',', ''), errors='coerce')
    t = t[(t.d >= '2010-01-01') & t.c.notna()]
    j = pos[c]
    for d, v, h, l, cl in zip(t.d, t.vol, t.h, t.l, t.c):
        i = di.get(d)
        if i is None or D.SRC[i, j] != 2: continue
        tot['n'] += 1
        tot['low_same'] += abs(D.L[i, j] - l) < 1e-6
        tot['vol_same'] += abs(D.V[i, j] - v) <= max(1.0, v * 1e-6)
        tot['high_same'] += abs(D.H[i, j] - h) < 1e-6
        tot['close_same'] += abs(D.C[i, j] - cl) < 1e-6
res['tsec_vs_k66_src2_sample150'] = {k: (int(v) if k == 'n' else float(v / max(tot['n'], 1))) for k, v in tot.items()}
print('tsec vs k66', res['tsec_vs_k66_src2_sample150'])
json.dump(res, open(HERE + 'data_check.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
