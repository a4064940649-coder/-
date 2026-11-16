#!/usr/bin/env python3
"""對帳診斷（只上市、還原值、H21、同檔 strict、單邊 0.6%）：研究中心訊號以外，XQ 可能多出的交易從哪裡來。
(1) 腳本其他條件都符合（腳本股池、季線上、RET20 > 10%）但研究中心 SURGE 缺值的股票（營收檔沒有：KY、金融、下市等）；
(2) 2016-01 訊號月（營收月 2015-12：研究中心因 2013 年 IFRS 斷點把 D 設缺值而沒有；XQ 有）；
(3) 創新板（若使用者沒排除）。把這些加進訊號後重算筆數、勝率、平均。python3 -I -B diag.py → diag.json。"""
import sys, json, gc
from pathlib import Path
import numpy as np, pandas as pd

H = Path(__file__).resolve().parent
AG = H.parent
XT = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/xqrecon')
sys.path.insert(0, str(AG / 's19_xq_threshold'))
import s19lib                                                       # noqa: E402
from s19lib import s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib   # noqa: E402
for mod, a in [(s19lib, 'S19'), (s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'), (s7lib, 'S7'), (s6lib, 'S6'), (s5lib, 'S5'),
               (s4lib, 'S4'), (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP')]:
    setattr(mod, a, XT)
F = 0.006
P = s19lib.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22')
INNOV, innov_names = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
UX, Cadj, liqX = s19lib.script_universe(P, INNOV)
with np.errstate(invalid='ignore'):
    UXI = (P.MK == 1) & P.not91[None, :] & P.has & (Cadj >= 10.0) & (liqX >= 1e7)        # 不排除創新板
del Cadj, liqX
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'NEARHI', 'V60', 'liq', 'LIMUP', 'LIMDN', 'REF', 'F_MA20', 'AMT', 'VOLF'):
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
S, _ = s19lib.rev_std(P)
months = list(S.index)
avB, _ = s19lib.b11_rows(P, months, 11)
names = pd.read_csv(s4lib.STOCK_LIST, dtype={'code': str}).drop_duplicates('code').set_index('code')['name'].to_dict()
names.update(innov_names)

# 2016-01 訊號月：營收月 2015-12 的 SURGE（不把 2013 年的 D 設缺值；其餘同 rev_std）
r = pd.read_csv(stlib.REV_CSV, dtype={'code': str}, parse_dates=['month'])
r['pri'] = (r['market'] != 'sii').astype(int)
r = r.sort_values(['month', 'code', 'pri']).drop_duplicates(['month', 'code'], keep='first')
Rv = r.pivot(index='month', columns='code', values='revenue_k_twd').sort_index()
Rv.index = pd.PeriodIndex(Rv.index, freq='M')
Rv = Rv.reindex(pd.period_range(Rv.index.min(), Rv.index.max(), freq='M')).reindex(columns=P.COLS).where(lambda x: x > 0)
D = Rv - Rv.shift(12)
m0 = pd.Period('2015-12', 'M')
W = D.loc[m0 - 24:m0 - 1].to_numpy(); d0 = D.loc[m0].to_numpy()
mu = W.mean(axis=0); sd = W.std(axis=0, ddof=1)
with np.errstate(invalid='ignore', divide='ignore'):
    s1512 = np.where(np.isfinite(W).all(axis=0) & (sd > 1e-9 * (np.abs(W).mean(axis=0) + 1e-12)), (d0 - mu) / sd, np.nan)
k1512 = s9lib.signal_rows(P, [m0], 'B', R=11)[0][m0]
del r, Rv, D, W

rows = []   # (k, j, surge, src)
for m in months + [m0]:
    k = avB[m]
    if k >= P.T or k < 61:
        continue
    s = s1512 if m == m0 else S.loc[m].to_numpy()
    if m != m0 and int((U[k] & np.isfinite(s)).sum()) < 50:
        continue
    with np.errstate(invalid='ignore'):
        pre = P.F_MA60[k] & (P.RET20[k] > 0.10)
        fin = np.isfinite(s)
        for j in np.flatnonzero(UX[k] & pre & fin & (np.where(fin, s, -1e9) >= 2.5)):
            rows.append((k, j, s[j], 'base' if m != m0 else 'm201601'))
        for j in np.flatnonzero(UX[k] & pre & ~fin):
            rows.append((k, j, np.nan, 'nan_surge' if m != m0 else 'nan_surge_m201601'))
        for j in np.flatnonzero(UXI[k] & ~UX[k] & pre):
            rows.append((k, j, s[j] if fin[j] else np.nan, 'innov'))
sg = pd.DataFrame(rows, columns=['k', 'j', 's', 'src'])
sg['code'] = P.COLS[sg['j'].to_numpy()]
sg['name'] = sg['code'].map(names)
sg['ky'] = sg['name'].fillna('').str.contains('KY')
sg['fin'] = sg['code'].str[:2].isin(['28', '58']) | sg['code'].isin(['2801', '2809', '2812', '2816', '2820', '2832', '2834', '2836', '2838', '2845', '2849', '2850', '2851', '2852', '2855', '2867', '2880', '2881', '2882', '2883', '2884', '2885', '2886', '2887', '2888', '2889', '2890', '2891', '2892', '2897', '5876', '5880', '6005', '6024'])

BARS = {}


def trades(sig):
    last = {}; out = []
    for k, j, s, src in sorted(sig, key=lambda z: (z[0], -(z[2] if np.isfinite(z[2]) else -9))):
        b = BARS.setdefault(j, np.flatnonzero(P.has[:, j] & np.isfinite(P.O[:, j]) & (P.O[:, j] > 0)))
        p = int(np.searchsorted(b, k, side='right'))
        if p >= len(b) or last.get(j, -1) > k:
            continue
        e = int(b[p])
        if p + 21 < len(b):
            x = int(b[p + 21]); why = 'hold'; px = P.AO[x, j]
        else:
            lastb = int(b[-1])
            if not P.MK[lastb + 1:, j].any() and lastb < P.T - 1:
                x = lastb; why = 'delist'
            else:
                x = P.T - 1; why = 'open_end'
            px = P.G[x, j]
        out.append((k, e, x, j, src, why, (px / P.AO[e, j]) * (1 - F) / (1 + F) - 1))
        last[j] = x
    return pd.DataFrame(out, columns=['k', 'e', 'x', 'j', 'src', 'why', 'r'])


def st(t):
    c = t[t['why'] != 'open_end']
    return dict(n=int(len(c)), n_open=int((t['why'] == 'open_end').sum()), win=float((c['r'] > 0).mean()), mean=float(c['r'].mean()),
                max=float(c['r'].max()), min=float(c['r'].min()), sum=float(c['r'].sum()))


res = {}
tup = lambda d: list(d[['k', 'j', 's', 'src']].itertuples(index=False, name=None))
base = sg[sg['src'] == 'base']
res['base'] = st(trades(tup(base)))
for extra in (['m201601'], ['nan_surge'], ['nan_surge', 'nan_surge_m201601'], ['innov'], ['m201601', 'nan_surge', 'nan_surge_m201601', 'innov']):
    t = trades(tup(sg[sg['src'].isin(['base'] + extra)]))
    res['base+' + '+'.join(extra)] = dict(all=st(t), extra_only=st(t[t['src'].isin(extra)]) if t['src'].isin(extra).any() else None)
nn = sg[sg['src'].str.startswith('nan_surge')]
res['nan_surge_pool'] = dict(n_rows=int(len(nn)), n_codes=int(nn['code'].nunique()), n_ky_rows=int(nn['ky'].sum()), n_fin_rows=int(nn['fin'].sum()),
                             n_no_name_rows=int(nn['name'].isna().sum()),
                             by_year=nn.assign(y=pd.DatetimeIndex(P.IDX[nn['k'].to_numpy()]).year).groupby('y').size().to_dict(),
                             top_codes=nn.groupby(['code']).size().sort_values(ascending=False).head(25).rename(lambda c: f'{c} {names.get(c, "?")}').to_dict())
inn = sg[sg['src'] == 'innov']
res['innov_pool'] = dict(n_rows=int(len(inn)), rows=[f'{P.IDX[k].date()} {c} {names.get(c, "?")} surge={s:.2f}' for k, c, s in zip(inn['k'], inn['code'], inn['s'])][:40])
res['m201601'] = dict(sig_date=str(P.IDX[k1512].date()), n_base=int((sg['src'] == 'm201601').sum()), n_nan=int((sg['src'] == 'nan_surge_m201601').sum()))
# 研究中心訊號裡淨報酬最接近 XQ 最大虧損 −41.64% 的交易（全部候選，含 SURGE < 2.5 的）
allpre = []
for m in months:
    k = avB[m]
    if k >= P.T or k < 61:
        continue
    s = S.loc[m].to_numpy()
    with np.errstate(invalid='ignore'):
        for j in np.flatnonzero(UXI[k] & P.F_MA60[k] & (P.RET20[k] > 0.10)):
            allpre.append((k, j, s[j], 'pre'))
tp = trades(allpre)
tp['code'] = P.COLS[tp['j'].to_numpy()]; tp['date'] = P.IDX[tp['e'].to_numpy()].strftime('%Y-%m-%d')
tp['surge'] = [S.loc[[m for m in months if avB[m] == k][0]].iloc[j] if any(avB[m] == k for m in months) else np.nan for k, j in zip(tp['k'], tp['j'])]
cl = tp.iloc[(tp['r'] + 0.4164).abs().argsort()[:6]]
res['closest_to_xq_min'] = [f'{d} {c} {names.get(c, "?")} r={r:.4f} surge={s:.2f}' for d, c, r, s in zip(cl['date'], cl['code'], cl['r'], cl['surge'])]
cl = tp.iloc[(tp['r'] - 1.1836).abs().argsort()[:3]]
res['closest_to_xq_max'] = [f'{d} {c} {names.get(c, "?")} r={r:.4f} surge={s:.2f}' for d, c, r, s in zip(cl['date'], cl['code'], cl['r'], cl['surge'])]
tn = trades(tup(sg[sg['src'].str.startswith('nan_surge')]))
tn['code'] = P.COLS[tn['j'].to_numpy()]; tn['date'] = P.IDX[tn['e'].to_numpy()].strftime('%Y-%m-%d')
cl = tn.iloc[(tn['r'] + 0.4164).abs().argsort()[:5]]
res['nan_pool_closest_to_xq_min'] = [f'{d} {c} {names.get(c, "?")} r={r:.4f}' for d, c, r in zip(cl['date'], cl['code'], cl['r'])]
# 不擋同檔重複（每個訊號都進）
lastfree = []
for k, j, s_, src in tup(base):
    b = BARS[j]; p = int(np.searchsorted(b, k, side='right'))
    if p < len(b):
        lastfree.append(p + 21 < len(b))
res['base_no_block'] = dict(n_signals=int(len(base)), n_closed=int(sum(lastfree)))
(H / 'diag.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
