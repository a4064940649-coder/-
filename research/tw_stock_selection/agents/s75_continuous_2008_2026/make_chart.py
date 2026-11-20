import json, sys, pandas as pd, numpy as np
src, out_art, out_repo = sys.argv[1:4]
d = pd.read_csv(src, parse_dates=['date']).set_index('date')
d = d[['V_BRIDGE', 'V_MASK', '0050']].dropna()
def dd_in(s, a, b):
    w = s[a:b]; pk = w.cummax(); r = w / pk - 1; t = r.idxmin(); p = w[:t].idxmax()
    return float(r.min()), p, t
bears = []
for name, a, b in [("2008 金融海嘯", "2008-01-01", "2008-12-31"), ("2011 歐債", "2011-01-01", "2011-12-31"), ("2015", "2015-01-01", "2015-12-31"),
                   ("2018", "2018-01-01", "2018-12-31"), ("2020 疫情", "2020-01-01", "2020-06-30"), ("2022", "2022-01-01", "2022-12-31"),
                   ("2024-07～2025-04", "2024-07-01", "2025-04-30")]:
    e, p, t = dd_in(d['0050'], a, b); s, sp, st = dd_in(d['V_BRIDGE'], a, b); m, _, _ = dd_in(d['V_MASK'], a, b)
    bears.append(dict(n=name, a=str(min(p, sp).date()), b=str(max(t, st).date()), s=round(s * 100, 2), m=round(m * 100, 2), e=round(e * 100, 2)))
w = d.groupby([d.index.isocalendar().year, d.index.isocalendar().week]).tail(1)
if w.index[0] != d.index[0]: w = pd.concat([d.iloc[[0]], w])
yr = {}
for c in d.columns:
    y = d[c].groupby(d.index.year).last(); prev = y.shift(1); prev.iloc[0] = 1_000_000
    yr[c] = {str(k): round(float(v), 2) for k, v in ((y / prev - 1) * 100).items()}
n = (d.index[-1] - d.index[0]).days / 365.25
def mdd(s): return float((s / s.cummax() - 1).min())
st = {c: dict(final=float(d[c].iloc[-1]), cagr=float((d[c].iloc[-1] / 1e6) ** (1 / n) - 1), dd=mdd(d[c])) for c in d.columns}
D = dict(dates=[str(x.date()) for x in w.index], br=[round(float(v)) for v in w['V_BRIDGE']], mk=[round(float(v)) for v in w['V_MASK']],
         etf=[round(float(v)) for v in w['0050']], years=dict(br=yr['V_BRIDGE'], mk=yr['V_MASK'], etf=yr['0050']),
         stats=dict(br=st['V_BRIDGE'], mk=st['V_MASK'], etf=st['0050']), bears=bears, start=str(d.index[0].date()), end=str(d.index[-1].date()))
print(json.dumps(dict(stats=D['stats'], bears=bears), ensure_ascii=False, indent=1))
page = open(sys.argv[4], encoding='utf-8').read().replace('/*DATA*/null', json.dumps(D, ensure_ascii=False, separators=(',', ':')))
open(out_art, 'w', encoding='utf-8').write(page)
open(out_repo, 'w', encoding='utf-8').write('<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head><body>' + page + '</body></html>')
print(len(page))
