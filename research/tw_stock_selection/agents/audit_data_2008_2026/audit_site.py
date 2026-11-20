#!/usr/bin/env python3
"""項目 7：網站 nav.json／yearly.json／risk.json 對 navs_official.csv.gz 重算（只讀）。python3 -I -B audit_site.py"""
import sys, json, glob; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import alib as A
SITE = Path('/home/user/-/xq_scripts/site')
n = pd.read_csv(SITE / 'navs_official.csv.gz', parse_dates=['date']).set_index('date')
out = {}; chk = []
def add(name, site, re, tol, unit=''):
    ok = abs(site - re) <= tol
    chk.append(dict(item=name, site=site, recalc=round(float(re), 4), diff=round(float(site - re), 4), tol=tol, verdict='通過' if ok else '有問題')); return ok
def cagr(s, a=None, b=None):
    s = s.dropna(); a = a or s.index[0]; b = b or s.index[-1]
    return float((s[b] / s[a]) ** (365.25 / (b - a).days) - 1)
def mdd(s):
    d = s / s.cummax() - 1; tr = d.idxmin(); pk = s[:tr].idxmax(); return float(d.min()), str(pk.date()), str(tr.date())
E, X, B = n['E0|W10_70'], n['XT|W10_70'], n['0050']
d19 = n.index[n.index < '2019-01-01'][-1]
out['recalc'] = dict(first=str(n.index[0].date()), last=str(n.index[-1].date()), rows=int(len(n)), cagr_since2019=cagr(E, d19), cagr_full=cagr(E), maxdd=mdd(E), maxdd_0050=mdd(B),
                     cagr_0050_since2019=cagr(B, d19), cagr_0050_full=cagr(B), final=float(E.iloc[-1]))
risk = {r['項目']: r for r in json.load(open(SITE / 'data/risk.json'))['rows']}
add('risk 年化_全部（2019 起）%', risk['年化_全部']['數值'], out['recalc']['cagr_since2019'] * 100, 0.005)
add('全期（2015-07 起）年化 %（任務給的 34.40）', 34.40, out['recalc']['cagr_full'] * 100, 0.005)
add('risk 回撤_單一路徑 %', risk['回撤_單一路徑']['數值'], out['recalc']['maxdd'][0] * 100, 0.05)
add('risk 0050_回撤 %', risk['0050_回撤']['數值'], out['recalc']['maxdd_0050'][0] * 100, 0.05)
# 滾動 12 個月（252 日）落後
r = (E / E.shift(252) - 1) - (B / B.shift(252) - 1); r = r.dropna()
out['recalc']['roll252'] = dict(worst=float(r.min() * 100), worst_date=str(r.idxmin().date()), frac_behind=float((r < 0).mean() * 100), last=float(r.iloc[-1] * 100))
add('risk 落後0050最多（252 日滾動）個百分點', risk['落後0050最多']['數值'], -r.min() * 100, 0.05)
add('risk 落後時間比例 %', risk['落後時間比例']['數值'], (r < 0).mean() * 100, 0.05)
# 官方引擎交易檔（若有）
tf = sorted(glob.glob(str(A.RS / 'agents/official_nav/*trades*')))
out['official_trades_file'] = tf
if tf:
    t = pd.read_csv(tf[0]); t = t[t['why'] != 'open_end'] if 'why' in t else t
    if 'version' in t: pass
    out['official_trades'] = dict(n=int(len(t)), cols=list(t.columns))
# yearly
ys = {r_['年']: r_ for r_ in json.load(open(SITE / 'data/yearly.json'))['rows']}
ye = n.groupby(n.index.year).last(); p = n.iloc[0]
yrec = {}
for y in ye.index:
    yrec[int(y)] = ((ye.loc[y] / p - 1) * 100).to_dict(); p = ye.loc[y]
bad = 0
for y, r_ in ys.items():
    for lab, col in (('現行策略', 'E0|W10_70'), ('XQ版', 'XT|W10_70'), ('0050', '0050')):
        ok = add(f'yearly {y} {lab} %', r_[lab], yrec[y][col], 0.05); bad += (not ok)
out['yearly_recalc'] = {y: {k: round(v, 3) for k, v in d.items()} for y, d in yrec.items()}
# nav.json
nv = pd.DataFrame(json.load(open(SITE / 'data/nav.json'))['rows']); nv['日期'] = pd.to_datetime(nv['日期'])
nv = nv.set_index('日期')
on_cal = nv.index.isin(n.index)
m = n.reindex(nv.index)
dif = pd.DataFrame({'E': (nv['現行策略'] - m['E0|W10_70']).abs(), 'X': (nv['XQ版'] - m['XT|W10_70']).abs(), 'B': (nv['0050'] - m['0050']).abs()})
out['nav_json'] = dict(rows=int(len(nv)), dates_in_navs_official=int(on_cal.sum()), dates_not_in=[str(d.date()) for d in nv.index[~on_cal]][:10], monotonic=bool(nv.index.is_monotonic_increasing), dup=int(nv.index.duplicated().sum()),
                       max_abs_diff_yuan=dif.max().to_dict(), n_rows_diff_gt_1=int((dif.max(axis=1) > 1).sum()),
                       worst=[dict(date=str(d.date()), **{k: float(v) for k, v in r_.items()}) for d, r_ in dif[dif.max(axis=1) > 1].head(8).iterrows()],
                       is_week_last_trading_day=bool(all(d == n.index[(n.index.isocalendar().week.values == d.isocalendar()[1]) & (n.index.isocalendar().year.values == d.isocalendar()[0])][-1] for d in nv.index[on_cal])))
chk.append(dict(item='nav.json 每列對 navs_official 同日（元）', site='584 列', recalc=f"最大差 現行 {dif['E'].max():.2f}／XQ {dif['X'].max():.2f}／0050 {dif['B'].max():.2f}", diff=None, tol=1.0,
                verdict='通過' if (dif.max().max() <= 1.0 and on_cal.all()) else '有問題'))
# navs_official 的 0050 欄對引擎來源 A 與自建 B
a = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj']
ra = (B / B.iloc[0]) / (a.reindex(n.index).ffill() / a.reindex(n.index).ffill().iloc[0]) - 1
out['navs_0050_vs_jonatw_max_rel'] = float(ra.abs().max())
# 與 S75 重新基準化前的官方值、與 S75 accept_is
acc = json.load(open(A.S75 / 'accept_is.json'))
out['s75_accept_is'] = {k: acc[k] for k in ('since2019', 'full', 'maxdd', 'n_closed', 'final_nav', 'max_abs_diff_vs_navs_official_csv')}
add('S75 accept_is 2019 起年化 對 navs_official 重算 %', acc['since2019'] * 100, out['recalc']['cagr_since2019'] * 100, 1e-6)
add('S75 accept_is 期末淨值 對 navs_official（元）', acc['final_nav'], out['recalc']['final'], 0.01)
out['checks'] = chk; out['n_bad'] = int(sum(c['verdict'] != '通過' for c in chk))
(A.HERE / 'results_site.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
for c in chk: print(c)
print(json.dumps({k: v for k, v in out.items() if k not in ('checks',)}, ensure_ascii=False, indent=1, default=str)[:3500])
