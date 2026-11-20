#!/usr/bin/env python3
"""補充：(a) Asoul 對 k66（data_panel src＝2）全市場 2010～2014 開收盤相符率（推論 Asoul 可靠度；2008～2009 沒有第二來源）；(b) Asoul 內部連鎖一致性逐年；
(c) 交易日曆對加權指數檔；(d) S75 淨值檔的標題數字重算；(e) 官方日 K 對 data_panel（k66／steven 段）全市場相符率 2015 起。python3 -I -B audit_misc.py"""
import sys, json; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import alib as A
out = {}
a = pd.read_csv(A.RS / 'raw/pre2015/asoul_twse_2007_2015H1.csv.gz', dtype={'code': str}, parse_dates=['date'])
dp = A.RS / 'data_panel/panel'
src = pd.read_parquet(dp / 'src.parquet'); do = pd.read_parquet(dp / 'open.parquet'); dc = pd.read_parquet(dp / 'close_raw.parquet')
# (a)
x = a[(a['date'] >= '2010-01-04') & (a['date'] <= '2014-12-31') & (a['close'] > 0)]
ii = src.index.get_indexer(x['date']); jj = src.columns.get_indexer(x['code']); ok = (ii >= 0) & (jj >= 0)
x = x[ok]; ii = ii[ok]; jj = jj[ok]
s = src.to_numpy()[ii, jj]; o2 = do.to_numpy()[ii, jj]; c2 = dc.to_numpy()[ii, jj]
m = s == 2
eo = np.abs(o2 - x['open'].to_numpy()) > 0.005; ec = np.abs(c2 - x['close'].to_numpy()) > 0.005
d = pd.DataFrame(dict(date=x['date'].to_numpy()[m], eo=eo[m], ec=ec[m], year=x['date'].dt.year.to_numpy()[m]))
byday = d.groupby('date')[['eo', 'ec']].mean()
out['asoul_vs_k66_2010_2014'] = dict(n_cells=int(m.sum()), open_mismatch=int(d['eo'].sum()), close_mismatch=int(d['ec'].sum()), open_match_rate=float(1 - d['eo'].mean()), close_match_rate=float(1 - d['ec'].mean()),
                                    by_year={int(y): dict(n=int(len(g)), open_mis=int(g['eo'].sum()), close_mis=int(g['ec'].sum())) for y, g in d.groupby('year')},
                                    worst_days={str(k.date()): round(float(v), 4) for k, v in byday['ec'].nlargest(5).items()}, n_days=int(len(byday)), n_days_close_mis_gt_1pct=int((byday['ec'] > 0.01).sum()))
# (b)
a = a.sort_values(['code', 'date']); a['pc'] = a.groupby('code')['close'].shift().where(lambda v: v > 0)
g = a[(a['close'] > 0) & a['change'].notna() & (a['xflag'] == 0) & a['pc'].notna()]
bad = np.abs(g['close'] - g['change'] - g['pc']) > 0.011
bd = bad.groupby(g['date']).mean()
out['asoul_chain'] = dict(by_year={int(y): dict(n=int(len(v)), bad=int(v.sum())) for y, v in bad.groupby(g['date'].dt.year)}, n_days_bad_gt_5pct=int((bd > 0.05).sum()), worst_days={str(k.date()): round(float(v), 4) for k, v in bd.nlargest(5).items()},
                          n_xflag=int(a['xflag'].sum()), xflag_by_year=a[a['xflag'] == 1].groupby(a['date'].dt.year).size().to_dict())
# 2008～2009 漲跌是否在 7% 內（對前收；非 X 日）
g2 = a[(a['close'] > 0) & a['pc'].notna() & (a['xflag'] == 0) & (a['date'] < '2015-06-01')]
mv = g2['close'] / (g2['close'] - g2['change'].fillna(0)) - 1
out['asoul_moves_gt_7p2pct'] = {int(y): int(v) for y, v in (mv.abs() > 0.072).groupby(g2['date'].dt.year).sum().items()}
# (c) 交易日曆
tw = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date'])['date']
nav = pd.read_csv(A.S75 / 'nav_2008_2026.csv.gz', parse_dates=['date']).set_index('date')
off = pd.read_parquet(A.TMP / 'off.parquet', columns=['date']); offd = pd.DatetimeIndex(sorted(off['date'].unique()))
asd = pd.DatetimeIndex(sorted(a['date'].unique()))
cal = asd[asd < A.SEAM].append(offd[offd <= A.END]); cal = cal[cal >= '2008-01-02']
twd = pd.DatetimeIndex(tw[(tw >= '2008-01-02') & (tw <= A.END)])
out['calendar'] = dict(nav_rows=int(len(nav)), my_calendar=int(len(cal)), nav_equals_my_calendar=bool(nav.index.equals(cal)), twii_days=int(len(twd)),
                       in_nav_not_twii=[str(d.date()) for d in nav.index.difference(twd)][:30], in_twii_not_nav=[str(d.date()) for d in twd.difference(nav.index)][:30],
                       weekend_days_in_nav=[str(d.date()) for d in nav.index if d.weekday() >= 5][:40], n_weekend=int(sum(d.weekday() >= 5 for d in nav.index)))
# (d) S75 標題數字
def cagr(s): return float((s.iloc[-1] / s.iloc[0]) ** (365.25 / (s.index[-1] - s.index[0]).days) - 1)
def mdd(s):
    d_ = s / s.cummax() - 1; tr = d_.idxmin(); return float(d_.min()), str(s[:tr].idxmax().date()), str(tr.date())
out['s75_headline'] = {c: dict(final=float(nav[c].iloc[-1]), cagr=cagr(nav[c]), maxdd=mdd(nav[c])) for c in nav.columns}
yr = pd.read_csv(A.S75 / 'yearly.csv').set_index('year'); ye = nav.groupby(nav.index.year).last(); p = nav.iloc[0]; dif = 0
for y in ye.index:
    r_ = ye.loc[y] / p - 1; p = ye.loc[y]
    dif = max(dif, abs(r_['V_BRIDGE'] - yr.at[y, 'V_BRIDGE']), abs(r_['V_MASK'] - yr.at[y, 'V_MASK']), abs(r_['0050'] - yr.at[y, 'e0050']))
out['s75_headline']['yearly_csv_vs_nav_max_abs_diff'] = float(dif)
out['s75_headline']['beat_years_V_BRIDGE'] = int((yr['V_BRIDGE'] > yr['e0050']).sum())
out['s75_headline']['nav_daily_moves_gt_10pct'] = {c: int((nav[c].pct_change().abs() > 0.10).sum()) for c in nav.columns}
out['s75_headline']['max_daily_move'] = {c: [float(nav[c].pct_change().max()), float(nav[c].pct_change().min())] for c in nav.columns}
# (e) 官方日 K 對 data_panel 非 Yahoo 段，全市場 2015 起（上市）
o = pd.read_parquet(A.TMP / 'off.parquet', columns=['date', 'code', 'market', 'open', 'close']); o = o[(o['market'] == 'twse') & (o['close'] > 0) & (o['date'] <= A.END)]
ii = src.index.get_indexer(o['date']); jj = src.columns.get_indexer(o['code']); ok = (ii >= 0) & (jj >= 0)
o = o[ok]; ii = ii[ok]; jj = jj[ok]
s = src.to_numpy()[ii, jj]; o2 = do.to_numpy()[ii, jj]; c2 = dc.to_numpy()[ii, jj]
res = {}
for k, nm in ((1, 'asoul'), (2, 'k66'), (3, 'yahoo5m'), (4, 'steven')):
    m = (s == k) & np.isfinite(c2)
    if m.sum():
        eo = np.abs(o2[m] - o['open'].to_numpy()[m]) > 0.005; ec = np.abs(c2[m] - o['close'].to_numpy()[m]) > 0.005
        res[nm] = dict(n=int(m.sum()), open_mis=int(np.nansum(eo)), close_mis=int(ec.sum()), open_match=float(1 - np.nanmean(eo)), close_match=float(1 - ec.mean()))
out['official_vs_data_panel_2015on'] = res
(A.HERE / 'results_misc.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
