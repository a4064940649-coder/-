#!/usr/bin/env python3
"""52 週高點因子的實務版本：流動性前 300 名中，依 收盤/52週高 排序，只買前 30 名，等權，月底換股；含成本 0.5%／1%；對照 0050 含息。"""
import json, math, glob, os, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, '/home/user/-/research/tw_rule_edges')
from cross_sectional_factors import load_one
d = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/tsec_all'; out = Path('/home/user/-/research/tw_rule_edges/output')
closes, vals, exd = {}, {}, {}
for p in sorted(glob.glob(os.path.join(d, '*.csv'))):
    code = Path(p).stem
    if not (code.isdigit() and len(code) == 4): continue
    try: df = load_one(p)
    except Exception: continue
    if df is None or len(df) < 300: continue
    closes[code] = df['c']; vals[code] = df['val']; exd[code] = df['exday']
C = pd.DataFrame(closes).sort_index().loc[:'2019-04-30']; V = pd.DataFrame(vals).reindex(C.index); X = pd.DataFrame(exd).reindex(C.index).fillna(False).astype(bool)
R = C.pct_change(); R = R.where(R.abs() < 0.30).mask(X, 0.0)
Cadj = ((1 + R.fillna(0)).cumprod() * C.bfill().iloc[0]).where(C.notna())
me = pd.DatetimeIndex(C.groupby([C.index.year, C.index.month]).apply(lambda x: x.index[-1]).values)
Cm = Cadj.reindex(me).ffill(limit=5); Rm = Cm.pct_change().shift(-1); Vm = V.rolling(60).mean().reindex(me); Pm = C.reindex(me).ffill(limit=5)
S = (Cadj / Cadj.rolling(252).max()).reindex(me)
vol60 = R.rolling(60).std().reindex(me)
e50 = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj']
e50m = e50.reindex(me, method='ffill').pct_change().shift(-1)
def ann(r):
    r = r.dropna(); yrs = len(r) / 12; eq = (1 + r).cumprod(); vol = r.std() * math.sqrt(12); cagr = eq.iloc[-1] ** (1 / yrs) - 1
    return {'CAGR': round(float(cagr), 4), 'vol': round(float(vol), 4), 'sharpe': round(float(cagr / vol), 2), 'maxDD': round(float((eq / eq.cummax() - 1).min()), 4), 'months': int(len(r))}
def run(start, end, n_hold=30, topn=300, lowvol_filter=False):
    port, bench, to = {}, {}, []; prev = set()
    for dt in me:
        if dt < pd.Timestamp(start) or dt > pd.Timestamp(end) or dt not in Rm.index or pd.isna(e50m.get(dt, np.nan)): continue
        elig = (Vm.loc[dt] >= 1e7) & (Pm.loc[dt] >= 10) & S.loc[dt].notna() & Rm.loc[dt].notna()
        liq = Vm.loc[dt][elig].sort_values(ascending=False); cand = liq.index[:topn]
        s = S.loc[dt][cand].dropna()
        if lowvol_filter:
            v = vol60.loc[dt][s.index]; s = s[v <= v.median()]      # 先剔除高波動一半
        pick = s.sort_values(ascending=False).index[:n_hold]
        port[dt] = Rm.loc[dt][pick].mean(); bench[dt] = e50m.loc[dt]
        to.append(len(set(pick) - prev) / n_hold); prev = set(pick)
    port, bench = pd.Series(port), pd.Series(bench); tov = float(np.mean(to))
    o = {'turnover': round(tov, 3), 'gross': ann(port), 'net_0.5pct': ann(port - tov * 0.005), 'net_1pct': ann(port - tov * 0.01), '0050_TR': ann(bench)}
    ex = (port - tov * 0.005) - bench; o['net_minus_0050_monthly_pct'] = round(float(ex.mean() * 100), 3); o['t'] = round(float(ex.mean() / (ex.std() / math.sqrt(len(ex)))), 2)
    net = port - tov * 0.005
    o['yearly_net_pct'] = {int(y): round(float(((1 + g).prod() - 1) * 100), 1) for y, g in net.groupby(net.index.year)}
    o['yearly_0050_pct'] = {int(y): round(float(((1 + g).prod() - 1) * 100), 1) for y, g in bench.groupby(bench.index.year)}
    o['years_beating_0050'] = int(sum(o['yearly_net_pct'][y] > o['yearly_0050_pct'][y] for y in o['yearly_net_pct']))
    return o
res = {'top30_of_top300_2005_2019': run('2005-01-01', '2019-04-30'), 'top30_of_top300_lowvolhalf_2005_2019': run('2005-01-01', '2019-04-30', lowvol_filter=True),
       'top50_of_top500_2005_2019': run('2005-01-01', '2019-04-30', n_hold=50, topn=500), 'top30_of_top300_2010_2019': run('2010-01-01', '2019-04-30')}
(out / 'factor_52w_top30.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
for k, v in res.items():
    print(k, '| to', v['turnover'], '| gross', v['gross']['CAGR'], '| net0.5', v['net_0.5pct'], '| net1', v['net_1pct']['CAGR'], '| 0050TR', v['0050_TR'], '| ex %/mo', v['net_minus_0050_monthly_pct'], 't', v['t'], '| yrs beat', v['years_beating_0050'], '/', len(v['yearly_net_pct']))
print('yearly net:', res['top30_of_top300_2005_2019']['yearly_net_pct']); print('yearly 0050:', res['top30_of_top300_2005_2019']['yearly_0050_pct'])
