#!/usr/bin/env python3
"""52 週高點因子穩健性：流動性前 300 名 vs 全體、成交值加權 vs 等權、子期間、逐年、成本敏感度。資料與處理同 cross_sectional_factors.py。"""
import json, math, glob, os, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, '/home/user/-/research/tw_rule_edges')
from cross_sectional_factors import load_one

d = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/tsec_all'; out = Path('/home/user/-/research/tw_rule_edges/output'); out.mkdir(exist_ok=True)
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
twii = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date']).set_index('date')['close']
twii_m = twii.reindex(me, method='ffill').pct_change().shift(-1)
def ann(r):
    r = r.dropna(); yrs = len(r) / 12; eq = (1 + r).cumprod(); vol = r.std() * math.sqrt(12); cagr = eq.iloc[-1] ** (1 / yrs) - 1
    return {'CAGR': round(float(cagr), 4), 'vol': round(float(vol), 4), 'sharpe': round(float(cagr / vol), 2), 'maxDD': round(float((eq / eq.cummax() - 1).min()), 4), 'months': int(len(r))}
def run(start, end, universe='all', weighting='EW', topn=300, rt_cost=0.005, nq=5):
    q5, q1, ew, bench, to = {}, {}, {}, {}, []; prev = set()
    for dt in me:
        if dt < pd.Timestamp(start) or dt > pd.Timestamp(end) or dt not in Rm.index: continue
        elig = (Vm.loc[dt] >= 1e7) & (Pm.loc[dt] >= 10) & S.loc[dt].notna() & Rm.loc[dt].notna()
        if universe == 'top300':
            liq = Vm.loc[dt][elig].sort_values(ascending=False); elig = elig & elig.index.isin(liq.index[:topn])
        s = S.loc[dt][elig]
        if s.count() < 60: continue
        ranks = pd.qcut(s.rank(method='first'), nq, labels=False); r = Rm.loc[dt][elig]; w = Vm.loc[dt][elig]
        if weighting == 'EW':
            q5[dt] = r[ranks == nq - 1].mean(); q1[dt] = r[ranks == 0].mean(); ew[dt] = r.mean()
        else:
            w5 = w[ranks == nq - 1]; q5[dt] = float((r[ranks == nq - 1] * w5).sum() / w5.sum()); w1 = w[ranks == 0]; q1[dt] = float((r[ranks == 0] * w1).sum() / w1.sum()); ew[dt] = float((r * w).sum() / w.sum())
        top = set(s.index[ranks == nq - 1]); to.append(len(top - prev) / max(len(top), 1)); prev = top
        bench[dt] = twii_m.loc[dt]
    q5, q1, ew, bench = pd.Series(q5), pd.Series(q1), pd.Series(ew), pd.Series(bench)
    tov = float(np.mean(to)); net = q5 - tov * rt_cost
    ex = q5 - ew; t_ex = float(ex.mean() / (ex.std() / math.sqrt(len(ex))))
    return {'n_months': int(len(q5)), 'avg_names_top': None, 'turnover': round(tov, 3), 'Q5_gross': ann(q5), 'Q5_net': ann(net), 'Q1': ann(q1), 'universe': ann(ew), 'TAIEX_price': ann(bench),
            'Q5_minus_universe_monthly_pct': round(float(ex.mean() * 100), 3), 't': round(t_ex, 2), 'Q5net_minus_TAIEX_CAGR_pp': round((ann(net)['CAGR'] - ann(bench)['CAGR']) * 100, 2),
            'yearly_Q5_net_pct': {int(y): round(float(((1 + g).prod() - 1) * 100), 1) for y, g in net.groupby(net.index.year)},
            'yearly_TAIEX_pct': {int(y): round(float(((1 + g).prod() - 1) * 100), 1) for y, g in bench.groupby(bench.index.year)}}
res = {}
res['all_EW_2005_2019'] = run('2005-01-01', '2019-04-30')
res['all_EW_2005_2019_cost1pct'] = run('2005-01-01', '2019-04-30', rt_cost=0.01)
res['top300_EW_2005_2019'] = run('2005-01-01', '2019-04-30', universe='top300')
res['top300_VW_2005_2019'] = run('2005-01-01', '2019-04-30', universe='top300', weighting='VW')
res['all_VW_2005_2019'] = run('2005-01-01', '2019-04-30', weighting='VW')
res['all_EW_2005_2009'] = run('2005-01-01', '2009-12-31')
res['all_EW_2010_2014'] = run('2010-01-01', '2014-12-31')
res['all_EW_2015_2019'] = run('2015-01-01', '2019-04-30')
res['top300_EW_decile10_2005_2019'] = run('2005-01-01', '2019-04-30', universe='top300', nq=10)
(out / 'factor_52w_robustness.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
for k, v in res.items():
    print(k, '| n', v['n_months'], '| Q5 gross', v['Q5_gross']['CAGR'], 'net', v['Q5_net']['CAGR'], 'DD', v['Q5_net']['maxDD'], '| univ', v['universe']['CAGR'], '| TAIEX', v['TAIEX_price']['CAGR'], '| Q5-univ %/mo', v['Q5_minus_universe_monthly_pct'], 't', v['t'], '| to', v['turnover'])
print('yearly Q5 net (top300 EW):', res['top300_EW_2005_2019']['yearly_Q5_net_pct']); print('yearly TAIEX:', res['top300_EW_2005_2019']['yearly_TAIEX_pct'])
