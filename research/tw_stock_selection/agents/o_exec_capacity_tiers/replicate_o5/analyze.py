import sys, os, pickle, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from rstats import *
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
b = load_bench(); sig = signals()
e50 = b['E50'].loc[T0:T1]; idx = e50.index
z = pd.read_parquet(os.path.join(D, 'zero_impact_seed_navs.parquet'))
zr = z.pct_change().fillna(0)
zab = (1 + zr.mean(axis=1)).cumprod()
TN = {3e5: '30萬', 1e6: '100萬', 3e6: '300萬', 1e7: '1000萬'}

def ab_nav(res, nm, tier, mode, seeds=range(40)):
    R = np.array([np.r_[0, res[(nm, s, tier, mode)]['nav'][1:] / res[(nm, s, tier, mode)]['nav'][:-1] - 1] for s in seeds])
    return pd.Series(np.cumprod(1 + R.mean(0)), index=idx), R

def paired(nav):
    d = win_rets(nav, sig) - win_rets(zab, sig)
    se = d.std(ddof=1) / np.sqrt(len(d))
    return d.mean() * 100, d.mean() / se, 2.8 * se * 12 * 100

def row(res, nm, tier, mode, full=True):
    nav, R = ab_nav(res, nm, tier, mode)
    out = dict(cagr=cagr(nav) * 100, t=tvs(nav, e50, sig))
    if not full: return out
    s24 = sig[sig >= '2024-01-01'][0]
    out.update(SEL=tvs(nav, e50, sig, hi='2018-12-31'), VAL=tvs(nav, e50, sig, '2019-01-01', '2023-12-31'),
               HOLD_cagr=cagr(nav.loc[s24:]) * 100, HOLD_t=tvs(nav, e50, sig, '2024-01-01'),
               F08=tvs(nav, e50, sig, '2008-01-01'),
               db1=tvs(nav, e50, sig, drop_best=1), db3=tvs(nav, e50, sig, drop_best=3), db5=tvs(nav, e50, sig, drop_best=5),
               mdd=mdd(nav) * 100)
    ts = []
    for s in range(40):
        n = pd.Series(res[(nm, s, tier, mode)]['nav'], index=idx); ts.append(tvs(n, e50, sig))
    ts = np.array(ts)
    out.update(seed_t_med=np.median(ts), seed_t_min=ts.min(), seed_t_max=ts.max(), seed_lt20=int((ts < 2.0).sum()), seed_lt23=int((ts < 2.3).sum()))
    out['pair_pp_m'], out['pair_t'], out['MDE80'] = paired(nav)
    if mode == 'GROW':
        out['Eend_med'] = np.median([res[(nm, s, tier, mode)]['E_end'] for s in range(40)])
    yrs = (idx[-1] - idx[0]).days / 365.25
    comp = np.mean([res[(nm, s, tier, mode)]['comp'].sum(0) / yrs for s in range(40)], axis=0) * 100
    out.update(cost_tax=comp[0], cost_fee=comp[1], cost_impx=comp[2], cost_base=comp[3], cost_odd=comp[4], turnover=comp[5])
    out['cash_pct'] = np.mean([res[(nm, s, tier, mode)]['cashf'][1:].mean() for s in range(40)]) * 100
    st = {k: sum(res[(nm, s, tier, mode)]['st'][k] for s in range(40)) for k in
          ('new', 'new_oddonly', 'new_pre', 'new_oddonly_pre', 'lot_orders', 'ex10', 'ex20', 'lot_pre19', 'ex10_pre19', 'lot_post19', 'ex10_post19')}
    out.update(oddonly=st['new_oddonly'] / st['new'] * 100, oddonly_pre=st['new_oddonly_pre'] / st['new_pre'] * 100,
               oddonly_post=(st['new_oddonly'] - st['new_oddonly_pre']) / (st['new'] - st['new_pre']) * 100,
               ex10=st['ex10'] / max(st['lot_orders'], 1) * 100, ex20=st['ex20'] / max(st['lot_orders'], 1) * 100,
               ex10_pre19=st['ex10_pre19'] / max(st['lot_pre19'], 1) * 100, ex10_post19=st['ex10_post19'] / max(st['lot_post19'], 1) * 100)
    return out

if __name__ == '__main__':
    nm = sys.argv[1]; full = len(sys.argv) < 3
    res = pickle.load(open(os.path.join(OUT, nm + '.pkl'), 'rb'))
    modes = sorted({k[3] for k in res})
    allout = {}
    for mode in modes:
        for tier in TN:
            if (nm, 0, tier, mode) not in res: continue
            o = row(res, nm, tier, mode, full)
            allout[f'{TN[tier]}|{mode}'] = o
            print(nm, TN[tier], mode, ' '.join(f'{k}={v:.3f}' if isinstance(v, float) else f'{k}={v}' for k, v in o.items()))
    json.dump(allout, open(os.path.join(OUT, nm + '_stats.json'), 'w'), ensure_ascii=False, indent=1, default=float)
