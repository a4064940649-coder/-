import numpy as np, pandas as pd, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rlib
C = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'calib.json')))['6.1414']
P = lambda s: pd.Period(s, 'M')
med = rlib.decision_line(); med_st = rlib.decision_line(cell='K5T|stress|100萬|FIX')
e = pd.read_csv(os.path.join(rlib.RAW, 'finlab/data/raw/etf_0050.csv'), parse_dates=['date']).set_index('date')['adj']
pl = pd.read_csv(os.path.join(rlib.RAW, 'v7/plan_V7_B10.csv'), parse_dates=['signal'])
sig = list(pl.signal) + [pd.Timestamp('2026-10-12')]
nv = pd.read_parquet(os.path.join(rlib.RAW, 'v7/navs_v7.parquet'))
months = [s.to_period('M') for s in sig[:-1]]
def wret(series, a, b): return (series.asof(b) / series.asof(a) - 1) * 100
r50 = pd.Series([wret(e, a, b) for a, b in zip(sig[:-1], sig[1:])], index=months)
off = pd.Series([wret(nv['V7_B10|normal'], a, b) for a, b in zip(sig[:-1], sig[1:])], index=months)
ew = pd.Series([wret(nv['V7_B10|EW'], a, b) for a, b in zip(sig[:-1], sig[1:])], index=months)
def ann(r):
    r = np.asarray(r) / 100; return (np.prod(1 + r) ** (12 / len(r)) - 1) * 100
def mdd(r):
    w = np.cumprod(1 + np.asarray(r) / 100); w = np.concatenate([[1], w]); return (w / np.maximum.accumulate(w) - 1).min() * 100
def r2_path(x, sig_):
    n = np.arange(1, len(x) + 1); z = np.cumsum(x) / (sig_ * np.sqrt(n)); act = None
    for t in range(12, len(x) + 1, 6):
        if z[t - 1] >= C['c_u'] * np.sqrt(36 / t): act = ('up', t); break
        if z[t - 1] <= -C['c_d']: act = ('down', t); break
    return z, act
def run(start, end, sig_, label):
    idx = [m for m in med.index if P(start) <= m <= P(end)]
    x = med.loc[idx].values; xs = med_st.loc[idx].values
    z, act = r2_path(x, sig_)
    main = r50.loc[idx].values + x; main_st = r50.loc[idx].values + xs
    w30 = np.full(len(x), 0.3); w100 = np.ones(len(x))
    if act:
        k = act[1] + 1  # effective from month n+2 (0-based index n+1)
        if act[0] == 'up': w30[k:] = 1.0
        else: w30[k:] = 0; w100[k:] = 0
    b = r50.loc[idx].values
    p30 = w30 * main + (1 - w30) * b; p100 = w100 * main + (1 - w100) * b
    p30s = w30 * main_st + (1 - w30) * b
    res = dict(n=len(x), mean_ex=x.mean(), sd=x.std(ddof=1), z_end=z[-1], action=(act[0], str(idx[act[1] - 1]), act[1]) if act else None,
               P30=(ann(p30), mdd(p30)), P100=(ann(p100), mdd(p100)), main=(ann(main), mdd(main)), e50=(ann(b), mdd(b)),
               EW=ann(ew.loc[idx]), official0=ann(off.loc[idx]), P30_stress=ann(p30s), main_stress=ann(main_st))
    d = b - main  # main vs 0050 t
    ex = main - b; res['t_vs_0050'] = ex.mean() / (ex.std(ddof=1) / np.sqrt(len(ex)))
    if start == '2019-05':
        no21 = [i for i, m in enumerate(idx) if m.year != 2021]
        res['z_end_no2021'] = x[no21].sum() / (sig_ * np.sqrt(len(no21)))
        imax = np.argmax(x); keep = [i for i in range(len(x)) if i != imax]
        res['max_month'] = (str(idx[imax]), x[imax]); res['z_end_nomax'] = x[keep].sum() / (sig_ * np.sqrt(len(keep)))
    print(label, json.dumps(res, default=lambda o: round(float(o), 3) if not isinstance(o, str) else o))
    return res
if __name__ == '__main__':
    out = {}
    sel = med[med.index <= P('2019-04')]
    out['A'] = run('2019-05', '2026-09', rlib.hac_sd(sel), 'from 2019-05, sigma=SEL HAC %.3f' % rlib.hac_sd(sel))
    s24 = max(rlib.hac_sd(med[med.index < P('2024-01')]), rlib.hac_sd(med[(med.index >= P('2019-05')) & (med.index < P('2024-01'))]))
    out['B'] = run('2024-01', '2026-09', s24, 'from 2024-01, sigma=%.3f' % s24)
    json.dump(out, open('pf.json', 'w'), default=str, indent=1)
