import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rlib, numpy as np, pandas as pd
P = rlib.load_panel(); TD = rlib.td_index(); close = P['close_raw'].ffill(); S = rlib.survivors()
o8 = pd.read_parquet(os.path.join(rlib.RAW, 'center', 'o8_shares_pit.parquet')); o8.index = pd.DatetimeIndex(o8.index).as_unit('ns')
o12 = pd.read_parquet(os.path.join(rlib.RAW, 'center', 'o12_shares_full.parquet')); o12.index = pd.DatetimeIndex(o12.index).as_unit('ns')
AD = pd.Timestamp('2021-03-31')
anc = o8.loc[AD].copy()
for c in ['4763', '2317']:
    anc[c] = o12.loc[AD, c]
anc = anc.dropna()
# official events
tw = pd.concat([pd.read_csv(f, dtype={'code': str}) for f in sorted(glob.glob(os.path.join(rlib.RAW, 'twt49u', '*.csv.gz')))])
tw = tw[tw.code.str.fullmatch(r'\d{4}')]
tw['date'] = pd.to_datetime(tw.date)
tw = tw[(tw.date >= '2014-01-01') & (tw.date <= AD) & (tw.ref_price > 0)]
tw['r'] = tw.prev_close / tw.ref_price
div = tw[tw.type == '息']
mkt = div.groupby(div.date.dt.year).r.median() - 1
def yhat(code, d):
    g = div[(div.code == code) & (div.date >= d - pd.DateOffset(years=3)) & (div.date <= d + pd.DateOffset(years=3))]
    return (g.r.median() - 1) if len(g) else mkt.get(d.year, mkt.median())
ev = []
for _, r in tw[tw.type.isin(['權', '權息'])].iterrows():
    m = r.r if r.type == '權' else r.r / (1 + yhat(r.code, r.date))
    if m <= 1.02: m = 1.0
    ev.append((r.date, r.code, min(max(m, 0.2), 3.0), 'tw'))
E = rlib.events()
E = E[(E.date >= '2014-01-01') & (E.date <= AD)]
keys = set(zip(tw.date, tw.code))
f3 = E[(E.flag == 3) & (E.m != 1)]
f3x = f3[[(d, c) not in keys for d, c in zip(f3.date, f3.code)]]
ev += [(d, c, m, 'f3') for d, c, m in zip(f3x.date, f3x.code, f3x.m)]
EVoff = pd.DataFrame(ev, columns=['date', 'code', 'm', 'src'])
EVgap = E[E.m != 1][['date', 'code', 'm']]
print('official events m!=1', (EVoff.m != 1).sum(), 'flag3 extra', len(f3x), 'gap events', len(EVgap))
QE = rlib.to_td(pd.period_range('2014Q1', '2020Q4', freq='Q').end_time.normalize())
def backcast(EV):
    EV = EV[EV.m != 1]
    out = {}
    for d in QE:
        sub = EV[(EV.date > d) & (EV.date <= AD)]
        cum = np.log(sub.m).groupby(sub.code).sum()
        out[d] = np.exp(np.log(anc) - cum.reindex(anc.index).fillna(0))
    return out
def evaluate(est, upper=False, errset='truth100'):
    ov = []; errs = []
    for d in QE:
        codes = S.loc[d][S.loc[d] & (close.loc[d] > 0)].index
        tr = o8.loc[d, codes].dropna()
        mt = (tr * close.loc[d, tr.index])
        e = est[d].reindex(codes)
        if upper:
            miss = e.isna() & tr.reindex(codes).notna()
            e[miss] = tr.reindex(codes)[miss]
        me = (e * close.loc[d, codes]).dropna()
        t100 = mt.nlargest(100).index; e100 = me.nlargest(100).index
        ov.append(len(set(t100) & set(e100)))
        x = np.log(e.reindex(t100) / tr.reindex(t100)).dropna()
        errs += list(x.abs())
    return np.median(errs), np.median(ov), min(ov), len(errs)
for name, EV in [('official', EVoff), ('gap', EVgap)]:
    est = backcast(EV)
    for up in [False, True]:
        print(name, 'upper' if up else 'base', 'med %.4f ov_med %.1f ov_min %d n %d' % evaluate(est, up))
ADV = rlib.adv(120)
def evaluate_imp(est):
    ov = []; errs = []
    for d in QE:
        codes = S.loc[d][S.loc[d] & (close.loc[d] > 0)].index
        tr = o8.loc[d, codes].dropna()
        mt = (tr * close.loc[d, tr.index])
        e = est[d].reindex(codes)
        me = e * close.loc[d, codes]
        r = rlib.impute_ratio(me, ADV.loc[d, codes], pd.Series(True, index=codes))
        me = me.fillna(ADV.loc[d, codes] * r)
        e2 = me / close.loc[d, codes]
        t100 = mt.nlargest(100).index; e100 = me.nlargest(100).index
        ov.append(len(set(t100) & set(e100)))
        x = np.log(e2.reindex(t100) / tr.reindex(t100)).dropna()
        errs += list(x.abs())
    return np.median(errs), np.median(ov), min(ov), len(errs), ov
for name, EV in [('official', EVoff), ('gap', EVgap)]:
    print(name, 'base+ADVimpute', 'med %.4f ov_med %.1f ov_min %d n %d %s' % evaluate_imp(backcast(EV)))
