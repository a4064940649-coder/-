import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rlib, numpy as np, pandas as pd
R = rlib.RAW
P = rlib.load_panel(); TD = rlib.td_index()
v = P['value']; c = P['close_raw']
variant = sys.argv[1] if len(sys.argv) > 1 else 'a'
good = v.notna().sum(axis=1) >= 300
vv = v[good]
if variant in ('a', 'c'):
    adv60 = vv.rolling(60, min_periods=60).mean().reindex(TD).ffill()
else:
    adv60 = vv.fillna(0).rolling(60, min_periods=60).mean().reindex(TD).ffill()
s = pd.Series(TD, index=TD).loc['2006-01':'2013-12']
reb = s[s.index.day >= 11].groupby(s[s.index.day >= 11].index.to_period('M')).min().values
reb = pd.DatetimeIndex(reb)
L = rlib.listing_mask()
pool = []
for d in reb:
    a = adv60.loc[d]; cl = c.loc[d]
    ok = (a >= 1e7) & (cl >= 10)
    ok &= pd.Series([not (x.startswith('0') or x.startswith('91')) for x in a.index], index=a.index)
    if variant == 'c':
        ok &= L.loc[d]
    top = a[ok].nlargest(300).index
    pool += [(d, x) for x in top]
pool = pd.DataFrame(pool, columns=['date', 'code'])
print('pool stock-months', len(pool))
# snapshots
IS = os.path.join(R, 'industry_snap'); CEN = os.path.join(R, 'center')
codemap = pd.read_csv(os.path.join(IS, 'goristock_c3c718ab_grs_industry_code.csv'), header=None, names=['k', 'name'])
cm = dict(zip(codemap.k.astype(str), codemap.name))
def toomore(fn):
    d = pd.read_csv(os.path.join(IS, fn), dtype=str)
    d.columns = ['code', 'name', 'ind']
    d = d[d.code.str.fullmatch(r'\d{4}', na=False) & d.ind.notna()]
    return dict(zip(d.code, d.ind.map(lambda k: cm.get(str(int(k)), k))))
snaps = []
snaps.append(('2011-07-28', toomore('goristock_14393dd2_grs_stock_no.csv')))
snaps.append(('2011-08-17', toomore('goristock_d9d8662c_grs_stock_no.csv')))
snaps.append(('2011-09-26', toomore('goristock_235da71f_grs_old_stock_no.csv')))
snaps.append(('2012-03-28', toomore('goristock_c3c718ab_grs_stock_no.csv')))
snaps.append(('2012-07-31', toomore('grs_893c8772_grs_stock_no.csv')))
d = pd.read_csv(os.path.join(IS, 'grs_568e6685_tools_twse_list.csv'), skiprows=1, dtype=str)
d = d[d.iloc[:, 0].str.fullmatch(r'\d{4}', na=False)]
snaps.append(('2013-12-18', dict(zip(d.iloc[:, 0], d.iloc[:, 3]))))
for dt, fn in [('2015-05-22', 'StockList_20150522.csv'), ('2016-04-01', 'StockList_20160401.csv'), ('2017-07-06', 'StockList_20170706.csv')]:
    d = pd.read_csv(os.path.join(CEN, fn), dtype=str)
    d = d[d['代號'].str.fullmatch(r'\d{4}', na=False)]
    snaps.append((dt, dict(zip(d['代號'], d['產業別']))))
d = pd.read_csv(os.path.join(CEN, 'isin_20230116_listed.csv'), dtype=str)
snaps.append(('2023-01-16', dict(zip(d.code, d.industry))))
snaps = [(pd.Timestamp(a), b) for a, b in snaps]
print([(str(a.date()), len(b)) for a, b in snaps])
def pick(code, d):
    have = [(t, m[code]) for t, m in snaps if code in m]
    before = [h for h in have if h[0] <= d]
    if before:
        t, ind = before[-1]
    else:
        after = [h for h in have if h[0] > d and h[0] <= d + pd.DateOffset(months=24)]
        if not after:
            return None, None
        t, ind = after[0]
    if abs((t - d).days) > 731:
        return None, None
    return t, ind
sl = pd.read_csv(os.path.join(CEN, 'stock_list_202610.csv'), dtype=str)
fin_main = set(sl[sl.industry == '金融保險業'].code) | {x for x in v.columns if x.startswith('28')} | {'5876', '5880'}
res = []
for d, x in zip(pool.date, pool.code):
    t, ind = pick(x, d)
    res.append((t is not None, ind))
pool['dated'] = [r[0] for r in res]; pool['ind'] = [r[1] for r in res]
pool['fin_main'] = pool.code.isin(fin_main)
pool['fin_pit'] = pool.ind.fillna('').str.startswith('金融保險')
print('dated share %.4f' % pool.dated.mean(), 'undet', (~pool.dated).sum())
print(pool.groupby(pool.date.dt.year).dated.mean().round(4))
print('undet & fin_main', ((~pool.dated) & pool.fin_main).sum())
dd = pool[pool.dated]
mis = dd[dd.fin_main != dd.fin_pit]
print('mismatch', mis.code.nunique(), len(mis)); print(mis.groupby('code').agg(n=('date', 'size'), ind=('ind', 'first'), fm=('fin_main', 'first')))
