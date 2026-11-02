"""Fund IR vs 0050 from parsed SITCA long table."""
import sys, re, json, math
import numpy as np, pandas as pd
OUT = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/rep_o11'
d = pd.read_parquet(OUT + '/fund_long.parquet')
import os
STRIP = os.environ.get('NOSTRIP') is None
TAG = '' if STRIP else '_nostrip'

def cname(n):
    n = n.replace('　', '').strip()
    n = re.sub(r'_.*$', '', n)
    if STRIP:
        n = re.sub(r'[\*#＊]+$', '', n).strip()
    return n
d['name'] = d.name.map(cname)
d = d[~d.name.str.contains(r'平均|評比基金數|指數$|報酬指數')]
EXC = re.compile(r'TISA|ETF|指數|N類|-N(?![A-Za-z])|N\d|NM|NB|N\(|N$|B類|B-|\(B\)|B\(|新台幣B|新臺幣B|-B$|B$|美元|USD|人民幣|日圓|日幣|歐元|澳幣|南非')
p1 = d[d.part == 1].drop_duplicates(['ym', 'name']).drop(columns=['incep', 'code'])
p2 = d[d.part == 2].drop_duplicates(['ym', 'name'])[['ym', 'name', 'incep']]
t = p1.merge(p2, on=['ym', 'name'], how='left')
t['excl'] = t.name.map(lambda s: bool(EXC.search(s)))
months = sorted(t.ym.unique())
mi = {m: i for i, m in enumerate(months)}
t['mi'] = t.ym.map(mi)
# overlap check m1 vs ytd diff
t = t.sort_values(['name', 'mi'])

def ytd_ret(t, key):
    t = t.sort_values([key, 'mi']).copy()
    prev_ytd = t.groupby(key).ytd.shift(1); prev_mi = t.groupby(key).mi.shift(1)
    jan = t.ym.str[4:] == '01'
    diff = np.where(jan, t.ytd / 100, (1 + t.ytd / 100) / (1 + prev_ytd / 100) - 1)
    t['r_ytd'] = np.where(jan | (prev_mi == t.mi - 1), diff, np.nan)
    return t
t = ytd_ret(t, 'name')
both = t.m1.notna() & t.r_ytd.notna()
overlap_mad_pp = float(np.median(np.abs(t.m1[both] / 100 - t.r_ytd[both])) * 100)
t['r'] = np.where(t.m1.notna(), t.m1 / 100, t.r_ytd)

# rename linking by identical inception date
info = t.groupby('name').agg(first=('mi', 'min'), last=('mi', 'max'), incep=('incep', lambda x: x.dropna().iloc[-1] if x.notna().any() else None),
                             excl=('excl', 'first'), cat=('cat', 'last'))
last_m = len(months) - 1
link = {}
used = set()
for nm, row in info.sort_values('last').iterrows():
    if row['last'] == last_m or row.excl or row.incep is None:
        continue
    c = info[(info['first'] > row['last']) & (info['first'] <= row['last'] + 3) & (info.incep == row.incep) & (~info.excl)]
    c = c[[x not in used for x in c.index]]
    if len(c) == 0:
        continue
    def score(x):
        s = 0
        if re.search(r'-?A(類|$|股|累積|\(|-)|A類型|不配息|累積', x): s -= 2
        if re.search(r'-I|I類|I級|-R|R類|C\(', x): s += 3
        return (s, len(x))
    best = sorted(c.index, key=score)[0]
    link[nm] = best; used.add(best)
# build chains
root = {}
for nm in info.index:
    root[nm] = nm
inv = {v: k for k, v in link.items()}
def chain_root(n):
    while n in inv:
        n = inv[n]
    return n
t['fund'] = t.name.map(chain_root)
# a later name inherits excl status from itself; chain excluded if final name excluded
t = t.sort_values(['fund', 'mi']).drop_duplicates(['fund', 'mi'], keep='last')
# recompute ytd-diff across renamed links
t = ytd_ret(t, 'fund')
t['r'] = np.where(t.m1.notna(), t.m1 / 100, t.r_ytd)

# 0050 month-end
e = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj']
me = e.resample('ME').last()
r50 = me.pct_change()
r50.index = r50.index.strftime('%Y%m')
t['b'] = t.ym.map(r50)
t['x'] = t.r - t.b
R = t.pivot(index='ym', columns='fund', values='r').reindex(months)
XB = t.pivot(index='ym', columns='fund', values='x').reindex(months)
finfo = t.groupby('fund').agg(first=('mi', 'min'), last=('mi', 'max'), n=('x', lambda x: x.notna().sum()), cat=('cat', 'last'),
                              excl=('excl', 'last'), last_name=('name', 'last'))
finfo['alive'] = finfo['last'] == last_m
y19 = np.array([m >= '201901' for m in months])

def stats_of(cols, mask=None):
    out = {}
    for c in cols:
        x = XB[c].values.copy(); r = R[c].values.copy(); b = np.array([r50.get(m, np.nan) for m in months])
        if mask is not None:
            x = np.where(mask, x, np.nan)
        ok = ~np.isnan(x)
        if ok.sum() < 24:
            continue
        xx = x[ok]
        irv = xx.mean() / xx.std(ddof=1) * math.sqrt(12)
        n = ok.sum()
        ga = np.exp(np.log1p(r[ok]).sum() * 12 / n) - np.exp(np.log1p(b[ok]).sum() * 12 / n)
        out[c] = {'IR': irv, 't': xx.mean() / xx.std(ddof=1) * math.sqrt(n), 'exann': ga, 'te': xx.std(ddof=1) * math.sqrt(12), 'n': int(n)}
    return pd.DataFrame(out).T

ACT = ['general', 'tech', 'smallmid', 'value', 'otc']
inc = finfo[~finfo.excl]
gen_surv = inc[(inc.cat == 'general') & inc.alive & (inc.n >= 180)].index
all_surv = inc[inc.cat.isin(ACT) & inc.alive & (inc.n >= 180)].index
res = {'overlap_mad_pp': overlap_mad_pp, 'n_links': len(link)}
def q(s, p): return float(np.percentile(s, p))
for lab, cols in [('general', gen_surv), ('all', all_surv)]:
    S = stats_of(cols)
    res[lab] = {'n': int(len(S)), 'IR_p90': q(S.IR, 90), 'IR_med': q(S.IR, 50), 'IR_max': float(S.IR.max()),
                'IR_max_fund': S.IR.idxmax(), 't_of_max': float(S.t[S.IR.idxmax()]),
                'exann_p90_pp': q(S.exann, 90) * 100, 'exann_med_pp': q(S.exann, 50) * 100, 'te_med_pct': q(S.te, 50) * 100,
                'n_t3': int((S.t >= 3).sum()), 'n_t2': int((S.t >= 2).sum())}
    pre = np.array([m <= '201812' for m in months])
    S1 = stats_of(cols, pre); S2 = stats_of(cols, y19)
    res[lab]['pre2019'] = {'med': q(S1.IR, 50), 'p90': q(S1.IR, 90)}
    res[lab]['y19'] = {'med': q(S2.IR, 50), 'p90': q(S2.IR, 90)}
    # survivorship lower bound
    cats = ['general'] if lab == 'general' else ACT
    dead = inc[inc.cat.isin(cats) & (~inc.alive) & (inc['first'] <= mi['201110'])].index
    Sd = stats_of(dead)
    lb = np.concatenate([S.IR.values, np.full(len(dead), S.IR.min())])
    act = np.concatenate([S.IR.values, Sd.IR.values])
    res[lab]['dead'] = {'n_dead': int(len(dead)), 'attrition': len(dead) / (len(dead) + len(S)), 'dead_IR_med': q(Sd.IR, 50) if len(Sd) else None,
                        'p90_lower': q(lb, 90), 'p90_actual_dead': q(act, 90), 'n_dead_with_ir': int(len(Sd))}
    res[lab]['top'] = S.sort_values('IR', ascending=False).head(5).assign(name=lambda z: [finfo.last_name[i] for i in z.index]).to_dict('index')
json.dump(res, open(OUT + '/funds%s.json' % TAG, 'w'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(res, ensure_ascii=False, indent=1, default=str))

if len(sys.argv) > 1 and sys.argv[1] == 'dump':
    inc2 = inc.copy()
    inc2['incep'] = [t[t.fund == f].incep.dropna().iloc[-1] if t[t.fund == f].incep.notna().any() else None for f in inc2.index]
    inc2['first_ym'] = [months[i] for i in inc2['first']]
    inc2['last_ym'] = [months[i] for i in inc2['last']]
    inc2.to_csv(OUT + '/fund_info.csv')
    S = stats_of(inc2.index)
    inc2.join(S, rsuffix='_s').to_csv(OUT + '/fund_info.csv')
