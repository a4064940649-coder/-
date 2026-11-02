import sys, json
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_etf_flow_predict/replicate_o9')
import pandas as pd, numpy as np
import rlib as R

ev = R.events(); ev = ev[ev.etf == '0056']
pe = pd.read_csv(R.SCR + '/extras/pe_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
ex = pd.read_csv(R.SCR + '/extras/exdiv_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
close = R.panel('close_raw')
pv = {k: pe.pivot_table(index='date', columns='code', values=k) for k in ['pe', 'yield']}


def close_at(d):
    return close.loc[R.last_td_le(d)]


def proxies(C):
    mc, rk = R.universe_mcap(C, 0, True)
    U = rk[rk <= 150].index
    months = pv['pe'].index
    m_now = months[months <= C][-1]
    m_ago = months[months <= m_now - pd.Timedelta(days=360)][-1]
    y1 = pv['yield'].loc[m_now].reindex(U)
    pe_now = pv['pe'].loc[m_now].reindex(U)
    div = y1 / 100 * close_at(m_now).reindex(U)
    eps_ago = close_at(m_ago).reindex(U) / pv['pe'].loc[m_ago].reindex(U)
    payout = (div / eps_ago).clip(0, 1.5)
    y4 = payout / pe_now * 100
    e = ex[(ex.date > C - pd.Timedelta(days=365)) & (ex.date <= C)]
    d12 = e.groupby('code')['value'].sum().reindex(U).fillna(0)
    y2 = d12 / close.loc[C].reindex(U) * 100
    out = {}
    for n, s in [('y1', y1), ('y2', y2), ('y4', y4)]:
        s = s.replace([np.inf, -np.inf], np.nan)
        r = s.rank(ascending=False, method='first')
        r = r.fillna(len(U) + 1)  # NaN -> bottom
        out[n] = r
    return out, U


res = {}
allrows = []
for rv in sorted(ev.review.unique()):
    e = ev[ev.review == rv]
    C = R.last_td_le(pd.Timestamp(e.cutoff.iloc[0]))
    prx, U = proxies(C)
    for _, r in e.iterrows():
        row = dict(review=rv, C=str(C.date()), code=r.code, action=r.action, inU=r.code in U)
        for n in prx:
            row[n] = float(prx[n].get(r.code, np.nan))
        allrows.append(row)
df = pd.DataFrame(allrows)
df.to_csv(R.AG + '/o_etf_flow_predict/replicate_o9/r2_0056_ranks.csv', index=False)
A = df[df.action == 'add']; D = df[df.action == 'del']
for n in ['y1', 'y2', 'y4']:
    ra = A[n]; rd = D[n]
    res[n] = dict(n_add=len(A), n_del=len(D), rec35=float((ra <= 35).mean()), rec50=float((ra <= 50).mean()),
                  med_rank_inU=float(ra[A.inU].median()), med_rank_all_nan151=float(ra.fillna(151).median()),
                  del_rec_ge66_or_outU=float(((rd >= 66) | (~D.inU)).mean()), del_rec_ge66_inU=float((rd[D.inU] >= 66).mean()),
                  rec35_ex2212=float((A[A.review != '2022-12'][n] <= 35).mean()))
    print(n, {k: round(v, 3) for k, v in res[n].items()})
print('adds not in top150:', int((~A.inU).sum()), 'dels not in top150:', int((~D.inU).sum()))
print('deleted with lagged-yield (y1) rank 1-2:', D[D.y1 <= 2][['review', 'code', 'y1', 'y4']].to_dict('records'))
json.dump(res, open(R.AG + '/o_etf_flow_predict/replicate_o9/r2_0056_p1.json', 'w'), indent=1)
