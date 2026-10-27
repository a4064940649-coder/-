import sys; sys.path.insert(0, "/home/user/-/research/tw_stock_selection/agents/o_disposal_rejudge/replicate_o3")
import pickle
import numpy as np, pandas as pd
import r_core as R
D = pickle.load(open(R.HERE / 'my_runs.pkl', 'rb')); PL, NAV = D['PL'], D['NAV']
IDX, Cn, On, G, TR = R.IDX, R.Cn, R.On, R.GROW, R.TRADED
# 同池等權
ew = R.simulate([dict(e=p['e'], pick=p['pool']) for p in PL[('E1', 10)]], buy=0, sell=0)
print('EW all', R.cagr(ew, 'all')['CAGR'])
SIGX = list(R.SIGX)
def stock_win(p, nxt):
    """訊號日收盤 → 下一訊號日收盤的各股報酬（成交日開盤買、下一成交日開盤賣；中間用 close 近似）"""
    t, e = p['t'], p['e']; t2 = R.IDX.get_loc(nxt)
    return {j: float(np.prod(G[t + 1:t2 + 1, j]) - 1) for j in p['pick']}
def contrib(v, a='2006-04-01', b='2021-08-31'):
    A = {p['d']: p for p in PL[(v, 10)]}; B = {p['d']: p for p in PL[('E1', 10)]}
    con = {}; months = []
    for i, d in enumerate(R.SIG):
        if not (pd.Timestamp(a) <= d <= pd.Timestamp(b)): continue
        nxt = SIGX[i + 1]; pa, pb = A[d], B[d]
        ra, rb = stock_win(pa, nxt), stock_win(pb, nxt)
        sa, sb = set(pa['pick']), set(pb['pick'])
        mc = {}
        for j in sa - sb: mc[j] = mc.get(j, 0) + ra[j] / len(sa)
        for j in sb - sa: mc[j] = mc.get(j, 0) - rb[j] / len(sb)
        months.append((d, mc))
        for j, x in mc.items(): con[R.COLS[j]] = con.get(R.COLS[j], 0) + x
    return con, months
for v in ('E5', 'E0'):
    con, months = contrib(v)
    tot = sum(con.values()); top = sorted(con.items(), key=lambda x: -x[1])[:3]
    print(v, 'sum pp', round(tot * 100, 2), 'top3', [(c, round(x * 100, 2)) for c, x in top], 'share', round(sum(x for _, x in top) / tot * 100, 0))
    # 剔除前 3 檔後的配對差：實際配對差減去這 3 檔貢獻
    pdif = R.sub(R.win_ret(NAV[(v, 10, 'normal')]) - R.win_ret(NAV[('E1', 10, 'normal')]), 'sel')
    tc = {c for c, _ in top}
    adj = pd.Series({d: pdif[d] - sum(x for j, x in mc.items() if R.COLS[j] in tc) for d, mc in months})
    print('  ex-top3 paired', round(adj.mean() * 100, 3), round(R.tstat(adj), 2))
print('--- open-to-open holding version')
def hold_ret(pl, i, t_end=R.T1):
    p = pl[i]; t = p['e']; t2 = pl[i + 1]['e'] if i + 1 < len(pl) else t_end
    pk = p['pick']; r = Cn[t, pk] / On[t, pk]
    if t2 > t: r = r * np.prod(G[t + 1:t2 + 1, pk], axis=0)
    if i + 1 < len(pl):
        tr = TR[t2, pk] & np.isfinite(On[t2, pk])
        r = np.where(tr, r * On[t2, pk] / np.where(tr, Cn[t2, pk], 1.0), r)
    return dict(zip(pk, r - 1))
def contrib2(v):
    A, B = PL[(v, 10)], PL[('E1', 10)]
    con = {}; months = []
    for i, (pa, pb) in enumerate(zip(A, B)):
        assert pa['d'] == pb['d']
        if not (pa['d'] <= pd.Timestamp('2021-08-31')): continue
        ra, rb = hold_ret(A, i), hold_ret(B, i)
        mc = {}
        for j, x in ra.items(): mc[j] = mc.get(j, 0) + x / len(ra)
        for j, x in rb.items(): mc[j] = mc.get(j, 0) - x / len(rb)
        months.append((pa['d'], mc))
        for j, x in mc.items(): con[R.COLS[j]] = con.get(R.COLS[j], 0) + x
    return con, months
for v in ('E5', 'E0'):
    con, months = contrib2(v)
    tot = sum(con.values()); top = sorted(con.items(), key=lambda x: -x[1])[:5]
    print(v, 'sum pp', round(tot * 100, 2), 'top', [(c, round(x * 100, 2)) for c, x in top], 'share3', round(sum(x for _, x in top[:3]) / tot * 100, 0))
    pdif = R.sub(R.win_ret(NAV[(v, 10, 'normal')]) - R.win_ret(NAV[('E1', 10, 'normal')]), 'sel')
    tc = {c for c, _ in top[:3]}
    adj = pd.Series({d: pdif[d] - sum(x for j, x in mc.items() if R.COLS[j] in tc) for d, mc in months})
    print('  ex-top3 paired', round(adj.mean() * 100, 3), round(R.tstat(adj), 2))
    gm = pd.Series({d: sum(mc.values()) for d, mc in months}); adj2 = gm - pd.Series({d: sum(x for j, x in mc.items() if R.COLS[j] in tc) for d, mc in months})
    print('  gross-holding paired', round(gm.mean()*100,3), round(R.tstat(gm),2), 'ex-top3', round(adj2.mean()*100,3), round(R.tstat(adj2),2))
