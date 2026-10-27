"""P4：主策略（EXFIN n10 gapfix 正式名單）持股遇到指數納入生效日。規則事前寫死在 PREREG，不調參數。"""
import sys, os, json, math
HERE = os.path.dirname(os.path.abspath(__file__))
RES = json.load(open(os.path.join(HERE, 'results.json'))); assert 'PREREG' in RES, '先執行 prereg.py'
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/main_v2')
import v2_core as V
sys.path.pop(0); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import iflib as L
K = V.K
YL, Y5, Y5f = V.load_revenue()
S = V.V2('gapfix', YL, Y5, exfin=True)
pl = S.plan(10, 'official')
IDX = K.IDX
D = L.panel(); assert D['adj'].index.equals(IDX)
A = D['adj'].ffill(limit=30).to_numpy(); AO = D['adj_open'].to_numpy()
E50 = K.E50.to_numpy()
ev = L.load_events()
inc = ev[ev.type.isin(['MSCI_STD_ADD', 'MSCI_SC_ADD', 'TW50_ADD']) & ev.in_panel].copy()
inc['i'] = [IDX.get_loc(e) for e in inc.eff]
COLI = {c: j for j, c in enumerate(K.COLS)}
NAV = pd.read_parquet('/home/user/-/research/tw_stock_selection/agents/main_v2/navs_gapfix.parquet')['EXFIN|n10|normal'].reindex(IDX)
rows = []; per = []
for k in range(len(pl) - 1):
    e0, picks, _, d = pl[k]; e1 = pl[k + 1][0]
    held = [K.COLS[j] for j in picks]
    # 期間報酬（主策略淨值）：e0−1 收盤 → e1−1 收盤（進出都在開盤，近似）
    nv0, nv1 = NAV.iloc[e0 - 1], NAV.iloc[e1 - 1]
    base = nv1 / nv0 - 1 if np.isfinite(nv0) and np.isfinite(nv1) else np.nan
    delta = 0.0
    hits = inc[(inc.i >= e0) & (inc.i < e1 - 1) & inc.code.isin(held)]
    for r in hits.drop_duplicates(['code', 'i']).itertuples():
        j = COLI[r.code]; iE = r.i
        Ri = AO[e1, j] / A[iE, j] - 1
        peers = [COLI[c] for c in held if c != r.code]
        Rp = np.nanmean([AO[e1, q] / A[iE, q] - 1 for q in peers])
        # E 時點權重
        g_i = A[iE, j] / AO[e0, j]; g_all = np.nanmean([A[iE, q] / AO[e0, q] for q in [COLI[c] for c in held]])
        w = (g_i / g_all) / len(held)
        r50 = E50[e1 - 1] / E50[iE] - 1
        delta += w * ((r50 - 0.005) - Ri)
        rows.append(dict(code=r.code, name=r.name, type=r.type, eff=r.eff, sig=d, R_i=Ri, R_peers=Rp, diff=Ri - Rp, w=w, r50=r50,
                         src3=bool(L.SRC3[0] <= r.eff <= L.SRC3[1])))
    per.append(dict(sig=d, start=IDX[e0], end=IDX[e1], base=base, mod=base + delta if np.isfinite(base) else np.nan, n_hits=len(hits)))
P = pd.DataFrame(rows); Q = pd.DataFrame(per)
P.to_csv(os.path.join(HERE, 'p4_events.csv'), index=False)
def tc(x, g):
    s = pd.Series(np.asarray(x, float), index=np.asarray(g)).dropna().groupby(level=0).mean()
    return {'mean%': round(float(s.mean()) * 100, 2), 't_month': round(float(s.mean() / (s.std(ddof=1) / math.sqrt(len(s)))), 2) if len(s) > 2 else None, 'n_months': int(len(s))}
M = P[~P.src3].copy(); M['m'] = M.eff.dt.to_period('M').astype(str)
out = {'n_events_all': int(len(P)), 'n_events_main': int(len(M)), 'by_type_n': M.groupby('type').size().to_dict(),
       'paired_diff_main': {**tc(M['diff'], M.m), 'median%': round(float(M['diff'].median()) * 100, 2), 'pooled_mean%': round(float(M['diff'].mean()) * 100, 2)},
       'half_2006_2015': tc(M[M.eff <= '2015-12-31']['diff'], M[M.eff <= '2015-12-31'].m),
       'half_2016_2026': tc(M[M.eff >= '2016-01-01']['diff'], M[M.eff >= '2016-01-01'].m),
       'by_type': {t: tc(g['diff'], g.m) for t, g in M.groupby('type')},
       'incl_src3': tc(P['diff'], P.eff.dt.to_period('M').astype(str)),
       'raw_R_i_mean%': round(float(M.R_i.mean()) * 100, 2), 'raw_R_peers_mean%': round(float(M.R_peers.mean()) * 100, 2)}
def ann(s, Q):
    s = s.dropna(); n = (1 + s).cumprod(); yrs = (Q.end.iloc[-1] - Q.start.iloc[0]).days / 365.25
    return {'CAGR%': round(float(n.iloc[-1] ** (1 / yrs) - 1) * 100, 2), 'maxDD_period%': round(float((n / n.cummax() - 1).min()) * 100, 1)}
Q = Q.dropna(subset=['base'])
out['strategy_compare'] = {'base': ann(Q.base, Q), 'sell_on_inclusion': ann(Q['mod'], Q), 'n_periods': int(len(Q)), 'n_periods_affected': int((Q.n_hits > 0).sum()),
                           'paired_period_diff_t': round(float((Q['mod'] - Q.base)[Q.n_hits > 0].mean() / ((Q['mod'] - Q.base)[Q.n_hits > 0].std(ddof=1) / math.sqrt((Q.n_hits > 0).sum()))), 2)}
cond = (out['paired_diff_main']['mean%'] < -1) and (out['paired_diff_main']['t_month'] is not None and out['paired_diff_main']['t_month'] <= -2) and \
       (out['half_2006_2015']['mean%'] < 0) and (out['half_2016_2026']['mean%'] < 0)
out['adopt_into_SOP'] = bool(cond)
out['yearly'] = {int(y): {'n': int(len(g)), 'mean%': round(float(g['diff'].mean()) * 100, 2)} for y, g in M.groupby(M.eff.dt.year)}
RES = json.load(open(os.path.join(HERE, 'results.json'))); RES['P4'] = out
json.dump(RES, open(os.path.join(HERE, 'results.json'), 'w'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
print(P[['code', 'name', 'type', 'eff', 'R_i', 'R_peers', 'diff']].to_string())
