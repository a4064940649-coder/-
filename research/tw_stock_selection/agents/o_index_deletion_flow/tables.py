"""P1／P3 彙總表與樣本外 CAR（事後描述；判定仍以 PREREG 為準）。"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import iflib as L
RES = json.load(open(os.path.join(L.HERE, 'results.json'))); assert 'PREREG' in RES
T = pd.read_csv(os.path.join(L.HERE, 'event_returns_all.csv'), parse_dates=['eff'], dtype={'code': str})
T['era'] = np.where(T.eff <= '2018-12-31', 'select', 'oos')
out = {'P1_all_deletions_net': {}, 'P3_by_class_net20': {}, 'P3_by_liq_net20': {}, 'CAR_oos': {}}
for era, g in T.groupby('era'):
    out['P1_all_deletions_net'][era] = {f'h{h}': L.stats(g[f'x_closeE_{h}'], g.m, 0.005) for h in (5, 10, 20)}
    out['P3_by_class_net20'][era] = {c: {f'h{h}': L.stats(h_[f'x_closeE_{h}'], h_.m, 0.005) for h in (5, 20)} for c, h_ in g.groupby('cls')}
    g = g.copy(); g['lq'] = pd.qcut(g.v20.rank(method='first'), 3, labels=['lowADV', 'midADV', 'highADV'])
    out['P3_by_liq_net20'][era] = {str(c): L.stats(h_.x_closeE_20, h_.m, 0.005) for c, h_ in g.groupby('lq', observed=False)}
D = L.panel(); R = D['ret'].clip(-.5, .5); EW = D['ew']; idx = R.index
ev = L.load_events(); ev = ev[ev.in_panel & (ev.eff >= '2019-01-01') & ~((ev.eff >= L.SRC3[0]) & (ev.eff <= L.SRC3[1]))]
for c, g in ev.groupby('cls'):
    if c not in ('MSCI_STD_DEL', 'MSCI_SC_DEL', 'TW50_DEL', 'HDETF_DEL', 'MSCI_STD_ADD', 'MSCI_SC_ADD', 'TW50_ADD'): continue
    M = []
    for r in g.itertuples():
        j = R.columns.get_loc(r.code); i = idx.get_loc(r.eff)
        if i + 20 >= len(idx): continue
        M.append((R.iloc[i - 10:i + 21, j] - EW.iloc[i - 10:i + 21]).to_numpy())
    M = np.array(M); m = np.nanmean(M, 0)
    out['CAR_oos'][c] = {'n': int(len(M)), 'CAR_-10_-1': round(float(np.nansum(m[:10])) * 100, 2), 'AR_0': round(float(m[10]) * 100, 2),
                         'AR_+1': round(float(m[11]) * 100, 2), 'CAR_+1_+5': round(float(np.nansum(m[11:16])) * 100, 2), 'CAR_+1_+20': round(float(np.nansum(m[11:])) * 100, 2)}
RES['TABLES'] = out
json.dump(RES, open(os.path.join(L.HERE, 'results.json'), 'w'), ensure_ascii=False, indent=1, default=str)
for k, v in out.items(): print(k, json.dumps(v, ensure_ascii=False)[:2500])
