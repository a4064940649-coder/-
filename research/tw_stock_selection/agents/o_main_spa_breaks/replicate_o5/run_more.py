import sys, json, re; sys.path.insert(0, '.')
from spa import *
exec(open('famdef.py').read())
res = {}
# ---- 有效試驗數、DSR ----
for nm, cols, w in (('F_long', FL, 'W_long'), ('F_main', FM, 'W_long'), ('F_all_W21', list(C.index[C.W_21]), 'W_21')):
    s0 = WSTART[w]; D = X50.loc[:, cols].iloc[s0:].fillna(0.0).to_numpy()
    e = eff_tests(D); res[f'eff|{nm}'] = e; print('eff', nm, e)
    if w == 'W_long':
        srs = D.mean(0) / D.std(0, ddof=1)
        x = D[:, cols.index(FOCUS)]
        for lab, N in (('liji', e['liji']), ('nyholt', e['nyholt']), ('raw', e['N'])):
            o = dsr(x, srs, N); res[f'dsr|{nm}|{lab}'] = o; print('  DSR', lab, round(N, 1), {k: round(v, 4) for k, v in o.items()})
# ---- 子期間（F_long，L6）----
for k, (lab, sl) in enumerate((('SEL', SEL), ('VAL', VAL), ('HOLD', HOLD))):
    D = X50.loc[:, FL].iloc[sl].fillna(0.0).to_numpy(); n = D.shape[0]
    idx = sb_indices(n, 6, 10000, 20261028 + 6 + 100 * (k + 1))
    o = spa(D, FL.index(FOCUS), 6, idx, stepm=False, unstd=False); o['best'] = FL[o.pop('argmax')]
    res[f'sub|{lab}'] = o; print('sub', lab, n, round(o['t_main'], 3), o['rank'], o['p_SPA_c_focus'], o['best'])
# ---- 判定欄的穩健性 ----
x = X50[FOCUS].to_numpy()
def run_rows(rows, tag):
    D = X50.loc[:, FL].iloc[rows].fillna(0.0).to_numpy(); n = D.shape[0]
    idx = sb_indices(n, 6, 10000, 20261028 + 6)
    o = spa(D, FL.index(FOCUS), 6, idx, stepm=False, unstd=False)
    res[f'rob|{tag}'] = dict(n=n, t=o['t_main'], p=o['p_SPA_c_focus']); print('rob', tag, n, round(o['t_main'], 3), o['p_SPA_c_focus'])
allr = np.arange(NW)
run_rows(allr[12:], 'start+1y'); run_rows(allr[24:], 'start+2y'); run_rows(allr[:-1], 'drop_last1')
order = np.argsort(-x)
for m in (1, 3, 5, 10): run_rows(np.setdiff1d(allr, order[:m]), f'drop_best{m}')
yr = pd.Series(x, index=G[:-1]).groupby(G[:-1].year).sum()
print('yearly sum (signal-year) pp', (yr * 100).round(1).to_dict())
ye = pd.Series(x, index=G[1:]).groupby(G[1:].year).sum(); print('yearly sum (end-year) pp', (ye * 100).round(1).to_dict())
best_y = int(yr.idxmax()); run_rows(allr[G[:-1].year != best_y], f'drop_year{best_y}')
run_rows(allr[G[:-1].year != 2021], 'drop_year2021_signal'); run_rows(allr[G[1:].year != 2021], 'drop_year2021_end')
run_rows(allr[12:], 'drop_first_year')
json.dump(res, open(OUT / 'more.json', 'w'), ensure_ascii=False, indent=1, default=float)
