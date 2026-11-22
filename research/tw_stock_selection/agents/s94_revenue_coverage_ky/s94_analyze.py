#!/usr/bin/env python3
"""S94 步驟 2、4：股池影響、名單與交易差異、20 條路徑配對差。用法：python3 -I -B s94_analyze.py [blk]"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; OUT = HERE / 'out'
SC = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s94')
R = {}
inv = json.load(open(OUT / 'inventory.json')); KY = set(inv['ky_codes'])
z = np.load(SC / 'cache' / 'panel_cs.npz', allow_pickle=False)
IDX = pd.DatetimeIndex(z['IDX']); COLS = z['COLS'].astype(str); MK = z['MK']
rawl = pd.read_parquet(SC / 'raw_long.parquet')
NAME = rawl.sort_values('period').groupby('stock_id').name.last().to_dict()
rev = {v: pd.read_csv(SC / f'rev_{v}.csv', dtype={'code': str}) for v in 'ABC'}
for v in rev: rev[v]['period'] = rev[v].month.str[:7]

# ---- 涵蓋率（稽核同口徑：月底最後交易日 MK==1 的四碼代號；有該月營收列）
per = pd.PeriodIndex(IDX, freq='M'); rows = []
have = {v: rev[v].groupby('period').code.agg(set) for v in rev}
for m in sorted(set(per)):
    if str(m) > '2026-09': continue
    t = np.flatnonzero(per == m)[-1]; tw = set(COLS[MK[t] == 1]); tw91 = {c for c in tw if not c.startswith('91')}
    r = dict(month=str(m), n_listed=len(tw), n_ky=len(tw & KY))
    for v in 'ABC':
        r['cov_' + v] = len(tw & have[v].get(str(m), set())) / len(tw) * 100; r['n_' + v] = len(tw & have[v].get(str(m), set()))
    r['missing_B'] = len(tw - have['B'].get(str(m), set()))
    rows.append(r)
cov = pd.DataFrame(rows); cov.to_csv(OUT / 'coverage_twse_monthly.csv', index=False)
cov['y'] = cov.month.str[:4]
R['coverage_by_year'] = {y: dict(n_listed=float(g.n_listed.median()), n_ky=float(g.n_ky.median()), A=round(float(g.cov_A.median()), 2), B=round(float(g.cov_B.median()), 2), C=round(float(g.cov_C.median()), 2),
                                 still_missing_B=float(g.missing_B.median())) for y, g in cov.groupby('y')}
t = np.flatnonzero(per == pd.Period('2026-09', 'M'))[-1]; tw = set(COLS[MK[t] == 1])
R['missing_2026_09'] = dict(A=sorted(tw - have['A']['2026-09']).__len__(), B=sorted(tw - have['B']['2026-09']), B_names={c: NAME.get(c, '') for c in sorted(tw - have['B']['2026-09'])})
# ---- 下市公司：曾上市、面板最後一天距結束 > 30 天
pres = MK > 0; END = IDX[-1]; dl = []
lastrev = {v: rev[v].groupby('code').period.max() for v in rev}
for j, c in enumerate(COLS):
    if not (MK[:, j] == 1).any(): continue
    k = np.flatnonzero(pres[:, j])[-1]
    if IDX[k] < END - pd.Timedelta(days=30) and MK[k, j] == 1:
        lm = pd.Period(IDX[k], 'M')
        d = dict(code=c, name=NAME.get(c, ''), last_trade=str(IDX[k].date()), ky=c in KY)
        for v in 'AB':
            lr = lastrev[v].get(c); d['last_rev_' + v] = lr; d['gap_' + v] = (lm - pd.Period(lr, 'M')).n if lr else None
        dl.append(d)
dl = pd.DataFrame(dl); dl.to_csv(OUT / 'delisted_twse.csv', index=False)
ok = lambda s: int((s.notna() & (s <= 3)).sum())
R['delisted'] = dict(n=len(dl), n_ky=int(dl.ky.sum()), A_rev_to_within3m=ok(dl.gap_A), B_rev_to_within3m=ok(dl.gap_B), A_no_rev_at_all=int(dl.last_rev_A.isna().sum()), B_no_rev_at_all=int(dl.last_rev_B.isna().sum()),
                     fixed_by_B=dl[(dl.gap_B.fillna(999) <= 3) & ~(dl.gap_A.fillna(999) <= 3)].code.tolist(),
                     A_last_rev_2012_12=int((dl.last_rev_A == '2012-12').sum()), B_last_rev_2012_12=int((dl.last_rev_B == '2012-12').sum()))

# ---- 可選股池
res = {v: json.load(open(OUT / v / 'res.json')) for v in 'ABC'}
el = {v: pd.read_csv(OUT / v / 'elig.csv.gz', dtype={'code': str}) for v in 'ABC'}
R['C_identical_to_A'] = dict(elig=bool(el['A'].equals(el['C'])), targets=res['A']['targets'] == res['C']['targets'], paths=res['A']['paths'] == res['C']['paths'],
                             trades=bool(pd.read_csv(OUT / 'A/trades.csv').equals(pd.read_csv(OUT / 'C/trades.csv'))))
pm = pd.DataFrame({'n_univ': res['A']['n_univ'], 'n_univ_ma': res['A']['n_univ_ma'], 'fin_A': res['A']['n_fin'], 'fin_B': res['B']['n_fin'], 'ma_A': res['A']['n_ma'], 'ma_B': res['B']['n_ma']})
pm['d_fin'] = pm.fin_B - pm.fin_A; pm['d_ma'] = pm.ma_B - pm.ma_A; pm.to_csv(OUT / 'pool_by_signal_month.csv')
pm['y'] = pm.index.str[:4]
R['pool'] = dict(by_year={y: dict(n_univ=float(g.n_univ.mean().round(1)), fin_A=float(g.fin_A.mean().round(1)), fin_B=float(g.fin_B.mean().round(1)), ma_A=float(g.ma_A.mean().round(1)), ma_B=float(g.ma_B.mean().round(1)),
                                  d_ma=float(g.d_ma.mean().round(1))) for y, g in pm.groupby('y')},
                 d_fin=dict(mean=float(pm.d_fin.mean()), min=int(pm.d_fin.min()), max=int(pm.d_fin.max())), d_ma=dict(mean=float(pm.d_ma.mean()), min=int(pm.d_ma.min()), max=int(pm.d_ma.max())),
                 share_univ_with_surge=dict(A=float((pm.fin_A / pm.n_univ).mean()), B=float((pm.fin_B / pm.n_univ).mean())))
a = set(zip(el['A'].month, el['A'].code)); nb = el['B'][[k not in a for k in zip(el['B'].month, el['B'].code)]].copy()
bset = set(zip(el['B'].month, el['B'].code)); lost = el['A'][[k not in bset for k in zip(el['A'].month, el['A'].code)]]
R['pool']['lost_pairs_in_B'] = len(lost)
nb['ky'] = nb.code.isin(KY); nb['name'] = nb.code.map(NAME)
R['pool']['new_pairs'] = len(nb); R['pool']['new_codes'] = int(nb.code.nunique()); R['pool']['new_codes_ky'] = int(nb[nb.ky].code.nunique()); R['pool']['new_nonky'] = sorted(nb[~nb.ky].code.unique())
g = nb.groupby('code').agg(name=('name', 'last'), n_months=('month', 'size'), n_months_ma=('ma60', 'sum'), first=('month', 'min'), last=('month', 'max'), v60_mean_m=('v60_m', 'mean'), v60_last_m=('v60_m', 'last'),
                           max_surge=('surge', 'max'), n_surge_ge2=('surge', lambda s: int((s >= 2).sum()))).sort_values('v60_mean_m', ascending=False)
g.round(2).to_csv(OUT / 'new_pool_codes.csv'); R['pool']['top30'] = g.head(30).round(1).reset_index().to_dict('records')

# ---- 名單差異
TA, TB = res['A']['targets'], res['B']['targets']
diff = []
for m in TA:
    sa, sb = set(TA[m]), set(TB[m])
    if sa != sb:
        diff.append(dict(month=m, sig_date=res['A']['sig_dates'][m], n_A=len(sa), n_B=len(sb), added=' '.join(f'{c}{NAME.get(c, "")}' for c in sorted(sb - sa)), dropped=' '.join(f'{c}{NAME.get(c, "")}' for c in sorted(sa - sb))))
dfm = pd.DataFrame(diff); dfm.to_csv(OUT / 'targets_diff_months.csv', index=False)
R['targets'] = dict(n_months=len(TA), n_months_diff=len(diff), n_added=int(sum(len(set(TB[m]) - set(TA[m])) for m in TA)), n_dropped=int(sum(len(set(TA[m]) - set(TB[m])) for m in TA)),
                    n_added_ky=int(sum(len((set(TB[m]) - set(TA[m])) & KY) for m in TA)), total_A=int(sum(len(v) for v in TA.values())), total_B=int(sum(len(v) for v in TB.values())),
                    diff_by_year=dfm.groupby(dfm.month.str[:4]).size().to_dict())

# ---- 交易差異（標準路徑）
ta = pd.read_csv(OUT / 'A/trades.csv', dtype={'code': str}); tb = pd.read_csv(OUT / 'B/trades.csv', dtype={'code': str})
for x in (ta, tb): x['name'] = x.code.map(NAME); x['ky'] = x.code.isin(KY)
ka = set(zip(ta.code, ta.ent_date)); kb = set(zip(tb.code, tb.ent_date))
newt = tb[[k not in ka for k in zip(tb.code, tb.ent_date)]].copy(); gone = ta[[k not in kb for k in zip(ta.code, ta.ent_date)]].copy()
cols = ['code', 'name', 'ky', 'sig_date', 'ent_date', 'exit_date', 'hold_days', 'why', 'ret', 'pnl', 'cost', 'score']
newt[cols].round(4).to_csv(OUT / 'trades_new_in_B.csv', index=False); gone[cols].round(4).to_csv(OUT / 'trades_gone_from_A.csv', index=False)
both = ta.merge(tb, on=['code', 'ent_date'], suffixes=('_A', '_B'))
R['trades'] = dict(n_A=len(ta), n_B=len(tb), n_common_entry=len(both), n_common_exit_differs=int((both.exit_date_A != both.exit_date_B).sum()),
                   new=dict(n=len(newt), n_ky=int(newt.ky.sum()), n_nonky=int((~newt.ky).sum()), ky_mean_ret=float(newt[newt.ky].ret.mean()), ky_median_ret=float(newt[newt.ky].ret.median()),
                            ky_win=float((newt[newt.ky].ret > 0).mean()), ky_pnl=float(newt[newt.ky].pnl.sum()), nonky_mean_ret=float(newt[~newt.ky].ret.mean()) if (~newt.ky).any() else None,
                            nonky_pnl=float(newt[~newt.ky].pnl.sum())),
                   gone=dict(n=len(gone), mean_ret=float(gone.ret.mean()), median_ret=float(gone.ret.median()), win=float((gone.ret > 0).mean()), pnl=float(gone.pnl.sum())),
                   all_A=dict(mean_ret=float(ta.ret.mean()), median_ret=float(ta.ret.median()), win=float((ta.ret > 0).mean())),
                   all_B_ky=dict(n=int(tb.ky.sum()), mean_ret=float(tb[tb.ky].ret.mean())))
ex = both[both.exit_date_A != both.exit_date_B]
ex[['code', 'name_A', 'ent_date', 'exit_date_A', 'exit_date_B', 'ret_A', 'ret_B', 'pnl_A', 'pnl_B']].round(4).to_csv(OUT / 'trades_exit_changed.csv', index=False)
kyt = newt[newt.ky & (newt.why != 'open_end')].sort_values('ret', ascending=False)
top3 = kyt.head(3)
blk = {}
for r in top3.itertuples(): blk.setdefault(r.code, []).append([int(r.ent_t), int(r.exit_t)])
json.dump(blk, open(OUT / 'block_top3.json', 'w')); R['top3_new'] = top3[cols].round(4).to_dict('records')

# ---- 20 條路徑配對差
def paired(va, vb):
    ra, rb = json.load(open(OUT / va / 'res.json')), json.load(open(OUT / vb / 'res.json')); o = {}
    for k, lab in (('paths', '2019起'), ('paths_full', '全期'), ('paths_1621', '2016–2021'), ('paths_2226', '2022–2026'), ('paths_dd', '最大回撤')):
        d = np.array([rb[k][p] - ra[k][p] for p in ra[k]]) * 100
        o[lab] = dict(median=round(float(np.median(d)), 2), mean=round(float(d.mean()), 2), min=round(float(d.min()), 2), max=round(float(d.max()), 2), n_pos=int((d > 0).sum()), n=len(d), std_path=round(float((rb[k]['100萬|+0'] - ra[k]['100萬|+0']) * 100), 2))
    o['by_shift_2019'] = {f'+{s}': round(float(np.median([(rb['paths'][p] - ra['paths'][p]) * 100 for p in ra['paths'] if p.endswith(f'+{s}')])), 2) for s in range(4)}
    return o
R['paired_B_minus_A'] = paired('A', 'B'); R['paired_C_minus_A'] = paired('A', 'C')
if (OUT / 'B_blk' / 'res.json').exists():
    R['paired_Bblk_minus_A'] = paired('A', 'B_blk'); rb = json.load(open(OUT / 'B_blk/res.json')); R['B_blk_std'] = {k: v for k, v in rb['std'].items() if k != 'yearly'}; R['B_blk_paths'] = rb['paths_summary']
R['yearly'] = {v: res[v]['std']['yearly'] for v in 'ABC'}
R['std'] = {v: {k: x for k, x in res[v]['std'].items() if k != 'yearly'} for v in 'ABC'}; R['paths_summary'] = {v: res[v]['paths_summary'] for v in 'ABC'}
R['peak_mb'] = {v: res[v]['peak_mb'] for v in 'ABC'}
json.dump(R, open(OUT / 'analysis.json', 'w'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(R, ensure_ascii=False, indent=None, default=str)[:200])
