#!/usr/bin/env python3
"""S17 事後分析（標「事後」，不影響判定）：重疊率、逐年、最賺 5 筆占比、大贏家；因為 NOMU 依事前規則「穩定地比較好」，另做 NOMU 深挖：
2019 前（2015-07～2018-12）逐條比較、逐月配對差、互相禁買對方最賺 5 筆、整段不准買 2408、NOMU 獨有／STD 獨有交易、選到的股票特徵、
給 XQ 的門檻（L_MA 第 10／15／20 名的分數分布）。python3 -I -B posthoc.py → results_posthoc.json。單行程。"""
import sys, json, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s17lib
from s17lib import s10lib, s4lib, s2lib, stlib, COSTS, DEFS

H = Path(__file__).resolve().parent
t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
RES = json.loads((H / 'results.json').read_text())
TR = pd.read_csv(H / 'trades.csv.gz', dtype={'code': str})
NAV = pd.read_csv(H / 'navs.csv.gz', parse_dates=['date'], index_col='date')
out = {'note': '全部是事後分析，不影響事前判定。'}
WIN = ['W10_70', 'WN_70']
BIG = ['2408', '3653', '1325', '2603']

# ---------------------------------------------------------------- 1. 重疊率（位移 0）
L = RES['lists']
ov = {}
for d in DEFS[1:]:
    rows_t, rows_20 = [], []
    for sd, tg in L['STD']['targets'].items():
        a = set(tg); b = set(L[d]['targets'].get(sd, []))
        if a:
            rows_t.append(len(a & b) / len(a))
        a2 = set(L['STD']['top20'][sd]); b2 = set(L[d]['top20'].get(sd, []))
        if a2:
            rows_20.append(len(a2 & b2) / len(a2))
    ov[d] = dict(buy_list_overlap_mean=float(np.mean(rows_t)), top20_overlap_mean=float(np.mean(rows_20)), n_months=len(rows_t))
out['overlap_vs_STD'] = ov

# ---------------------------------------------------------------- 2. 逐年、最賺 5 筆、交易統計
yr = {}; t5 = {}; ts = {}
for cid in WIN:
    for d in DEFS:
        R = RES['combos'][f'{d}|{cid}']
        yr[f'{d}|{cid}'] = {y: v for y, v in R['yearly'].items()}
        t5[f'{d}|{cid}'] = dict(main=R['top5_share'], path_median=R['paths20']['top5_median'])
        ts[f'{d}|{cid}'] = dict(full=R['trades_full'], since2019=R['trades_since2019'], avg_npos=R['avg_npos'], turnover=R['turnover_oneway_ann'], cost=R['cost_ann_pct_of_nav'])
out['yearly'] = yr; out['top5_share'] = t5; out['trade_stats'] = ts

# ---------------------------------------------------------------- 3. 大贏家
bw = {}
for d in DEFS:
    for cid in WIN:
        x = TR[(TR['run'] == f'{d}|{cid}') & TR['code'].isin(BIG)]
        bw[f'{d}|{cid}'] = [dict(code=r.code, ent=r.ent_date, exit=r.exit_date, ret=float(r.ret), pnl=float(r.pnl), why=r.why) for r in x.itertuples()]
    inlist = {}
    for c in BIG:
        months = [sd for sd, tg in L[d]['targets'].items() if c in tg]
        inlist[c] = months
    bw[f'{d}|in_buy_list'] = inlist
out['big_winners'] = bw

# 最賺 5 筆名單（主跑）
out['top5_trades'] = {f'{d}|{cid}': RES['combos'][f'{d}|{cid}']['block5_top'] for d in DEFS for cid in WIN}

# ---------------------------------------------------------------- 4. NOMU 深挖（需要面板）
P = s17lib.load()
U = stlib.universe(P, 'twse') & ~s4lib.innov_panel(P)[0]
F = s17lib.rev_family(P, defs=['STD', 'NOMU'])
Rv = F['Rv']
RTD = {d: dict(S=F['S'][d], av=F['av']) for d in ('STD', 'NOMU')}
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq'):
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
s17lib.e50_fix_suspension(P)
bench = pd.Series(P.E50[P.t0:], index=P.IDX[P.t0:])
D0, DE = P.IDX[P.t0], P.IDX[-1]
D19 = P.IDX[P.IDX < pd.Timestamp('2019-01-01')][-1]
D24 = P.IDX[P.IDX < pd.Timestamp('2024-01-01')][-1]
cg = lambda nav, a, b: stlib.cagr(nav, a, b)
CELLS = {c['id']: c for c in json.loads((H / 'PREREG.json').read_text())['grid']['cells']}
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)


def sim(c, S, **kw):
    S2, ekw = s10lib.variant_S(S, 'E0'); ekw.update(kw)
    return s10lib.simulate_s10(P, S2, U, sat_w=c['sat_w'], sizing=c['sizing'], lock=c['lock'], cost=COSTS['strict'], **ekw)


# 4a 驗證重跑主跑與 results 相同（確保事後分析用的是同一套）
def mksig(d, s):
    S = s10lib.make_signal_s10(P, U, RTD[d], shift=s)
    S['XV'] = {'E0': S['XV']['E0']}                       # 只留 E0 出場矩陣，省記憶體
    return S


SIG0 = {d: mksig(d, 0) for d in ('STD', 'NOMU')}
chk = {}
main = {}
for d in ('STD', 'NOMU'):
    for cid, c in CELLS.items():
        nav, lg, _ = sim(c, SIG0[d])
        main[(d, cid)] = (nav, lg)
        chk[f'{d}|{cid}'] = abs(cg(nav, D19, DE) - RES['combos'][f'{d}|{cid}']['windows']['since2019']['cagr']) < 1e-9
out['rerun_identical'] = chk
assert all(chk.values()), chk

# 4b 2019 前（2015-07-01～2018-12-28）20 條路徑、逐月配對差、互相禁買、不准買 2408
pre = {}; blk = {}; no2408 = {}
j2408 = int(np.flatnonzero(P.COLS == '2408')[0])
for cid, c in CELLS.items():
    # 雙方最賺 5 筆的聯集（主跑），整段持有期都不准買
    union = {}
    for d in ('STD', 'NOMU'):
        b, _ = s2lib.block_from(main[(d, cid)][1], 5)
        for j, v in b.items():
            union.setdefault(j, []).extend(v)
    bl2408 = {j2408: [(0, P.T)]}
    rows = {d: dict(pre=[], blk=[], no2408=[], s19=[]) for d in ('STD', 'NOMU')}
    for s in range(4):
        for d in ('STD', 'NOMU'):
            Sx = SIG0[d] if s == 0 else mksig(d, s)
            for cap in CAPS:
                nav = sim(c, Sx, capital=cap)[0]
                rows[d]['pre'].append(cg(nav, D0, D19)); rows[d]['s19'].append(cg(nav, D19, DE))
                rows[d]['blk'].append(cg(sim(c, Sx, capital=cap, block=union)[0], D19, DE))
                rows[d]['no2408'].append(cg(sim(c, Sx, capital=cap, block=bl2408)[0], D19, DE))
            if s:
                del Sx; gc.collect()
    f = lambda k: np.array(rows['NOMU'][k]) - np.array(rows['STD'][k])
    pre[cid] = dict(STD_median=float(np.median(rows['STD']['pre'])), NOMU_median=float(np.median(rows['NOMU']['pre'])),
                    diff_npos=int((f('pre') > 0).sum()), diff_median_pp=float(np.median(f('pre'))) * 100)
    blk[cid] = dict(blocked=[str(P.COLS[j]) for j in union], STD_median=float(np.median(rows['STD']['blk'])), NOMU_median=float(np.median(rows['NOMU']['blk'])),
                    diff_npos=int((f('blk') > 0).sum()), diff_median_pp=float(np.median(f('blk'))) * 100)
    no2408[cid] = dict(STD_median=float(np.median(rows['STD']['no2408'])), NOMU_median=float(np.median(rows['NOMU']['no2408'])),
                       diff_npos=int((f('no2408') > 0).sum()), diff_median_pp=float(np.median(f('no2408'))) * 100)
    print(cid, 'pre', pre[cid], 'blk', blk[cid], '2408', no2408[cid], f'{time.time()-t_all:.0f}s', flush=True)
out['NOMU_pre2019_paths'] = pre; out['NOMU_block_union_top5_paths'] = blk; out['NOMU_no2408_paths'] = no2408

pm = {}
for cid in CELLS:
    a, b = main[('NOMU', cid)][0], main[('STD', cid)][0]
    pm[cid] = {k: s2lib.paired(a, b, k) for k in ('2016-03', '2019-01', '2024-01')}
    pm[cid]['2016-03_to_2018-12'] = s2lib.paired(a.loc[:'2018-12-31'], b.loc[:'2018-12-31'], '2016-03')
    pm[cid]['2019-01_to_2023-12'] = s2lib.paired(a.loc[:'2023-12-31'], b.loc[:'2023-12-31'], '2019-01')
out['NOMU_monthly_paired'] = pm

# 4c NOMU 獨有／STD 獨有交易（主跑，代號＋進場日）
ex = {}
for cid in CELLS:
    a = main[('NOMU', cid)][1]; b = main[('STD', cid)][1]
    ka = set(zip(a['code'], a['ent_t'])); kb = set(zip(b['code'], b['ent_t']))
    def st(lg, keys):
        x = lg[[k in keys for k in zip(lg['code'], lg['ent_t'])]]
        return dict(n=int(len(x)), mean_ret=float(x['ret'].mean()) if len(x) else None, median_ret=float(x['ret'].median()) if len(x) else None, sum_pnl=float(x['pnl'].sum()),
                    top=[dict(code=r.code, ent=str(r.ent_date.date()), ret=round(float(r.ret), 4), pnl=round(float(r.pnl))) for r in x.sort_values('pnl', ascending=False).head(5).itertuples()])
    ex[cid] = dict(NOMU_only=st(a, ka - kb), STD_only=st(b, kb - ka), common_in_NOMU=st(a, ka & kb), common_in_STD=st(b, ka & kb))
out['NOMU_exclusive_trades'] = ex

# 4d 選到的股票特徵（位移 0 買進名單）：μ÷σ（年增金額的趨勢強度）、近 12 個月營收、股價、60 日均成交值
D = Rv - Rv.shift(12); D[D.index.year == 2013] = np.nan
mu = D.shift(1).rolling(24, min_periods=24).mean(); sd = D.shift(1).rolling(24, min_periods=24).std(ddof=1)
musd = (mu / sd.where(sd > 0)).replace([np.inf, -np.inf], np.nan)
rev12 = Rv.rolling(12, min_periods=12).sum()
av = F['av']
feat = {}
for d in ('STD', 'NOMU'):
    S = SIG0[d]
    vals = dict(mu_over_sigma=[], rev12_bn=[], price=[], yoy=[])
    for x in S['info']:
        k = x['k']; m = pd.Period(x['month'], 'M')
        for code in x['targets']:
            j = P.ci[code]
            vals['mu_over_sigma'].append(musd.at[m, code]); vals['rev12_bn'].append(rev12.at[m, code] / 1e6)
            vals['price'].append(P.C[k, j]); vals['yoy'].append(Rv.at[m, code] / Rv.at[m - 12, code] - 1)
    feat[d] = {k: dict(median=float(np.nanmedian(v)), p25=float(np.nanpercentile(v, 25)), p75=float(np.nanpercentile(v, 75)), n=len(v)) for k, v in vals.items()}
out['picked_features'] = feat

# 4e 給 XQ 的門檻：L_MA（股池內、還原收盤 > 60 日均線）第 10／15／20 名的分數分布（位移 0）
thr = {}
for d in ('STD', 'NOMU'):
    Sd = F['S'][d]
    nth = {10: [], 15: [], 20: []}
    for x in SIG0[d]['info']:
        k = x['k']; s = Sd.loc[pd.Period(x['month'], 'M')].to_numpy()
        ok = U[k] & np.isfinite(s) & P.F_MA60[k]
        v = np.sort(s[ok])[::-1]
        for n in nth:
            if len(v) >= n:
                nth[n].append(float(v[n - 1]))
    thr[d] = {str(n): dict(median=float(np.median(v)), q25=float(np.percentile(v, 25)), q75=float(np.percentile(v, 75)), min=float(np.min(v)), max=float(np.max(v)), n=len(v)) for n, v in nth.items()}
out['xq_thresholds'] = thr

# 4f 回撤四段：STD／NOMU 20 條路徑中位數（取自 results）
out['seg_median'] = {cid: {d: RES['combos'][f'{d}|{cid}']['paths20']['seg_median'] for d in DEFS} for cid in WIN}
out['bench_seg'] = RES['bench_0050']['seg']
out['meta'] = dict(runtime_s=round(time.time() - t_all, 1), maxrss_mb=rss())
(H / 'results_posthoc.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
print(json.dumps({k: out[k] for k in ('overlap_vs_STD', 'NOMU_pre2019_paths', 'NOMU_block_union_top5_paths', 'NOMU_no2408_paths', 'picked_features', 'xq_thresholds', 'meta')}, ensure_ascii=False, indent=1))
for cid in CELLS:
    print(cid, {k: (round(v['mean'] * 100, 3), round(v['t'], 2), [round(z * 100, 3) for z in v['ci95']]) for k, v in pm[cid].items()})
    print(cid, json.dumps(ex[cid], ensure_ascii=False))
