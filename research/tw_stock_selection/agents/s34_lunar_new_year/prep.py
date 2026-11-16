#!/usr/bin/env python3
"""S34 第 0 步（描述，不算任何策略報酬）：春節日期表、休市空檔、1～2 月交易日數與年對年比、全股池 SURGE 分布（SURGE ≥ 2 比例）依營收月份、
全市場 1、2 月與 1＋2 月合計年增率、F1 名單變動（換手率）依營收月份、各候選規則改動名單的檔數（依營收月份）。
輸出 data_check.json、describe_months.csv.gz。python3 -I -B prep.py"""
import sys, json, time, resource
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s34lib
from s34lib import s10lib, s4lib, stlib

H = Path(__file__).resolve().parent
t = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = s34lib.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗'
A, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~A
del A
TD = s34lib.trading_days()
Rv = s34lib.revenue(P)
RT = {}
for mode in ('none', 'jf', 'adj2', 'wd'):
    RT[mode] = s34lib.surge_rule(P, Rv, mode, TD)
F = s34lib.s17lib.rev_family(P, defs=['STD'])
same = np.allclose(np.nan_to_num(F['S']['STD'].to_numpy(), nan=-9e9), np.nan_to_num(RT['none']['S'].to_numpy(), nan=-9e9), rtol=0, atol=1e-12)
assert same, "mode='none' 與 s17lib STD 不同"
print('載入', f'{time.time()-t:.0f}s', rss(), 'MB', flush=True)
chk = dict(panel=dict(T=P.T, N=P.N, last=str(P.IDX[-1].date()), exright_fix=P.exright_fix), A_none_equals_STD=bool(same))

# ---------------------------------------------------------------- 1. 春節日期表、休市、交易日
idx = pd.DatetimeIndex(P.IDX)
tdp = pd.Series(1, index=idx).groupby(idx.to_period('M')).sum()
ov = [m for m in tdp.index if m >= pd.Period('2015-02', 'M') and m <= pd.Period('2026-09', 'M')]
chk['taiex_vs_panel_tradingdays'] = dict(months=len(ov), mismatch=[(str(m), int(TD[m]), int(tdp[m])) for m in ov if TD.get(m) != tdp[m]])
clo = s34lib.lny_closure()
lt = []
for y, d in s34lib.LNY.items():
    m = pd.Timestamp(d).month
    j, f = pd.Period(f'{y}-01', 'M'), pd.Period(f'{y}-02', 'M')
    row = dict(year=y, lny=d, lny_month=m, closure=clo[y], td_jan=int(TD[j]), td_feb=int(TD[f]))
    if y - 1 in s34lib.LNY:
        row['prev_lny_month'] = pd.Timestamp(s34lib.LNY[y - 1]).month
        row['month_shift'] = bool(row['prev_lny_month'] != m)
        row['td_ratio_jan'] = float(TD[j] / TD[j - 12]); row['td_ratio_feb'] = float(TD[f] / TD[f - 12])
        row['td_ratio_jf'] = float((TD[j] + TD[f]) / (TD[j - 12] + TD[f - 12]))
    lt.append(row)
chk['lny_table'] = lt
chk['lny_source'] = s34lib.LNY_SOURCE

# ---------------------------------------------------------------- 2. 全市場加總年增率（只上市股池、兩年都有營收的公司）
agg = []
for y in range(2016, 2027):
    o = dict(year=y)
    for lab, ms in (('jan', [1]), ('feb', [2]), ('jf', [1, 2]), ('mar', [3])):
        cur = [pd.Period(f'{y}-{mm:02d}', 'M') for mm in ms]
        if any(c not in Rv.index for c in cur) or any((c - 12) not in Rv.index for c in cur):
            continue
        a = Rv.loc[cur].sum(min_count=len(ms)); b = Rv.loc[[c - 12 for c in cur]].sum(min_count=len(ms))
        ok = a.notna() & b.notna() & np.isin(P.COLS, P.COLS[U[min(s34lib.cslib.rev_avail_days(P, cur, True)[cur[-1]], P.T - 1)]])
        o[lab] = float(a[ok].sum() / b[ok].sum() - 1)
    agg.append(o)
chk['aggregate_yoy_listed'] = agg

# ---------------------------------------------------------------- 3. SURGE 分布依營收月份（訊號列 shift 0；股池 U 且 SURGE 有值）
rows = []
S0 = s10lib.make_signal_s10(P, U, RT['none'], shift=0)
prev = None
info0 = {x['month']: x for x in S0['info']}
for m in RT['none']['S'].index:
    if str(m) not in info0:
        continue
    k = info0[str(m)]['k']
    r = dict(month=str(m), cal=m.month, k=int(k), sig_date=str(P.IDX[k].date()))
    for mode in ('none', 'jf', 'adj2', 'wd'):
        s = RT[mode]['S'].loc[m].to_numpy()
        fin = U[k] & np.isfinite(s)
        if fin.sum() == 0:
            r[f'{mode}_n'] = 0; continue
        v = s[fin]
        r[f'{mode}_n'] = int(fin.sum()); r[f'{mode}_ge2'] = float((v >= 2).mean()); r[f'{mode}_med'] = float(np.median(v)); r[f'{mode}_p90'] = float(np.quantile(v, 0.9))
    tg = info0[str(m)]['targets']
    r['f1_n'] = len(tg)
    r['f1_new_vs_prev'] = (float(np.mean([c not in prev for c in tg])) if (prev is not None and tg) else np.nan)
    prev = set(tg)
    if m.month in (1, 2):
        y = m.year
        r['td_ratio'] = float(TD[m] / TD[m - 12])
    rows.append(r)
DF = pd.DataFrame(rows)
DF.to_csv(H / 'describe_months.csv.gz', index=False)
grp = lambda c: ('1月' if c == 1 else '2月' if c == 2 else '3月' if c == 3 else '其他（4～12 月）')
DF['grp'] = DF['cal'].map(grp)
summ = {}
for g, d in DF.groupby('grp'):
    summ[g] = dict(n_months=int(len(d)), ge2_mean=float(d['none_ge2'].mean()), ge2_min=float(d['none_ge2'].min()), ge2_max=float(d['none_ge2'].max()),
                   ge2_sd=float(d['none_ge2'].std()), med_mean=float(d['none_med'].mean()), p90_mean=float(d['none_p90'].mean()),
                   f1_n_mean=float(d['f1_n'].mean()), f1_new_mean=float(d['f1_new_vs_prev'].mean()),
                   jf_ge2_mean=float(d['jf_ge2'].mean()) if 'jf_ge2' in d and d['jf_ge2'].notna().any() else None,
                   adj2_ge2_mean=float(d['adj2_ge2'].mean()), wd_ge2_mean=float(d['wd_ge2'].mean()),
                   ge2_sd_jf=float(d['jf_ge2'].std()) if d['jf_ge2'].notna().sum() > 1 else None, ge2_sd_adj2=float(d['adj2_ge2'].std()), ge2_sd_wd=float(d['wd_ge2'].std()))
chk['surge_by_month_group'] = summ
jf = DF[DF['cal'].isin([1, 2])].copy()
chk['jan_feb_detail'] = jf[['month', 'sig_date', 'td_ratio', 'none_ge2', 'jf_ge2', 'adj2_ge2', 'wd_ge2', 'none_med', 'f1_n', 'f1_new_vs_prev']].to_dict('records')
with np.errstate(divide='ignore'):
    x = np.log(jf['td_ratio'].to_numpy()); yv = jf['none_ge2'].to_numpy(); yw = jf['wd_ge2'].to_numpy()
bk = lambda r: '今年交易日多（比 ≥ 1.3）' if r >= 1.3 else ('今年交易日少（比 ≤ 0.77）' if r <= 0.77 else '差不多（0.77～1.3）')
jf['bucket'] = jf['td_ratio'].map(bk)
chk['ge2_by_tdratio_bucket'] = {b: dict(n=int(len(d)), months=list(d['month']), STD=float(d['none_ge2'].mean()), JF=float(d['jf_ge2'].mean()) if d['jf_ge2'].notna().any() else None,
                                       ADJ2=float(d['adj2_ge2'].mean()), WD=float(d['wd_ge2'].mean())) for b, d in jf.groupby('bucket')}
chk['corr_logtdratio_ge2'] = dict(STD=float(np.corrcoef(x, yv)[0, 1]), WD=float(np.corrcoef(x, yw)[0, 1]), n=int(len(x)))
# 全部月份裡 SURGE ≥ 2 比例最高的 12 個月
chk['top12_ge2_months'] = DF.sort_values('none_ge2', ascending=False).head(12)[['month', 'none_ge2']].to_dict('records')

# ---------------------------------------------------------------- 4. 各候選規則改動名單（只數檔數；依營收月份分組；分訓練／驗證）
per = lambda m: 'train' if pd.Period(m, 'M') <= s34lib.TRAIN_LAST_MONTH else ('valid' if pd.Period(m, 'M') >= s34lib.VALID_FIRST_MONTH else 'pre')
eff = {}
for r in s34lib.RULES:
    S1, _ = s34lib.build(P, U, Rv, TD, r, 0, S_cur=S0, RT_cur=RT['none'])
    a = {x['month']: x['targets'] for x in S0['info']}
    b = {x['month']: (x['targets'] if S1['sig'][x['k']].any() or r['kind'] != 'nobuy' else []) for x in S1['info']}
    if r['kind'] == 'nobuy':
        b = {x['month']: ([] if pd.Period(x['month'], 'M').month in r['months'] else x['targets']) for x in S0['info']}
    o = {}
    for p in ('train', 'valid'):
        oo = {}
        for g in ('1月', '2月', '3月', '其他（4～12 月）'):
            ms = [m for m in a if per(m) == p and grp(pd.Period(m, 'M').month) == g]
            n_orig = sum(len(a[m]) for m in ms)
            rem = sum(len(set(a[m]) - set(b.get(m) or [])) for m in ms)
            add = sum(len(set(b.get(m) or []) - set(a[m])) for m in ms)
            chg = sum(1 for m in ms if a[m] != (b.get(m) or []))
            skipped = sum(1 for m in ms if m not in b)
            oo[g] = dict(n_months=len(ms), months_changed=chg, months_skipped=skipped, n_orig=n_orig, removed=rem, added=add)
        o[p] = oo
    # 出場（L_MA 前 20 名）在訊號列改變的格數：只對 surge 類
    if r['kind'] == 'surge':
        xs = {x['k']: x['month'] for x in S1['info']}
        d = {}
        for x in S0['info']:
            k = x['k']
            if k in xs:
                d.setdefault(grp(pd.Period(x['month'], 'M').month), []).append(int(((S1['X'][k] != S0['X'][k]) & U[k]).sum()) / 2)
        o['top20_changed_per_row_mean'] = {g: float(np.mean(v)) for g, v in d.items()}
    eff[r['id']] = o
chk['rule_effect_preview'] = eff
chk['runtime_s'] = round(time.time() - t, 1); chk['maxrss_mb'] = rss()
(H / 'data_check.json').write_text(json.dumps(chk, ensure_ascii=False, indent=1, default=str))
print(json.dumps({k: v for k, v in chk.items() if k not in ('jan_feb_detail',)}, ensure_ascii=False, indent=1, default=str)[:12000])
