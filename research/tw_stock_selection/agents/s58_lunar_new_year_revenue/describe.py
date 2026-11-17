#!/usr/bin/env python3
"""S58 第 0 步（描述，不算任何策略報酬）：
(1) 2012～2026 春節（正月初一）／除夕日期、所在月份、月份互換年；加權指數 1～2 月最長休市空檔與 1、2 月交易日數（核對）；
(2) 1 月、2 月、1＋2 月合計營收 YoY（年增率）的截面中位數與離散度（P75−P25），對照同年 3～12 月；
(3) 換股日 SURGE 分布：營收月份 1 月（2 月換股）、2 月（3 月換股）對其他月份——股池 SURGE 中位數、≥ 2 比例、L_MA 前 10 名平均 SURGE、F1 名單換手率；
(4) 各候選改動名單的檔數（只數檔數）。
輸出 results_describe.json。python3 -I -B describe.py"""
import sys, json, time, resource
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s58lib as L
from s58lib import cslib

H = Path(__file__).resolve().parent
t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P, U, RT, Z = L.setup()
print('載入', f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)

# (1) 春節表與休市空檔
TX = L.taiex_dates()
lny = {}
for y in range(2013, 2027):
    d1 = pd.Timestamp(L.LNY[y])
    x = TX[(TX >= pd.Timestamp(f'{y}-01-01')) & (TX <= pd.Timestamp(f'{y}-03-10'))]
    g = np.diff(x.values).astype('timedelta64[D]').astype(int)
    a = int(np.argmax(g))
    closed = pd.bdate_range(x[a] + pd.Timedelta(days=1), x[a + 1] - pd.Timedelta(days=1))
    tdj = int(((TX.year == y) & (TX.month == 1)).sum()); tdf = int(((TX.year == y) & (TX.month == 2)).sum())
    lny[y] = dict(lny_day1=str(d1.date()), chuxi=str((d1 - pd.Timedelta(days=1)).date()), lny_month=int(d1.month),
                  swap_vs_prev=L.SWAP.get(y), last_trade=str(x[a].date()), reopen=str(x[a + 1].date()),
                  closed_weekdays_jan=int((closed.month == 1).sum()), closed_weekdays_feb=int((closed.month == 2).sum()),
                  tdays_jan=tdj, tdays_feb=tdf)
    lny[y]['closure_major_month'] = 1 if lny[y]['closed_weekdays_jan'] > lny[y]['closed_weekdays_feb'] else 2
check_month = {y: v['closure_major_month'] == v['lny_month'] for y, v in lny.items()}

# (2) 營收 YoY 截面
Rv = Z['Rv']
cols_ok = np.asarray(Rv.columns.isin(P.COLS))
def xs(a, b):
    r = (a / b - 1).replace([np.inf, -np.inf], np.nan).dropna()
    return dict(n=int(len(r)), median=float(r.median()), iqr=float(r.quantile(0.75) - r.quantile(0.25)), pos_share=float((r > 0).mean()))
yoy = {}
for y in range(2014, 2027):
    def row(mm):
        return Rv.loc[pd.Period(f'{y}-{mm:02d}', 'M')], Rv.loc[pd.Period(f'{y-1}-{mm:02d}', 'M')]
    o = {}
    if pd.Period(f'{y}-01', 'M') in Rv.index:
        o['jan'] = xs(*row(1))
    if pd.Period(f'{y}-02', 'M') in Rv.index and Rv.loc[pd.Period(f'{y}-02', 'M')].notna().sum() > 100:
        o['feb'] = xs(*row(2))
        a1, b1 = row(1); a2, b2 = row(2)
        o['jf'] = xs(a1 + a2, b1 + b2)
    oth = [xs(*row(mm)) for mm in range(3, 13) if pd.Period(f'{y}-{mm:02d}', 'M') in Rv.index and Rv.loc[pd.Period(f'{y}-{mm:02d}', 'M')].notna().sum() > 100]
    if oth:
        o['m3_12'] = dict(n_months=len(oth), median_mean=float(np.mean([v['median'] for v in oth])), iqr_mean=float(np.mean([v['iqr'] for v in oth])),
                          median_min=float(min(v['median'] for v in oth)), median_max=float(max(v['median'] for v in oth)))
    # 方向：春節由 2 月移到 1 月（今年 1 月、去年 2 月）→ 1 月假性跌、2 月假性漲
    dirn = None
    if L.LNY_MONTH[y] != L.LNY_MONTH[y - 1]:
        dirn = 'feb→jan（1 月應偏低、2 月偏高）' if L.LNY_MONTH[y] == 1 else 'jan→feb（1 月應偏高、2 月偏低）'
    o['shift'] = dirn
    if 'jan' in o and 'feb' in o:
        o['jan_minus_feb_median_pp'] = (o['jan']['median'] - o['feb']['median']) * 100
    yoy[y] = o

# (3) 換股日 SURGE 分布（現行，位移 0）
sg = L.signals(P, U, RT, Z, 0)
rows = []
prev_t = None; prev_top20 = None
info = {x['month']: x for x in sg['CUR']['info']}
for x in sg['CUR']['info']:
    m = pd.Period(x['month'], 'M'); k = x['k']
    s = cslib.ins15_mask_row(P, m, k, RT['S'].loc[m].to_numpy())
    fin = U[k] & np.isfinite(s)
    ok = fin & P.F_MA60[k]
    o_ma = np.flatnonzero(ok)[np.argsort(-s[ok], kind='stable')]
    tg = set(x['targets'])
    top20 = set(o_ma[:20].tolist())
    r = dict(month=str(m), mon=int(m.month), year=int(m.year), sig_date=x['sig_date'], n=int(fin.sum()), med=float(np.median(s[fin])),
             p90=float(np.quantile(s[fin], 0.9)), ge2=float((s[fin] >= 2).mean()), le_m2=float((s[fin] <= -2).mean()),
             lma_top10_mean=float(s[o_ma[:10]].mean()), f1_n=int(x['n_tgt']),
             f1_new_share=(float(len(tg - prev_t) / len(tg)) if (prev_t is not None and tg) else None),
             top20_overlap_prev=(int(len(top20 & prev_top20)) if prev_top20 is not None else None))
    rows.append(r); prev_t = tg; prev_top20 = top20
R = pd.DataFrame(rows)
def grp(df):
    return dict(n_rows=int(len(df)), med=float(df['med'].mean()), ge2=float(df['ge2'].mean()), ge2_min=float(df['ge2'].min()), ge2_max=float(df['ge2'].max()),
                le_m2=float(df['le_m2'].mean()), lma_top10_mean=float(df['lma_top10_mean'].mean()), f1_n=float(df['f1_n'].mean()),
                f1_new_share=float(df['f1_new_share'].dropna().mean()), top20_overlap_prev=float(df['top20_overlap_prev'].dropna().mean()))
G = {'1 月營收（2 月換股）': grp(R[R.mon == 1]), '2 月營收（3 月換股）': grp(R[R.mon == 2]), '3 月營收（4 月換股）': grp(R[R.mon == 3]),
     '其他（4～12 月營收）': grp(R[R.mon >= 4])}
for mo in (1, 2):
    sw = R[(R.mon == mo) & R.year.map(lambda y: bool(L.SWAP.get(y)))]
    ns = R[(R.mon == mo) & ~R.year.map(lambda y: bool(L.SWAP.get(y)))]
    G[f'{mo} 月營收｜互換年'] = grp(sw); G[f'{mo} 月營收｜非互換年'] = grp(ns)
jf_rows = R[R.mon.isin([1, 2])][['month', 'sig_date', 'n', 'med', 'ge2', 'le_m2', 'lma_top10_mean', 'f1_n', 'f1_new_share', 'top20_overlap_prev']].to_dict('records')

# (4) 候選改動名單檔數（只數檔數）
def split(m):
    return 'train' if pd.Period(m, 'M') <= L.TRAIN_LAST_MONTH else 'valid'
chg = {}
base = {x['month']: x['targets'] for x in sg['CUR']['info']}
for c in ('C1', 'C2', 'C3'):
    new = {x['month']: x['targets'] for x in sg[c]['info']}
    out = {}
    for part in ('train', 'valid'):
        for mo, lab in ((1, '1 月營收'), (2, '2 月營收'), (0, '其他月份')):
            ms = [m for m in base if split(m) == part and ((pd.Period(m, 'M').month == mo) if mo else pd.Period(m, 'M').month > 2)]
            changed = [m for m in ms if base[m] != new.get(m)]
            skipped = [m for m in ms if m not in new]
            rem = sum(len(set(base[m]) - set(new.get(m) or [])) for m in ms)
            add = sum(len(set(new.get(m) or []) - set(base[m])) for m in ms)
            out[f'{part}|{lab}'] = dict(n_rows=len(ms), rows_changed=len(changed), rows_skipped=len(skipped), n_orig=int(sum(len(base[m]) for m in ms)), removed=int(rem), added=int(add))
    chg[c] = out
# 其他月份必須逐列相同（影響範圍檢查）
scope_ok = {c: all(chg[c][f'{p}|其他月份']['rows_changed'] == 0 for p in ('train', 'valid')) for c in chg}

res = dict(note='第 0 步描述：沒有算任何策略報酬。', lny=lny, lny_source=L.LNY_SOURCE, lny_closure_matches_month=check_month, yoy_xs=yoy,
           surge_rows_group=G, surge_rows_jan_feb=jf_rows, candidate_list_changes=chg, scope_other_months_unchanged=scope_ok,
           swap_years=sorted(y for y, v in L.SWAP.items() if v), meta=dict(runtime_s=round(time.time() - t0, 1), maxrss_mb=rss()))
(H / 'results_describe.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
for y, v in lny.items():
    print(y, v)
for y, v in yoy.items():
    print(y, json.dumps(v, ensure_ascii=False))
for k, v in G.items():
    print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})
print(json.dumps(chg, ensure_ascii=False)); print('scope', scope_ok, 'closure_match', check_month, res['meta'])
