#!/usr/bin/env python3
"""D1 資料閘門（看任何報酬之前）：PIT 股數 2006～2012 回推、閘門檢核、PIT 產業表統計、前 100 大下市股價格完整度。
只用價格（市值名次）、除權因子、公開股本；不算任何策略或基準報酬。

D1 事前寫死的通過條件：
 (a) 「假裝不知道」檢核：以 2021-03-31 的 o8 股數為錨點，用同一套除權因子規則回推到 2014-01；
     2014Q1～2020Q4 共 28 個季末，與 o8 真值比：真值前 100 名股數對數誤差（絕對值）合併中位數 ≤ 3%，
     且回推市值前 100 與真值前 100 的重疊檔數：中位數 ≥ 95、最小 ≥ 93。（錨點日沒有真值的股票一律改 ADV 插補，模擬 2013 年以前已下市股的處境）
 (b) 2006～2012 獨立粗檢：twsec 年度 EPS、純益率 × 月營收年合計 → 隱含股數；回推股數與隱含股數的對數差，
     中位數（絕對值）≤ 10%（隱含股數本身有 EPS 四捨五入與加權平均股數誤差，門檻放寬）。
 (a)(b) 都過 → 2006～2012 標「近似 PIT」並計入 G2／G4；否則 2006～2012 只報描述（改用 120 日均成交值前 100 代理池，標「非 PIT 市值」）。
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lclib as X
L = X.L
t0 = time.time()
IDX, N, COLS = X.IDX, X.N, X.COLS
def ti(d): return int(IDX.searchsorted(pd.Timestamp(d), side='right') - 1)

S13 = X.shares_2013()
f, Fn = X.event_factors()
LST = X.listed_mask(); ADV = X.adv120()
def alive_at(t): return L.TRADED[max(t - 5, 0):t + 1].any(axis=0) & np.isfinite(L.CFF[t]) & LST[t] & ~X.DRMASK
out = {'rule': __doc__}

# ---------------- (a) 假裝不知道 ----------------
ta = ti('2021-03-31'); tlo = ti('2014-01-02') - 1
mult = X.share_multipliers(f, Fn, tlo, ta)
anc = S13[ta].copy(); tan = np.where(np.isfinite(anc), ta, -1)
Sh = X.backcast(anc, tan, tlo, mult)
qs = pd.date_range('2014-03-31', '2020-12-31', freq='QE')
rows = []; errs = []
for q in qs:
    t = ti(q); al = alive_at(t)
    mt = np.where(al, S13[t] * L.CFF[t], np.nan)
    mh = np.where(al, Sh[t] * L.CFF[t], np.nan)
    mh, imp = X.impute_mcap(t, mh, al, ADV)
    T = np.argsort(-np.nan_to_num(mt, nan=-1))[:100]; H = np.argsort(-np.nan_to_num(mh, nan=-1))[:100]
    e = np.abs(np.log(mh[T] / mt[T])); e = e[np.isfinite(e)]
    errs.append(e)
    rows.append(dict(q=str(q.date()), overlap=len(set(T) & set(H)), med_abs_log=float(np.median(e)), p90=float(np.percentile(e, 90)),
                     n_imputed_in_true_top100=int(imp[T].sum()), years_back=round((ta - t) / 245, 1)))
E = np.concatenate(errs)
ov = np.array([r['overlap'] for r in rows])
ga = dict(pooled_med_abs_log=float(np.median(E)), pooled_p90=float(np.percentile(E, 90)), overlap_median=float(np.median(ov)), overlap_min=int(ov.min()),
          by_q=rows, n_events_with_m=int(sum(len(v) for v in mult.values())))
ga['pass'] = bool(ga['pooled_med_abs_log'] <= 0.03 and ga['overlap_median'] >= 95 and ga['overlap_min'] >= 93)
out['D1a'] = ga
print('D1a', {k: v for k, v in ga.items() if k != 'by_q'}, flush=True)

# ---------------- 正式回推 2006～2012 ----------------
p0 = ti(str(X.PIT_START.date()))
first = np.where(np.isfinite(S13).any(0), np.argmax(np.isfinite(S13), axis=0), -1)
anc = np.full(N, np.nan); tan = np.full(N, -1)
for j in range(N):
    if 0 <= first[j] <= p0 + 60:
        anc[j] = S13[first[j], j]; tan[j] = first[j]
tlo2 = ti('2005-06-01')
mult2 = X.share_multipliers(f, Fn, tlo2, int(tan.max()))
Sb = X.backcast(anc, tan, tlo2, mult2)
SF = S13.copy()
pre = np.arange(len(IDX)) < p0
SF[pre] = Sb[pre]
for j in range(N):           # 錨點落在 2013-01-02 之後（≤60 日）者，錨點前也用回推
    if tan[j] > p0: SF[p0:tan[j], j] = Sb[p0:tan[j], j]
pd.DataFrame(SF.astype(np.float32), index=IDX, columns=COLS).to_parquet(X.DATA / 'shares_full.parquet')
out['shares_full'] = dict(file='data/shares_full.parquet', sha256=X.sha(X.DATA / 'shares_full.parquet'),
                          n_anchor=int((tan >= 0).sum()), n_events_with_m=int(sum(len(v) for v in mult2.values())))

# 前 100 名中無股數（插補）比例，2006～2012 與 2013 起
caps = X.Caps(SF)
imp_rows = []
for d in L.official_days('2006-01-01'):
    t = IDX.get_loc(d); U, mcU, impU = caps.universe(t)
    imp_rows.append(dict(d=str(d.date()), n_imp=int(impU.sum()), w_imp=float(mcU[impU].sum() / mcU.sum()), codes=[COLS[U[k]] for k in np.where(impU)[0]]))
IR = pd.DataFrame(imp_rows); IR['y'] = IR.d.str[:4]
out['imputed_top100'] = dict(by_year=IR.groupby('y').agg(n_imp=('n_imp', 'mean'), w_imp=('w_imp', 'mean')).round(4).reset_index().to_dict('records'),
                            codes_count=pd.Series([c for cs in IR.codes for c in cs]).value_counts().head(30).to_dict())
print('imputed', out['imputed_top100']['by_year'], flush=True)

# ---------------- (b) 2006～2012 隱含股數粗檢 ----------------
rev = pd.read_csv('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv', dtype={'code': str}, parse_dates=['month'])
rev = rev[rev.market == 'sii']; rev['y'] = rev.month.dt.year
ry = rev.groupby(['code', 'y']).agg(rev=('revenue_k_twd', 'sum'), n=('revenue_k_twd', 'count')).reset_index()
ry = ry[ry.n == 12].set_index(['code', 'y'])['rev']
brow = []
for y in range(2006, 2013):
    t = ti(f'{y}-06-30'); al = alive_at(t)
    mc = np.where(al, SF[t] * L.CFF[t], np.nan)
    top = np.argsort(-np.nan_to_num(mc, nan=-1))[:100]
    for j in top:
        c = COLS[j]; fp = X.RAW / 'twsec_yearly' / f'{c}.csv'
        if not fp.exists() or (c, y) not in ry.index: continue
        try:
            Y = pd.read_csv(fp, skiprows=1); Y = Y.set_index('Year')
            eps = float(Y.loc[y, 'EPS']); mg = float(Y.loc[y, '純益率(%)'])
        except Exception:
            continue
        if not (np.isfinite(eps) and np.isfinite(mg)) or eps < 0.5 or mg <= 0: continue
        imp_sh = ry[(c, y)] * 1000 * mg / 100 / eps
        brow.append(dict(y=y, code=c, logdiff=float(np.log(SF[t, j] / imp_sh))))
B = pd.DataFrame(brow)
gb = dict(n=len(B), med_abs_log=float(B.logdiff.abs().median()), med_signed=float(B.logdiff.median()), p75_abs=float(B.logdiff.abs().quantile(0.75)),
          by_year=B.groupby('y').logdiff.agg(lambda s: [round(float(s.median()), 4), round(float(s.abs().median()), 4), int(len(s))]).to_dict(),
          worst=B.reindex(B.logdiff.abs().sort_values(ascending=False).index).head(12).round(3).to_dict('records'))
gb['pass'] = bool(gb['med_abs_log'] <= 0.10)
out['D1b'] = gb
print('D1b', {k: v for k, v in gb.items() if k != 'worst'}, flush=True)
out['D1_pass'] = bool(ga['pass'] and gb['pass'])
out['label_2006_2012'] = '近似 PIT（計入 G2／G4）' if out['D1_pass'] else '非 PIT 市值（只報描述；不計入 G2／G4）'

# ---------------- 前 100 大下市股價格完整度 ----------------
LM = pd.read_csv(X.RAW / 'listing_master.csv', dtype=str).set_index('code')
ever = set()
for d in L.official_days('2006-01-01'):
    U, _, _ = caps.universe(IDX.get_loc(d)); ever |= set(U.tolist())
lt = L.LAST_TRADE
dl = []
for j in sorted(ever):
    c = COLS[j]
    if lt[j] >= len(IDX) - 5: continue
    last_listed = LM['last_listed_day'].get(c) if c in LM.index else None
    seg = L.TRADED[max(lt[j] - 120, 0):lt[j] + 1, j]
    dl.append(dict(code=c, panel_last=str(IDX[lt[j]].date()), last_listed=last_listed, miss_last120=int((~seg).sum()),
                   gap_days=int((pd.Timestamp(last_listed) - IDX[lt[j]]).days) if isinstance(last_listed, str) else None))
DL = pd.DataFrame(dl)
out['delisted_top100'] = dict(n=len(DL), n_gap_gt7d=int((DL.gap_days.fillna(0) > 7).sum()), n_miss_last120_gt5=int((DL.miss_last120 > 5).sum()),
                              focus=DL[DL.code.isin(['2448', '2325', '2311', '3474', '3009', '2823', '2888'])].to_dict('records'),
                              problems=DL[(DL.gap_days.fillna(0) > 7) | (DL.miss_last120 > 5)].to_dict('records'))
print('delisted', {k: v for k, v in out['delisted_top100'].items() if k not in ('focus', 'problems')}, flush=True)
# 產業表摘要
T = pd.read_csv(X.RAW / 'industry_pit.csv', dtype=str)
evc = [COLS[j] for j in ever]
Te = T[T.code.isin(evc)]
out['industry_pit'] = dict(sha256=X.sha(X.RAW / 'industry_pit.csv'), n_ever_top100=len(evc), n_rows=len(Te),
                           n_reclassified=int((Te.groupby('code').size() > 1).sum()),
                           n_manual=int(Te.source.str.startswith('人工').sum()), n_prefix=int(Te.source.str.startswith('代碼').sum()),
                           n_unclassified=int((Te.industry == '未分類').sum()),
                           reclassified=Te[Te.code.isin(Te.code[Te.groupby('code').code.transform('size') > 1])].to_dict('records'))
out['data_sha256'] = {'shares_full.parquet': out['shares_full']['sha256'], 'industry_pit.csv': out['industry_pit']['sha256'],
                      'o8 shares_pit.parquet': X.sha(X.O8 / 'data' / 'shares_pit.parquet'), 'lclib.py': X.sha(X.HERE / 'lclib.py'), 's1_gate.py': X.sha(__file__)}
(X.DATA / 'gate.json').write_text(X.dumps(out), encoding='utf-8')
print('D1_pass', out['D1_pass'], out['label_2006_2012'], round(time.time() - t0), 's')
