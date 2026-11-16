#!/usr/bin/env python3
"""S39 獨立複製（python3 -I -B rep_s39.py）。只依 SPEC.json 文字與 FINDINGS.md 條件撰寫；未讀、未 import、未執行 s39 的任何 .py。
原始來源：
- Tasi（shohokuno10/Tasi）營收/YYYYMM.csv 各版本：blobless clone 到暫存區，以 `git log --raw` 列版本、`git show <commit>:<path>` 匯出成暫存檔（dl/snaps/<commit>.csv），本程式只讀這些 CSV（不執行 repo 內任何程式）。
- agents/o_rev_event_clock_ts/data/firstseen_sources.csv.gz（核對首次出現時間）。
- 研究中心營收檔 stlib.REV_CSV（CUR）。
- 引擎 s20lib → s17lib → s10lib（不改），cslib.EXRIGHT_FIX=True。
輸出：本資料夾 results_rep.json。暫存：scratchpad/s3940rep/（結束刪除）。"""
import sys, json, time, resource, gc, re
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent.parent
ROOT = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s3940rep')
DL = ROOT / 'dl'
SC = ROOT / 'cache'
SC.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(AG / 's20_persistent_growth'))
import s20lib                                                  # noqa: E402
from s20lib import s17lib, s10lib, s4lib, stlib, cslib, COSTS   # noqa: E402
s20lib.S20 = SC; s17lib.S17 = SC; s10lib.S10 = SC
for n in ('s9lib', 's8lib', 's7lib', 's6lib', 's5lib', 's4lib', 's3lib', 's2lib'):
    setattr(getattr(s10lib, n), 'S' + n[1:-3], SC)
s10lib.stlib.S1 = SC; cslib.TMP = SC
cslib.EXRIGHT_FIX = True

T0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
MONTHS = pd.period_range('2024-06', '2026-09', freq='M')
CODE_RE = re.compile(r'^\d{4}[0-9A-Z]{0,2}$')


# ================================================================ 1. Tasi 首次值抽取
def extract():
    v = pd.read_csv(DL / 'vers_use.txt', sep=' ', header=None, names=['file', 'commit', 'T', 'blob', 'st'])
    v['ym'] = v['file'].str.extract(r'(\d{6})')[0]
    v['T'] = pd.to_datetime(v['T'])
    rows = []; meta = {}
    for ym, g in v.groupby('ym'):
        m = pd.Period(ym[:4] + '-' + ym[4:], 'M')
        if m not in MONTHS:
            continue
        g = g.sort_values('T')
        seen = {}                                   # code -> dict
        last_vals = {}
        snaps_T = []
        for _, r in g.iterrows():
            snaps_T.append(r['T'])
            fp = DL / 'snaps' / f"{r['commit']}.csv"
            if fp.stat().st_size == 0:
                cur_vals = {}
            else:
                d = pd.read_csv(fp, dtype=str)
                d.columns = [c.strip() for c in d.columns]
                d['公司代號'] = d['公司代號'].str.strip()
                d = d[d['公司代號'].str.match(CODE_RE)].drop_duplicates('公司代號', keep='first')
                val = pd.to_numeric(d['當月營收'].str.replace(',', ''), errors='coerce')
                v12 = pd.to_numeric(d['去年當月營收'].str.replace(',', ''), errors='coerce')
                v1 = pd.to_numeric(d['上月營收'].str.replace(',', ''), errors='coerce')
                cur_vals = {c: (a, b, e) for c, a, b, e in zip(d['公司代號'], val, v12, v1)}
            for c, (a, b, e) in cur_vals.items():
                if c not in seen:
                    seen[c] = dict(T_first=r['T'], commit_first=r['commit'], FP=np.nan, V12=np.nan, V1=np.nan, T_fp=pd.NaT, commit_fp=None, nchg=0, prev=None)
                s = seen[c]
                if np.isnan(s['FP']) and np.isfinite(a):
                    s.update(FP=a, V12=b, V1=e, T_fp=r['T'], commit_fp=r['commit'])
                if np.isfinite(a):
                    if s['prev'] is not None and a != s['prev']:
                        s['nchg'] += 1
                    s['prev'] = a
            last_vals = {c: x[0] for c, x in cur_vals.items()}
        for c, s in seen.items():
            rows.append(dict(code=c, ym=str(m), FP=s['FP'], V1=s['V1'], V12=s['V12'], T_fp=s['T_fp'], commit_fp=s['commit_fp'],
                             T_first=s['T_first'], commit_first=s['commit_first'], TL=last_vals.get(c, np.nan), n_chg_window=s['nchg']))
        meta[str(m)] = dict(n_commits=int(len(g)), first_commit=str(g['T'].iloc[0]), last_commit=str(g['T'].iloc[-1]), n_codes=len(seen))
    return pd.DataFrame(rows), meta


FPD, EXMETA = extract()
FPD = FPD[np.isfinite(FPD['FP'])].reset_index(drop=True)
res = dict(extract=dict(n_rows_fp=int(len(FPD)), per_month=EXMETA))
print('抽取', len(FPD), round(time.time() - T0), 's', flush=True)

# 與 firstseen_sources 核對（tasi，(code, ym)）
fs = pd.read_csv(AG / 'o_rev_event_clock_ts/data/firstseen_sources.csv.gz', dtype=str)
fs = fs[fs['source'] == 'tasi'][['code', 'ym', 'T', 'commit']]
fs['T'] = pd.to_datetime(fs['T'])
mg = FPD.merge(fs, on=['code', 'ym'], how='left')
has = mg['T'].notna()
res['extract']['firstseen_check'] = dict(n_fp=int(len(FPD)), n_in_firstseen=int(has.sum()),
                                         T_fp_eq_firstseen_T=int((mg.loc[has, 'T_fp'] == mg.loc[has, 'T']).sum()),
                                         commit_fp_eq=int((mg.loc[has, 'commit_fp'] == mg.loc[has, 'commit']).sum()),
                                         T_first_eq_firstseen_T=int((mg.loc[has, 'T_first'] == mg.loc[has, 'T']).sum()))
print('firstseen 核對', res['extract']['firstseen_check'], flush=True)

# 原研究 data/first_print.csv.gz 對照（只比對結果，不參與計算）
try:
    o = pd.read_csv(AG / 's39_first_print_revenue/data/first_print.csv.gz', dtype={'code': str, 'ym': str})
    o = o[o['code'].str.match(CODE_RE)]
    mm = FPD.merge(o[['code', 'ym', 'FP', 'TL', 'V12']], on=['code', 'ym'], how='outer', suffixes=('', '_o'), indicator=True)
    both = mm['_merge'] == 'both'
    res['extract']['vs_original_first_print'] = dict(n_mine=int(len(FPD)), n_orig_codes=int(len(o)), n_both=int(both.sum()),
                                                     only_mine=int((mm['_merge'] == 'left_only').sum()), only_orig=int((mm['_merge'] == 'right_only').sum()),
                                                     FP_equal=int((mm.loc[both, 'FP'] == mm.loc[both, 'FP_o']).sum()),
                                                     TL_equal_or_both_nan=int(((mm.loc[both, 'TL'] == mm.loc[both, 'TL_o']) | (mm.loc[both, 'TL'].isna() & mm.loc[both, 'TL_o'].isna())).sum()),
                                                     V12_equal=int((mm.loc[both, 'V12'] == mm.loc[both, 'V12_o']).sum()))
except Exception as e:                                          # noqa: BLE001
    res['extract']['vs_original_first_print'] = repr(e)

# ================================================================ 2. CUR 營收（自讀，同 SPEC：同代碼優先上市、≤0 缺值）
r = pd.read_csv(stlib.REV_CSV, dtype={'code': str}, parse_dates=['month'])
r['pri'] = (r['market'] != 'sii').astype(int)
r = r.sort_values(['month', 'code', 'pri']).drop_duplicates(['month', 'code'], keep='first')
r['ym'] = r['month'].dt.to_period('M').astype(str)
CURL = r[['code', 'ym', 'market', 'revenue_k_twd']].rename(columns={'revenue_k_twd': 'CUR'})

# ---------------------------------------------------------------- 描述 1
win = CURL[CURL['ym'].isin([str(m) for m in MONTHS])]
J = win.merge(FPD[['code', 'ym', 'FP', 'TL', 'V12', 'n_chg_window', 'T_fp']], on=['code', 'ym'], how='outer', indicator=True)
cov = {}
for mk in ('sii', 'otc'):
    x = J[J['market'] == mk]
    cov[mk] = dict(both=int((x['_merge'] == 'both').sum()), only_cur=int((x['_merge'] == 'left_only').sum()))
cov['only_fp_any_market'] = int((J['_merge'] == 'right_only').sum())

# 載入面板（股池、P.COLS、現行 SURGE）
P = s20lib.load()
assert P.exright_fix is True
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
del INNOV
F = s17lib.rev_family(P, defs=['STD'])
S_ref = F['S']['STD']; AV = F['av']; Rv_ref = F['Rv']; del F
s17lib.e50_fix_suspension(P)
for a in ('CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq'):
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
print('載入面板', round(time.time() - T0), 's', rss(), 'MB', flush=True)
COLS = pd.Index(P.COLS)
onlyfp = J[J['_merge'] == 'right_only']
cov['only_fp_in_P_COLS'] = int(onlyfp['code'].isin(COLS).sum())


def stats(x):
    x = x[(x['FP'] > 0) & (x['CUR'] > 0)]
    corr = np.round(x['FP']) != np.round(x['CUR'])
    rel = (x['CUR'] / x['FP'] - 1)[corr]
    a = rel.abs()
    return dict(n=int(len(x)), n_corr=int(corr.sum()), share_corr=float(corr.mean()) if len(x) else float('nan'),
                up=int((rel > 0).sum()), down=int((rel < 0).sum()), down_share=float((rel < 0).mean()) if corr.sum() else float('nan'),
                abs_rel_bins={'<0.1%': int((a < 0.001).sum()), '0.1~1%': int(((a >= 0.001) & (a < 0.01)).sum()), '1~10%': int(((a >= 0.01) & (a < 0.1)).sum()), '>=10%': int((a >= 0.1).sum())},
                abs_rel_median=float(a.median()) if len(a) else float('nan'), rel_median=float(rel.median()) if len(rel) else float('nan'))


B = J[J['_merge'] == 'both'].copy()
B['m'] = B['ym']
# 策略股池：P.COLS 內且現行 SURGE 有值
sv = S_ref.stack(future_stack=True) if hasattr(pd.DataFrame, 'stack') else S_ref.stack()
sv = sv[np.isfinite(sv)]
pool_keys = set((str(m), c) for m, c in sv.index)
pool_keys_u = set((str(m), c) for m, c in sv.index if AV[m] < P.T and U[AV[m], P.ci[c]])
B['in_pool_cols'] = [(m, c) in pool_keys for m, c in zip(B['ym'], B['code'])]
B['in_pool'] = [(m, c) in pool_keys_u for m, c in zip(B['ym'], B['code'])]
sii = B[B['market'] == 'sii']
d1 = dict(coverage=cov, sii_all=stats(sii), strategy_pool=stats(sii[sii['in_pool']]), pool_PCOLS_surge_only=stats(sii[sii['in_pool_cols']]), otc_reference=stats(B[B['market'] == 'otc']))
x = sii[(sii['FP'] > 0) & (sii['CUR'] > 0)].copy()
x['corr'] = np.round(x['FP']) != np.round(x['CUR'])
tl_ok = x['TL'].notna()
d1['decompose'] = dict(n=int(len(x)), in_window_changed=int((tl_ok & (np.round(x['TL']) != np.round(x['FP']))).sum()),
                       post_window_changed=int(((~tl_ok) | (np.round(x['CUR']) != np.round(x['TL']))).sum()),
                       corrected_only_in_window=int((x['corr'] & tl_ok & (np.round(x['TL']) == np.round(x['CUR']))).sum()),
                       corrected_post_window=int((x['corr'] & ((~tl_ok) | (np.round(x['CUR']) != np.round(x['TL'])))).sum()),
                       tl_missing=int((~tl_ok).sum()))
cx = x[x['corr']]
vc = cx['code'].value_counts()
d1['company_concentration'] = dict(n_companies_corrected=int(len(vc)), n_companies=int(x['code'].nunique()), top10=vc.head(10).to_dict(),
                                   share_corr_from_repeaters_ge3=float(vc[vc >= 3].sum() / vc.sum()))
d1['by_month_n_corr'] = cx.groupby('ym').size().to_dict()
d1['corrections_list_check'] = None
try:
    oc = pd.read_csv(AG / 's39_first_print_revenue/data/corrections.csv.gz', dtype={'code': str, 'm': str})
    mine = set(zip(cx['code'], cx['ym'])); orig = set(zip(oc['code'], oc['m']))
    d1['corrections_list_check'] = dict(n_mine=len(mine), n_orig=len(orig), both=len(mine & orig), only_mine=sorted(mine - orig), only_orig=sorted(orig - mine))
except Exception as e:                                          # noqa: BLE001
    d1['corrections_list_check'] = repr(e)
# 舊月份 V12 vs CUR（m−12）
fv = FPD[['code', 'ym', 'V12']].copy()
fv['ym'] = (pd.PeriodIndex(fv['ym'], freq='M') - 12).astype(str)
fv = fv.rename(columns={'V12': 'FP'}).merge(CURL, on=['code', 'ym'], how='inner')
d1['older_months_V12_sii'] = stats(fv[fv['market'] == 'sii'])
res['describe_1'] = d1
print('描述1', json.dumps({k: d1[k] for k in ('sii_all', 'decompose')}, ensure_ascii=False), flush=True)

# ================================================================ 3. SURGE 包裝與變體
REVC = pd.read_csv(stlib.REV_CSV, dtype={'code': str}, parse_dates=['month'])


def rv_matrix():
    rr = REVC.copy()
    rr['pri'] = (rr['market'] != 'sii').astype(int)
    rr = rr.sort_values(['month', 'code', 'pri']).drop_duplicates(['month', 'code'], keep='first')
    Rv = rr.pivot(index='month', columns='code', values='revenue_k_twd').sort_index()
    Rv.index = pd.PeriodIndex(Rv.index, freq='M')
    Rv = Rv.reindex(pd.period_range(Rv.index.min(), Rv.index.max(), freq='M')).reindex(columns=P.COLS)
    return Rv.where(Rv > 0)


def z_from_D(D):
    D = D.copy()
    D[D.index.year == 2013] = np.nan
    Z, _ = s17lib.zscore(D, 24, 'mean')
    Z[Z.index < s17lib.FIRST_MONTH] = np.nan
    return Z


Rv = rv_matrix()
FPM_ = FPD.pivot(index='ym', columns='code', values='FP')
FPM_.index = pd.PeriodIndex(FPM_.index, freq='M')
FPmat = FPM_.reindex(index=Rv.index, columns=Rv.columns)
inwin = pd.Series(Rv.index.isin(MONTHS), index=Rv.index)
REPL = FPmat.notna() & Rv.notna() & inwin.to_numpy()[:, None]   # 只替換兩邊都有值的格子
FPv = FPmat.where(FPmat > 0)                                     # FP ≤ 0 → 缺值


def surge_variant(mode):
    if mode == 'CUR':
        return z_from_D(Rv - Rv.shift(12))
    if mode == 'FPA':
        R2 = Rv.where(~REPL, FPv)
        return z_from_D(R2 - R2.shift(12))
    if mode == 'FPM':
        D = Rv - Rv.shift(12)
        D[D.index.year == 2013] = np.nan
        Z = z_from_D(Rv - Rv.shift(12))
        Rm = Rv.where(~REPL, FPv)
        for m in MONTHS:
            i = Rv.index.get_loc(m)
            D2 = D.iloc[i - 24:i + 1].copy()
            D2.iloc[-1] = (Rm.iloc[i] - Rv.iloc[i - 12]).to_numpy()
            z, _ = s17lib.zscore(D2, 24, 'mean')
            Z.iloc[i] = z.iloc[-1].to_numpy()
        return Z
    raise ValueError(mode)


eqn = lambda a, b: bool(np.array_equal(np.asarray(a, float), np.asarray(b, float), equal_nan=True))
SV = {k: surge_variant(k) for k in ('CUR', 'FPM', 'FPA')}
acc = dict(cur_Rv_bitwise=eqn(Rv.to_numpy(), Rv_ref.to_numpy()), cur_wrapper_bitwise=eqn(SV['CUR'].to_numpy(), S_ref.to_numpy()),
           cells_replaced_in_Rv=int(REPL.to_numpy().sum()), cells_replaced_value_differs=int((REPL & (np.round(FPmat) != np.round(Rv))).to_numpy().sum()))
pre = SV['CUR'].index < MONTHS[0]
for k in ('FPM', 'FPA'):
    acc[f'{k}_rows_before_2024-06_identical'] = eqn(SV[k].to_numpy()[pre], SV['CUR'].to_numpy()[pre])
    dif = ~((SV[k].to_numpy() == SV['CUR'].to_numpy()) | (np.isnan(SV[k].to_numpy()) & np.isnan(SV['CUR'].to_numpy())))
    acc[f'{k}_cells_changed'] = int(dif.sum())
    acc[f'{k}_months_changed'] = [str(m) for m in SV['CUR'].index[dif.any(axis=1)]]
assert acc['cur_wrapper_bitwise'], 'CUR 包裝與 s17lib 不同'
print('包裝', acc, flush=True)

# ================================================================ 4. 引擎
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D0, DE, D19, DW = P.IDX[P.t0], P.IDX[-1], lastb('2019-01-01'), lastb('2024-07-01')
CELL = dict(sat_w=0.7, sizing='W10', lock=False)
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6); SH = (0, 1, 2, 3)
SIGS = {}
for k in SV:
    RT = dict(S=SV[k], av=AV)
    for s in SH:
        SIGS[(k, s)] = s10lib.variant_S(s10lib.make_signal_s10(P, U, RT, shift=s), 'E0')[0]


def run(k, s, cap, cost='strict'):
    return s10lib.simulate_s10(P, SIGS[(k, s)], U, capital=cap, cap_pct=0.05, cost=COSTS[cost], **CELL)


def summ(nav, lg):
    ent = pd.to_datetime(lg['ent']) if 'ent' in lg else None
    return dict(cagr_window=stlib.cagr(nav, DW, DE), maxdd_window=stlib.maxdd(nav, DW, DE), since2019=stlib.cagr(nav, D19, DE), full=stlib.cagr(nav, D0, DE),
                maxdd_full=stlib.maxdd(nav, D0, DE), n_closed_all=int((lg['why'] != 'open_end').sum()), end_nav=float(nav.iloc[-1]))


def trade_keys(lg):
    x = lg.copy()
    x['ent'] = [str(P.IDX[t].date()) for t in x['ent_t']]; x['exit'] = [str(P.IDX[t].date()) for t in x['exit_t']]
    cols = ['code', 'ent', 'exit', 'why']
    return set(tuple(str(v) for v in row) for row in x[cols].to_numpy()), cols


RUN = {}
for k in SV:
    for s in SH:
        for cap in CAPS:
            nav, lg, st = run(k, s, cap)
            RUN[(k, s, cap)] = (nav, lg)
print('引擎 60 次', round(time.time() - T0), 's', rss(), 'MB', flush=True)
nav0, lg0 = RUN[('CUR', 0, 1e6)]
a0 = summ(nav0, lg0)
acc['A0_engine_current'] = dict(since2019=a0['since2019'], full=a0['full'], maxdd=a0['maxdd_full'], n_closed=a0['n_closed_all'],
                                ok=bool(abs(a0['since2019'] - 0.4899) < 0.00025 and abs(a0['full'] - 0.3454) < 0.00025 and abs(a0['maxdd_full'] + 0.3211) < 0.00025 and a0['n_closed_all'] == 381))
assert acc['A0_engine_current']['ok'], acc['A0_engine_current']
for k in ('FPM', 'FPA'):
    nv, _ = RUN[(k, 0, 1e6)]
    d = (nv.to_numpy() != nav0.to_numpy()) & ~(np.isnan(nv.to_numpy()) & np.isnan(nav0.to_numpy()))
    first = nv.index[np.flatnonzero(d)[0]] if d.any() else None
    acc[f'{k}_first_nav_diff_date'] = str(first.date()) if first is not None else None
    rel = np.abs(nv.to_numpy() / nav0.to_numpy() - 1)
    big = np.flatnonzero(np.nan_to_num(rel) > 1e-9)
    acc[f'{k}_first_nav_reldiff_gt_1e-9_date'] = str(nv.index[big[0]].date()) if len(big) else None
    acc[f'{k}_max_reldiff_before_first_gt_1e-9'] = float(np.nanmax(rel[:big[0]])) if len(big) else float(np.nanmax(rel))
    for th in (1e-6, 1e-4, 1e-3):
        bb = np.flatnonzero(np.nan_to_num(rel) > th)
        acc[f'{k}_first_nav_reldiff_gt_{th:g}_date'] = str(nv.index[bb[0]].date()) if len(bb) else None
    acc[f'{k}_reldiff_at_first_diff'] = float(rel[big[0]]) if len(big) else 0.0
    assert nv.index.equals(nav0.index)
    acc[f'{k}_nav_identical_before_window'] = bool(first is None or first > DW)
res['acceptance'] = acc

std = {}
for k in SV:
    nav, lg = RUN[(k, 0, 1e6)]
    o = summ(nav, lg)
    nx2, lx2, _ = run(k, 0, 1e6, 'strict_x2')
    o['cagr_window_costx2'] = stlib.cagr(nx2, DW, DE)
    kk, cols = trade_keys(lg)
    o['n_trades_entered_window'] = int((P.IDX[lg['ent_t'].to_numpy()] > DW).sum())
    o['n_closed_window'] = int(((P.IDX[lg['ent_t'].to_numpy()] > DW) & (lg['why'] != 'open_end')).sum())
    if k != 'CUR':
        k0, _ = trade_keys(lg0)
        o['trades_only_in_this'] = len(kk - k0); o['trades_only_in_CUR'] = len(k0 - kk)
        o['trade_diff'] = dict(only_this=sorted(kk - k0), only_cur=sorted(k0 - kk), cols=cols)
        b = std['CUR']
        for f in ('cagr_window', 'maxdd_window', 'cagr_window_costx2', 'since2019', 'full'):
            o[f'd_{f}_pp'] = (o[f] - b[f]) * 100
    std[k] = o
paths = {}; pairs = {}
for k in SV:
    paths[k] = {f'{int(c/1e4)}萬|+{s}': summ(*RUN[(k, s, c)]) for s in SH for c in CAPS}
for k in ('FPM', 'FPA'):
    pairs[k] = {}
    for f in ('cagr_window', 'since2019', 'maxdd_window'):
        d = np.array([(paths[k][p][f] - paths['CUR'][p][f]) * 100 for p in paths['CUR']])
        pairs[k][f] = dict(median_pp=float(np.median(d)), min_pp=float(d.min()), max_pp=float(d.max()), n_lower=int((d < -1e-12).sum()),
                           n_higher=int((d > 1e-12).sum()), n_equal=int((np.abs(d) <= 1e-12).sum()))
res['engine'] = dict(window=[str(DW.date()), str(DE.date())], since2019_from=str(D19.date()), standard=std, paths=paths, paths_pairs=pairs)
print('標準路徑', {k: (round(v['cagr_window'] * 100, 2), round(v['since2019'] * 100, 2)) for k, v in std.items()}, flush=True)
print('20 條', json.dumps(pairs, ensure_ascii=False), flush=True)

# ================================================================ 5. 名單差異（位移 0）
lists = {}
cur_info = {x['month']: x for x in SIGS[('CUR', 0)]['info']}
for k in ('FPM', 'FPA'):
    inf = {x['month']: x for x in SIGS[(k, 0)]['info']}
    t20d = {}; f1d = {}
    for m in MONTHS:
        ms = str(m)
        if ms not in cur_info:
            continue
        kk = cur_info[ms]['k']; assert inf[ms]['k'] == kk
        a = set(np.flatnonzero(~SIGS[('CUR', 0)]['X'][kk])); b = set(np.flatnonzero(~SIGS[(k, 0)]['X'][kk]))
        if a != b:
            t20d[ms] = dict(added=[str(P.COLS[j]) for j in sorted(b - a)], removed=[str(P.COLS[j]) for j in sorted(a - b)])
        fa, fb = set(cur_info[ms]['targets']), set(inf[ms]['targets'])
        if fa != fb:
            f1d[ms] = dict(added=sorted(fb - fa), removed=sorted(fa - fb))
    sch = 0
    for m in MONTHS:
        kk = AV[m]
        if kk >= P.T:
            continue
        a = SV['CUR'].loc[m].to_numpy(); b = SV[k].loc[m].to_numpy(); u = U[kk]
        sch += int((u & ~((a == b) | (np.isnan(a) & np.isnan(b)))).sum())
    lists[k] = dict(t20_months_diff=len(t20d), t20=t20d, f1_months_diff=len(f1d), f1=f1d, n_months=int(sum(str(m) in cur_info for m in MONTHS)),
                    surge_changed_cells_in_univ=sch)
res['lists'] = lists
print('名單', json.dumps({k: (v['t20_months_diff'], v['f1_months_diff'], v['t20']) for k, v in lists.items()}, ensure_ascii=False), flush=True)

res['runtime_s'] = round(time.time() - T0, 1); res['maxrss_mb'] = rss()


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


json.dump(clean(res), open(HERE / 'results_rep.json', 'w'), ensure_ascii=False, indent=1)
print('完成', res['runtime_s'], 's', res['maxrss_mb'], 'MB')
