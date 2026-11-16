#!/usr/bin/env python3
"""S40 獨立複製（python3 -I -B rep_s40.py）。只依 SPEC.json 文字與 FINDINGS.md 條件撰寫；未讀、未 import、未執行 s40 的任何 .py。
原始來源：
- 法定期限：證交法 36 條（次月 10 日、遇假日順延；以交易日近似上班日）；特例日期依 SPEC（2019-01→2019-02-15、2020-01→2020-02-15 敏感度、2023-03→2023-04-12；
  2026-01 起保險業及具保險子公司 15 日）。
- 逐檔首次可見時間：agents/o_rev_event_clock_ts/data/firstseen_sources.csv.gz（source＝tasi 的 T、T_prev）；
  Tasi 各月檔快照時間：本複製員以 `git log --raw` 列出（dl/vers_use.txt，不執行 repo 內任何程式）；
  2026-09 另用 agents/rev_announce/firstseen_2026-09.csv（tw-stock-screener 首見時間），取兩者較早者。
- 引擎 s20lib → s17lib → s10lib（不改），cslib.EXRIGHT_FIX=True；可得日／SURGE 的替換寫在本檔（最小包裝）。
輸出：本資料夾 results_rep.json。暫存：scratchpad/s3940rep/（結束刪除）。"""
import sys, json, time, resource, gc
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
TZ = 'Asia/Taipei'
SPECIAL = {pd.Period('2019-01', 'M'): ('2019-02-15', '金管會特例：春節連假延至 2019-02-15（全體）', 'all'),
           pd.Period('2020-01', 'M'): ('2020-02-15', '疫情：受影響公司經申請可延至 2020-02-15（非全體，敏感度）', 'sens'),
           pd.Period('2023-03', 'M'): ('2023-04-12', '金管會公告：清明連假延至 2023-04-12（全體）', 'all')}
INS15_FROM = pd.Period('2026-01', 'M')
# 上市的保險業及具保險子公司公發公司（11 家金控中上市的 10 家、上市保險公司、三商投控 2905、和泰車 2207；台灣金未上市）
INS15 = {'2880', '2881', '2882', '2883', '2885', '2886', '2887', '2891', '2892', '5880',
         '2816', '2832', '2850', '2851', '2852', '2867', '2905', '2207'}
VMONTHS = pd.period_range('2024-06', '2026-09', freq='M')

P = s20lib.load()
assert P.exright_fix is True
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
del INNOV
F = s17lib.rev_family(P, defs=['STD'])
S_ref = F['S']['STD']; AV_ref = F['av']; Rv = F['Rv']; del F
s17lib.e50_fix_suspension(P)
for a in ('CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq'):
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
IDXv = P.IDX.values
print('載入', round(time.time() - T0), 's', rss(), 'MB', flush=True)


def first_td_on_or_after(d):
    return int(np.searchsorted(IDXv, np.datetime64(pd.Timestamp(d)), side='left'))


def avail(months, override=None):
    """同 cslib.rev_avail_days(pit=True) 規則：期限日（10 日起第一個交易日）的下一個交易日；override＝{m: 延後後的期限日}。"""
    out = {}
    for m in months:
        d = (m + 1).to_timestamp() + pd.Timedelta(days=9)
        if override and m in override:
            d = pd.Timestamp(override[m])
        out[m] = first_td_on_or_after(d) + 1
    return out


res = {}
AV0 = avail(list(S_ref.index))
AVX = avail(list(pd.period_range('2015-01', '2026-10', freq='M')))
res['acceptance'] = dict(av_rebuilt_equal=bool(AV0 == AV_ref))
assert res['acceptance']['av_rebuilt_equal']

# ================================================================ 第 1 步：期限盤點
cal = []
for m in pd.period_range('2015-01', '2026-10', freq='M'):
    k = AVX[m]
    if k >= P.T:
        continue
    tenth = (m + 1).to_timestamp() + pd.Timedelta(days=9)
    dstat = P.IDX[first_td_on_or_after(tenth)]
    sp = SPECIAL.get(m)
    dact = P.IDX[first_td_on_or_after(sp[0])] if sp else dstat
    sd = P.IDX[k]
    case = 'safe' if dact < sd else ('boundary' if dact == sd else 'problem')
    row = dict(rev_month=str(m), tenth=str(tenth.date()), deadline_statutory=str(dstat.date()), special=sp[1] if sp else '', special_scope=sp[2] if sp else '',
               deadline_actual=str(dact.date()), signal_day=str(sd.date()), case=case,
               trading_days_short=int(first_td_on_or_after(dact) - k) if case == 'problem' else 0)
    if m >= INS15_FROM:
        d15 = P.IDX[first_td_on_or_after((m + 1).to_timestamp() + pd.Timedelta(days=14))]
        row['deadline_ins15'] = str(d15.date())
        row['ins15_case'] = 'safe' if d15 < sd else ('boundary' if d15 == sd else 'problem')
        row['ins15_trading_days_short'] = int(first_td_on_or_after(d15) - k)
    cal.append(row)
CAL = pd.DataFrame(cal)
res['step1'] = dict(n_months=int(len(CAL)), case_counts=CAL['case'].value_counts().to_dict(),
                    problem_months=CAL[CAL['case'] != 'safe'].to_dict('records'),
                    ins15=CAL[CAL['rev_month'] >= str(INS15_FROM)][['rev_month', 'signal_day', 'deadline_ins15', 'ins15_case', 'ins15_trading_days_short']].to_dict('records'))
try:
    oc = pd.read_csv(AG / 's40_revenue_deadline_pit/data/deadline_calendar.csv', dtype=str)
    mg = CAL.merge(oc, on='rev_month', suffixes=('', '_o'))
    res['step1']['vs_original_calendar'] = dict(n=int(len(mg)), signal_day_eq=int((mg['signal_day'] == mg['signal_day_o']).sum()),
                                                deadline_actual_eq=int((mg['deadline_actual'] == mg['deadline_actual_o']).sum()),
                                                deadline_statutory_eq=int((mg['deadline_statutory'] == mg['deadline_statutory_o']).sum()),
                                                ins15_eq=int((mg['deadline_ins15'].fillna('') == mg['deadline_ins15_o'].fillna('')).sum()))
except Exception as e:                                          # noqa: BLE001
    res['step1']['vs_original_calendar'] = repr(e)
print('第1步', res['step1']['case_counts'], [(r['rev_month'], r['deadline_actual'], r['signal_day']) for r in res['step1']['problem_months']], flush=True)

# ================================================================ 首次可見時間
fs = pd.read_csv(AG / 'o_rev_event_clock_ts/data/firstseen_sources.csv.gz', dtype=str)
fs = fs[fs['source'] == 'tasi'][['code', 'ym', 'T', 'T_prev']]
fs['T'] = pd.to_datetime(fs['T']).dt.tz_convert(TZ); fs['T_prev'] = pd.to_datetime(fs['T_prev']).dt.tz_convert(TZ)
vers = pd.read_csv(DL / 'vers_use.txt', sep=' ', header=None, names=['file', 'commit', 'T', 'blob', 'st'])
vers['ym'] = vers['file'].str.extract(r'(\d{6})')[0].str[:4] + '-' + vers['file'].str.extract(r'(\d{6})')[0].str[4:]
vers['T'] = pd.to_datetime(vers['T']).dt.tz_convert(TZ)
SNAPS = {ym: sorted(g['T']) for ym, g in vers.groupby('ym')}
LAST = {ym: s[-1] for ym, s in SNAPS.items()}
# T_prev 核對：firstseen 的 T_prev 是否等於 git log 的前一個快照
chk = 0; tot = 0
for ym, g in fs[fs['ym'].isin([str(m) for m in VMONTHS])].groupby('ym'):
    sn = pd.Series(SNAPS[ym])
    for t, tp in zip(g['T'], g['T_prev']):
        prev = sn[sn < t]
        exp = prev.iloc[-1] if len(prev) else pd.NaT
        tot += 1; chk += int((pd.isna(exp) and pd.isna(tp)) or (exp == tp))
res['acceptance']['firstseen_T_prev_eq_gitlog_prev_snapshot'] = f'{chk}/{tot}'
FS = {(c, ym): (t, tp) for c, ym, t, tp in zip(fs['code'], fs['ym'], fs['T'], fs['T_prev'])}
scr = pd.read_csv(AG / 'rev_announce/firstseen_2026-09.csv', dtype=str).rename(columns={'Unnamed: 0': 'code'})
scr['first_seen'] = pd.to_datetime(scr['first_seen']).dt.tz_convert(TZ)
SCR = dict(zip(scr['code'], scr['first_seen']))
SCR_RUNS = sorted(scr['first_seen'].unique())


def classify(code, m, sd, mode='spec'):
    """回傳 (cat, T, T_prev, src)。sd＝訊號日（Timestamp，日期）。"""
    ym = str(m)
    night = pd.Timestamp(sd.date()).tz_localize(TZ) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    s18 = pd.Timestamp(sd.date()).tz_localize(TZ) + pd.Timedelta(hours=18)
    t, tp = FS.get((code, ym), (pd.NaT, pd.NaT)); src = 'tasi'
    last = LAST[ym]
    if ym == '2026-09' and code in SCR:
        t2 = SCR[code]
        if pd.isna(t) or t2 < t:
            runs = [r for r in SCR_RUNS if r < t2]
            tp2 = runs[-1] if runs else pd.NaT
            if pd.notna(tp) and (pd.isna(tp2) or tp > tp2) and tp < t2:
                tp2 = tp
            t, tp, src = t2, tp2, 'screener'
        last = max(last, SCR_RUNS[-1])
    if pd.notna(t):
        if t <= night:
            return 'OK', t, tp, src
        return ('LATE_SURE' if (pd.notna(tp) and tp >= s18) else 'LATE_GAP'), t, tp, src
    return ('UNSEEN_LATE' if last >= s18 else 'UNKNOWN'), t, tp, src


def legal_deadline(code, m):
    """該檔該月的法定期限日（交易日）。"""
    if m >= INS15_FROM and code in INS15:
        return P.IDX[first_td_on_or_after((m + 1).to_timestamp() + pd.Timedelta(days=14))]
    sp = SPECIAL.get(m)
    if sp and sp[2] == 'all':
        return P.IDX[first_td_on_or_after(sp[0])]
    return P.IDX[first_td_on_or_after((m + 1).to_timestamp() + pd.Timedelta(days=9))]


def refine(cat, code, m, sd, tp):
    """依法定期限細分（原研究結果中出現的 LATE_GAP_DL／LATE_LEGAL，本複製員依其名稱與筆數推定的定義）：
    LATE_GAP_DL：LATE_GAP，且法定期限 < 訊號日、前一個快照 ≤ 法定期限當晚（守法公告即訊號日當晚可見）→ 視為可見；
    LATE_LEGAL：UNKNOWN，且該檔適用 15 日期限、15 日期限 > 訊號日（法律允許晚交、快照到最後都沒有）→ 視為看不到。"""
    dl = legal_deadline(code, m)
    if cat == 'LATE_GAP' and dl < sd:
        dn = pd.Timestamp(dl.date()).tz_localize(TZ) + pd.Timedelta(hours=23, minutes=59, seconds=59)
        if pd.isna(tp) or tp <= dn:
            return 'LATE_GAP_DL'
    if cat == 'UNKNOWN' and m >= INS15_FROM and code in INS15 and dl > sd:
        return 'LATE_LEGAL'
    return cat


# ================================================================ 第 2 步：可見度（位移 0）＋各位移的遮罩
SIG0 = s10lib.make_signal_s10(P, U, dict(S=S_ref, av=AV0), shift=0)
info0 = {x['month']: x for x in SIG0['info']}
CELLS = {}                                    # shift -> DataFrame
for sh in (0, 1, 2, 3):
    rows = []
    for m in VMONTHS:
        k = AV0[m] + sh
        if k >= P.T or str(m) not in info0:
            continue
        sd = P.IDX[k]
        s = S_ref.loc[m].to_numpy()
        js = np.flatnonzero(U[k] & np.isfinite(s))
        for j in js:
            c = str(P.COLS[j])
            cat, t, tp, src = classify(c, m, sd)
            rows.append((str(m), str(sd.date()), c, cat, refine(cat, c, m, sd, tp), src, t, tp, bool(m >= INS15_FROM and c in INS15), float(s[j])))
    CELLS[sh] = pd.DataFrame(rows, columns=['rev_month', 'signal_day', 'code', 'cat', 'cat2', 'src', 'T', 'T_prev', 'ins15', 'surge'])
C0 = CELLS[0]
top20 = {}; f1 = {}
for ms, x in info0.items():
    if pd.Period(ms, 'M') in VMONTHS:
        top20[ms] = set(str(P.COLS[j]) for j in np.flatnonzero(~SIG0['X'][x['k']]))
        f1[ms] = set(x['targets'])
C0['in_top20'] = [c in top20.get(m, ()) for m, c in zip(C0['rev_month'], C0['code'])]
C0['in_f1'] = [c in f1.get(m, ()) for m, c in zip(C0['rev_month'], C0['code'])]
s2 = dict(n_months=int(C0['rev_month'].nunique()), univ_cells_total=int(len(C0)),
          univ_cat_spec=C0['cat'].value_counts().to_dict(), univ_cat_refined=C0['cat2'].value_counts().to_dict(),
          f1_total=int(C0['in_f1'].sum()), f1_by_cat_spec=C0[C0['in_f1']]['cat'].value_counts().to_dict(), f1_by_cat_refined=C0[C0['in_f1']]['cat2'].value_counts().to_dict(),
          top20_total=int(C0['in_top20'].sum()), top20_by_cat_spec=C0[C0['in_top20']]['cat'].value_counts().to_dict(),
          top20_by_cat_refined=C0[C0['in_top20']]['cat2'].value_counts().to_dict())
late = C0[~C0['cat2'].isin(['OK'])]
s2['f1_not_ok'] = late[late['in_f1']][['rev_month', 'signal_day', 'code', 'cat', 'cat2', 'src', 'ins15']].astype(str).to_dict('records')
s2['top20_not_ok'] = late[late['in_top20']][['rev_month', 'signal_day', 'code', 'cat', 'cat2', 'src', 'ins15']].astype(str).to_dict('records')
MASK_SETS = {'PM': {'LATE_SURE', 'LATE_GAP', 'UNSEEN_LATE', 'LATE_LEGAL'}, 'PMS': {'LATE_SURE', 'UNSEEN_LATE', 'LATE_LEGAL'},
             'PMW': {'LATE_SURE', 'LATE_GAP', 'UNSEEN_LATE', 'LATE_LEGAL', 'LATE_GAP_DL'}}
pm0 = C0[C0['cat2'].isin(MASK_SETS['PM'])]
s2['pm_masked_by_month'] = pm0.groupby('rev_month').size().to_dict()
s2['pm_masked_ins15'] = int(pm0['ins15'].sum())
s2['pm_masked_non_ins15'] = pm0[~pm0['ins15']][['rev_month', 'code', 'cat2', 'src']].astype(str).to_dict('records')
s2['non_ins15_late_spec'] = C0[(~C0['ins15']) & C0['cat'].isin(['LATE_SURE', 'LATE_GAP', 'UNSEEN_LATE'])].groupby(['rev_month', 'cat']).size().astype(int).reset_index().astype(str).to_dict('records')
# S36「0 天」6 筆
s36 = [('2025-05', '5469'), ('2025-06', '3712'), ('2025-10', '2344'), ('2026-03', '2347'), ('2026-04', '6005'), ('2026-07', '3003')]
s2['s36_zero_day'] = [dict(rev_month=m, code=c, **{k: str(v) for k, v in C0[(C0['rev_month'] == m) & (C0['code'] == c)][['cat', 'T', 'T_prev', 'in_f1']].iloc[0].to_dict().items()})
                      if len(C0[(C0['rev_month'] == m) & (C0['code'] == c)]) else dict(rev_month=m, code=c, cat='not_in_univ') for m, c in s36]
# 原研究 visibility_cells 對照
try:
    ov = pd.read_csv(AG / 's40_revenue_deadline_pit/data/visibility_cells.csv', dtype={'code': str})
    mg = C0.merge(ov[['rev_month', 'code', 'cat', 'ins15']], on=['rev_month', 'code'], how='outer', suffixes=('', '_o'), indicator=True)
    b = mg[mg['_merge'] == 'both']
    s2['vs_original_cells'] = dict(n_orig=int(len(ov)), n_both=int(len(b)), orig_not_in_mine=int((mg['_merge'] == 'right_only').sum()),
                                   cat2_eq_orig=int((b['cat2'] == b['cat_o']).sum()),
                                   mismatches=b[b['cat2'] != b['cat_o']][['rev_month', 'code', 'cat', 'cat2', 'cat_o', 'src']].astype(str).to_dict('records'),
                                   ins15_mismatch=b[b['ins15'].astype(bool) != b['ins15_o'].astype(bool)][['rev_month', 'code', 'ins15', 'ins15_o']].astype(str).to_dict('records'))
except Exception as e:                                          # noqa: BLE001
    s2['vs_original_cells'] = repr(e)

# Tasi 每月可見比例（P.COLS 內、當月有營收者）
tm = []
for m in VMONTHS:
    ym = str(m); row = CAL[CAL['rev_month'] == ym].iloc[0]
    codes = [c for c in Rv.columns[np.isfinite(Rv.loc[m].to_numpy())]]
    Ts = pd.Series([FS.get((c, ym), (pd.NaT, pd.NaT))[0] for c in codes])
    nightof = lambda d: pd.Timestamp(d).tz_localize(TZ) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    n = len(codes)
    tt = Ts.dropna().sort_values()
    d95 = d99 = None
    for q, nm in ((0.95, 'd95'), (0.99, 'd99')):
        need = int(np.ceil(q * n))
        if len(tt) >= need:
            v = str(tt.iloc[need - 1].date())
            d95 = v if nm == 'd95' else d95; d99 = v if nm == 'd99' else d99
    tm.append(dict(rev_month=ym, signal_day=row['signal_day'], deadline_actual=row['deadline_actual'], tasi_last_snapshot=str(LAST[ym]), n_rev=n, n_seen=int(Ts.notna().sum()),
                   share_by_deadline_night=float((Ts <= nightof(row['deadline_actual'])).sum() / n), share_by_signal_night=float((Ts <= nightof(row['signal_day'])).sum() / n),
                   date_95=d95, date_99=d99))
s2['tasi_monthly'] = tm
try:
    ot = pd.read_csv(AG / 's40_revenue_deadline_pit/data/tasi_monthly.csv', dtype=str)
    mt = pd.DataFrame(tm).merge(ot, on='rev_month', suffixes=('', '_o'))
    s2['tasi_monthly_vs_original'] = dict(n=int(len(mt)), n_rev_eq=int((mt['n_rev'].astype(str) == mt['n_rev_o']).sum()), n_seen_eq=int((mt['n_seen'].astype(str) == mt['n_seen_o']).sum()),
                                          max_abs_diff_share_signal=float((mt['share_by_signal_night'] - mt['share_by_signal_night_o'].astype(float)).abs().max()),
                                          max_abs_diff_share_deadline=float((mt['share_by_deadline_night'] - mt['share_by_deadline_night_o'].astype(float)).abs().max()))
except Exception as e:                                          # noqa: BLE001
    s2['tasi_monthly_vs_original'] = repr(e)
res['step2'] = s2
print('第2步', json.dumps({k: s2[k] for k in ('univ_cells_total', 'univ_cat_spec', 'univ_cat_refined', 'f1_by_cat_refined', 'top20_by_cat_refined')}, ensure_ascii=False), flush=True)
print('對照', json.dumps(s2.get('vs_original_cells'), ensure_ascii=False)[:1500], flush=True)

# ================================================================ 第 3 步：引擎
OV_DL = {m: SPECIAL[m][0] for m in SPECIAL if SPECIAL[m][2] == 'all'}
OV_DL20 = {m: SPECIAL[m][0] for m in SPECIAL}
AVS = {'CUR': AV0, 'DL': avail(list(S_ref.index), OV_DL), 'DL20': avail(list(S_ref.index), OV_DL20)}


def masked_S(sh, cats, mode='nan'):
    S2 = S_ref.copy()
    x = CELLS[sh]
    x = x[x['cat2'].isin(cats)] if not isinstance(cats, tuple) else x[x[cats[0]].isin(cats[1])]
    ci = S2.columns.get_indexer(x['code']); ri = S2.index.get_indexer(pd.PeriodIndex(x['rev_month'], freq='M'))
    A = S2.to_numpy(copy=True)
    if mode == 'nan':
        A[ri, ci] = np.nan
    else:                                                       # PST：上月 SURGE
        A[ri, ci] = S_ref.to_numpy()[ri - 1, ci]
    return pd.DataFrame(A, index=S2.index, columns=S2.columns), int(len(x))


SPEC_LIT = ('cat', {'LATE_SURE', 'LATE_GAP', 'UNSEEN_LATE'})            # SPEC 字面版 PM（不細分）
VAR = {'CUR': ('CUR', None, 'nan'), 'DL': ('DL', None, 'nan'), 'DL20': ('DL20', None, 'nan'),
       'PM': ('CUR', MASK_SETS['PM'], 'nan'), 'PMS': ('CUR', MASK_SETS['PMS'], 'nan'), 'PMW': ('CUR', MASK_SETS['PMW'], 'nan'),
       'PST': ('CUR', MASK_SETS['PM'], 'prev'), 'ALL': ('DL', MASK_SETS['PM'], 'nan'),
       'PM_SPEC': ('CUR', SPEC_LIT, 'nan'), 'ALL_SPEC': ('DL', SPEC_LIT, 'nan')}
SH = (0, 1, 2, 3); CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D0, DE, D19, DW = P.IDX[P.t0], P.IDX[-1], lastb('2019-01-01'), lastb('2024-07-01')
CELL = dict(sat_w=0.7, sizing='W10', lock=False)


def run(SG, cap, cost='strict'):
    return s10lib.simulate_s10(P, SG, U, capital=cap, cap_pct=0.05, cost=COSTS[cost], **CELL)


def summ(nav, lg):
    return dict(since2019=stlib.cagr(nav, D19, DE), full=stlib.cagr(nav, D0, DE), maxdd_full=stlib.maxdd(nav, D0, DE),
                cagr_window=stlib.cagr(nav, DW, DE), maxdd_window=stlib.maxdd(nav, DW, DE), n_closed_all=int((lg['why'] != 'open_end').sum()), end_nav=float(nav.iloc[-1]))


def tkeys(lg):
    return set((str(c), str(P.IDX[e].date()), str(P.IDX[x].date()), w) for c, e, x, w in zip(lg['code'], lg['ent_t'], lg['exit_t'], lg['why']))


RUN = {}; NMASK = {}; INFO0 = {}; X2 = {}
for v, (avk, cats, mode) in VAR.items():
    for sh in SH:
        if cats is None:
            S = S_ref
        else:
            S, nm = masked_S(sh, cats, mode); NMASK[f'{v}|+{sh}'] = nm
        SG = s10lib.variant_S(s10lib.make_signal_s10(P, U, dict(S=S, av=AVS[avk]), shift=sh), 'E0')[0]
        SG.pop('XV', None)                                         # 省記憶體（E0 只用 X）
        for cap in CAPS:
            nav, lg, _ = run(SG, cap)
            RUN[(v, sh, cap)] = (nav, lg)
        if sh == 0:
            INFO0[v] = SG['info']
            X2[v] = run(SG, 1e6, 'strict_x2')[0]
        del SG, S
        gc.collect()
print('引擎', len(RUN), '次', round(time.time() - T0), 's', rss(), 'MB', flush=True)
nav0, lg0 = RUN[('CUR', 0, 1e6)]
a0 = summ(nav0, lg0)
res['acceptance']['A0_engine_current'] = dict(since2019=a0['since2019'], full=a0['full'], maxdd=a0['maxdd_full'], n_closed=a0['n_closed_all'],
                                              ok=bool(abs(a0['since2019'] - 0.4899) < 0.00025 and abs(a0['full'] - 0.3454) < 0.00025 and abs(a0['maxdd_full'] + 0.3211) < 0.00025 and a0['n_closed_all'] == 381))
res['acceptance']['cur_S_identical'] = True
assert res['acceptance']['A0_engine_current']['ok']
std = {}
for v in VAR:
    nav, lg = RUN[(v, 0, 1e6)]
    o = summ(nav, lg)
    nx2 = X2[v]
    o['since2019_costx2'] = stlib.cagr(nx2, D19, DE); o['cagr_window_costx2'] = stlib.cagr(nx2, DW, DE)
    if v != 'CUR':
        d = (nav.to_numpy() != nav0.to_numpy())
        o['first_nav_diff_date'] = str(nav.index[np.flatnonzero(d)[0]].date()) if d.any() else None
        a, b = tkeys(lg), tkeys(lg0)
        o['trades_only_in_this'] = len(a - b); o['trades_only_in_CUR'] = len(b - a)
        for f in ('since2019', 'full', 'maxdd_full', 'cagr_window', 'maxdd_window', 'since2019_costx2', 'cagr_window_costx2'):
            o[f'd_{f}_pp'] = (o[f] - std['CUR'][f]) * 100
    std[v] = o
pairs = {}
for v in VAR:
    if v == 'CUR':
        continue
    pairs[v] = {}
    for f in ('since2019', 'cagr_window', 'maxdd_full'):
        d = np.array([(summ(*RUN[(v, sh, c)])[f] - summ(*RUN[('CUR', sh, c)])[f]) * 100 for sh in SH for c in CAPS])
        pairs[v][f] = dict(median_pp=float(np.median(d)), min_pp=float(d.min()), max_pp=float(d.max()), n_lower=int((d < -1e-12).sum()),
                           n_higher=int((d > 1e-12).sum()), n_equal=int((np.abs(d) <= 1e-12).sum()))
# F1 改變的月份
f1d = {}
c0 = {x['month']: x for x in INFO0['CUR']}
for v in VAR:
    if v == 'CUR':
        continue
    dd = {}
    for x in INFO0[v]:
        y = c0.get(x['month'])
        if y is None or set(y['targets']) != set(x['targets']) or y['k'] != x['k']:
            dd[x['month']] = dict(sig_date=x['sig_date'], cur_sig_date=y['sig_date'] if y else None, f1=x['targets'], cur_f1=y['targets'] if y else None)
    f1d[v] = dd
res['engine'] = dict(window=[str(DW.date()), str(DE.date())], since2019_from=str(D19.date()), standard=std, paths_pairs=pairs, n_cells_masked=NMASK, f1_diff_months=f1d)
print('標準路徑 2019 起差', {v: round(o.get('d_since2019_pp', 0), 3) for v, o in std.items()}, flush=True)
print('20 條中位數', {v: round(p['since2019']['median_pp'], 3) for v, p in pairs.items()}, flush=True)
print('遮罩', NMASK, flush=True)
res['runtime_s'] = round(time.time() - T0, 1); res['maxrss_mb'] = rss()


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [clean(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return str(o)
    return o


json.dump(clean(res), open(HERE / 'results_rep.json', 'w'), ensure_ascii=False, indent=1)
print('完成', res['runtime_s'], 's', res['maxrss_mb'], 'MB')
