#!/usr/bin/env python3
"""S40 第 1、2 步：(1) 2015-01～2026-10 法定期限盤點＋與引擎訊號日比較；Tasi 每月累積可見比例交叉核對。
(2) 2024-06～2026-09 現行引擎 F1／L_MA 前 20 名（與全股池）在訊號日當晚是否已可見（Tasi 首次可見時間）。
輸出 data/deadline_calendar.csv、data/tasi_monthly.csv、data/visibility_cells.csv（股池內所有非 OK 的格子＋F1／前 20 名全部）、results.json 的 step1／step2。
python3 -I -B visibility.py"""
import sys, json, time, resource
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s40lib as L
from s40lib import s17lib, s10lib, s4lib, stlib

H = Path(__file__).resolve().parent
RES = H / 'results.json'
res = json.loads(RES.read_text()) if RES.exists() else {}
t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)

P = L.load()
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
F = s17lib.rev_family(P, defs=['STD'])
S, av, Rv = F['S']['STD'], F['av'], F['Rv']

# ---------------------------------------------------------------- (1) 期限盤點
months = list(pd.period_range('2015-01', '2026-10', freq='M'))
cal = L.deadline_table(P, months)
cal.to_csv(H / 'data' / 'deadline_calendar.csv', index=False)
s1 = dict(n_months=int(len(cal)), case_counts=cal['case'].value_counts().to_dict(),
          problem_months=cal[cal['case'].str.startswith('deadline>')][['rev_month', 'signal_day', 'deadline_statutory', 'deadline_actual', 'special', 'days_short']].to_dict(orient='records'),
          boundary_months_n=int(cal['case'].str.startswith('deadline=').sum()),
          ins15=cal.dropna(subset=['deadline_ins15'])[['rev_month', 'signal_day', 'deadline_ins15', 'ins15_case']].to_dict(orient='records'),
          special_2020=cal[cal['rev_month'] == '2020-01'][['rev_month', 'signal_day', 'deadline_actual', 'case']].to_dict(orient='records'),
          note='engine：訊號日＝期限日（10 日起第一個交易日）的下一個交易日，所以一般月份與 10 日遇假日月份都是 deadline<signal；只有主管機關特例延長（及 2026 起 15 日期限公司）會晚於訊號日。')
print(json.dumps(s1, ensure_ascii=False, default=str)[:3000], flush=True)

# ---------------------------------------------------------------- Tasi
seen = L.load_seen()
win = json.loads(L.TASI_WIN.read_text())['per_month']
ra = pd.read_csv(L.RA_2609, dtype=str).rename(columns={'Unnamed: 0': 'code'})
ra.columns = ['code'] + list(ra.columns[1:])
ra['first_seen'] = pd.to_datetime(ra['first_seen'], utc=True).dt.tz_convert(L.TZ)
ra = ra.set_index('code')['first_seen']
cols = set(P.COLS)
tasi_codes = set(seen['code'])
tm, cells = [], []
SIG = s10lib.make_signal_s10(P, U, dict(S=S, av=av), shift=0)
f1map = {x['k']: x['targets'] for x in SIG['info']}
for m in pd.period_range(L.FIRST_M, L.LAST_M, freq='M'):
    k = av[m]
    sd = P.IDX[k]
    send = pd.Timestamp(sd.date()).tz_localize(L.TZ) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    s18 = pd.Timestamp(sd.date()).tz_localize(L.TZ) + pd.Timedelta(hours=18)
    row = cal[cal['rev_month'] == str(m)].iloc[0]
    dd = pd.Timestamp(row['deadline_actual'])
    dend = dd.tz_localize(L.TZ) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    last = pd.Timestamp(win[str(m)]['last_commit']).tz_convert(L.TZ)
    g = seen[seen['m'] == m].drop_duplicates('code').set_index('code')
    # 分母：tw_revenue 該月有營收（>0）的上市代碼（P.COLS）
    has = Rv.loc[m].dropna().index
    has = [c for c in has if c in cols and c in tasi_codes]
    T = g['T'].reindex(has)
    if str(m) == '2026-09':
        T2 = ra.reindex(has)
        T = pd.concat([T, T2], axis=1).min(axis=1)
    Ts = T.dropna().sort_values()
    n = len(has)
    q = lambda p: str(Ts.iloc[int(np.ceil(p * n)) - 1].date()) if len(Ts) >= int(np.ceil(p * n)) else None
    tm.append(dict(rev_month=str(m), signal_day=str(sd.date()), deadline_actual=str(dd.date()), tasi_last_snapshot=str(last), n_rev=n,
                   n_seen=int(T.notna().sum()), share_by_deadline_night=float((T <= dend).sum() / n), share_by_signal_night=float((T <= send).sum() / n),
                   date_95=q(0.95), date_99=q(0.99), modal_first_date=str(Ts.dt.date.value_counts().idxmax()) if len(Ts) else None))
    # 逐格分類（股池內，SURGE 有值）
    s = S.loc[m].to_numpy()
    fin = U[k] & np.isfinite(s)
    ok_ma = fin & P.F_MA60[k]
    o_ma = np.flatnonzero(ok_ma)[np.argsort(-s[ok_ma], kind='stable')]
    t20 = set(P.COLS[j] for j in o_ma[:20])
    rank = {P.COLS[j]: i + 1 for i, j in enumerate(o_ma)}
    f1 = set(f1map.get(k, []))
    for j in np.flatnonzero(fin):
        c = P.COLS[j]
        Tc = g['T'].get(c) if c in g.index else None
        Tp = g['T_prev'].get(c) if c in g.index else None
        src = 'tasi'
        if str(m) == '2026-09' and c in ra.index and pd.notna(ra[c]) and (Tc is None or pd.isna(Tc) or ra[c] < Tc):
            Tc, Tp, src = ra[c], None, 'screener'
        ins_legal = (c in L.INS15) and (m >= L.INS15_START) and str(row.get('ins15_case')) == '問題'
        if Tc is not None and pd.notna(Tc):
            if Tc <= send:
                cat = 'OK'
            elif src == 'tasi':
                if pd.notna(Tp) and Tp >= s18:
                    cat = 'LATE_SURE'                      # 訊號日 18 時以後的快照仍沒有
                elif pd.notna(Tp) and Tp <= dend and not ins_legal:
                    cat = 'LATE_GAP_DL'                    # 快照缺口跨過期限當晚：可能準時公布（推定可見）
                else:
                    cat = 'LATE_GAP'
            else:
                cat = 'LATE_SURE' if Tc.date() > (sd + pd.Timedelta(days=1)).date() else 'LATE_GAP'
        else:
            cat = 'UNSEEN_LATE' if last >= s18 else ('LATE_LEGAL' if ins_legal else 'UNKNOWN')
        in20, inf1 = c in t20, c in f1
        if cat != 'OK' or in20 or inf1:
            cells.append(dict(rev_month=str(m), signal_day=str(sd.date()), code=c, cat=cat, src=src, T=str(Tc) if Tc is not None and pd.notna(Tc) else None,
                              T_prev=str(Tp) if Tp is not None and pd.notna(Tp) else None, in_top20=in20, in_f1=inf1, rank_lma=rank.get(c),
                              ins15=c in L.INS15, surge=float(s[j]), tasi_last=str(last)))
TM = pd.DataFrame(tm); TM.to_csv(H / 'data' / 'tasi_monthly.csv', index=False)
C = pd.DataFrame(cells); C.to_csv(H / 'data' / 'visibility_cells.csv', index=False)
late = C[C['cat'].isin(L.LATE_PM)]
by = lambda d: {str(k): int(v) for k, v in d.groupby('cat').size().items()}
s2 = dict(n_months=int(len(TM)),
          univ_cells_total=int(sum(int((U[av[m]] & np.isfinite(S.loc[m].to_numpy())).sum()) for m in pd.period_range(L.FIRST_M, L.LAST_M, freq='M'))),
          univ_noncompliant=by(C[C['cat'] != 'OK']),
          univ_late_by_month={k: int(v) for k, v in late.groupby('rev_month').size().items()},
          univ_late_ins15=int(late['ins15'].sum()),
          f1_total=int(C['in_f1'].sum()), f1_by_cat=by(C[C['in_f1']]),
          top20_total=int(C['in_top20'].sum()), top20_by_cat=by(C[C['in_top20']]),
          f1_late=C[C['in_f1'] & (C['cat'] != 'OK')].to_dict(orient='records'),
          top20_late=C[C['in_top20'] & (C['cat'] != 'OK')].to_dict(orient='records'),
          late_rank_le30=late[late['rank_lma'].notna() & (late['rank_lma'] <= 30)][['rev_month', 'code', 'cat', 'rank_lma', 'surge', 'ins15']].to_dict(orient='records'),
          tasi_monthly=TM.to_dict(orient='records'))
# S36「0 天」：可見列 r（T 的台北日期以前最後一個交易日）＝官方訊號列 av
ok = C[C['in_f1'] & C['T'].notna()].copy()
ok['r'] = [int(np.searchsorted(P.IDX.values, np.datetime64(pd.Timestamp(t[:10])), side='right') - 1) for t in ok['T']]
ok['k'] = [av[pd.Period(x, 'M')] for x in ok['rev_month']]
s2['s36_zero_day_f1'] = ok[ok['r'] >= ok['k']][['rev_month', 'signal_day', 'code', 'T', 'T_prev', 'cat']].to_dict(orient='records')
res['step1'] = s1; res['step2'] = s2
res['visibility_runtime_s'] = round(time.time() - t0, 1); res['visibility_maxrss_mb'] = rss()
RES.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
print(json.dumps({k: v for k, v in s2.items() if k not in ('tasi_monthly',)}, ensure_ascii=False, indent=0, default=str)[:6000])
print(TM.to_string())
print('done', res['visibility_runtime_s'], rss())
