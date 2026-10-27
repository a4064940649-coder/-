#!/usr/bin/env python3
"""專項檢查 (a)～(i)。輸出 results_checks.json。需先跑 run_main.py（讀 picks.json）。"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import ia_core as K

OUT = K.HERE
R = {}
picks = json.loads((OUT / 'picks.json').read_text(encoding='utf-8'))
IDX = K.IDX; C = K.C; O = K.O; RET = K.RET; FLAG = K.FLAG; SRC = K.SRC

# ---------- (a) 還原價與除權息 ----------
a = {}
q = K.MYADJ / K.ADJ
rq = (q / q.ffill().shift(1) - 1).abs()
a['adj_vs_cumprod_ret_mismatch_cells'] = int((rq > 1e-9).sum().sum())
prevc = C.ffill().shift(1)
raw = C / prevc - 1
m0 = (FLAG == 0) & RET.notna() & C.notna()
dif = (RET - raw).abs()
a['flag0_ret_ne_rawclose_ratio_cells'] = int(((dif > 1e-6) & m0).sum().sum())
a['flag_counts'] = {int(k): int(v) for k, v in FLAG.stack().value_counts().items()}
f1 = (FLAG == 1)
a['flag1_exday_ret0_by_year'] = {int(y): int(v) for y, v in f1.sum(axis=1).groupby(IDX.year).sum().items() if v > 0}
# 2021-10 起 flag 2（有 exdiv 參考價，報酬正確）的除權息日：真實除權息日報酬分佈
f2 = (FLAG == 2)
r2 = RET.where(f2).stack().dropna()
ref2 = (C / (1 + RET)).where(f2)
gap2 = (O / ref2 - 1).where(f2).stack().dropna(); oc2 = (C / O - 1).where(f2).stack().dropna()
a['flag2_exday_true_ret'] = {'n': int(len(r2)), 'mean_pct': round(float(r2.mean() * 100), 3), 'median_pct': round(float(r2.median() * 100), 3),
                             'mean_gap_open_vs_ref_pct': round(float(gap2.mean() * 100), 3), 'mean_open_to_close_pct': round(float(oc2.mean() * 100), 3)}
# flag 1 日當天開盤到收盤（已知部分）
oc1 = (C / O - 1).where(f1).stack().dropna()
a['flag1_open_to_close'] = {'n': int(len(oc1)), 'mean_pct': round(float(oc1.mean() * 100), 3)}
# exdiv_5y 與 flag 2 的對應（上市、面板期內）
ex = pd.read_csv(K.X / 'exdiv_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
ex = ex[(ex['market'] == 'TWSE') & ex['code'].isin(K.COLS) & ex['date'].isin(IDX)]
fl = np.array([FLAG.at[d, c] for d, c in zip(ex['date'], ex['code'])])
cl = np.array([C.at[d, c] for d, c in zip(ex['date'], ex['code'])])
ok = np.isfinite(cl)
a['exdiv5y_events_in_panel'] = int(ok.sum())
a['exdiv5y_events_flag_dist'] = {str(int(k)) if np.isfinite(k) else 'nan': int(v) for k, v in pd.Series(fl[ok]).value_counts().items()}
# 這些事件中，ret 是否 = 收盤/參考價 −1（參考價用 exdiv_5y 的 ref_price）
rr = np.array([RET.at[d, c] for d, c in zip(ex['date'], ex['code'])])
pcl = np.array([prevc.at[d, c] for d, c in zip(ex['date'], ex['code'])])
implied = cl / (ex['ref_price'].to_numpy() * pcl / ex['prev_close'].to_numpy()) - 1
e = np.abs(rr - implied)
a['exdiv5y_ret_matches_ref_within_0.1pct'] = round(float(np.nanmean(e[ok] < 1e-3)), 4)
R['a_adjustment'] = a
print('a', a, flush=True)

# ---------- (b) 52 週高點：還原價／假新高 ----------
b = {}
lim = pd.Series(np.where(IDX < pd.Timestamp('2015-06-01'), 0.07, 0.10), index=IDX)
nth = C.notna().cumsum()            # 上市後第幾個交易日
over = RET.gt(lim + 0.011, axis=0) & (nth > 5)
b['days_ret_above_limit_plus1pct'] = int(over.sum().sum())
b['by_flag'] = {str(k): int(v) for k, v in FLAG.where(over).stack().value_counts().items()}
# 選中股票在訊號日之前 252 日內有沒有「超過漲停」的日報酬（可能的假新高）
sus = []
for key in ('P1_long|n10', 'P2_rev5y|n10', 'P1_long|n30', 'P2_rev5y|n30'):
    for ed, cs in picks[key].items():
        e_i = IDX.get_loc(pd.Timestamp(ed)); t = e_i - 1
        for c in cs:
            w = over[c].iloc[max(0, t - 252):t + 1]
            if w.any():
                for dd in w.index[w]:
                    sus.append((key, ed, c, str(dd.date()), round(float(RET.at[dd, c]), 3), int(FLAG.at[dd, c]), int(SRC.at[dd, c])))
b['picks_with_above_limit_day_in_252d'] = sorted(set(sus))[:80]
b['n_pick_events_affected'] = len({(s[0], s[1], s[2]) for s in sus})
# 原始價 vs 還原價的 52 週高點比
S52, _ = K.rolling_signals('strict')
rawhi = C.loc[K.GOOD].rolling(252, min_periods=252).max().reindex(IDX)
rawr = (C / rawhi).to_numpy()
rows = []
for key in ('P1_long|n10', 'P2_rev5y|n10'):
    for ed, cs in picks[key].items():
        t = IDX.get_loc(pd.Timestamp(ed)) - 1
        for c in cs:
            j = K.CI[c]; rows.append((S52[t, j], rawr[t, j]))
rw = np.array(rows)
b['picks_n10_s52_adj_mean'] = round(float(np.nanmean(rw[:, 0])), 3)
b['picks_n10_s52_raw_mean'] = round(float(np.nanmean(rw[:, 1])), 3)
b['picks_n10_share_adj_ge_0.95_but_raw_lt_0.8'] = round(float(np.nanmean((rw[:, 0] >= 0.95) & (rw[:, 1] < 0.8))), 3)
R['b_52w'] = b
print('b', {k: v for k, v in b.items() if k != 'picks_with_above_limit_day_in_252d'}, flush=True)

# ---------- (c) 營收月份對齊 ----------
c_ = {}
Y5 = K.yoy_from_rev5y(); Y5s = K.yoy_from_rev5y(use_col=False); YL = K.yoy_from_long()
common = Y5.index.intersection(Y5s.index)
dd = (Y5.reindex(common) - Y5s.reindex(index=common, columns=Y5.columns)).abs().stack().dropna()
c_['rev5y_yoycol_vs_selfcomputed_within_0.1pp'] = round(float((dd < 0.1).mean()), 4)
# rev_5y 與長資料同月同碼營收是否相同（確認 ym 都是營收所屬月份，沒有錯位一個月）
r5 = pd.read_csv(K.X / 'rev_5y.csv.gz', dtype={'code': str}); r5['p'] = pd.PeriodIndex(r5['ym'], freq='M')
lg = pd.read_csv(K.LONG, dtype={'code': str}); lg['p'] = pd.PeriodIndex(pd.to_datetime(lg['month']), freq='M')
mg = r5.merge(lg[['p', 'code', 'revenue_k_twd']], on=['p', 'code'])
c_['same_month_match_rate'] = round(float((abs(mg['revenue'] / mg['revenue_k_twd'] - 1) < 1e-3).mean()), 4)
lg2 = lg.copy(); lg2['p'] = lg2['p'] + 1
mg2 = r5.merge(lg2[['p', 'code', 'revenue_k_twd']], on=['p', 'code'])
c_['shifted_1m_match_rate'] = round(float((abs(mg2['revenue'] / mg2['revenue_k_twd'] - 1) < 1e-3).mean()), 4)
# 例：各訊號月用到哪三個月
ex_ = {}
for d in ['2007-01-11', '2010-01-11', '2016-01-11', '2022-01-11', '2024-02-15', '2026-01-12']:
    m = pd.Period(pd.Timestamp(d), 'M'); ex_[d] = [str(m - 1), str(m - 2), str(m - 3)]
c_['months_used_examples'] = ex_
# 1 月換股：yoy3 是否跨年正確（以 2330 2022-01 為例手算）
x = YL['2330']
c_['2330_signal_2022-01_yoy3_long'] = {'2021-12': round(float(x[pd.Period('2021-12')]), 3), '2021-11': round(float(x[pd.Period('2021-11')]), 3),
                                        '2021-10': round(float(x[pd.Period('2021-10')]), 3), 'avg': round(float(K.yoy3_for_month(YL, pd.Period('2022-01'))[K.CI['2330']]), 3)}
R['c_revenue'] = c_
print('c', c_, flush=True)

# ---------- (d) 下市股 / (e) 停牌 ----------
d_ = {}
last_trade = C.apply(lambda s: s.last_valid_index())
rows = []; susp = []
for key in ('P1_long|n30', 'P1_long|n10', 'P2_rev5y|n30', 'P2_rev5y|n10'):
    eds = sorted(picks[key]);
    for k, ed in enumerate(eds):
        e0 = IDX.get_loc(pd.Timestamp(ed)); e1 = IDX.get_loc(pd.Timestamp(eds[k + 1])) if k + 1 < len(eds) else len(IDX) - 1
        for c in picks[key][ed]:
            lt = last_trade[c]
            if lt < IDX[e1]:
                seg = C[c].iloc[max(0, IDX.get_loc(lt) - 20):IDX.get_loc(lt) + 1]
                rows.append((key, ed, c, str(lt.date()), float(C.at[lt, c]), round(float(seg.iloc[-1] / seg.iloc[0] - 1), 3)))
            w = C[c].iloc[e0:e1 + 1]
            if w.isna().any() and lt >= IDX[e1]:
                susp.append((key, ed, c, int(w.isna().sum()), bool(np.isnan(K.On[e1, K.CI[c]]))))
d_['picks_delisted_during_holding'] = rows
d_['n'] = len(rows)
R['d_delist'] = d_
e_ = {'picks_with_missing_days_in_holding': susp[:60], 'n': len(susp), 'n_unsellable_at_next_rebalance': int(sum(s[4] for s in susp))}
# strict（窗內須全有值）vs lenient：池子少了多少
S52s, V60s = K.rolling_signals('strict'); S52l, V60l = K.rolling_signals('lenient')
def poolsize(S52, V60, d):
    t = IDX.get_loc(d); v = V60[t]; px = K.Cn[t]
    base = np.isfinite(v) & (v >= 1e7) & (px >= 10)
    o = np.argsort(-np.where(base, v, -np.inf))[:min(300, base.sum())]
    top = np.zeros(K.N, bool); top[o] = True
    return top & np.isfinite(S52[t])
lost = {}
for d in K.rebalance_days('2006-04-01'):
    a1 = poolsize(S52s, V60s, d); a2 = poolsize(S52l, V60l, d)
    lost.setdefault(d.year, []).append((int(a1.sum()), int(a2.sum()), int((a2 & ~a1).sum())))
e_['pool_strict_vs_lenient_by_year'] = {y: {'strict': round(np.mean([x[0] for x in v]), 1), 'lenient': round(np.mean([x[1] for x in v]), 1), 'extra_in_lenient': round(np.mean([x[2] for x in v]), 1)} for y, v in lost.items()}
R['e_suspension'] = e_
print('d', d_['n'], rows[:10], flush=True); print('e', {k: v for k, v in e_.items() if k != 'picks_with_missing_days_in_holding'}, flush=True)

# ---------- (f) 成交價 ----------
f_ = {}
AO_ex = (O * (K.ADJ / C)).to_numpy()   # 既有程式的還原開盤
ACf = K.ADJ.ffill().to_numpy()
chk = []
for key in ('P1_long|n10', 'P2_rev5y|n10'):
    for ed, cs in picks[key].items():
        e0 = IDX.get_loc(pd.Timestamp(ed))
        for c in cs:
            j = K.CI[c]
            mine = K.GROW[e0, j] * K.On[e0, j] / K.Cn[e0, j]           # 前收→今開
            theirs = AO_ex[e0, j] / ACf[e0 - 1, j]
            chk.append(abs(mine - theirs))
f_['overnight_factor_mine_vs_exlib_maxabs'] = float(np.nanmax(chk))
# 換股日是除權息日（flag 1）的檔次：賣方應拿到股利，引擎用 O/C 代替 O/參考價
n1 = 0; tot = 0
for key in ('P1_long|n30',):
    for ed, cs in picks[key].items():
        e0 = IDX.get_loc(pd.Timestamp(ed)); tot += len(cs)
        n1 += int(sum(K.Fn[e0, K.CI[c]] == 1 for c in cs))
f_['P1_n30_exec_on_flag1_exday_share'] = round(n1 / tot, 4)
f_['exec_day_is_next_panel_day'] = True
R['f_exec'] = f_
print('f', f_, flush=True)

# ---------- (g) 2024-01-02～02-14 缺口 ----------
g_ = {}
d0, d1 = pd.Timestamp('2023-12-29'), pd.Timestamp('2024-02-15')
g_['panel_days_between'] = int(((IDX > d0) & (IDX < d1)).sum())
x = pd.DataFrame({'ret': RET.loc[d1], 'raw': C.loc[d1] / C.loc[d0] - 1, 'flag': FLAG.loc[d1]}).dropna()
g_['n_stocks_on_0215'] = int(len(x)); g_['flag_dist_0215'] = {str(int(k)): int(v) for k, v in x['flag'].value_counts().items()}
g_['ret_eq_raw_share'] = round(float((abs(x['ret'] - x['raw']) < 1e-6).mean()), 3)
exg = ex[(ex['date'] > d0) & (ex['date'] < d1)]
g_['exdiv_events_inside_gap'] = int(len(exg))
gg = x.loc[[c for c in exg['code'] if c in x.index]]
g_['inside_gap_ret_differs_from_raw'] = int((abs(gg['ret'] - gg['raw']) > 1e-4).sum())
miss = [c for c in K.COLS if np.isfinite(C.at[d0, c]) and not np.isfinite(C.at[d1, c])]
g_['traded_1229_but_missing_0215'] = len(miss)
first_back = {c: C[c].loc[d1:].first_valid_index() for c in miss}
g_['missing_first_back_dates'] = pd.Series({c: str(v.date()) if v is not None else None for c, v in first_back.items()}).value_counts().head(8).to_dict()
held = picks['P2_rev5y|n30'].get('2023-12-12', [])
g_['held_from_2023-12-12_n30'] = held
g_['held_missing_on_0215_0216'] = [c for c in held if c in miss]
r = (1 + RET.loc[d1, held]) * 1
g_['held_ret_gap_mean_pct'] = round(float(r.mean() * 100 - 100), 2)
e50 = K.E50_raw
g_['0050_has_jan2024_days'] = int(((e50.index > d0) & (e50.index < d1)).sum())
g_['0050_ret_1229_to_0215_pct'] = round(float((K.E50.loc[d1] / K.E50.loc[d0] - 1) * 100), 2)
R['g_gap2024'] = g_
print('g', g_, flush=True)

# ---------- (h) 0050 ----------
h_ = {}
td = pd.read_csv('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/data_ext/tsec_0050.csv', header=None)
p = td[0].str.split('/', expand=True).astype(int)
td['date'] = pd.to_datetime(dict(year=p[0] + 1911, month=p[1], day=p[2])); c50 = td.set_index('date')[6].astype(float)
j = pd.concat([c50.rename('c'), e50.rename('a')], axis=1).dropna()
jump = (j['a'].pct_change() - j['c'].pct_change()).abs() > 0.003
impl = (j['c'].shift(1) - j['c'] * j['a'].shift(1) / j['a'])[jump]
h_['implied_dividends_2005_2019'] = {str(k.date()): round(float(v), 3) for k, v in impl.items()}
h_['split_2025_06'] = {str(k.date()): float(v) for k, v in e50.loc['2025-06-09':'2025-06-20'].items()}
rng = IDX[(IDX >= '2006-01-01')]
h_['panel_days_without_0050_obs'] = [str(d.date()) for d in rng if d not in e50.index][:20]
h_['n_panel_days_without_0050_obs'] = int(sum(d not in e50.index for d in rng))
h_['0050_days_not_in_panel_2006plus'] = int(sum((d >= pd.Timestamp('2006-01-01')) and (d not in IDX) for d in e50.index))
R['h_0050'] = h_
print('h', {k: v for k, v in h_.items() if k != 'implied_dividends_2005_2019'}, flush=True)

(OUT / 'results_checks.json').write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print('done')
