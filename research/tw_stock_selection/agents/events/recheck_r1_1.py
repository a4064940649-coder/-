#!/usr/bin/env python3
"""重測 r1-1：處置股主策略「逐年扣大盤 +2.4%～+4.6%，每年都正」是否站得住。
事前固定的重測項目（全部報告，不挑）：
 (0) 原樣重建 ev5 可成交事件表（處置第一天收盤買、最後一天收盤賣；進場鎖漲停剔除；異常報酬對流動性合格等權），
     核對逐年 n 與平均（16/64/76/129/83/353；2.43/3.61/2.83/3.37/4.59/3.70）。
 (1) 逐年同時報：n、平均、中位數、勝率、t_month（進場月份分群，主要）、t_date（參考）、月份數、正月份比例、
     剔除前 3 名平均、前 3 名占總和比例、Wilcoxon 符號秩 p、符號檢定 p、扣 0.5% 成本後平均。
 (2) 去重疊：A＝同一檔持有中遇到新處置 → 略過新事件（保留第一筆）；B＝併成一段持有（出場延到新處置最後一天）。
     一檔同時只有一個部位。逐年同 (1) 的全部統計。
 (3) 跨股同期重疊：日曆時間組合（同時持有所有事件股、等權每日再平衡），逐年報「投入日的每日超額（對流動性合格等權）」
     的累積、bp/日、t（以日為單位），未扣成本與扣成本兩版；用去重疊 A 的交易。
 (4) 空頭：加權指數收盤 < 200 日均線。
     4a 2022 年內：以「進場日前一交易日」收盤判定（進場前已知）；另報以公告日判定（ev5 的定義）。
     4b 日曆時間組合在 2022 年 TAIEX<MA200 的日子（以前一日收盤判定）的每日超額。
     4c 全期間空頭事件（2021～2026）彙總做對照。
"""
import sys, json, math
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats as SS
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evlib as L

HERE = Path(__file__).resolve().parent
D = L.load(); idx = D['idx']; C = D['close_raw']; adj = D['adj']
cols = {c: j for j, c in enumerate(C.columns)}
An, LU = adj.to_numpy(), D['limit_up_close'].to_numpy()
car = L.Car(D, 'mkt_liq')
dp = pd.read_csv(L.X / 'disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['date', 'start', 'end'])
dp = dp[dp['code'].isin(C.columns)].sort_values(['start', 'code']).reset_index(drop=True)
tw = D['twii'].sort_index(); ma = tw.rolling(200).mean()
bear_s = (tw < ma).reindex(idx, method='ffill')           # 當日收盤 < 200MA
bear_prev = bear_s.shift(1)                                 # 前一交易日收盤判定（進場時已知）

# ---------- (0) 重建 ----------
rows = []
for _, r in dp.iterrows():
    c = r['code']; j = cols[c]
    i_a = idx.searchsorted(r['date'], 'right') - 1; i_s = idx.searchsorted(r['start']); i_e = idx.searchsorted(r['end'], 'right') - 1
    if i_s >= len(idx) or i_e <= i_s or i_a < 6 or not np.isfinite(An[i_s, j]): continue
    o = {'date': idx[i_s], 'code': c, 'i_s': i_s, 'i_e': i_e, 'lu': bool(LU[i_s, j]),
         'bear_ann': bool(bear_s.iat[i_a]) if pd.notna(bear_s.iat[i_a]) else np.nan,
         'bear_prev': bool(bear_prev.iat[i_s]) if pd.notna(bear_prev.iat[i_s]) else np.nan}
    o['raw'], o['ab'], o['ab50'] = car(c, i_s, i_e)
    rows.append(o)
E = pd.DataFrame(rows)
T = E[~E['lu']].dropna(subset=['ab']).copy()


def one_position(E, mode):
    out = []
    for c, g in E.sort_values('i_s').groupby('code', sort=False):
        cur = None
        for _, r in g.sort_values('i_s').iterrows():
            if cur is not None and r['i_s'] <= cur['i_e']:
                if mode == 'merge' and r['i_e'] > cur['i_e']: cur['i_e'] = r['i_e']; cur['n_merged'] += 1
                continue
            if cur is not None: out.append(cur)
            cur = None
            if r['lu']: continue
            cur = {'date': r['date'], 'code': c, 'i_s': r['i_s'], 'i_e': r['i_e'], 'bear_ann': r['bear_ann'], 'bear_prev': r['bear_prev'], 'n_merged': 0}
        if cur is not None: out.append(cur)
    K = pd.DataFrame(out)
    K['raw'], K['ab'], K['ab50'] = zip(*[car(c, a, b) for c, a, b in zip(K['code'], K['i_s'], K['i_e'])])
    return K.dropna(subset=['ab']).sort_values('date').reset_index(drop=True)


KA = one_position(E, 'first'); KB = one_position(E, 'merge')


def st(df, col='ab'):
    df = df.dropna(subset=[col]); x = df[col]
    o = {'n': int(len(df))}
    if len(df) == 0: return o
    o.update({'mean_pct': round(x.mean() * 100, 2), 'median_pct': round(x.median() * 100, 2), 'hit': round(float((x > 0).mean()), 3),
              'net_mean_pct_cost0.5': round((x.mean() - 0.005) * 100, 2), 'net_median_pct_cost0.5': round((x.median() - 0.005) * 100, 2)})
    if len(df) < 3: return o
    m = df.groupby(df['date'].dt.to_period('M'))[col].mean(); d = df.groupby('date')[col].mean()
    s = x.sort_values(ascending=False)
    o.update({'t_month': L.tstat(m), 't_date': L.tstat(d), 't_event_naive': L.tstat(x), 'n_months': int(len(m)),
              'share_pos_months': round(float((m > 0).mean()), 3),
              'month_means_pct': {str(k): round(v * 100, 2) for k, v in m.items()} if len(m) <= 6 else None,
              'mean_ex_top3_pct': round(s.iloc[3:].mean() * 100, 2) if len(s) > 6 else None,
              'top3_share_of_sum': round(float(s.iloc[:3].sum() / x.sum()), 2) if x.sum() > 0 else None,
              'top3_pct': [round(v * 100, 1) for v in s.iloc[:3]],
              'wilcoxon_p': round(float(SS.wilcoxon(x).pvalue), 4),
              'sign_test_p': round(float(SS.binomtest(int((x > 0).sum()), int((x != 0).sum()), 0.5).pvalue), 4)})
    return o


def yearly(df):
    return {int(y): st(g) for y, g in df.groupby(df['date'].dt.year)}


res = {'rebuild_check': {'n_events': int(len(E)), 'n_tradable': int(len(T)),
                         'yearly_n_mean': {int(y): [int(len(g)), round(g['ab'].mean() * 100, 2)] for y, g in T.groupby(T['date'].dt.year)}},
       'yearly_base': yearly(T), 'yearly_dedup_A_keep_first': yearly(KA), 'yearly_dedup_B_merge': yearly(KB),
       'all_base': st(T), 'all_A': st(KA), 'all_B': st(KB),
       'dedup_counts': {'A': int(len(KA)), 'B': int(len(KB)), 'B_extended': int((KB['n_merged'] > 0).sum())}}

# ---------- (3) 日曆時間組合（跨股重疊）：逐年投入日超額 ----------
R = D['ret'].fillna(0).to_numpy(); mk = D['mkt_liq'].to_numpy()


def cal_daily(trades, rt):
    w = np.zeros(R.shape, dtype=np.float32); ent = np.zeros(len(idx))
    for code, i0, i1 in trades:
        w[i0 + 1:i1 + 1, cols[code]] += 1.0; ent[i0] += 1
    n = w.sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        g = (w * R).sum(axis=1) / np.where(n > 0, n, 1)
        cost = np.where(n > 0, np.r_[0, ent[:-1]] / np.where(n > 0, n, 1) * rt, 0)
    ex = pd.Series(np.where(n > 0, g - cost - mk, np.nan), index=idx)
    return ex.dropna()


def cal_stats(ex):
    if len(ex) < 3: return {'n_days': int(len(ex))}
    return {'n_days': int(len(ex)), 'excess_bp_day': round(float(ex.mean() * 1e4), 2), 't_day': L.tstat(ex),
            'cum_excess_pct': round(float((1 + ex).prod() - 1) * 100, 1), 'share_pos_days': round(float((ex > 0).mean()), 3)}


trA = list(zip(KA['code'], KA['i_s'], KA['i_e']))
res['calendar_dedupA_invested_days_vs_EWliq'] = {}
for rt in (0.0, 0.005):
    ex = cal_daily(trA, rt)
    res['calendar_dedupA_invested_days_vs_EWliq'][f'cost{rt}'] = {int(y): cal_stats(g) for y, g in ex.groupby(ex.index.year)}
    bp = bear_prev.reindex(ex.index)
    e22 = ex[ex.index.year == 2022]
    res['calendar_dedupA_invested_days_vs_EWliq'][f'cost{rt}_2022_bear_days'] = cal_stats(e22[bp.reindex(e22.index) == True])
    res['calendar_dedupA_invested_days_vs_EWliq'][f'cost{rt}_2022_bull_days'] = cal_stats(e22[bp.reindex(e22.index) == False])
    res['calendar_dedupA_invested_days_vs_EWliq'][f'cost{rt}_all_bear_days'] = cal_stats(ex[bp == True])

# ---------- (4) 空頭 ----------
b22 = bear_s[(bear_s.index.year == 2022)]
res['twii_2022_bear_days'] = {'n_trading_days': int(len(b22)), 'n_below_ma200': int(b22.sum()),
                              'first_below': str(b22[b22].index.min().date()) if b22.any() else None,
                              'last_below': str(b22[b22].index.max().date()) if b22.any() else None}
reg = {}
for name, K in (('base', T), ('A', KA), ('B', KB)):
    y22 = K[K['date'].dt.year == 2022]
    reg[name] = {'2022_bear_prevday': st(y22[y22['bear_prev'] == True]), '2022_bull_prevday': st(y22[y22['bear_prev'] == False]),
                 '2022_bear_ann': st(y22[y22['bear_ann'] == True]), '2022_bull_ann': st(y22[y22['bear_ann'] == False]),
                 'all_bear_prevday': st(K[K['bear_prev'] == True]), 'all_bull_prevday': st(K[K['bear_prev'] == False]),
                 'all_bear_by_year_n': {int(y): int(len(g)) for y, g in K[K['bear_prev'] == True].groupby(K['date'].dt.year)}}
res['regime'] = reg
res['regime_2022_bear_events_list'] = T[(T['date'].dt.year == 2022) & (T['bear_prev'] == True)][['date', 'code', 'ab']].assign(
    date=lambda x: x['date'].dt.strftime('%Y-%m-%d'), ab=lambda x: (x['ab'] * 100).round(2)).to_dict('records')

out = HERE / 'recheck_r1_1.json'
out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else str(x)), encoding='utf-8')

K_ = ['n', 'mean_pct', 'median_pct', 'hit', 't_month', 't_date', 'n_months', 'share_pos_months', 'mean_ex_top3_pct', 'top3_share_of_sum', 'wilcoxon_p', 'sign_test_p', 'net_median_pct_cost0.5']
print('rebuild', res['rebuild_check'], res['dedup_counts'])
for k in ('yearly_base', 'yearly_dedup_A_keep_first', 'yearly_dedup_B_merge'):
    print('==', k)
    for y, v in res[k].items(): print(y, {q: v.get(q) for q in K_}, v.get('top3_pct'), v.get('month_means_pct'))
for k in ('all_base', 'all_A', 'all_B'): print(k, {q: res[k].get(q) for q in K_})
print('== calendar'); print(json.dumps(res['calendar_dedupA_invested_days_vs_EWliq'], ensure_ascii=False))
print('== 2022 TWII', res['twii_2022_bear_days'])
for n_, v in reg.items():
    for k, s in v.items(): print(n_, k, s if k.endswith('_n') else {q: s.get(q) for q in K_})
print(res['regime_2022_bear_events_list'])
