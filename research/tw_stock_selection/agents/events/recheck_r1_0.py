#!/usr/bin/env python3
"""重測 r1-0：處置股主策略（ev5_disposal.py 的 base）在質疑點下是否成立。
事前固定的重測項目（全部報告）：
 (0) 原樣重建 ev5 事件表，核對 n=721、扣大盤 +3.62%。
 (1) 主要 t 一律用 t_month（進場月份分群）；t_date 只當參考。
 (2) 同一檔處置期重疊：A＝只保留第一筆（持有中遇到的新處置事件略過）；
     B＝併成一段持有（持有中遇到新處置 → 出場日延到新處置的最後一天）。一檔同時只有一個部位。
 (3) 剔除前 1%、前 10 名；對稱剔除上下 1%；winsorize 5%/95%（各期間內計算）。
 (4) 期間：挑選 2021-10～2023（再拆 2021-10～2022、2023）、驗證 2024～2026（再拆 2024～2025、2026）。
 (5) 出場日收盤跌停 → 延到第一個非跌停日收盤。跌停判定：收盤/前收 − 1 ≤ −9.5%（無公司行動）；
     2025-09-24 起另要求收盤 ≤ 最低價×1.001（history.csv.gz 有 low）。
 (6) 櫃買：用 history.csv.gz（2025-09-24～2026-10-22，上市＋上櫃）重建 2025-09-25 起開始的處置事件；
     報酬＝收盤價報酬，除息日用 exdiv_5y 的參考價還原；異常報酬對「同市場流動性合格等權」；同時用同一管線算上市股做對照。
 另報：多空（TAIEX vs 200MA）、同時套用 (2A)+(5) 的「嚴格版」與其日曆時間組合。
"""
import sys, json, math
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evlib as L

HERE = Path(__file__).resolve().parent
DATA = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data')
D = L.load(); idx = D['idx']; C = D['close_raw']; O = D['open']; adj = D['adj']; F = D['flag']
cols = {c: j for j, c in enumerate(C.columns)}
An, Cn, On, LQ, LU, Fn, Sn = adj.to_numpy(), C.to_numpy(), O.to_numpy(), D['liq'].to_numpy(), D['limit_up_close'].to_numpy(), F.to_numpy(), D['src'].to_numpy()
car = L.Car(D, 'mkt_liq')
dp_all = pd.read_csv(L.X / 'disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['date', 'start', 'end'])
sl = pd.read_csv(DATA / 'stock_list.csv', dtype={'code': str})
dp = dp_all[dp_all['code'].isin(C.columns)].sort_values(['start', 'code']).reset_index(drop=True)
tw = D['twii'].sort_index(); bull = (tw > tw.rolling(200).mean()).reindex(idx, method='ffill')

PER = {'sel_2021-10_2023': ('2021-10-01', '2023-12-31'), 'sel_a_2021-10_2022': ('2021-10-01', '2022-12-31'), 'sel_b_2023': ('2023-01-01', '2023-12-31'),
       'val_2024_2026': ('2024-01-01', '2026-12-31'), 'val_a_2024_2025': ('2024-01-01', '2025-12-31'), 'val_b_2026': ('2026-01-01', '2026-12-31'),
       'all': ('2021-01-01', '2026-12-31')}


def stats(df, col='ab'):
    df = df.dropna(subset=[col])
    o = {'n': int(len(df))}
    if len(df) < 5: return o
    m = df.groupby(df['date'].dt.to_period('M'))[col].mean(); d = df.groupby('date')[col].mean()
    o.update({'mean_ab_pct': round(df[col].mean() * 100, 2), 'median_ab_pct': round(df[col].median() * 100, 2),
              'mean_raw_pct': round(df['raw'].mean() * 100, 2) if 'raw' in df else None,
              'mean_vs0050_pct': round(df['ab50'].mean() * 100, 2) if 'ab50' in df and df['ab50'].notna().any() else None,
              'hit': round(float((df[col] > 0).mean()), 3), 't_month_MAIN': L.tstat(m), 't_date_ref': L.tstat(d),
              'n_months': int(len(m)), 'share_pos_months': round(float((m > 0).mean()), 3)})
    x = df[col].sort_values()
    k1 = max(1, int(round(len(x) * 0.01)))
    o['ex_top1pct_mean'] = round(x.iloc[:-k1].mean() * 100, 2)
    o['ex_top10_mean'] = round(x.iloc[:-10].mean() * 100, 2) if len(x) > 20 else None
    o['trim_both1pct_mean'] = round(x.iloc[k1:-k1].mean() * 100, 2)
    lo, hi = df[col].quantile([0.05, 0.95]); w = df.assign(**{col: df[col].clip(lo, hi)})
    o['winsor5_mean'] = round(w[col].mean() * 100, 2)
    o['winsor5_t_month'] = L.tstat(w.groupby(w['date'].dt.to_period('M'))[col].mean())
    ex10 = df.drop(x.index[-10:]) if len(x) > 20 else df
    o['ex_top10_t_month'] = L.tstat(ex10.groupby(ex10['date'].dt.to_period('M'))[col].mean())
    return o


def by_per(df, col='ab'):
    return {p: stats(df[(df['date'] >= a) & (df['date'] <= b)], col) for p, (a, b) in PER.items()}


def limit_down(j, i):
    if i < 1 or Fn[i, j] != 0 or not (np.isfinite(Cn[i, j]) and np.isfinite(Cn[i - 1, j])): return False
    return Cn[i, j] / Cn[i - 1, j] - 1 <= -0.095


# ---------- (0) 原樣重建 ----------
rows = []
for _, r in dp.iterrows():
    c = r['code']; j = cols[c]
    i_a = idx.searchsorted(r['date'], 'right') - 1; i_s = idx.searchsorted(r['start']); i_e = idx.searchsorted(r['end'], 'right') - 1
    if i_s >= len(idx) or i_e <= i_s or i_a < 6 or not np.isfinite(An[i_s, j]): continue
    o = {'date': idx[i_s], 'code': c, 'i_s': i_s, 'i_e': i_e, 'lu': bool(LU[i_s, j]),
         'bull': bool(bull.iat[i_a]) if pd.notna(bull.iat[i_a]) else np.nan,
         'src_s': int(Sn[i_s, j]) if np.isfinite(Sn[i_s, j]) else -1, 'src_e': int(Sn[i_e, j]) if np.isfinite(Sn[i_e, j]) else -1}
    o['raw'], o['ab'], o['ab50'] = car(c, i_s, i_e)
    # (5) 跌停出場延後
    k = i_e
    while k < len(idx) - 1 and limit_down(j, k): k += 1
    o['ld_exit'] = k > i_e; o['i_e2'] = k
    o['raw2'], o['ab2'], o['ab502'] = car(c, i_s, k)
    rows.append(o)
E = pd.DataFrame(rows)
T = E[~E['lu']].copy()
res = {'rebuild_check': {'n_events': int(len(E)), 'n_tradable': int(len(T)), 'tradable': by_per(T), 'untradable_included': by_per(E)}}


# ---------- (2) 重疊處理：一檔同時只有一個部位 ----------
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
            cur = {'date': r['date'], 'code': c, 'i_s': r['i_s'], 'i_e': r['i_e'], 'bull': r['bull'], 'n_merged': 0}
        if cur is not None: out.append(cur)
    K = pd.DataFrame(out)
    j_ = K['code'].map(cols)
    R = [car(c, a, b) for c, a, b in zip(K['code'], K['i_s'], K['i_e'])]
    K['raw'], K['ab'], K['ab50'] = zip(*R)
    e2 = []
    for c, a, b in zip(K['code'], K['i_s'], K['i_e']):
        j = cols[c]; k = b
        while k < len(idx) - 1 and limit_down(j, k): k += 1
        e2.append(k)
    K['i_e2'] = e2; K['ld_exit'] = K['i_e2'] > K['i_e']
    R2 = [car(c, a, b) for c, a, b in zip(K['code'], K['i_s'], K['i_e2'])]
    K['raw2'], K['ab2'], K['ab502'] = zip(*R2)
    return K.sort_values('date').reset_index(drop=True)


T2 = T.sort_values(['code', 'i_s']).copy()
prevE = E.sort_values(['code', 'i_s']).groupby('code')['i_e'].shift()      # 同檔前一筆（含買不到的）處置的結束日
T2['overlap_prev_any'] = (T2['i_s'] <= prevE.reindex(T2.index)).fillna(False)
prevT = T2.groupby('code')['i_e'].shift()
T2['overlap_prev_tradable'] = T2['i_s'] <= prevT
KA = one_position(E, 'first'); KB = one_position(E, 'merge')
res['overlap'] = {'n_tradable_overlapping_prev_any_event': int(T2['overlap_prev_any'].sum()),
                  'n_tradable_overlapping_prev_tradable_event': int(T2['overlap_prev_tradable'].sum()),
                  'overlap_only_events': by_per(T2[T2['overlap_prev_any']]),
                  'A_keep_first': by_per(KA), 'B_merge_holding': by_per(KB),
                  'B_n_positions_extended': int((KB['n_merged'] > 0).sum())}

# ---------- (5) 跌停出場 ----------
ld = T[T['ld_exit']]
res['limit_down_exit'] = {'n_tradable_with_limit_down_exit': int(len(ld)),
                          'mean_ab_close_exit_pct': round(ld['ab'].mean() * 100, 2), 'mean_ab_delayed_exit_pct': round(ld['ab2'].mean() * 100, 2),
                          'mean_extra_days': round(float((ld['i_e2'] - ld['i_e']).mean()), 2),
                          'base_delayed_exit': by_per(T.assign(raw=T['raw2'], ab=T['ab2'], ab50=T['ab502'])),
                          'A_keep_first_delayed_exit_STRICT': by_per(KA.assign(raw=KA['raw2'], ab=KA['ab2'], ab50=KA['ab502'])),
                          'B_merge_delayed_exit': by_per(KB.assign(raw=KB['raw2'], ab=KB['ab2'], ab50=KB['ab502']))}

# ---------- 其他：多空、src=3 ----------
KS = KA.assign(raw=KA['raw2'], ab=KA['ab2'], ab50=KA['ab502'])
res['regime_strict'] = {'bull': by_per(KS[KS['bull'] == True]), 'bear': by_per(KS[KS['bull'] == False]),
                        'share_bull_tradable': round(float((T['bull'] == True).mean()), 3)}
s3 = T[(T['src_s'] == 3) | (T['src_e'] == 3)]
res['src3'] = {'n_entry_src3': int((T['src_s'] == 3).sum()), 'n_exit_src3': int((T['src_e'] == 3).sum()),
               'src3_any': stats(s3), 'non_src3': stats(T.drop(s3.index))}

# ---------- 日曆時間組合（嚴格版：保留第一筆＋跌停延後） ----------
res['calendar_strict'] = {}
tr = list(zip(KA['code'], KA['i_s'], KA['i_e2']))
for p in ('sel_2021-10_2023', 'val_a_2024_2025', 'val_b_2026', 'val_2024_2026'):
    a, b = PER[p]
    for rt in (0.005, 0.01):
        v = L.calendar_portfolio(D, tr, a, min(pd.Timestamp(b), idx[-1]), rt=rt, idle='cash')
        res['calendar_strict'][f'{p}_cost{rt}'] = {'strategy': v['strategy'], '0050': v['0050'], 'EW_liquid': v['EW_liquid'],
                                                   't_vs_0050': v['t_vs_0050'], 't_vs_EW_invested': v['t_vs_EW_invested'],
                                                   'excess_vs_EW_bp_on_invested_days': v['excess_vs_EW_bp_on_invested_days'], 'share_days_invested': v['share_days_invested']}

# ---------- 範圍：不在面板的處置事件 ----------
miss = dp_all[~dp_all['code'].isin(C.columns)].merge(sl[['code', 'market']], on='code', how='left')
res['excluded_not_in_panel'] = {'by_market': {str(k): int(v) for k, v in miss['market'].fillna('unknown(delisted?)').value_counts().items()},
                                'unknown_market_by_year': {int(k): int(v) for k, v in miss[miss['market'].isna()].groupby(miss['start'].dt.year).size().items()}}

# ---------- (6) 櫃買：history.csv.gz ----------
h = pd.read_csv(DATA / 'history.csv.gz', dtype={'code': str}, parse_dates=['date'])
h = h[h['close'] > 0]
mk = sl.set_index('code')['market']
W = {k: h.pivot(index='date', columns='code', values=k).sort_index() for k in ('open', 'high', 'low', 'close', 'volume')}
hidx = W['close'].index; HC = W['close']
prev = HC.shift()
xd = pd.read_csv(L.X / 'exdiv_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
xd = xd[(xd['date'] >= hidx[0]) & xd['ref_price'].gt(0)]
ref = xd.pivot_table(index='date', columns='code', values='ref_price', aggfunc='last').reindex(index=hidx, columns=HC.columns)
base_prev = ref.where(ref.notna(), prev)
HR = (HC / base_prev - 1).where(HC.notna() & prev.notna())
HR = HR.where(HR.abs() < 0.5)          # 極端值（減資、合併等）視為缺值
val = W['volume'] * HC
hliq = (val.rolling(60, min_periods=20).mean() >= 1e7) & (HC >= 10)
HLU = (HC / prev - 1 >= 0.095) & (HC >= W['high'] * 0.999) & ref.isna()
HLD = (HC / prev - 1 <= -0.095) & (HC <= W['low'] * 1.001) & ref.isna()
LAh = np.log1p(HR.fillna(0).clip(lower=-0.99)).cumsum()
e50h = D['e50'].reindex(hidx).fillna(0); LE = np.log1p(e50h).cumsum()
pools = {}
for m_ in ('TWSE', 'TPEX'):
    cs = [c for c in HC.columns if mk.get(c) == m_]
    pools[m_] = np.log1p(HR[cs].where(hliq[cs]).mean(axis=1).fillna(0)).cumsum()
hcols = {c: j for j, c in enumerate(HC.columns)}
LAn, HLDn, HLUn = LAh.to_numpy(), HLD.to_numpy(), HLU.to_numpy()
dh = dp_all.merge(sl[['code', 'market']], on='code', how='inner')
dh = dh[dh['start'] > hidx[0]].sort_values(['start', 'code'])
hrows = []
for _, r in dh.iterrows():
    c = r['code']
    if c not in hcols: continue
    j = hcols[c]; i_s = hidx.searchsorted(r['start']); i_e = hidx.searchsorted(r['end'], 'right') - 1
    if i_s >= len(hidx) or i_e <= i_s or not np.isfinite(HC.iat[i_s, j]): continue
    k = i_e
    while k < len(hidx) - 1 and HLDn[k, j]: k += 1
    P = pools[r['market']].to_numpy()
    raw = math.expm1(LAn[k, j] - LAn[i_s, j]); raw0 = math.expm1(LAn[i_e, j] - LAn[i_s, j])
    hrows.append({'date': hidx[i_s], 'code': c, 'market': r['market'], 'i_s': i_s, 'i_e': i_e, 'lu': bool(HLUn[i_s, j]),
                  'raw0': raw0, 'ab0': raw0 - math.expm1(P[i_e] - P[i_s]),
                  'raw': raw, 'ab': raw - math.expm1(P[k] - P[i_s]), 'ab50': raw - math.expm1(LE.iat[k] - LE.iat[i_s]),
                  'ab_vs_twse_pool': raw - math.expm1(pools['TWSE'].iat[k] - pools['TWSE'].iat[i_s]), 'ld_exit': k > i_e})
H = pd.DataFrame(hrows)


def dedup_first(df):
    keep, last = [], {}
    for i, r in df.sort_values('i_s').iterrows():
        if r['code'] in last and r['i_s'] <= last[r['code']]: continue
        if r['lu']: continue      # 買不到：不建部位（但下一筆仍可進）
        keep.append(i); last[r['code']] = r['i_e']
    return df.loc[keep]


otc = {}
for m_ in ('TWSE', 'TPEX'):
    g = H[H['market'] == m_]
    gt = g[~g['lu']]
    gd = dedup_first(g)
    otc[m_] = {'n_events': int(len(g)), 'n_limit_up_entry': int(g['lu'].sum()),
               'tradable_close_exit': stats(gt.assign(ab=gt['ab0'], raw=gt['raw0'])),
               'tradable_delayed_exit': stats(gt), 'STRICT_dedup_delayed_exit': stats(gd),
               'STRICT_vs_TWSE_pool': stats(gd.assign(ab=gd['ab_vs_twse_pool'])),
               'n_limit_down_exit_tradable': int(gt['ld_exit'].sum()),
               'STRICT_ex_edge_months_2025-09_2026-10': stats(gd[(gd['date'] >= '2025-10-01') & (gd['date'] < '2026-10-01')]),
               'monthly_n_strict': {str(k): int(v) for k, v in gd.groupby(gd['date'].dt.to_period('M')).size().items()},
               'monthly_mean_ab_strict_pct': {str(k): round(v * 100, 2) for k, v in gd.groupby(gd['date'].dt.to_period('M'))['ab'].mean().items()}}
# 同期面板管線對照（上市，2025-09-25 起）
pw = KS[KS['date'] > hidx[0]]
otc['TWSE_panel_pipeline_same_window_STRICT'] = stats(pw)
res['otc_history_2025-09_2026-10'] = otc

out = HERE / 'recheck_r1_0.json'
out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')


def show(name, d, keys=('n', 'mean_ab_pct', 'median_ab_pct', 't_month_MAIN', 't_date_ref', 'share_pos_months', 'ex_top1pct_mean', 'ex_top10_mean', 'ex_top10_t_month', 'winsor5_mean', 'winsor5_t_month')):
    print(f'== {name}')
    for p, x in d.items():
        print(f'  {p:22s}', ' '.join(f'{k}={x.get(k)}' for k in keys))


show('rebuild tradable', res['rebuild_check']['tradable']); show('rebuild untradable incl', res['rebuild_check']['untradable_included'])
print({k: v for k, v in res['overlap'].items() if not isinstance(v, dict)})
show('overlap-only events', res['overlap']['overlap_only_events'])
show('A keep first', res['overlap']['A_keep_first']); show('B merge', res['overlap']['B_merge_holding'])
print({k: v for k, v in res['limit_down_exit'].items() if not isinstance(v, dict)})
show('base delayed LD exit', res['limit_down_exit']['base_delayed_exit'])
show('STRICT A+LD', res['limit_down_exit']['A_keep_first_delayed_exit_STRICT'])
show('B+LD', res['limit_down_exit']['B_merge_delayed_exit'])
show('STRICT bull', res['regime_strict']['bull']); show('STRICT bear', res['regime_strict']['bear'])
print('src3', res['src3']['n_entry_src3'], res['src3']['n_exit_src3'], res['src3']['src3_any'].get('mean_ab_pct'), res['src3']['non_src3'].get('mean_ab_pct'))
for k, v in res['calendar_strict'].items(): print('cal', k, v['strategy'], '0050', v['0050']['CAGR'], 'EW', v['EW_liquid']['CAGR'], 't0050', v['t_vs_0050'], 'tEW', v['t_vs_EW_invested'], v['share_days_invested'])
print('excluded', res['excluded_not_in_panel'])
for m_ in ('TWSE', 'TPEX'):
    x = otc[m_]; print('OTC-check', m_, x['n_events'], x['n_limit_up_entry'], x['n_limit_down_exit_tradable'])
    show(m_ + ' history', {k: x[k] for k in ('tradable_close_exit', 'tradable_delayed_exit', 'STRICT_dedup_delayed_exit', 'STRICT_vs_TWSE_pool', 'STRICT_ex_edge_months_2025-09_2026-10')})
    print('  monthly', x['monthly_mean_ab_strict_pct']); print('  monthly_n', x['monthly_n_strict'])
show('TWSE panel same window', {'strict': otc['TWSE_panel_pipeline_same_window_STRICT']})
