#!/usr/bin/env python3
"""重測 r1_0：跳空下跌 ≥3%／≥5% 開盤買的加總方式、事件數分桶、拿掉大事件日、成本制度與斷點比較。

事前固定（不依結果調整）：
  出場（保守，同 r2_gaps.py）：收盤賣；收盤跌停 → 隔日開盤賣、成本 0.5%。
  加總方式：
    D  每日等權 cohort（原程式）：先同日事件等權，再對事件日平均；t＝事件日序列 t。
    T  每筆等權：所有事件平均；t 值按日期 cluster（CR1 小樣本修正 G/(G−1)）；另報中位數。
  分桶：當日事件數（該窗口內、該規則）1、2–3、4–10、11–30、>30。
  拿掉事件數最多的前 5／10／20 天後重算（T 與 D 都報）。
  成本（兩套）：
    base   ：原程式（當沖 0.23%，跌停改隔日開盤 0.5%）。
    regime ：2014-01-06 前不能現股當沖 → 開盤買、隔日收盤賣、0.585%（0.1425%×2＋0.3%稅）；
             2014-01-06～2017-04-27 當沖 0.38%（跌停隔日開盤 0.5%）；2017-04-28 起 0.23%（同 base）。
  斷點：2014-01-06（現股當沖開放）、2015-06-01（10% 漲跌幅）、2017-04-28（當沖稅減半）。
    各做「前 12 個月 vs 後 12 個月」與「相鄰整段」兩種前後比較，用毛報酬（保守出場）避免成本制度混入，
    另列 regime 淨報酬；差異 t＝(後−前)/sqrt(se前²+se後²)，se 用日期 cluster（T）或日序列（D）。
窗口同 stlib.WINDOWS：S0 2005-01～2015-05、S1 2015-06～2019-04、V_official（只用官方來源日）、V_all（含 src 3）。
"""
import sys, time, math, json
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/short_term')
import numpy as np, pandas as pd
import stlib as L

t0 = time.time()
D = L.load()
on, idr, elig, F, R = D['on'], D['id'], D['elig'], D['flag'], D['ret']
gdt, cdt = L.daytrade_exit(D)
r_next = R.shift(-1).where(F.shift(-1).isin([0, 1, 2]) & (R.shift(-1).abs() <= D['lim'].shift(-1).values[:, None] + 0.03))
g2 = (1 + idr) * (1 + r_next) - 1          # 開盤買、隔日收盤賣
CAL = D['CAL']
DT_OPEN = pd.Timestamp('2014-01-06'); TAX_CUT = pd.Timestamp('2017-04-28'); LIM10 = pd.Timestamp('2015-06-01')
COST_NC = 0.00585

# regime 版每筆毛報酬與成本
pre14 = pd.Series(CAL < DT_OPEN, index=CAL)
mid = pd.Series((CAL >= DT_OPEN) & (CAL < TAX_CUT), index=CAL)
g_reg = gdt.copy()
g_reg.loc[pre14.values] = g2.loc[pre14.values]
c_reg = cdt.copy()
c_reg.loc[pre14.values] = COST_NC
cldm = D['close_limit_down'].fillna(False).astype(bool)
c_reg.loc[mid.values] = c_reg.loc[mid.values].where(cldm.loc[mid.values], L.COST_DT_PRE2017)

rules = {'gap_down_3': on <= -0.03, 'gap_down_5': on <= -0.05}


def events(mask, g, c, days):
    """回傳事件長表：date, code, g, c, net（限 days 內）。"""
    m = (mask & g.notna()).reindex(days).fillna(False).astype(bool)
    st = m.stack(); st = st[st]
    ix = st.index
    gg = g.reindex(days).stack().reindex(ix).values
    cc = c.reindex(days).stack().reindex(ix).values
    df = pd.DataFrame({'date': ix.get_level_values(0), 'code': ix.get_level_values(1), 'g': gg, 'c': cc})
    df['net'] = df['g'] - df['c']
    return df


def cl_stats(df, col='net'):
    """每筆等權平均，日期 cluster t。"""
    if len(df) < 3: return {'n': int(len(df))}
    x = df[col].values; m = x.mean(); N = len(x)
    S = pd.Series(x - m).groupby(df['date'].values).sum().values
    G = len(S)
    v = (G / (G - 1)) * (S ** 2).sum() / N ** 2 if G > 1 else float('nan')
    se_naive = x.std(ddof=1) / math.sqrt(N)
    se = math.sqrt(v) if v > 0 else float('nan')
    return {'n_trades': int(N), 'n_days': int(G), 'mean_pct': round(m * 100, 3), 'median_pct': round(float(np.median(x)) * 100, 3),
            't_cluster': round(m / se, 2) if se == se else None, 't_naive': round(m / se_naive, 2),
            'se_cluster_pct': round(se * 100, 4),
            'eff_n_design': round(N * (se_naive / se) ** 2, 1) if se == se else None}


def day_stats(df, col='net'):
    """每日等權 cohort：先同日平均，再對日平均。"""
    if len(df) < 3: return {'n_days': int(df['date'].nunique())}
    d = df.groupby('date')[col].mean()
    if len(d) < 3: return {'n_days': int(len(d)), 'mean_pct': round(float(d.mean()) * 100, 3)}
    se = d.std(ddof=1) / math.sqrt(len(d))
    return {'n_days': int(len(d)), 'mean_pct': round(float(d.mean()) * 100, 3), 't': round(float(d.mean() / se), 2) if se > 0 else None,
            'se_pct': round(float(se) * 100, 4)}


def both(df, col='net'):
    return {'per_trade': cl_stats(df, col), 'per_day': day_stats(df, col)}


BUCKETS = [(1, 1, '1'), (2, 3, '2-3'), (4, 10, '4-10'), (11, 30, '11-30'), (31, 10 ** 9, '>30')]


def analyse(df):
    o = {'all': both(df), 'gross': both(df, 'g')}
    cnt = df.groupby('date').size()
    df = df.assign(ndays=df['date'].map(cnt))
    o['buckets'] = {}
    for lo, hi, lab in BUCKETS:
        sub = df[(df['ndays'] >= lo) & (df['ndays'] <= hi)]
        o['buckets'][lab] = both(sub)
        o['buckets'][lab]['share_of_trades'] = round(len(sub) / len(df), 3) if len(df) else None
    o['drop_top_days'] = {}
    order = cnt.sort_values(ascending=False, kind='mergesort')
    for k in [5, 10, 20]:
        drop = set(order.index[:k])
        sub = df[~df['date'].isin(drop)]
        o['drop_top_days'][str(k)] = both(sub)
        o['drop_top_days'][str(k)]['dropped_trades'] = int(df['date'].isin(drop).sum())
    top = order.index[:12]
    o['top_days'] = [{'date': str(d.date()), 'n': int(cnt[d]), 'mean_net_pct': round(float(df.loc[df['date'] == d, 'net'].mean()) * 100, 2)} for d in top]
    # 貢獻：前 10 天佔每筆等權總淨和的比例
    tot = df['net'].sum()
    o['top10_days_share_of_sum_net'] = round(float(df[df['date'].isin(set(order.index[:10]))]['net'].sum() / tot), 2) if tot != 0 else None
    return o


res = {'description': __doc__, 'rules': {}}
pr = []
for name, cond in rules.items():
    mask = (elig & cond).fillna(False).astype(bool)
    rr = {}
    for w in L.WINDOWS:
        dm = L.day_set(D, w); days = dm[dm].index
        dfb = events(mask, gdt, cdt, days)
        dfr = events(mask, g_reg, c_reg, days)
        rr[w] = {'base_cost': analyse(dfb), 'regime_cost': analyse(dfr)}
        # 對照：S0 隔日收盤全期、0.585% / 1%
        if w == 'S0_2005_2015':
            dfn = events(mask, g2, pd.DataFrame(COST_NC, index=g2.index, columns=g2.columns), days)
            rr[w]['S0_all_next_close_0.585'] = both(dfn)
            dfn1 = dfn.assign(net=dfn['g'] - 0.01)
            rr[w]['S0_all_next_close_1pct'] = both(dfn1)
        for cname in ['base_cost', 'regime_cost']:
            a = rr[w][cname]
            pr.append([name, w, cname, a['all']['per_day'].get('mean_pct'), a['all']['per_day'].get('t'),
                       a['all']['per_trade'].get('mean_pct'), a['all']['per_trade'].get('median_pct'), a['all']['per_trade'].get('t_cluster'),
                       a['all']['per_trade'].get('n_trades'), a['all']['per_trade'].get('n_days')]
                      + [a['drop_top_days'][k]['per_trade'].get('mean_pct') for k in ['5', '10', '20']]
                      + [a['drop_top_days'][k]['per_trade'].get('t_cluster') for k in ['5', '10', '20']])
    # V_all 內 src 3 期間單獨
    dm = L.day_set(D, 'V_all'); days = dm[dm].index
    s3 = days[(days >= pd.Timestamp('2024-02-15')) & (days <= pd.Timestamp('2025-09-23'))]
    rr['src3_2024-02_2025-09'] = {'base_cost': analyse(events(mask, gdt, cdt, s3))}
    # 斷點比較（毛報酬，保守當沖出場；pre-2014 另列 regime 淨）
    bp = {}
    alld = L.day_set(D, 'V_all'); alld = alld[alld].index   # 所有窗口共用（含 src3 以外均官方；斷點都在 2019 前）
    for lab, b, prev_b, next_b in [('2014-01-06_daytrade_open', DT_OPEN, pd.Timestamp('2005-01-01'), LIM10),
                                   ('2015-06-01_limit10', LIM10, DT_OPEN, TAX_CUT),
                                   ('2017-04-28_tax_cut', TAX_CUT, LIM10, pd.Timestamp('2019-05-01'))]:
        cal_all = CAL[D['all_day'].values]
        out = {}
        for wl, (a0, a1, b0, b1) in {'12m': (b - pd.DateOffset(months=12), b, b, b + pd.DateOffset(months=12)),
                                     'adjacent_segment': (prev_b, b, b, next_b)}.items():
            dA = cal_all[(cal_all >= a0) & (cal_all < a1)]; dB = cal_all[(cal_all >= b0) & (cal_all < b1)]
            eA = events(mask, gdt, cdt, dA); eB = events(mask, gdt, cdt, dB)
            rA = events(mask, g_reg, c_reg, dA); rB = events(mask, g_reg, c_reg, dB)
            sA, sB = cl_stats(eA, 'g'), cl_stats(eB, 'g')
            dA_, dB_ = day_stats(eA, 'g'), day_stats(eB, 'g')
            out[wl] = {'before': f'{a0.date()}~{(a1 - pd.Timedelta(days=1)).date()}', 'after': f'{b0.date()}~{(b1 - pd.Timedelta(days=1)).date()}',
                       'gross_per_trade_before': sA, 'gross_per_trade_after': sB,
                       'diff_per_trade_pct': round(sB['mean_pct'] - sA['mean_pct'], 3),
                       't_diff_per_trade': round((sB['mean_pct'] - sA['mean_pct']) / math.hypot(sA['se_cluster_pct'], sB['se_cluster_pct']), 2),
                       'gross_per_day_before': dA_, 'gross_per_day_after': dB_,
                       'diff_per_day_pct': round(dB_['mean_pct'] - dA_['mean_pct'], 3),
                       't_diff_per_day': round((dB_['mean_pct'] - dA_['mean_pct']) / math.hypot(dA_['se_pct'], dB_['se_pct']), 2),
                       'regime_net_before': both(rA), 'regime_net_after': both(rB)}
            # 只看事件數 ≤10 的日子（個股自身跳空）
            def small(e):
                n = e.groupby('date').size(); return e[e['date'].map(n) <= 10]
            qA, qB = small(eA), small(eB)
            a_, b_ = cl_stats(qA, 'g'), cl_stats(qB, 'g')
            out[wl]['small_days_le10_gross_per_trade'] = {'before': a_, 'after': b_, 'diff_pct': round(b_['mean_pct'] - a_['mean_pct'], 3),
                                                         't_diff': round((b_['mean_pct'] - a_['mean_pct']) / math.hypot(a_['se_cluster_pct'], b_['se_cluster_pct']), 2)}
        bp[lab] = out
    rr['breakpoints'] = bp
    res['rules'][name] = rr

res['runtime_sec'] = round(time.time() - t0, 1)
L.save('results_recheck_r1_0.json', res)

pd.set_option('display.width', 250)
cols = ['rule', 'win', 'cost', 'D_mean', 'D_t', 'T_mean', 'T_med', 'T_tcl', 'trades', 'days', 'T_d5', 'T_d10', 'T_d20', 't_d5', 't_d10', 't_d20']
print(pd.DataFrame(pr, columns=cols).to_string())
for name in rules:
    print('\n#####', name)
    for w in list(L.WINDOWS) + ['src3_2024-02_2025-09']:
        for cname in ['base_cost', 'regime_cost']:
            if cname not in res['rules'][name][w]: continue
            a = res['rules'][name][w][cname]
            print(f'-- {w} {cname} buckets (T mean / med / tcl / trades / days | D mean)')
            for lab, v in a['buckets'].items():
                pt, pdy = v['per_trade'], v['per_day']
                print(f"   {lab:>6}: {pt.get('mean_pct')} / {pt.get('median_pct')} / {pt.get('t_cluster')} / {pt.get('n_trades', pt.get('n'))} / {pt.get('n_days')} | {pdy.get('mean_pct')}  share {v['share_of_trades']}")
            if cname == 'base_cost':
                print('   top days:', a['top_days'][:10], 'top10 share of sum net', a['top10_days_share_of_sum_net'])
                print('   eff_n_design all:', a['all']['per_trade'].get('eff_n_design'))
    s0 = res['rules'][name]['S0_2005_2015']
    print('S0 next close 0.585:', s0['S0_all_next_close_0.585'], '\nS0 next close 1%:', s0['S0_all_next_close_1pct'])
    for lab, v in res['rules'][name]['breakpoints'].items():
        for wl, x in v.items():
            print(f"BP {lab} {wl}: before {x['before']} T {x['gross_per_trade_before']['mean_pct']} D {x['gross_per_day_before']['mean_pct']} | after {x['after']} T {x['gross_per_trade_after']['mean_pct']} D {x['gross_per_day_after']['mean_pct']} | diffT {x['diff_per_trade_pct']} t {x['t_diff_per_trade']} diffD {x['diff_per_day_pct']} t {x['t_diff_per_day']} | regime net T {x['regime_net_before']['per_trade'].get('mean_pct')}->{x['regime_net_after']['per_trade'].get('mean_pct')} D {x['regime_net_before']['per_day'].get('mean_pct')}->{x['regime_net_after']['per_day'].get('mean_pct')}")
for name in rules:
    for lab, v in res['rules'][name]['breakpoints'].items():
        for wl, x in v.items():
            q = x['small_days_le10_gross_per_trade']
            print(f"SMALL<=10 {name} {lab} {wl}: before {q['before']['mean_pct']} (n {q['before']['n_trades']}) after {q['after']['mean_pct']} (n {q['after']['n_trades']}) diff {q['diff_pct']} t {q['t_diff']}")
print('runtime', res['runtime_sec'])
