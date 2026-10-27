#!/usr/bin/env python3
"""main_refine 診斷：長期營收（tw_revenue.csv）與 rev_5y 造成 n10 在 2024～2026 差距的原因。同一引擎（lhlib.Setup.selection(10)＋sim_portfolio、滑價 0.1%）。
寫入 results.json 的 DIAG_REV_SOURCE（不參與判定）。"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.dont_write_bytecode = True
import rlib as R
L = R.L
X = L.X
B, SL = L.cost_rates(0.001)

def run(YOY, start):
    S = L.Setup(YOY, start=start); ev, _ = S.selection(10)
    nav, _ = L.sim_portfolio(ev, B, SL, L.pos(S.dates[0]), L.pos(S.dates[-1]))
    return S, ev, nav

def cagr(n, a, b):
    n = n[(n.index >= pd.Timestamp(a)) & (n.index <= pd.Timestamp(b))]
    # 以 a 之前最後一日為起點
    return round(float((n.iloc[-1] / n.iloc[0]) ** (365.25 / (n.index[-1] - n.index[0]).days) - 1), 4)

def main():
    res = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
    out = {}
    # ---------- 原始資料 ----------
    lg = pd.read_csv(L.LONG, dtype={'code': str, 'market': str, 'name': str}); lg['ym'] = lg['month'].str[:7]
    lg = lg.sort_values(['ym', 'market']).drop_duplicates(['ym', 'code'], keep='first')
    r5 = pd.read_csv(X / 'rev_5y.csv.gz', dtype={'code': str}); rh = pd.read_csv(X / 'rev_hist.csv.gz', dtype={'code': str})
    sl = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv', dtype={'code': str})
    FIN = set(sl.loc[sl['industry'] == '金融保險業', 'code']) | {c for c in R.COLS if c[:2] == '28' or c in ('5876', '5880')}
    # ---------- 1. 重現 ----------
    Yl = L.yoy_long(); Y5 = L.yoy_rev5y()
    S5, e5, n5 = run(Y5, '2021-01-01'); Sl, el, nl = run(Yl, '2005-01-01')
    mn = pd.read_parquet(HERE.parent / 'exec_rebuild' / 'main_nav.parquet')['M2_n10_slip0.001']
    x = n5.reindex(mn.index)
    out['repro_exec_rebuild_main_nav_n10_maxdiff'] = float((x / x.iloc[0] / (mn / mn.iloc[0]) - 1).abs().max())
    END = str(S5.dates[-1].date())
    out['CAGR_same_engine'] = {'rev5y': {'2021-03~': cagr(n5, '2021-03-01', END), '2024-01~': cagr(n5, '2024-01-01', END)},
                               'long': {'2021-03~': cagr(nl, '2021-03-01', END), '2024-01~': cagr(nl, '2024-01-01', END)}, 'end': END}
    # ---------- 2. 逐期持股差異（2024 起） ----------
    exs = sorted(set(e5) & set(el)); rows = []; nper = 0; ndiff = 0
    tend = L.pos(S5.dates[-1])
    for q, ex in enumerate(exs):
        if L.IDX[ex] < pd.Timestamp('2024-01-01'): continue
        nper += 1
        nx = exs[q + 1] if q + 1 < len(exs) else tend
        p5 = set(e5[ex][1].tolist()); pL = set(el[ex][1].tolist()); d = L.IDX[ex - 1]
        if p5 != pL: ndiff += 1
        for j in p5 ^ pL:
            c = R.COLS[j]
            fwd = (L.AO[nx, j] if nx != tend else L.AC[nx, j]) / L.AO[ex, j] - 1
            y5, yl = S5.yoy3.loc[d, c], Sl.yoy3.loc[d, c]
            direct = (np.isnan(y5) != np.isnan(yl)) or (np.isfinite(y5) and np.isfinite(yl) and abs(y5 - yl) > 1)
            rows.append({'signal': str(d.date()), 'code': c, 'side': 'rev5y_only' if j in p5 else 'long_only', 'fwd_ret': round(float(fwd), 4),
                         'yoy3_rev5y': None if np.isnan(y5) else round(float(y5), 2), 'yoy3_long': None if np.isnan(yl) else round(float(yl), 2),
                         'financial': c in FIN, 'cause': '營收年增率不同' if direct else '被排擠（排名邊界）',
                         'contrib_to_gap': round(float((1 if j in p5 else -1) * fwd / 10), 4)})
    df = pd.DataFrame(rows)
    out['pick_diff_2024plus'] = {'n_rebalances': nper, 'n_rebalances_with_diff': ndiff, 'n_name_diffs': int(len(df)),
        'sum_contrib_simple': round(float(df['contrib_to_gap'].sum()), 4),
        'contrib_by_cause_side': {f'{a}|{b}': round(float(v), 4) for (a, b), v in df.groupby(['cause', 'side'])['contrib_to_gap'].sum().items()},
        'financial_names_rev5y_only': sorted(df.loc[(df.side == 'rev5y_only') & df.financial, 'code'].unique().tolist()),
        'n_rev5y_only_financial': int(((df.side == 'rev5y_only') & df.financial).sum()), 'n_rev5y_only': int((df.side == 'rev5y_only').sum()),
        'rows': df.to_dict('records')}
    print(json.dumps({k: v for k, v in out['pick_diff_2024plus'].items() if k != 'rows'}, ensure_ascii=False))
    # ---------- 3. 年增率差異的成因（2021-01～2026-08 全部重疊格） ----------
    a = r5[['code', 'ym', 'revenue', 'yoy']].merge(lg[['code', 'ym', 'revenue_k_twd']], on=['code', 'ym'], how='outer')
    a = a[a['code'].isin(R.COLS)]
    lg2 = lg[['code', 'ym', 'revenue_k_twd']].copy(); lg2['ym'] = (pd.PeriodIndex(lg2['ym'], freq='M') + 12).astype(str)
    a = a.merge(lg2.rename(columns={'revenue_k_twd': 'long_prev'}), on=['code', 'ym'], how='left')
    a = a[(a['ym'] >= '2021-01') & (a['ym'] <= '2026-08')]
    a['yoy_self'] = np.where(a['long_prev'] > 0, (a['revenue_k_twd'] / a['long_prev'] - 1) * 100, np.nan)
    a['yoy_mopsform'] = np.where(a['long_prev'].abs() > 0, (a['revenue_k_twd'] - a['long_prev']) / a['long_prev'].abs() * 100, np.nan)
    a['implied_prev'] = a['revenue'] / (1 + a['yoy'] / 100)
    both = a[a['revenue'].notna() & a['revenue_k_twd'].notna()]
    lvl_eq = (both['revenue'] - both['revenue_k_twd']).abs() <= 0.0001 * both['revenue'].abs().clip(lower=1)
    bad = both[both['yoy'].notna() & ~((both['yoy'] - both['yoy_self']).abs() <= 1)]
    def cat(r):
        if pd.isna(r['long_prev']): return '去年同月長資料缺'
        if r['long_prev'] <= 0: return '去年同月營收 ≤0（MOPS 用 |去年| 當分母，長資料自算為空）'
        if abs(r['revenue'] - r['revenue_k_twd']) > 0.0001 * max(abs(r['revenue']), 1): return '當月營收數值不同'
        if abs(r['implied_prev'] - r['long_prev']) > 0.01 * abs(r['long_prev']): return 'MOPS 去年同月數字被重編（與當年原申報不同）'
        return '其他（四捨五入）'
    bad = bad.assign(cat=bad.apply(cat, axis=1))
    out['yoy_mismatch_2021_2026'] = {'cells_both_present': int(len(both)), 'level_equal_share': round(float(lvl_eq.mean()), 5),
        'yoy_diff_gt1pp_cells': int(len(bad)), 'share': round(float(len(bad) / max(1, both['yoy'].notna().sum())), 5),
        'by_cause': bad['cat'].value_counts().to_dict(),
        'by_cause_financial': bad[bad['code'].isin(FIN)]['cat'].value_counts().to_dict(),
        'restated_by_year': bad[bad['cat'].str.startswith('MOPS 去年')].assign(y=lambda z: z['ym'].str[:4])['y'].value_counts().sort_index().to_dict(),
        'restated_top_codes': bad[bad['cat'].str.startswith('MOPS 去年')]['code'].value_counts().head(15).to_dict(),
        'only_in_rev5y': int((a['revenue'].notna() & a['revenue_k_twd'].isna()).sum()), 'only_in_long': int((a['revenue'].isna() & a['revenue_k_twd'].notna()).sum()),
        'only_in_rev5y_codes': a.loc[a['revenue'].notna() & a['revenue_k_twd'].isna(), 'code'].value_counts().head(10).to_dict(),
        'only_in_long_codes': a.loc[a['revenue'].isna() & a['revenue_k_twd'].notna(), 'code'].value_counts().head(10).to_dict()}
    print(json.dumps(out['yoy_mismatch_2021_2026'], ensure_ascii=False))
    # ---------- 4. 反事實（同引擎） ----------
    Lv = R.rev_level_long(); prev = Lv.shift(12)
    Y_mops = ((Lv - prev) / prev.abs() * 100).where(prev.abs() > 0)
    r5c = r5[r5['code'].isin(R.COLS)].copy(); r5c['p'] = pd.PeriodIndex(r5c['ym'], freq='M')
    Lv5 = r5c.pivot_table(index='p', columns='code', values='revenue', aggfunc='last').sort_index()
    Lv5 = Lv5.reindex(pd.period_range(Lv5.index.min(), Lv5.index.max(), freq='M')); p5 = Lv5.shift(12)
    Y5_self = ((Lv5 / p5 - 1) * 100).where(p5 > 0)
    def exfin(Y): Y = Y.copy(); Y[[c for c in Y.columns if c in FIN]] = np.nan; return Y
    cf = {'long_MOPSformula': (Y_mops, '2005-01-01'), 'rev5y_selfcomputed': (Y5_self, '2021-01-01'),
          'long_exFIN': (exfin(Yl), '2005-01-01'), 'rev5y_exFIN': (exfin(Y5), '2021-01-01'),
          'long_MOPSformula_exFIN': (exfin(Y_mops), '2005-01-01')}
    out['counterfactual_CAGR'] = {}
    base_full = {'2006-04~2013-12': cagr(nl, '2006-04-01', '2013-12-31'), '2014~2020': cagr(nl, '2013-12-31', '2020-12-31')}
    out['counterfactual_CAGR']['long（原版）'] = dict(out['CAGR_same_engine']['long'], **base_full)
    out['counterfactual_CAGR']['rev5y（原版）'] = out['CAGR_same_engine']['rev5y']
    for k, (Y, st) in cf.items():
        S, ev, nav = run(Y, st)
        o = {'2021-03~': cagr(nav, '2021-03-01', END), '2024-01~': cagr(nav, '2024-01-01', END)}
        if st < '2010':
            o['2006-04~2013-12'] = cagr(nav, '2006-04-01', '2013-12-31'); o['2014~2020'] = cagr(nav, '2013-12-31', '2020-12-31')
            com = [e for e in ev if e in el and L.IDX[e] < pd.Timestamp('2021-01-01')]
            o['pick_overlap_with_long_2006_2020'] = round(float(np.mean([len(set(ev[e][1]) & set(el[e][1])) / 10 for e in com])), 3)
        com = [e for e in ev if e in e5 and L.IDX[e] >= pd.Timestamp('2024-01-01')]
        o['pick_overlap_with_rev5y_2024plus'] = round(float(np.mean([len(set(ev[e][1]) & set(e5[e][1])) / 10 for e in com])), 3) if com else None
        out['counterfactual_CAGR'][k] = o; print(k, o, flush=True)
    # 財務股在 2006～2020 長資料（MOPS 公式）被選入的次數
    # ---------- 5. rev_hist 比對 ----------
    h = rh[rh['code'].isin(R.COLS)].copy()
    m = h.merge(r5[['code', 'ym', 'revenue', 'yoy']], on=['code', 'ym'], how='left', suffixes=('', '_5y')).merge(lg[['code', 'ym', 'revenue_k_twd']], on=['code', 'ym'], how='left')
    def eqs(x, y): z = m[x].notna() & m[y].notna(); return round(float(((m.loc[z, x] - m.loc[z, y]).abs() <= 0.0001 * m.loc[z, x].abs().clip(lower=1)).mean()), 5), int(z.sum())
    out['rev_hist_check'] = {'months': f"{h['ym'].min()}~{h['ym'].max()}", 'revenue_equal_rev5y': eqs('revenue', 'revenue_5y'),
                             'revenue_equal_long': eqs('revenue', 'revenue_k_twd'), 'yoy_equal_rev5y': eqs('yoy', 'yoy_5y'),
                             'ann_date_nonnull_months': sorted(h.loc[h['ann_date'].notna(), 'ym'].unique().tolist())}
    ad = pd.to_datetime(h.loc[h['ann_date'].notna(), 'ann_date']); sig = Sl.dates[-1]
    out['rev_hist_check']['2026-09_ann_date_dist'] = ad.dt.strftime('%Y-%m-%d').value_counts().sort_index().to_dict()
    out['rev_hist_check']['signal_date_2026-10'] = str(sig.date())
    late = set(h.loc[h['ann_date'].notna() & (pd.to_datetime(h['ann_date']) > sig), 'code'])
    out['rev_hist_check']['n_announced_after_signal_in_panel'] = len(late)
    # 2026-10-12 長資料的訊號（holdout 引擎會在 10-13 開盤成交）
    F = R.Frames(Sl.dates, Yl); pl = F.plan(F.yoy3); last = [p for p in pl if L.IDX[p[0] - 1] == sig]
    if last:
        ex, cand = last[0]
        picks = [R.COLS[j] for j in cand[:12]]
        out['rev_hist_check']['top12_at_2026-10-12'] = {c: (h.loc[(h.code == c) & (h.ym == '2026-09'), 'ann_date'].tolist() or [None])[0] for c in picks}
        el9 = F.t300.loc[sig] & F.s52.loc[sig].notna() & ~F.inD.loc[sig]
        pool = [c for c in R.COLS[el9.to_numpy()]]
        out['rev_hist_check']['pool_announced_after_signal'] = sorted(set(pool) & late)
    print(json.dumps(out['rev_hist_check'], ensure_ascii=False))
    # ---------- 6. 事後（POSTHOC，非事前、不可採用）：金控公告晚於訊號日的影響 ----------
    FHC = {'2880', '2881', '2882', '2883', '2884', '2885', '2886', '2887', '2888', '2889', '2890', '2891', '2892', '5880'}
    out['FHC_ann_date_2026-09'] = {c: (h.loc[(h.code == c) & (h.ym == '2026-09'), 'ann_date'].tolist() or [None])[0] for c in sorted(FHC)}
    def nfin(ev, a, b, S=FIN):
        ks = [e for e in ev if pd.Timestamp(a) <= L.IDX[e] <= pd.Timestamp(b)]
        return {'rebalances': len(ks), 'fin_name_slots': int(sum(sum(R.COLS[j] in S for j in ev[e][1]) for e in ks)),
                'fhc_name_slots': int(sum(sum(R.COLS[j] in FHC for j in ev[e][1]) for e in ks))}
    out['POSTHOC_fin_picks'] = {'long_2006_2020': nfin(el, '2006-01-01', '2020-12-31'), 'long_2021_2026': nfin(el, '2021-01-01', '2026-12-31'),
                                'rev5y_2021_2026': nfin(e5, '2021-01-01', '2026-12-31'), 'long_2024_2026': nfin(el, '2024-01-01', '2026-12-31'),
                                'rev5y_2024_2026': nfin(e5, '2024-01-01', '2026-12-31')}
    def exset(Y, S): Y = Y.copy(); Y[[c for c in Y.columns if c in S]] = np.nan; return Y
    Sb = L.Setup(Yl); Fb = R.Frames(Sb.dates, Yl); T0 = Fb.plan(Fb.yoy3)[0][0]; T1 = len(R.IDX) - 1
    nbase, _ = R.simulate(Fb.plan(Fb.yoy3), B, SL, T0, T1)
    SEG = {'SEL': ('2006-04-01', '2013-12-31'), 'VAL': ('2014-01-01', '2020-12-31'), 'HOLD': ('2021-03-01', '2026-10-31')}
    ph = {}
    for k, S_ in (('exFIN_all_financials', FIN), ('exFHC_financial_holding_only', FHC)):
        Fx = R.Frames(Sb.dates, exset(Yl, S_)); nx, _ = R.simulate(Fx.plan(Fx.yoy3), B, SL, T0, T1)
        ph[k] = {sk: {'stats': R.seg_stats(nx, a_, b_), 'paired_vs_BASE': R.paired(nx, nbase, a_, b_)} for sk, (a_, b_) in SEG.items()}
        print('POSTHOC', k, {sk: (v['stats']['CAGR'], v['paired_vs_BASE']['diff_pct_m'], v['paired_vs_BASE']['t_paired_m']) for sk, v in ph[k].items()}, flush=True)
    out['POSTHOC_exclude_financials_longdata'] = ph
    print(json.dumps(out['POSTHOC_fin_picks']), out['FHC_ann_date_2026-09'])
    res['DIAG_REV_SOURCE'] = out
    (HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')

if __name__ == '__main__':
    main()
