#!/usr/bin/env python3
"""主策略（52 週高點 × 月營收 3 個月年增率）M2 可成交版的真正樣本外：2006-04～2020-12（長期月營收）。輸出 results.json、oos_nav.parquet。"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lhlib as L

OUT = L.HERE
PREREG = {
    '寫於執行前': True,
    '規則': 'bt_final_rule.py／exlib M2，參數不改：前 300 名流動性池（60 日均值 ≥1,000 萬、股價 ≥10）、0.5×52 週高點百分位＋0.5×近 3 月營收年增率平均百分位、前 30／10 檔等權、每月第一個 ≥11 日交易日收盤訊號、隔日開盤成交、開盤漲停且未持有者跳過遞補',
    '營收': 'NiklavsD/bottleneck-signals data/processed/tw_revenue.csv（MOPS t21sc03 每月彙總頁，國內公司 _0 頁），年增率由營收水準自算（同月比去年同月）',
    '成本': 'exlib：買 0.0855%+滑價、賣 0.0855%+稅 0.3%+滑價；滑價 0.1%／0.3%；壓力 買 0.5%＋賣 0.5%',
    '執行面調整': '漲跌幅 2015-06-01 前 7%、後 10%（開盤漲停判定 = 還原開盤／前收 −1 ≥ 上限 −0.6%）；另報 exlib 原樣 10% 為診斷',
    '面板修補': '2018-01-13 只有 46 檔有值（面板缺漏），計算 52 週高點與 60 日均值時把該日視為不存在；另報未修補版為診斷',
    '處置濾網': 'disposal_5y 只有 2021-09 起，2005～2020 無法套用（等於不濾）',
    '期間': {'OOS_full': '2006-04～2020-12', 'A_clean': '2006-04～2013-01 訊號（營收無存活者偏誤、個體營收口徑一致）',
             'IFRS_break': '2013-02～2014-03 訊號（近 3 月年增率含 2013 合併 vs 2012 個體，營收腿失真）',
             'B_survivor': '2014-04～2020-12 訊號（合併口徑一致，但 2013 起營收只剩目前仍上市公司；2019–2020 價格面板也只有存活者）'},
    '主要判定': 'OOS_full 與 A_clean 的 n30 滑價 0.1%：對同池等權 t ≥ 2 → 樣本外支持選股能力；對 0050 t ≥ 2 → 樣本外勝過 0050',
    '診斷（不影響判定）': ['全期 10% 漲停判定', '價格腿單獨（w52=1）', '營收腿單獨（w52=0）', 'rev_5y 原資料 2021 起重現 exec_rebuild M2', '缺營收股票的同池等權報酬（存活者偏誤方向）'],
}

def seg_bounds(dates):
    D = pd.DatetimeIndex(dates)
    def first_on_or_after(s): return D[D >= pd.Timestamp(s)][0]
    a0 = D[0]; d13 = first_on_or_after('2013-02-01'); d14 = first_on_or_after('2014-04-01')
    one = pd.Timedelta(days=1)
    return {
        'OOS_full_2006-04_2020-12': [(a0, '2020-12-31')],
        'A_clean_2006-04_2013-01sig': [(a0, d13 - one)],
        'IFRS_break_2013-02_2014-03sig': [(d13, d14 - one)],
        'B_survivor_2014-04_2020-12': [(d14, '2020-12-31')],
        'OOS_ex_IFRS': [(a0, d13 - one), (d14, '2020-12-31')],
        'bear_2008': [('2008-01-01', '2008-12-31')], 'bear_2011': [('2011-01-01', '2011-12-31')],
        'bear_2015': [('2015-01-01', '2015-12-31')], 'bear_2018': [('2018-01-01', '2018-12-31')],
        'year_2020': [('2020-01-01', '2020-12-31')],
        'overlap_2021-03_2026-10': [('2021-03-01', '2026-10-31')],
        'all_2006-04_2026-10': [(a0, '2026-10-31')],
    }

def main():
    res = {'PREREG': PREREG}
    (OUT / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')  # 先落地事前設定
    S = L.Setup(L.yoy_long(), start='2005-01-01')
    W = seg_bounds(S.dates)
    t0 = L.pos(S.dates[0]); t1 = L.pos(S.dates[-1])
    res['dates'] = {'first_signal': str(S.dates[0].date()), 'last_signal': str(S.dates[-1].date()), 'n_rebalances': len(S.dates) - 1}
    navs = {}
    def summarize(nav, ew):
        return {k: L.daily_stats(nav, s, S.DATES, L.E50, ew) for k, s in W.items()}
    # 主結果
    for n in (30, 10):
        ev, pools = S.selection(n)
        nb = S.nblk
        ew, _ = L.sim_portfolio(pools, 0.0, 0.0, t0, t1); navs['EW_pool'] = ew
        res[f'pool_size_by_year'] = {int(y): round(float(np.mean([len(pools[k][1]) for k in pools if L.IDX[k].year == y])), 1) for y in sorted({L.IDX[k].year for k in pools})}
        for slip in (0.001, 0.003, 'stress'):
            b, s = L.cost_rates(slip)
            nav, tov = L.sim_portfolio(ev, b, s, t0, t1)
            key = f'M2_n{n}_slip{slip}'; navs[key] = nav
            res[key] = {'turnover_value_per_rebalance': round(tov, 3), 'skipped_limit_up_total': int(nb), 'rebalances': len(ev),
                        'mean_names': round(float(np.mean([len(v[1]) for v in ev.values()])), 1), 'windows': summarize(nav, ew)}
            w = res[key]['windows']
            print(key, {k: (v['CAGR'], v['maxDD'], v.get('t_vs_0050_m'), v.get('t_vs_EW_m')) for k, v in w.items() if v}, flush=True)
        g, _ = L.sim_portfolio(ev, 0.0, 0.0, t0, t1); navs[f'M2_n{n}_gross'] = g
        res[f'M2_n{n}_gross'] = {'windows': {k: v for k, v in summarize(g, ew).items() if k in ('OOS_full_2006-04_2020-12', 'A_clean_2006-04_2013-01sig', 'B_survivor_2014-04_2020-12')}}
    res['0050_windows'] = {k: L.daily_stats(L.E50.iloc[t0:t1 + 1], s, S.DATES) for k, s in W.items()}
    # ---- 診斷（n30、滑價 0.1%）----
    b, s = L.cost_rates(0.001)
    diag = {}
    keepw = ('OOS_full_2006-04_2020-12', 'A_clean_2006-04_2013-01sig', 'IFRS_break_2013-02_2014-03sig', 'B_survivor_2014-04_2020-12', 'OOS_ex_IFRS', 'overlap_2021-03_2026-10')
    def short(w): return {k: {kk: w[k][kk] for kk in ('CAGR', 'maxDD', 't_vs_0050_m', 't_vs_EW_m', 'excess_vs_EW_pct_m')} for k in keepw if w.get(k)}
    ew = navs['EW_pool']
    for name, kw in {'gap10_everywhere': dict(gap='10'), 'price_leg_only_w52=1': dict(w52=1.0), 'revenue_leg_only_w52=0': dict(w52=0.0)}.items():
        ev, _ = S.selection(30, **kw)
        nav, tov = L.sim_portfolio(ev, b, s, t0, t1); navs[f'DIAG_{name}'] = nav
        diag[name] = {'turnover': round(tov, 3), 'skipped_limit_up_total': int(S.nblk), 'windows': short(summarize(nav, ew))}
        diag[name]['yearly_pct'] = summarize(nav, ew)['OOS_full_2006-04_2020-12']['yearly_pct']
        print('DIAG', name, diag[name]['windows'].get('OOS_full_2006-04_2020-12'), flush=True)
    # 未修補面板（exlib 原式滾動值）
    Sr = L.Setup(L.yoy_long(), start='2005-01-01', raw_panel=True)
    evr, poolr = Sr.selection(30)
    navr, tovr = L.sim_portfolio(evr, b, s, t0, t1); ewr, _ = L.sim_portfolio(poolr, 0, 0, t0, t1)
    wr = {k: L.daily_stats(navr, sg, Sr.DATES, L.E50, ewr) for k, sg in W.items()}
    diag['raw_panel_no_fix'] = {'rebalances': len(evr), 'bad_days': L.BAD_DAYS, 'windows': short(wr), 'yearly_pct': wr['OOS_full_2006-04_2020-12']['yearly_pct']}
    print('DIAG raw_panel', diag['raw_panel_no_fix']['windows'].get('OOS_full_2006-04_2020-12'), len(evr), flush=True)
    # 缺營收股票（池內但 yoy3 缺）的同池等權：存活者偏誤方向
    el_all = S.t300 & S.s52.notna() & ~S.inD
    ev_m, ev_h = {}, {}
    nmiss = {}
    for d in S.dates[:-1]:
        ex = L.pos(d) + 1
        trad = np.isfinite(L.OPEN_RAW[ex]) & np.isfinite(L.ADJ_RAW[ex])
        a = el_all.loc[d].to_numpy() & trad; hv = S.yoy3.loc[d].notna().to_numpy()
        if (a & hv).sum() >= 50: ev_h[ex] = ('open', np.where(a & hv)[0])
        if (a & ~hv).sum() >= 1: ev_m[ex] = ('open', np.where(a & ~hv)[0])
        nmiss.setdefault(d.year, []).append(int((a & ~hv).sum()))
    navm, _ = L.sim_portfolio(ev_m, 0, 0, t0, t1); navh, _ = L.sim_portfolio(ev_h, 0, 0, t0, t1)
    ym = L.daily_stats(navm, [(S.dates[0], '2026-10-31')], S.DATES)['yearly_pct']; yh = L.daily_stats(navh, [(S.dates[0], '2026-10-31')], S.DATES)['yearly_pct']
    diag['pool_names_missing_revenue'] = {'mean_count_by_year': {y: round(float(np.mean(v)), 1) for y, v in nmiss.items()},
                                          'EW_missing_yearly_pct': ym, 'EW_with_revenue_yearly_pct': yh}
    # rev_5y 原資料 2021 起重現 exec_rebuild（應 ≈ 47.3%）
    S5 = L.Setup(L.yoy_rev5y(), start='2021-01-01')
    ev5, pools5 = S5.selection(30)
    a5 = L.pos(S5.dates[0]); b5 = L.pos(S5.dates[-1])
    nav5, _ = L.sim_portfolio(ev5, b, s, a5, b5); ew5, _ = L.sim_portfolio(pools5, 0, 0, a5, b5)
    w5 = L.daily_stats(nav5, [('2021-03-01', '2026-10-31')], S5.DATES, L.E50, ew5)
    diag['repro_exec_rebuild_M2_n30_rev5y'] = {k: w5[k] for k in ('CAGR', 'maxDD', 't_vs_0050_m', 't_vs_EW_m')}
    # 同期間用長資料（overlap）
    diag['overlap_longdata_M2_n30'] = {k: res['M2_n30_slip0.001']['windows']['overlap_2021-03_2026-10'][k] for k in ('CAGR', 'maxDD', 't_vs_0050_m', 't_vs_EW_m')}
    # 持股重疊：長資料 vs rev_5y，2021-03 之後同日
    pk_l, _ = S.selection(30); com = [k for k in ev5 if k in pk_l]
    diag['pick_overlap_long_vs_rev5y_2021plus'] = round(float(np.mean([len(set(ev5[k][1]) & set(pk_l[k][1])) / 30 for k in com])), 3) if com else None
    res['DIAG'] = diag
    vf = OUT / 'verify.json'
    if vf.exists(): res['DATA_VERIFY'] = json.loads(vf.read_text(encoding='utf-8'))   # 由 verify_data.py 產生
    (OUT / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    pd.DataFrame(navs).to_parquet(OUT / 'oos_nav.parquet')
    print(json.dumps(diag, ensure_ascii=False, default=str)[:4000])

if __name__ == '__main__':
    main()
