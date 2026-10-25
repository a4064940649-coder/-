#!/usr/bin/env python3
"""新股／現增 公開申購抽籤 期望值分析（資料：mjib007/company-law-cases 彙整之證交所公開申購紀錄 2020–2026-09，
價格為 FinMind 之「申購結束日市價」，非撥券日價格 —— 這是本分析最大的近似）。
用法：python3 ipo_lottery_ev.py --csv <path> --out <dir>
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

def roc_to_date(s):
    s = str(s).strip()
    if not s or s == 'nan': return pd.NaT
    y, m, d = s.split('/')
    return pd.Timestamp(int(y) + 1911, int(m), int(d))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--csv', required=True); ap.add_argument('--out', default='./output')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.csv, encoding='utf-8-sig')
    df['抽籤日'] = df['抽籤日期'].map(roc_to_date)
    df['year'] = df['抽籤日'].dt.year
    for c in ['承銷價_num', '申購股數_num', '每張價差', '價差報酬率', '中籤率(%)', '申購結束日市價', '總合格件']:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', ''), errors='coerce')
    # 只保留有價格與承銷價的有效紀錄，且未取消
    cancelled = df['取消公開抽籤'].astype(str).str.strip().isin(['Y', 'y', '是', 'True', '1'])
    valid = df[~cancelled & df['承銷價_num'].notna() & df['申購結束日市價'].notna() & (df['申購股數_num'] > 0)].copy()
    valid['lot_cost'] = valid['承銷價_num'] * valid['申購股數_num']          # 一申購單位成本（通常 1,000 股）
    valid['lot_gain'] = (valid['申購結束日市價'] - valid['承銷價_num']) * valid['申購股數_num']
    valid['ret'] = valid['lot_gain'] / valid['lot_cost']
    valid['p_win'] = valid['中籤率(%)'] / 100.0
    valid['ev_per_app'] = valid['p_win'] * valid['lot_gain'] - 20            # 每件申購期望值（含 NT$20 處理費）
    valid['ev_gross'] = valid['p_win'] * valid['lot_gain']
    summary = {}
    g = valid.groupby('year')
    tbl = pd.DataFrame({
        'n_lotteries': g.size(),
        'n_markets': g['發行市場'].nunique(),
        'avg_win_rate_pct': g['中籤率(%)'].mean().round(2),
        'median_win_rate_pct': g['中籤率(%)'].median().round(2),
        'share_profitable_pct': (g['ret'].apply(lambda s: (s > 0).mean()) * 100).round(1),
        'avg_ret_pct': (g['ret'].mean() * 100).round(1),
        'median_ret_pct': (g['ret'].median() * 100).round(1),
        'avg_lot_gain_NT': g['lot_gain'].mean().round(0),
        'median_lot_gain_NT': g['lot_gain'].median().round(0),
        'sum_ev_per_id_NT': g['ev_per_app'].sum().round(0),          # 全部都申購時，一個身分證一年的期望值
        'sum_fees_NT': (g.size() * 20),
        'exp_wins_per_id': g['p_win'].sum().round(2),
        'avg_lot_cost_NT': g['lot_cost'].mean().round(0),
        'max_lot_cost_NT': g['lot_cost'].max().round(0),
    })
    tbl.to_csv(out / 'ipo_lottery_by_year.csv', encoding='utf-8-sig')
    bymkt = valid.groupby('發行市場').agg(n=('ret', 'size'), avg_ret_pct=('ret', lambda s: round(s.mean() * 100, 1)),
                                        share_profitable_pct=('ret', lambda s: round((s > 0).mean() * 100, 1)),
                                        avg_win_rate_pct=('中籤率(%)', 'mean'), ev_per_app_NT=('ev_per_app', 'mean'))
    bymkt.to_csv(out / 'ipo_lottery_by_market.csv', encoding='utf-8-sig')
    # 分配：報酬率分位數、尾部
    q = valid['ret'].quantile([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]).round(3)
    # 模擬：一個人每年全部申購、中籤即以申購結束日市價賣出 → 年損益分配（依中籤率 Bernoulli 抽樣）
    rng = np.random.default_rng(0); sims = {}
    for y, grp in g:
        p = grp['p_win'].values; gain = grp['lot_gain'].values
        draws = rng.random((5000, len(p))) < p
        pnl = (draws * gain).sum(axis=1) - 20 * len(p)
        sims[int(y)] = {'mean': float(pnl.mean()), 'p10': float(np.percentile(pnl, 10)), 'median': float(np.median(pnl)),
                        'p90': float(np.percentile(pnl, 90)), 'prob_negative': float((pnl < 0).mean()), 'prob_zero_wins': float((draws.sum(axis=1) == 0).mean())}
    res = {'rows_total': int(len(df)), 'rows_valid': int(len(valid)), 'cancelled_rows': int(cancelled.sum()),
           'date_range': [str(valid['抽籤日'].min().date()), str(valid['抽籤日'].max().date())],
           'overall': {'share_profitable_pct': round(float((valid['ret'] > 0).mean() * 100), 1), 'avg_ret_pct': round(float(valid['ret'].mean() * 100), 1),
                       'median_ret_pct': round(float(valid['ret'].median() * 100), 1), 'avg_win_rate_pct': round(float(valid['中籤率(%)'].mean()), 2),
                       'avg_ev_per_app_NT': round(float(valid['ev_per_app'].mean()), 0), 'ret_quantiles': q.to_dict()},
           'by_year': tbl.reset_index().to_dict(orient='records'), 'by_market': bymkt.reset_index().to_dict(orient='records'), 'sim_annual_pnl': sims,
           'caveats': ['價格為申購結束日市價（FinMind），非撥券日可賣出價格；IPO 興櫃價與掛牌後價格可能不同', '未計入資金凍結成本與郵寄費 NT$50（中籤才收）',
                       '2026 年資料僅至 9 月', '承銷價「未訂出」或無市價之紀錄已排除']}
    (out / 'ipo_lottery_ev.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    print(json.dumps({k: res[k] for k in ['rows_total', 'rows_valid', 'date_range', 'overall']}, ensure_ascii=False, indent=1))
    print(tbl.to_string()); print(bymkt.to_string()); print(json.dumps(sims, indent=1))

if __name__ == '__main__':
    main()
