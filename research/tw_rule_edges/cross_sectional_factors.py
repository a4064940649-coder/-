#!/usr/bin/env python3
"""台股全市場橫斷面因子檢定（資料：Asoul/tsec 證交所／櫃買每日行情 1994–2019-05，未還原權息）。
每月最後交易日依訊號分五組，等權持有一個月；只看「做多最佳組 vs 全市場等權」，含成本（來回 0.5% × 周轉）。
因子：1 月反轉、12-1 動能、60 日波動（低波動）、MAX（過去 20 日最大單日漲幅，低者佳）、成交值（流動性）、52 週高點距離、價格。
注意：未還原除權息 → 月報酬略被低估（對所有組一致），高股息股較吃虧；小型股成本未加碼。
用法：python3 cross_sectional_factors.py --dir <tsec dir> --out ./output [--start 2005-01-01] [--min-value 10000000]
"""
import argparse, json, math, glob, os
from pathlib import Path
import numpy as np, pandas as pd

def load_one(p):
    df = pd.read_csv(p, header=None, names=['d', 'vol', 'val', 'o', 'h', 'l', 'c', 'chg', 'n'], dtype=str)
    df = df[~df['o'].isin(['---', '--'])]
    if df.empty: return None
    parts = df['d'].str.split('/', expand=True)
    if parts.shape[1] != 3: return None
    parts = parts.astype(int); parts[0] += 1911
    df['date'] = pd.to_datetime(dict(year=parts[0], month=parts[1], day=parts[2]), errors='coerce')
    for c in ['c', 'o', 'h', 'l', 'val', 'vol']:
        df[c] = pd.to_numeric(df[c].str.replace(',', ''), errors='coerce')
    df = df.dropna(subset=['date', 'c']).set_index('date').sort_index()
    df = df[~df.index.duplicated()]
    return df[['c', 'h', 'l', 'val']]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dir', required=True); ap.add_argument('--out', default='./output')
    ap.add_argument('--start', default='2005-01-01'); ap.add_argument('--end', default='2019-04-30'); ap.add_argument('--min-value', type=float, default=1e7)
    ap.add_argument('--rt-cost', type=float, default=0.005); ap.add_argument('--max-files', type=int, default=0)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.dir, '*.csv')))
    if a.max_files: files = files[:a.max_files]
    closes, highs, vals = {}, {}, {}
    for p in files:
        code = Path(p).stem
        if not (code.isdigit() and len(code) == 4): continue   # 四碼普通股（含下市）
        try: df = load_one(p)
        except Exception: continue
        if df is None or len(df) < 300: continue
        closes[code] = df['c']; highs[code] = df['h']; vals[code] = df['val']
    C = pd.DataFrame(closes).sort_index(); H = pd.DataFrame(highs).reindex(C.index); V = pd.DataFrame(vals).reindex(C.index)
    C = C.loc[:a.end]; H = H.loc[:a.end]; V = V.loc[:a.end]
    print('universe files', len(closes), 'dates', C.index.min().date(), C.index.max().date(), 'shape', C.shape)
    R = C.pct_change()
    R = R.where(R.abs() < 0.30)   # 去除資料錯誤（台股日漲跌幅 ≤ 10%，±30% 為容錯）
    month_end = C.groupby([C.index.year, C.index.month]).apply(lambda x: x.index[-1]).values
    month_end = pd.DatetimeIndex(month_end); month_end = month_end[(month_end >= pd.Timestamp(a.start) - pd.Timedelta(days=400))]
    # 訊號（以月末資料計算，用於下個月）
    logC = np.log(C)
    sig = {
        'rev_1m': -(C / C.shift(21) - 1),                    # 1 月反轉：過去一月跌越多越好
        'mom_12_1': C.shift(21) / C.shift(252) - 1,          # 12-1 動能
        'low_vol_60d': -R.rolling(60).std(),                 # 低波動
        'low_max_20d': -R.rolling(20).max(),                 # 低 MAX（避開樂透股）
        'liquidity_val_60d': V.rolling(60).mean(),           # 高成交值（流動性）
        'near_52w_high': C / H.rolling(252).max(),           # 接近 52 週高點
        'price_level': C,                                    # 價格（高價 vs 低價股）
        'vol_shock_20_250': -(R.rolling(20).std() / R.rolling(250).std()),  # 近期波動相對低
    }
    # 月報酬矩陣（月末到月末）
    Cm = C.reindex(month_end).ffill(limit=5)
    Rm = Cm.pct_change().shift(-1)   # 下個月報酬，對齊訊號月末
    Vm = V.rolling(60).mean().reindex(month_end)
    results = {}
    me = [d for d in month_end if d >= pd.Timestamp(a.start)]
    for name, S in sig.items():
        Sm = S.reindex(month_end)
        q_rets = {k: [] for k in range(5)}; ew = []; turnover = []; prev_top = set()
        for d in me:
            if d not in Rm.index: continue
            elig = (Vm.loc[d] >= a.min_value) & (Cm.loc[d] >= 10) & Sm.loc[d].notna() & Rm.loc[d].notna()
            s = Sm.loc[d][elig]
            if s.count() < 100: continue
            ranks = pd.qcut(s.rank(method='first'), 5, labels=False)
            r_next = Rm.loc[d][elig]
            for k in range(5): q_rets[k].append((d, r_next[ranks == k].mean()))
            ew.append((d, r_next.mean()))
            top = set(s.index[ranks == 4]); turnover.append(len(top - prev_top) / max(len(top), 1)); prev_top = top
        if len(ew) < 24: continue
        ewS = pd.Series(dict(ew)); qS = {k: pd.Series(dict(v)) for k, v in q_rets.items()}
        to = float(np.mean(turnover)) if turnover else 1.0
        def ann(r):
            r = r.dropna(); yrs = len(r) / 12; eq = (1 + r).cumprod(); vol = r.std() * math.sqrt(12); cagr = eq.iloc[-1] ** (1 / yrs) - 1
            return {'CAGR': round(float(cagr), 4), 'vol': round(float(vol), 4), 'sharpe': round(float(cagr / vol), 2), 'maxDD': round(float((eq / eq.cummax() - 1).min()), 4)}
        top_gross = qS[4]; top_net = qS[4] - to * a.rt_cost   # 每月換掉 to 比例 × 來回成本
        spread = qS[4] - qS[0]
        tstat = float(spread.mean() / (spread.std() / math.sqrt(len(spread))))
        excess = qS[4] - ewS
        t_ex = float(excess.mean() / (excess.std() / math.sqrt(len(excess))))
        results[name] = {'months': int(len(ewS)), 'period': [str(ewS.index[0].date()), str(ewS.index[-1].date())],
                         'quintile_avg_monthly_pct': {f'Q{k+1}': round(float(qS[k].mean() * 100), 3) for k in range(5)},
                         'universe_EW_avg_monthly_pct': round(float(ewS.mean() * 100), 3),
                         'Q5_minus_Q1_monthly_pct': round(float(spread.mean() * 100), 3), 't_Q5_Q1': round(tstat, 2),
                         'Q5_minus_EW_monthly_pct': round(float(excess.mean() * 100), 3), 't_Q5_EW': round(t_ex, 2),
                         'avg_monthly_turnover_top_quintile': round(to, 3),
                         'Q5_gross': ann(top_gross), 'Q5_net_of_cost': ann(top_net), 'universe_EW': ann(ewS),
                         'Q5_net_minus_EW_CAGR_pp': round((ann(top_net)['CAGR'] - ann(ewS)['CAGR']) * 100, 2)}
        print(name, json.dumps({k: results[name][k] for k in ['months', 'Q5_minus_Q1_monthly_pct', 't_Q5_Q1', 'Q5_minus_EW_monthly_pct', 't_Q5_EW', 'avg_monthly_turnover_top_quintile', 'Q5_net_minus_EW_CAGR_pp']}, ensure_ascii=False))
    meta = {'n_stocks': int(C.shape[1]), 'date_range': [str(C.index.min().date()), str(C.index.max().date())], 'min_daily_value_NT': a.min_value, 'rt_cost': a.rt_cost, 'start': a.start, 'end': a.end,
            'note': '未還原權息；四碼代號含已下市；等權；月末換股；成本 = 周轉率 × 來回成本'}
    (out / 'cross_sectional_factors.json').write_text(json.dumps({'meta': meta, 'results': results}, ensure_ascii=False, indent=1), encoding='utf-8')
    print('saved', out / 'cross_sectional_factors.json')

if __name__ == '__main__':
    main()
