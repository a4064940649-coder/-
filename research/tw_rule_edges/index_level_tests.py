#!/usr/bin/env python3
"""台股指數／ETF 日線層級的規則化策略檢定（季節性、隔夜 vs 日內、趨勢過濾、結算週、農曆年）

輸入：任何含 date, open, high, low, close(, volume) 欄位的 CSV（欄名不分大小寫；可為中文「日期/開盤/最高/最低/收盤」）。
輸出：各檢定的統計表（平均報酬、t 值、bootstrap p 值）、以及成本後的策略 vs 買進持有比較。
用法：python3 index_level_tests.py --csv path/to/taiex.csv --name TAIEX [--cost-bps 10] [--out out_dir]
     python3 index_level_tests.py --smoke-test   # 合成資料，僅確認程式可執行
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd

COLMAP = {
    'date': ['date', '日期', 'Date', 'DATE', 'time', '時間'],
    'open': ['open', '開盤', '開盤價', 'Open', 'OPEN', '開盤指數'],
    'high': ['high', '最高', '最高價', 'High', 'HIGH', '最高指數'],
    'low':  ['low', '最低', '最低價', 'Low', 'LOW', '最低指數'],
    'close':['close', '收盤', '收盤價', 'Close', 'CLOSE', 'Adj Close', 'adj_close', '收盤指數'],
    'volume':['volume', '成交量', '成交股數', 'Volume', 'VOLUME', '成交金額'],
}

def load_csv(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    cols = {}
    for std, names in COLMAP.items():
        for n in names:
            if n in raw.columns:
                cols[std] = n; break
    missing = [k for k in ('date','close') if k not in cols]
    if missing:
        raise SystemExit(f"CSV 缺少欄位 {missing}；有的欄位：{list(raw.columns)}")
    df = pd.DataFrame({std: raw[c] for std, c in cols.items()})
    # 民國年處理（例：113/01/02）
    d = df['date'].astype(str).str.strip()
    if d.str.match(r'^\d{2,3}/\d{1,2}/\d{1,2}$').all():
        parts = d.str.split('/', expand=True).astype(int)
        parts[0] += 1911
        d = parts[0].astype(str) + '-' + parts[1].astype(str) + '-' + parts[2].astype(str)
    df['date'] = pd.to_datetime(d, errors='coerce')
    for c in ('open','high','low','close','volume'):
        if c in df:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', ''), errors='coerce')
    df = df.dropna(subset=['date','close']).sort_values('date').drop_duplicates('date').set_index('date')
    if 'open' not in df: df['open'] = df['close'].shift(1)
    for c in ('high','low'):
        if c not in df: df[c] = df[['open','close']].max(axis=1) if c=='high' else df[['open','close']].min(axis=1)
    return df

def validate(df: pd.DataFrame) -> dict:
    bad_ohlc = int(((df['low'] > df[['open','close']].min(axis=1)) | (df['high'] < df[['open','close']].max(axis=1))).sum())
    gaps = df.index.to_series().diff().dt.days
    big_moves = int((df['close'].pct_change().abs() > 0.12).sum())
    return {'rows': int(len(df)), 'start': str(df.index[0].date()), 'end': str(df.index[-1].date()),
            'bad_ohlc_rows': bad_ohlc, 'gaps_gt_10d': int((gaps > 10).sum()), 'moves_gt_12pct': big_moves}

def tstat(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))) if len(x) > 2 and x.std(ddof=1) > 0 else float('nan')

def boot_p(x: np.ndarray, pool: np.ndarray, n_boot=2000, seed=0) -> float:
    """單尾：x 的平均是否顯著高於從 pool 隨機抽同樣筆數的平均"""
    rng = np.random.default_rng(seed)
    x = x[~np.isnan(x)]; pool = pool[~np.isnan(pool)]
    if len(x) < 5 or len(pool) < 50: return float('nan')
    sims = np.array([rng.choice(pool, size=len(x), replace=True).mean() for _ in range(n_boot)])
    return float((sims >= x.mean()).mean())

def ann(ret_daily: pd.Series) -> dict:
    r = ret_daily.dropna()
    if len(r) < 20: return {}
    eq = (1 + r).cumprod()
    years = len(r) / 245
    cagr = eq.iloc[-1] ** (1 / years) - 1
    vol = r.std() * math.sqrt(245)
    dd = (eq / eq.cummax() - 1).min()
    return {'CAGR': round(float(cagr), 4), 'vol': round(float(vol), 4), 'sharpe': round(float(cagr / vol), 2) if vol > 0 else None,
            'maxDD': round(float(dd), 4), 'years': round(years, 1)}

def lunar_new_year_dates():
    # 農曆新年（初一）西曆日期 1990–2030（公開曆法事實）
    s = """1990-01-27 1991-02-15 1992-02-04 1993-01-23 1994-02-10 1995-01-31 1996-02-19 1997-02-07 1998-01-28 1999-02-16
    2000-02-05 2001-01-24 2002-02-12 2003-02-01 2004-01-22 2005-02-09 2006-01-29 2007-02-18 2008-02-07 2009-01-26
    2010-02-14 2011-02-03 2012-01-23 2013-02-10 2014-01-31 2015-02-19 2016-02-08 2017-01-28 2018-02-16 2019-02-05
    2020-01-25 2021-02-12 2022-02-01 2023-01-22 2024-02-10 2025-01-29 2026-02-17 2027-02-06 2028-01-26 2029-02-13 2030-02-03"""
    return pd.to_datetime(s.split())

def run_tests(df: pd.DataFrame, name: str, cost_bps: float, out: Path) -> dict:
    res = {'name': name, 'data': validate(df)}
    cc = df['close'].pct_change()                       # 收盤到收盤
    overnight = df['open'] / df['close'].shift(1) - 1   # 前收 → 今開
    intraday = df['close'] / df['open'] - 1             # 今開 → 今收
    res['overnight_vs_intraday'] = {
        'overnight': ann(overnight), 'intraday': ann(intraday), 'close_to_close': ann(cc),
        'overnight_mean_bps': round(float(overnight.mean() * 1e4), 2), 'intraday_mean_bps': round(float(intraday.mean() * 1e4), 2),
        'overnight_t': round(tstat(overnight.values), 2), 'intraday_t': round(tstat(intraday.values), 2),
    }
    # 月份效應
    monthly = cc.groupby(cc.index.month).agg(['mean', 'count'])
    res['month_of_year_mean_bps'] = {int(m): round(float(v * 1e4), 2) for m, v in monthly['mean'].items()}
    # 月初／月末（最後 1 日 + 前 3 日）效應
    dom = cc.index.to_series().groupby([cc.index.year, cc.index.month]).cumcount()
    rev_dom = cc.index.to_series().groupby([cc.index.year, cc.index.month]).cumcount(ascending=False)
    tom = cc[(rev_dom <= 0) | (dom <= 2)]
    rest = cc[~((rev_dom <= 0) | (dom <= 2))]
    res['turn_of_month'] = {'tom_mean_bps': round(float(tom.mean() * 1e4), 2), 'rest_mean_bps': round(float(rest.mean() * 1e4), 2),
                            'tom_t': round(tstat(tom.values), 2), 'p_boot': round(boot_p(tom.values, cc.values), 3), 'n_tom': int(tom.count())}
    # 週幾效應
    res['day_of_week_mean_bps'] = {int(d): round(float(v * 1e4), 2) for d, v in cc.groupby(cc.index.dayofweek).mean().items()}
    # 農曆年前 5 日／後 5 日
    lny = lunar_new_year_dates()
    pre, post = [], []
    idx = df.index
    for d in lny:
        if d < idx[0] or d > idx[-1]: continue
        before = cc[idx < d].iloc[-5:]
        after = cc[idx > d].iloc[:5]
        pre.append(before.sum()); post.append(after.sum())
    pre, post = np.array(pre), np.array(post)
    # 隨機 5 日視窗作為對照
    rng = np.random.default_rng(1)
    win5 = cc.rolling(5).sum().dropna().values
    res['lunar_new_year'] = {'n_years': int(len(pre)), 'pre5_mean_pct': round(float(pre.mean() * 100), 2) if len(pre) else None,
                             'post5_mean_pct': round(float(post.mean() * 100), 2) if len(post) else None,
                             'pre5_t': round(tstat(pre), 2) if len(pre) > 2 else None,
                             'pre5_p_boot_vs_random5d': round(boot_p(pre, win5), 3) if len(pre) > 4 else None,
                             'random5d_mean_pct': round(float(np.nanmean(win5) * 100), 2)}
    # 台指期結算週（每月第 3 個星期三所在週 vs 其他）
    third_wed = df.index.to_series().apply(lambda d: d.weekday() == 2 and 15 <= d.day <= 21)
    settle_week = cc.index.to_series().apply(lambda d: 15 <= d.day - (d.weekday() - 0) <= 21 and True)  # 粗略：該週含第三個星期三
    wk = cc.groupby([cc.index.year, cc.index.isocalendar().week]).transform('sum')
    sw_mask = third_wed.groupby([third_wed.index.year, third_wed.index.isocalendar().week]).transform('max')
    res['settlement_week'] = {'settle_week_mean_bps': round(float(cc[sw_mask].mean() * 1e4), 2), 'other_mean_bps': round(float(cc[~sw_mask].mean() * 1e4), 2),
                              't_settle': round(tstat(cc[sw_mask].values), 2), 'n_days': int(sw_mask.sum())}
    # 趨勢過濾：收盤 > 200 日均線持有，否則空手（成本：每次切換 cost_bps）
    out_rows = {}
    for L in (100, 200):
        ma = df['close'].rolling(L).mean()
        pos = (df['close'] > ma).astype(int).shift(1).fillna(0)
        trades = pos.diff().abs().fillna(0)
        strat = pos * cc - trades * cost_bps / 1e4
        out_rows[f'MA{L}'] = {**ann(strat), 'switches': int(trades.sum()), 'time_in_market': round(float(pos.mean()), 3)}
    out_rows['buy_and_hold'] = ann(cc)
    res['trend_filter'] = out_rows
    # 儲存
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{name}_index_tests.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    return res

def synthetic(n=6000, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2000-01-03', periods=n)
    r = rng.normal(0.0003, 0.012, n)
    close = 5000 * np.cumprod(1 + r)
    open_ = close * (1 + rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.008, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.008, n))
    return pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close, 'volume': rng.integers(1e6, 1e7, n)}, index=dates)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv'); ap.add_argument('--name', default='SERIES'); ap.add_argument('--cost-bps', type=float, default=10.0)
    ap.add_argument('--out', default='./output'); ap.add_argument('--smoke-test', action='store_true')
    a = ap.parse_args()
    if a.smoke_test:
        print('=== SMOKE TEST — SYNTHETIC DATA — NOT REAL RESULTS ===')
        df = synthetic(); name = 'SYNTHETIC'
    else:
        if not a.csv: ap.error('--csv 必填（或用 --smoke-test）')
        df = load_csv(a.csv); name = a.name
    res = run_tests(df, name, a.cost_bps, Path(a.out))
    print(json.dumps(res, ensure_ascii=False, indent=1)[:4000])

if __name__ == '__main__':
    main()
