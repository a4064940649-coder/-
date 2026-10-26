#!/usr/bin/env python3
"""每月選股清單：「52 週高點 × 月營收動能」規則（上市股，回測 2021-01～2026-10 淨年化 49%，見研究報告）。

用法（每月 11 日或之後第一個交易日、收盤前執行）：
  python3 screen_monthly.py --data <tw-stock-screener>/data [--asof 2026-11-11] [--panel <面板資料夾>]
  --data   ：stevenhsiehsheng-ctrl/tw-stock-screener 的 data 資料夾（history.csv.gz、stock_list.csv、extras/rev_5y.csv.gz、
             extras/rev_hist.csv.gz、extras/exdiv_5y.csv.gz、extras/disposal_5y.csv.gz）；第三方整理的官方資料，重要數字請以 MOPS／證交所核對
  --panel  ：（選用）build_twse_panel.py 產生的面板；給了就用面板的還原價，可重現回測持股
規則：
  1 可投資：上市普通股、60 日平均成交值 ≥ NT$1,000 萬、收盤 ≥ 10 元、非處置期間；取 60 日均成交值前 300 名
  2 訊號 A：收盤（還原）／過去 252 個交易日最高還原收盤
  3 訊號 B：最近 3 個月營收年增率平均（用「上個月」為最新月份；該月申報家數不足 80% 時停止並提示）
  4 分數 = A 的百分位 × 0.5 + B 的百分位 × 0.5；取前 30 檔、等權；持有到下個月同一規則換股
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))

def is4(c): return isinstance(c, str) and len(c) == 4 and c.isdigit() and c[0] != '0'

def adjusted_from_history(D: Path):
    from build_twse_panel import compute_returns
    lst = pd.read_csv(D / 'stock_list.csv', dtype=str)
    twse = {c for c in lst.loc[lst['market'] == 'TWSE', 'code'] if is4(c)}
    h = pd.read_csv(D / 'history.csv.gz', dtype={'code': str}, parse_dates=['date'])
    h = h[h['code'].isin(twse) & (h['volume'] > 0) & (h['close'] > 0)]
    ex = pd.read_csv(D / 'extras/exdiv_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
    ex = ex[(ex['ref_price'] > 0) & (ex['prev_close'] > 0)].copy(); ex['f'] = ex['prev_close'] / ex['ref_price']
    ex = ex[(ex['f'] >= 1) & (ex['f'] < 3)]; exg = {c: g.sort_values('date') for c, g in ex.groupby('code')}
    CAL = pd.DatetimeIndex(sorted(h['date'].unique()))
    adj, close, value = {}, {}, {}
    for code, g in h.groupby('code'):
        g = g.set_index('date').sort_index()
        df = pd.DataFrame({'o': g['open'], 'h': g['high'], 'c': g['close'], 'val': g['volume'] * g['close'], 'x': False, 'chgv': np.nan, 'src': 4.0})
        r, _ = compute_returns(df, [(CAL[0], CAL[-1])], exg.get(code), CAL)
        adj[code] = pd.Series((1 + pd.Series(r, index=df.index).fillna(0)).cumprod().to_numpy(), index=df.index)
        close[code] = df['c']; value[code] = df['val']
    return pd.DataFrame(adj).reindex(CAL), pd.DataFrame(close).reindex(CAL), pd.DataFrame(value).reindex(CAL)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True); ap.add_argument('--asof'); ap.add_argument('--panel'); ap.add_argument('--n', type=int, default=30)
    a = ap.parse_args(); D = Path(a.data)
    if a.panel:
        Pp = Path(a.panel); adj = pd.read_parquet(Pp / 'adj.parquet'); C = pd.read_parquet(Pp / 'close_raw.parquet'); V = pd.read_parquet(Pp / 'value.parquet')
    else:
        adj, C, V = adjusted_from_history(D)
    asof = pd.Timestamp(a.asof) if a.asof else adj.index[-1]
    t = adj.index[adj.index <= asof][-1]
    adjc = adj.where(C.notna())
    s52 = (adjc / adjc.rolling(252).max()).loc[t]
    v60 = V.rolling(60).mean().loc[t]; px = C.loc[t]
    rev = pd.read_csv(D / 'extras/rev_5y.csv.gz', dtype={'code': str})
    hist = D / 'extras/rev_hist.csv.gz'
    if hist.exists():
        rh = pd.read_csv(hist, dtype={'code': str})
        if 'ann_date' in rh:   # 只用已公告者（ann_date ≤ 換股日；空白＝早期回補月份，視為已公告）
            ad = pd.to_datetime(rh['ann_date'], errors='coerce'); rh = rh[ad.isna() | (ad <= t)]
        rev = pd.concat([rev, rh[['code', 'ym', 'revenue', 'yoy', 'mom']]]).drop_duplicates(['code', 'ym'], keep='first')   # 重疊月份以 rev_5y 為準
    rev['p'] = pd.PeriodIndex(rev['ym'], freq='M')
    YOY = rev.pivot_table(index='p', columns='code', values='yoy', aggfunc='last').sort_index()
    YOY = YOY.reindex(pd.period_range(YOY.index.min(), YOY.index.max(), freq='M'))
    m = pd.Period(t, 'M') - 1
    if m not in YOY.index: sys.exit(f'營收資料尚無 {m}')
    cnt = YOY.notna().sum(axis=1); typical = cnt.loc[m - 6:m - 1].median()
    if cnt.loc[m] < 0.8 * typical:
        sys.exit(f'{m} 營收僅 {int(cnt.loc[m])} 家申報（平常約 {int(typical)} 家）——請等多數公司公告（法定 10 日前）後再執行')
    yoy3 = YOY.rolling(3, min_periods=3).mean().loc[m].reindex(adj.columns)
    dsp = pd.read_csv(D / 'extras/disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['start', 'end'])
    in_disp = set(dsp.loc[(dsp['start'] <= t) & (dsp['end'] >= t), 'code'])
    base = (v60 >= 1e7) & (px >= 10)
    top300 = base & (v60.where(base).rank(ascending=False) <= 300)      # 先取流動性前 300 名，再剔除處置股（與回測一致）
    el = top300 & ~pd.Series(adj.columns.isin(list(in_disp)), index=adj.columns) & s52.notna() & yoy3.notna()
    score = 0.5 * s52.where(el).rank(pct=True) + 0.5 * yoy3.where(el).rank(pct=True)
    pick = score[el].sort_values(ascending=False).head(a.n)
    names = {}
    lst = pd.read_csv(D / 'stock_list.csv', dtype=str)
    if 'name' in lst: names = dict(zip(lst['code'], lst['name']))
    out = pd.DataFrame({'name': [names.get(c, '') for c in pick.index], 'close': px[pick.index].round(2), 'near_52w_high': s52[pick.index].round(3),
                        'rev_yoy3_pct': yoy3[pick.index].round(1), 'score': pick.round(3), 'avg_value_60d_NT$m': (v60[pick.index] / 1e6).round(0)})
    print(f'換股日 {t.date()}｜營收月份 {m}｜候選 {int(el.sum())} 檔（前 300 名流動性、扣除處置 {len(in_disp)} 檔）')
    print(out.to_string())
    return out

if __name__ == '__main__':
    main()
