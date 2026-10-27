#!/usr/bin/env python3
"""上櫃面板驗證：來源重疊期逐筆比對收盤、除權息處理、逐年涵蓋、下櫃股、缺漏日。輸出 verify.json。"""
import json, glob, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_tpex_panel as BP  # noqa: E402

S = BP.S; LS = BP.LS; OUT = BP.OUT
TP = {k: pd.read_parquet(OUT / f'{k}.parquet') for k in ['close_raw', 'adj', 'ret', 'flag', 'value', 'srcname']}
rows = pickle.load(open(OUT / 'rows.pkl', 'rb'))['raw']
codes = sorted(rows)
res = {}


def cmp(a, b, name):
    """a, b: {code: Series close}，回傳重疊筆數與相符比例。"""
    n = eq = eq1 = 0; worst = []
    for c in set(a) & set(b):
        j = pd.concat([a[c], b[c]], axis=1, join='inner').dropna()
        if not len(j): continue
        d = (j.iloc[:, 0] / j.iloc[:, 1] - 1).abs()
        n += len(d); eq += int((d < 1e-6).sum()); eq1 += int((d < 0.001).sum())
        if (d >= 0.001).any(): worst.append((c, int((d >= 0.001).sum())))
    worst.sort(key=lambda x: -x[1])
    res.setdefault('overlap_close', {})[name] = {'n_rows': n, 'n_codes': len(set(a) & set(b)), 'exact_pct': round(100 * eq / max(n, 1), 3),
                                                 'within_0.1pct': round(100 * eq1 / max(n, 1), 3), 'codes_with_mismatch_top': worst[:8]}
    print(name, res['overlap_close'][name], flush=True)


def main():
    asoul = {c: BP.BT.read_exchange(p, False)['c'] for c, p in ((Path(p).stem, p) for p in glob.glob(str(S / 'tsec_all/*.csv'))) if c in rows}
    twan = {Path(p).stem: BP.read_B(p)['c'] for p in glob.glob(str(LS / 'twan/otc_trading_raw_data/*.csv')) if Path(p).stem in rows}
    henry = {Path(p).stem: BP.read_fm(p)['c'] for p in glob.glob(str(LS / 'henry/data/stocks/*.csv')) if Path(p).stem in rows}
    hschi = {Path(p).stem: BP.read_fm(p)['c'] for p in glob.glob('/home/user/hschi1106/stock_ga/data/stock_dayK/*.csv') if Path(p).stem in rows}
    cal = pd.DatetimeIndex(pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date'])['date'])
    E = BP.load_tasi(set(cal)); E = {c: g['c'] for c, g in E.items() if c in rows}
    hist = pd.read_csv(BP.SCR / 'history.csv.gz', dtype={'code': str}, parse_dates=['date'])
    hist = hist[hist['code'].isin(rows) & (hist['volume'] > 0)]
    F = {c: g.set_index('date')['close'] for c, g in hist.groupby('code')}
    sl = lambda D, a, b: {c: s.loc[a:b] for c, s in D.items()}
    cmp(sl(asoul, '2013-01-01', '2014-12-31'), sl(henry, '2013-01-01', '2014-12-31'), 'Asoul_vs_FinMind(henry)_2013-2014')
    cmp(sl(asoul, '2015-01-01', '2016-12-31'), sl(twan, '2015-01-01', '2016-12-31'), 'Asoul_vs_taiwanan_2015-2016')
    cmp(sl(twan, '2015-01-01', '2021-04-29'), sl(henry, '2015-01-01', '2021-04-29'), 'taiwanan_vs_FinMind(henry)_2015-2021-04')
    cmp(sl(henry, '2020-01-01', '2022-12-30'), sl(hschi, '2020-01-01', '2022-12-30'), 'henry_vs_hschi_2020-2022')
    cmp(sl(hschi, '2023-10-01', '2025-05-28'), E, 'hschi_vs_Tasi_2023-10_2025-05')
    cmp(sl(E, '2025-09-24', '2026-10-31'), F, 'Tasi_vs_history.csv.gz_2025-09-24_')
    # Tasi 2025-09-24 之後完整的 kbar 也拿來比（E 只載到 2025-09-23，另讀）
    fs = sorted(glob.glob(str(LS / 'tasi/kbar/kbar2025-1*.csv')) + glob.glob(str(LS / 'tasi/kbar/kbar2026-*.csv')))
    parts = []
    for f in fs:
        try:
            x = pd.read_csv(f, dtype={'stoc': str})
        except Exception:
            continue
        if 'stoc' in x.columns and len(x): parts.append(x)
    t = pd.concat(parts); t['date'] = pd.to_datetime(t['date']); t = t[t['stoc'].isin(rows) & (t['vol'] > 0)]
    tt = {c: g.drop_duplicates('date').set_index('date')['clo'] for c, g in t.groupby('stoc')}
    cmp(tt, sl(F, '2025-10-01', '2026-10-31'), 'Tasi_vs_history.csv.gz_2025-10_2026-10')
    # 成交量口徑：Tasi 張數×1000 / history 股數
    vt = t.drop_duplicates(['stoc', 'date']).set_index(['stoc', 'date'])['vol'] * 1000
    vh = hist.set_index(['code', 'date'])['volume']
    j = pd.concat([vt, vh], axis=1, join='inner').dropna(); j = j[j.iloc[:, 1] > 0]
    res['volume_ratio_Tasi_over_history_median'] = round(float((j.iloc[:, 0] / j.iloc[:, 1]).median()), 4)
    # 逐年涵蓋與來源
    C = TP['close_raw']; srcn = TP['srcname']
    res['stocks_with_price_median_by_year'] = {int(y): int(v) for y, v in C.notna().sum(axis=1).groupby(C.index.year).median().items() if y >= 2007}
    res['source_share_by_year'] = {int(y): {k: round(float(v), 3) for k, v in g.stack().value_counts(normalize=True).items()} for y, g in srcn.groupby(srcn.index.year) if y >= 2007}
    tpex_now = set(pd.read_csv(BP.SCR / 'stock_list.csv', dtype=str).query("market=='TPEX'")['code'])
    gone = [c for c in C.columns if c not in tpex_now]
    last = C[gone].apply(lambda s: s.last_valid_index())
    res['not_in_current_TPEX_list'] = {'n': len(gone), 'last_date_year_counts': {int(k): int(v) for k, v in last.dt.year.value_counts().sort_index().items()}}
    # 缺漏日：上櫃有值檔數相對 21 日中位數
    cnt = C.notna().sum(axis=1); med = cnt.rolling(21, center=True, min_periods=5).median()
    bad = cnt[(cnt < 0.85 * med) & (C.index >= '2007-05-01')]
    res['partial_days'] = {'n': int(len(bad)), 'by_year': {int(y): int(v) for y, v in bad.groupby(bad.index.year).size().items()}, 'examples': [str(d.date()) for d in bad.index[:15]]}
    # flag
    fl = TP['flag']
    res['flag_counts_by_year'] = {int(y): {int(f): int(n) for f, n in g.stack().value_counts().sort_index().items() if f > 0} for y, g in fl.groupby(fl.index.year) if y >= 2007}
    # 除權息還原檢查：exdiv_5y 事件日（2021-10 起）面板報酬 vs 收盤/參考價−1
    ex = pd.read_csv(BP.SCR / 'extras/exdiv_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
    ex = ex[ex['code'].isin(C.columns) & (ex['ref_price'] > 0)]
    R = TP['ret']; out = []
    for _, r in ex.iterrows():
        d = r['date']
        if d not in C.index or pd.isna(C.at[d, r['code']]): continue
        rr = R.at[d, r['code']]; want = C.at[d, r['code']] / r['ref_price'] - 1
        out.append((rr, want, srcn.at[d, r['code']]))
    o = pd.DataFrame(out, columns=['panel', 'want', 'src']).dropna()
    o['ok'] = (o['panel'] - o['want']).abs() < 0.002
    res['exdiv_check_2021-10_'] = {'n': int(len(o)), 'match_pct': round(100 * float(o['ok'].mean()), 2), 'by_src': {k: round(100 * float(g['ok'].mean()), 1) for k, g in o.groupby('src')},
                                   'naive_close_over_prevclose_would_mismatch_pct': None}
    # 年化報酬合理性：上櫃面板等權（每日 ret 截面平均）逐年 vs 只用原始收盤（不還原）
    rr = R.where(R.abs() < 0.5)
    raw_r = (C / C.ffill().shift() - 1).where(C.notna() & C.ffill().shift().notna())
    res['EW_daily_mean_ret_yearly_pct'] = {int(y): round(float(((1 + g.mean(axis=1).fillna(0)).prod() - 1) * 100), 1) for y, g in rr.groupby(rr.index.year) if y >= 2008}
    res['EW_raw_price_only_yearly_pct'] = {int(y): round(float(((1 + g.mean(axis=1).fillna(0)).prod() - 1) * 100), 1) for y, g in raw_r.where(raw_r.abs() < 0.5).groupby(raw_r.index.year) if y >= 2008}
    (HERE / 'verify.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    print(json.dumps({k: v for k, v in res.items() if k != 'overlap_close'}, ensure_ascii=False, default=str)[:5000])


if __name__ == '__main__':
    main()
