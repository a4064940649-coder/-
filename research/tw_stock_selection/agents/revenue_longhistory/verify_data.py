#!/usr/bin/env python3
"""驗證 NiklavsD/bottleneck-signals 的 data/processed/tw_revenue.csv（MOPS t21sc03 每月彙總頁解析，2005-01 起）。
1. 與 rev_5y 重疊月份逐筆比對營收與年增率；2. 逐年涵蓋檔數（上市 sii）與面板覆蓋率；3. 已下市公司；4. 公告日；5. 2013 IFRS 口徑斷點。
輸出 verify.json。只讀資料檔，不執行下載來源的程式。
"""
import json
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
SRC = Path('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv')
X = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data')
P = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')

def load_long():
    d = pd.read_csv(SRC, dtype={'code': str, 'market': str, 'name': str})
    d['p'] = pd.PeriodIndex(pd.to_datetime(d['month']), freq='M')
    d = d[d['code'].str.fullmatch(r'\d{4}')]            # 普通股四碼
    d = d.sort_values(['p', 'market']).drop_duplicates(['p', 'code'], keep='first')   # sii 優先
    return d

def yoy_from_levels(d):
    L = d.pivot_table(index='p', columns='code', values='revenue_k_twd', aggfunc='last').sort_index()
    L = L.reindex(pd.period_range(L.index.min(), L.index.max(), freq='M'))
    prev = L.shift(12)
    Y = (L / prev - 1) * 100
    Y = Y.where(prev > 0)
    return L, Y

def main():
    out = {'source': 'https://github.com/NiklavsD/bottleneck-signals (data/processed/tw_revenue.csv, commit ae3126d)'}
    d = load_long()
    out['rows'] = int(len(d)); out['months'] = [str(d['p'].min()), str(d['p'].max())]
    out['rows_by_market'] = d['market'].value_counts().to_dict()
    L, Y = yoy_from_levels(d)
    # 1. 與 rev_5y 比對
    r5 = pd.read_csv(X / 'extras/rev_5y.csv.gz', dtype={'code': str}); r5['p'] = pd.PeriodIndex(r5['ym'], freq='M')
    m = r5.merge(d[['p', 'code', 'revenue_k_twd', 'market']], on=['p', 'code'], how='left')
    m['has'] = m['revenue_k_twd'].notna()
    mm = m[m['has']].copy()
    mm['rel'] = (mm['revenue_k_twd'] / mm['revenue'] - 1).abs()
    ym = Y.rename_axis(index='p', columns='code').stack().rename('yoy_long').reset_index()
    mm = mm.merge(ym, on=['p', 'code'], how='left')
    mm['dy'] = (mm['yoy_long'] - mm['yoy']).abs()
    out['vs_rev5y'] = {
        'rev5y_rows': int(len(m)), 'found_in_long': int(m['has'].sum()), 'found_rate': round(float(m['has'].mean()), 4),
        'revenue_exact_rate(<0.01%)': round(float((mm['rel'] < 1e-4).mean()), 4),
        'revenue_within_1pct': round(float((mm['rel'] < 0.01).mean()), 4),
        'yoy_rows_both': int(mm['dy'].notna().sum()),
        'yoy_within_0.1pp': round(float((mm['dy'] < 0.1).mean()), 4) if mm['dy'].notna().any() else None,
        'yoy_within_1pp': round(float((mm['dy'].dropna() < 1).mean()), 4),
        'by_market_of_long': mm['market'].value_counts().to_dict(),
    }
    miss = m[~m['has']]
    out['vs_rev5y']['missing_examples'] = miss.groupby('code').size().sort_values(ascending=False).head(15).to_dict()
    bad = mm[mm['rel'] >= 0.01].groupby('code').size().sort_values(ascending=False).head(10)
    out['vs_rev5y']['mismatch_codes_top'] = bad.to_dict()
    # 不吻合是否集中在 KY（rev_5y 有、長資料沒有 → _1 外國公司頁沒抓）
    sl = pd.read_csv(X / 'stock_list.csv', dtype={'code': str})
    nm = dict(zip(sl['code'], sl['name']))
    out['vs_rev5y']['missing_codes_KY_share'] = round(float(np.mean(['KY' in str(nm.get(c, '')) for c in miss['code'].unique()])), 3) if len(miss) else None
    out['vs_rev5y']['missing_codes_n'] = int(miss['code'].nunique())
    # 2. 逐年涵蓋：上市（sii）家數；面板中當年有交易的股票有營收的比例
    adj = pd.read_parquet(P / 'adj.parquet')
    traded = adj.notna().groupby(adj.index.year).any()
    cov = {}
    sii = d[d['market'] == 'sii']
    for y in range(2005, 2027):
        cy = set(sii.loc[sii['p'].dt.year == y, 'code']); call = set(d.loc[d['p'].dt.year == y, 'code'])
        tr = set(traded.columns[traded.loc[y]]) if y in traded.index else set()
        tr4 = {c for c in tr if len(c) == 4 and c.isdigit() and not c.startswith('00')}
        cov[y] = {'sii_codes': len(cy), 'panel_traded_common': len(tr4), 'panel_with_rev_any_market': len(tr4 & call),
                  'panel_cov_rate': round(len(tr4 & call) / max(len(tr4), 1), 3)}
    out['coverage_by_year'] = cov
    # 3. 已下市：長資料中出現、但不在目前 stock_list（上市+上櫃）的代碼
    cur = set(sl['code'])
    gone = sorted(set(sii['code']) - cur)
    lastp = sii.groupby('code')['p'].max()
    out['delisted'] = {'sii_codes_total': int(sii['code'].nunique()), 'sii_codes_not_in_current_list': len(gone),
                       'by_last_year': pd.Series([lastp[c].year for c in gone]).value_counts().sort_index().to_dict(),
                       'examples': gone[:20]}
    pan = set(adj.columns)
    out['delisted']['not_in_current_list_but_in_panel'] = len(set(gone) & pan)
    # 4. 公告日：資料無公告日欄位；法定期限 = 次月 10 日（證交法第 36 條）。換股日取 ≥ 11 日第一個交易日即符合。
    out['announce'] = 'tw_revenue.csv 沒有公告日欄位；彙總頁為次月 10 日法定期限後的全體公告。回測只在 ≥ 11 日的第一個交易日使用上月營收。'
    # 5. 2013 IFRS 斷點：上市公司 2013 起申報合併營收（之前為個體）。看 YoY 截面中位數與極端值比例
    med = {}
    for y in range(2006, 2027):
        v = Y[(Y.index.year == y)].stack()
        med[y] = {'median_yoy': round(float(v.median()), 1), 'share_abs_gt_100': round(float((v.abs() > 100).mean()), 3), 'n': int(len(v))}
    out['yoy_cross_section_by_year'] = med
    # 年內連續性：同一公司 2012-12 → 2013-01 營收倍數的分布（合併 vs 個體）
    if pd.Period('2012-12', 'M') in L.index:
        jump = (L.loc[pd.Period('2013-01', 'M')] / L.loc[pd.Period('2012-01', 'M')]).dropna()
        jump0 = (L.loc[pd.Period('2012-01', 'M')] / L.loc[pd.Period('2011-01', 'M')]).dropna()
        out['ifrs_break'] = {'ratio_2013-01_over_2012-01_p90': round(float(jump.quantile(0.9)), 2), 'p99': round(float(jump.quantile(0.99)), 2),
                             'ratio_2012-01_over_2011-01_p90': round(float(jump0.quantile(0.9)), 2), 'p99_prev': round(float(jump0.quantile(0.99)), 2)}
    (HERE / 'verify.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))

if __name__ == '__main__':
    main()
