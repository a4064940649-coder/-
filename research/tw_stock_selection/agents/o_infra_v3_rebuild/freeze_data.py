#!/usr/bin/env python3
"""第 0 步：三份資料表凍結（sha256）＋覆蓋率＋「每個期間被誰看過」。在任何策略報酬之前執行；已存在就拒絕覆寫。
python3 -I -B freeze_data.py → data_freeze.json"""
import json, hashlib, datetime
from pathlib import Path
import pandas as pd, numpy as np
HERE = Path(__file__).resolve().parent
F = HERE / 'data_freeze.json'
assert not F.exists(), 'data_freeze.json 已存在，不覆寫'
def sha(f): return hashlib.sha256((HERE / f).read_bytes()).hexdigest()
files = ['delist_prices.parquet', 'merger_consideration.csv', 'disposal_hist.csv', 'delist_official.csv']
dm = json.loads((HERE / 'delist_prices_meta.json').read_text(encoding='utf-8'))
hm = json.loads((HERE / 'disposal_hist_meta.json').read_text(encoding='utf-8'))
pr = json.loads((HERE / 'probe_dead2.json').read_text(encoding='utf-8'))
E = pd.read_csv(HERE / 'merger_consideration.csv', dtype={'code': str}).set_index('code')

# 併購：凍結（持有中永久停止交易）事件中，有對價資料的比例
cov_m = {}
for key, v in pr.items():
    if key.startswith('_'): continue
    n = sum(x['n'] for x in v['dead'].values())
    if n == 0: cov_m[key] = {'dead_events': 0}; continue
    typed = sum(x['n'] for c, x in v['dead'].items() if c in E.index and E.loc[c, 'type'] != 'unknown')
    exact = sum(x['n'] for c, x in v['dead'].items() if c in E.index and ((E.loc[c, 'type'] == 'cash' and pd.notna(E.loc[c, 'cash'])) or E.loc[c, 'type'] in ('swap', 'failure')))
    verified = sum(x['n'] for c, x in v['dead'].items() if c in E.index and E.loc[c, 'verified'] == 'news')
    codes = len(v['dead'])
    cov_m[key] = {'dead_events': n, 'dead_codes': codes, 'type_known_share': typed / n, 'usable_consideration_share': exact / n, 'news_verified_share': verified / n}

# 處置：逐年事件數（近似名單 vs 2021-09-24 以後真實名單年均）
H = pd.read_csv(HERE / 'disposal_hist.csv', parse_dates=['start'])
real = {int(k): v for k, v in hm['real_events_per_year_panel_codes'].items()}
real_mean = float(np.mean([real[y] for y in (2022, 2023, 2024, 2025)]))
yr = H.groupby(H['start'].dt.year).size()
gap_years = [int(y) for y, n in yr.items() if n < 0.5 * real_mean]
ch = [x for x in hm['iterations'] if x['version'] == hm['chosen']][0]

out = {'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
       'note': '在任何策略報酬之前凍結；之後所有回測斷言下列 sha256。',
       'files': {f: {'sha256': sha(f)} for f in files},
       'coverage': {
           'delist_prices': {'should_have_4digit_2019-05-17_2023': dm['n_target_4digit_2019-05-17_2023'], 'got_stocks_any': dm['n_target_in_panel'],
                             'stocks_full': dm['stocks_full'], 'stockdays_need': dm['coverage_stockdays']['need'], 'stockdays_got': dm['coverage_stockdays']['got'],
                             'stockday_rate': dm['coverage_stockdays']['rate'], 'cross_check': dm['cross_check'],
                             'new_listing_after_2019_06_then_delisted_by_2023': 0,
                             'note': '終止上市名單 2019-05-17～2023 共 37 列，其中 5 檔為 6 碼 DR（不在面板宇宙）；32 檔 4 碼全部在面板內、資料停在 2019-05-16 前後。'},
           'merger': cov_m,
           'disposal': {'proxy_chosen': hm['chosen'], 'select_precision': ch['select_2021-09-24_2023']['precision'], 'select_recall': ch['select_2021-09-24_2023']['recall'],
                        'validate_precision': ch['validate_2024_2026']['precision'], 'validate_recall': ch['validate_2024_2026']['recall'],
                        'proxy_events_per_year': {int(k): int(v) for k, v in yr.items()}, 'real_events_per_year': real, 'real_mean_2022_2025': real_mean,
                        'gap_years_lt_half_real_mean': gap_years,
                        'source_search': 'GitHub 找不到 2021-09 以前的 TWSE punish 封存；新聞鏡像每年 23～105 篇提到處置，不足以當名單 → 價格規則近似'}},
       'periods_seen_by': {
           'S1 2006-04～2013': ['main_v2（正式數字）', 'o_rev_survivorship（R_DEL 不改 S1 以外）', 'disposal_bear_longhistory（2005～2021-09 近似處置事件，事件層）', 'disposal_x_main（近似處置事件，事件層）'],
           'S2 2014～2020': ['main_v2', 'o_rev_survivorship（R_DEL、B05、B10、去凍結 B10）', 'disposal_bear_longhistory／disposal_x_main（事件層）'],
           'S3 2021-03～2026-10': ['main_v2', 'o_rev_survivorship', 'o_rev_survivorship 執行審查（不排除處置 +5.1pp）', 'disposal_reconcile／disposal_v2（真實名單事件層）'],
           'S4 2024～2026-10': ['main_v2', 'o_rev_survivorship', 'integrate_o2（TSMC30）'],
           'conclusion': '全部期間都已被看過；本題不產生新訊號，只修資料與引擎，因此沒有「盲測」期間。處置近似規則的版本選擇只用 2021-09-24～2023 真實名單（事件層），驗證 2024～2026 只報告。'},
       'delist_prices_meta_sha256': hashlib.sha256((HERE / 'delist_prices_meta.json').read_bytes()).hexdigest(),
       'disposal_hist_meta_sha256': hashlib.sha256((HERE / 'disposal_hist_meta.json').read_bytes()).hexdigest()}
F.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
print(json.dumps(out['coverage']['delist_prices'], ensure_ascii=False)[:600])
for k, v in cov_m.items():
    if v.get('dead_events'): print(k, v)
print(json.dumps(out['coverage']['disposal'], ensure_ascii=False))
