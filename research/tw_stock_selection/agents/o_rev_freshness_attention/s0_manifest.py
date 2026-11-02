#!/usr/bin/env python3
"""raw/manifest.json：本題 clone 的公開 GitHub 封存（blobless、no-checkout）的 commit 與用到的衍生檔 sha256，
以及搜尋紀錄（gh search 的查詢字串與結論）。raw/ 只讀日期與檔名，不執行其中任何程式。"""
import subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fa_common as C

def git(repo, *a): return subprocess.run(['git', '-C', str(repo), *a], capture_output=True, text=True, check=True).stdout.strip()
M = {'clones': {}, 'search_log': [
    {'tool': 'gh search repos', 'queries': ['月營收 daily', 'monthly revenue taiwan stock crawler', 'twse revenue data actions', 'mops revenue csv daily update',
                                            '台股 月營收 資料 自動更新', '營收', 'monthly revenue taiwan', 'taiwan stock revenue', 'mops monthly revenue',
                                            'twse monthly revenue', 'tw stock data github actions', 'taiwan stock daily data archive'],
     'result': '只找到即時爬蟲或單次快照（LouisLee0801/Taiwan-Stock-Revenue 2026-06 建立、RainBoltz/TaiwanStockRevenueInspector 2019-04 一週、mjib007/revenue-yoy-backtest 單一 commit）；沒有 2019～2024 的每日 commit 營收封存'},
    {'tool': 'gh search code', 'queries': ['t21sc03', 't21sc03_108', 'mops.twse.com.tw/nas/t21/sii', 'TaiwanStockMonthRevenue filename:csv'],
     'result': '找到 wenchiehlee/Python-Actions.FinMind（GitHub Actions 每日更新，type5 營收檔 2026-09-21 才加入、140 檔）；其餘為爬蟲程式碼'},
    {'tool': 'local', 'repo': 'shohokuno10/Tasi', 'result': '營收/YYYYMM.csv 在 2024-06 起有每日快照 commit（每月約 5～10 個 commit 日）；2021-01～2024-05 月檔在 2024-06-14 一次補進（非 PIT）'},
    {'tool': 'local', 'repo': 'a607ernie/ptt-stock-data', 'result': '2023-01～2025-05 的 PTT 文章，期間太短（最多 29/89 個月），不納入'},
]}
for nm, url in (('tasi', 'https://github.com/shohokuno10/Tasi'), ('finmind', 'https://github.com/wenchiehlee/Python-Actions.FinMind')):
    r = C.RAW / nm
    M['clones'][nm] = {'url': url, 'clone': 'git clone --filter=blob:none --no-checkout', 'head': git(r, 'rev-parse', 'HEAD'),
                       'n_commits': int(git(r, 'rev-list', '--count', 'HEAD')), 'first_commit_date': git(r, 'log', '--reverse', '--format=%cs').splitlines()[0],
                       'last_commit_date': git(r, 'log', '-1', '--format=%cs')}
M['derived'] = {str(p.relative_to(C.HERE)): C.sha(p) for p in sorted(C.DATA.glob('gate_population.csv.gz'))}
M['external_inputs_sha256'] = {str(p): C.sha(p) for p in [C.AG / 'o_rev_event_trigger' / 'data' / 'ann_news.csv', C.SCR / 'extras' / 'rev_hist.csv.gz',
                                                           C.AG / 'o_sbl_foreign_level' / 'fo_panel.parquet',
                                                           C.AG / 'o_capmatched_selection' / 'raw' / 'agents' / 'tw50_members.json',
                                                           C.AG / 'o_capmatched_selection' / 'data' / 'membership_pre2015.json']}
C.jdump(M, C.RAW / 'manifest.json')
print(M['clones'])
