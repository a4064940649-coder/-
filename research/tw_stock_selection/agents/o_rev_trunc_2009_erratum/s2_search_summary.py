#!/usr/bin/env python3
"""第 2 步：把 raw/search_log.jsonl（GitHub 搜尋原始紀錄）與逐 repo 檢查結果彙整成 search_log.json 與 raw/manifest.json。
本步不下載任何營收值；所有候選 repo 都先 add_repo（read），再用 --depth 1 --filter=blob:none --no-checkout clone 到 scratchpad/o15_rev/clone/，
只看檔案清單、必要時讀 1 個 blob；檢查完即刪除。用法：python3 -I -B s2_search_summary.py"""
import json, datetime, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
Q = [json.loads(l) for l in open(HERE / 'raw/search_log.jsonl', encoding='utf-8')]
repos_hit = sorted({i['repo'] for q in Q for i in q['items']})
INSPECT = [
 dict(repo='jkzhong/taiwan_stock_info', commit='c9aa08a545512049b3b83149e02480298e67d807', commit_date='2014-01-29T00:05:21+08:00', method='blob-less clone，看檔案清單',
      finding='2014 年課堂專案：程式庫、作業、result.7z；沒有月營收表', usable=False),
 dict(repo='litefunc/mops', commit='f1b928ba6896bd28bc41569efa0a1eed252c4651', commit_date='2024-12-21T21:34:45+08:00', method='blob-less clone，看檔案清單',
      finding='只有季財報（income／balance／incomedu csvfiles），沒有月營收', usable=False),
 dict(repo='angel870326/monthly-revenue-forecasting', commit='d63155798f4d17cefeb2dab01161abb08be1164a', commit_date='2023-07-17T17:34:12+08:00', method='blob-less clone，看檔案清單',
      finding='只有 notebook，沒有資料檔', usable=False),
 dict(repo='henry753951/finance', commit='e2c22b5fb8f992e70b0c2ca691f2932ea54b6afe', commit_date='2024-01-03T06:42:48+08:00', method='blob-less clone，讀 crawl/data/雷凌.csv 1 個 blob',
      finding='crawl/data/<名稱>.csv 是 Google 搜尋熱度（2023-10 起日資料），data/roa_roe 是年度 ROA／ROE；沒有月營收', usable=False),
 dict(repo='hawan0228/fintech_ai', commit='92037ea2d7b97eb18eccd8a35b95b115c859a89b', commit_date='2026-06-24T23:08:36+08:00', method='depth-1 blob-less clone，看檔案清單',
      finding='113 個 FinMind 逐檔營收檔，都是現存股；19 檔與 TRUNC 股（3534、5854、2473、3367、2403、2463、1716、6119）0 檔', usable=False),
 dict(repo='ChiJiun/tuc-6274-financial-forecast', commit='83701d22c08f6643e16fdcd4abf297a130c493d1', commit_date='2026-10-11T02:39:56+08:00', method='depth-1 blob-less clone，讀 manifest 與 t21sc03_100_6_0.html',
      finding='有 98～101 年 t21sc03 頁，但是上櫃（otc）頁；manifest 的 captured_at_utc 是 2026-10-10（重新產生的頁面）；100_6 頁 1,074 個代碼，3534／5854／2473／3367／2330 都不在內', usable=False,
      snapshot_note='快照日 2026-10 晚於全部目標股下市日，就算有值也不能算進下市股覆蓋'),
 dict(repo='wenchiehlee/python-actions.finmind', commit='19eda174b815ba17b1fb36c3c0b3340db9623459', commit_date='2026-11-05T22:36:04+00:00', method='depth-1 blob-less clone，看檔案清單',
      finding='140 個 FinMind 逐檔營收檔，沒有任何目標代碼', usable=False),
 dict(repo='koreal6803/finlab_course_20191218', commit='6c5b03766963cb0e3b5262ae8501c993ca9fd5a3', commit_date='2019-12-18T18:07:25+08:00', method='depth-1 blob-less clone，看檔案清單',
      finding='只有 3 個 pickle（adj_close_price、close_price、fundamental_data）；pickle 載入會執行程式碼，屬不可信格式，不開啟；檔名顯示不是月營收', usable=False),
 dict(repo='taiwanan/TW_Stock_Exchange_Crawler', commit='（o_rev_survivorship/raw 已有複本，不重抓）', commit_date='2018-06～2019-09', method='沿用 o_rev_survivorship 紀錄',
      finding='已知只有 2015-01 起的逐檔月營收，不覆蓋 2009-12～2012-12；記錄為不重複', usable=False),
]
out = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
           n_queries=len(Q), n_code_queries=sum(q['kind'] == 'code' for q in Q), n_repo_queries=sum(q['kind'] == 'repo' for q in Q),
           n_distinct_repos_in_results=len(repos_hit), n_repos_inspected=len(INSPECT), n_repos_cloned=sum(1 for x in INSPECT if 'clone' in x['method']),
           queries=[dict(ts=q['ts'], kind=q['kind'], q=q['q'], total=q['total'], err=q['err'], n_items=len(q['items']),
                         repos=sorted({i['repo'] for i in q['items']})[:20]) for q in Q],
           inspected=INSPECT,
           conclusion=f'本次 {len(Q)} 次搜尋、{len(INSPECT)} 個 repo 內找不到含 2009-12～2012-12 已下市上市公司月營收的公開封存；通過一致性的補值 0 筆，下市股覆蓋 0%。',
           not_used=['MOPS／mopsov、twse.com.tw、FinMind API、archive.org 等非 GitHub 來源（題目只允許 GitHub 公開 repo；o_rev_survivorship 已記錄 proxy 403）'])
(HERE / 'search_log.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
man = dict(utc=out['utc'], note='本題沒有任何候選封存通過篩選，因此 raw/ 只有搜尋原始紀錄，沒有抽出的營收檔。',
           files={'raw/search_log.jsonl': dict(sha256=hashlib.sha256((HERE / 'raw/search_log.jsonl').read_bytes()).hexdigest(), use='GitHub 搜尋 API 原始回應的中繼資料（不可信資料）')},
           sources_extracted=[])
(HERE / 'raw/manifest.json').write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding='utf-8')
print(out['conclusion'], out['n_code_queries'], out['n_repo_queries'], out['n_distinct_repos_in_results'])
