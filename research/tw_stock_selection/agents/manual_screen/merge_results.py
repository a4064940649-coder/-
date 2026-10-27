#!/usr/bin/env python3
"""把 results_bt.json、results_today.json、results_pool.json 併入 results.json（保留 PREREG 不動）。"""
import json, hashlib
from pathlib import Path
H = Path(__file__).resolve().parent
R = json.loads((H / 'results.json').read_text())
assert hashlib.sha256(R['PREREG'].encode('utf-8')).hexdigest() == R['PREREG_sha256']
R['backtest'] = json.loads((H / 'results_bt.json').read_text())
R['signal_2026_10_12'] = json.loads((H / 'results_today.json').read_text())
R['pool_threshold'] = json.loads((H / 'results_pool.json').read_text())
R['web_sources'] = json.loads((H / 'raw' / 'web_search_log.json').read_text())
R['implementation_notes'] = [
 '第一次執行（PREREG 之後）發現實作錯誤：252 日原始最高價窗內不足 200 筆時 np.fmax(NaN, 收盤) 回傳收盤，使上市未滿 200 天的新股 D=0、被當成在一年高點。'
 '已改成 NaN（符合 PREREG『窗內有值 ≥ 200 天』）。錯誤版的數字（例如 M1 2006-04～2013-12 年化 14.1%、最大回撤 −55.7%）作廢；主策略數字不受影響。',
 '主策略 n10 本引擎重建與 independent_audit 相同：2006-04～2020-12 18.3%（t 1.87），2021-03～2026-10-12 56.7%（t 1.95），2024～2026 83.4%（t 1.27）。',
 '2026-10-12 主策略 ex-fin 本引擎重算名單與 main_picks_today.py --ex-fin 10 檔完全相同。',
 '網站頁面本身無法開啟（goodinfo.tw 等主機被本環境網路政策拒絕），欄位名稱只依搜尋引擎索引的頁面標題與摘要，未在頁面上逐一點選核對。']
(H / 'results.json').write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print('merged', list(R))
