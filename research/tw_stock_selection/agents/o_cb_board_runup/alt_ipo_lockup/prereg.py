# -*- coding: utf-8 -*-
"""凍結：把最終規格寫成 PREREG（sha256、時間）存進 results.json。之後才可以跑 oos.py。"""
import os, sys, json, hashlib, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
HERE = Path(__file__).resolve().parent
IT = json.loads((HERE / 'iterations.json').read_text(encoding='utf-8'))
assert IT['chosen'] == 'A5'
sha = lambda p: hashlib.sha256((HERE / p).read_bytes()).hexdigest()
PREREG = f"""PREREG｜alt_ipo_lockup｜新上市股的內部人鎖定期供給（替代題；主題 CB 董事會推升在閘門 (a) 關閉）
凍結版本：A5（依事前選版規則：受影響 ≥ 30 的版本中挑選期配對 t 最大；A5 本身是看過探索「初次上市 3～4 年殘差負」後追加的資料導向版本，贏家詛咒風險高）。
規則：主策略 v2 正式版（52 週比百分位 0.5＋近 3 月營收年增率百分位 0.5、前 300 流動性池、排除金融、月換股 10 檔、M2 成交、開盤漲停遞補、gapfix 窗）不變，
  唯一修改：換股訊號日 d，若候選股為「初次上市」（面板第一個交易日 ≥ 2004-06-01、非上櫃轉上市、非控股轉換、非 DR）且上市年資 ∈ [2, 4) 年
  （年資＝(d − 面板第一個交易日)/365.25），且本期之前未持有，則跳過並由下一名遞補；已持有者若仍在前 10 名則續抱。
資料：listings.csv sha256 {sha('listings.csv')}；iterations.json sha256 {sha('iterations.json')}。
比較：原版主策略（同引擎、同成本）、0050 含息、同池等權。成本：一般（手續費 0.0855%＋稅 0.3%＋滑價 0.1%／邊），壓力（每邊 0.5%）。
期間：挑選 2006-04～2019-04（已看過）；驗證 2019-05～2026-10，內分 2019-05～2023-12 與保留段 2024-01～2026-10；跨循環 2006-04～2026-10。驗證期只跑一次。
另在 o_infra_v3_rebuild 的 V3 基礎設施（下市股價格 P1、營收 R1、併購現金出清 X1、處置 Dhi）上同規格重跑一次。
判定（全部通過才待觀察，否則不可用）：
  (1) 挑選期與驗證期，對主策略的月配對差（版本 − 原版，訊號窗淨報酬）都 > 0；
  (2) 跨循環年化 ≥ 原版；
  (3) 剔除配對差貢獻最大的 3 檔後，驗證期配對差仍 > 0；
  (4) 安慰劑（同規則改套上櫃轉上市股 2～4 年）配對差的絕對值 < 主版本配對差的一半（驗證期）；
  (5) 壓力成本下驗證期配對差仍 > 0。
檢定總數揭露：探索 P1～P4 約 30 個子群組檢定；策略層 6 版 × n10。驗證期污染：2019～2026 有台積電／AI 行情；2019～2023 新上市後下市的股票幾乎不在面板（183 檔只有 1 檔），V3 也沒補到，年輕股的驗證期報酬偏樂觀（對本規則＝避開年輕股，偏誤方向是讓避開看起來較差）。
可成交性：規則只改選股、不改檔數與成交時點，30 萬元資金單筆不超過當日成交值 5% 的檢查沿用主策略（前 300 流動性池）。"""
R = {'PREREG': PREREG, 'PREREG_sha256': hashlib.sha256(PREREG.encode('utf-8')).hexdigest(),
     'PREREG_time_utc': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
     'PREREG_params': dict(version='A5', lo=2, hi=4, ipo_only=True, placebo='transfer 2-4y')}
f = HERE / 'results.json'
assert not f.exists() or 'PREREG' not in json.loads(f.read_text(encoding='utf-8')), 'PREREG 已存在，不能覆寫'
f.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding='utf-8')
print(R['PREREG_sha256'], R['PREREG_time_utc'])
