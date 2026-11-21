#!/usr/bin/env python3
"""S87 事前登記：寫 PREREG.json 與 sha256（在任何候選格的訓練／驗證報酬計算之前；只看過 data_check.json 的驗收與機械統計（檔數、換手次數，不含報酬）
與乾跑恆等結果）。python3 -I -B prereg_write.py"""
import json, hashlib, datetime
from pathlib import Path
H = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (H / 'PREREG.json').exists() and not (H / 'FROZEN.json').exists() and not (H / 'results_train.json').exists(), '已登記過，拒絕覆寫'
dc = json.loads((H / 'data_check.json').read_text())
assert dc['A0']['ok'] and dc['A2']['ok']
import sys
sys.path.insert(0, str(H))
import s87lib as L
PRE = {
 'id': 'S87', 'written_at': datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
 'title': '講義單元 02「風格 ETF 順風檢查」做成規則，放在現行策略的 0050 部位（底倉＋閒置資金）：訓練→凍結→驗證一次',
 'source': 'course_notes/README.md 單元 02：「用風格 ETF 兩個數字分順風／中性／逆風：自己 12 個月含息報酬、同期大盤」；單元 03：類股同樣檢查，但「不拿來擇時」。',
 'why_untested': 'S25 用自建多空因子測過因子動能（描述）與「營收動能因子逆風就減碼 SURGE」（不通過）；S66 比過 17 檔 ETF 的靜態持有與 70/30 固定組合；'
                 'S30 測過低 β 股票籃子當底倉；S54 測過空位留現金／大盤濾網；S85 測過四套第二策略放閒置資金。'
                 '「照講義的兩個數字，每月把 0050 部位換到順風的風格 ETF」沒有做過。',
 'hypothesis': '若台股存在可交易的風格延續（講義的因子動能），底倉與閒置資金放在順風風格 ETF 應比一律放 0050 多賺；尤其 SURGE 名單短、閒置資金多的月份，這部分錢占六到七成。',
 'what_failure_looks_like': '成對差落在 ±1 個百分點內且池／回看月數之間正負號不一致（＝沒有可量到的東西）；或一致為負（順風多半是高股息在大盤弱時相對抗跌，之後大盤由權值股帶頭反彈時被甩開，加上換倉成本）。',
 'definitions': {
  '順風': 'ETF 過去 L 個月含息報酬 R_etf > 0 且 > 同期 0050 含息報酬（官方日 K＋TWT49U 除息自建；月底對月底）。',
  '決定與執行': '每月最後一個交易日收盤決定（只用當日以前資料），下一個交易日開盤換倉。',
  '合格 ETF': '在池內、決定日有收盤價、L 個月前的月底已有資料（上市滿 L 個月）、決定日前 60 個交易日平均成交值 ≥ 2,000 萬（至少 40 天有資料）。',
  '池 S（風格）': L.STYLE, '池 SX（風格＋產業／主題）': dict(L.STYLE, **L.SECTOR),
  '池的界定': '只看名稱、在算任何報酬前寫死：台股、上市（上櫃 ETF 缺除息資料不收）、非槓桿反向、非債券、非海外；市值型複製品（006208、00692、00850 等）不算風格、不收。含已下市者。',
  '規則 TOP1': '順風 ETF 中 R_etf 最高的一檔；沒有順風 → 0050。', '規則 EW': '所有順風 ETF 等權（每月拉回等權）；沒有順風 → 0050。',
  '範圍 W100／W50': '0050 部位全部照規則／一半照規則、一半固定留 0050（每月拉回 50／50）。',
 },
 'grid': dict(n_cells=len(L.CELLS), cells=[c['id'] for c in L.CELLS], dims=dict(pool=list(L.POOLS), look=list(L.LOOKS), rule=list(L.RULES), scope=list(L.SCOPES)),
              fixed=dict(liq_min=L.LIQ_MIN, liq_win=L.LIQ_WIN, liq_minobs=L.LIQ_MINOBS, xslip_non0050_per_side=L.XSL)),
 'engine': {
  'chain': 's87lib → s20lib → s17lib → s10lib → cslib（全部不改）；引擎函式 s10lib.simulate_s10 直接呼叫',
  'flags': dc['flags'],
  'wrapper': '引擎把底倉＋空位當同一個 0050 部位（價格 P.E50／P.E50O）。本題用 P 的淺複本把這兩個陣列換成「載具指數」，其餘不動（同 S30 base_swap 的做法）。'
             'ETF 之間換倉成本直接扣在載具指數：每邊手續費 0.0399%＋半跳動點，賣方另加 0.1% 證交稅，非 0050 的 ETF 每邊另加 0.1% 滑價。'
             '策略買賣股票時進出載具的成本照引擎原本的 0050 口徑（手續費＋0.1% 稅＋0050 的半跳動點），不另加。',
  'fixed': '選股、進出場（E0）、W10_70、嚴成本、股池、5% 成交值上限等全部不動。',
  'acceptance_done_before_prereg': dict(A0=dc['A0'], A1_0050自建含息對引擎=dc['A1'], A2_恆等=dc['A2'], official_file_pre_switch=dc['official_file_pre_switch'],
                                        official_file_now=dc['official_file_now'],
                                        note='正式結果檔在本題執行途中由工程切換為修正版（多開 REDUCE_FIX、0050 含息改自建；2019 起 48.86%）。本題依指示只開三個正式開關、不開 REDUCE_FIX，'
                                             '現行標準路徑＝48.76%（與切換前正式檔逐位相同）；與切換後正式檔差 −0.11 個百分點，來源即那兩項修正。候選與現行在同一引擎下比較，差值不受影響。'),
  'mech_train（只有檔數與換手，不含報酬）': dc['mech_train'], 'eligible_count_train_L12': dc['eligible_count_train_L12'],
  'dry_run': 'train.py --dry、validate.py --dry（16 格都換成現行）全部差 0',
 },
 'paths': '20 條＝起始資金 {60, 80, 100, 150, 200} 萬 × 訊號位移 {0, 1, 2, 3}（載具的月底決定不隨位移變）',
 'windows': {
  'train': '連續模擬，量 2016-01-30 → 2021-12-30（訊號 2016-02～2021-12）的年化與窗內最大回撤；載具換倉只到 2021-12 第一個交易日',
  'valid': '重新起跑：只保留營收月份 ≥ 2021-12 的訊號（2022 前全在 0050；載具第一次換倉＝2022 第一個交易日，依 2021-12-30 的決定），起始資金換算使 2021-12-30 淨值約 60～200 萬；量 2021-12-30 → 2026-10-22',
  'bear_segments_V4': '2022、2024-07～2025-04 的段內最大跌幅（s17lib.seg_dd）',
 },
 'step1_train_selection': {
  'rule': '16 格中取訓練窗 20 條路徑年化中位數最高者（同分依 ≥ +0.25 路徑數、再依格子順序）。若「其中位數 − 現行中位數 ≥ +1 個百分點」且「回撤中位數不比現行深 2 個百分點以上」→ 選它；否則選「不改」。',
  'frozen_for_validation': '不論選它或「不改」，都凍結中位數最高的格跑驗證期一次；選「不改」時驗證只作描述，總判定一律不通過。',
  'T0（S25 標準）': '凍結格對現行的成對差（逐條路徑）中位數 ≥ +1 個百分點，且差 ≥ +0.25 個百分點的路徑 ≥ 14/20。',
 },
 'step2_validation（凍結後只跑一次）': {
  'V1': '驗證窗年化差（凍結格 − 現行）> 0 的路徑 ≥ 15/20', 'V2': '驗證窗年化差中位數 ≥ +2 個百分點',
  'V3': '兩邊主跑（100 萬換算、位移 0）各自最賺 5 筆已平倉交易的聯集，整段持有期兩邊都禁買，重跑 20 條路徑後年化差中位數 > 0',
  'V4': '2022、2024-07～2025-04 兩段的段內最大跌幅差中位數都 ≥ −2 個百分點',
  'overall': '訓練選擇不是「不改」且 T0、V1～V4 全過 → 建議；否則不改。',
 },
 'descriptive_only（登記但不作判定）': [
  '其餘 15 格在驗證期的年化／回撤成對差', '凍結格載具指數本身對 0050 的年化、回撤、換手與成本；每月持有哪幾檔',
  '凍結格在「現行輸 0050 的月份」與「0050 下跌的月份」的月報酬差', '凍結格全期連續 20 條路徑（2019 起、全期、回撤）',
  'ETF 層級：順風 vs 非順風，下一個月對 0050 的超額（訓練／驗證／全期；池 SX、回看 12）',
 ],
 'predictions（事前）': {
  'P0': '約 90% 機率「不改」。理由：訓練期合格池中位數只有 1 檔（0056），順風月份只占三到四成；S25 的因子動能 t 1.55、S66 風格 ETF 長期輸 0050；每年換倉成本約 0.4～1.0%（占載具）。',
  'P1': '訓練期最好一格的成對差中位數落在 −0.5～+1.5 個百分點；16 格中位數的中位數落在 −1.5～+0.3。',
  'P2': '驗證期凍結格 V2（≥ +2）不過；年化差中位數落在 −3～+0.5（2023～2026 權值股集中行情，順風的高股息 ETF 之後多半被 0050 甩開）；整體通過機率 < 3%。',
  'P3': '回撤：2022 段可能淺 0～2 個百分點（高股息抗跌），2024-07 段差不多。',
 },
 'caveat_validation_not_fresh': '現行策略與 0050、各 ETF 的 2022～2026 靜態成績（S66）早已看過；研究員知道 2024 起高股息 ETF 大輸 0050。本題的輪動規則本身沒有在任何期間算過。',
 'data_snooping': {'added': 16, 'note': '16 格，N +16。'},
 'code_sha256_at_write': {n: sha(H / n) for n in ('s87lib.py', 'etf_load.py', 'prep.py', 'train.py', 'validate.py')},
 'data_check_sha256': sha(H / 'data_check.json'),
}
txt = json.dumps(PRE, ensure_ascii=False, indent=1)
(H / 'PREREG.json').write_text(txt)
psha = hashlib.sha256(txt.encode()).hexdigest()
(H / 'PREREG.sha256').write_text(f'{psha}  PREREG.json\n')
print('PREREG', psha)
