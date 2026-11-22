#!/usr/bin/env python3
"""寫 PREREG.json＋sha256＋code.sha256（訓練期模擬之前）。"""
import json, hashlib, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (HERE / 'PREREG.json').exists()
td = json.loads((HERE / 'step0.json').read_text())['train_desc']
P = {
 'id': 'S96', 'title': '官方本益比／淨值比／殖利率和營收驚奇（SURGE）交易賺賠的關係', 'written_at': datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
 'baseline': '現行正式策略 E0｜W10_70（五個開關全開）：2019 起 48.86%、全期 34.46%、381 筆已平倉（389 筆含未平倉）；step0.py 已重現，與 xq_scripts/site/navs_official.csv.gz 逐日差 0。',
 'data': 'raw/tsd_extra/per_twse_early＋per_twse（官方每日本益比、殖利率、淨值比；四捨五入 2 位；收盤價欄不用）。抽查與涵蓋率見 data_check.json、spot_check.csv。',
 'timing': '訊號日收盤後用當日（含）以前的官方估值；次一交易日開盤成交（引擎原規則）。出場規則完全不動。',
 'cells': {'n': 10, 'list': ['NOPER|Z', 'NOPER|F', 'PBLOW|Z', 'PBLOW|F', 'NOPER_PBLOW|Z', 'NOPER_PBLOW|F', 'P3LOW|F', 'PE30|F', 'PBHI|F', 'YLD0|F'],
           'rules': {'NOPER': '官方本益比為空（虧損或無法計算）→ 不買', 'PBLOW': '淨值比在同日股池最低 1/3 → 不買', 'NOPER_PBLOW': '上兩條任一成立 → 不買',
                     'P3LOW': '本益比在自己過去 756 個資料日（至少 250 筆）最低 1/3 → 不買；虧損或歷史不足照買', 'PE30': '對照（傳統方向）：本益比 > 30 → 不買',
                     'PBHI': '對照（傳統方向）：淨值比在同日股池最高 20% → 不買', 'YLD0': '殖利率 0 → 不買'},
           'modes': {'Z': '被擋名額不遞補，錢照引擎放 0050', 'F': '由季線上名單第 11～20 名中近 20 日漲幅 > 10% 且通過同一條件者，依 SURGE 高低遞補'}, 'code': 's96cells.py'},
 'protocol': {'train': '全期連續模擬，取訓練窗 2016-01-29 → 2021-12-30（訊號日 2016-02～2021-12）；20 條路徑＝起始資金 60／80／100／150／200 萬 × 訊號位移 +0～+3 日；嚴成本。',
              'selection': '訓練窗年化中位數最高的格（同分看 ≥ +0.25 個百分點的路徑數）；要「選用」須同時：中位數 ≥ 現行 + 1.0 個百分點、最大回撤中位數不比現行深超過 2 個百分點、2016–2018 與 2019–2021 兩半成對差中位數皆 > 0。否則 selected＝「不改」。',
              'T0': '最高格對現行的成對差中位數 ≥ +1.0 個百分點，且 ≥ +0.25 個百分點的路徑 ≥ 14/20。',
              'freeze': '寫 FROZEN.json＋sha256；即使「不改」，仍以訓練中位數最高的格跑驗證一次，只作描述。',
              'validation': '重新起跑（只留營收月份 ≥ 2021-12 的進場訊號；起始資金按現行 2021-12-30 淨值縮放）× 20 條路徑；驗證窗 2021-12-30 → 2026-10-22；只跑一次（results.json 已存在就拒絕）。',
              'verdict（照 S25）': {'V1': '成對差 > 0 的路徑 ≥ 15/20', 'V2': '成對差中位數 ≥ +2.0 個百分點', 'V3': '禁買「現行與凍結格各自最賺 5 筆」聯集後，成對差中位數 > 0',
                                'V4': '2022 全年、2024-07～2025-04 兩段空頭的段內最大回撤，成對差中位數皆 ≥ −2.0 個百分點',
                                'overall': 'selected≠不改 且 T0 且 V1～V4 全過才算通過；其餘一律不通過、不改正式策略。'},
              'descriptive_only': ['凍結格在驗證段擋掉的大贏家（現行驗證段報酬率前 5%）與大輸家（後 5%）清單', '隨機剔除同檔數 20 次（seed 96；F 模式照樣遞補）', '成本加倍', '全期連續 2019 起年化',
                                   '2008–2026 連續帳戶（S75）632 筆中 2012 年後有估值資料者的大贏家估值分佈（desc632.py；不登記、不判定）']},
 'train_description_seen_before_prereg': {'note': '登記前只看了現行訓練期 186 筆已平倉交易的分組描述（step0.json），沒有跑過任何加估值條件的模擬（train.py --dry 只印剔除／遞補檔數，不印報酬）。', 'train_desc_groups': td['groups'], 'spearman_ret': td['spearman_ret']},
 'predictions（事前寫下，可被否證）': [
   '訓練期描述：虧損或無本益比的 31 筆勝率 35%、平均 +0.8%（全體 186 筆平均約 +9%）；淨值比股池最低 1/3 的 45 筆勝率 36%、平均 +0.9% → NOPER、PBLOW、NOPER_PBLOW 在訓練期小幅為正（成對中位數 0～+2 個百分點），但達不到 T0。',
   '本益比 > 30 的 53 筆平均 +18.5%、9 筆大贏家占 4 筆、最賺 10 筆占 6 筆 → PE30|F 訓練期明顯為負；PBHI|F 也為負。',
   '驗證期：S70 已知「帳面虧損不買」會擋掉 2408 這類翻身大贏家 → NOPER 類在驗證期為負或不顯著；10 格沒有一格會通過，凍結格與隨機剔除沒有分別。'],
 'dedup': 'S29（QMV／GARP、E/P 含虧損股，粗資料）、S70（最近一季營業虧損不買）、S88（財報品質）、w1_pe_river_low、w4_buffett_roe_pe_screen、w4_roe_yield_compounder、w4_sun_7pct、w4_peg（獨立選股法，非 SURGE 過濾）。本題新增：官方逐日本益比／淨值比／殖利率、同日股池百分位、自身 3 年本益比百分位。',
 'data_snooping': {'added': 10, 'note': '10 格全部計入；N 以研究中心彙總為準。'},
 'resources': '單行程、峰值記憶體 ≤ 3GB、python3 -I -B、不連外網；暫存 scratchpad/s96/（用完刪）。',
}
(HERE / 'PREREG.json').write_text(json.dumps(P, ensure_ascii=False, indent=1))
(HERE / 'PREREG.sha256').write_text(f"{sha(HERE / 'PREREG.json')}  PREREG.json\n")
(HERE / 'code.sha256').write_text(''.join(f"{sha(HERE / f)}  {f}\n" for f in ('s96lib.py', 's96cells.py', 'train.py', 'validate.py')) + f"{sha(HERE.parent / 's90_institutional_flow' / 's90lib.py')}  ../s90_institutional_flow/s90lib.py\n")
print((HERE / 'PREREG.sha256').read_text(), (HERE / 'code.sha256').read_text())
