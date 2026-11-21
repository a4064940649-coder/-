#!/usr/bin/env python3
"""S92 事前登記：寫 PREREG.json＋PREREG.sha256（含登記時程式碼 sha256）。已存在就拒絕。python3 -I -B prereg_write.py"""
import json, hashlib, datetime
from pathlib import Path
H = Path(__file__).resolve().parent
assert not (H / 'PREREG.json').exists(), 'PREREG.json 已存在'
dc = json.loads((H / 'data_check.json').read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
G = dc['train_returns_by_group']
PR = {
 'id': 'S92', 'written_at': datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
 'question': '補完 S88 的兩個缺口：(1) 應收帳款、存貨成長快於營收的股票要不要擋；(2) S88 的失敗是不是因為財報時點用估的（改用實際公告日重跑）',
 'engine': '正式管線（official_nav/run_official.py 的 E0｜W10_70 寫法）五個開關全開；s10lib.simulate_s10 原樣呼叫；只改訊號列買進名單；嚴成本、空位 0050',
 'acceptance': dc['acceptance'],
 'data': {'fs_xbrl': 'raw/tsd_extra/fs_xbrl（元；一家一季一列）', 'filing_dates': 'raw/tsd_extra/filing_dates（一份文件一列；取同檔同季所有列上傳時間最小值當首次可得日）',
          's88_inputs': 'o_pead_margin_surprise_low_attention 的 fin_long／fin_raw／avail_dates（同 S88）',
          'pit_fd': '季可用 ⇔ 首次上傳日期 < 訊號日（嚴格小於）；公告日缺值的季退回 S88 估法；由季底 < 訊號日的最新一季往回最多 5 季',
          'pit_s88': '法定期限 < 訊號日 且（市場可得日缺值或 < 訊號日），往回 1 季（S88 原樣）',
          'features': 's92lib.featx 的文件字串；只判一般業（ci）；缺值不擋'},
 'cells': {'new（10 格，參與選格）': list(__import__('s92rules').NEW_CELLS) if False else ['INV20|Z', 'INV20|F', 'INV30|Z', 'INV30|F', 'AR30|Z', 'AR30|F', 'BOTH10|Z', 'EITHER30|Z', 'DIO20|Z', 'DIO20|F'],
           's88_timing（2 格，不參與選格）': ['R09FD|Z', 'R09FD|F'],
           'reference（不算格、不參與選格）': ['R09S88|Z（S88 凍結格原時點估法，在五開關新口徑下重跑，只用來和 R09FD|Z 對照）'],
           'defs': 'INV20／INV30＝存貨年增率 − 單季營收年增率 > 0.20／0.30；AR30＝應收年增率 − 營收年增率 > 0.30；BOTH10＝兩者都 > 0.10；EITHER30＝任一 > 0.30；DIO20＝存貨周轉天數年增 > 20 天；R09＝應計比率在股池最高 20% 或近四季營業現金流為負。Z＝不遞補放 0050；F＝第 11～20 名遞補',
           'total': '12 格（≤ 12）'},
 'why_these（只看訓練期分組）': '訓練期 186 筆：存貨成長比營收快越多、平均報酬越低（>20 個百分點 39 筆平均 +2.1%、沒有大贏家、不在損益前 10；>30 個百分點 24 筆 +0.2%）；存貨周轉天數多 20 天以上 23 筆平均 −1.0%、沒有大贏家。應收方向弱（高三分位平均 +11.3% 與低組相同、含 5 筆損益前 10），只有 >30 個百分點 11 筆沒有大贏家，仍依派工列入。應收周轉天數變化沒有可用的門檻（>20 天只有 4 筆），不設格，只作描述。',
 'step0_seen_before_prereg': {'note': '登記前只看了：資料涵蓋與時點統計（不含報酬）、訓練期（訊號 2016-02～2021-12）現行交易的報酬分組；驗證期報酬完全沒算。train.py 與 validate.py 各做過一次乾跑（所有格換成恆等，結果全為 0，不含任何條件的效果）。',
                              'train_returns_by_group': G, 'timing': dc['timing']},
 'step1_train_selection': {'window': '訓練窗＝2016-02-01 前最後交易日 → 2022-01-01 前最後交易日，連續模擬，20 條路徑（60／80／100／150／200 萬 × 位移 0～3）',
                           'rule': '只在 10 個新格中取訓練窗 20 條路徑年化中位數最高者（同分依 ≥ +0.25 路徑數、再依格子順序）。同時符合 (a) 中位數 − 現行 ≥ +1 個百分點、(b) 回撤中位數不比現行深 2 個百分點以上、(c) 2016～2018 與 2019～2021 兩半成對差中位數都 > 0 → 選它；否則「不改」。',
                           'frozen_for_validation': '不論選它或「不改」，都凍結中位數最高的新格跑驗證期一次；「不改」時只作描述，總判定不通過。',
                           'T0（S25 標準）': '成對差中位數 ≥ +1 個百分點，且差 ≥ +0.25 的路徑 ≥ 14/20。'},
 'step2_validation（凍結後只跑一次）': {'window': '重新起跑（只保留營收月份 ≥ 2021-12 的訊號，起始資金縮放），驗證窗 2021-12-30 → 2026-10-22，20 條路徑',
                              'V1': '年化差 > 0 的路徑 ≥ 15/20', 'V2': '年化差中位數 ≥ +2 個百分點', 'V3': '現行與該格各自最賺 5 筆的聯集兩邊都禁買後，年化差中位數 > 0',
                              'V4': '2022、2024-07～2025-04 兩段空頭段內最大跌幅差中位數都 ≥ −2 個百分點', 'overall': '訓練選擇不是「不改」且 T0、V1～V4 全過 → 建議加；否則不加。'},
 's88_timing_check': 'R09FD|Z、R09FD|F 固定跑（規則已由 S88 凍結）。判讀：R09FD|Z 的 V1～V4 全過才算「時點估計是 S88 失敗的原因」；否則不是。另報 R09FD|Z − R09S88|Z 的成對差。',
 'descriptive_only': ['其餘新格驗證窗 20 條路徑年化差（不做 V3）', '現行 389 筆依四個指標三分位（切點取自訓練期）、財報新舊分組（全期／訓練／驗證）', '每格驗證段擋掉的大贏家（前 5%）與大輸家（後 5%）清單', '凍結格與 R09FD|Z 的隨機剔除對照 20 次（seed 92）', '成本加倍'],
 'predictions（事前）': {'P0': '很可能「不改」或驗證不過（約 85%）。', 'P1': '訓練期存貨類（INV20、INV30、DIO20）的 Z 版比現行好 0～+2 個百分點；應收類接近 0 或為負。', 'P2': 'F 版不比 Z 版好。',
                     'P3': '時點檢驗：R09FD|Z 與 R09S88|Z 幾乎一樣（389 筆中旗標只翻 7 筆），驗證期仍為負 → 時點不是 S88 失敗的原因。', 'P4': '驗證期凍結格會擋到至少 1 筆大贏家（景氣循環股翻身時存貨天數偏高）。'},
 'data_snooping': {'added': 12},
 'caveats': ['現行策略 2022～2026 的大贏家名單在先前題目已看過（如 2408）；本題的應收／存貨特徵在任何期間都沒算過。', 'fs_xbrl 可能是更正後數字；公告日若被更正檔覆蓋會偏晚（保守）。', '門檻是看訓練期交易分組挑的，訓練期成績因此偏樂觀。'],
}
PR['code_sha256_at_write'] = {n: sha(H / n) for n in ('s92lib.py', 's92rules.py', 'train.py', 'validate.py', 'prep.py')}
txt = json.dumps(PR, ensure_ascii=False, indent=1, default=str)
(H / 'PREREG.json').write_text(txt)
h = hashlib.sha256(txt.encode()).hexdigest()
(H / 'PREREG.sha256').write_text(f'{h}  PREREG.json\n')
print(h)
