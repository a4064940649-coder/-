#!/usr/bin/env python3
"""S88 事前登記：寫 PREREG.json＋PREREG.sha256（含登記時程式碼 sha256）。已存在就拒絕。python3 -I -B prereg_write.py"""
import json, hashlib, datetime
from pathlib import Path
H = Path(__file__).resolve().parent
assert not (H / 'PREREG.json').exists(), 'PREREG.json 已存在'
dc = json.loads((H / 'data_check.json').read_text())
PEAD = H.parent / 'o_pead_margin_surprise_low_attention'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
PR = {
 'id': 'S88', 'written_at': datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
 'question': '營收暴增但財報品質差的股票是不是大輸家來源？能不能用財報把壞交易濾掉（不改正式策略，只回答要不要加）',
 'engine': '正式引擎鏈 s70lib→s47lib→…→s10lib.simulate_s10（不改）；cslib.EXRIGHT_FIX、DEADLINE_FIX、INS15_MASK 全開；REDUCE_FIX、E50_SOURCE 維持預設關閉；F1＋E0、W10_70、嚴成本、空位 0050',
 'acceptance': dc['acceptance'],
 'acceptance_note': '本題重現的是三開關口徑的標準路徑 2019 起 48.756%（＝official_nav/archive_2026-11-21_pre_v2/res_official.json）。執行當下 official_nav/out/res_official.json 已由工程切到 v2（多開 REDUCE_FIX＋0050 官方含息，48.862%）；本題依派工指示不自行開 REDUCE_FIX，所以沒有對 48.862% 做重現。',
 'data': {'fin_long.parquet': sha(PEAD / 'data' / 'fin_long.parquet'), 'fin_raw.parquet': sha(PEAD / 'replicate_o10' / 'data' / 'fin_raw.parquet'), 'avail_dates.parquet': sha(PEAD / 'data' / 'avail_dates.parquet'),
          'pit': '訊號日 d 用「法定期限 < d 且（市場可得日 A 缺值或 A < d）」的最新一季，晚交或無資料往回 1 季；法定期限 Q1 5/15、Q2 8/14、Q3 11/14、年報次年 3/31；年增用同檔 kq−4',
          'missing': '應收帳款、存貨：本機無可讀資料（bs_hist 只有目錄、內容讀不到）→ 不測「應收／存貨成長 > 營收成長」，改用應計比率、營業現金流、總資產成長−營收成長三個替代'},
 'flags': {'OPI_DOWN': '最近一季營業利益（金額）< 去年同季', 'GM_DOWN': '最近一季毛利率 < 去年同季', 'LOSS': '最近一季營業利益 < 0（＝S70 C1）',
           'ACC_HI': '（近四季本期淨利 − 近四季營業現金流）÷ 最近一季總資產，在同期股池非金融公司的百分位 ≥ 0.80', 'OCF_NEG': '近四季營業現金流 < 0',
           'AG_GAP': '總資產年增率 − 最近一季營收年增率 > 0.10', 'common': '缺值一律不擋；證券、金控、銀行、保險業不判 ACC_HI、OCF_NEG、AG_GAP'},
 'rules（10 條，事前固定）': {'R01_OPI_DOWN': 'OPI_DOWN', 'R02_GM_DOWN': 'GM_DOWN', 'R03_LOSS': 'LOSS', 'R04_ACC_HI': 'ACC_HI', 'R05_OCF_NEG': 'OCF_NEG', 'R06_AG_GAP': 'AG_GAP',
                          'R07_OPI_GM_DOWN': 'OPI_DOWN 且 GM_DOWN', 'R08_LOSS_OCFNEG': 'LOSS 且 OCF_NEG', 'R09_CASH_BAD': 'ACC_HI 或 OCF_NEG',
                          'R10_ANY2': 'OPI_DOWN、GM_DOWN、LOSS、ACC_HI、OCF_NEG 五項中至少兩項'},
 'modes': {'Z': '名單（L_MA 前 10 名中 RET20 > 10%）中被規則擋到的不買，不遞補，錢照引擎放 0050', 'F': '被擋的名額由 L_MA 第 11～20 名中 RET20 > 10% 且沒被同一規則擋到者，依 SURGE 由高到低遞補（最多補到被擋檔數）；出場規則不變'},
 'cells': '10 條規則 × 2 種處理＝20 格（≤ 30）；現行 CUR 為對照。不得增減。',
 'dedup': 'S28（毛利率／營業利益率年減 > 2 或 5 個百分點且營收年增才擋）、S70（C1 營業虧損不買＝本題 R03|Z；C2 毛利／資產）、S29（ROE、含虧損 E/P）、S13（毛利率改善等特徵）、S11（季報 SUE）都已不通過。本題新增：營業利益金額年減、毛利率年減（不設門檻與保護）、現金流類（應計、營業現金流、資產膨脹）、組合，以及「遞補」處理。R03|Z 與 S70 C1 重複，仍列入以便同表比較，但不重複計入資料窺探次數。',
 'step0_seen_before_prereg': {'note': '登記前只看了：全期旗標出現率（不含報酬）與訓練期（訊號 2016-02～2021-12）現行交易的報酬分組；驗證期報酬在本題完全沒算。',
                              'train_returns_by_group': {k: v for k, v in dc['train_returns_by_group'].items() if k.endswith('=1') or k in ('ALL', 'thresholds')}},
 'step1_train_selection': {'window': '訓練窗＝2016-02-01 前最後交易日 → 2022-01-01 前最後交易日（訊號 2016-02～2021-12），連續模擬，20 條路徑（60／80／100／150／200 萬 × 位移 0～3）',
                           'rule': '20 格中取訓練窗 20 條路徑年化中位數最高者（同分依 ≥ +0.25 路徑數、再依格子順序）。若同時符合 (a) 其中位數 − 現行中位數 ≥ +1 個百分點、(b) 其回撤中位數不比現行深 2 個百分點以上、(c) 2016～2018 與 2019～2021 兩半的成對差中位數都 > 0 → 選它；否則「不改」（不往下找第二名）。',
                           'frozen_for_validation': '不論選它或「不改」，都凍結中位數最高的那一格跑驗證期一次；「不改」時驗證只作描述，總判定一律不通過。',
                           'T0（S25 標準）': '凍結格對現行的逐路徑成對差中位數 ≥ +1 個百分點，且差 ≥ +0.25 的路徑 ≥ 14/20。'},
 'step2_validation（凍結後只跑一次）': {'window': '重新起跑（只保留營收月份 ≥ 2021-12 的訊號，起始資金縮放），驗證窗 2021-12-30 → 2026-10-22，20 條路徑',
                              'V1': '年化差（凍結格 − 現行）> 0 的路徑 ≥ 15/20', 'V2': '年化差中位數 ≥ +2 個百分點',
                              'V3': '現行與凍結格（重新起跑主跑）各自最賺 5 筆已平倉交易的聯集，整段持有期兩邊都禁買，重跑 20 條路徑後年化差中位數 > 0',
                              'V4': '2022、2024-07～2025-04 兩段空頭的段內最大跌幅差中位數都 ≥ −2 個百分點',
                              'overall': '訓練選擇不是「不改」且 T0、V1～V4 全過 → 建議加；否則不加。'},
 'descriptive_only（登記但不作判定）': ['現行全部交易依旗標分組（全期／訓練／驗證）：筆數、勝率、平均、中位數、大贏家（報酬前 5%）與大輸家（後 5%）比例',
                                '每一格擋掉的現行交易：筆數、損益合計、擋到幾筆大贏家／大輸家／損益最大 10 筆（S82 教訓的檢查）',
                                '其餘 19 格驗證窗重新起跑 20 條路徑年化差（不做 V3）；全期連續 20 條路徑', '凍結格的隨機剔除對照（20 次，seed 88）；成本加倍'],
 'predictions（事前）': {'P0': '很可能「不改」或驗證不過（約 85%）。',
                     'P1': '訓練期：現金流類（R04、R05、R08、R09）的 Z 版比現行好 0～+2 個百分點（訓練期這幾組平均報酬為負、幾乎沒有大贏家）；R01、R02、R07、R10 明顯較差（−2 以下；訓練期被擋組反而賺得多，含 4 筆大贏家）。',
                     'P2': 'F（遞補）版多半不如 Z 版或差不多：補進來的是 SURGE 較低的第 11～20 名（S81 已知遞補沒有好處）。',
                     'P3': '驗證期：即使凍結的是現金流類，V2（≥ +2）不過的機率 > 80%；循環翻身股（如 2408）財報落後約 105 天，現金流也可能還在谷底，擋到 1～2 筆大贏家就會輸。',
                     'P4': '全期分組：品質差的各組勝率與中位數略低，但平均報酬不一定低（大贏家拉高）；大輸家（後 5%）在品質差組的比例不會比全體高出一倍。'},
 'data_snooping': {'added': 19, 'note': '20 格中 R03|Z 與 S70 C1 相同不重複計；N +19。'},
 'caveats': ['現行策略 2022～2026 的結果早已看過，且 S28／S70 已知 2408 等翻身股在驗證期是大贏家：對 R01～R03、R07 的驗證期方向已有先驗，不算乾淨。現金流類旗標在任何期間都沒算過。',
             '財報封存可能含更正後數字；市場可得日 A 在 2017-04 前是推斷值。', '資料齡中位數約 105 天。'],
}
PR['code_sha256_at_write'] = {n: sha(H / n) for n in ('s88lib.py', 'train.py', 'validate.py', 'prep.py')}
txt = json.dumps(PR, ensure_ascii=False, indent=1, default=str)
(H / 'PREREG.json').write_text(txt)
h = hashlib.sha256(txt.encode()).hexdigest()
(H / 'PREREG.sha256').write_text(f'{h}  PREREG.json\n')
print(h)
