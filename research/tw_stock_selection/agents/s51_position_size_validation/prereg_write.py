#!/usr/bin/env python3
"""S51 事前登記：寫 SPEC.json、PREREG.json 與各自 sha256（在任何候選格的訓練／驗證報酬計算之前；只看過 data_check.json 的現行格驗收與乾跑恆等結果）。
python3 -I -B prereg_write.py"""
import json, hashlib, datetime
from pathlib import Path

H = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (H / 'PREREG.json').exists() and not (H / 'FROZEN.json').exists(), '已登記過，拒絕覆寫'
dc = json.loads((H / 'data_check.json').read_text())
assert dc['A0']['tol_ok'] and dc['A0']['bitwise_vs_deadline_ALL'] and dc['A2']['ok'] and dc['A3_all_ok']

SPEC = {
 'id': 'S51', 'title': '每檔多押一點（每檔約占總資產 8～8.75%，現行 7%）：訓練→凍結→驗證一次',
 'written_at': datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
 'origin': 'S47（agents/s47_allocation_family）事後看到 S70_D8、S80_D10、S100_D12 三格 20/20 條為正（+4.41／+2.59／+2.67 個百分點），回撤深 1.2～1.7 個百分點，領先一大塊來自 2020 年；S47 只列為後續題目、不改。',
 'engine': {
  'chain': 's51lib → s47lib（不改；simulate_s47 為 s10lib.simulate_s10 原始碼字串替換副本）→ s20lib → s17lib → s10lib（全部不改）',
  'simulate_s51': 's47lib._SRC 再加一處：買進迴圈 5% 成交值上限檢查前加計數器 st[n_sized]（只計數，不影響計算）；函式名 simulate_s51。',
  'flags': {'EXRIGHT_FIX': True, 'DEADLINE_FIX': True, 'INS15_MASK': True},
  'fixed': '選股 F1（L_MA 依 SURGE 前 10 名中 RET20 > 10%）、出場 E0、嚴成本（strict）、股池只上市排除創新板、最多 10 檔、帶狀拉回 ±10 個百分點、2 萬掃 0050、空位 0050、5% 成交值上限、目標一半不買，全部不動。',
  'cells': {'S70_D10（現行）': '衛星 70%、每檔＝衛星 ÷ 10（約占總資產 7%）',
            'C1 S70_D8': '衛星 70%、每檔＝衛星 ÷ 8（約 8.75%）',
            'C2 S80_D10': '衛星 80%、每檔＝衛星 ÷ 10（約 8%）',
            'C3 S100_D12': '衛星 100%、每檔＝衛星 ÷ 12（約 8.33%；檔數上限仍 10）'},
  'acceptance_done_before_prereg': {'A0': '現行 100 萬位移 0：2019 起 48.76%、全期 34.40%、回撤 −32.12%、已平倉 381 筆，與 deadline_engine_fix ALL 逐位相同',
                                    'A2': '現行 20 條路徑與 deadline_engine_fix ALL 逐條差 0',
                                    'A3': 'simulate_s51（ndiv=10）與原引擎 simulate_s10 在 20 條連續路徑、重新起跑主跑、禁買主跑淨值與交易逐位相同',
                                    'dry_run': 'train.py --dry、validate.py --dry（三個候選都換成現行格）全部差 0，程式路徑正常'},
 },
 'paths': '20 條＝起始資金 {60, 80, 100, 150, 200} 萬 × 訊號位移 {0, 1, 2, 3}',
 'windows': {
  'train': '連續模擬（全部訊號），量 2016-01-29 → 2021-12-30（＝訊號 2016-02～2021-12）的年化（stlib.cagr）與窗內最大回撤',
  'train_ex2020': '敏感度：訓練窗日報酬剔除 2020 曆年各日後連乘；年數＝(窗內曆日數 − 2020 年在窗內曆日數) ÷ 365.25；回撤用剔除後的連乘序列',
  'valid': '重新起跑：只保留營收月份 ≥ 2021-12 的訊號（2022-01-01 前無交易，全在 0050），起始資金 ÷ 2.558 使 2021-12-30 淨值約＝60～200 萬；量 2021-12-30 → 2026-10-22',
  'bear_segments_V4': '2022（2022-01-01～2022-12-31）、2024-07～2025-04（2024-07-01～2025-04-30），段內高低點的最大跌幅（s17lib.seg_dd）',
 },
 'capacity': '1,000 萬、5% 當日成交值上限、全期連續、嚴成本、位移 0～3；capped_rate＝被上限縮量的買進嘗試 ÷ 算出股數的買進嘗試（S37 口徑；現行格試算位移 0 為 13.2%，S37 當時 12.9%）',
 'files': ['s51lib.py', 'prep.py', 'prereg_write.py', 'train.py', 'validate.py', 'data_check.json', 'results_train.json', 'FROZEN.json', 'results.json'],
 'code_sha256_at_write': {n: sha(H / n) for n in ('s51lib.py', 'prep.py', 'prereg_write.py', 'train.py', 'validate.py')},
 'temp': '/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s51/（結束刪除）',
}

PRE = {
 'id': 'S51', 'written_at': SPEC['written_at'],
 'candidates（事前固定，不得增減）': {'C1': 'S70_D8', 'C2': 'S80_D10', 'C3': 'S100_D12'}, 'current': 'S70_D10',
 'step1_train_selection': {
  'rule': '三個候選中，取訓練窗 20 條路徑年化中位數最高者（同分依 ≥ +0.25 路徑數、再依 C1→C3 順序）。'
          '若「其中位數 − 現行中位數 ≥ +1 個百分點」且「其回撤中位數 − 現行回撤中位數 ≥ −2 個百分點（不深於現行 2 個百分點以上）」→ 選它；'
          '否則選「不改」（不往下找第二名）。',
  'frozen_for_validation': '不論選它或「不改」，都凍結中位數最高的候選跑驗證期一次；選「不改」時驗證只作描述，總判定一律不通過。',
  'T0（S25 標準，訓練期）': '凍結候選對現行的成對差（逐條路徑）中位數 ≥ +1 個百分點，且差 ≥ +0.25 個百分點的路徑 ≥ 14/20。',
 },
 'step2_ex2020_sensitivity': '訓練期另算「去掉 2020 年」的年化與回撤（定義見 SPEC.windows.train_ex2020），報告三個候選的成對差中位數、為正路徑數、以及若套同一選擇規則會不會被選；只報告，不作判定、不影響凍結。',
 'step3_validation（凍結後只跑一次）': {
  'V1': '驗證窗年化差（凍結候選 − 現行，重新起跑）> 0 的路徑 ≥ 15/20',
  'V2': '驗證窗年化差中位數 ≥ +2 個百分點',
  'V3': '兩邊（現行與凍結候選的重新起跑主跑：100 萬換算、位移 0）各自最賺 5 筆已平倉交易的聯集，整段持有期兩邊都禁買，重跑 20 條路徑後年化差中位數 > 0',
  'V4': '2022、2024-07～2025-04 兩段空頭的段內最大跌幅差中位數都 ≥ −2 個百分點（不深於現行 2 個百分點以上）',
  'overall': '訓練選擇不是「不改」且 T0、V1、V2、V3、V4 全過 → 建議改成凍結候選；否則不改。',
  'also_report': '驗證窗最大回撤與 Calmar（驗證窗年化 ÷ |驗證窗最大回撤|）的成對差；2022～2023、2024 起兩段年化差；成本加倍。',
 },
 'descriptive_only（登記但不作判定）': ['另兩個候選在驗證期的同樣 V1～V4 統計（V3 各自用與現行的聯集）', '全期連續 20 條路徑（2019 起、全期、回撤、Calmar、四段空頭）',
                                     '容量：1,000 萬 capped_rate（四格）；每檔加碼會讓單筆金額變大、容量變差，需註明'],
 'predictions': {
  'P1': '訓練期 C1（S70_D8）中位數最高，比現行多約 +3～+5 個百分點，回撤深約 1～2 個百分點 → 大概會被選（機率約 60%）。',
  'P2': '去掉 2020 年後，三個候選的領先幅度縮小一半以上（S47 指出領先大塊來自 2020 年）。',
  'P3': '驗證期 V1 大概會過（S47 全期 20/20 為正），V2（≥ +2）約一半機率，V3 不確定；整體通過機率約 30%。',
  'P4': '容量：每檔加碼後 1,000 萬 capped_rate 從約 13% 升到約 15～18%。',
 },
 'caveat_validation_not_fresh': 'S47 已看過 2015-07～2026-10 全期結果（含 2022～2026），三個候選本身就是看了全期才挑出來的；本題的「驗證期」對這些候選並非真正未見過的資料，通過只能算「沒有被推翻」，不通過則是較強的證據。',
 'data_snooping': {'added': 0, 'note': '三格在 S47 已計入 N（+13），本題不新增格。'},
 'spec_sha256': None,
}
(H / 'SPEC.json').write_text(json.dumps(SPEC, ensure_ascii=False, indent=1))
(H / 'SPEC.sha256').write_text(sha(H / 'SPEC.json') + '  SPEC.json\n')
PRE['spec_sha256'] = sha(H / 'SPEC.json')
(H / 'PREREG.json').write_text(json.dumps(PRE, ensure_ascii=False, indent=1))
(H / 'PREREG.sha256').write_text(sha(H / 'PREREG.json') + '  PREREG.json\n')
print(open(H / 'SPEC.sha256').read(), open(H / 'PREREG.sha256').read())
