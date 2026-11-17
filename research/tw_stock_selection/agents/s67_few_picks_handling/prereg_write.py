#!/usr/bin/env python3
"""S67 事前登記：寫 SPEC.json、PREREG.json 與各自 sha256（在任何候選的訓練／驗證報酬計算之前；只看過 data_check.json 的現行驗收、
第 0 步訓練期描述（現行口徑）、mech_check.json 的機械檢查（只有筆數與金額占比）與乾跑恆等結果）。python3 -I -B prereg_write.py"""
import json, hashlib, datetime
from pathlib import Path

H = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (H / 'PREREG.json').exists() and not (H / 'FROZEN.json').exists(), '已登記過，拒絕覆寫'
dc = json.loads((H / 'data_check.json').read_text())
mc = json.loads((H / 'mech_check.json').read_text())
assert dc['A0']['tol_ok'] and dc['A0']['bitwise_vs_deadline_ALL'] and dc['A2']['ok'] and dc['A3_all_ok'] and dc['A4_all_ok']
assert mc['M1_mult1_identical'] and mc['M2_c3_lo10_identical'] and mc['M4_C2']['n_buys_in_few_months'] == 0 and mc['M5_C3']['ok']
s0 = dc['step0']; tw = s0['trades_train_window']

SPEC = {
 'id': 'S67', 'title': '名單只有 1～2 檔怎麼辦（加倍／不買／放寬門檻補到 3 檔）：訓練→凍結→驗證一次',
 'written_at': datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
 'question': '現行策略每月名單平均約 4 檔，約四分之一的月份只有 0～2 檔（2026-11 只有 1 檔）。散戶遇到「名單只有 1～2 檔」該怎麼辦？',
 'dedup（先查過 BACKLOG.md 與 agents/，以下做過、不重做）': {
  'S4（s4_e3_purity）': '遞補到前 20 名（F2）變差、F1 不遞補＋空位 0050 成為現行；Nd15（前 15 名鄰域）沒差；「晚 2 天特別差」不成立。',
  'S6（s6_satellite_sizing）': 'WN＝每檔＝衛星 ÷ max(當月名單檔數, 5)（名單 ≤5 檔時每檔加大，最多 2 倍）：報酬較高、回撤較深，屬風險偏好；全樣本、非訓練／驗證。本題 C1 是它的窄版（只在名單 ≤2 檔時加倍、上限 14%），未做過。',
  'S47／S51（配置家族、每檔多押）': '每個月都把每檔加大（÷8 等）：S51 勉強通過、列為風險偏好選項。與「只在名單太少月份加倍」不同。',
  'S54': '空位留現金／空位買 0050：做過，不重做（本題空位一律 0050）。',
  'S56': '同產業上限＋遞補：遞補一律較差（第 11～20 名不太賺）。本題不做任何「遞補第 11 名以後」。',
  'S59／S61': '檔數／曝險對超額的解釋力很小（10.7 個百分點中約 0.4）；燈號③「連 3 月選出 ≤2 檔」誤報多。只是監控，沒有測「該月怎麼下單」。',
  'S63／S64／S65': '只持 5 檔（集中度）、名單只買一半（−6.0）、漏做月份（−0.7～−2.1）、新手進場：都不是依「名單檔數」條件處理。',
  'S46': '動能門檻家族（45 格，RET20 門檻每個月都改）：現行最好。本題 C3 只在名單 ≤2 檔的月份把門檻放寬到 5%，條件式，未做過。',
  '結論': 'C1（少檔月加倍）、C2（少檔月不買）、C3（少檔月放寬門檻補到 3 檔，仍限前 10 名）三者都沒做過。',
 },
 'definitions': {
  '名單': '當月訊號名單＝股池（上市、排除創新板）中站上 60 日均線者依 SURGE 排前 10 名，其中 RET20 > 10% 者（make_signal_s10 的 n_tgt；含已持有的股票）。',
  '名單太少月份': 'n_tgt ≤ 2（含 0 檔）；依各自位移後訊號日的名單判斷（位移 1～3 的路徑各自重算）。',
  'C1': '名單太少月份（1～2 檔），該月新買進每檔目標金額＝現行目標（衛星 ÷ 10，約占總資產 7%）× 2，上限＝開盤估總資產 × 14%；已持有的股票不加碼；其他月份不變。資金不足時照引擎原規則（賣 0050 支應、不足縮量、不到目標一半不買）。',
  'C2': '名單太少月份不買新股（該訊號列買進名單清空）；已持有的照 E0 出場；其他月份不變。',
  'C3': '名單太少月份，在同一批前 10 名中把 RET20 門檻由 > 10% 放寬到 > 5%：5% < RET20 ≤ 10% 者依 SURGE 由高到低補進，補到名單滿 3 檔為止（不夠就有幾檔補幾檔；0 檔月份最多補 3 檔）；每檔金額不變；其他月份不變。',
 },
 'engine': {
  'chain': 's67lib → s47lib（不改；simulate_s47 為 s10lib.simulate_s10 原始碼字串替換副本）→ s20lib → s17lib → s10lib（全部不改）',
  'simulate_s67': 's47lib._SRC 再做三處字串替換：新參數 mult／mult_cap；u = sv / div 後依訊號列倍數調整（上限 mult_cap × eq）；5% 成交值上限檢查前加計數器 n_sized（只計數）。函式名 simulate_s67。原檔不動。',
  'flags': {'EXRIGHT_FIX': True, 'DEADLINE_FIX': True, 'INS15_MASK': True},
  'fixed': '出場 E0、嚴成本（strict）、股池只上市排除創新板、W10_70（衛星 70%、每檔＝衛星 ÷ 10）、最多 10 檔、帶狀拉回 ±10 個百分點、2 萬掃 0050、空位 0050、5% 成交值上限、目標一半不買，全部不動。',
  'src_sha256': dc['src_sha256'],
  'acceptance_done_before_prereg': {
   'A0': '現行 100 萬位移 0：2019 起 48.76%、全期 34.40%、回撤 −32.12%、已平倉 381 筆，與 deadline_engine_fix ALL 逐位相同',
   'A2': '現行 20 條路徑與 deadline_engine_fix ALL 逐條差 0',
   'A3': 'simulate_s67（mult=None）與原引擎 simulate_s10 在 20 條連續路徑、重新起跑主跑、禁買主跑淨值與交易逐位相同',
   'A4': 'C3 用的前 10 名重建（pool_of）在 129 個訊號列 × 位移 0～3 與 make_signal_s10 的名單逐檔相同',
   'mech_check（只看筆數與金額占比，不看報酬）': mc,
   'dry_run': 'train.py --dry、validate.py --dry（三個候選都換成現行）全部差 0，程式路徑正常',
  },
 },
 'paths': '20 條＝起始資金 {60, 80, 100, 150, 200} 萬 × 訊號位移 {0, 1, 2, 3}',
 'windows': {
  'train': '連續模擬（全部訊號），量 2016-01-30 → 2021-12-30（＝訊號 2016-02～2021-12）的年化（stlib.cagr）與窗內最大回撤',
  'valid': '重新起跑：只保留營收月份 ≥ 2021-12 的訊號（2022-01-01 前無交易，全在 0050），起始資金 ÷ 2.558 使 2021-12-30 淨值約＝60～200 萬；量 2021-12-30 → 2026-10-22',
  'bear_segments_V4': '2022（2022-01-01～2022-12-31）、2024-07～2025-04（2024-07-01～2025-04-30），段內高低點的最大跌幅（s17lib.seg_dd）',
 },
 'step0_descriptive（現行口徑；寫 PREREG 前只看訓練期報酬）': {
  'list_size_all_129_months': s0['list_size']['all'], 'list_size_train': s0['list_size']['train_window'], 'list_size_valid（只是檔數）': s0['list_size']['valid'],
  'new_buys_all': s0['new_buys']['all'], 'new_buys_train': s0['new_buys']['train_window'],
  'trades_train_by_list_size': tw,
  'reading': '訓練期名單 ≤2 檔月份買進 17 筆（11 個月），平均報酬 +2.6%、同持有期 0050 +0.3%、超額 +2.3%（逐月 t 0.71）；名單 ≥5 檔月份 105 筆，平均 +14.3%、超額 +10.1%（逐月 t 2.47）。'
             '少檔月的那 1～2 檔沒有比較準，反而較平庸；但樣本只有 17 筆。',
  'c3_fill_counts_train': {k: v for k, v in s0['c3_fill_counts_train'].items() if k != 'detail'},
 },
 'capacity': '1,000 萬、5% 當日成交值上限、全期連續、嚴成本、位移 0～3；capped_rate＝被上限縮量的買進嘗試 ÷ 算出股數的買進嘗試（S37／S51 口徑；現行位移 0 為 13.2%）',
 'late2': '「晚 2 天」＝位移 2（訊號日延後 2 個交易日，名單、名次、RET20 都用延後那天的資料）對位移 0；各 5 種資金中位數；只描述。',
 'files': ['s67lib.py', 'prep.py', 'mech_check.py', 'prereg_write.py', 'train.py', 'validate.py', 'data_check.json', 'mech_check.json', 'results_train.json', 'FROZEN.json', 'results.json'],
 'code_sha256_at_write': {n: sha(H / n) for n in ('s67lib.py', 'prep.py', 'mech_check.py', 'prereg_write.py', 'train.py', 'validate.py')},
 'temp': '/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s67/（結束刪除）',
}

PRE = {
 'id': 'S67', 'written_at': SPEC['written_at'],
 'candidates（事前固定，不得增減）': {'C1': '名單 ≤2 檔月份每檔金額加倍（上限每檔占總資產 14%）', 'C2': '名單 ≤2 檔月份不買新股（已持有照 E0）',
                           'C3': '名單 ≤2 檔月份 RET20 門檻 10% → 5%，補到至少 3 檔（仍限前 10 名）'},
 'current': '現行：名單幾檔就買幾檔、每檔＝衛星 ÷ 10、其餘放 0050',
 'step1_train_selection': {
  'rule': '三個候選中，取訓練窗 20 條路徑年化中位數最高者（同分依 ≥ +0.25 路徑數、再依 C1→C3 順序）。'
          '若「其中位數 − 現行中位數 ≥ +1 個百分點」且「其回撤中位數 − 現行回撤中位數 ≥ −2 個百分點（不深於現行 2 個百分點以上）」→ 選它；'
          '否則選「不改」（不往下找第二名）。',
  'frozen_for_validation': '不論選它或「不改」，都凍結中位數最高的候選跑驗證期一次；選「不改」時驗證只作描述，總判定一律不通過。',
  'T0（S25 標準，訓練期）': '凍結候選對現行的成對差（逐條路徑）中位數 ≥ +1 個百分點，且差 ≥ +0.25 個百分點的路徑 ≥ 14/20。',
 },
 'step2_validation（凍結後只跑一次）': {
  'V1': '驗證窗年化差（凍結候選 − 現行，重新起跑）> 0 的路徑 ≥ 15/20',
  'V2': '驗證窗年化差中位數 ≥ +2 個百分點',
  'V3': '兩邊（現行與凍結候選的重新起跑主跑：100 萬換算、位移 0）各自最賺 5 筆已平倉交易的聯集，整段持有期兩邊都禁買，重跑 20 條路徑後年化差中位數 > 0',
  'V4': '2022、2024-07～2025-04 兩段空頭的段內最大跌幅差中位數都 ≥ −2 個百分點（不深於現行 2 個百分點以上）',
  'overall': '訓練選擇不是「不改」且 T0、V1、V2、V3、V4 全過 → 建議改成凍結候選；否則不改。',
  'also_report': '驗證窗最大回撤與 Calmar 的成對差；2022～2023、2024 起兩段年化差；成本加倍。',
 },
 'descriptive_only（登記但不作判定）': [
  '另兩個候選在驗證期的同樣 V1～V4 統計（V3 各自用與現行的聯集）',
  '全期連續 20 條路徑（2019 起、全期、回撤、Calmar、四段空頭）',
  '容量：1,000 萬 capped_rate（現行與三個候選）',
  '「晚 2 天」：位移 2 對位移 0，以及各位移下候選 − 現行',
  '第 0 步的全樣本與驗證期版（現行口徑）；各候選在名單太少月份實際加倍／少買／補進的交易報酬',
 ],
 'predictions（事前）': {
  'P0': '很可能「不改」（約 85%）。理由：訓練期少檔月買進的交易只有 17 筆、占全部損益 0.3%，超額 +2.3%（t 0.71），比名單多的月份平庸；任何只動這幾筆的處理方式很難讓年化差到 +1 個百分點。',
  'P1': 'C1（加倍）：訓練期年化差約 −0.3～+0.8 個百分點、回撤差不多；到不了 +1。',
  'P2': 'C2（不買）：訓練期年化差約 −0.5～+0.2 個百分點（少賺那一點超額），回撤幾乎不變。',
  'P3': 'C3（放寬補到 3 檔）：訓練期年化差約 −0.8～+0.5 個百分點（補進的是動能較弱的；S46 門檻家族中現行 10% 最好）。',
  'P4': '驗證期：凍結候選 V2（≥ +2）不過；整體通過機率 < 5%。',
  'P5': '容量：C1 的 1,000 萬縮量比例比現行高約 1 個百分點以內；C2 略低；C3 差不多。「晚 2 天」各候選與現行的差距在 ±1 個百分點內。',
 },
 'caveat_validation_not_fresh': '現行策略 2022～2026 的全期結果早已看過（S47、S59 等），但三個候選本身沒有在任何期間算過；驗證期對候選是未見過的。'
                                '少檔月份只占約四分之一、每月只 1～2 筆，統計力低：不通過多半代表「沒有可量到的差別」，不代表候選有害。',
 'data_snooping': {'added': 3, 'note': '三個事前固定的候選，N +3。'},
 'spec_sha256': None,
}
txt = json.dumps(SPEC, ensure_ascii=False, indent=1)
(H / 'SPEC.json').write_text(txt)
ssha = hashlib.sha256(txt.encode()).hexdigest()
(H / 'SPEC.sha256').write_text(f'{ssha}  SPEC.json\n')
PRE['spec_sha256'] = ssha
txt = json.dumps(PRE, ensure_ascii=False, indent=1)
(H / 'PREREG.json').write_text(txt)
psha = hashlib.sha256(txt.encode()).hexdigest()
(H / 'PREREG.sha256').write_text(f'{psha}  PREREG.json\n')
print('SPEC', ssha); print('PREREG', psha)
