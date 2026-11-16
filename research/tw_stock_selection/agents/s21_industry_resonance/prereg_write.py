#!/usr/bin/env python3
"""S21 事前登記：在本題算任何報酬之前寫 PREREG.json 與 PREREG.sha256。已存在就拒絕覆寫。python3 -I -B prereg_write.py
（第 0 步 prep.py 只建面板、PIT 產業對照、共振特徵涵蓋率與分位數、各規則改動名單的檔數，沒有算任何報酬。）"""
import sys, json, hashlib, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
H = Path(__file__).resolve().parent
f = H / 'PREREG.json'
assert not f.exists(), 'PREREG.json 已存在，拒絕覆寫'
import s21lib                                                     # noqa: E402
dc = json.loads((H / 'data_check.json').read_text())
lib_sha = hashlib.sha256((H / 's21lib.py').read_bytes()).hexdigest()
ser = lambda r: {k: (list(v) if isinstance(v, tuple) else v) for k, v in r.items()}

PRE = {
 'topic': 'S21 產業共振：同 PIT 產業很多公司一起營收爆發時，SURGE 選股會不會更準？',
 'written_at': datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
 'status_at_write': '只跑過 prep.py（data_check.json）：面板、PIT 產業對照、共振特徵分布、各規則改動名單檔數；沒有算任何報酬。',
 's21lib_sha256_at_write': lib_sha,
 'baseline（不改定義）': 'S10 的 E0／W10_70（XQ 20＋21 號預設）：嚴成本、只上市、排除創新板、A10 訊號日次日開盤、F1 進場（L_MA 依 SURGE 前 10 名中 RET20 > 10%）、'
                      'E0 出場（跌出 L_MA 前 20 名）；衛星 70%、每檔＝衛星 ÷ 10；股數 × 開盤 < 目標一半不買；單筆 ≤ 當日成交值 5%；底倉 ±10 個百分點帶狀拉回；'
                      '0050 停牌日用前一日收盤（s17lib）。SURGE＝STD（s17lib.rev_family，σ > 0）。引擎 s10lib.simulate_s10，直接 import 重用。',
 'industry_data': dict(file='agents/o_lc_sector_rotation_pit/raw/industry_pit.csv（只讀、不複製）',
                       rule='訊號日當天有效的那一段（start ≤ 日 < end）。2015-05 前為回溯（訓練期從 2016-02 開始，不受影響）；2023-07-03 新產業別起生效。「未分類」→ 缺值。',
                       coverage='股池（只上市、流動性、非創新板、SURGE 有值）產業涵蓋率 100%（data_check.json）；F1 缺產業 0%；同業不足 5 家（BRX 缺值）訓練期 F1 1.4%。',
                       caveat='2017-07-06～2023-01-16 之間沒有快照：這段期間改類別的公司會在 2023-01-16 才換（落後，不是前視）；2017-07 後新上市公司的類別取自 2023 快照（輕微回填，產業別很少變）。'),
 'features': dict(
     universe='訊號列 k：股池 U[k] 且當月 SURGE 有值的股票，依當時 PIT 產業分組',
     PEER='同產業（不含自己）SURGE ≥ 2 的家數',
     BRX='PEER ÷（同業有 SURGE 者家數 − 1）；同業有 SURGE 者 < 5 家 → 缺值（＝題目的「產業 SURGE 廣度」，不含自己以免機械相關）',
     MKT='全股池（不含自己）SURGE ≥ 2 的比例',
     REL='BRX − MKT（扣掉全市場同時爆發：data_check 顯示全股池 SURGE ≥ 2 比例在 1%～35% 間大幅跳動，農曆年基期月份最高）'),
 'rule_form': {
     'candidates（5 條，全部計入 N；參與訓練期選擇）': [ser(r) for r in s21lib.RULES],
     'control（1 條，只描述，不參與選擇與判定）': ser(s21lib.CONTROL),
     'common': '所有規則只改當月買進名單；出場 X（E0）不動。遞補與重排只在 L_MA 前 20 名內進行，所以新買的股票同樣在跌出前 20 名時賣出。篩掉的名額不遞補，錢照引擎掃進 0050。',
     'rule_effect_preview（data_check，只數檔數）': dc['rule_effect_preview'],
 },
 'windows': dict(train='訊號日 2016-02～2021-12（營收月份 2016-01～2021-11）；訓練窗淨值 2016-01-29 → 2021-12-30（全期連續模擬切出來，與 S20 train.py 相同）',
                 valid='訊號日 2022-01～2026-10（營收月份 2021-12～2026-09）；重新起跑：只保留營收月份 ≥ 2021-12 的訊號，驗證窗 2021-12-30 → 2026-10-22（同 S20 validate.py）'),
 'paths': '20 條路徑＝訊號日位移 0～3 天 × 起始資金 60／80／100／150／200 萬（同 S20）',
 'selection': '訓練期分數＝候選規則 − 現行（CUR）的訓練窗年化差，20 條路徑的中位數（個百分點）；分數最高者凍結（同分依 npos、再依改動檔數少者）。'
              'T0（訓練期最低要求）：分數 ≥ +1 個百分點且 ≥ 14／20 條為正。T0 不過仍凍結分數最高的一條，驗證期只描述、判定必為不通過。',
 'verdict（驗證期只跑一次，沿用 S20）': dict(
     V1='驗證窗（重新起跑）年化差為正的路徑 ≥ 15／20',
     V2='配對差中位數 ≥ +2 個百分點',
     V3='禁買 CUR 與凍結規則各自（重新起跑主跑）最賺 5 筆期間的聯集後，配對差中位數 > 0',
     V4='兩段空頭（2022；2024-07～2025-04）段內最大回撤配對差中位數都不深於 −2 個百分點',
     overall='T0 且 V1～V4 全過＝通過；否則不通過，停在描述，不換規則重跑、不寫 XQ 修改建議'),
 'acceptance': dict(A0='CUR 全期主跑重現 S10 W10_70：2019 起 48.13%、全期 34.08%、回撤 −33.40%（±0.025 個百分點）',
                    A1='build_variant 每個訊號列重現的 F1 與 make_signal_s10 完全相同（assert）',
                    A2='「全部通過」的篩選規則（PEER ≥ −1）交易與淨值與 CUR 逐筆相同'),
 'posthoc（未登記，只描述）': '逐筆超額（同 S20 outcomes）依共振高低分組；產業別貢獻；FILL_ALL 對照；全期連續模擬；隨機篩掉同比例對照（如有篩選規則被凍結）。',
 'N': '資料窺探族群 N：2,844 → 2,850（5 條候選＋1 條對照）',
 'predictions（事前預測）': dict(
     P1='兩條篩選規則（FLT_P2、FLT_REL）訓練期都為負（像 S20：篩掉近 45% 的 F1、錢放 0050，機械效果約 −5～−10 個百分點）',
     P2='最好的是 RANK_BR 或 FILL_P2，分數在 −2～+2 之間；T0 通過機率約 25%',
     P3='FILL_P2 比 FILL_ALL 好（共振的第 11～20 名比不共振的好），但兩者都不會比 CUR 好很多（S4 F2 遞補到前 20 名較差）',
     P4='整體通過（T0＋V1～V4）機率約 10%；最可能卡在 T0 或 V2'),
}
txt = json.dumps(PRE, ensure_ascii=False, indent=1)
f.write_text(txt)
sha = hashlib.sha256(txt.encode()).hexdigest()
(H / 'PREREG.sha256').write_text(f'{sha}  PREREG.json\n')
print(sha)
