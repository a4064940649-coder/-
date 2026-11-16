#!/usr/bin/env python3
"""S33 事前登記：在本題算任何報酬之前寫 PREREG.json 與 PREREG.sha256。已存在就拒絕覆寫。python3 -I -B prereg_write.py
（第 0 步 prep.py 只建面板、PTH 涵蓋率與分位數、各規則改動名單的檔數，沒有算任何報酬。）"""
import sys, json, hashlib, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
H = Path(__file__).resolve().parent
f = H / 'PREREG.json'
assert not f.exists(), 'PREREG.json 已存在，拒絕覆寫'
import s33lib                                                     # noqa: E402
dc = json.loads((H / 'data_check.json').read_text())
sha = lambda p: hashlib.sha256((H / p).read_bytes()).hexdigest()

PRE = {
 'topic': 'S33 George & Hwang（2004, JF）52 週高點接近度（PTH）能不能改進 SURGE 名單內的選股？',
 'written_at': datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
 'status_at_write': '只跑過 prep.py（data_check.json）：面板、PTH 涵蓋率、分位數、相關、各規則改動名單檔數；沒有算任何報酬。寫入前沒有修改過規則。',
 'prior_work（避免重做）': {
     '第二輪／w 系列獨立策略': '52 週高點相關的既有研究都是「獨立選股」（全市場接近新高買進，如 bt_52w_oos.py、主策略的 52 週高＋營收、w3_ath_breakout_duke、w1_minervini_trend_template）或接近 52 週低點（w3_52w_low_nearness），不是 SURGE 名單內。',
     'S2 D2': '舊版 C1（S1 月營收驚奇，非現行 F1、未開除權息修正、無 20 條路徑判定）上加「還原收盤 ÷ 240 日還原最高價 ≥ 0.90」門檻（疊加在 RET20 之上）→「沒差」。本題 GATE_PTH 是「取代」RET20，不是疊加。',
     'S13 NEAR52': 'L_MA 前 20 名內的描述性特徵（大贏家 AUC 0.451，未被選中、未進驗證、未做策略測試）。',
     '結論': '沒有做過現行 F1（EXRIGHT_FIX）名單內、S25 判定架構下的 PTH 排序／篩選／遞補／換門檻 → 不算重做，繼續。'},
 'code_sha256_at_write': {p: sha(p) for p in ('s33lib.py', 'prep.py')},
 'baseline（不改定義）': 'S10 的 E0／W10_70（XQ 20＋21 號預設），cslib.EXRIGHT_FIX＝True：嚴成本、只上市、排除創新板、A10 訊號日次日開盤、F1 進場（L_MA 依 SURGE 前 10 名中 RET20 > 10%）、'
                      'E0 出場（跌出 L_MA 前 20 名）；衛星 70%、每檔＝衛星 ÷ 10；0050 停牌日用前一日收盤（s17lib）。SURGE＝STD（σ > 0）。引擎 s20lib→s17lib→s10lib，直接 import。',
 'score': dict(
     PTH='還原收盤（含息累積指數 G）÷ 過去 252 個交易日（含訊號日）最高還原收盤；上市滿 252 根才有值。只用訊號日收盤以前的資料（PIT；prep.py 自檢 pit_selfcheck＝True）。'
         'George & Hwang 原文用收盤 ÷ 52 週最高價；本題用還原收盤以免除權息造成假性遠離高點。',
     pPTH='同期百分位：訊號列 k 的股池 U[k] 內有值者排名，0～1；缺值（上市未滿 252 根）以 0.5 代入（排序、相對篩選用）。',
     coverage='F1 上涵蓋率：訓練 99.6%、驗證 98.8%（data_check.json）。F1 的 PTH 中位數約 0.97、四分之一以上恰為 1（訊號日即創 252 日新高）。'),
 'rule_form': {
     'candidates（4 條，全部計入 N；參與訓練期選擇）': s33lib.RULES,
     'control（1 條，只描述，不參與選擇與判定）': s33lib.CONTROL,
     'common': '所有規則只改當月買進名單；出場 X（E0）不動。遞補、重排與換門檻只在 L_MA 前 20 名內進行，所以新買的股票同樣在跌出前 20 名時賣出。篩掉的名額不遞補，錢照引擎掃進 0050。',
     'rule_effect_preview（data_check，只數檔數）': dc['rule_effect_preview'],
 },
 'windows': dict(train='訊號日 2016-02～2021-12（營收月份 2016-01～2021-11）；訓練窗淨值 2016-01-29 → 2021-12-30（全期連續模擬切出來，同 S20／S21／S29）',
                 valid='訊號日 2022-01～2026-10（營收月份 2021-12～2026-09）；重新起跑：只保留營收月份 ≥ 2021-12 的訊號，驗證窗 2021-12-30 → 2026-10-22'),
 'paths': '20 條路徑＝訊號日位移 0～3 天 × 起始資金 60／80／100／150／200 萬（同 S20／S21／S25／S29）',
 'selection': '訓練期分數＝候選規則 − 現行（CUR）的訓練窗年化差，20 條路徑的中位數（個百分點）；分數最高者凍結（同分依改善路徑數、再依改動檔數少者）。'
              'T0（沿用 S25）：分數 ≥ +1 個百分點，且年化差 ≥ +0.25 個百分點的路徑 ≥ 14／20。T0 不過仍凍結分數最高者，驗證期只描述、判定必為不通過。',
 'verdict（驗證期只跑一次，沿用 S25）': dict(
     V1='驗證窗（重新起跑）年化差為正的路徑 ≥ 15／20',
     V2='配對差中位數 ≥ +2 個百分點',
     V3='禁買 CUR 與凍結規則各自（重新起跑主跑）最賺 5 筆期間的聯集後，配對差中位數 > 0',
     V4='兩段空頭（2022；2024-07～2025-04）段內最大回撤配對差中位數都不深於 −2 個百分點',
     overall='T0 且 V1～V4 全過＝通過；否則不通過，停在描述，不換規則重跑；通過才寫 XQ 20 號修改建議（只寫建議，不改檔）'),
 'acceptance': dict(A0='CUR 全期主跑（EXRIGHT_FIX＝True）重現：2019 起 48.99%、全期 34.54%、回撤 −32.11%（±0.025 個百分點）、381 筆已出場',
                    A1='build_variant 每個訊號列重現的 F1 與 make_signal_s10 完全相同（assert）',
                    A2='「什麼都不剔除」的篩選規則（FLT_PTH 比例 0）交易與淨值與 CUR 逐筆相同'),
 'describe（不論通過與否都做，未參與判定）': 'F1 逐筆（s20lib.outcomes：E0 持有期，不扣成本）對 0050 超額與對同業超額（同 S25／S29：PIT 產業、股池 U、排除自己與產業最大市值股、同業 ≥ 5 家，等權），'
                                       '依 PTH 三分組：(i) 當月 F1 內名次三分（主表）、(ii) 同期股池百分位 pPTH（< 1/3、1/3～2/3、≥ 2/3），各組平均、中位數、勝率、月份群聚 t；高組 − 低組逐月差的平均與 t；'
                                       '逐月等級相關（IC）平均與 t。訓練、驗證、全期分列。另描述 L_MA 第 11～20 名依 PTH ≥ 0.90 分兩組（解釋 FILL_PTH）、L_MA 前 10 名中 RET20 ≤ 10% 但 PTH ≥ 0.90 者（解釋 GATE_PTH）。'
                                       '隨機對照：凍結規則的 PTH 換成「當月 L_MA 前 20 名 PTH 值的隨機重排」（分布與機制不變），20 次。',
 'N': '資料窺探族群 N 加 5（4 條候選＋1 條對照）。',
 'predictions（事前預測）': dict(
     P1='FLT_PTH 剔除約 25% 的 F1、錢放 0050，訓練期為負（S20 機械效果；F1 的 PTH 幾乎都 > 0.9，區分度低）',
     P2='GATE_PTH 改動最多（訓練期加 296、減 68 檔），結果變異最大；S2 D2「沒差」、S1 診斷「最賺 20 筆共同點是已在漲」→ 預期 GATE_PTH 為負',
     P3='RANK_PTH 分數 −2～+2；S13 NEAR52 在門內 AUC 0.451（略反向）→ 門內 PTH 排序可能無用或反向',
     P4='FILL_PTH 會比 FILL_ALL 遞補更多（不看 RET20），S2 診斷「遞補的 11 名以後只有 +1.4～+2.5%」→ 預期不比 CUR 好',
     P5='描述：F1 內 PTH 三分組高 − 低接近 0（|t| < 2）；整體通過機率約 5～10%'),
}
txt = json.dumps(PRE, ensure_ascii=False, indent=1)
f.write_text(txt)
h = hashlib.sha256(txt.encode()).hexdigest()
(H / 'PREREG.sha256').write_text(f'{h}  PREREG.json\n')
(H / 'code_prereg.sha256').write_text(''.join(f'{sha(p)}  {p}\n' for p in ('s33lib.py', 'prep.py', 'prereg_write.py')))
print(h)
