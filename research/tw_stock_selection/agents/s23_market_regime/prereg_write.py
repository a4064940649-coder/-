#!/usr/bin/env python3
"""S23 事前登記：在本題算任何報酬之前寫 PREREG.json 與 PREREG.sha256。已存在就拒絕覆寫。python3 -I -B prereg_write.py
（第 0 步 prep.py 只建面板、環境變數分布、訓練期門檻、各規則觸發月數與少買檔數，沒有算任何報酬。）"""
import sys, json, hashlib, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
H = Path(__file__).resolve().parent
f = H / 'PREREG.json'
assert not f.exists(), 'PREREG.json 已存在，拒絕覆寫'
import s23lib                                                     # noqa: E402
dc = json.loads((H / 'data_check.json').read_text())
lib_sha = hashlib.sha256((H / 's23lib.py').read_bytes()).hexdigest()
THR = dc['thresholds']
rules = []
for r in s23lib.RULES:
    x = dict(r); x['thr'] = THR.get(r['id'])
    rules.append(x)
eff = {k: {p: {kk: vv for kk, vv in d.items()} for p, d in v.items()} for k, v in dc['rule_effect_preview'].items()}

PRE = {
 'topic': 'S23 市場環境減碼：市場轉弱時把衛星新買改放 0050（或清出衛星），能不能降低回撤而不犧牲太多報酬？',
 'written_at': datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
 'status_at_write': '只跑過 prep.py（data_check.json、regime_features.csv）：環境變數分布、訓練期分位數門檻、各規則觸發月數與少買檔數；沒有算任何報酬（包括 0050 段內跌幅）。',
 's23lib_sha256_at_write': lib_sha,
 'baseline（不改定義）': 'S10 的 E0／W10_70（XQ 20＋21 號預設）：嚴成本、只上市、排除創新板、A10 訊號日次日開盤、F1 進場（L_MA 依 SURGE 前 10 名中 RET20 > 10%）、'
                      'E0 出場（跌出 L_MA 前 20 名）；衛星 70%、每檔＝衛星 ÷ 10；股數 × 開盤 < 目標一半不買；單筆 ≤ 當日成交值 5%；底倉 ±10 個百分點帶狀拉回；'
                      '0050 停牌日用前一日收盤（s17lib）。SURGE＝STD（s17lib.rev_family，σ > 0）。引擎 s10lib.simulate_s10，直接 import 重用（s23lib→s20lib→s17lib→s10lib）。',
 'scope': '只動衛星曝險：規則只改當月買進名單 sig（不買或少買），MA_EXIT 另在訊號列把全部衛星持股標記賣出；選股排序、F1、E0 出場、每檔＝衛星 ÷ 10、底倉 30% 全部不動。'
          '沒買或賣出的錢照引擎掃進 0050（不放現金）。只在換股日（訊號列）判斷，月中不進出。',
 'difference_from_past': {
     'breadth_derisk': '過去在舊主策略（n10 全市場前 300 流動性池、2006 起）用「創 52 週新高家數 B」把曝險降到現金；只過挑選期、驗證期沒過，B 與下月報酬幾乎無關（Spearman 0.06），代價集中在空頭後反彈。'
                       '本題不同處：(1) 現行 S10 衛星（S5～S10 定型的策略），不是舊 n10；(2) 減的是「衛星新買」，錢進 0050 而不是現金——底倉與空位本來就是 0050，所以本題只能把回撤壓到接近 0050 本身，不能躲開大盤下跌；'
                       '(3) 廣度改用營收面 SURGE 廣度與 L_MA 家數（與選股同源），另加 0050 對 200 日均線；(4) 主目標是回撤改善、報酬容忍 −1／−2 個百分點，而不是 Calmar＋−3 個百分點。'
                       '預期結果很可能一樣是否定（breadth_derisk 的失敗機制——錯過空頭後反彈——同樣適用）。',
     'o_concentration_regime': '那題是 0050 與主策略整體在「權值集中／擴散」兩態之間 100% 輪動（相對強弱訊號），不可用；本題不做整體輪動，只在絕對弱勢時暫停衛星新買（或清出衛星），底倉不動。',
     'timing_switch／timing_daily': '過去大盤均線開關轉現金套在 n10 上少 8～9 個百分點；本題 MA 規則改成月頻、只影響衛星、錢進 0050。',
     'structure_note': 'data_check：F1 名單 98% 的月份不滿 10 檔（中位數 3 檔），衛星本來就大半放 0050；所以「不新買」影響的金額有限，預期效果與代價都小。'},
 'features（訊號日收盤可得）': dict(
     BRD='股池 U[k]（只上市、流動性、非創新板）且 SURGE 有值的股票中 SURGE ≥ 2 的比例',
     BRD3='本月與前 2 個訊號列 BRD 的平均（data_check／S21：BRD 在農曆年基期月份大幅跳動，先平滑；這是看報酬前決定的）',
     LMA='股池中還原收盤 > 60 日均線（F_MA60）且 SURGE ≥ 2 的家數',
     MA200='0050 含息還原收盤 E50[k] < 含當日過去 200 個交易日平均 → 弱',
     threshold_rule='BRD3、LMA 的門檻＝訓練期訊號列（營收月份 2016-01～2021-11，位移 0，71 列）該變數的分位數，在 prep.py 算出、寫死於下方 thr；位移 1～3 用同一組數值。'),
 'rule_form': {
     'candidates（7 條＝5 種規則，其中 BRD／LMA 各 2 個分位數門檻；全部計入 N，參與訓練期選擇）': rules,
     'rule_effect_preview（data_check，只數月數與檔數）': eff,
 },
 'windows': dict(train='訊號日 2016-02～2021-12（營收月份 2016-01～2021-11）；訓練窗淨值 2016-01-29 → 2021-12-30（全期連續模擬切出來，同 S20／S21 train.py）',
                 valid='訊號日 2022-01～2026-10（營收月份 2021-12～2026-09）；重新起跑：只保留營收月份 ≥ 2021-12 的訊號，驗證窗 2021-12-30 → 2026-10-22（同 S20／S21 validate.py）'),
 'paths': '20 條路徑＝訊號日位移 0～3 天 × 起始資金 60／80／100／150／200 萬（同 S20／S21）；環境變數在各位移自己的訊號列上重算',
 'selection': {
     'path_improved': '訓練窗內，某條路徑算「改善」＝（最大回撤較淺：規則回撤 − 現行回撤 > 0，或 Calmar（訓練窗年化 ÷ |訓練窗最大回撤|）較高）且 年化差 ≥ −1 個百分點',
     'T0': '改善路徑 ≥ 14／20，且 20 條年化差中位數 ≥ −1 個百分點',
     'freeze': '改善路徑數最多者凍結；同數依 Calmar 差中位數、再依訓練期觸發月數少者。T0 不過仍凍結第一名，驗證期只描述、判定必為不通過。'},
 'verdict（驗證期只跑一次；以回撤改善為主目標）': dict(
     V1='驗證窗（重新起跑）最大回撤較淺（規則 − 現行 > 0）的路徑 ≥ 15／20',
     V2='驗證窗年化差（規則 − 現行）20 條中位數 ≥ −2 個百分點',
     V3='禁買 CUR 與凍結規則各自（重新起跑主跑，位移 0、100 萬）最賺 5 筆期間的聯集後，年化差中位數 ≥ −2 個百分點',
     V4='兩段空頭（2022 全年；2024-07～2025-04）段內最大跌幅配對差中位數都 > 0（兩段都較淺）',
     overall='T0 且 V1～V4 全過＝通過；否則不通過，停在描述，不換規則、不換門檻重跑、不寫 XQ 修改建議'),
 'random_control（登記的描述，不參與判定）': '凍結規則在同一窗口觸發 n 個月 → 在同窗口的訊號列中隨機挑 n 個月做同樣動作（stop／half／exit），20 個種子（20261116＋i）；'
                                    '驗證窗重新起跑與全期連續各一組，位移 0、100 萬；報告凍結規則年化差與回撤差在隨機分布中的百分位。',
 'acceptance': dict(A0='CUR 全期主跑重現 S10 W10_70：2019 起 48.13%、全期 34.08%、回撤 −33.40%（±0.025 個百分點）',
                    A1='apply_regime 不觸發（門檻 −1）時交易與淨值與 CUR 逐筆相同',
                    A2='MA_EXIT 觸發月的次日開盤，衛星持股全部被標記賣出（停板延後除外）'),
 'posthoc（未登記，只描述，驗證判定後才跑）': '其餘 6 條候選在驗證窗的結果（純描述，不得拿來換規則）；全期連續 2019 起與四段空頭；觸發月份與錯過／避開的報酬；放現金版（只描述）。',
 'N': '資料窺探族群 N：2,850 → 2,857（7 條候選；隨機對照不計）',
 'predictions（事前預測）': dict(
     P1='BRD_STOP 兩條訓練期回撤幾乎不變、年化小幅為負（營收廣度落後股價；2019 觸發月正逢多頭起漲）；T0 不過',
     P2='MA 規則在訓練期 2018、2020-03 回撤較淺 1～4 個百分點，但 2019-01～02、2020-04 錯過反彈，年化差 −1～−3；MA_EXIT 回撤改善最多、年化代價也最大',
     P3='凍結的最可能是 MA_HALF 或 MA_STOP；T0 通過機率約 35%',
     P4='驗證期 2022 段較淺（MA 在 2022-02～11 觸發）但 2024-07～2025-04 段 MA 只在 2025-02～04 觸發，太晚，最大跌幅大多已發生或剛好躲過 4 月急跌，難判；整體通過機率約 10%，最可能卡在 V4 或 V3',
     P5='因為錢進 0050、F1 本來就常不滿 10 檔，所有規則回撤改善幅度都不超過 3 個百分點'),
}
txt = json.dumps(PRE, ensure_ascii=False, indent=1)
f.write_text(txt)
sha = hashlib.sha256(txt.encode()).hexdigest()
(H / 'PREREG.sha256').write_text(f'{sha}  PREREG.json\n')
code = {n: hashlib.sha256((H / n).read_bytes()).hexdigest() for n in ('s23lib.py', 'prep.py', 'prereg_write.py')}
(H / 'code_prereg.sha256').write_text(''.join(f'{v}  {k}\n' for k, v in code.items()))
print(sha)
