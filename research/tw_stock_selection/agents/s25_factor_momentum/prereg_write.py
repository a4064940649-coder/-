#!/usr/bin/env python3
"""S25 B 部分事前登記：寫 PREREG.json 與 PREREG.sha256（只在 prep.py 之後、describe.py／train.py 之前跑一次；已存在就拒絕）。python3 -I -B prereg_write.py"""
import json, hashlib, datetime
from pathlib import Path
H = Path(__file__).resolve().parent
assert not (H / 'PREREG.json').exists(), 'PREREG.json 已存在，拒絕覆寫'
import sys
sys.path.insert(0, str(H))
import s25lib
chk = json.loads((H / 'data_check.json').read_text())
eff = chk['rule_effect_preview']
sha = lambda p: hashlib.sha256((H / p).read_bytes()).hexdigest()
PRE = {
    'topic': 'S25 B 部分：因子自身 12 個月趨勢為負時，衛星新買減半／不新買（錢掃進 0050），能不能讓現行策略更好？',
    'written_at': datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
    'status_at_write': '只跑過 prep.py（data_check.json、trend_features.csv）：各規則觸發月數與少買檔數；沒有算任何策略報酬，也還沒跑 describe.py（A 部分因子動能檢定）。因子趨勢用的是 s24 已存檔的因子月報酬（全期 monthly.csv 已在 S24 公開，非新資訊）。',
    's25lib_sha256_at_write': sha('s25lib.py'), 'prep_sha256_at_write': sha('prep.py'), 'SPEC_sha256': sha('SPEC.json'),
    'baseline（不改定義）': 'S10 的 E0／W10_70（XQ 20＋21 號預設）：嚴成本、只上市、排除創新板、A10 訊號日次日開盤、F1 進場（L_MA 依 SURGE 前 10 名中 RET20 > 10%）、E0 出場（跌出 L_MA 前 20 名）；衛星 70%、每檔＝衛星 ÷ 10；股數 × 開盤 < 目標一半不買；單筆 ≤ 當日成交值 5%；底倉 ±10 個百分點帶狀拉回；0050 停牌日用前一日收盤（s17lib）。SURGE＝STD。引擎 s10lib.simulate_s10，直接 import 重用（s25lib→s20lib→s17lib→s10lib），共用引擎旗標用預設。',
    'scope': '只動衛星新買：規則只改當月買進名單 sig（減半或不買）；選股排序、F1、E0 出場、每檔＝衛星 ÷ 10、底倉 30% 全部不動。沒買的錢照引擎掃進 0050。只在換股日（訊號列）判斷。',
    'feature': '訊號日所在月份的前一個月月底已知的因子趨勢＝該因子最近 12 個月報酬算術和（s24 monthly.csv；RMOM 2016-07 訊號起、MOM 2017-02 訊號起才有值，之前不觸發）。觸發＝趨勢 ≤ 0。門檻固定 0（講義定義），不調參。',
    'rule_form': {'candidates': [dict(r) for r in s25lib.RULES], 'rule_effect_preview（data_check，只數月數與檔數）': eff,
                  'note': '4 條全部計入 N、參與訓練期選擇；沒有對照組規則（S21／S23 的隨機對照在驗證期另做，不參與選擇）。MOM 與 RMOM 趨勢相關 0.68。'},
    'difference_from_past': {
        'S23 市場環境減碼': '同樣只動衛星新買、錢進 0050；S23 用 SURGE 廣度／L_MA 家數／0050 均線，本題改用「策略自己吃的因子（MOM、RMOM）」的 12 個月趨勢，依據是講義的因子動能（Ehsani & Linnainmaa 2022）。S23 的結構限制（衛星股票平均只占總淨值約 43%、F1 98% 月份不滿 10 檔）同樣適用 → 預期效果與代價都小。',
        'S24 歸因': 'S24 顯示現行 MOM 曝險 0.24、RMOM 0.21（主模型），因子逆風時策略可能較差，這是本題的動機。'},
    'windows': {'train': '訊號日 2016-02～2021-12（營收月份 2016-01～2021-11）；訓練窗淨值 2016-01-29 → 2021-12-30（全期連續模擬切出來，同 S20／S21／S23）',
                'valid': '訊號日 2022-01～2026-10（營收月份 2021-12～2026-09）；重新起跑：只保留營收月份 ≥ 2021-12 的訊號，驗證窗 2021-12-30 → 2026-10-22'},
    'paths': '20 條路徑＝訊號日位移 0～3 天 × 起始資金 60／80／100／150／200 萬（同 S20／S21／S23）',
    'selection': '訓練期分數＝規則 − 現行 CUR 的訓練窗年化差，20 條路徑中位數（個百分點）；最高者凍結（同分依改善路徑數、再依觸發月數少者）。'
                 'T0（訓練期最低要求，含改善最小幅度）：分數 ≥ +1 個百分點，且年化差 ≥ +0.25 個百分點的路徑 ≥ 14／20（差 0～0.25 不算改善，避免 S23 那種「幾乎不觸發也算改善」的技術性通過）。'
                 'T0 不過仍凍結分數最高者，驗證期只描述、判定必為不通過。',
    'verdict（驗證期只跑一次，沿用 S21）': {
        'V1': '驗證窗（重新起跑）年化差 > 0 的路徑 ≥ 15／20',
        'V2': '配對差中位數 ≥ +2 個百分點',
        'V3': '禁買 CUR 與凍結規則各自（重新起跑主跑）最賺 5 筆期間的聯集後，配對差中位數 > 0',
        'V4': '兩段空頭（2022；2024-07～2025-04）段內最大回撤配對差中位數都不深於 −2 個百分點',
        'overall': 'T0 且 V1～V4 全過＝通過；否則不通過，停在描述，不換規則重跑、不寫 XQ 修改建議'},
    'acceptance': {'A0': 'CUR 全期主跑重現 W10_70：2019 起 48.13%、全期 34.08%、回撤 −33.40%（±0.025 個百分點）',
                   'A1': '永不觸發的規則（趨勢門檻 −∞）交易與淨值與 CUR 逐筆相同'},
    'random_control（描述，不參與判定）': '驗證期：同窗口同月數隨機觸發同動作，20 個種子（位移 0、100 萬），重新起跑與全期連續各一組',
    'N': '資料窺探族群 N：2,857 → 2,861（4 條候選）',
    'predictions（事前預測）': {
        'P1': '訓練期 4 條的分數都在 −3～+1.5 之間；STOP 版代價比 HALF 大（衛星本來就大半在 0050）',
        'P2': 'RMOM 趨勢 ≤ 0 的月份集中在 2016 下半～2017、2019（訓練）與 2022 全年（驗證）；2022 可能讓 RMOM 規則在驗證期回撤略淺，但 2023 反彈前就恢復買進，代價有限',
        'P3': 'T0 通過機率約 20%；整體通過（T0＋V1～V4）約 5%，最可能卡在 T0 或 V2（+2 很難，觸發只有 6～11 個月）',
        'P4': 'A 部分：台股因子動能方向與講義相同但不顯著（t < 2）；扣同業後選股超額比對 0050 小，但仍為正'},
}
txt = json.dumps(PRE, ensure_ascii=False, indent=1)
(H / 'PREREG.json').write_text(txt)
h = hashlib.sha256(txt.encode()).hexdigest()
(H / 'PREREG.sha256').write_text(f'{h}  PREREG.json\n')
print(h)
