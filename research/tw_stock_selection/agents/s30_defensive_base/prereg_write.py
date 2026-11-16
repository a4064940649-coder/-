#!/usr/bin/env python3
"""S30 事前登記：在本題算任何防禦組合或新規則的報酬之前寫 PREREG.json 與 PREREG.sha256。已存在就拒絕覆寫。python3 -I -B prereg_write.py
（第 0 步 prep.py 只做引擎驗收、包裝逐位相同驗收、防禦組合持股檔數／重疊／事前 β；沒有算任何組合或規則的報酬。）"""
import sys, json, hashlib, datetime
from pathlib import Path
H = Path(__file__).resolve().parent
f = H / 'PREREG.json'
assert not f.exists(), 'PREREG.json 已存在，拒絕覆寫'
dc = json.loads((H / 'data_check.json').read_text())
sha = lambda p: hashlib.sha256((H / p).read_bytes()).hexdigest()

PRE = {
 'topic': "S30 底倉與空位換成「低 β＋高品質」防禦組合（Buffett's Alpha：Frazzini、Kabiller、Pedersen 2018；Asness 等 QMJ）：能不能讓 W10_70 跌少一點而不犧牲太多報酬？",
 'written_at': datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
 'status_at_write': '只跑過 prep.py（data_check.json）：引擎驗收 A0、包裝驗收 A1、防禦組合每季持股（檔數、重疊、事前 β、ROE／殖利率涵蓋）；沒有算任何防禦組合或新規則的報酬。',
 'code_sha256_at_write': {p: sha(p) for p in ('s30lib.py', 'prep.py')},
 'prior_knowledge（誠實揭露）': '研究中心過去做過 w12_betting_against_beta（低 β 只做多：驗證期 2019-05～2026-10 年化 9.96% 對 0050 28.27%；2024～2025 只有 1.3%）、'
     'w3_dividend_lowvol、w12_conservative_formula（低波動＋殖利率：2022 年明顯抗跌、2024 年起大輸 0050 30 個百分點以上）。研究員寫 PREREG 前已讀過這些結論；'
     '所以本題的組合與規則沒有在 2016 起的資料上試過，但「低 β 在 2024 起大輸大盤」是已知的事前資訊，反映在下面的預測。',
 'baseline（不改定義）': '現行 E0｜W10_70（XQ 20＋21 號預設）：嚴成本、只上市、排除創新板、A10 訊號日次日開盤、F1 進場、E0 出場；衛星 70%、每檔＝衛星 ÷ 10；'
     '底倉 ±10 個百分點帶狀拉回；衛星沒用完的錢（空位）也放 0050；0050 停牌日用前一日收盤（s17lib）。cslib.EXRIGHT_FIX＝True。引擎 s10lib.simulate_s10（直接 import，不改）。'
     f"驗收 A0：{json.dumps({k: v for k, v in dc['acceptance']['A0'].items()}, ensure_ascii=False)}",
 'engine_wrapper': '引擎把「底倉＋空位」當成同一個 0050 部位（v50），價格來自 P.E50（收盤）、P.E50O（開盤）；資金進出成本＝手續費＋0.1% 證交稅＋半個 0050 跳動點。'
     '本題不改引擎，在 s30lib.base_swap() 模擬期間暫時把這四個陣列換成防禦組合指數，並把引擎內的 0050 進出成本換成組合成本（證交稅 0.3%、每邊滑價 0.4%；MIX 為 0.2%／0.21%），'
     '離開時還原。所以底倉與空位一起換（引擎無法分開），衛星選股、出場、每檔金額、帶狀拉回全部不動。'
     f"驗收 A1（傳入 0050 陣列複本＝逐位相同）：{json.dumps(dc['acceptance']['A1'], ensure_ascii=False)}",
 'defensive_baskets（事前定義 3 個＋0050 基準）': {
     'common': '每季最後一個交易日 d（2015-12-31 起，44 次）形成；股池＝上市、d 有成交、非 91、非創新板、PIT 市值（股數落後 63 交易日，同 s24）前 150 名、'
               '過去 252 交易日有效日報酬 ≥ 200 天。β＝過去 252 日日報酬對 0050 含息日報酬 OLS 斜率（只用有成交日；剔除 0050 分割停牌日）。'
               'd 的下一個交易日收盤再平衡成等權；成本＝手續費 0.1425%×0.28＋賣出稅 0.3%＋滑價（半個跳動點＋0.3%）。'
               '第一次形成前（2015-07-01～2016-01-04 收盤）＝0050，第一次換倉計「賣 0050＋買組合」成本。成分股停牌、下市 → 報酬 0（價值凍結到下次再平衡）。',
     'LOWB': '低 β：β 最低 30 檔等權（Frazzini–Pedersen 低 β 腿）',
     'LBQ': '低 β ∩ 高品質：β 最低一半（75 檔）中，最近已公告 TTM ROE（歸屬母公司淨利 ÷ 歸屬母公司權益，PIT 可得日同 s24）最高 30 檔等權（課程「Buffett 的 Alpha」主組合）',
     'HDLV': '高股息低波動：252 日波動最低一半（75 檔）中，過去 365 天 TWT49U「息／權息」事件（權值＋息值）÷ 除權息前收盤 合計最高 30 檔等權',
     'MIX': '50% 0050＋50% LBQ，每月最後交易日收盤拉回 50／50（移轉金額計雙邊成本）',
     'E50（基準）': '0050 含息（現行）',
     'ex_ante_stats（data_check，事前值）': dict(avg_beta=dict(LOWB=dc['per_formation_summary']['beta_LOWB'], LBQ=dc['per_formation_summary']['beta_LBQ'],
                                                         HDLV=dc['per_formation_summary']['beta_HDLV'], top150_median=dc['per_formation_summary']['beta_med_top']),
                                              overlap=dc['overlap_avg_names'], new_names_per_quarter=dc['name_turnover']),
 },
 'rules（4 條，全部計入 N；參與訓練期選擇）': {
     'R_LOWB': '底倉＋空位＝LOWB（取代 0050）',
     'R_LBQ': '底倉＋空位＝LBQ',
     'R_HDLV': '底倉＋空位＝HDLV',
     'R_MIX': '底倉＋空位＝MIX（一半 0050、一半 LBQ）',
     'common': '只換底倉與空位持有的東西；衛星 70%、每檔＝衛星 ÷ 10、選股、出場、±10 個百分點帶狀拉回全部與現行相同。',
 },
 'windows': dict(train='訊號日 2016-02～2021-12；訓練窗淨值 2016-01-29 → 2021-12-30（全期連續模擬切出來，同 S20／S21）',
                 valid='訊號日 2022-01～2026-10；重新起跑（只保留營收月份 ≥ 2021-12 的訊號；起始資金依各設定自己在 2021-12-30 的淨值縮放成 60～200 萬），驗證窗 2021-12-30 → 2026-10-22'),
 'paths': '20 條路徑＝訊號日位移 0～3 天 × 起始資金 60／80／100／150／200 萬（同 S20／S21）',
 'selection_and_T0': dict(
     T0='訓練窗最大回撤較現行淺 ≥ 0.5 個百分點的路徑 ≥ 14／20，且訓練窗年化差（規則 − 現行）20 條中位數 ≥ −1 個百分點',
     freeze='T0 通過的規則中，選訓練窗回撤改善（20 條中位數）最大者；同分依年化差中位數大者。沒有規則通過 T0 → 凍結「回撤較淺 ≥ 0.5 的路徑數」最多者（同分依年化差中位數），驗證期只描述，判定必為不通過。'),
 'verdict（驗證期只跑一次）': dict(
     V1='驗證窗（重新起跑）最大回撤較現行淺 ≥ 0.5 個百分點的路徑 ≥ 15／20',
     V2='驗證窗年化差（凍結 − 現行）20 條中位數 ≥ −2 個百分點',
     V3='禁買現行與凍結規則各自（重新起跑主跑、位移 0、100 萬）最賺 5 筆期間的聯集後，驗證窗年化差中位數 ≥ −2 個百分點',
     V4='兩段空頭（2022；2024-07～2025-04）段內最大回撤：凍結 − 現行 20 條中位數都 ≥ +0.5 個百分點（較淺）',
     overall='T0 且 V1～V4 全過＝通過；否則不通過，停在描述，不換組合、不換規則重跑、不寫操作建議'),
 'also_report（描述，不進判定）': [
     '各防禦組合單獨：2016-01-29 起與訓練窗、驗證窗的年化、年化波動、最大回撤、對 0050 日報酬 CAPM β、四段空頭跌幅、每季換手與成本（describe.py；在 FROZEN 之後才跑）',
     'CAPM β 變化：現行與凍結規則的淨值日報酬（與月報酬）對 0050 的 β，全期連續與驗證窗重新起跑',
     '全期連續主跑：2019 起、全期、最大回撤、逐年、四段空頭、最賺 5 筆占比、成本加倍'],
 'posthoc（未登記，只描述）': '其餘 3 條規則的驗證窗與全期數字（看過凍結結果後才算，不能用來改判定）；底倉比例敏感度不做。',
 'N': '資料窺探族群 N：2,861 → 2,865（4 條規則；3 個組合是規則的一部分，不另計）',
 'predictions（事前預測）': dict(
     P1='三個組合單獨 2016 起年化都低於 0050（約 10%～15% 對 0050 約 20%），波動與 β 明顯較低（β 約 0.4～0.6），2022 年跌得比 0050 少',
     P2='訓練窗：R_LOWB、R_LBQ、R_HDLV 年化差約 −2～−5 個百分點（底倉約占六成 × 組合落後 0050），回撤 2018、2020-03 可能較淺；T0 年化條件（≥ −1）最可能失敗，只有 R_MIX 有機會',
     P3='T0 通過機率約 30%；整體通過（T0＋V1～V4）機率約 5%：驗證期 2024 起 0050 每年 50% 以上，低 β 組合大幅落後，V2 幾乎一定失敗',
     P4='CAPM β 由約 0.97 降到 0.6～0.8（R_MIX 約 0.8）'),
}
txt = json.dumps(PRE, ensure_ascii=False, indent=1)
f.write_text(txt)
s = hashlib.sha256(txt.encode()).hexdigest()
(H / 'PREREG.sha256').write_text(f'{s}  PREREG.json\n')
print(s)
