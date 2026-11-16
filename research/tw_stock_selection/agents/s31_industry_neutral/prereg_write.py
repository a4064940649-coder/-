#!/usr/bin/env python3
"""S31 事前登記：在本題算任何報酬之前寫 PREREG.json 與 PREREG.sha256。已存在就拒絕覆寫。python3 -I -B prereg_write.py
（第 0 步 prep.py 只建面板、PIT 產業涵蓋、同業家數、各規則改動名單的檔數與名單層產業集中度，沒有算任何報酬。）"""
import sys, json, hashlib, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
H = Path(__file__).resolve().parent
f = H / 'PREREG.json'
assert not f.exists(), 'PREREG.json 已存在，拒絕覆寫'
import s31lib                                                     # noqa: E402
dc = json.loads((H / 'data_check.json').read_text())
sh = lambda p: hashlib.sha256((H / p).read_bytes()).hexdigest()

PRE = {
 'topic': 'S31 產業中性化：SURGE 改「產業內排名」或「限制單一產業檔數」，能不能更好、更分散？',
 'written_at': datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
 'status_at_write': '只跑過 prep.py（data_check.json）：面板（除權息修正開）、PIT 產業涵蓋、同業家數、各規則改動名單檔數與名單層產業集中度；沒有算任何報酬。',
 'basis': 'S25：扣同業籃子後每筆超額 +6.75%（t 4.00），產業本身貢獻 −0.07% → 選股贏在同業內挑對公司；S21：產業共振篩選失敗（會擋掉自己先爆的大贏家）；'
          '課程單元 03：類股集中是陷阱、尺要對準同業；學術上產業中性化常降低雜訊。',
 's31lib_sha256_at_write': sh('s31lib.py'), 'prep_sha256_at_write': sh('prep.py'),
 'baseline（不改定義）': 'S10 的 E0／W10_70（XQ 20＋21 號預設）：嚴成本、只上市、排除創新板、A10 訊號日次日開盤、F1 進場（L_MA 依 SURGE 前 10 名中 RET20 > 10%）、'
                      'E0 出場（跌出 L_MA 前 20 名）；衛星 70%、每檔＝衛星 ÷ 10；0050 停牌日用前一日收盤（s17lib）。SURGE＝STD（s17lib.rev_family，σ > 0）。'
                      '引擎 s20lib→s17lib→s10lib.simulate_s10，直接 import 重用，不改共用引擎。**cslib.EXRIGHT_FIX = True（除權息參考價修正）**。',
 'industry_data': dict(file='agents/o_lc_sector_rotation_pit/raw/industry_pit.csv（只讀、不複製；用 s21lib.industry_matrix：訊號日當天有效段，「未分類」→ 缺值）',
                       coverage=f"股池產業涵蓋率：訓練期最低 {dc['train']['ind_coverage_min']:.3f}、驗證期最低 {dc['valid']['ind_coverage_min']:.3f}；"
                                f"股池內同業（不含自己）≥ 5 家比例：訓練 {dc['train']['share_universe_peers_ge5']:.3f}、驗證 {dc['valid']['share_universe_peers_ge5']:.3f}；"
                                f"F1 同業 < 5 家：訓練 {dc['train']['F1_peers_lt5']:.3f}、驗證 {dc['valid']['F1_peers_lt5']:.3f}（同業家數中位數 {dc['train']['F1_peers_median']:.0f}／{dc['valid']['F1_peers_median']:.0f}）",
                       caveat='同 S21：2017-07-06～2023-01-16 沒有快照（改類別的公司落後換、2017-07 後新上市用 2023 快照，輕微回填）。'),
 'peer_group': '同業＝訊號列股池 U[k]（只上市、流動性、非創新板）且當月 SURGE 有值、同一 PIT 產業的股票（同 S25 A3 的同業候選股池；排名用途不排除最大市值股）。'
               '同業（不含自己）≥ 5 家才用產業內統計，不足或產業缺值者用全股池統計。',
 'rule_form': {
     'candidates（4 條，全部計入 N；參與訓練期選擇）': s31lib.RULES,
     'common': 'score 類（IND_PCT、DEMED、MIX）：新分數取代 SURGE 做 L_MA 排序（同分依 SURGE），進場＝前 10 名中 RET20 > 10%，出場＝跌出新排序前 20 名'
               '（現行 E0 同一條規則，排序換成新分數；進出場一致，否則依新分數買進的股票會被舊排序立刻賣掉）。'
               'cap 類（CAP2）：只改當月買進名單，遞補只在 L_MA 前 20 名內，出場 E0 不動。L_MA 的 60 日均線條件、RET20 > 10%、每檔金額、衛星比例全部不變。',
     'rule_effect_preview（data_check，只數檔數）': dc['rule_effect_preview'],
 },
 'windows': dict(train='訊號日 2016-02～2021-12（營收月份 2016-01～2021-11）；訓練窗淨值 2016-01-29 → 2021-12-30（全期連續模擬切出，同 S21 train.py）',
                 valid='訊號日 2022-01～2026-10（營收月份 2021-12～2026-09）；重新起跑：只保留營收月份 ≥ 2021-12 的訊號，驗證窗 2021-12-30 → 2026-10-22（同 S21 validate.py）'),
 'paths': '20 條路徑＝訊號日位移 0～3 天 × 起始資金 60／80／100／150／200 萬（同 S21、S25）',
 'selection': '訓練期分數＝規則 − 現行 CUR 的訓練窗年化差，20 條路徑中位數（個百分點）；最高者凍結（同分依改善路徑數、再依改動檔數少者）。'
              'T0（沿用 S25，含改善最小幅度）：分數 ≥ +1 個百分點，且年化差 ≥ +0.25 個百分點的路徑 ≥ 14／20。T0 不過仍凍結分數最高者，驗證期只描述、判定必為不通過。',
 'verdict（驗證期只跑一次，沿用 S25）': dict(
     V1='驗證窗（重新起跑）年化差為正的路徑 ≥ 15／20',
     V2='配對差中位數 ≥ +2 個百分點',
     V3='禁買 CUR 與凍結規則各自（重新起跑主跑）最賺 5 筆期間的聯集後，配對差中位數 > 0',
     V4='兩段空頭（2022；2024-07～2025-04）段內最大回撤配對差中位數都不深於 −2 個百分點',
     overall='T0 且 V1～V4 全過＝通過；否則不通過，停在描述，不換規則重跑、不寫 XQ 修改建議'),
 'diversification（描述，不參與判定）': dict(
     holdings='每日持股依進場訊號日 PIT 產業分組（檔數；W10 每檔等額，近似金額）：單一產業最大占比的日平均（主）、最大產業檔數、產業數、「同產業 ≥ 3 檔」天數比例、單一產業最大占比的日最大值',
     top5='最賺 5 筆占比（s17lib.top5_share：已平倉最賺 5 筆損益 ÷ 淨值增加）',
     list='每月買進名單最大同產業檔數平均、「同產業 ≥ 3 檔」月份比例',
     where='訓練窗：4 條規則全報；驗證窗（重新起跑）與全期連續：CUR 與凍結規則'),
 'random_control（描述，不參與判定）': '凍結規則的「產業標籤隨機打散」對照：每個訊號列把股池內產業代碼隨機重排（各產業家數不變），其餘照凍結規則，20 次（種子 3101～3120）；'
                                   '報驗證窗（重新起跑，100 萬、位移 0）與全期 2019 起年化分布，以及凍結規則落在第幾名。用來分辨「真的產業」與「任意分組」。',
 'acceptance': dict(A0='CUR 全期主跑（除權息修正開）重現：2019 起 48.99%、全期 34.54%、回撤 −32.11%（±0.025 個百分點）、已平倉 381 筆',
                    A1='恆等分數（score＝SURGE）的 build_rule 與 make_signal_s10 的 sig、X 逐格相同（prep 已驗：' + json.dumps(dc['A1_identity_builder']) + '），且淨值與交易逐筆相同',
                    A2='build_rule 每個訊號列重現的 F1 與 make_signal_s10 完全相同（assert）'),
 'N': '資料窺探族群：S25 完成時 2,861（S26～S30 並行，以研究中心彙整為準）；本題 +4（4 條候選；隨機對照不計）',
 'predictions（事前預測）': dict(
     P1='IND_PCT 改動最大（訓練期換掉約 35% 的 F1、前 20 名每月約 7 檔不同），很可能為負：會把熱門產業第 2、3 強（常是真正的大贏家，如半導體、電子零組件）換成冷門產業的第 1 名',
     P2='DEMED、MIX 介於 −2～+1；CAP2 改動很小（訓練期 10 個月、16 檔），分數接近 0，難以達到 +1',
     P3='分散度：IND_PCT、MIX 單一產業最大占比明顯下降；最賺 5 筆占比不一定下降（S21 經驗：換股反而更集中）',
     P4='整體通過機率約 10%；最可能卡在 T0'),
}
txt = json.dumps(PRE, ensure_ascii=False, indent=1)
f.write_text(txt)
sha = hashlib.sha256(txt.encode()).hexdigest()
(H / 'PREREG.sha256').write_text(f'{sha}  PREREG.json\n')
(H / 'code_prereg.sha256').write_text(''.join(f'{sh(p)}  {p}\n' for p in ('s31lib.py', 'prep.py', 'prereg_write.py')))
print(sha)
