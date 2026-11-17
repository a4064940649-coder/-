#!/usr/bin/env python3
"""S44 事前登記：在本題算任何策略報酬之前寫 PREREG.json 與 PREREG.sha256。已存在就拒絕覆寫。python3 -I -B prereg_write.py
（第 0 步 prep.py 只做描述：L_MA 第 10 名 SURGE 分布、F1 中 SURGE < 2／< 2.5、F1 在全股池的分位、各規則剔除檔數；沒有算任何報酬。
 引擎驗收（CUR 主跑 48.99／34.54／−32.11／381）放在 train.py 開頭，只看驗收數字。）"""
import sys, json, hashlib, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
H = Path(__file__).resolve().parent
f = H / 'PREREG.json'
assert not f.exists(), 'PREREG.json 已存在，拒絕覆寫'
import s44lib                                                     # noqa: E402
dc = json.loads((H / 'data_check.json').read_text())
lib_sha = hashlib.sha256((H / 's44lib.py').read_bytes()).hexdigest()

PRE = {
 'topic': 'S44 名次＋最低 SURGE 門檻：F1（L_MA 前 10 名且 RET20 > 10%）再要求 SURGE 不低於下限，能不能挑掉全市場營收平淡月份的「假性爆發」、讓策略更好？',
 'written_at': datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
 'status_at_write': '只跑過 prep.py（data_check.json、describe_months.csv.gz、describe_f1.csv.gz：描述，不含任何報酬）。',
 's44lib_sha256_at_write': lib_sha,
 'prior_work_check': ('S19（XQ 門檻版）只用固定門檻 2.5／2.0、不看名次（XU−XR＝−2.06 個百分點），是「門檻取代名次」，會買到第 11～20 名；'
                      'S4／S17／S18 只把門檻當 XQ 近似名次的工具；S20 篩 F1 用的是 μ÷σ 不是 SURGE 水準；S34 改的是 SURGE 定義。'
                      'BACKLOG／agents 全文搜尋沒有「名次＋門檻雙條件」（F1 且 SURGE ≥ θ）的回測 → 本題不重複。'),
 'baseline（現行，不改）': ('S10 的 E0／W10_70（XQ 20＋21 號預設）：嚴成本、只上市、排除創新板、A10 訊號日次日開盤、F1 進場（L_MA 依 SURGE 前 10 名中 RET20 > 10%，不遞補）、'
                         'E0 出場（跌出 L_MA 前 20 名）；衛星 70%、每檔＝衛星 ÷ 10；0050 停牌日用前一日收盤。SURGE＝STD（s17lib.rev_family，σ > 0）。'
                         '引擎 s20lib→s17lib→s10lib.simulate_s10，直接 import 重用，不改共用引擎。cslib.EXRIGHT_FIX＝True。'),
 'description_summary（prep，未含報酬）': dict(
     s10_dist=dc['s10_dist'], f1_frac_lt2=dc['f1_surge']['all']['frac_lt2'], f1_frac_lt25=dc['f1_surge']['all']['frac_lt25'],
     f1_pct=dc['f1_surge']['all']['pct'], rule_effect_preview=dc['rule_effect_preview'],
     note=('F1 幾乎全部落在當月全股池第 95 百分位以上（只有 0.8% 低於），所以使用者舉例的 P95 幾乎不剔除任何股票；'
           '在看過這個描述（只有檔數、沒有報酬）後加登記 P97.5 一條，讓相對門檻的剔除量（訓練期 12%）和 ABS_2.5（16%）同一量級。P99 會剔除一半 F1，是另一個策略，不登記。')),
 'rule_form': {
     'candidates（5 條，全部計入 N；R1 的 θ 只能在訓練期從 {1.5, 2.0, 2.5} 選）': s44lib.RULES,
     'common': ('只改進場：F1 目標中不合規則的不買，不遞補第 11 名以後，空出的名額與資金照引擎掃進 0050；出場 E0（跌出 L_MA 前 20 名）完全不動，'
                '已持有的股票即使當月 SURGE 低於門檻也不賣。SURGE 定義、股池、名額 10、W10 都不動。'),
     'pool_definition': '當月全股池＝訊號列 k 的股池（stlib.universe twse & ~創新板）且當月 SURGE 有值（即 make_signal_s10 套季線前的 fin 集合）；百分位用 numpy.quantile linear，每個訊號列（含位移）各自算。',
 },
 'windows': dict(train='訊號日 2016-02～2021-12（營收月份 2016-01～2021-11）；訓練窗淨值 2016-01-29 → 2021-12-30（全期連續模擬切出來，同 S20／S21／S34）',
                 valid='訊號日 2022-01～2026-10（營收月份 2021-12～2026-09）；重新起跑：只保留營收月份 ≥ 2021-12 的訊號，驗證窗 2021-12-30 → 2026-10-22；起始資金縮放使 2021-12-30 淨值約＝各路徑資金'),
 'paths': '20 條路徑＝訊號日位移 0～3 天 × 起始資金 60／80／100／150／200 萬（同 S20／S21／S34）',
 'selection': ('訓練期分數＝規則 − 現行 CUR 的訓練窗年化差，20 條路徑中位數（個百分點）；5 條中最高者凍結（同分依 ≥ +0.25 的路徑數、再依剔除檔數少者）。'
               'T0（沿用 S25，含改善最小幅度）：分數 ≥ +1 個百分點，且年化差 ≥ +0.25 個百分點的路徑 ≥ 14／20。T0 不過仍凍結分數最高者，驗證期只描述、判定必為不通過。'),
 'verdict（驗證期只跑一次，沿用 S25）': dict(
     V1='驗證窗（重新起跑）年化差為正的路徑 ≥ 15／20',
     V2='配對差中位數 ≥ +2 個百分點',
     V3='禁買 CUR 與凍結規則各自（重新起跑主跑）最賺 5 筆期間的聯集後，配對差中位數 > 0',
     V4='兩段空頭（2022；2024-07～2025-04）段內最大回撤配對差中位數都不深於 −2 個百分點',
     overall='T0 且 V1～V4 全過＝通過；否則不通過，停在描述，不換門檻重跑、不寫 XQ 修改建議'),
 'random_control（事前登記、只描述、不參與判定）': dict(
     how='對凍結規則：每個訊號月從 F1 目標中隨機剔除「與規則同檔數」的股票（逐月配對檔數），其餘同規則（不遞補、放 0050）。',
     n=200, seeds='20261117＋0～199', path='主跑 100 萬、位移 0',
     metrics='訓練窗（全期連續切出）與驗證窗（重新起跑）年化差（隨機 − CUR）的分布；報凍結規則在分布中的百分位。用來區分「挑掉弱訊號」與「少買幾檔、多放 0050」。'),
 'removed_trade_level（事前登記、只描述、不參與判定）': dict(
     rows='F1 名單（位移 0）逐筆；分訓練（營收月份 ≤ 2021-11）與驗證（≥ 2021-12）；對 5 條規則各自分「留下／剔除」。',
     outcome='每檔 E0 持有期報酬減同期 0050（s20lib.outcomes：次日開盤進場、E0 出場、跌停順延、下市結算），扣成本前。',
     stats='剔除組與留下組的筆數、平均、中位數、勝率、最大／最小；同月群聚差（剔除組 − 當月 F1 全部，s20lib.month_cluster_diff）與群聚 t；被剔除的最賺 5 筆列出。'),
 'acceptance': dict(A0='CUR 全期主跑：2019 起 48.99%、全期 34.54%、回撤 −32.11%（±0.025 個百分點）、已平倉 381 筆',
                    A1='prep：f1_rows 的 F1 名單與 make_signal_s10 info 相同（已驗 true）',
                    A2='門檻 θ＝−∞ 的規則跑出的交易與淨值和 CUR 逐筆相同'),
 'N': '資料窺探族群 N：＋5（ABS 三個 θ＋REL 兩個百分位；隨機對照不計）',
 'predictions（事前預測）': dict(
     P1='ABS_1.5 訓練期剔除 0 檔 → 分數 0；ABS_2.0 只剔除 7 檔、REL_P95 2 檔 → 分數在 ±1 內。',
     P2='ABS_2.5 與 REL_P975 剔除 12～16%，訓練期分數多半為負（S19：門檻版在分數低的年份錯過 2482 這類股票；S20：篩 F1 多半是持股變少的機械效果）。',
     P3='T0 通過機率約 15%；整體通過機率約 5%。',
     P4='被剔除股票的逐筆超額平均略低於留下組，但群聚 t 不會 > 2（S19 的「SURGE < 2.5 的前 10 名」81 筆平均 +1.41%，不差）。'),
}
txt = json.dumps(PRE, ensure_ascii=False, indent=1, default=str)
f.write_text(txt)
sha = hashlib.sha256(txt.encode()).hexdigest()
(H / 'PREREG.sha256').write_text(f'{sha}  PREREG.json\n')
code = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(H.glob('*.py'))}
(H / 'code_prereg.sha256').write_text(''.join(f'{v}  {k}\n' for k, v in code.items()))
print(sha)
