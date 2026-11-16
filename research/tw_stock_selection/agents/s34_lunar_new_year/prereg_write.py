#!/usr/bin/env python3
"""S34 事前登記：在本題算任何策略報酬之前寫 PREREG.json 與 PREREG.sha256。已存在就拒絕覆寫。python3 -I -B prereg_write.py
（第 0 步 prep.py 只做描述：春節日期、交易日、SURGE ≥ 2 比例、F1 名單變動、各規則改動名單檔數；沒有算任何報酬。
 引擎驗收（CUR 主跑 48.99／34.54／−32.11／381）在計時測試時跑過一次，只看驗收數字。）"""
import sys, json, hashlib, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
H = Path(__file__).resolve().parent
f = H / 'PREREG.json'
assert not f.exists(), 'PREREG.json 已存在，拒絕覆寫'
import s34lib                                                     # noqa: E402
dc = json.loads((H / 'data_check.json').read_text())
lib_sha = hashlib.sha256((H / 's34lib.py').read_bytes()).hexdigest()

PRE = {
 'topic': 'S34 農曆年（春節）讓 1、2 月營收年增金額失真：SURGE 在這兩個月會不會選錯股？事前定義的校正能不能讓策略更好？',
 'written_at': datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
 'status_at_write': '只跑過 prep.py（data_check.json、describe_months.csv.gz：描述，不含報酬）與一次引擎驗收（只看 CUR 驗收數字）。',
 's34lib_sha256_at_write': lib_sha,
 'prior_work_check': 'S17 定義家族的 SM3（全年每月都用近 3 個月合計）是全年平滑，不是針對春節的校正；BACKLOG／S17／S2 沒有做過 1、2 月或春節校正 → 本題不重複。',
 'baseline（不改定義）': 'S10 的 E0／W10_70（XQ 20＋21 號預設）：嚴成本、只上市、排除創新板、A10 訊號日次日開盤、F1 進場（L_MA 依 SURGE 前 10 名中 RET20 > 10%）、'
                      'E0 出場（跌出 L_MA 前 20 名）；衛星 70%、每檔＝衛星 ÷ 10；0050 停牌日用前一日收盤（s17lib）。SURGE＝STD（s17lib.rev_family，σ > 0）。'
                      '引擎 s20lib→s17lib→s10lib.simulate_s10，直接 import 重用，不改共用引擎。cslib.EXRIGHT_FIX＝True。',
 'lny_table': dict(dates=s34lib.LNY, source=s34lib.LNY_SOURCE, use='只用於描述；規則只看營收的日曆月份（1、2、3 月）與交易日數（加權指數）。'),
 'description_summary（prep，未含報酬）': dict(
     ge2_by_month_group={k: dict(mean=v['ge2_mean'], min=v['ge2_min'], max=v['ge2_max'], sd=v['ge2_sd']) for k, v in dc['surge_by_month_group'].items()},
     ge2_by_tdratio_bucket=dc['ge2_by_tdratio_bucket'], corr_logtdratio_ge2=dc['corr_logtdratio_ge2'],
     note='WD（交易日比例全額校正）在描述階段已看到過度校正（交易日少的月份 SURGE ≥ 2 比例 4.0% → 24.6%）；仍照教科書定義登記、不調彈性，預測它最差。'),
 'rule_form': {
     'candidates（4 條，全部計入 N；參與訓練期選擇）': s34lib.RULES,
     'common': 'surge 類規則改的是 SURGE 定義本身（D 序列整條替換後重算 z 分數），所以 F1 進場與 E0 出場（L_MA 前 20 名）一起用校正後的 SURGE；'
               '只有 JF_SKIP 的 1 月營收訊號列整列缺值（make_signal_s10 跳過該列＝不買不賣）。NOBUY 只清 1、2 月營收訊號列的買進，出場照現行。',
     'rule_effect_preview（data_check，只數檔數）': dc['rule_effect_preview'],
 },
 'windows': dict(train='訊號日 2016-02～2021-12（營收月份 2016-01～2021-11）；訓練窗淨值 2016-01-29 → 2021-12-30（全期連續模擬切出來，同 S20／S21／S31）',
                 valid='訊號日 2022-01～2026-10（營收月份 2021-12～2026-09）；重新起跑：只保留營收月份 ≥ 2021-12 的訊號，驗證窗 2021-12-30 → 2026-10-22；起始資金縮放使 2021-12-30 淨值約＝各路徑資金'),
 'paths': '20 條路徑＝訊號日位移 0～3 天 × 起始資金 60／80／100／150／200 萬（同 S20／S21／S31）',
 'selection': '訓練期分數＝規則 − 現行 CUR 的訓練窗年化差，20 條路徑中位數（個百分點）；最高者凍結（同分依改善路徑數、再依改動檔數少者）。'
              'T0（沿用 S25，含改善最小幅度）：分數 ≥ +1 個百分點，且年化差 ≥ +0.25 個百分點的路徑 ≥ 14／20。T0 不過仍凍結分數最高者，驗證期只描述、判定必為不通過。',
 'verdict（驗證期只跑一次，沿用 S25）': dict(
     V1='驗證窗（重新起跑）年化差為正的路徑 ≥ 15／20',
     V2='配對差中位數 ≥ +2 個百分點',
     V3='禁買 CUR 與凍結規則各自（重新起跑主跑）最賺 5 筆期間的聯集後，配對差中位數 > 0',
     V4='兩段空頭（2022；2024-07～2025-04）段內最大回撤配對差中位數都不深於 −2 個百分點',
     overall='T0 且 V1～V4 全過＝通過；否則不通過，停在描述，不換規則重跑、不寫 XQ 修改建議'),
 'affected_months_trade_level（事前登記、只描述、不參與判定）': dict(
     rows='受影響訊號列＝營收月份 1、2 月（ADJ2 另加 3 月）；位移 0；分訓練（營收月份 ≤ 2021-11）與驗證（≥ 2021-12）。',
     outcome='每檔 E0 持有期報酬減同期 0050（s20lib.outcomes：次日開盤進場、依各自名單的出場矩陣、跌停順延、下市結算），扣成本前。',
     compare='每個受影響月份：校正後名單平均超額 − 校正前（現行）名單平均超額；被跳過或不新買的月份，校正後名單＝空（超額 0，等於放 0050）。群聚 t＝逐月差的平均 ÷（標準差 ÷ √月數）。'
             '另報兩邊共同的股票、只在現行名單的股票、只在校正後名單的股票三組平均超額。'),
 'random_control（事前登記、只描述、不參與判定）': dict(
     how='對凍結規則的機制：每年隨機挑相鄰兩月 (a, a＋1)，a ∈ 3～10（不碰 12、1、2 月），套用同機制取代 1、2 月：'
         'JF_SKIP→兩月 D 平均且第一月跳過換股；ADJ2→a、a＋1、a＋2 三月改「當月與上月平均」；WD→兩月套交易日比例；NOBUY→兩月不新買。',
     n=200, seeds='20261116＋0～199',
     metrics='訓練窗與驗證窗（重新起跑）20 條路徑配對差中位數的分布；報凍結規則在分布中的百分位。用來區分「春節校正」與「隨便動兩個月」。'),
 'acceptance': dict(A0='CUR 全期主跑：2019 起 48.99%、全期 34.54%、回撤 −32.11%（±0.025 個百分點）、已平倉 381 筆',
                    A1="mode='none' 的 SURGE 與 s17lib STD 逐格相同（prep 已驗）；以它建的訊號跑出的交易與淨值和 CUR 逐筆相同"),
 'N': '資料窺探族群 N：＋4（4 條候選；隨機對照不計）',
 'predictions（事前預測）': dict(
     P1='受影響的訊號列只有每年 2 個（ADJ2 3 個），全期年化差多半在 ±2 個百分點內；T0 通過機率約 20%。',
     P2='WD 最差（描述階段已看到過度校正）；JF_SKIP 與 ADJ2 訓練期接近 0；NOBUY 小幅為負（1、2 月 F1 平均約 4 檔，少買的錢放 0050）。',
     P3='逐筆比較：交易日多的 1 月（2018、2021、2024、2026）現行名單裡「只在現行」的股票超額較差（被假性爆發灌進來），但月數少，群聚 t 不會 > 2。',
     P4='整體通過（T0＋V1～V4）機率約 5%；最可能卡在 T0 或 V2（年化差太小）。'),
}
txt = json.dumps(PRE, ensure_ascii=False, indent=1, default=str)
f.write_text(txt)
sha = hashlib.sha256(txt.encode()).hexdigest()
(H / 'PREREG.sha256').write_text(f'{sha}  PREREG.json\n')
code = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(H.glob('*.py'))}
(H / 'code_prereg.sha256').write_text(''.join(f'{v}  {k}\n' for k, v in code.items()))
print(sha)
