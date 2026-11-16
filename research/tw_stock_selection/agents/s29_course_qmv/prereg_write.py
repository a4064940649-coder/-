#!/usr/bin/env python3
"""S29 事前登記：在本題算任何報酬之前寫 PREREG.json 與 PREREG.sha256。已存在就拒絕覆寫。python3 -I -B prereg_write.py
（第 0 步 prep.py 只建面板、Q／M／V 涵蓋率與分位數、各規則改動名單的檔數，沒有算任何報酬。）"""
import sys, json, hashlib, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
H = Path(__file__).resolve().parent
f = H / 'PREREG.json'
assert not f.exists(), 'PREREG.json 已存在，拒絕覆寫'
import s29lib                                                     # noqa: E402
dc = json.loads((H / 'data_check.json').read_text())
sha = lambda p: hashlib.sha256((H / p).read_bytes()).hexdigest()

PRE = {
 'topic': 'S29 課程 QMV（品質、動能、估值）能不能改進 SURGE 的選股排序？',
 'written_at': datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
 'status_at_write': '只跑過 prep.py（data_check.json）：面板、Q／M／V 涵蓋率、分位數、相關、各規則改動名單檔數；沒有算任何報酬。'
                    '寫入前唯一一次規則修改：FILL_QMV 原為「依 QMV 排序遞補」，data_check 顯示與 FILL_ALL 只差 19 個月 24 檔（幾乎相同），改為「只遞補 QMV ≥ 0.5 者」，仍未看任何報酬。',
 'code_sha256_at_write': {p: sha(p) for p in ('s29lib.py', 'prep.py')},
 'baseline（不改定義）': 'S10 的 E0／W10_70（XQ 20＋21 號預設），cslib.EXRIGHT_FIX＝True：嚴成本、只上市、排除創新板、A10 訊號日次日開盤、F1 進場（L_MA 依 SURGE 前 10 名中 RET20 > 10%）、'
                      'E0 出場（跌出 L_MA 前 20 名）；衛星 70%、每檔＝衛星 ÷ 10；0050 停牌日用前一日收盤（s17lib）。SURGE＝STD（σ > 0）。引擎 s20lib→s17lib→s10lib，直接 import。',
 'scores': dict(
     Q='ROE＝TTM 歸屬母公司淨利（連續 4 季）÷ 最近一季歸屬母公司權益（權益 ≤ 0 → 缺值）。季報＝o_pead_margin_surprise_low_attention/data/fin_long.parquet（MOPS XBRL 單季），'
       '取市場可得日 A（avail_dates.parquet；缺 → 法定期限 5/15、8/14、11/14、3/31）≤ 訊號日的最新一季（PIT，同 S11／S24）。ROIC 需要有息負債與現金，本機沒有 → 用 ROE（Hou-Xue-Zhang q 因子）。',
     M='12-1 個股動能＝含息累積指數 G[k−21] ÷ G[k−252] − 1（k＝訊號列，跳過最近約一個月；k−252 時須已掛牌）。',
     V='E/P＝TTM 歸屬母公司淨利 ÷ PIT 市值（shares_full 股數落後 63 交易日 × 收盤，同 S24／S25）；虧損股 E/P 為負、排最貴。EV/EBIT 需要負債與現金，沒有 → 用 E/P。',
     pct='同期百分位：訊號列 k 的股池 U[k]（只上市、流動性、非創新板）內有值者排名，0～1，越大越好（高 ROE、強動能、便宜）。',
     QMV='(pQ＋pM＋pV) ÷ 3，缺值分項以 0.5 代入；QM＝(pQ＋pM) ÷ 2。',
     coverage='F1 上涵蓋率：訓練 Q 97.2%、M 99.6%、V 97.2%；驗證 Q 97.5%、M 98.8%、V 97.5%（data_check.json）→ 三分數都夠，不縮小。'),
 'rule_form': {
     'candidates（5 條，全部計入 N；參與訓練期選擇）': s29lib.RULES,
     'control（1 條，只描述，不參與選擇與判定）': s29lib.CONTROL,
     'common': '所有規則只改當月買進名單；出場 X（E0）不動。遞補與重排只在 L_MA 前 20 名內進行，所以新買的股票同樣在跌出前 20 名時賣出。篩掉的名額不遞補，錢照引擎掃進 0050。',
     'rule_effect_preview（data_check，只數檔數）': dc['rule_effect_preview'],
 },
 'windows': dict(train='訊號日 2016-02～2021-12（營收月份 2016-01～2021-11）；訓練窗淨值 2016-01-29 → 2021-12-30（全期連續模擬切出來，同 S20／S21）',
                 valid='訊號日 2022-01～2026-10（營收月份 2021-12～2026-09）；重新起跑：只保留營收月份 ≥ 2021-12 的訊號，驗證窗 2021-12-30 → 2026-10-22'),
 'paths': '20 條路徑＝訊號日位移 0～3 天 × 起始資金 60／80／100／150／200 萬（同 S20／S21／S25）',
 'selection': '訓練期分數＝候選規則 − 現行（CUR）的訓練窗年化差，20 條路徑的中位數（個百分點）；分數最高者凍結（同分依改善路徑數、再依改動檔數少者）。'
              'T0（沿用 S25，含改善最小幅度）：分數 ≥ +1 個百分點，且年化差 ≥ +0.25 個百分點的路徑 ≥ 14／20。T0 不過仍凍結分數最高者，驗證期只描述、判定必為不通過。',
 'verdict（驗證期只跑一次，沿用 S25）': dict(
     V1='驗證窗（重新起跑）年化差為正的路徑 ≥ 15／20',
     V2='配對差中位數 ≥ +2 個百分點',
     V3='禁買 CUR 與凍結規則各自（重新起跑主跑）最賺 5 筆期間的聯集後，配對差中位數 > 0',
     V4='兩段空頭（2022；2024-07～2025-04）段內最大回撤配對差中位數都不深於 −2 個百分點',
     overall='T0 且 V1～V4 全過＝通過；否則不通過，停在描述，不換規則重跑；通過才寫 XQ 20 號修改建議（只寫建議，不改檔）'),
 'acceptance': dict(A0='CUR 全期主跑（EXRIGHT_FIX＝True）重現：2019 起 48.99%、全期 34.54%、回撤 −32.11%（±0.025 個百分點）、381 筆',
                    A1='build_variant 每個訊號列重現的 F1 與 make_signal_s10 完全相同（assert）',
                    A2='「全部通過」的篩選規則（FLT_V 門檻 −1）交易與淨值與 CUR 逐筆相同'),
 'describe（不論通過與否都做，未參與判定）': 'F1 逐筆（s20lib.outcomes：E0 持有期，不扣成本）對 0050 超額與對同業超額（同 S25：PIT 產業、股池 U、排除自己與產業最大市值股、同業 ≥ 5 家，等權），'
                                       '依 pQ、pM、pV、QMV 三分組（同期股池百分位 < 1/3、1/3～2/3、≥ 2/3）的平均、中位數、勝率、月份群聚 t；高組 − 低組的月份群聚差；逐月等級相關（IC）平均與 t。訓練、驗證、全期分列。'
                                       '另做隨機對照：凍結規則的分數換成每月獨立均勻亂數（規則機制不變），20 次。',
 'N': '資料窺探族群 N 加 6（5 條候選＋1 條對照）；S25 後為 2,861，S26～S28 由研究中心另計。',
 'predictions（事前預測）': dict(
     P1='兩條純篩選（FLT_Q 剔除約 25%、FLT_V 剔除約 33% 的 F1、錢放 0050）訓練期為負（S20 機械效果）；FLT_V 最差（S20 由衰轉盛的大贏家多半虧損或高本益比）',
     P2='最好的是 RANK_QMV／RANK_QM 之一，分數 −2～+2；T0 通過機率約 20%',
     P3='FILL_QMV 比 FILL_ALL 好，但兩者都不會比 CUR 好很多（S21 FILL_ALL −0.21）',
     P4='描述：M（12-1）在 F1 內最有用的機率最高但弱（S24 動能因子 +4.1），V 可能反向（高估值的營收爆發股反而賺），Q 接近 0',
     P5='整體通過（T0＋V1～V4）約 5～10%'),
}
txt = json.dumps(PRE, ensure_ascii=False, indent=1)
f.write_text(txt)
h = hashlib.sha256(txt.encode()).hexdigest()
(H / 'PREREG.sha256').write_text(f'{h}  PREREG.json\n')
(H / 'code_prereg.sha256').write_text(''.join(f'{sha(p)}  {p}\n' for p in ('s29lib.py', 'prep.py', 'prereg_write.py')))
print(h)
