#!/usr/bin/env python3
"""S32 事前登記：在本題算任何規則的報酬之前寫 PREREG.json 與 PREREG.sha256。已存在就拒絕覆寫。python3 -I -B prereg_write.py
（第 0 步 prep.py 只做引擎驗收 A0／A1（現行＝等額，本來就公布過）、特徵涵蓋率與各規則倍數分布；沒有算任何規則的報酬。）"""
import sys, json, hashlib, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
H = Path(__file__).resolve().parent
f = H / 'PREREG.json'
assert not f.exists(), 'PREREG.json 已存在，拒絕覆寫'
import s32lib                                                     # noqa: E402
dc = json.loads((H / 'data_check.json').read_text())
sha = lambda p: hashlib.sha256((H / p).read_bytes()).hexdigest()

PRE = {
 'topic': 'S32 衛星每檔金額依波動度調整（波動大買少、波動小買多）：能不能降低回撤、讓報酬不那麼靠少數大贏家，同時不犧牲太多報酬？',
 'written_at': datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
 'status_at_write': '只跑過 prep.py（data_check.json）：引擎驗收 A0、包裝引擎 A1（等額時逐位相同）、F1 名單上 σ60／σ250／β60 涵蓋率與分布、各規則倍數 m 的分布；沒有算任何規則的報酬。',
 'code_sha256_at_write': {p: sha(p) for p in ('s32lib.py', 'prep.py', 'prereg_write.py')},
 'not_done_before': '已讀 s6_satellite_sizing、s8_drawdown_cap 的 FINDINGS：做過的是 W10／W7／WN（等額 ÷10、÷7、÷實際檔數）、名次加權（L_MA 1～5 名 1.5 倍）、'
                    '單檔上限 8／10／12% 總資產、衛星比例、底倉鎖定；沒有任何依波動度或 β 的權重。w12_betting_against_beta、w3_dividend_lowvol 是選股題，不是 SURGE 衛星內配重。故本題四條規則都是沒做過的。',
 'baseline（不改定義）': 'S10 的 E0／W10_70（XQ 20＋21 號預設），cslib.EXRIGHT_FIX＝True：嚴成本、只上市、排除創新板、A10 訊號日次日開盤、F1 進場（L_MA 依 SURGE 前 10 名中 RET20 > 10%）、'
                      'E0 出場（跌出 L_MA 前 20 名）；衛星 70%、每檔＝衛星 ÷ 10；股數 × 開盤 < 目標一半不買；單筆 ≤ 當日成交值 5%；底倉 ±10 個百分點帶狀拉回；0050 停牌日用前一日收盤（s17lib）。',
 'engine': '共用引擎不改。s32lib.simulate_w＝執行時讀 s10lib.simulate_s10 原始碼做字串替換（PATCH 逐字記錄於 data_check.json）：目標金額 u × m(k, j)，沒給 m 時 ×1.0；另只讀地記錄每日最大單檔權重與集中度。'
           'A1：等額時與 s10lib.simulate_s10 日淨值、交易逐位相同（位移 0／100 萬、位移 2／150 萬、位移 3／60 萬，已通過）。',
 'what_changes': '只改「新買進」的每檔目標金額：target＝(衛星資金 ÷ 10) × m。m 只在當月 F1 名單（含已持有、當天不買的）內計算，名單內 Σm＝檔數（預算中性：名單滿 10 檔時目標總額仍＝衛星資金）；'
                 '續抱股不重新平衡；衛星比例、底倉、拉回、出場、名單、買進順序（SURGE）、最小買進門檻（股數 × 開盤 < 該檔自己目標一半不買）全部不動。',
 'bounds': f'm ∈ [{s32lib.LO}, {s32lib.HI}] ＝ 單檔目標 ≥ 衛星 5%、≤ 衛星 15%（等額＝10%）。夾住後差額依原始分數比例分給未夾住者，反覆到穩定。名單只有 1 檔 → m＝1。',
 'features': dict(sig60='訊號日 k（含當天收盤；與 RET20 同時點，無前視）往前 60 個交易日 P.R（含息還原日報酬）樣本標準差，有值 ≥ 40 天',
                  beta60='同窗對 0050（P.E50 還原）日報酬的 OLS 斜率，成對有值 ≥ 40 天',
                  missing='缺值用名單內有值者原始分數中位數；全缺 → 等額（data_check：訓練／驗證期 F1 缺值 0%）'),
 'rules（4 條候選，全部計入 N；參與訓練期選擇）': [dict(r) for r in s32lib.RULES],
 'control（1 條，只描述，不參與選擇與判定）': dict(s32lib.CONTROL),
 'weight_preview（data_check，只看權重）': {r['id']: dc[r['id']] for r in s32lib.RULES + [s32lib.CONTROL]},
 'windows': dict(train='訊號日 2016-02～2021-12（營收月份 2016-01～2021-11）；訓練窗淨值 2016-01-29 → 2021-12-30（全期連續模擬切出來，同 S21）',
                 valid='訊號日 2022-01～2026-10（營收月份 2021-12～2026-09）；重新起跑：只保留營收月份 ≥ 2021-12 的訊號，驗證窗 2021-12-30 → 2026-10-22（同 S21）'),
 'paths': '20 條路徑＝訊號日位移 0～3 天 × 起始資金 60／80／100／150／200 萬（同 S21）',
 'definitions': dict(improved_path='同一路徑上 規則 − 現行：Calmar（窗內年化 ÷ |窗內最大回撤|）差 ≥ +0.05，或 窗內最大回撤淺 ≥ 0.5 個百分點（兩者之一）',
                     ann_diff='同一路徑窗內年化差（個百分點）'),
 'selection': '訓練期：每條候選算 改善路徑數 n_imp（/20）、年化差中位數、Calmar 差中位數。先看「年化差中位數 ≥ −1」者，其中 n_imp 最多者凍結（同分依 Calmar 差中位數、再依年化差中位數）；'
              '若沒有任何一條年化差中位數 ≥ −1，仍依同順序凍結一條。'
              'T0：凍結者 n_imp ≥ 14／20 且年化差中位數 ≥ −1 個百分點。T0 不過仍凍結，驗證期只描述、判定必為不通過。',
 'verdict（驗證期只跑一次）': dict(
     V1='驗證窗（重新起跑）改善路徑（同 improved_path 定義）≥ 15／20',
     V2='驗證窗年化差中位數 ≥ −1.5 個百分點',
     V3='禁買 CUR 與凍結規則各自（重新起跑主跑）最賺 5 筆期間的聯集後，驗證窗年化差中位數 ≥ −1.5 個百分點',
     V4='兩段空頭（2022；2024-07～2025-04）段內最大回撤配對差中位數都 > 0（凍結規則較淺）',
     overall='T0 且 V1～V4 全過＝通過；否則不通過，停在描述，不換規則重跑、不寫 XQ 修改建議'),
 'also_report': '最賺 5 筆占淨值增加比（全期連續與重新起跑）、CAPM β（日報酬對 0050）、單檔最大權重（每日最大單檔市值 ÷ 淨值的最大值與平均）、有效持股數（1 ÷ Σ個股權重²）、八項檢查、逐年、四段空頭。',
 'acceptance': dict(A0='CUR 全期主跑（EXRIGHT_FIX＝True）重現 48.99%／34.54%／−32.11%（±0.025 個百分點）、已平倉 381 筆（prep 已通過）',
                    A1='simulate_w 權重＝等額時與 s10lib.simulate_s10 逐位相同（prep 已通過；train／validate 再驗位移 0）'),
 'N': '資料窺探族群 N：BACKLOG 最後記錄 2,869（S31）＋4 條候選＋1 條對照＝2,874（S29／S30 平行進行，若有新增以研究中心為準）',
 'predictions（事前預測）': dict(
     P1='四條都會讓訓練期回撤略淺（0～2 個百分點），但年化也會少（−0.5～−3）；β 約 0.95～1.0 幾乎不變',
     P2='IB60 與 IV60 最可能過 T0；IVAR60 傾斜最大、最可能年化差 < −1；HV3 效果最小',
     P3='最賺 5 筆占比會下降（大贏家多半是高波動股，被減碼），但這正是報酬來源，所以年化差多半為負',
     P4='整體通過（T0＋V1～V4）機率約 15%；最可能卡在 V1（2024-07～2025-04 的回撤是全市場同跌，個股權重救不了）或 V3'),
}
txt = json.dumps(PRE, ensure_ascii=False, indent=1)
f.write_text(txt)
h = hashlib.sha256(txt.encode()).hexdigest()
(H / 'PREREG.sha256').write_text(f'{h}  PREREG.json\n')
(H / 'code_prereg.sha256').write_text(''.join(f'{v}  {k}\n' for k, v in PRE['code_sha256_at_write'].items()))
print(h)
