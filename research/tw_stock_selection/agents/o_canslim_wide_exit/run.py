#!/usr/bin/env python3
"""o_canslim_wide_exit：CAN SLIM 進場條件（腳本原文預設參數，不改）＋較寬的出場。
用法（一律 python3 -I -B）：
  python3 -I -B run.py check     # 引擎等價檢查：只跑腳本原樣（6% 停損，已看過的結果），確認與 cslib.simulate 逐日相同
  python3 -I -B run.py prereg    # 事前登記 → PREREG.json、PREREG.sha256、results.json['PREREG']（此時不得有任何新報酬）
  python3 -I -B run.py main      # 全部事前登記的回測（先核對 PREREG sha256）
  python3 -I -B run.py posthoc   # 看過報酬後才加的：逐筆出場驗證、與複製員數字對照、事後分析（寫進 results.json['POSTHOC']、['EXIT_VERIFY']）
中間檔：scratchpad/cs_wide/（panel_cs.npz、compact.npz、e0050_ohlc.csv；可刪，會自動重建）。"""
import json, sys, time, datetime, math, os
from pathlib import Path
from statistics import NormalDist
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import cwlib as cw
cs = cw.cs

DRY = os.environ.get('CW_DRY') == '1'      # 乾跑：全部候選換成腳本原樣（6% 停損）、路徑各 2 條、輸出到暫存夾；只用來在登記前確認程式不當機
OUT = (cw.TMP / 'dry') if DRY else HERE
RES = OUT / 'results.json'
START = '2017-03-31'
A2019 = '2019-01-01'
SEGS = [('2017-03-31', '2018-12-31'), ('2019-01-01', '2023-12-31'), ('2024-01-01', '2026-10-22')]
N_BEFORE = 2619
NPATH = 2 if DRY else 200
COLS_ALL = ['MAIN', 'STRESS', 'ZERO', 'E50', 'ALL', 'NEXTDAY', 'NOLOT']                     # 每個候選都跑的 7 欄
COLS_PRIMARY = ['ALL_E50', 'E50_STRESS', 'S20', 'C30', 'C300', 'C1000', 'TPEX', 'EXT', 'EXT_NOLOT']   # 只有主規格跑
COLS_BASE = ['POOL_RS', 'POOL_RS_ZERO']                                                     # 歸因基準（不看 C、A 的池子）

PREREG = {
 'title': 'o_canslim_wide_exit 事前登記（寫於任何「寬出場」報酬由本研究員計算之前）',
 'question': '公開貼文的 CAN SLIM 台股版進場腳本（raw/canslim_entry.xs，預設參數不改），把出場放寬，扣成本後能不能顯著打敗 0050 含息',
 'already_seen': {
  'statement': '2017-04～2026-10 的全樣本已經被看過，歷史上沒有乾淨的樣本外；真正的樣本外只有前瞻（G8）。本題全部歷史數字都是樣本內。',
  'list': [
   '腳本原樣（收盤從 20 日最高收盤回落 6% 或 8 日線下穿 20 日線）：只上市、PIT、100 萬、10 檔、一般成本、空位現金，2019 起年化 12.33% 對 0050 28.91%（t −1.36）；全期 11.79% 對 23.93%；單邊周轉 16.9 倍／年；成本拖累 10.76pp；平均持有 8 日；平均 5.05 檔、現金 50.7%',
   '複製員事後把移動停損放寬（保留均線下穿、只上市、空位現金、一般成本）：10% → 2019 起超額 −6.52pp；11% → −0.07；12% → +3.03（t 0.37）；15% → +3.36（t 0.41）；20% → +4.48（t 0.51）；12% 全期 +6.79pp（t 0.74）；12% 壓力成本 2019 起 25.70%（輸 0050）',
   '上市＋上櫃、空位放 0050、停損 10%：2019 起 −0.52pp、全期 +3.84pp（t 0.61）、壓力 2019 起 −8.31pp、分段 +21.2／+6.5／−15.8pp；同口徑停損 20% 全期 t 1.65',
   '每持有日毛報酬在各停損幅度都是 21～24bp（貼文口徑）',
   '策略對 0050 的 β 0.37、β 調整後月 α +0.40%（t +0.69）（腳本原樣）；和主策略月報酬相關 0.59、持股日 48% 落在主策略前 30 名',
   '引擎慣例：當日賣出騰出的空位隔日才可買 → 腳本原樣全期 +2.56pp（14.35%）、2019 起 15.48%',
   '整股限制：2017-03-31～2020-10-23 有限制 9.86%、無限制 15.51%；無限制全期 13.84%',
   '延伸段 2015-12-25～2017-03-31（腳本原樣）：可執行 8.25% 對 0050 18.53%；貼文口徑 132 筆、勝率 39.4%、每筆 +2.0%',
   '消融、參數鄰域（EPS、營收、距高點、大盤均線、停損 8%／10%）全部為負（見 o_ext_canslim_xs/FINDINGS.md 5.1、5.5）',
   'replicate/review_numbers.json 另存有「只上市／上市＋上櫃 × 空位現金／0050 × 停損 6、8、10、12、15、20%」共 24 格的數字；報告文字只引用上列幾格。這 24 格一律視為已看過。本研究員在登記前沒有開啟該檔的數值',
   '本研究員在登記前自己算過的報酬：只有腳本原樣（6% 停損）的引擎等價檢查（run.py check），數字全部是上列已知結果的重現'],
 },
 'entry': '完全照 raw/canslim_entry.xs 預設參數（C：單季 EPS 年增 ≥18；A：年度 EPS 三年複合成長 ≥15、近 3 個月營收年增平均 ≥20；N：收盤距 240 日最高價 ≥ −10% 且滿 240 根；S：20 日均量 ≥ 500 張；L：120 日漲幅 > 加權指數同期且 > 0；M：加權指數收盤 ≥ 60 日均線）。'
          '實作直接 import o_ext_canslim_xs/cslib.py 的 load_eps、load_rev3、entry_fields、entry_signal（已被獨立複製逐列確認：2017-03-31 起上市＋上櫃 19,072 列）。原始價、PIT、無成交日沿用前收（高＝收、量＝0）。',
 'bars_and_alignment': {
  'calendar': '官方日 K 的交易日曆（2015-01-05～2026-10-22）。訊號日 t 盤後算訊號 → t+1 開盤成交。',
  'exit_check': '出場條件自進場當日收盤起（含）逐日在收盤後檢查；第 k 日成立 → 第 k+1 日開盤賣（無開盤價或開盤 ≤ 跌停價則順延到下一日再試）。一旦成立就維持待賣，不再重算。',
  'float': '全部比較用 IEEE 雙精度、不加容差，運算順序照下列公式；例外只有漲跌停判斷（開盤 ≥ 漲停價 − 1e-9、開盤 ≤ 跌停價 + 1e-9）與現金足額判斷（+1e-6）。',
  'CF': 'CF＝收盤價，無成交日沿用前一個有成交日的收盤（原始價，不還原）',
  'R': '含息日報酬 R[t]＝收盤÷(收盤−漲跌價差)−1；|R|>0.5 或缺值時改用 收盤÷前收−1；無成交日 0'},
 'exit_rules': {
  'trail_p': 'CF[t] <= max(CF[t−19..t]) × (1 − p/100)（需滿 20 根；p＝12、15、20）',
  'cross': '8 日均線向下穿越 20 日均線：SMA8[t] < SMA20[t] 且 SMA8[t−1] >= SMA20[t−1]（SMA 用 CF，需滿根數）',
  'fstop8': '以進場價為基準的固定停損：g[k] <= 1 − 8/100，其中 g[k]＝(CF[e]÷O[e]) × Π_{i=e+1..k}(1+R[i])，e＝進場日、O[e]＝進場日開盤價（不含滑價與費用）。'
            '用含息累積報酬而不是原始價，避免除權息日的機械性跌幅觸發停損（trail 與 cross 仍照腳本用原始價）',
  'ma50': 'CF[t] < SMA50[t]（嚴格小於；需滿 50 根）',
  'hold60': '持有滿 60 個交易日：在第 e+59 日收盤後成立 → 第 e+60 日開盤賣（交易日以市場日曆計，不論該股當日有無成交）',
  'combine': '同一候選的多條規則是「任一成立即出場」；成交紀錄的 why 欄記下當日成立的全部規則（trail、cross、ma50、fstop、hold，以 + 連接）'},
 'candidates': {
  'T15_MA': '【主規格】移動停損 15% ＋ 均線下穿',
  'T12_MA': '移動停損 12% ＋ 均線下穿', 'T20_MA': '移動停損 20% ＋ 均線下穿',
  'T12': '只有移動停損 12%', 'T15': '只有移動停損 15%', 'T20': '只有移動停損 20%',
  'FS8_T15': '進場價固定停損 8% ＋ 移動停損 15%（不含均線下穿；O’Neil／Minervini 原法沒有均線下穿規則）',
  'FS8_T15_MA': '進場價固定停損 8% ＋ 移動停損 15% ＋ 均線下穿（主規格只多加一條固定停損，用來單獨量固定停損的效果）',
  'MA50_FS8': '收盤跌破 50 日均線 ＋ 進場價固定停損 8%。機制理由：O’Neil 原法以 50 日線為機構支撐，跌破視為趨勢結束；固定停損是原法的 7～8% 認賠規則。不含移動停損',
  'HOLD60': '固定持有 60 個交易日、沒有任何停損。機制理由：季報與營收動能的漂移期約一季；這個候選把「進場訊號有沒有用」和「出場時機」分開，周轉最低',
  'primary_rule': '主規格＝T15_MA：使用者指定的「已看過範圍（12／15／20）的中間值、保留均線下穿」。不用績效挑。其餘 9 個是鄰域／穩健性，只用來判「不翻號」與 SPA'},
 'engine': {
  'base': '與 o_ext_canslim_xs/cslib.simulate 逐行相同（已用腳本原樣驗證逐日淨值差 0）：100 萬起始、最多 10 檔；每檔目標＝成交日開盤估算總權益÷檔數、不超過現金；'
          '買＝開盤×(1+滑價)、賣＝開盤×(1−滑價)，滑價＝0.5×跳動點÷開盤價；手續費 0.1425%×0.28 每邊、整股與零股各一筆、每筆最低 20 元；賣方證交稅 0.3%；'
          '開盤 ≥ 漲停價不買（該次訊號作廢）；2020-10-26 前只買整張（買不起換下一名）；單筆 ≤ 當日成交值 5%（買縮量、賣分日）；持股以含息日報酬滾動；下市以最後收盤結清',
  'same_day_order': '同一日開盤先賣後買。賣出依部位建立先後。買進候選＝訊號日（前一日）盤後入選、且前一日收盤未持有（今日開盤剛賣掉的也不算未持有）',
  'ranking': '訊號多於空位時依訊號日「超越大盤%」（個股 120 日漲幅−加權指數 120 日漲幅）由大到小；同分依代號由小到大',
  'slot_convention': '主欄＝「當日開盤賣出騰出的空位當日可買」（SAME）。事前指定理由：(1) 已看過的全部結果都用這個慣例；(2) 開盤前同時掛賣單與買單、當日買賣款項淨額交割，實務上做得到；'
                     '(3) 在唯一看過的比較（腳本原樣）它是較低的一邊，不是挑有利的。並列欄 NEXTDAY＝當日可買檔數不得超過前一日收盤時的空位數（現金仍可用當日賣出所得）',
  'costs': 'normal 如上；stress＝每邊滑價再加 0.3%；zero＝無費用無滑價（回測上緣，實現不了）',
  'idle_e50x': '空位放 0050（E50 欄）：閒置資金全部持有 0050。每日：(1) 閒置資金 ×(1+ov[t])，ov＝0050 前收→當日開盤的含息報酬＝(1+含息日報酬)÷(收盤÷開盤)−1；'
               '(2) 開盤先賣股、後買股，買股可用資金＝目前閒置資金 − 保留款，保留款＝I0 − max(I0−20, 0)×(1−c_s)，I0＝隔夜後、買賣前的閒置資金；'
               '(3) 淨額 d＝買賣後閒置資金−I0：d>0 淨買 0050，成本＝d×(滑價+壓力加碼)+max(20, d×0.0399%)；d<0 淨賣 0050 毛額 g＝|d|÷(1−c_s)，成本＝g×(0.1% 證交稅+滑價+壓力加碼)+max(20, g×0.0399%)，'
               'c_s＝0.0399%+0.1%+滑價+壓力加碼；0050 滑價＝0.5×跳動點（<50 元 0.01、≥50 元 0.05）÷當日開盤；(4) 閒置資金 ×(1+intr[t])，intr＝0050 收盤÷開盤−1。'
               '0050 開盤與收盤取自官方日 K；缺開盤價的日子 intr＝0、ov＝整日報酬、滑價 0.03%。0050 部位不套整股限制（近似，已知偏樂觀：2020-10-26 前 0050 一張約 8 萬元）',
  'universes': 'TWSE＝訊號日當天市場別為上市（主欄）；ALL＝上市＋上櫃；TPEX＝只上櫃',
  'start': '第一個訊號日 2017-03-31、起始淨值 100 萬（與前一題相同）；延伸段另從 2015-12-25 起跑'},
 'columns': {
  'MAIN': '只上市、PIT、100 萬、最多 10 檔、一般成本、空位放現金、SAME 慣例。第一個報的數字＝2019-01-01～2026-10-22 年化對 0050 含息',
  'per_candidate(7)': 'MAIN、STRESS（壓力成本）、ZERO（零成本，回測上緣）、E50（空位放 0050 精確換手）、ALL（上市＋上櫃）、NEXTDAY、NOLOT（無整股限制）',
  'primary_only(9)': 'ALL_E50、E50_STRESS、S20（20 檔）、C30／C300／C1000（30 萬／300 萬／1,000 萬）、TPEX（只上櫃，偽樣本外 b）、EXT（2015-12-25 起，偽樣本外 a）、EXT_NOLOT',
  'baseline(2)': 'POOL_RS、POOL_RS_ZERO：訊號改成「通過 N、S、L、M（不看 C、A）」的全部股票，同排序、同主規格出場、同主欄口徑'},
 'stats': {
  'annualized': '日曆年化＝(終值÷基期值)^(365.25÷日曆天數)−1；視窗基期＝視窗起日之前最後一個交易日的淨值（全期基期＝2017-03-31）',
  'monthly_t': '日曆月報酬（月底對月底；第一個月自基期起）策略−0050 的平均÷(標準差(ddof=1)÷√n)。全期 n＝115（2017-04～2026-10）、2019 起 n＝94。另報 Newey-West（Bartlett 核、6 期落後）t',
  'segments': '2017-03-31～2018-12-31、2019～2023、2024-01-01～2026-10-22',
  'drop_best3': '拿掉年度超額（月報酬連乘的年報酬差）最高的 3 個日曆年後重算月超額 t（全期）',
  'turnover': '單邊周轉＝買進金額（股數×開盤價）合計÷日均淨值÷年數；成本拖累＝ZERO 年化−MAIN 年化（pp）；另報費用＋稅＋滑價÷日均淨值÷年數',
  'expectancy': '每筆期望值＝勝率×平均獲利−(1−勝率)×平均虧損，用主欄成交紀錄（淨額÷買進總成本−1；含期末未平倉）；零成本版並列'},
 'path_perturbation': {
  'why': '起始日偏移在本題沒有資訊量（前一題 13 條範圍只有 0.3pp），改用排序擾動',
  'P1(用於 G1)': '主規格主欄。每條路徑 i＝0..199：排序鍵＝Z＋ε，Z＝訊號日當天全部上市訊號股的「超越大盤%」標準化（母體標準差；只有 1 檔或標準差 0 → 0），'
                'ε＝np.random.default_rng(1000+i).standard_normal((T,N)) 的對應格。報 2019 起超額的中位數、p10、p90（全期並列）',
  'P2(並列)': '排序鍵＝np.random.default_rng(5000+i).random((T,N))（完全隨機排序），200 條，同樣報中位數、p10、p90'},
 'pseudo_oos': {
  'statement': '只能當輔助證據，不能當樣本外，不進入 G1～G7 判定',
  'a_EXT': '2015-12-25～2017-03-31 延伸段（只上市）：FY2011、FY2012 年度 EPS 用「(y+1) 年 4/10～4/30 每日 收盤÷本益比 的中位數」反推（o_pead per_long.parquet 與 data_panel/panel/close_raw.parquet），'
           '只補 EpsY3 缺的格子；其餘同主規格主欄，自 2015-12-25（官方日 K 滿 240 根的第一天）以 100 萬起跑。報該段年化對 0050、月超額 t（n＝16），EXT_NOLOT 並列。'
           '讀法事前寫死：該段主欄超額 > 0 記「方向一致」，否則「方向不一致」。已知限制：反推誤差 2% 內只有 82～85%；100 萬帳戶在這段受整股限制很重',
  'b_TPEX': '只上櫃的組合（主規格、其餘同主欄）。寬出場在上市＋上櫃合併口徑被看過（至少 10% 與 20% 兩格），只上櫃單獨沒有被報過。讀法：2019 起與全期超額 > 0 記「方向一致」。'
            '已知限制：上櫃公告日全靠推斷、沒有處置資料、與上市段同一時期同一多頭'},
 'placebo_and_attribution': {
  'placebo': '主規格主欄。每條 i＝0..199：每個交易日 t 從當日池子（上市、通過 N、S、L、M 與在榜，不看 C、A）抽 k_t 檔當訊號，k_t＝當日真實上市訊號檔數。'
             '抽法：U＝np.random.default_rng(9000+i).random((月數, N))（月份自面板第一個月 2015-01 起依序；N＝精簡面板欄數，欄序＝代號由小到大），取當日池內 U[當月] 最小的 k_t 檔。'
             '排序、出場、成本、引擎全部同主規格。報安慰劑 2019 起年化與全期年化的分布，主規格落在第幾百分位（＝安慰劑小於主規格的比例）。'
             '讀法事前寫死：≥ 95 百分位才寫「C、A 條件有增量」；< 95 寫「無法區分於只買近新高強勢股」',
  'pool_baseline': 'POOL_RS：池子全部當訊號（不抽樣）跑主規格出場；另報貼文口徑（不限檔數、零成本）每持有日毛報酬：真實訊號對池子',
  'beta': '月報酬對 0050 月報酬 OLS（含截距、無風險利率視為 0）：β、月 α、α 的 t（OLS 與 NW6）；並和「β×0050＋(1−β)×現金」（每月再平衡、零成本）的年化比較。全期與 2019 起',
  'vs_main': '主策略序列＝integrate_o20/data/v7x_navs_jmean_100w.csv.gz 的 base|REAL_opt|j|normal|100萬|std（j＝0..12），各 j 在 2017-03-31 歸一後等權平均（只用來算相關、配對差、混合；不是正式數字）。'
             '報：月報酬配對差（本題−主策略）的平均、t、NW6 t、6 個月循環區塊拔靴 95% CI（5,000 次、seed 20261114）；月報酬相關、對 0050 月超額相關；50/50 每月再平衡混合（零再平衡成本）的年化、對 0050 的 t、月頻最大回撤。主規格 MAIN 與 E50 兩欄都做',
  'cost': '成本拖累、周轉、平均持有交易日（成交紀錄 exit_t−ent_t 的平均）、每筆期望值代入數字'},
 'power': {
  'formula': 't＝月超額平均÷(sd÷√n)。達到 t ≥ 3（n＝115）需要月超額平均 ≥ 3×sd÷√115；達到 t ≥ 2（n＝94）需要 ≥ 2×sd÷√94。80% 檢定力的最小可偵測月超額＝(門檻+0.84)×sd÷√n',
  'planning_numbers': '用已知的腳本原樣月超額標準差（全期 7.35%、2019 起 7.92%）先算：全期 t ≥ 3 需要月超額 ≥ 2.06%（約年化 24.7pp）；2019 起 t ≥ 2 需要 ≥ 1.63%（約 19.6pp）。'
                      '如果真實超額只有年化 +3～5pp（月 0.25～0.42%），全期 t 的期望值只有 0.36～0.61，通過 t ≥ 3 的機率約 0.4%～0.8%',
  'statement': '所以在這個樣本長度，G2 幾乎不可能通過，除非真實超額在年化 25pp 左右或以上。G2 不過只能讀成「本樣本證明不了」，不能讀成「證明沒有超額」；同樣，不顯著的正超額也不能讀成「有超額」。跑完後用各候選實際的標準差重算一次並列'},
 'gates': {
  'G1': '主規格：MAIN 與 STRESS 的 2019 起超額與全期超額都 > 0，且路徑擾動 P1 的 2019 起超額 p10 > 0',
  'G2': '主規格 MAIN：全期月超額 t ≥ 3.0 且 2019 起 t ≥ 2.0（NW6 並列，不用來放寬）',
  'G3': '本題 10 個候選（MAIN 欄、全期月超額）Hansen SPA consistent p < 0.05，且主規格全期 t ≥ 新 N 下的 Bonferroni 雙尾門檻；DSR（N＝新 N、試驗間變異數取 10 個候選月 SR 的變異數）≥ 0.95 並列。全族群報酬矩陣無法重建，不跑全族群 SPA',
  'G4': '主規格 MAIN 三個分段超額都 > 0；拿掉最好 3 年後 t ≥ 2.0；其餘 9 個候選 MAIN 欄的 2019 起超額全部 > 0（不翻號）',
  'G5': '全部 PIT、含下市股、無前視；沿用前一題閘門並補 0050 開盤價與延伸段反推的檢查。已知附註照列',
  'G6': '待不同研究員只看規格獨立重做（本題無法自證）',
  'G7': '主規格 C（100 萬）、C300、C1000 的 2019 起超額都 > 0；C30 並列',
  'G8': '前瞻 12 個月（尚未開始）',
  'verdict': '機械套用：G1～G5、G7 全過 → 「待 G6 複製一致後建議進前瞻追蹤」；任一不過 → 「不可用」（不建議進前瞻追蹤）。不論結果，真錢配置都要等 G8'},
 'family': {
  'N_before': N_BEFORE,
  'counting': '保守計法：本題事前登記的每一條「對 0050 比較的報酬序列」都計入，共 10×7＋9＋2＝81 條（路徑擾動與安慰劑不是候選，不計）。新 N＝2,619＋81＝2,700',
  'N_after': N_BEFORE + 81},
 'implementation_note': '為了省記憶體，程式分塊算訊號後只保留「曾經通過 N、S、L、M 的股票」1,723 欄（欄序仍是代號由小到大）；池子包含全部真實訊號，引擎結果與整個面板相同（check 階段以腳本原樣驗證逐日淨值差 0）。'
                        '路徑擾動與安慰劑的亂數矩陣形狀是 (2,876 日, 1,723 欄)；複製員不必逐條相同，只要分布（中位數、p10、p90、百分位）一致',
 'post_hoc_rule': '看過報酬後新增的任何分析一律放在 results.json 的 POSTHOC、FINDINGS 另立「事後」一節，與事前登記的主欄分開；事後讀法不得寫成建議',
}


def prereg_sha():
    return cw.sha_bytes(json.dumps(PREREG, ensure_ascii=False, sort_keys=True).encode('utf-8'))


def load_res():
    return json.load(open(RES)) if RES.exists() else {}


def save_res(r):
    json.dump(r, open(RES, 'w'), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))


def setup():
    P = cw.Compact()
    t0 = int(P.IDX.get_loc(pd.Timestamp(START)))
    bench = pd.Series(P.E50, index=P.IDX)
    U = {'TWSE': P.MK == 1, 'ALL': P.MK > 0, 'TPEX': P.MK == 2}
    disp = cs.disposal_mask(P)
    e50 = cw.load_e50(P)
    return P, t0, bench, U, disp, e50


# =============================================================== check（只用已看過的腳本原樣）
def stage_check():
    P, t0, bench, U, disp, e50 = setup()
    out = dict(compact_info=P.info, panel_days=P.T, compact_cols=P.N)
    for nm, un, kw in [('MAIN', 'TWSE', {}), ('ALL', 'ALL', {}), ('S20', 'TWSE', dict(slots=20)), ('C30', 'TWSE', dict(capital=3e5)),
                       ('STRESS', 'TWSE', dict(cost='stress')), ('ZERO', 'TWSE', dict(cost='zero'))]:
        a, la, _ = cs.simulate(P, P.sig, P.RS, P.X['trail6'], P.X['cross'], U[un], t0, disp=disp, **kw)
        b, lb, sb = cw.simulate2(P, P.sig, P.RS, P.X, cw.ORIG, U[un], t0, disp=disp, **kw)
        out['orig_' + nm] = dict(max_abs_nav_diff_vs_cslib=float(np.abs(a - b).max()), nav_end=float(b.iloc[-1]), n_trades=int(len(lb)), n_trades_cslib=int(len(la)))
    prior = json.load(open(cw.AG / 'o_ext_canslim_xs' / 'results.json'))
    out['prior_nav_end'] = prior['ENGINE_CHECK']['nav_end']
    out['signals_all_from_2017-03-31'] = int((P.sig[t0:] & U['ALL'][t0:]).sum())
    for nm, kw in [('NEXTDAY', dict(same_day_slot=False)), ('NOLOT', dict(lot_rule=False)), ('E50', dict(idle='e50x', e50=e50))]:
        b, _, _ = cw.simulate2(P, P.sig, P.RS, P.X, cw.ORIG, U['TWSE'], t0, disp=disp, **kw)
        out['orig_' + nm] = dict(full=cw.wstats(b, bench)['cagr'], from2019=cw.wstats(b, bench, A2019)['cagr'])
    b, _, _ = cw.simulate2(P, P.sig_ext, P.RS, P.X, cw.ORIG, U['TWSE'], P.ts_ext, disp=disp)
    w = cw.wstats(b, bench, None, START)
    out['orig_EXT_window'] = dict(start=str(P.IDX[P.ts_ext].date()), cagr=w['cagr'], cagr_0050=w['cagr_0050'], n_months=w['n_months'],
                                  pre_signal_rows_twse=int((P.sig_ext[P.ts_ext:t0] & U['TWSE'][P.ts_ext:t0]).sum()),
                                  post_signals_identical=bool(((P.sig_ext & U['TWSE'])[t0:] == (P.sig & U['TWSE'])[t0:]).all()))
    m = cw.wstats(cw.simulate2(P, P.sig, P.RS, P.X, cw.ORIG, U['TWSE'], t0, disp=disp)[0], bench)
    out['orig_MAIN_monthly'] = dict(n=m['n_months'], sd_ex_m=m['sd_ex_m'], t=m['t_ex_m'], nw6=m['nw6_t'])
    out['e50'] = dict(days_missing_open=e50['n_missing'], mean_overnight_bp=float(e50['ov'][t0 + 1:].mean() * 1e4), mean_intraday_bp=float(e50['intr'][t0 + 1:].mean() * 1e4),
                      product_check=float(np.prod(1 + e50['ov'][t0 + 1:]) * np.prod(1 + e50['intr'][t0 + 1:]) / (P.E50[-1] / P.E50[t0])))
    res = load_res(); res['ENGINE_CHECK_PRE'] = cw.rnd(out)
    res['ENGINE_CHECK_PRE']['note'] = '全部是腳本原樣（6% 停損）＝已看過的結果；用來確認本題引擎與 cslib.simulate 逐日相同（差 0）、並重現複製員的 NEXTDAY／NOLOT／延伸段數字'
    save_res(res); print(json.dumps(res['ENGINE_CHECK_PRE'], ensure_ascii=False, indent=1))


def stage_prereg():
    res = load_res()
    assert 'ENGINE_CHECK_PRE' in res, '先跑 check'
    assert 'EXEC' not in res, '已有回測結果，不得重寫 PREREG'
    sha = prereg_sha()
    txt = json.dumps(PREREG, ensure_ascii=False, sort_keys=True, indent=1)
    (HERE / 'PREREG.json').write_text(txt, encoding='utf-8')
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
    (HERE / 'PREREG.sha256').write_text('%s  PREREG(canonical: json.dumps(PREREG, ensure_ascii=False, sort_keys=True) utf-8)\n%s  PREREG.json(file)\n%s  cwlib.py(at prereg)\n%s  run.py(at prereg)\nwritten_utc %s\n'
                                        % (sha, cw.sha_file(HERE / 'PREREG.json'), cw.sha_file(HERE / 'cwlib.py'), cw.sha_file(HERE / 'run.py'), now), encoding='utf-8')
    res['PREREG'] = dict(spec=PREREG, sha256=sha, written_utc=now, cwlib_sha256=cw.sha_file(HERE / 'cwlib.py'), run_sha256=cw.sha_file(HERE / 'run.py'),
                         note='此時本研究員只算過腳本原樣（6% 停損）的引擎等價檢查；沒有算過任何寬出場的報酬')
    save_res(res); print(sha, now)


# =============================================================== main
def trade_block(lg):
    r = lg['ret'].to_numpy(); w = r > 0
    pw = float(w.mean()); aw = float(r[w].mean()) if w.any() else float('nan'); al = float(-r[~w].mean()) if (~w).any() else float('nan')
    hd = (lg['exit_t'] - lg['ent_t']).to_numpy()
    return dict(n=int(len(r)), win_rate=pw, avg_win=aw, avg_loss=al, payoff=aw / al if al else float('nan'), expectancy=pw * aw - (1 - pw) * al, mean=float(r.mean()),
                median=float(np.median(r)), max=float(r.max()), min=float(r.min()), hold_days_mean=float(hd.mean()), hold_days_median=float(np.median(hd)),
                n_open_end=int((lg['why'] == 'open_end').sum()),
                by_why={k: dict(n=int(len(g)), mean=float(g['ret'].mean()), win_rate=float((g['ret'] > 0).mean()), hold=float((g['exit_t'] - g['ent_t']).mean())) for k, g in lg.groupby('why')},
                disp_buys=dict(n=int(lg['disp'].sum()), mean=float(lg.loc[lg['disp'] == 1, 'ret'].mean()) if lg['disp'].sum() else None))


def stage_main():
    T0 = time.time()
    if DRY:
        OUT.mkdir(parents=True, exist_ok=True); res = {'PREREG': dict(sha256=prereg_sha())}
        for k in list(cw.CANDS):
            cw.CANDS[k] = cw.ORIG
    else:
        res = load_res()
    assert 'PREREG' in res and res['PREREG']['sha256'] == prereg_sha(), 'PREREG 不存在或內容與登記不符'
    P, t0, bench, U, disp, e50 = setup()
    EXEC = {}; navs = {}; logs = {}

    def pack(nav, lg, st, a0=None):
        o = dict(full=cw.wstats(nav, bench), from2019=cw.wstats(nav, bench, A2019),
                 segments={a + '~' + b: cw.wstats(nav, bench, a, b) for a, b in SEGS}, drop_best3=cw.drop_best3(nav, bench), exec=st)
        if lg is not None and len(lg):
            o['trades'] = trade_block(lg)
        return o

    def kw_of(col):
        return {'MAIN': dict(), 'STRESS': dict(cost='stress'), 'ZERO': dict(cost='zero'), 'E50': dict(idle='e50x', e50=e50), 'ALL': dict(un='ALL'),
                'NEXTDAY': dict(same_day_slot=False), 'NOLOT': dict(lot_rule=False), 'ALL_E50': dict(un='ALL', idle='e50x', e50=e50),
                'E50_STRESS': dict(idle='e50x', e50=e50, cost='stress'), 'S20': dict(slots=20), 'C30': dict(capital=3e5), 'C300': dict(capital=3e6),
                'C1000': dict(capital=1e7), 'TPEX': dict(un='TPEX')}[col]

    def run(name, sp, col, sig=None, ts=None, **over):
        kw = dict(kw_of(col)) if col in ('MAIN', 'STRESS', 'ZERO', 'E50', 'ALL', 'NEXTDAY', 'NOLOT', 'ALL_E50', 'E50_STRESS', 'S20', 'C30', 'C300', 'C1000', 'TPEX') else {}
        kw.update(over); un = kw.pop('un', 'TWSE')
        nav, lg, st = cw.simulate2(P, P.sig if sig is None else sig, P.RS, P.X, sp, U[un], t0 if ts is None else ts, disp=disp, **kw)
        EXEC[name] = pack(nav, lg, st); navs[name] = nav; logs[name] = lg
        return nav, lg, st

    for c, sp in cw.CANDS.items():
        for col in COLS_ALL:
            run('%s|%s' % (c, col), sp, col)
    pr = cw.PRIMARY; spp = cw.CANDS[pr]
    for col in ('ALL_E50', 'E50_STRESS', 'S20', 'C30', 'C300', 'C1000', 'TPEX'):
        run('%s|%s' % (pr, col), spp, col)
    # 偽樣本外 a：延伸段
    for nm, over in (('EXT', {}), ('EXT_NOLOT', dict(lot_rule=False))):
        nav, lg, st = cw.simulate2(P, P.sig_ext, P.RS, P.X, spp, U['TWSE'], P.ts_ext, disp=disp, **over)
        o = dict(window=cw.wstats(nav, bench, None, START), ext_full=cw.wstats(nav, bench), ext_from2019=cw.wstats(nav, bench, A2019), exec=st, start=str(P.IDX[P.ts_ext].date()))
        pre = lg[lg['ent_t'] <= t0]
        o['trades_entered_in_window'] = trade_block(pre) if len(pre) else None
        EXEC['%s|%s' % (pr, nm)] = o; navs['%s|%s' % (pr, nm)] = nav; logs['%s|%s' % (pr, nm)] = lg
    # 歸因基準：池子全部當訊號
    run('POOL_RS', spp, 'MAIN', sig=P.pool); run('POOL_RS_ZERO', spp, 'ZERO', sig=P.pool)
    print('runs done', len(EXEC), round(time.time() - T0, 1), flush=True)

    M = '%s|MAIN' % pr
    # ---------------- 路徑擾動
    sig_u = P.sig & U['TWSE']; Z = cw.zrank(sig_u, P.RS)
    PATH = {}
    for tag, mk in (('P1_z_plus_noise', lambda i: Z + np.random.default_rng(1000 + i).standard_normal(Z.shape)),
                    ('P2_pure_random', lambda i: np.random.default_rng(5000 + i).random(Z.shape))):
        ex19 = []; exf = []; c19 = []
        for i in range(NPATH):
            nav, _, _ = cw.simulate2(P, P.sig, mk(i), P.X, spp, U['TWSE'], t0, disp=disp, keep_log=False)
            a = cw.wstats(nav, bench, A2019); b = cw.wstats(nav, bench)
            ex19.append(a['excess_pp']); exf.append(b['excess_pp']); c19.append(a['cagr'])
        q = lambda v: dict(median=float(np.median(v)), p10=float(np.quantile(v, 0.10)), p90=float(np.quantile(v, 0.90)), min=float(np.min(v)), max=float(np.max(v)),
                           share_gt0=float((np.array(v) > 0).mean()))
        PATH[tag] = dict(n=NPATH, from2019_excess_pp=q(ex19), full_excess_pp=q(exf), from2019_cagr=q(c19),
                         main_from2019_excess_pp=EXEC[M]['from2019']['excess_pp'], main_pctile_in_paths=float((np.array(ex19) < EXEC[M]['from2019']['excess_pp']).mean()))
    print('paths done', round(time.time() - T0, 1), flush=True)

    # ---------------- 安慰劑
    pool_u = P.pool & U['TWSE']
    assert not (sig_u & ~pool_u).any()
    pl19 = []; plf = []; plb = []; plz19 = []
    for i in range(NPATH):
        ps = cw.placebo_signal(P, sig_u, pool_u, 9000 + i)
        nav, _, st = cw.simulate2(P, ps, P.RS, P.X, spp, U['TWSE'], t0, disp=disp, keep_log=False)
        pl19.append(cw.wstats(nav, bench, A2019)['cagr']); plf.append(cw.wstats(nav, bench)['cagr']); plb.append(st['n_buy'])
    pl19 = np.array(pl19); plf = np.array(plf)
    m19 = EXEC[M]['from2019']['cagr']; mf = EXEC[M]['full']['cagr']
    PLACEBO = dict(n=NPATH, pool_stock_days_twse=int(pool_u[t0:].sum()), signal_stock_days_twse=int(sig_u[t0:].sum()),
                   from2019_cagr=dict(main=m19, placebo_mean=float(pl19.mean()), placebo_median=float(np.median(pl19)), p5=float(np.quantile(pl19, .05)), p95=float(np.quantile(pl19, .95)),
                                      main_percentile=float((pl19 < m19).mean())),
                   full_cagr=dict(main=mf, placebo_mean=float(plf.mean()), placebo_median=float(np.median(plf)), p5=float(np.quantile(plf, .05)), p95=float(np.quantile(plf, .95)),
                                  main_percentile=float((plf < mf).mean())),
                   n_buy=dict(main=EXEC[M]['exec']['n_buy'], placebo_mean=float(np.mean(plb)), placebo_min=int(np.min(plb)), placebo_max=int(np.max(plb))),
                   from2019_0050=EXEC[M]['from2019']['cagr_0050'], placebo_share_beating_0050_from2019=float((pl19 > EXEC[M]['from2019']['cagr_0050']).mean()))
    # 貼文口徑每持有日毛報酬：真實訊號對池子（主規格出場、不限檔數、零成本）
    PD_ = {}
    for nm, sg in (('signal', P.sig), ('pool', P.pool)):
        tr = cw.post_trades2(P, sg, P.X, spp, U['TWSE'], t0)
        PD_[nm] = dict(n=int(len(tr)), mean_ret=float(tr['ret_tr'].mean()), win_rate=float((tr['ret_tr'] > 0).mean()), hold_mean=float(tr['hold'].mean()),
                       gross_bp_per_hold_day=float(tr['ret_tr'].sum() / tr['hold'].sum() * 1e4))
    PLACEBO['post_caliber_per_hold_day'] = PD_
    print('placebo done', round(time.time() - T0, 1), flush=True)

    # ---------------- β、和主策略
    def mm(nav, a=None):
        return cw.mret(nav, a), cw.mret(bench.reindex(nav.index), a)
    BETA = {}
    for nm in (M, '%s|E50' % pr, '%s|ALL' % pr, '%s|ALL_E50' % pr):
        BETA[nm] = {}
        for tag, a in (('full', None), ('from2019', A2019)):
            ms, mb = mm(navs[nm], a); BETA[nm][tag] = cw.beta_alpha(ms, mb)
    mn = cw.main_strategy_nav(START).reindex(P.IDX[t0:]).ffill()
    VS = {'main_series_check': dict(from2019=cw.wstats(mn, bench, A2019), full=cw.wstats(mn, bench))}
    for nm in (M, '%s|E50' % pr):
        VS[nm] = {}
        for tag, a in (('full', None), ('from2019', A2019)):
            ms, mb = mm(navs[nm], a); mo = cw.mret(mn, a); ix = ms.index.intersection(mo.index)
            d = (ms[ix] - mo[ix]).to_numpy(); mix = 0.5 * ms[ix] + 0.5 * mo[ix]; yrs = len(ix) / 12.0
            cum = np.cumprod(1 + mix.to_numpy()); cumo = np.cumprod(1 + mo[ix].to_numpy())
            VS[nm][tag] = dict(n=int(len(ix)), paired_mean_m=float(d.mean()), paired_t=cw.tstat(d), paired_nw6_t=cw.nw_t(d, 6), boot=cw.block_boot_ci(d),
                               corr_ret=float(np.corrcoef(ms[ix], mo[ix])[0, 1]), corr_excess=float(np.corrcoef(ms[ix] - mb[ix], mo[ix] - mb[ix])[0, 1]),
                               mix_cagr_m=float(cum[-1] ** (1 / yrs) - 1), main_cagr_m=float(cumo[-1] ** (1 / yrs) - 1), strat_cagr_m=float(np.prod(1 + ms[ix]) ** (1 / yrs) - 1),
                               e50_cagr_m=float(np.prod(1 + mb[ix]) ** (1 / yrs) - 1), mix_t_vs_0050=cw.tstat((mix - mb[ix]).to_numpy()), main_t_vs_0050=cw.tstat((mo[ix] - mb[ix]).to_numpy()),
                               mix_maxdd_m=float((cum / np.maximum.accumulate(np.concatenate([[1.0], cum]))[1:] - 1).min()),
                               main_maxdd_m=float((cumo / np.maximum.accumulate(np.concatenate([[1.0], cumo]))[1:] - 1).min()))

    # ---------------- 成本
    COST = {}
    for c in cw.CANDS:
        a = EXEC['%s|MAIN' % c]; z = EXEC['%s|ZERO' % c]; s = EXEC['%s|STRESS' % c]
        COST[c] = dict(drag_full_pp=(z['full']['cagr'] - a['full']['cagr']) * 100, drag_from2019_pp=(z['from2019']['cagr'] - a['from2019']['cagr']) * 100,
                       drag_stress_full_pp=(z['full']['cagr'] - s['full']['cagr']) * 100, turnover=a['exec']['turnover_oneway_per_year'], cost_acct_pct=a['exec']['cost_acct_pct_per_year'],
                       hold_days=a['trades']['hold_days_mean'], avg_npos=a['exec']['avg_npos'], avg_cash=a['exec']['avg_cash_frac'], n_buy=a['exec']['n_buy'],
                       expectancy_net=a['trades']['expectancy'], win_rate=a['trades']['win_rate'], avg_win=a['trades']['avg_win'], avg_loss=a['trades']['avg_loss'],
                       expectancy_zero=z['trades']['expectancy'], win_rate_zero=z['trades']['win_rate'], avg_win_zero=z['trades']['avg_win'], avg_loss_zero=z['trades']['avg_loss'])

    # ---------------- 檢定力（實際標準差）
    nd = NormalDist(); POWER = {}
    for c in cw.CANDS:
        f = EXEC['%s|MAIN' % c]['full']; g = EXEC['%s|MAIN' % c]['from2019']
        se_f = f['sd_ex_m'] / math.sqrt(f['n_months']); se_g = g['sd_ex_m'] / math.sqrt(g['n_months'])
        POWER[c] = dict(sd_full=f['sd_ex_m'], sd_2019=g['sd_ex_m'], need_m_full_t3=3 * se_f, need_ann_pp_full_t3=3 * se_f * 12 * 100, need_m_2019_t2=2 * se_g, need_ann_pp_2019_t2=2 * se_g * 12 * 100,
                        mde80_ann_pp_full_t3=(3 + 0.8416) * se_f * 12 * 100,
                        power_t3_if_true_3pp=1 - nd.cdf(3 - (0.03 / 12) / se_f), power_t3_if_true_5pp=1 - nd.cdf(3 - (0.05 / 12) / se_f),
                        power_t3_if_true_10pp=1 - nd.cdf(3 - (0.10 / 12) / se_f), expected_t_if_true_5pp=(0.05 / 12) / se_f)

    # ---------------- G3：SPA、DSR、Bonferroni
    N_new = PREREG['family']['N_after']
    names = ['%s|MAIN' % c for c in cw.CANDS]
    Xf = np.column_stack([(cw.mret(navs[n]) - cw.mret(bench.reindex(navs[n].index))).to_numpy() for n in names])
    X19 = np.column_stack([(cw.mret(navs[n], A2019) - cw.mret(bench.reindex(navs[n].index), A2019)).to_numpy() for n in names])
    srs = Xf.mean(0) / Xf.std(0, ddof=1)
    allx = []
    for n, v in navs.items():
        if '|EXT' in n:
            continue
        allx.append((cw.mret(v) - cw.mret(bench.reindex(v.index))).to_numpy())
    allx = np.column_stack(allx); sr_all = allx.mean(0) / allx.std(0, ddof=1)
    G3 = dict(N_before=N_BEFORE, n_new=81, N_after=N_new, n_series_run=len(navs), bonferroni_two_sided_t=float(nd.inv_cdf(1 - 0.025 / N_new)),
              bonferroni_one_sided_t=float(nd.inv_cdf(1 - 0.05 / N_new)),
              spa_full_10=cw.spa_test(Xf), spa_2019_10=cw.spa_test(X19), spa_full_all_series=cw.spa_test(allx),
              dsr_primary_var10=cw.dsr(Xf[:, 0], N_new, float(srs.var(ddof=1))), dsr_primary_var_all=cw.dsr(Xf[:, 0], N_new, float(sr_all.var(ddof=1))),
              sr_m_candidates={c: float(s) for c, s in zip(cw.CANDS, srs)}, best_candidate_by_t=list(cw.CANDS)[int(np.argmax(srs))])

    # ---------------- 判定
    e = EXEC; A = e[M]; S = e['%s|STRESS' % pr]
    seg = [A['segments'][a + '~' + b]['excess_pp'] for a, b in SEGS]
    nb19 = {c: e['%s|MAIN' % c]['from2019']['excess_pp'] for c in cw.CANDS if c != pr}
    cap = {k: e['%s|%s' % (pr, k)]['from2019']['excess_pp'] for k in ('C30', 'MAIN', 'C300', 'C1000')}
    p10 = PATH['P1_z_plus_noise']['from2019_excess_pp']['p10']
    G = {}
    G['G1'] = dict(passed=bool(A['from2019']['excess_pp'] > 0 and A['full']['excess_pp'] > 0 and S['from2019']['excess_pp'] > 0 and S['full']['excess_pp'] > 0 and p10 > 0),
                   main_2019=A['from2019']['excess_pp'], main_full=A['full']['excess_pp'], stress_2019=S['from2019']['excess_pp'], stress_full=S['full']['excess_pp'], path_P1_p10=p10,
                   path_P2_p10=PATH['P2_pure_random']['from2019_excess_pp']['p10'])
    G['G2'] = dict(passed=bool(A['full']['t_ex_m'] >= 3.0 and A['from2019']['t_ex_m'] >= 2.0), t_full=A['full']['t_ex_m'], t_2019=A['from2019']['t_ex_m'], nw6_full=A['full']['nw6_t'], nw6_2019=A['from2019']['nw6_t'])
    G['G3'] = dict(passed=bool(G3['spa_full_10']['p_consistent'] < 0.05 and A['full']['t_ex_m'] >= G3['bonferroni_two_sided_t']), spa_p=G3['spa_full_10']['p_consistent'],
                   spa_p_2019=G3['spa_2019_10']['p_consistent'], t_full=A['full']['t_ex_m'], bonferroni_t=G3['bonferroni_two_sided_t'], dsr=G3['dsr_primary_var10']['dsr'], N_after=N_new)
    G['G4'] = dict(passed=bool(min(seg) > 0 and A['drop_best3']['t'] >= 2.0 and min(nb19.values()) > 0), segments_excess_pp=seg, drop_best3_t=A['drop_best3']['t'], best3=A['drop_best3']['best3'],
                   neighbors_from2019_excess_pp=nb19, n_neighbors_negative=int(sum(v <= 0 for v in nb19.values())))
    G['G5'] = dict(passed=None, note='PIT、含下市股、無前視沿用前一題閘門（o_ext_canslim_xs 第 6 節與複製報告第 6 點：未發現實質前視）；附註見 FINDINGS')
    G['G6'] = dict(passed=False, note='待不同研究員只看規格獨立重做')
    G['G7'] = dict(passed=bool(cap['MAIN'] > 0 and cap['C300'] > 0 and cap['C1000'] > 0), from2019_excess_pp=cap)
    G['G8'] = dict(passed=False, note='前瞻尚未開始')
    hist_ok = all(G[k]['passed'] for k in ('G1', 'G2', 'G3', 'G4', 'G7'))
    G['verdict'] = '待 G6 複製一致後建議進前瞻追蹤' if hist_ok else '不可用（不建議進前瞻追蹤）'

    res['EXEC'] = cw.rnd(EXEC); res['PATH'] = cw.rnd(PATH); res['PLACEBO'] = cw.rnd(PLACEBO); res['BETA'] = cw.rnd(BETA); res['VS_MAIN'] = cw.rnd(VS)
    res['COST'] = cw.rnd(COST); res['POWER'] = cw.rnd(POWER); res['G3'] = cw.rnd(G3); res['GATES'] = cw.rnd(G)
    # ---------------- 引擎恆等式與輸出檔
    lg = logs[M]
    res['ENGINE_CHECK'] = cw.rnd(dict(nav_end=float(navs[M].iloc[-1]), capital_plus_sum_trade_pnl=float(1e6 + (lg['proceeds'] - lg['cost']).sum())), 2)
    keep = ['%s|MAIN' % c for c in cw.CANDS] + ['%s|%s' % (pr, k) for k in ('ZERO', 'STRESS', 'E50', 'ALL', 'ALL_E50', 'TPEX', 'EXT', 'NEXTDAY', 'NOLOT')] + ['POOL_RS']
    T_ = []
    for n in keep:
        x = logs[n][['code', 'sig_date', 'ent_date', 'exit_date', 'ret', 'why', 'shares', 'disp', 'cost', 'proceeds']].copy(); x.insert(0, 'variant', n); T_.append(x)
    T_ = pd.concat(T_, ignore_index=True)
    for c in ('sig_date', 'ent_date', 'exit_date'):
        T_[c] = pd.to_datetime(T_[c]).dt.strftime('%Y-%m-%d')
    T_['ret'] = T_['ret'].round(6); T_['cost'] = T_['cost'].round(2); T_['proceeds'] = T_['proceeds'].round(2)
    T_.to_csv(OUT / 'trades.csv.gz', index=False, compression='gzip')
    nv = pd.DataFrame({k: v for k, v in navs.items()}); nv['0050'] = bench.reindex(nv.index); nv['V7X_jmean_norm'] = mn.reindex(nv.index)
    nv.round(2).to_csv(OUT / 'navs.csv.gz', compression='gzip')
    res['RUN'] = dict(finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'), seconds=round(time.time() - T0, 1),
                      cwlib_sha256=cw.sha_file(HERE / 'cwlib.py'), run_sha256=cw.sha_file(HERE / 'run.py'), prereg_sha256=res['PREREG']['sha256'],
                      cslib_sha256=cw.sha_file(cw.AG / 'o_ext_canslim_xs' / 'cslib.py'), n_series=len(navs))
    save_res(res)
    print('done', round(time.time() - T0, 1), 's')


# =============================================================== posthoc（看過報酬後才加：引擎驗證、與複製員數字對照、事後分析；全部標「事後」）
def verify_exits(P, lg, sp, t_last):
    """獨立重算每筆成交的出場訊號日 k*（不經引擎）：k*＝進場日 e 起第一個任一規則成立的日子；預期賣出日＝k*+1 起第一個有開盤價且開盤 > 跌停價的日子。"""
    O, C, R = P.O, P.CF, P.R
    bad = []; n = 0; why_bad = 0
    for j, e, x, why in zip(lg['j'], lg['ent_t'], lg['exit_t'], lg['why']):
        if why == 'delist':
            continue
        n += 1
        g = np.concatenate([[C[e, j] / O[e, j]], 1 + R[e + 1:t_last + 1, j]]).cumprod()
        hit = np.zeros(t_last + 1 - e, bool); lab = {}
        if sp['trail']:
            lab['trail'] = P.X['trail%d' % sp['trail']][e:t_last + 1, j]
        if sp['cross']:
            lab['cross'] = P.X['cross'][e:t_last + 1, j]
        if sp['ma50']:
            lab['ma50'] = P.X['ma50'][e:t_last + 1, j]
        if sp['fstop']:
            lab['fstop'] = g <= (1 - sp['fstop'] / 100.0)
        if sp['hold']:
            h = np.zeros(len(hit), bool); h[sp['hold'] - 1:] = True; lab['hold'] = h
        for v in lab.values():
            hit |= v
        nz = np.flatnonzero(hit)
        if len(nz) == 0:
            ok = (why == 'open_end'); xs = None
        else:
            k = e + nz[0]; xs = k + 1
            while xs <= t_last and not (np.isfinite(O[xs, j]) and O[xs, j] > 0 and O[xs, j] > P.LIMDN[xs, j] + 1e-9):
                xs += 1
            if xs > t_last:
                ok = (why == 'open_end')
            else:
                ok = (x == xs)
                w = '+'.join(kk for kk in ('trail', 'cross', 'ma50', 'fstop', 'hold') if kk in lab and lab[kk][nz[0]])
                why_bad += int(w != why)
        if not ok:
            bad.append((P.COLS[j], int(e), int(x), None if xs is None else int(xs), why))
    return dict(n_checked=n, n_exit_day_mismatch=len(bad), n_why_mismatch=why_bad, mismatches=bad[:10])


def stage_posthoc():
    T0 = time.time()
    res = load_res(); assert 'EXEC' in res
    P, t0, bench, U, disp, e50 = setup()
    pr = cw.PRIMARY; spp = cw.CANDS[pr]; PH = {'label': '事後：以下全部是看過報酬之後才加的，不屬於事前登記的主欄，讀法不得當成建議'}
    # ---- (0) 引擎驗證：每個候選 MAIN 欄逐筆重算出場日
    EV = {}; logs = {}
    for c, sp in cw.CANDS.items():
        nav, lg, st = cw.simulate2(P, P.sig, P.RS, P.X, sp, U['TWSE'], t0, disp=disp)
        assert abs(cw.wstats(nav, bench)['cagr'] - res['EXEC']['%s|MAIN' % c]['full']['cagr']) < 1e-6
        EV[c] = verify_exits(P, lg, sp, P.T - 1); EV[c]['n_sell_partial_cap'] = st['sell_partial_cap']; logs[c] = (nav, lg, st)
    res['EXIT_VERIFY'] = cw.rnd(EV)
    # ---- (1) 與複製員 review_numbers.json 對照（登記後才開啟）
    rv = json.load(open(cw.AG / 'o_ext_canslim_xs' / 'replicate' / 'review_numbers.json'))['variants']
    XC = {}
    for mine, theirs in (('T12_MA|MAIN', 'twse_cash_trail12'), ('T15_MA|MAIN', 'twse_cash_trail15'), ('T20_MA|MAIN', 'twse_cash_trail20'),
                         ('T12_MA|ALL', 'both_cash_trail12'), ('T15_MA|ALL', 'both_cash_trail15'), ('T20_MA|ALL', 'both_cash_trail20'),
                         ('T12_MA|E50', 'twse_e50_trail12'), ('T15_MA|E50', 'twse_e50_trail15'), ('T20_MA|E50', 'twse_e50_trail20'), ('T15_MA|ALL_E50', 'both_e50_trail15')):
        a = res['EXEC'][mine]; b = rv[theirs]
        XC[mine] = dict(replicator_key=theirs, mine_2019=a['from2019']['cagr'], rep_2019=b['from2019']['ann'], mine_full=a['full']['cagr'], rep_full=b['full']['ann'],
                        mine_t_full=a['full']['t_ex_m'], rep_t_full=b['full']['t'], diff_2019_pp=(a['from2019']['cagr'] - b['from2019']['ann']) * 100, diff_full_pp=(a['full']['cagr'] - b['full']['ann']) * 100)
    PH['crosscheck_vs_replicator'] = XC
    # ---- (2) 安慰劑：零成本版（拿掉「安慰劑買進筆數較多 → 成本較高」的不對稱）與筆數
    sig_u = P.sig & U['TWSE']; pool_u = P.pool & U['TWSE']
    z19 = []; zf = []
    for i in range(NPATH):
        ps = cw.placebo_signal(P, sig_u, pool_u, 9000 + i)
        nav, _, _ = cw.simulate2(P, ps, P.RS, P.X, spp, U['TWSE'], t0, disp=disp, cost='zero', keep_log=False)
        z19.append(cw.wstats(nav, bench, A2019)['cagr']); zf.append(cw.wstats(nav, bench)['cagr'])
    z19 = np.array(z19); zf = np.array(zf); mz = res['EXEC']['%s|ZERO' % pr]
    PH['placebo_zero_cost'] = dict(n=NPATH, from2019=dict(main=mz['from2019']['cagr'], placebo_median=float(np.median(z19)), p5=float(np.quantile(z19, .05)), p95=float(np.quantile(z19, .95)),
                                                         main_percentile=float((z19 < mz['from2019']['cagr']).mean())),
                                   full=dict(main=mz['full']['cagr'], placebo_median=float(np.median(zf)), p5=float(np.quantile(zf, .05)), p95=float(np.quantile(zf, .95)),
                                             main_percentile=float((zf < mz['full']['cagr']).mean())))
    # ---- (3) 集中度：主規格主欄損益來自多少筆
    nav, lg, st = logs[pr]
    pnl = (lg['proceeds'] - lg['cost']).to_numpy(); o = np.argsort(-pnl); tot = pnl.sum()
    top = lg.iloc[o[:10]]
    PH['concentration_main'] = dict(n_trades=int(len(lg)), total_pnl=float(tot), top5_share=float(pnl[o[:5]].sum() / tot), top10_share=float(pnl[o[:10]].sum() / tot), top20_share=float(pnl[o[:20]].sum() / tot),
                                    top10=[dict(code=r.code, ent=str(r.ent_date.date()), exit=str(r.exit_date.date()), ret=float(r.ret), pnl=float(r.proceeds - r.cost), why=r.why) for r in top.itertuples()],
                                    pnl_by_exit_year={int(y): float(v) for y, v in (lg['proceeds'] - lg['cost']).groupby(lg['exit_date'].dt.year).sum().items()})
    # ---- (4) 空位放 0050 版（事前登記的並列欄）的路徑擾動與分段——事後補跑
    ex19 = []
    Z = cw.zrank(sig_u, P.RS)
    for i in range(NPATH):
        nav_, _, _ = cw.simulate2(P, P.sig, Z + np.random.default_rng(1000 + i).standard_normal(Z.shape), P.X, spp, U['TWSE'], t0, disp=disp, idle='e50x', e50=e50, keep_log=False)
        ex19.append(cw.wstats(nav_, bench, A2019)['excess_pp'])
    E5 = res['EXEC']['%s|E50' % pr]
    PH['e50_version'] = dict(path_P1_from2019_excess_pp=dict(median=float(np.median(ex19)), p10=float(np.quantile(ex19, .1)), p90=float(np.quantile(ex19, .9)), share_gt0=float((np.array(ex19) > 0).mean())),
                             segments_excess_pp=[E5['segments'][a + '~' + b]['excess_pp'] for a, b in SEGS], drop_best3_t=E5['drop_best3']['t'], best3=E5['drop_best3']['best3'],
                             stress_2019=res['EXEC']['%s|E50_STRESS' % pr]['from2019']['excess_pp'], stress_full=res['EXEC']['%s|E50_STRESS' % pr]['full']['excess_pp'],
                             t_full=E5['full']['t_ex_m'], t_2019=E5['from2019']['t_ex_m'])
    # ---- (5) 只上櫃為什麼差：持股檔數、現金、零成本
    nav_, lg_, st_ = cw.simulate2(P, P.sig, P.RS, P.X, spp, U['TPEX'], t0, disp=disp, cost='zero')
    tp = res['EXEC']['%s|TPEX' % pr]
    PH['tpex_diag'] = dict(avg_npos=tp['exec']['avg_npos'], avg_cash=tp['exec']['avg_cash_frac'], n_buy=tp['exec']['n_buy'], blk_lot=tp['exec']['blk_lot'], expectancy_net=tp['trades']['expectancy'],
                           win_rate=tp['trades']['win_rate'], zero_cost_2019=cw.wstats(nav_, bench, A2019)['cagr'], zero_cost_full=cw.wstats(nav_, bench)['cagr'],
                           signal_stock_days=int((P.sig & U['TPEX'])[t0:].sum()), signal_stock_days_twse=int(sig_u[t0:].sum()))
    # ---- (6) 逐年超額為正的年數、月超額為正比例（主規格）
    yr = res['EXEC']['%s|MAIN' % pr]['drop_best3']['yearly']
    PH['years_positive_main'] = dict(n_years=len(yr), n_pos=int(sum(v['ex_pp'] > 0 for v in yr.values())))
    res['POSTHOC'] = cw.rnd(PH)
    res['RUN']['posthoc_finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
    res['RUN']['run_sha256_final'] = cw.sha_file(HERE / 'run.py'); res['RUN']['cwlib_sha256_final'] = cw.sha_file(HERE / 'cwlib.py')
    save_res(res); print('posthoc done', round(time.time() - T0, 1))


if __name__ == '__main__':
    st = sys.argv[1] if len(sys.argv) > 1 else 'main'
    {'check': stage_check, 'prereg': stage_prereg, 'main': stage_main, 'posthoc': stage_posthoc}[st]()
