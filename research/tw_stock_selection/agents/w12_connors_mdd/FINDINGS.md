# FINDINGS：Connors 多日下跌（MDD, Multiple Days Down），200 日線上、5 日線下、5 日中 4 日收黑時買，站上 5 日線時賣（w12_connors_mdd）

## 結論：不可用
- **PRIMARY 是書中積極版 AGG_NT**：最多加碼到 2 單位，收盤站上 SMA5 時出場，不設停損，20 槽組合。
  - 挑選期組合淨年化 4.7%，輸 0050 的 7.7%，對等權月超額 t −0.64。
  - 驗證期淨年化 −18.9%，0050 是 28.3%、同池等權 16.8%；對 0050 t −6.01，對等權 t −7.95。
  - 兩段都沒有通過，所以判定不可用。
- **4 個變體全部不可用**：AGG／STD 各配 NT（不設時間停損）與 T10（10 日時間停損）。另一組敏感度 AGG40（每單位 1/40）也一樣。
- **扣成本前的數字**：
  - 標準版 STD_NT 挑選期事件毛異常每筆 +0.53%（t 6.5），驗證期只剩 +0.12%（t 2.3），2024–2025 是 +0.06%（t 0.9）。
  - 組合毛年化：挑選期 52.6%（積極版）／44.4%（標準版），驗證期 18.2%／14.2%，兩版驗證期都輸 0050 的 28.3%。
- **扣成本後的數字**：
  - 平均每筆持有約 3.9 個交易日。積極版一年約 730 筆進場加 420 次加碼，每單位 5% 淨值。
  - 來回成本 0.67% 時，每年成本約占淨值 39%。
  - 即使用教學的 0.5% 來回，驗證期淨年化也只有 −10.6%。
- **處理建議**：歸檔，不延伸調參。
  - 這和已測的 Double 7（w3_connors_double7）、RSI(2)+SMA5（w3_connors_rsi2_sma5）、乖離率反轉結論相同：台股短線超賣反彈有毛利，但 2012 年後毛利變小，扣台股成本後是負的。
  - 「5 日中 4 日收黑」這個計數條件沒有帶來新的資訊。71% 的 MDD 訊號同一天也是 Double 7 的 7 日最低收盤，51% 同時是 RSI(2) < 10。

## 事前設定（results.json → PREREG；prereg.py 在 2026-10-27 14:12 寫入，看結果前固定，之後沒改）
- **股池**
  - 上市普通股，排除 91xx。
  - t 日 60 日均成交值 ≥ NT$1,000 萬（停牌日記 0），原始收盤 ≥ 10 元，flag ∈ {0,1,2}。
  - 有值檔數 < 300 的日子（2018-01-13）整列剔除後才做 rolling。
  - flag 5 的價格視為缺值。
  - [t−5, t] 內有 flag 3/4/5 就剔除該訊號；全市場 flag 4 ≥ 50 檔的缺口日（2024-02-15、2024-03-18）不算。
  - t 或 t+1 在處置期間就剔除（處置名單 2021-09 起才有）。
- **指標**
  - 只用還原收盤 adj，避免除權息造成假收黑；先 ffill，最多 10 日，停牌日不算收黑。
  - SMA200、SMA5 是簡單均線。
  - DOWN5 = 最近 5 日中「收盤 < 前一日收盤」的天數，平盤不算。
- **訊號（t 日收盤判斷）**：adj > SMA200、adj < SMA5、DOWN5 ≥ 4。
- **進場**
  - t+1 開盤買。
  - 開盤漲停（2015-06 前 7%、之後 10%，各減 0.6%）或沒有開盤價時買不到：組合換下一個候選，事件層直接放棄。
- **積極版加碼**
  - 持有中某日收盤低於首筆進場價（還原開盤），隔日開盤加碼一單位，最多 2 單位。
  - 同一天也觸發出場時，出場優先。
  - 組合層的加碼只用現金，不用槓桿，執行順序是先賣、再加碼、再新進場。
- **出場**
  - 從進場日收盤起，收盤 > SMA5 就隔日開盤全部賣出。
  - 不設停損。
  - T10 變體：持有到第 10 個交易日收盤仍未出場，第 11 天開盤賣出。
  - 鎖跌停或停牌時順延；下市用最後收盤。
- **組合**
  - 最多 20 檔，每單位等於開盤淨值的 1/20。
  - 同日訊號依距 5 日均線的跌幅 adj/SMA5−1 由深到淺排序（任務指定）；同值時成交值大者優先。
  - 現金 0 報酬；2004-07 起模擬，統計從 2005-01 起。
- **成本**：判定用 exlib.cost_rates(0.001)。
  - 買：0.0855%＋0.1%。
  - 賣：0.0855%＋稅 0.3%＋0.1%。
  - 來回 0.67%。另報不扣成本（gross）、教學口徑來回 0.5%（rt0.5）、壓力情境來回 1%。
- **判定（與 Double 7／RSI(2) 同口徑）**
  - 一段要通過，PRIMARY 在該段須同時滿足兩個條件：
    1. 事件淨異常報酬（對同池等權，開盤對開盤）> 0，且 t ≥ 2（按進場月份分群）。
    2. 20 槽組合淨年化 ≥ 0050。
  - 兩段都過為可用，一段過為待觀察，都沒過為不可用。
- **0050 單一標的**
  - etf_0050.csv 只有含息收盤、沒有開盤價，所以改成隔日收盤成交（仍然可成交）。
  - 成本：ETF 稅 0.1%＋滑價 0.1%，來回約 0.47%。
  - 只當補充，不進主判定。

## PRIMARY（AGG_NT）20 槽組合（滑價 0.1%）
| 期間 | 毛年化 | 來回 0.5% | 淨年化 | 壓力 | 淨 MDD（日頻） | 0050 | 同池等權 | 對 0050 t | 對等權 t |
|---|---|---|---|---|---|---|---|---|---|
| 挑選 2005-01～2019-04 | 52.6% | 15.3% | 4.7% | −12.9% | −62.6% | 7.7% | 7.1% | −0.57 | −0.64 |
| 驗證 2019-05～2026-10 | 18.2% | −10.6% | −18.9% | −33.0% | −83.8% | 28.3% | 16.8% | −6.01 | −7.95 |
| 2024～2025 | 1.8% | −20.4% | −26.8% | −37.7% | −54.6% | 42.4% | 8.1% | −4.96 | −5.79 |
| 挑選前半 2005～2011 | 70.5% | 30.3% | 18.7% | −0.6% | −34.7% | 4.8% | 5.8% | 1.62 | 1.33 |
| 挑選後半 2012～2019-04 | 37.3% | 2.7% | −7.1% | −23.3% | −59.3% | 10.5% | 8.4% | −2.94 | −3.29 |
| 驗證（只有存活者 2019-05～2023-12） | 18.5% | −11.6% | −19.8% | | −69.8% | 14.7% | 16.9% | −4.04 | −6.35 |
| 驗證（只用官方價，剔除 src 3） | 24.5% | −6.9% | −15.9% | | −71.1% | 24.7% | 19.5% | −4.50 | −6.24 |
| 2021 起切分：2021～2023 | 12.6% | −15.5% | −23.6% | | −62.1% | 6.9% | 15.1% | −2.65 | −5.55 |
| 2021 起切分：2024～2026-10 | 17.6% | −9.1% | −17.4% | | −56.7% | 54.5% | 16.8% | −4.60 | −4.72 |

- **組合運作（全期，滑價 0.1%）**
  - 新進場 16,154 筆，加碼 9,290 次。
  - 現金不足沒能加碼 4,054 次，槽位滿沒買 125,297 次，開盤漲停跳過 282 次。
  - 平均持股 11.8 檔，平均曝險 87%。
- **逐年淨報酬（%）**
  - 2005 +0.1、2006 +62.4、2007 +31.0、2008 −8.9、2009 +93.9、2010 +22.5
  - 2011 −27.9、2012 −7.1、2013 +27.0、2014 −3.4、2015 −21.8、2016 −6.3
  - 2017 −5.8、2018 −29.4、2019 −3.6、2020 −12.9、2021 −19.4、2022 −47.3
  - 2023 +5.4、2024 −14.2、2025 −37.7、2026 +9.1
- **逐年毛報酬（%）**：2006 +152、2009 +175、2013 +100、2019 +43、2022 −24、2025 −13、2026 +53。不扣成本時，每年都有不錯的報酬。
- **同期 0050（%）**：2019 33.5、2020 31.1、2021 21.9、2022 −21.4、2023 27.5、2024 48.6、2025 36.8、2026 67.2。

## 事件層（不受槽位限制；事件以訊號日歸屬；t 為月分群）
| 變體／期間 | 筆數 | 持有 | 毛異常 (t) | 來回 0.5% (t) | 淨異常 (t) | 壓力 (t) |
|---|---|---|---|---|---|---|
| STD_NT 挑選 | 38,905 | 3.9 日 | +0.529% (6.49) | +0.027% (0.32) | −0.145% (−1.81) | −0.472% (−5.86) |
| STD_NT 驗證 | 29,792 | 3.9 日 | +0.118% (2.34) | −0.383% (−7.71) | −0.555% (−11.18) | −0.881% (−17.8) |
| STD_NT 2024～2025 | 7,284 | 4.0 日 | +0.061% (0.86) | −0.439% (−3.90) | −0.610% (−5.54) | −0.936% (−8.68) |
| AGG_NT 挑選（事前公式） | 38,905 | 3.9 日 | +0.845% (9.73) | +0.341% (4.14) | **+0.168% (2.21)** | −0.161% (−1.46) |
| AGG_NT 驗證（事前公式） | 29,792 | 3.9 日 | +0.488% (10.03) | −0.016% (−0.51) | −0.189% (−4.15) | −0.517% (−11.1) |
| AGG_NT 2024～2025（事前公式） | 7,284 | 4.0 日 | +0.446% (4.81) | −0.056% (−0.32) | −0.228% (−2.09) | −0.556% (−5.47) |
| **AGG_NT 挑選（每單位等金額，事後修正）** | 62,908 單位 | | +0.547% (5.56) | +0.045% (0.69) | −0.127% (−0.98) | −0.455% (−4.18) |
| **AGG_NT 驗證（每單位等金額，事後修正）** | 48,740 單位 | | +0.123% (2.25) | −0.379% (−7.41) | −0.551% (−10.75) | −0.877% (−17.1) |
| AGG_T10 挑選／驗證（事前公式） | 39,081／30,013 | 3.8 日 | +0.824% (9.21)／+0.486% (10.11) | | +0.147% (1.83)／−0.191% (−4.23) | |
| STD_T10 挑選／驗證 | 39,081／30,013 | 3.8 日 | +0.508% (5.95)／+0.116% (2.34) | | −0.165% (−2.14)／−0.557% (−11.4) | |

- **事前公式高估了積極版，這裡要特別說明**：
  - 事前公式把「有加碼」的事件算成兩單位報酬的平均，把「沒加碼」的事件算成滿額一單位，兩類事件用的資金不同，卻各算一筆、等權平均。
  - 沒加碼的事件通常是很快反彈的好交易（毛異常 +2.10%、持有 1.8 日）。有加碼的事件首筆單位平均 −0.87%，兩單位平均才把虧損攤薄。
  - 改成每單位等金額加權（加碼事件權重 2），挑選期淨異常是 −0.127%（t −0.98）。改成固定資金版（未加碼的另一半放現金）是 −0.103%（t −1.16）。
  - 所以挑選期事件層的 t 2.21 是口徑造成的，不是真的贏等權。
  - 判定仍照事前規則：挑選期組合年化 4.7% < 0050 7.7%，本來就不通過，判定不受這個問題影響。
- **加碼單位本身**：挑選期加碼單位毛異常 +0.58%，驗證期 +0.13%，2024–2025 只有 +0.04%。加碼也付不起 0.67% 的來回成本。
- **出場原因**：約 100% 是收盤站上 SMA5。T10 變體約 3% 碰到時間停損。
- **淨勝率**：STD 挑選期 55%、驗證期 53%；AGG 挑選期 61%、驗證期 59%。勝率高但每筆平均報酬小，是典型的均值回歸樣子。
- **逐年事件淨異常（STD_NT，滑價 0.1%）**
  - 只有 2004、2006、2007、2009 是正的，其中最大的 2006 年也只有 +0.15%。
  - 2012 起每年都是負的，2017 起 ≤ −0.35%，2022 是 −0.77%，2023–2026 在 −0.55%～−0.65%。
  - 毛異常從 2005–2011 年的 +0.3%～+0.8%，降到 2022–2026 年的 −0.11%～+0.12%。
- **每筆損益兩平的來回成本**（STD 毛異常）：挑選期 0.53%、驗證期 0.12%、2024–2025 為 0.06%。光證交稅 0.3% 就超過驗證期的兩平點。

## 其他變體的組合層（滑價 0.1%；淨年化／MDD／對 0050 t／對等權 t）
| 變體 | 挑選 2005-01～2019-04 | 驗證 2019-05～2026-10 | 2024～2025 | 毛年化（挑選／驗證） |
|---|---|---|---|---|
| AGG_NT（PRIMARY） | 4.7% / −62.6% / −0.57 / −0.64 | −18.9% / −83.8% / −6.01 / −7.95 | −26.8% / −54.6% / −4.96 / −5.79 | 52.6% / 18.2% |
| STD_NT | 3.9% / −58.5% / −1.01 / −1.07 | −18.9% / −83.4% / −6.56 / −9.67 | −25.9% / −51.3% / −5.12 / −6.61 | 44.4% / 14.2% |
| AGG_T10 | 3.6% / −66.4% / −0.82 / −0.89 | −21.4% / −87.3% / −6.19 / −8.26 | −28.1% / −56.7% / −4.92 / −5.90 | 53.5% / 17.2% |
| STD_T10 | 3.2% / −60.2% / −1.17 / −1.22 | −20.0% / −85.3% / −6.48 / −9.16 | −26.8% / −52.2% / −5.34 / −6.71 | 44.9% / 14.9% |
| AGG40_NT（每單位 1/40，敏感度） | 4.2% / −50.0% / −1.07 / −1.05 | −16.8% / −78.6% / −6.59 / −9.60 | −23.9% / −47.7% / −5.21 / −6.85 | 36.0% / 11.1% |

- 4 個判定變體在兩段都不通過，全部是不可用。10 日時間停損和加碼都救不回來。

## 0050 單一標的（補充，不進主判定；隔日收盤成交，ETF 成本來回約 0.47%）
| 變體 | 期間 | 淨年化 | MDD | 0050 買進持有 | 對 0050 t | 筆數 | 每筆淨報酬 | 每筆毛報酬 |
|---|---|---|---|---|---|---|---|---|
| AGG_NT | 挑選 | −0.8% | −15.3% | 7.7% | −2.29 | 100 | +0.08% | +0.56% |
| AGG_NT | 驗證 | +0.4% | −8.3% | 28.3% | −3.73 | 60 | +0.43% | +0.90% |
| AGG_NT | 2024～2025 | −2.3% | −7.4% | 42.4% | −2.73 | 18 | −0.26% | +0.21% |
| STD_NT | 挑選／驗證 | −0.7%／+1.4% | −16.9%／−13.6% | 7.7%／28.3% | −2.26／−3.56 | 100／60 | −0.08%／+0.30% | +0.39%／+0.78% |

- 0050 平均每年只有 7 次訊號，每筆持有約 3 日，大部分時間是空手。
- 每筆淨報酬的 t 都 < 2；扣成本後年化接近 0，遠輸買進持有。
- 原版在美國 ETF 上有效的前提是空手時間不花成本、每筆報酬夠大，這兩點在 0050 都不成立。

## 診斷
- **隔夜跳空不是主因**：
  - 訊號日收盤到隔日開盤平均跳空 +0.12%（挑選期）、+0.24%（驗證期）。
  - 假設能用訊號日收盤成交（實際做不到），STD 毛異常是 +0.64%（挑選期）／+0.35%（驗證期），驗證期仍不夠付 0.67%。
- **組合毛報酬來自跌幅排序**：
  - 事後分層（不改判定）顯示，挑選期距 SMA5 跌幅 > 6% 的訊號 STD 淨異常 +1.04%；跌幅 < 2% 的是 −0.45%。
  - 隨機排序 10 種子（PRIMARY）：挑選期淨年化平均 −5.5%（介於 −7.4%～−4.1%），對等權 t −3.26；驗證期平均 −21.8%，對等權 t −10.7。
  - 跌幅排序比隨機好約 10 個百分點／年，但仍然輸等權與 0050。
- **與已測規則重疊很高**：71% 的 MDD 訊號同日是 Double 7 訊號，51% 是 RSI(2) < 10。MDD 是同一個超賣反彈因子換一種寫法。
- **資料檢查**：
  - 抽 300 筆 PRIMARY 事件，用獨立程式重算訊號三條件、進場價、出場日，0 筆不符，沒有前視。
  - 共 68,896 筆事件，持有期碰到 flag 3/4/5 的 88 筆；報酬 > 100% 的 0 筆，< −50% 的 4 筆。
  - 剔除這些後，AGG 事前公式淨異常挑選期 +0.176%（t 3.1）、驗證期 −0.186%（t −4.1），結論方向不變。挑選期 t 偏高的原因見上面的口徑說明。
  - 驗證期事件 21% 碰到 src 3 近似收盤期。只用官方價的區段，STD 淨異常 −0.52%（t −7.3），組合淨年化 −15.9%，結論不變。
  - 只有存活者的 2019-05～2023-12（對抄底偏樂觀）：STD 淨異常 −0.52%（t −6.7），組合淨年化 −19.8%，仍為負。
  - STD_T10 有 1 筆（2432，2008-08 訊號）在時間停損日後長期停牌，價格凍結到 2023 年才賣出，造成組合賣出順延 3,754 次（diag_delay.py）。這只占 5% 部位，而且 STD_T10 不是 PRIMARY，不影響結論。

## 限制
- 2019–2023 面板只有存活者，抄底類結果偏樂觀，但結論是負的，不受影響。
- 2024-01-02～02-14 無資料；src 3 期間收盤是 13:20 近似值。
- 面板沒有最低價，鎖跌停用最高價判斷。
- 處置名單 2021-09 起才有。
- 原版是收盤成交，台股改成隔日開盤，已用「收盤進場」做上限比較，仍不夠付成本。
- 0050 沒有開盤價，單一標的改用隔日收盤成交。
- 事前的積極版事件公式口徑有瑕疵（見事件層說明），已在 results.json → diag_post_hoc 附上每單位加權與固定資金兩種修正口徑。判定照事前規則進行，結果不變。

## 檔案
- /home/user/-/research/tw_stock_selection/agents/w12_connors_mdd/prereg.py（事前設定）
- /home/user/-/research/tw_stock_selection/agents/w12_connors_mdd/run.py（回測；成本來自 exlib，股池與引擎沿用 w3_connors_double7）
- /home/user/-/research/tw_stock_selection/agents/w12_connors_mdd/results.json（PREREG、meta、events、portfolio、single_0050、data_checks、judgement、diag_post_hoc）
- /home/user/-/research/tw_stock_selection/agents/w12_connors_mdd/run_log.txt
- /home/user/-/research/tw_stock_selection/agents/w12_connors_mdd/diag_agg_unit.py、diag_agg_unit_log.txt（積極版每單位加權修正、300 筆抽樣重算）
- /home/user/-/research/tw_stock_selection/agents/w12_connors_mdd/diag_delay.py（STD_T10 賣出順延原因）
- /home/user/-/research/tw_stock_selection/agents/w12_connors_mdd/nav.parquet（各變體與各成本情境的 NAV、同池等權、0050）
- /home/user/-/research/tw_stock_selection/agents/w12_connors_mdd/primary_events.csv.gz（PRIMARY 事件明細）


## 反方審查

{
 "final": "不可用",
 "issues": [
  "Look-ahead bias: none found. The signal uses the close on day t, entry is at the t+1 open, and exits are triggered by the close on day s and filled at the s+1 open. SMA200, SMA5 and the down-day count use only data up to day t. The adjusted price is a ratio adjustment, so it does not shift signals. The researcher's 300-event spot check had 0 mismatches. One small point: the disposal-stock filter also uses whether t+1 is in a disposal period. That is announced after the day-t close and only removes signals, so it does not make results look better. Not counted as a problem.",
  "The claim that the signal makes money before costs is overstated. At the event level the gross edge is real (abnormal return vs same-pool equal weight: selection +0.845% per trade, t 9.7; validation +0.488%, t 10.0). But the 20-slot portfolio before costs in validation is 18.2% a year, against 0050 at 28.3% and equal weight at 16.8% (t vs EW 0.57, t vs 0050 -0.85; 2024–2025 only +1.75%). So even at zero cost it does not meet the rule of beating equal weight with t of at least 2 and not losing to 0050. Saying the losses come only from costs is wrong. The edge is too thin to survive even before costs.",
  "The gross edge shrinks steadily over time. Gross abnormal return per event: 2005–2011 1.02%, 2012–2019/04 0.69%, validation 0.49%, 2024–2025 0.45%. Net abnormal return by year has been negative every year since 2016. In 2012–2019/04, part of the selection period, the net portfolio already returned −7.1% a year (t vs EW −3.29). The selection-period net 4.7% comes almost entirely from 2005–2011 (2006 +62%, 2009 +94%), so even inside the selection period it relies on a few early years.",
  "The event-level pass in the selection period (net abnormal +0.168%, t 2.21) is an artifact of how the aggressive version's events are counted. For trades with an add-on, the two units' returns are averaged; trades without an add-on count at full size. Weighted per equal-money unit, the result is −0.127% (t −0.98). With fixed capital it is −0.103% (t −1.16). The standard version is −0.145% (t −1.81). The researcher noted this in diag_post_hoc and did not use it to change the verdict, but it means no reading of the selection period clearly passes.",
  "Costs are not set too harsh. The 0.67% round trip is just the 0.0855% fee each way (6/10 of the 0.1425% list fee), the 0.3% transaction tax, and 0.1% slippage each way. At 0.5% round trip the validation portfolio is still −10.6% a year (t vs EW −5.63). Holding is 3.9 days on average with about 16k entries plus 9.3k add-ons, so turnover is very high. The breakeven round-trip cost is only about 0.49% in validation and about 0.45% in 2024–2025, below the tax-plus-fee floor (about 0.47%).",
  "Limit-up and limit-down handling is if anything slightly favourable to the strategy. A limit-up open blocks a buy, which is fine. A sell counts as blocked only if the day's high is at limit-down. A stock that opens at limit-down and then rebounds is assumed sold at the open, which may not fill in practice. The panel has no daily low and no order-queue data, so this direction flatters the strategy. Only 19 sells were delayed, so the effect is small and does not change the verdict.",
  "Survivorship bias in 2019–2023 makes results look better for a buy-the-dip rule, yet the survivor-only segment (2019-05 to 2023-12) is still −19.8% a year net (t vs EW −6.35). Correcting the bias would only make it worse. The negative verdict holds.",
  "The src=3 approximate closes (13:20 prices) affect only signal and exit decisions; they cause no look-ahead. Excluding src 3, validation is still −15.9% a year (t vs EW −6.24), about 21% of validation events touch src 3, and the conclusion is unchanged. The 2024-01-02 to 2024-02-14 data gap means holdings jump over missing days; the effect is small.",
  "Sample size and extreme events are not a problem. There are about 39k selection events and 30k validation events over 173 and 89 months. No trade gained more than 100% and 4 lost more than 50%. Removing flag 3/4/5 trades and extremes gives selection +0.176% and validation −0.186% net abnormal per event, so the result does not hinge on a few events.",
  "Post-hoc tuning: PREREG was written at 14:12, run.py was written after that, and all variants (STD, T10, 1/40 NAV per unit) and 10 random orderings fail. The deepest-drop ordering was set by the task, and random ordering does worse (selection CAGR −5.5%). There is no sign of tuning after seeing results. diag_agg_unit was added afterwards as a diagnostic and does not change the verdict.",
  "How t is computed: the portfolio t uses non-overlapping monthly excess returns and the event t uses entry-month clustering, which is reasonable. t_date is larger (selection 2.78) and should not be quoted; the researcher already marks it reference-only.",
  "Minor diagnostic flaw: 'mean_ab_gross_if_close_entry' uses only the first unit's return (raw1), while the trade's main return under the aggressive version averages both units. The two aren't comparable, so the overnight-gap diagnostic is unreliable. It does not affect the verdict.",
  "Risk side: net maximum drawdown is −62.6% in the selection period and −83.8% in validation, which is clearly worse than 0050 and equal weight. There is no reason to downgrade to 'watch' (待觀察)."
 ],
 "note": "I agree it is not usable (不可用), and the verdict holds under every check I ran. I found no look-ahead bias, and survivorship bias and the src=3 approximate closes both push results upward, yet the strategy still fails.\n\nMy main objection is to how the conclusion is worded. \"The signal makes money before costs, and high turnover costs eat it up\" is overstated. The 20-slot portfolio before costs in validation is 18.2% a year, below 0050's 28.3% and only level with equal weight at 16.8% (t 0.57). So even at zero cost it would not pass in the validation period. The gross edge shrinks steadily over time (per-trade 1.02% → 0.69% → 0.49% → 0.45%), and net abnormal return has been negative every year since 2016.\n\nThe selection-period pass at the event level (t 2.21) comes from averaging the two units on add-on trades. Weighted per unit it is −0.98, and the standard version is −1.81, so no reading of the selection period clearly passes.\n\nSuggested rewording: \"It has a decaying gross edge at the trade level, but no edge at the portfolio level even before costs. After costs it is not usable at all.\"\n\nFiles reviewed: /home/user/-/research/tw_stock_selection/agents/w12_connors_mdd/run.py, prereg.py, results.json, diag_agg_unit.py, run_log.txt"
}