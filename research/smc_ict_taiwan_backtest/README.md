# SMC / ICT 與傳統支撐壓力「機械化日線代理規則」回測工具（台股版）

`backtest_smc_taiwan.py` 是一支**單檔、可離線執行**的 Python 回測腳本，讓台灣散戶在自己的電腦上，
用台股日線資料與**貼近實際的台灣交易成本**，檢驗「Smart Money Concepts (SMC) / ICT」常見概念
（流動性掃蕩、公平價值缺口 FVG、訂單塊 Order Block）以及傳統支撐壓力（反彈、突破）的**機械化代理規則**是否有統計優勢。

> **重要聲明**
> 1. SMC/ICT 在實務上多半是**盤中、裁量式**的操作，本工具測的是「寫得出來、能重複驗證」的**日線機械化代理版本**，
>    結果只能回答「這些可機械化的版本在台股日線上有沒有優勢」，無法證明或推翻裁量式交易者的做法。
> 2. 本工具**不會替你下載任何結論**：它輸出交易明細與統計量，請自行判讀。所有數字都取決於你抓到的資料與你設定的成本。
> 3. 預設成本（手續費折數、交易稅、期貨手續費）**請務必依你的券商與現行法規自行核對**。
> 4. `--smoke-test` 模式使用程式內產生的隨機漫步資料，只用來確認程式能跑完，**其輸出不是任何真實市場的結果**。

---

## 1. 安裝

需求：Python 3.10 以上（已在 3.13 測試）、pandas 2.x 或 3.x、numpy。`yfinance` 只在使用預設資料來源時需要。

```bash
cd research/smc_ict_taiwan_backtest
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

先確認程式本身能跑（純合成資料，數字無意義）：

```bash
python backtest_smc_taiwan.py --smoke-test
```

畫面會印出 `SMOKE TEST — SYNTHETIC RANDOM-WALK DATA — NOT REAL RESULTS` 的警示框，並把輸出寫到 `./output_smoke/`。

---

## 2. 執行

### 2.1 預設：Yahoo Finance（yfinance）

```bash
python backtest_smc_taiwan.py
```

預設標的：`2330.TW 2317.TW 2454.TW 2308.TW 2382.TW 2412.TW 2881.TW 2882.TW 3711.TW 2303.TW 0050.TW ^TWII`，
期間 `2015-01-01` 到今天。下載的原始資料會快取到 `./data/<代碼>.csv`（加上 `.meta.json` 記錄抓取範圍），
下次執行若範圍相同就直接讀快取；加 `--refresh` 強制重抓。

常用參數：

```bash
python backtest_smc_taiwan.py --tickers 2330.TW 2317.TW ^TWII --start 2018-01-01 --end 2025-12-31
python backtest_smc_taiwan.py --quick                       # 只跑前 4 檔、200 次 bootstrap、50 次洗牌（快速檢查）
python backtest_smc_taiwan.py --r1-entry next_open          # R1 改為次日開盤進場（最保守、無收盤前瞻）
python backtest_smc_taiwan.py --instrument futures          # 主要「淨報酬」欄位改用期貨成本（適合 ^TWII）
python backtest_smc_taiwan.py --comm-discount 0.28 --slippage-ticks 1   # 2.8 折手續費、每邊 1 檔滑價
python backtest_smc_taiwan.py --sides long                  # 只測多單
```

### 2.2 FinMind（備援來源）

```bash
python backtest_smc_taiwan.py --source finmind --finmind-token <你的token>
# 或 export FINMIND_TOKEN=...   （免 token 也能用，但有流量限制）
```

使用 FinMind API v4 的 `TaiwanStockPrice` 資料集；代碼自動轉換（`2330.TW` → `2330`，`^TWII` → `TAIEX`，
可用 `--finmind-index-id` 改）。

### 2.3 自備 CSV

```bash
python backtest_smc_taiwan.py --source csv --csv-dir ./my_csv --tickers 2330.TW ^TWII
```

每個標的一個檔案，檔名 `<代碼>.csv`（`^TWII` 請存成 `IDX_TWII.csv`），欄位 `date,open,high,low,close,volume`
（大小寫不拘；也接受 FinMind 的 `max/min/Trading_Volume`，以及可選的 `adj_close`）。

### 2.4 資料處理

* **除息還原**：若來源提供 `adj_close`（yfinance 有），預設把 OHLC 乘上 `adj_close/close`，避免除息跳空被誤判成缺口或掃蕩；`--no-adjust` 可關閉。
* **資料驗證**：刪除 `low > min(open,close)` 或 `high < max(open,close)`、非正價、缺值的 K 棒；回報連續交易日間隔 > 10 天的斷點、單日漲跌幅 > 10.5% 的筆數（可能是拆股或壞資料）、零成交量筆數。結果在 `data_quality.csv` 與摘要檔。
* 任一標的下載失敗只會跳過該標的，不會中斷整個回測。

---

## 3. 成本模型（可用參數覆寫，預設值請自行核對）

| 模式 `cost_mode` | 進場邊 | 出場邊 | 預設往返成本（不含滑價） |
|---|---|---|---|
| `gross` | 0 | 0 | 0 |
| `cash_overnight`（現股留倉）| 手續費 0.1425% × 折數 | 手續費 0.1425% × 折數 + 證交稅 0.30% | 6 折：0.0855%×2 + 0.30% = **0.471%** |
| `cash_daytrade`（現股當沖）| 同上 | 手續費 + 證交稅 **0.15%** | 6 折：**0.321%** |
| `futures`（期貨）| 期交稅 0.002% × 名目 + 每口手續費 | 同左 | 小台 @20000 點：每邊 ≈ 0.007%，往返 ≈ **0.014%** |

* 滑價：`--slippage-bps`（預設每邊 5 bp = 0.05%）或 `--slippage-ticks N`（依台股升降單位：<10 元 0.01、10–50 元 0.05、50–100 元 0.1、100–500 元 0.5、500–1000 元 1、≥1000 元 5；ETF <50 元 0.01、≥50 元 0.05；指數 1 點）。所有淨報酬模式都會套用滑價。
* 現股放空（`cash_overnight` 的 short）額外扣 `--short-borrow-bps`（預設 8 bp）當作借券費；**沒有**模擬平盤下不得放空、券源不足、強制回補等限制，空單結果只能當參考。
* 期貨名目 = 價格 × 乘數：指數用 50（小台 MTX），個股用 2000（股票期貨一口 = 2 張）；`--fut-multiplier` 可覆寫。`--fut-comm` 為每口每邊手續費（預設 NT$50）。**未考慮**保證金、基差、轉倉、以及並非每檔股票都有股票期貨。
* `cash_daytrade` 只是「如果稅率是 0.15% 會怎樣」的**假設情境**：日線策略必然留倉，實際上拿不到當沖稅率，請只當作成本下限參考。
* 淨報酬計算為乘法形式：多單 `淨報酬 = 出場價×(1−滑價)×(1−賣出成本) / [進場價×(1+滑價)×(1+買入成本)] − 1`；空單對稱處理。

執行後 `cost_table.csv` 會列出每檔標的在其中位數價格下、各模式的實際往返成本（含滑價）。

---

## 4. 規則說明（皆為日線；多空皆測並分開報告）

所有訊號只用到第 `t` 根（含）以前的 K 棒，**沒有前瞻偏誤**。進出場時點規則：

| 進場型態 `entry_kind` | 用於 | 說明 |
|---|---|---|
| `close` | R1（預設）| 訊號棒收盤價進場。假設在 13:25–13:30 集合競價以試撮價下單，屬輕微樂觀假設；`--r1-entry next_open` 可改為次日開盤。 |
| `open` | R4a、R4b、R1(`next_open`) | 訊號棒的**下一根開盤價**進場。 |
| `limit` | R2、R3 | 在價位掛限價單，之後第一根「觸及該價位」的 K 棒成交（成交價 = 掛單價，或若該棒開盤就跳過價位則以開盤價成交）。若該棒**開盤已越過停損價**，視為區域失效、撤單不進場。 |

出場：逐棒檢查停損 / 目標；若同一根 K 棒同時涵蓋停損與目標，**一律當作先打停損**（保守）；跳空穿越停損或目標則以開盤價出場；限價進場的那一根只檢查停損（高點可能發生在成交之前，不給目標）。時間出場在持有第 N 根 K 棒的收盤。

### R1 流動性掃蕩反轉（ICT「turtle soup」/ 前日低點掃蕩）
* 多：`low[t] < low[t-1]` 且 `close[t] > low[t-1]`（`--r1-strict` 再要求 `close > open`）。
* 進場：訊號棒收盤（預設）或次日開盤；停損 = `low[t] × (1 − buffer)`（`--stop-buffer-bps` 預設 10 bp）；目標 = 進場價 + `--target-r`（預設 2）× 風險；`--hold`（預設 3）根後時間出場。
* 空單對稱（`high[t] > high[t-1]` 且 `close[t] < high[t-1]`）。
* 測的是：「假突破前日低點後收回」是否有短線反轉優勢。

### R2 公平價值缺口 FVG（三根 K 棒失衡）
* 多方 FVG：`high[t-2] < low[t]`，缺口區 = `[high[t-2], low[t]]`（`--fvg-min-gap-pct` 可過濾太小的缺口）。
* (a) **描述統計**：缺口形成後 5/10/20 根內，價格「回到缺口內」(touch) 與「穿越整個缺口」(fill) 的比率，並與**洗牌 K 棒**虛無分布比較（保留每根 K 棒相對前收的形狀、打亂順序，消除時間序列記憶）。寫在 `fvg_fill_rates.csv`。
* (b) **交易規則**：缺口形成後在「靠近現價的那一邊」（多方 = `low[t]`）掛限價買單，有效 `--fvg-wait`（預設 20）根；停損 = 缺口遠端 `high[t-2] × (1 − buffer)`；目標 2R；`--level-hold`（預設 10）根後時間出場。
* 測的是：「價格回補缺口後會不會從缺口邊緣反轉」。

### R3 訂單塊 Order Block 回測
* 多方 OB：第 `t` 根創 `--lookback`（20）根新高時，往前 `--ob-lookback`（10）根內找**最後一根陰線** `b`，OB 區 = `[low[b], high[b]]`，且要求 `close[t] > high[b]`（價格已離開 OB）。每根 OB 只用一次。
* 在 `high[b]` 掛限價買單，有效 `--ob-wait`（20）根；停損 = `low[b] × (1 − buffer)`；目標 2R；`--level-hold` 根後時間出場。空單對稱（創新低前的最後一根陽線）。
* 測的是：「推動波起點的最後反向 K 棒，回測時是否成為支撐/壓力」。

### R4 傳統支撐壓力
* 水準 = 滾動 20 根（不含當根）的最低 / 最高價。
* (a) **反彈**：第 `t-1` 根收盤落在支撐上方 `--sr-tol-pct`（0.5%）內，第 `t` 根反轉向上（`close[t] > close[t-1]` 且 `close[t] > open[t]`）→ 第 `t+1` 根開盤買進；停損在支撐下方；目標 2R；`--level-hold` 根後時間出場。
* (b) **突破**：`close[t] >` 前 20 根最高 → 第 `t+1` 根開盤買進，持有 `--hold` 根，無停損；R 單位用 ATR(14)/進場價。
* 摘要檔有 R4a 與 R4b 的並列比較。

### R5 基準
* **買進持有**：每檔的總報酬、CAGR、年化波動、日報酬 Sharpe、最大回落；以及「無條件持有 N 根」的平均報酬與勝率（N = `--hold`、`--level-hold`），寫在 `baselines_buy_and_hold.csv`。
* **隨機進場 bootstrap**：對每個「規則 × 方向 × 標的（及 POOLED）」，抽同樣筆數的隨機進場，**同標的、同方向、同進場型態、持有期從該規則實際持有期重抽**，重複 `--n-boot`（1000）次，得到「隨機進場平均（毛）報酬」的分布；`p_boot` = 隨機平均 ≥ 規則平均的比例（單尾）。`rand_mean_pct` 為該虛無分布的平均。

### 「水準磁吸」描述統計（level magnet）
下一個交易日觸及前日最高 / 最低價的頻率（`level_magnet.csv`），對照 (a) 洗牌 K 棒虛無分布（平均、標準差、z 值）與 (b) 零漂移布朗運動盤中路徑（54 個 5 分鐘步、隔夜跳空與盤中波動度皆配對實際值）。**純描述**：隨機漫步本來就會頻繁觸及前日高低點，這個統計只是讓你看「比隨機多多少」。

### 其他共通設定
* 預設**不重疊**：同一規則、同方向、同標的一次只有一筆部位或一張掛單（`--allow-overlap` 可開）。這讓交易報酬較接近獨立樣本，t 值與 Sharpe 才有意義。
* `--min-risk-pct`（預設 0.2%）：停損距離小於此比例的訊號跳過（避免極小風險造成荒謬的 R 倍數與槓桿）。被跳過的訊號與原因列在 `skipped_signals.csv`。

---

## 5. 輸出與解讀

輸出目錄預設 `./output/`（`--out-dir` 可改）：

| 檔案 | 內容 |
|---|---|
| `results_summary.md` | 人看的總結：參數、各成本模式的 POOLED 表、R4a vs R4b、每檔明細、基準、FVG 回補率、磁吸統計、資料品質、被跳過訊號、限制。 |
| `trades_all.csv` | **每一筆交易**：標的、規則、方向、訊號/進場/出場日期與索引、進出場價、停損、目標、`risk_frac`、出場原因（stop/target/time/end_of_data）、毛報酬與三種淨報酬、各自的 R 倍數。可自行用 Excel 檢查。 |
| `metrics_long.csv` | 長表：規則 × 方向 × 標的（含 `POOLED`）× 成本模式 的全部統計量。 |
| `baselines_buy_and_hold.csv` | R5 買進持有與無條件 N 根報酬。 |
| `fvg_fill_rates.csv` | R2 描述統計（含 `_null` 洗牌虛無值）。 |
| `level_magnet.csv` | 前日高低點觸及率與兩種虛無值。 |
| `cost_table.csv` | 每檔在中位數價格下的往返成本。 |
| `data_quality.csv`、`skipped_signals.csv`、`run_config.json` | 資料品質、被跳過訊號、本次參數。 |

### 統計量定義

| 欄位 | 定義 / 假設 |
|---|---|
| `n_trades` | 交易筆數。少於 ~30 筆的結果基本上沒有統計意義。 |
| `win_rate` / `win_pct` | 報酬 > 0 的比例（毛或淨，依 `cost_mode`）。有 2R 目標的規則，勝率 33% 左右才打平；請與 `expectancy_R` 一起看。 |
| `avg_ret_pct`、`median_ret_pct` | 每筆報酬平均 / 中位數（%）。 |
| `expectancy_R` | 平均「報酬 ÷ 每筆風險」。風險 = \|進場價 − 停損\| / 進場價；R4b 無停損，用 ATR(14)/進場價當風險單位。>0 才有正期望。 |
| `profit_factor` | 總獲利 ÷ 總虧損。 |
| `sharpe_ann` | 每筆報酬平均 ÷ 標準差 × √(每年交易筆數)。**假設**交易報酬獨立同分布、不重疊、空手期間資金閒置、無風險利率 0。不能直接與基金年化 Sharpe 比較。 |
| `max_dd_pct` | 等風險部位（每筆冒 `--risk-per-trade`，預設 1% 權益；損益 = 1% × R）依出場日順序複利計算的最大回落。POOLED 表是把所有標的的交易串成一條權益曲線。 |
| `t_stat` | 平均報酬 ÷ (標準差/√n)；|t| > 2 大約對應 5% 顯著水準（前提是交易獨立）。 |
| `p_boot`、`rand_mean_pct`、`rand_sd_pct` | R5 隨機進場 bootstrap 的單尾 p 值與虛無分布平均/標準差（毛報酬基準；成本對規則與隨機進場是對稱的，所以結論相同）。 |
| `pct_stop`、`pct_target`、`pct_time` | 出場原因比例。若 `pct_stop` 很高，表示停損設在「水準正下方」常被掃到——這本身就是對「水準會被精準尊重」的檢驗。 |

### 怎麼判斷「有優勢」

一個規則要稍微值得相信，至少要同時滿足：**淨報酬模式**（`cash_overnight` 或 `futures`）下 `avg_ret_pct > 0`、`expectancy_R > 0`、`profit_factor > 1`，`p_boot` 小（例如 < 0.05），`t_stat` > 2，交易筆數夠多，而且**在多數標的上方向一致**（看每檔明細表），並且在 `--r1-entry next_open` 這類更保守設定下仍成立。
注意本工具一次測試 5 條規則 × 2 方向 × 12 檔 ≈ 120 個組合，**即使完全沒有優勢，也預期會出現幾個 p < 0.05**（多重比較）。

---

## 6. 限制

1. **日線代理 ≠ SMC/ICT 原始做法**：原方法多在盤中、搭配市場結構與時段（kill zone）等裁量判斷，這裡的 R1–R3 只是最常見、能寫成規則的版本。
2. **日線 OHLC 看不到盤中路徑**：停損/目標先後順序採保守假設（先停損）；限價成交那根不給目標。這會系統性地讓結果偏保守。
3. **成交假設**：收盤價進場（R1 預設）假設集合競價可以成交在收盤價；限價單假設一觸價即成交（實際可能排隊未成交）。
4. **放空**：現股空單未模擬台灣的融券限制（券源、平盤下不得放空、強制回補）、也沒有股票期貨的流動性問題；空單結果主要適用於 `^TWII` 的期貨成本模式。
5. **期貨模式**簡化：不含保證金、基差、轉倉、漲跌停限制；個股期貨乘數一律用 2000。
6. **資料**：yfinance 的台股資料偶有缺漏或錯價；本工具會刪掉不合理 K 棒並報告，但無法修正來源錯誤。除息還原依賴來源的 `adj_close`。
7. **存活者偏差 / 標的選擇**：預設標的是目前的大型權值股，回測 2015 年起必然帶有「它們後來變大」的偏誤；請自行補入其他標的。
8. **多重比較**：見上節。
9. **非投資建議**：本工具只做統計描述，不構成任何投資建議。

---

## 7. 參數總表

執行 `python backtest_smc_taiwan.py --help` 可看全部參數與預設值。重點整理：

| 類別 | 參數 | 預設 | 說明 |
|---|---|---|---|
| 資料 | `--tickers`、`--source {yfinance,finmind,csv}`、`--start`、`--end`、`--data-dir`、`--csv-dir`、`--finmind-token`、`--refresh`、`--no-adjust`、`--out-dir` | 見上文 | 資料來源與快取 |
| 規則 | `--sides`、`--hold 3`、`--level-hold 10`、`--target-r 2`、`--stop-buffer-bps 10`、`--min-risk-pct 0.2`、`--lookback 20`、`--sr-tol-pct 0.5`、`--fvg-wait 20`、`--fvg-min-gap-pct 0`、`--ob-lookback 10`、`--ob-wait 20`、`--r1-entry close`、`--r1-strict`、`--allow-overlap` | | 規則參數 |
| 成本 | `--instrument {cash,futures}`、`--comm-rate 0.1425`、`--comm-discount 0.6`、`--tax-rate 0.3`、`--daytrade-tax-rate 0.15`、`--short-borrow-bps 8`、`--fut-tax-rate 0.002`、`--fut-comm 50`、`--fut-multiplier`、`--slippage-bps 5`、`--slippage-ticks 0`、`--risk-per-trade 1` | | 單位為 %（bps 除外） |
| 統計 | `--n-boot 1000`、`--n-shuffle 200`、`--seed 42`、`--quick`、`--smoke-test`、`--log-level` | | 固定 seed 可完全重現 |

---

## 8. 檔案

```
research/smc_ict_taiwan_backtest/
├── backtest_smc_taiwan.py   # 主程式（CLI）
├── README.md                # 本說明
└── requirements.txt
```
