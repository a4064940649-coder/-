// ===== S15 強勢股拉回買進（選股腳本，進場）草稿 v0.1 =====
// ★ 研究中心判定：不通過。單獨做衛星依事前規則「輸 0050」；和現行 SURGE 各半合併「沒有補充價值」。不建議加進現行策略。
//   單獨（0050 底倉 30%＋衛星 70%）：20 條路徑 2019 年起中位數 23.34%，0050 28.91%，0／20 條較高；八項檢查 0／8。
//   一半現行 SURGE＋一半拉回：對「一半 SURGE＋一半 0050」0／20 條較好（中位數少 1.14 個百分點）；2019～2023 每條都多 2.00，2024 年起每條都少 7.40。
//   這支只給 XQ 交叉驗證研究中心的結果用，不要拿真錢跑。細節見 research/tw_stock_selection/agents/s15_strong_pullback/FINDINGS.md。
// 規則（研究中心 P 格，事前登記 PREREG.json sha256 8aff4e37…）：
//   1. 股池：上市、非創新板（股名不帶 -創）、近 60 日平均成交值 >= 1,000 萬、收盤 >= 10。
//   2. 強勢：近 120 日漲幅 − 0050 近 120 日漲幅（都含息還原），在股池內排前 20%。
//   3. 接近新高：收盤 >= 近 240 日最高價 × 0.85（距 52 週高 <= 15%）。
//   4. 拉回：昨天收盤在昨天 20 日均線的 0.97～1.02 倍。
//   5. 量縮：昨天的 5 日均量 < 昨天的 20 日均量。
//   6. 轉強：今天收盤 > 昨天最高價。
//   7. 今天盤後選出，明天開盤買；開盤就漲停不買。同一天多於空位時，「超越0050%」大的先買。
//   8. 固定持有 20 個交易日，第 20 個交易日的開盤賣（不設停損、不看均線）。持有中同一檔再出現訊號不加碼。
//   9. 資金：0050 底倉 30%、衛星 70%、每檔＝衛星 ÷ 10、最多 10 檔，空位放 0050；買不到目標一半不買。
// 研究中心回測（官方日 K、只上市、排除創新板、100 萬、嚴成本：手續費 0.1425%×0.28、賣出稅 0.3%、每邊半個跳動點＋0.3% 滑價）：
//   P：2019 年起 19.98%、2015-07 起 13.32%、2024 年起 21.04%、最大回撤 −38.06%（2018-01 → 2020-03）。
//   同期 0050：28.91%／21.14%／54.48%／−33.96%。每年單邊周轉約 8.1 倍，成本約每年淨值的 10.1%；衛星 86% 的日子 10 檔全滿。
//   扣成本前，訊號後持有 20 日對 0050 平均 −0.29%（中位數 −2.19%），買進隔天收盤就平均落後 0050 0.44%。
// XQ 回測時只能測衛星那一塊：起始資金設衛星資金（例如 70 萬）、最多 10 檔、每檔＝起始資金 ÷ 10、空位是現金（研究中心是 0050，會比較低）。
// 未在 XQ 編譯，也未在 XQ 回測。以下需在 XQ 確認：
//   (1) 「股池內前 20%」XS 單檔腳本算不出橫斷面排名：請在選股中心用「輸出欄位 1 排序、取前 20%」（需在 XQ 確認是否支援百分比取前）；
//       做不到時把 MinRS 設成一個固定門檻（例如 20，代表比 0050 多漲 20 個百分點）近似，結果會和研究中心不同；
//   (2) 0050 的 120 日漲幅要用含息還原價：GetSymbolField("0050.TW", "收盤價") 在「還原日線」下是否已還原除息與 2025-06 分割（需在 XQ 確認）；
//   (3) 「固定持有 20 個交易日」：XQ 回測的出場設定是否有「持有 N 根 K 棒後出場」，以及是用開盤還是收盤成交（需在 XQ 確認）；研究中心是第 20 個交易日開盤賣；
//   (4) 成交量單位：XQ 日線 Volume 是張，成交值＝收盤 × 張（千元），÷ 10 換成萬元（同 xq_scripts 20 號的寫法）。
Input: RSDays(120), MinRS(-999), NearHigh(85), PBLow(97), PBHigh(102), MinAvgValue(1000), MinPrice(10), LiqDays(60);
// MinRS：超越 0050 的百分點下限（預設 -999＝不設，改由選股中心排序取前 20%）；NearHigh：收盤 ÷ 240 日高 × 100 下限；PBLow／PBHigh：昨收 ÷ 昨 MA20 × 100 的區間
Var: High240(0), StockRet(-999), EtfRet(-999), RS(-999), MA20Prev(0), Ratio(0), V5(0), V20(0), AvgValue(0), EtfClose(0), EtfOld(0);
SetTotalBar(300);
SetBackBar(300);
if BarFreq <> "D" and BarFreq <> "AD" then RaiseRunTimeError("請使用日線（建議還原日線）盤後選股");
Ret = 0;
AvgValue = Average(Close * Volume, LiqDays) / 10;
High240 = Highest(High, 240);
StockRet = -999;
if Close[RSDays] > 0 then StockRet = (Close / Close[RSDays] - 1) * 100;
EtfClose = GetSymbolField("0050.TW", "收盤價");
EtfOld = GetSymbolField("0050.TW", "收盤價")[RSDays];
EtfRet = -999;
if EtfOld > 0 then EtfRet = (EtfClose / EtfOld - 1) * 100;
RS = -999;
if StockRet > -999 and EtfRet > -999 then RS = StockRet - EtfRet;
MA20Prev = Average(Close, 20)[1];
Ratio = 0;
if MA20Prev > 0 then Ratio = Close[1] / MA20Prev * 100;
V5 = Average(Volume, 5)[1];
V20 = Average(Volume, 20)[1];
if CurrentBar >= 240 and Close > 0 and High240 > 0
   and AvgValue >= MinAvgValue and Close >= MinPrice
   and RS > -999 and RS >= MinRS
   and Close / High240 * 100 >= NearHigh
   and Ratio >= PBLow and Ratio <= PBHigh
   and V5 < V20
   and Close > High[1] then Ret = 1;
OutputField(1, RS, 1, "超越0050%（取前20%、大的先買）");
OutputField(2, Close / High240 * 100 - 100, 1, "距240日高%");
OutputField(3, Ratio, 1, "昨收÷昨MA20×100");
OutputField(4, AvgValue, 0, "60日均成交值(萬)");
