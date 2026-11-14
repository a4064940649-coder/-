// ===== 05 強勢股池（選股腳本，進場）v1.0 =====
// 只看價格：接近 240 日新高、比大盤強、有成交量、大盤在 60 日線上。不看財報。
// 研究中心的發現：CAN SLIM 的財報條件拿掉後，報酬差不多，但回撤大很多（-48.8% 對 -25.2%）。
// 用法：選股腳本；日線盤後執行；符合就隔天開盤買；同時最多 10 檔；
//       訊號多於空位時依輸出欄位 1「超越大盤%」由大到小。出場搭配 03（寬出場）。未在 XQ 編譯或回測。
Input: MaxFromHigh(10), MinAvgQty(500), RSDays(120), UseMarket(1), MarketMA(60);
Var: High240(0), FromHigh(-999), AvgQty(0), StockRet(-999), IndexRet(-999),
     IdxClose(0), IdxCloseOld(0), IdxMA(0), MarketOK(true);
SetTotalBar(300);
SetBackBar(300);
if BarFreq <> "D" and BarFreq <> "AD" then RaiseRunTimeError("請使用日線盤後選股");
Ret = 0;
High240 = Highest(High, 240);
FromHigh = -999;
if High240 > 0 and CurrentBar >= 240 then FromHigh = (Close / High240 - 1) * 100;
AvgQty = Average(Volume, 20);
StockRet = -999;
if Close[RSDays] > 0 then StockRet = (Close / Close[RSDays] - 1) * 100;
IdxClose = GetSymbolField("TSE.TW", "收盤價");
IdxCloseOld = GetSymbolField("TSE.TW", "收盤價")[RSDays];
IndexRet = -999;
if IdxCloseOld > 0 then IndexRet = (IdxClose / IdxCloseOld - 1) * 100;
IdxMA = Average(GetSymbolField("TSE.TW", "收盤價"), MarketMA);
MarketOK = true;
if UseMarket = 1 and (IdxMA <= 0 or IdxClose < IdxMA) then MarketOK = false;
if Close > 0 and MarketOK
   and FromHigh >= -MaxFromHigh
   and AvgQty >= MinAvgQty
   and StockRet > IndexRet and StockRet > 0 then Ret = 1;
OutputField(1, StockRet - IndexRet, 1, "超越大盤%");
OutputField(2, FromHigh, 1, "距240日高%");
OutputField(3, AvgQty, 0, "20日均量");
