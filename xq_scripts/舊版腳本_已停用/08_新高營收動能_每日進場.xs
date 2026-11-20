// ===== 08 新高＋營收動能（選股腳本，每日進場版）v1.0 =====
// 01 的條件改成每天都可以進場，出場搭配 03（寬出場），不用等每月換股日。
// 這是新組合，研究中心沒有回測過，請自行回測後再決定。
// 用法：選股腳本；日線（建議還原日線）盤後執行；符合就隔天開盤買；同時最多 10 檔；
//       訊號多於空位時依輸出欄位 2「近3月營收年增%」由大到小。
Input: MinHighRatio(95), MinRevYoY3(30), MinAvgValue(10000), MinPrice(10),
       HighDays(252), LiqDays(60), UseMarket(1), MarketMA(60);
Var: MissingValue(-999999999), RevM(0), RevYoY3(0), RevOK(true), Idx(0),
     HighC(0), HighRatio(0), AvgValue(0), IdxClose(0), IdxMA(0), MarketOK(true);
SetTotalBar(300);
SetBackBar(300);
SetBackBar(6, "M");
if BarFreq <> "D" and BarFreq <> "AD" then RaiseRunTimeError("請使用日線盤後選股");
Ret = 0;
RevYoY3 = 0; RevOK = true;
for Idx = 0 to 2 begin
    RevM = GetField("月營收年增率", "M", Default:=-999999999)[Idx];
    if RevM = MissingValue then RevOK = false else RevYoY3 = RevYoY3 + RevM / 3;
end;
if RevOK = false then RevYoY3 = -999;
HighC = Highest(Close, HighDays);
HighRatio = 0;
if HighC > 0 and CurrentBar >= HighDays then HighRatio = Close / HighC * 100;
AvgValue = Average(Close * Volume, LiqDays) / 10;
IdxClose = GetSymbolField("TSE.TW", "收盤價");
IdxMA = Average(GetSymbolField("TSE.TW", "收盤價"), MarketMA);
MarketOK = true;
if UseMarket = 1 and (IdxMA <= 0 or IdxClose < IdxMA) then MarketOK = false;
if MarketOK and Close >= MinPrice
   and AvgValue >= MinAvgValue
   and HighRatio >= MinHighRatio
   and RevYoY3 >= MinRevYoY3 then Ret = 1;
OutputField(1, HighRatio, 1, "距52週高%");
OutputField(2, RevYoY3, 1, "近3月營收年增%");
OutputField(3, AvgValue, 0, "60日均成交值(萬)");
