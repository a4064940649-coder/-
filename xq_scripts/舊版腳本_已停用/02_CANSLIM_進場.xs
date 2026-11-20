// ===== 02 CAN SLIM 台股版（選股腳本，進場）=====
// 來源：公開貼文（Fred Richards〈Growth Investing with CAN SLIM〉的台股改編），條件與預設值照原文，未改。
// C：單季EPS年增 >= 18%
// A：年度EPS三年複合成長 >= 15%、近3月營收年增平均 >= 20%
// N：收盤距240日新高 10% 以內
// S：20日均量 >= 500 張
// L：120日漲幅贏加權指數，且為正
// M：加權指數在60日均線之上才選股
// 用法：選股腳本；日線盤後執行；符合就隔天開盤買。出場請搭配 03（寬出場）。
Input: MinEpsQYoY(18), MinEpsCAGR(15), MinRevYoY3(20), MaxFromHigh(10),
       MinAvgQty(500), RSDays(120), UseMarket(1), MarketMA(60);
Var: MissingValue(-999999999), EpsQ0(0), EpsQ4(0), EpsQYoY(-999),
     EpsY0(0), EpsY3(0), EpsCAGR(-999), RevM(0), RevYoY3(0), RevOK(true), Idx(0),
     High240(0), FromHigh(-999), AvgQty(0), StockRet(-999), IndexRet(-999),
     IdxClose(0), IdxCloseOld(0), IdxMA(0), MarketOK(true);
SetTotalBar(300);
SetBackBar(300);
SetBackBar(8, "Y");
SetBackBar(8, "Q");
SetBackBar(6, "M");
if BarFreq <> "D" and BarFreq <> "AD" then RaiseRunTimeError("請使用日線盤後選股");
Ret = 0;
// C：單季EPS年增
EpsQ0 = GetField("每股稅後淨利(元)", "Q", Default:=-999999999);
EpsQ4 = GetField("每股稅後淨利(元)", "Q", Default:=-999999999)[4];
EpsQYoY = -999;
if EpsQ0 <> MissingValue and EpsQ4 <> MissingValue and EpsQ4 > 0 then
    EpsQYoY = (EpsQ0 / EpsQ4 - 1) * 100;
// A：年度EPS三年複合成長＋近3月營收年增
EpsY0 = GetField("每股稅後淨利(元)", "Y", Default:=-999999999);
EpsY3 = GetField("每股稅後淨利(元)", "Y", Default:=-999999999)[3];
EpsCAGR = -999;
if EpsY0 > 0 and EpsY3 > 0 then
    EpsCAGR = (Power(EpsY0 / EpsY3, 1 / 3) - 1) * 100;
RevYoY3 = 0; RevOK = true;
for Idx = 0 to 2 begin
    RevM = GetField("月營收年增率", "M", Default:=-999999999)[Idx];
    if RevM = MissingValue then RevOK = false else RevYoY3 = RevYoY3 + RevM / 3;
end;
if RevOK = false then RevYoY3 = -999;
// N：接近240日新高
High240 = Highest(High, 240);
FromHigh = -999;
if High240 > 0 and CurrentBar >= 240 then FromHigh = (Close / High240 - 1) * 100;
// S：流動性
AvgQty = Average(Volume, 20);
// L：相對強勢（個股 vs 加權指數）
StockRet = -999;
if Close[RSDays] > 0 then StockRet = (Close / Close[RSDays] - 1) * 100;
IdxClose = GetSymbolField("TSE.TW", "收盤價");
IdxCloseOld = GetSymbolField("TSE.TW", "收盤價")[RSDays];
IndexRet = -999;
if IdxCloseOld > 0 then IndexRet = (IdxClose / IdxCloseOld - 1) * 100;
// M：大盤方向
IdxMA = Average(GetSymbolField("TSE.TW", "收盤價"), MarketMA);
MarketOK = true;
if UseMarket = 1 and (IdxMA <= 0 or IdxClose < IdxMA) then MarketOK = false;
if Close > 0 and MarketOK
   and EpsQYoY >= MinEpsQYoY
   and EpsCAGR >= MinEpsCAGR
   and RevYoY3 >= MinRevYoY3
   and FromHigh >= -MaxFromHigh
   and AvgQty >= MinAvgQty
   and StockRet > IndexRet and StockRet > 0 then Ret = 1;
OutputField(1, EpsQYoY, 1, "單季EPS年增%");
OutputField(2, EpsCAGR, 1, "EPS三年複合成長%");
OutputField(3, RevYoY3, 1, "近3月營收年增%");
OutputField(4, FromHigh, 1, "距240日高%");
OutputField(5, StockRet - IndexRet, 1, "超越大盤%");
OutputField(6, AvgQty, 0, "20日均量");
