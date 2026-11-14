// ===== 04 CAN SLIM 分數版（選股腳本，每月換股、固定檔數）v1.0 =====
// 做法參考 FinLab「排名法」：不用門檻一刀切，改成把五個條件各自換成 0～100 分，平均成綜合分數，
// 每月依綜合分數由大到小取前 20 檔，平均買進，抱到下個月。這樣不會忽多忽少、不會空手。
// 五個條件：C 單季EPS年增、A1 年度EPS三年複合成長、A2 近3月營收年增、N 距240日高、L 超越大盤。
// 注意：真正的排名法是「全市場百分位」；XS 單一腳本做不到，這裡用固定上下限換算成分數，是近似版。
// 用法：選股腳本；日線盤後執行；每月換股一次；排序欄位＝輸出欄位 1「綜合分數」，取前 20 檔。未在 XQ 編譯或回測。
Input: MinAvgQty(500), RSDays(120), UseMarket(1), MarketMA(60), MinScore(0);
Var: MissingValue(-999999999), EpsQ0(0), EpsQ4(0), EpsQYoY(-999),
     EpsY0(0), EpsY3(0), EpsCAGR(-999), RevM(0), RevYoY3(0), RevOK(true), Idx(0),
     High240(0), FromHigh(-999), AvgQty(0), StockRet(-999), IndexRet(-999), RS(-999),
     IdxClose(0), IdxCloseOld(0), IdxMA(0), MarketOK(true), DataOK(true),
     ScC(0), ScA1(0), ScA2(0), ScN(0), ScL(0), Score(0);
SetTotalBar(300);
SetBackBar(300);
SetBackBar(8, "Y");
SetBackBar(8, "Q");
SetBackBar(6, "M");
if BarFreq <> "D" and BarFreq <> "AD" then RaiseRunTimeError("請使用日線盤後選股");
Ret = 0;
DataOK = true;
// C：單季EPS年增（去年同季 <= 0 不算）
EpsQ0 = GetField("每股稅後淨利(元)", "Q", Default:=-999999999);
EpsQ4 = GetField("每股稅後淨利(元)", "Q", Default:=-999999999)[4];
EpsQYoY = -999;
if EpsQ0 <> MissingValue and EpsQ4 <> MissingValue and EpsQ4 > 0 then
    EpsQYoY = (EpsQ0 / EpsQ4 - 1) * 100
else DataOK = false;
// A1：年度EPS三年複合成長（兩年都要 > 0）
EpsY0 = GetField("每股稅後淨利(元)", "Y", Default:=-999999999);
EpsY3 = GetField("每股稅後淨利(元)", "Y", Default:=-999999999)[3];
EpsCAGR = -999;
if EpsY0 > 0 and EpsY3 > 0 then
    EpsCAGR = (Power(EpsY0 / EpsY3, 1 / 3) - 1) * 100
else DataOK = false;
// A2：近3月營收年增平均
RevYoY3 = 0; RevOK = true;
for Idx = 0 to 2 begin
    RevM = GetField("月營收年增率", "M", Default:=-999999999)[Idx];
    if RevM = MissingValue then RevOK = false else RevYoY3 = RevYoY3 + RevM / 3;
end;
if RevOK = false then begin RevYoY3 = -999; DataOK = false; end;
// N：距240日高
High240 = Highest(High, 240);
FromHigh = -999;
if High240 > 0 and CurrentBar >= 240 then FromHigh = (Close / High240 - 1) * 100
else DataOK = false;
// S：流動性
AvgQty = Average(Volume, 20);
// L：超越大盤
StockRet = -999;
if Close[RSDays] > 0 then StockRet = (Close / Close[RSDays] - 1) * 100;
IdxClose = GetSymbolField("TSE.TW", "收盤價");
IdxCloseOld = GetSymbolField("TSE.TW", "收盤價")[RSDays];
IndexRet = -999;
if IdxCloseOld > 0 then IndexRet = (IdxClose / IdxCloseOld - 1) * 100;
RS = -999;
if StockRet <> -999 and IndexRet <> -999 then RS = StockRet - IndexRet
else DataOK = false;
// M：大盤方向（只擋新進場）
IdxMA = Average(GetSymbolField("TSE.TW", "收盤價"), MarketMA);
MarketOK = true;
if UseMarket = 1 and (IdxMA <= 0 or IdxClose < IdxMA) then MarketOK = false;
// 換算成 0～100 分（超出上下限就壓在邊界）
// C：-50% → 0 分，+150% → 100 分
ScC = (EpsQYoY + 50) / 2;
if ScC < 0 then ScC = 0;
if ScC > 100 then ScC = 100;
// A1：-20% → 0 分，+60% → 100 分
ScA1 = (EpsCAGR + 20) / 0.8;
if ScA1 < 0 then ScA1 = 0;
if ScA1 > 100 then ScA1 = 100;
// A2：-30% → 0 分，+100% → 100 分
ScA2 = (RevYoY3 + 30) / 1.3;
if ScA2 < 0 then ScA2 = 0;
if ScA2 > 100 then ScA2 = 100;
// N：離高點 -40% → 0 分，0% → 100 分
ScN = (FromHigh + 40) / 0.4;
if ScN < 0 then ScN = 0;
if ScN > 100 then ScN = 100;
// L：-50% → 0 分，+100% → 100 分
ScL = (RS + 50) / 1.5;
if ScL < 0 then ScL = 0;
if ScL > 100 then ScL = 100;
Score = (ScC + ScA1 + ScA2 + ScN + ScL) / 5;
if DataOK and MarketOK and Close > 0 and AvgQty >= MinAvgQty and Score >= MinScore then Ret = 1;
OutputField(1, Score, 1, "綜合分數");
OutputField(2, EpsQYoY, 1, "單季EPS年增%");
OutputField(3, EpsCAGR, 1, "EPS三年複合成長%");
OutputField(4, RevYoY3, 1, "近3月營收年增%");
OutputField(5, FromHigh, 1, "距240日高%");
OutputField(6, RS, 1, "超越大盤%");
