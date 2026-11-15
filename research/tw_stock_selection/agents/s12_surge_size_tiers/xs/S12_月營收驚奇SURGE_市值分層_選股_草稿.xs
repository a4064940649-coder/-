// ===== S12 月營收驚奇 SURGE＋站上季線＋已在漲，加「市值區間」條件（選股腳本，草稿）v0.1 =====
// 以「20_月營收驚奇SURGE_已在漲_選股_可調版.xs」為底，只多兩個參數 MinMktCap／MaxMktCap（總市值區間，億元）與一個輸出欄。
// ★ 研究中心 S12 的結論是「不建議改現行」：本草稿只供 XQ 交叉驗證與紙上追蹤，不要拿真錢跑。預設值（0／99999999）＝不設市值條件＝20 號腳本。
// 研究中心的分層是「在當天的股池內依 PIT 市值排名」：大型＝前 100 名、中型＝101～300 名、小型＝其餘。XQ 選股腳本不容易做橫斷面排名，
//   這裡用固定市值門檻近似；門檻每年要換（研究中心各年第 100 名、第 300 名市值的平均，億元）：
//   2016 323／63、2017 389／84、2018 393／84、2019 400／90、2020 463／107、2021 666／150、2022 664／141、2023 809／164、
//   2024 1,022／201、2025 960／189、2026 1,304／227。
//   例：只做大型股（S12 的 TL 格）在 2026 年設 MinMktCap＝1300；只做中型（TM）設 MinMktCap＝227、MaxMktCap＝1300。
// 研究中心 S12 回測（只上市、排除創新板、嚴成本、0050 底倉 30%＋衛星 70%、每檔＝衛星 ÷ 10；20 條路徑中位數，2019 起年化／最大回撤）：
//   現行（不分層）46.23%／−34.01%；只做大型 42.48%／−41.03%；只做中型 26.90%／−38.45%；只做小型 36.70%／−39.38%；
//   大 5 檔＋中小 5 檔 41.83%／−39.50%；0050 28.91%／−33.96%。
// 注意：分層後「前 10 名」「前 20 名」是在層內排名，所以 MinSurge（近似前 10 名）與 21 號的 ExitSurge（近似前 20 名）在層內的門檻會比全股池低，
//   需另行校準；用全股池的 2.5／2.0 會讓大型層幾乎買不到股票。
// 未在 XQ 編譯，也未在 XQ 回測。語法只用了 xq_scripts 既有腳本出現過的寫法，另加一個欄位。
// 需在 XQ 確認：(1) 欄位名「總市值」與單位（本草稿假設日頻、單位億元）；(2) 回測時「總市值」是不是當天的值（不能用今天的股本回推）；
//   (3) 其餘同 20 號腳本開頭的確認事項。
Input: MinRet20(10), MaxRet20(1000), RetLen(20), UseMA60(1), MALen(60), MinAvgValue(1000), MinPrice(10), LiqDays(60), OnlyRebalDay(1), RebalDay(11), MinSurge(0),
       MinMktCap(0), MaxMktCap(99999999);
// MinMktCap／MaxMktCap：總市值下限（含）／上限（不含），億元；0／99999999 = 不設（＝20 號腳本）
Var: MissingValue(-999999999), Idx(0), Idx2(0), RevA(0), RevB(0), RevOK(true),
     D0(0), Dk(0), SumD(0), SumD2(0), MeanD(0), VarD(0), Surge(-999),
     AvgValue(0), IsRebal(true), DayNow(0), DayPrev(0), MAval(0), MAOK(true), Bias(0),
     Ret20(-999), RetOK(true), MktCap(0), CapOK(true);
SetTotalBar(100);
SetBackBar(100);
SetBackBar(40, "M");
if BarFreq <> "D" and BarFreq <> "AD" then RaiseRunTimeError("請使用日線盤後選股");
Ret = 0;
RevOK = true; D0 = 0; SumD = 0; SumD2 = 0;
for Idx = 0 to 24 begin
    Idx2 = Idx + 12;
    RevA = GetField("月營收", "M", Default:=-999999999)[Idx];
    RevB = GetField("月營收", "M", Default:=-999999999)[Idx2];
    if RevA = MissingValue or RevB = MissingValue then RevOK = false;
    if RevA <= 0 or RevB <= 0 then RevOK = false;
    Dk = RevA - RevB;
    if Idx = 0 then D0 = Dk;
    if Idx >= 1 then begin
        SumD = SumD + Dk;
        SumD2 = SumD2 + Dk * Dk;
    end;
end;
Surge = -999;
if RevOK = true then begin
    MeanD = SumD / 24;
    VarD = (SumD2 - 24 * MeanD * MeanD) / 23;
    if VarD > 0 then Surge = (D0 - MeanD) / Power(VarD, 0.5);
end;
AvgValue = Average(Close * Volume, LiqDays) / 10;
MAval = Average(Close, MALen);
Bias = 0;
if MAval > 0 then Bias = (Close / MAval - 1) * 100;
MAOK = true;
if UseMA60 = 1 then begin
    MAOK = false;
    if CurrentBar >= MALen and Close > MAval then MAOK = true;
end;
Ret20 = -999;
if Close[RetLen] > 0 then Ret20 = (Close / Close[RetLen] - 1) * 100;
RetOK = false;
if Ret20 > MinRet20 and Ret20 <= MaxRet20 then RetOK = true;
// 市值區間（需在 XQ 確認欄位名與單位）
MktCap = GetField("總市值", "D", Default:=-1);
CapOK = true;
if MinMktCap > 0 or MaxMktCap < 99999999 then begin
    CapOK = false;
    if MktCap >= MinMktCap and MktCap < MaxMktCap then CapOK = true;
end;
IsRebal = true;
if OnlyRebalDay = 1 then begin
    DayNow = DayOfMonth(Date);
    DayPrev = DayOfMonth(Date[1]);
    IsRebal = false;
    if DayNow >= RebalDay and (DayPrev < RebalDay or DayPrev > DayNow) then IsRebal = true;
end;
if IsRebal and Close >= MinPrice
   and AvgValue >= MinAvgValue
   and MAOK
   and RetOK
   and CapOK
   and Surge > -999
   and Surge >= MinSurge then Ret = 1;
OutputField(1, Surge, 2, "SURGE");
OutputField(2, AvgValue, 0, "60日均成交值(萬)");
OutputField(3, Bias, 1, "距季線%");
OutputField(4, Ret20, 1, "前20日漲幅%");
OutputField(5, D0, 0, "本月營收年增金額");
OutputField(6, MktCap, 0, "總市值(億)");
