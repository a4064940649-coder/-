// ===== 14 月營收驚奇 SURGE：營收一公布就進場（選股腳本，進場）v1.0 =====
// 和 12 的差別：12 是等到每月 11 日以後才一起換股；這支是「哪一天營收更新，哪一天就算」，隔天開盤買。
// 為什麼要做：研究中心回測發現進場越晚越差——晚 1／3／5／10 個交易日，年化從 43.0% 掉到 39.8%／38.0%／27.9%。
//             營收其實在每月 1～10 日陸續公布，多數公司在 6～8 日就公布了，等到 11 日已經晚了幾天。
// 研究中心沒有歷史的「實際公布日」資料，這個版本沒辦法自己回測，要靠 XQ 回測（XQ 的月營收欄位若依公布日更新就能測）。
// 規則：今天這檔的月營收數字和昨天不同（代表剛公布新月份）→ 算 SURGE → SURGE >= MinSurge 就入選。
// SURGE＝（本月營收年增金額 − 過去 24 個月年增金額的平均）÷ 過去 24 個月年增金額的標準差。
// 用法：選股腳本；日線盤後每天執行；股池選「上市」；同時最多 10 檔，每檔 1/10 資金；
//       訊號多於空位時依輸出欄位 1「SURGE」由大到小。
// 出場：在 XQ 回測設定「持有 20 個交易日後出場」（研究中心測過固定 20 日：年化 38.9%、最大回撤 -38.7%）；
//       或搭配 15（下次營收公布時 SURGE 不到門檻就賣）。
// MinSurge 預設 2.8＝研究中心各月「第 10 名的 SURGE」的中位數。
// 未在 XQ 編譯，也未在 XQ 回測。需在 XQ 確認：欄位名「月營收」；回測時月營收是不是在實際公布日才更新。
Input: MinSurge(2.8), MinAvgValue(1000), MinPrice(10), LiqDays(60);
Var: MissingValue(-999999999), Idx(0), Idx2(0), RevA(0), RevB(0), RevOK(true),
     D0(0), Dk(0), SumD(0), SumD2(0), MeanD(0), VarD(0), Surge(-999),
     AvgValue(0), RevNow(0), NewRev(false);
SetTotalBar(100);
SetBackBar(100);
SetBackBar(40, "M");
if BarFreq <> "D" and BarFreq <> "AD" then RaiseRunTimeError("請使用日線盤後選股");
Ret = 0;
// 今天的最新月營收和昨天不同 → 剛公布
RevNow = GetField("月營收", "M", Default:=-999999999);
NewRev = false;
if CurrentBar > 1 and RevNow <> MissingValue and RevNow <> RevNow[1] then NewRev = true;
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
if NewRev and Close >= MinPrice
   and AvgValue >= MinAvgValue
   and Surge >= MinSurge then Ret = 1;
OutputField(1, Surge, 2, "SURGE");
OutputField(2, AvgValue, 0, "60日均成交值(萬)");
OutputField(3, D0, 0, "本月營收年增金額");
