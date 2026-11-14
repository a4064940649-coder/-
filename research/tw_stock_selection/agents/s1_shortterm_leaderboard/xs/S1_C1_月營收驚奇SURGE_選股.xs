// ===== S1-C1 月營收驚奇 SURGE（選股腳本，進場）v1.0 =====
// 規則：每月營收公布後換股一次。SURGE＝（本月營收的年增「金額」− 過去 24 個月年增金額的平均）÷ 過去 24 個月年增金額的標準差。
//       也就是「這個月的營收年增金額，比它自己過去兩年的常態多了幾個標準差」。挑 SURGE 最大的 10 檔。
// 用法：選股腳本；日線盤後執行；股池在 XQ 選股畫面選「上市」（主欄）或「上市＋上櫃」（並列欄）。
//       換股日＝每月第一個「日期 >= RebalDay」的交易日（預設 11）；隔天開盤成交。
// 排序與檔數：依輸出欄位 1「SURGE」由大到小，取前 10 檔，每檔 1/10 資金。
// 出場：XQ 選股回測請設「每月換股、持有到下次換股」——下次換股日不在前 10 名就賣、還在就續抱，這樣不需要出場腳本。
//       不能設月換股時，搭配「S1_C1_月營收驚奇SURGE_出場_近似.xs」。
// 研究中心回測（官方日 K、只上市、10 檔、空位放現金、扣 2.8 折手續費＋證交稅＋每邊半個跳動點）：
//       2019 年起年化 42.8%（同期 0050 28.9%）、2015-07 起 31.4%（0050 21.1%）、最大回撤 -47.9%；平均持有 26 個交易日。
// 未在 XQ 編譯，也未在 XQ 回測。
// 需在 XQ 確認的三件事：
//   (1) 欄位名「月營收」未在現有腳本出現過（現有腳本只用過「月營收年增率」）；寫法相同，單位不影響結果。
//   (2) 回測時 GetField("月營收","M")[0] 在營收公布前是否仍是上上個月的值（不能偷看）。研究中心的口徑是
//       「次月 10 日（遇假日順延）之後的第一個交易日」才用；10 日遇假日時 XQ 版會比研究中心早一天，可把 RebalDay 改 12。
//   (3) 2016-02 以前研究中心沒有訊號（2013 年會計口徑改變，照原研究剔除），XQ 回測請從 2016-02 以後看。
Input: MinAvgValue(1000), MinPrice(10), LiqDays(60), OnlyRebalDay(1), RebalDay(11), MinSurge(0);
// MinAvgValue：近 60 日平均成交值下限（萬元）；1000 = 1,000 萬
// OnlyRebalDay：1 = 只在換股日選股；0 = 每天都算（看清單用）
// MinSurge：SURGE 下限；0 = 不設（只靠排序取前 10）
Var: MissingValue(-999999999), Idx(0), Idx2(0), RevA(0), RevB(0), RevOK(true),
     D0(0), Dk(0), SumD(0), SumD2(0), MeanD(0), VarD(0), Surge(-999),
     AvgValue(0), IsRebal(true), DayNow(0), DayPrev(0);
SetTotalBar(100);
SetBackBar(100);
SetBackBar(40, "M");
if BarFreq <> "D" and BarFreq <> "AD" then RaiseRunTimeError("請使用日線盤後選股");
Ret = 0;
// 年增金額 D(k) = 月營收[k] − 月營收[k+12]；k = 0 是最新月，k = 1..24 是過去 24 個月
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
// 近 60 日平均成交值（萬元）：成交量(張) × 收盤(元) = 千元，再 ÷ 10 = 萬元
AvgValue = Average(Close * Volume, LiqDays) / 10;
// 換股日：每月第一個日期 >= RebalDay 的交易日
IsRebal = true;
if OnlyRebalDay = 1 then begin
    DayNow = DayOfMonth(Date);
    DayPrev = DayOfMonth(Date[1]);
    IsRebal = false;
    if DayNow >= RebalDay and (DayPrev < RebalDay or DayPrev > DayNow) then IsRebal = true;
end;
if IsRebal and Close >= MinPrice
   and AvgValue >= MinAvgValue
   and Surge > -999
   and Surge >= MinSurge then Ret = 1;
OutputField(1, Surge, 2, "SURGE");
OutputField(2, AvgValue, 0, "60日均成交值(萬)");
OutputField(3, D0, 0, "本月營收年增金額");
