// ===== 現行策略：月營收驚奇 SURGE 進場＋出場合併版（交易腳本）v1.0（2026-11-16）=====
// 用途：給 XQ 需要「交易腳本」的回測（例如策略回測、投資組合回測）。進場條件＝20 號、出場條件＝21 號，合在一支裡。
// 請用「還原日線」。股票範圍：上市普通股，排除股名帶「-創」的。
// 回測資金設定（在回測畫面設）：最多同時持有 10 檔、滿了不買；每檔金額＝目前淨值的 1/10；訊號隔天開盤成交。
// 進場（只在每月換股日＝第一個日期 >= RebalDay 的交易日）：
//   SURGE >= MinSurge(2.5)、前 20 日漲幅 > 10%、收盤 > 60 日均線、60 日均成交值 >= 1,000 萬、收盤 >= 10。
// 出場（只在換股日）：SURGE < ExitSurge(2.0) 或 收盤 <= 60 日均線。月中不出場、不停損、不停利。
// 研究中心回測（照 XQ 固定門檻、只算衛星、空位現金、70 萬）：2019 年起約 40%（20 條路徑約 33%～45%）、最大回撤約 −32%～−34%。
// 未在 XQ 編譯。需在 XQ 確認：(1) SetPosition 的寫法與 MARKET 常數；(2) 回測是否在下一根 K 棒開盤成交；
//   (3) 部位數量由回測的資金設定決定（這裡的 1 代表「持有」）；(4) 欄位名「月營收」（20 號已編譯成功，應可用）。
Input: MinSurge(2.5), ExitSurge(2.0), MinRet20(10), RetLen(20), UseMA60(1), MALen(60), MinAvgValue(1000), MinPrice(10), LiqDays(60), RebalDay(11);
Var: MissingValue(-999999999), Idx(0), Idx2(0), RevA(0), RevB(0), RevOK(true),
     D0(0), Dk(0), SumD(0), SumD2(0), MeanD(0), VarD(0), Surge(-999),
     AvgValue(0), IsRebal(false), DayNow(0), DayPrev(0), MAval(0), MAOK(true), Ret20(-999),
     EntryOK(false), ExitOK(false);
SetTotalBar(100);
SetBackBar(100);
SetBackBar(40, "M");
// SURGE：本月營收年增金額，相對過去 24 個月年增金額的平均與標準差
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
MAOK = true;
if UseMA60 = 1 then begin
    MAOK = false;
    if CurrentBar >= MALen and Close > MAval then MAOK = true;
end;
Ret20 = -999;
if Close[RetLen] > 0 then Ret20 = (Close / Close[RetLen] - 1) * 100;
// 換股日：每月第一個日期 >= RebalDay 的交易日
DayNow = DayOfMonth(Date);
DayPrev = DayOfMonth(Date[1]);
IsRebal = false;
if DayNow >= RebalDay and (DayPrev < RebalDay or DayPrev > DayNow) then IsRebal = true;
// 進場條件（20 號）
EntryOK = false;
if IsRebal and Close >= MinPrice and AvgValue >= MinAvgValue and MAOK
   and Ret20 > MinRet20 and Surge > -999 and Surge >= MinSurge then EntryOK = true;
// 出場條件（21 號）
ExitOK = false;
if IsRebal and Surge < ExitSurge then ExitOK = true;
if IsRebal and UseMA60 = 1 and Close <= MAval then ExitOK = true;
// 下單：先判斷出場，再判斷進場
if Position > 0 and ExitOK then SetPosition(0, MARKET);
if Position = 0 and EntryOK then SetPosition(1, MARKET);
