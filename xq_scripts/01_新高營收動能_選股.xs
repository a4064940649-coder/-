// ===== 01 新高＋營收動能（選股腳本）v1.0 =====
// 研究中心主策略的 XQ 版：成交熱門股裡，挑「股價最接近 52 週新高」而且「近 3 個月營收年增率最高」的股票。
// 用法：選股腳本；日線（建議還原日線）盤後執行；每月換股一次（每月 11 日以後的第一個交易日）。
// 排序：依輸出欄位 1「距52週高%」由大到小，取前 10 檔，平均買進，抱到下個月換股。
// 注意：原版是用「成交值前 300 名」當股池、兩個分數各自排名後相加；XS 單一腳本做不到全市場排名，
//       這裡改成門檻（成交值、營收年增）＋排序（距高點），是近似版。未在 XQ 編譯或回測。
Input: MinHighRatio(85), MinRevYoY3(30), MinAvgValue(10000), MinPrice(10),
       HighDays(252), LiqDays(60), OnlyRebalDay(0), RebalDay(11);
// MinHighRatio：收盤 ÷ 52 週最高收盤 的下限(%)
// MinRevYoY3：近 3 個月營收年增率平均的下限(%)
// MinAvgValue：近 60 日平均成交值下限（萬元）；10000 = 1 億
// OnlyRebalDay：1 = 只在每月第一個「日期 >= RebalDay」的交易日選股；0 = 每天都選
Var: MissingValue(-999999999), RevM(0), RevYoY3(0), RevOK(true), Idx(0),
     HighC(0), HighRatio(0), AvgValue(0), IsRebal(true), DayNow(0), DayPrev(0);
SetTotalBar(300);
SetBackBar(300);
SetBackBar(6, "M");
if BarFreq <> "D" and BarFreq <> "AD" then RaiseRunTimeError("請使用日線盤後選股");
Ret = 0;
// 近 3 個月營收年增率平均
RevYoY3 = 0; RevOK = true;
for Idx = 0 to 2 begin
    RevM = GetField("月營收年增率", "M", Default:=-999999999)[Idx];
    if RevM = MissingValue then RevOK = false else RevYoY3 = RevYoY3 + RevM / 3;
end;
if RevOK = false then RevYoY3 = -999;
// 距 52 週最高收盤
HighC = Highest(Close, HighDays);
HighRatio = 0;
if HighC > 0 and CurrentBar >= HighDays then HighRatio = Close / HighC * 100;
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
   and HighRatio >= MinHighRatio
   and RevYoY3 >= MinRevYoY3 then Ret = 1;
OutputField(1, HighRatio, 1, "距52週高%");
OutputField(2, RevYoY3, 1, "近3月營收年增%");
OutputField(3, AvgValue, 0, "60日均成交值(萬)");
