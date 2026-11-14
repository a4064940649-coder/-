// ===== S1-C1 月營收驚奇 SURGE（出場腳本，近似版）v1.0 =====
// 搭配「S1_C1_月營收驚奇SURGE_選股.xs」。隔天開盤賣出。
// 研究中心的出場規則：每月換股日，持股不在「SURGE 最大的 10 檔」裡就賣，還在就續抱；月中不出場、不設停損。
// 單檔腳本看不到全市場排名，這裡用門檻近似：換股日當天，這檔的 SURGE < ExitSurge 就賣。
// ExitSurge 預設 2.8＝研究中心 2016-02～2026-10 各月「第 10 名的 SURGE」的中位數 2.79（四分位 2.46～3.24）。
// 這是近似：某個月第 10 名的 SURGE 高於或低於 2.8 時，會和研究中心的出場不一樣。
// 能用 XQ 選股回測的「每月換股」時，請用月換股，不要用這支。
// 未在 XQ 編譯，也未在 XQ 回測。欄位名「月營收」需在 XQ 確認（同進場腳本）。
Input: ExitSurge(2.8), RebalDay(11);
Var: MissingValue(-999999999), Idx(0), Idx2(0), RevA(0), RevB(0), RevOK(true),
     D0(0), Dk(0), SumD(0), SumD2(0), MeanD(0), VarD(0), Surge(-999),
     IsRebal(false), DayNow(0), DayPrev(0);
SetTotalBar(100);
SetBackBar(100);
SetBackBar(40, "M");
if BarFreq <> "D" and BarFreq <> "AD" then RaiseRunTimeError("請使用日線");
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
// 換股日：每月第一個日期 >= RebalDay 的交易日
DayNow = DayOfMonth(Date);
DayPrev = DayOfMonth(Date[1]);
IsRebal = false;
if DayNow >= RebalDay and (DayPrev < RebalDay or DayPrev > DayNow) then IsRebal = true;
// 營收缺值（Surge = -999）也視為不在名單
if IsRebal and Surge < ExitSurge then Ret = 1;
