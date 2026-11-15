// ===== S4-E3' 月營收驚奇 SURGE（出場腳本，近似版）v1.0 =====
// 搭配「S4_E3p_月營收驚奇SURGE_已在漲_選股.xs」。隔天開盤賣出。請用「還原日線」。
// 以「19_月營收驚奇SURGE_寬出場_近似.xs」為底，多一個開關 UseF4Exit（預設 0 = E3'）。
// ★ 請在 XQ 排除股名帶「-創」的股票、設「最多 10 檔、滿了不買」、沒訊號時的資金放 0050（說明見進場腳本開頭）。
// 只在每月換股日判斷；月中不出場、不設停損。換股日當天任一條成立就賣：
//   (1) 收盤 <= 60 日均線（不是近似）；
//   (2) SURGE < ExitSurge（近似「掉出季線上股票的 SURGE 前 20 名」；E3' 用 2.0）；
//   (3) UseRet20Exit = 1 時：前 20 日漲幅 <= MinRet20(%)（E2 用；E3' 不用）；
//   (4) UseF4Exit = 1 時：SURGE < ExitSurge10（近似掉出前 10 名）而且前 20 日漲幅 <= 0（F4：11～20 名要還在漲才續抱）。
// 門檻的根據（研究中心 2016-02～2026-10 共 129 個月、只上市、排除創新板，「收盤 > 季線的股票」裡第 N 名的 SURGE）：
//   第 10 名中位數 2.52（四分位 2.23～3.01）；第 15 名 2.27（1.93～2.57）；第 20 名 2.01（1.72～2.37，最小 0.69、最大 4.05）。
//   固定門檻只是近似名次（四分位寬約 ±0.3），建議 ExitSurge 設 1.7、2.0、2.4 各跑一次。
// 研究中心回測（嚴成本、只上市、排除創新板、100 萬、10 檔；用真正的名次，不是這支的近似）：
//   E3'（空位現金）：2019 年起年化 48.60%（0050 28.91%）、2015-07 起 37.53%（0050 21.14%）、2024 年起 48.91%（0050 54.48%）、最大回撤 -45.3%。
//   F4（UseF4Exit = 1）：45.36%／36.07%、最大回撤 -46.1%；相對 E3' 判定「沒差」（2019 年起少 3.24 個百分點、逐月每月 -0.15 個百分點，t -0.94）。
//   所以預設 UseF4Exit = 0。
// 未在 XQ 編譯，也未在 XQ 回測。語法只用了 xq_scripts 既有腳本出現過的寫法。欄位名「月營收」需在 XQ 確認。
Input: ExitSurge(2.0), UseF4Exit(0), ExitSurge10(2.5), UseMA60(1), MALen(60), UseRet20Exit(0), MinRet20(10), RetLen(20), RebalDay(11);
// ExitSurge：換股日 SURGE 低於這個值就賣；2.0 = 近似「掉出前 20 名」（E3）；1.75 = E2；2.5 = 近似「掉出前 10 名」（D1，同 17 號腳本）
// UseF4Exit：1 = F4 的續抱條件：SURGE < ExitSurge10（近似掉出前 10 名）而且前 RetLen 日漲幅 <= 0 就賣；0 = 不用（E3'，預設）
// ExitSurge10：近似「第 10 名」的 SURGE；2.5 = 129 個月中位數 2.52
// UseMA60：1 = 換股日收盤 <= MALen 日均線就賣；0 = 不看均線
// UseRet20Exit：1 = 換股日前 RetLen 日漲幅 <= MinRet20(%) 就賣（E2）；0 = 不看（E3）
Var: MissingValue(-999999999), Idx(0), Idx2(0), RevA(0), RevB(0), RevOK(true),
     D0(0), Dk(0), SumD(0), SumD2(0), MeanD(0), VarD(0), Surge(-999),
     IsRebal(false), DayNow(0), DayPrev(0), MAval(0), Ret20(-999);
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
MAval = Average(Close, MALen);
Ret20 = -999;
if Close[RetLen] > 0 then Ret20 = (Close / Close[RetLen] - 1) * 100;
// 換股日：每月第一個日期 >= RebalDay 的交易日
DayNow = DayOfMonth(Date);
DayPrev = DayOfMonth(Date[1]);
IsRebal = false;
if DayNow >= RebalDay and (DayPrev < RebalDay or DayPrev > DayNow) then IsRebal = true;
// 營收缺值（Surge = -999）也視為不在名單
if IsRebal and Surge < ExitSurge then Ret = 1;
if IsRebal and UseMA60 = 1 and Close <= MAval then Ret = 1;
if IsRebal and UseRet20Exit = 1 and Ret20 <= MinRet20 then Ret = 1;
if IsRebal and UseF4Exit = 1 and Surge < ExitSurge10 and Ret20 <= 0 then Ret = 1;
