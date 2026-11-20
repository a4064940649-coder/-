// ===== S3-E2／E3 月營收驚奇 SURGE（出場腳本，近似版）v1.0 =====
// 搭配「18_月營收驚奇SURGE_加季線_已在漲_選股.xs」。隔天開盤賣出。請用「還原日線」。
// 以「17_月營收驚奇SURGE_加季線_出場_近似.xs」為底，多兩個東西：ExitSurge 的預設值改成「第 20 名」的近似門檻；多一個開關 UseRet20Exit。
// 只在每月換股日判斷；月中不出場、不設停損。換股日當天任一條成立就賣：
//   (1) 收盤 <= 60 日均線（研究中心的規則裡，這種股票一定不在名單，這一條不是近似）；
//   (2) SURGE < ExitSurge（近似；單檔腳本看不到全市場名次，用歷史上「第 N 名的 SURGE」的中位數代替名次）；
//   (3) UseRet20Exit = 1 時：前 20 日漲幅 <= MinRet20(%)（只有 E2 用；E3 的續抱不看前 20 日漲幅）。
// 兩組參數：
//   E3（寬出場，掉出前 20 名才賣）：ExitSurge = 2.0、UseRet20Exit = 0。
//       2.0＝研究中心 2016-02～2026-10 共 129 個月，「收盤 > 季線的股票」裡第 20 名 SURGE 的中位數 2.01（四分位 1.72～2.37，最小 0.69、最大 4.05）。
//       參考：第 10 名的中位數 2.52（四分位 2.23～3.01）；第 30 名的中位數 1.72（四分位 1.40～2.04）。
//   E2（不在「季線上＋前 20 日漲幅 > 10%」的前 10 名就賣）：ExitSurge = 1.75、UseRet20Exit = 1、MinRet20 = 10。
//       1.75＝E2 名單第 10 名 SURGE 的中位數（四分位 1.30～2.29；129 個月裡有 3 個月名單不足 10 檔，不計）。
// 某個月第 20 名（或第 10 名）的 SURGE 高於或低於門檻時，會和研究中心的出場不一樣：
//   門檻偏低的月份會多抱（實際名次已掉出去卻沒賣），門檻偏高的月份會早賣。四分位寬度約 ±0.3，所以這是粗略近似。
//   可以把 ExitSurge 設 1.7、2.0、2.4 各跑一次，看結果對這個門檻敏不敏感。
// 研究中心回測（嚴成本、只上市、100 萬、10 檔、空位放現金；用真正的名次，不是這支的近似）：
//   E3：2019 年起年化 59.20%（同期 0050 28.91%、D1 41.56%）、2015-07 起 44.28%（0050 21.14%、D1 30.63%）、2024 年起 81.09%、最大回撤 -44.4%。
//       其中 7610 一筆（+1,291%）占已平倉損益 40.7%；整段不准買 7610 時是 48.94%／37.79%。
//   只換出場、進場不加漲幅門檻（E1）：42.15%／32.04%，和 D1 沒有差別；出場門檻放寬到前 30 名（E4）：43.26%／32.61%。
//   E2：44.10%／34.26%、最大回撤 -38.8%。
// E2 能用 XQ 選股回測的「每月換股」時，請用月換股，不要用這支。E3 一定要用這支（或自己在 XQ 用名次做）。
// 未在 XQ 編譯，也未在 XQ 回測。欄位名「月營收」需在 XQ 確認（同進場腳本）。
Input: ExitSurge(2.0), UseMA60(1), MALen(60), UseRet20Exit(0), MinRet20(10), RetLen(20), RebalDay(11);
// ExitSurge：換股日 SURGE 低於這個值就賣；2.0 = 近似「掉出前 20 名」（E3）；1.75 = E2；2.5 = 近似「掉出前 10 名」（D1，同 17 號腳本）
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
