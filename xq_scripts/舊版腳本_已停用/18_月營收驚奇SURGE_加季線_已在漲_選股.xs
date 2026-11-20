// ===== S3-E2／E3 月營收驚奇 SURGE＋站上季線＋進場前已在漲（選股腳本，進場）v1.0 =====
// 以「16_月營收驚奇SURGE_加季線_選股.xs」為底，多一個門檻 MinRet20：換股日當天「收盤 ÷ 20 個交易日前的收盤 − 1」要大於 MinRet20(%)，才列入排名。
// 規則：每月營收公布後換股一次。SURGE＝（本月營收的年增「金額」− 過去 24 個月年增金額的平均）÷ 過去 24 個月年增金額的標準差。
//       在「收盤 > 60 日均線」而且「前 20 日漲幅 > 10%」的股票裡，挑 SURGE 最大的 10 檔。
// 用法：選股腳本；請用「還原日線」盤後執行（研究中心的季線與前 20 日漲幅都用含息還原價）；
//       股池在 XQ 選股畫面選「上市」；換股日＝每月第一個「日期 >= RebalDay」的交易日（預設 11）。
// 排序與檔數：依輸出欄位 1「SURGE」由大到小，取前 10 檔，每檔 1/10 資金；最多持有 10 檔，有空位才買；買不到的不遞補。
// 兩種測法（差別只在出場）：
//   E2：下次換股日不在這份名單（前 10 名）就賣。XQ 選股回測設「每月換股、持有到下次換股」即可；
//       不能設月換股時，搭配「19_月營收驚奇SURGE_寬出場_近似.xs」，參數 ExitSurge = 1.75、UseRet20Exit = 1。
//   E3：買進用這支；出場改成「換股日掉出『收盤 > 季線的股票裡 SURGE 前 20 名』才賣」，還在前 20 名就續抱（續抱不看前 20 日漲幅）。
//       XQ 的月換股回測做不到這個出場，要搭配出場腳本，參數 ExitSurge = 2.0、UseRet20Exit = 0。
// 研究中心回測（官方日 K、只上市、100 萬、10 檔、空位放現金；嚴成本＝手續費 2.8 折＋證交稅＋每邊半個跳動點＋每邊再加 0.3% 滑價）：
//   D1（MinRet20 = -100，不設門檻）：2019 年起年化 41.56%（同期 0050 28.91%）、2015-07 起 30.63%（0050 21.14%）、2024 年起 43.60%（0050 54.48%）、最大回撤 -41.8%。
//   E2：2019 年起 44.10%、2015-07 起 34.26%、2024 年起 52.38%、最大回撤 -38.8%；輕成本 55.70%／43.41%；成本再加倍 27.27%／19.01%（低於 0050）。
//       和 D1 逐月比較每月平均多 0.42 個百分點，t 值 0.96。門檻改 5% 是 40.69%／32.44%、改 15% 是 41.39%／31.31%，兩邊都沒有比 D1 好，
//       所以 10% 這個數字本身不穩，請在 XQ 把 MinRet20 設 5、10、15 各跑一次。
//   E3：2019 年起 59.20%、2015-07 起 44.28%、2024 年起 81.09%、最大回撤 -44.4%；輕成本 64.62%／49.23%；成本再加倍 46.62%／34.08%。
//       其中一筆 7610（2025-12-12 買、2026-06-12 賣，+1,291%）占全部已平倉損益的 40.7%。整段不准買 7610 時是 48.94%／37.79%（同樣不准買的 D1 是 39.55%／29.35%）。
//       禁買它自己最賺的 5 筆後是 42.03%／33.35%。和 D1 逐月比較每月平均多 1.19 個百分點，t 值 2.29；不准買 7610 時是 0.81 個百分點，t 值 1.75。
// MinRet20 = -100 時就是 16 號腳本（D1）。
// 未在 XQ 編譯，也未在 XQ 回測。
// 需在 XQ 確認：
//   (1) 欄位名「月營收」（現有腳本只在 12～17 號用過，同樣未編譯）。
//   (2) 回測時 GetField("月營收","M")[0] 在營收公布前是否仍是上上個月的值（不能偷看）。研究中心的口徑是
//       「次月 10 日（遇假日順延）之後的第一個交易日」才用；10 日遇假日時 XQ 版會早一天，可把 RebalDay 改 12。
//   (3) 研究中心的營收檔沒有 KY 股與已下市公司，XQ 的選股結果會多出這些股票。
//   (4) 2016-02 以前研究中心沒有訊號，XQ 回測請從 2016-02 以後看。
//   (5) E3 需要「最多 10 檔、滿了就不買」；XQ 回測能不能這樣設，需在 XQ 確認。
Input: MinRet20(10), RetLen(20), UseMA60(1), MALen(60), MinAvgValue(1000), MinPrice(10), LiqDays(60), OnlyRebalDay(1), RebalDay(11), MinSurge(0);
// MinRet20：前 RetLen 日漲幅下限（%），要「大於」才入選；10 = E2／E3；-100 = 不設（D1）
// UseMA60：1 = 收盤要高於 MALen 日均線才入選；0 = 不看均線
// MinAvgValue：近 60 日平均成交值下限（萬元）；1000 = 1,000 萬
// OnlyRebalDay：1 = 只在換股日選股；0 = 每天都算（看清單用）
// MinSurge：SURGE 下限；0 = 不設（只靠排序取前 10）
Var: MissingValue(-999999999), Idx(0), Idx2(0), RevA(0), RevB(0), RevOK(true),
     D0(0), Dk(0), SumD(0), SumD2(0), MeanD(0), VarD(0), Surge(-999),
     AvgValue(0), IsRebal(true), DayNow(0), DayPrev(0), MAval(0), MAOK(true), MaBias(0),
     Ret20(-999), RetOK(true);
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
// 季線：收盤 > MALen 日均線
MAval = Average(Close, MALen);
MaBias = 0;
if MAval > 0 then MaBias = (Close / MAval - 1) * 100;
MAOK = true;
if UseMA60 = 1 then begin
    MAOK = false;
    if CurrentBar >= MALen and Close > MAval then MAOK = true;
end;
// 進場前已在漲：收盤 ÷ RetLen 個交易日前的收盤 − 1（%）要大於 MinRet20
Ret20 = -999;
if Close[RetLen] > 0 then Ret20 = (Close / Close[RetLen] - 1) * 100;
RetOK = false;
if Ret20 > MinRet20 then RetOK = true;
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
   and MAOK
   and RetOK
   and Surge > -999
   and Surge >= MinSurge then Ret = 1;
OutputField(1, Surge, 2, "SURGE");
OutputField(2, AvgValue, 0, "60日均成交值(萬)");
OutputField(3, MaBias, 1, "距季線%");
OutputField(4, Ret20, 1, "前20日漲幅%");
OutputField(5, D0, 0, "本月營收年增金額");
