// ===== S2-D1／D6 月營收驚奇 SURGE＋站上季線（選股腳本，進場）v1.0 =====
// 以「12_月營收驚奇SURGE_選股.xs」為底，多一個開關 UseMA60：換股日當天收盤要在 60 日均線之上，才列入排名。
// 規則：每月營收公布後換股一次。SURGE＝（本月營收的年增「金額」− 過去 24 個月年增金額的平均）÷ 過去 24 個月年增金額的標準差。
//       在「收盤 > 60 日均線」的股票裡，挑 SURGE 最大的 10 檔。
// 用法：選股腳本；請用「還原日線」盤後執行（研究中心的季線用含息還原價；用一般日線時，除權息後幾天的判斷會不同）；
//       股池在 XQ 選股畫面選「上市」；換股日＝每月第一個「日期 >= RebalDay」的交易日（預設 11）。
// 排序與檔數：依輸出欄位 1「SURGE」由大到小，取前 10 檔，每檔 1/10 資金；買不到的不遞補。
// 出場：XQ 選股回測請設「每月換股、持有到下次換股」——下次換股日不在前 10 名（含跌破季線被排除）就賣、還在就續抱。
//       不能設月換股時，搭配「17_月營收驚奇SURGE_加季線_出場_近似.xs」。
// 兩種測法：
//   D1：隔天開盤買、隔天開盤賣（XQ 選股回測的一般設定）。
//   D6：賣出仍在隔天開盤，買進改在隔天「收盤」；收盤漲停不買。XQ 回測能不能把進場價設成隔日收盤，需在 XQ 確認；不能設就只測 D1。
// 研究中心回測（官方日 K、只上市、100 萬、10 檔、空位放現金；嚴成本＝手續費 2.8 折＋證交稅＋每邊半個跳動點＋每邊再加 0.3% 滑價）：
//   D1：2019 年起年化 41.56%（同期 0050 28.91%；不加季線的原版 37.03%）、2015-07 起 30.63%（0050 21.14%）、2024 年起 43.60%（0050 54.48%）、最大回撤 -41.8%。
//   D6：2019 年起 41.32%、2015-07 起 31.31%、2024 年起 45.43%、最大回撤 -43.3%。
//   輕成本（不加 0.3% 滑價）D1 是 50.63%／38.83%。成本再加倍（每邊滑價 0.6%）D1 剩 27.99%，低於 0050。
//   和原版逐月比較：D1 每月平均多 0.33 個百分點，但 t 值只有 1.50，還不能排除是運氣。
// UseMA60 = 0 時就是原版（12 號腳本）。
// 未在 XQ 編譯，也未在 XQ 回測。
// 需在 XQ 確認：
//   (1) 欄位名「月營收」（現有腳本只在 12～15 號用過，同樣未編譯）。
//   (2) 回測時 GetField("月營收","M")[0] 在營收公布前是否仍是上上個月的值（不能偷看）。研究中心的口徑是
//       「次月 10 日（遇假日順延）之後的第一個交易日」才用；10 日遇假日時 XQ 版會早一天，可把 RebalDay 改 12。
//   (3) 研究中心的營收檔沒有 KY 股與已下市公司，XQ 的選股結果會多出這些股票。
//   (4) 2016-02 以前研究中心沒有訊號，XQ 回測請從 2016-02 以後看。
Input: UseMA60(1), MALen(60), MinAvgValue(1000), MinPrice(10), LiqDays(60), OnlyRebalDay(1), RebalDay(11), MinSurge(0);
// UseMA60：1 = 收盤要高於 MALen 日均線才入選（D1／D6）；0 = 不看均線（原版）
// MinAvgValue：近 60 日平均成交值下限（萬元）；1000 = 1,000 萬
// OnlyRebalDay：1 = 只在換股日選股；0 = 每天都算（看清單用）
// MinSurge：SURGE 下限；0 = 不設（只靠排序取前 10）
Var: MissingValue(-999999999), Idx(0), Idx2(0), RevA(0), RevB(0), RevOK(true),
     D0(0), Dk(0), SumD(0), SumD2(0), MeanD(0), VarD(0), Surge(-999),
     AvgValue(0), IsRebal(true), DayNow(0), DayPrev(0), MAval(0), MAOK(true), Bias(0);
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
Bias = 0;
if MAval > 0 then Bias = (Close / MAval - 1) * 100;
MAOK = true;
if UseMA60 = 1 then begin
    MAOK = false;
    if CurrentBar >= MALen and Close > MAval then MAOK = true;
end;
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
   and Surge > -999
   and Surge >= MinSurge then Ret = 1;
OutputField(1, Surge, 2, "SURGE");
OutputField(2, AvgValue, 0, "60日均成交值(萬)");
OutputField(3, Bias, 1, "距季線%");
OutputField(4, D0, 0, "本月營收年增金額");
