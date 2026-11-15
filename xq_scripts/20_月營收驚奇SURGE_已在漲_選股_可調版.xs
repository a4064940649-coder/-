// ===== S4-E3' 月營收驚奇 SURGE＋站上季線＋進場前已在漲（選股腳本，進場）v1.0 =====
// 以「18_月營收驚奇SURGE_加季線_已在漲_選股.xs」為底，多一個參數 MaxRet20（前 20 日漲幅上限），預設值全部等於 E3'。
// ★ 在 XQ 回測前請先設定三件事（研究中心 S4 的固定規格）：
//   (1) 股票範圍排除股名帶「-創」（含「-KY創」）的創新板股票。研究中心的數字全部是排除後的；不排除時 E3 會抱到 7610 聯友金屬-創（+1,291%），
//       2019 年起年化從 48.60% 變 59.27%，這多出來的部分來自一檔股票。
//   (2) 「最多持有 10 檔、滿了就不買」；每檔 1/10 資金；買得到的金額不到 1/10 資金的一半就不買（研究中心的最小買進金額規則）。
//   (3) 沒有訊號、持股不足 10 檔時，剩下的資金放 0050（研究中心 F1、F5 的主欄是空位放 0050；E3' 主欄空位放現金，但放 0050 結果相近）。
// 規則（E3'）：每月營收公布後換股一次。在「收盤 > 60 日均線」而且「前 20 日漲幅 > 10%」的上市股票裡，挑 SURGE 最大的 10 檔。
//   出場要搭配「21_月營收驚奇SURGE_寬出場_可調版.xs」：換股日掉出「季線上股票的 SURGE 前 20 名」或跌破季線才賣。
// SURGE＝（本月營收的年增「金額」− 過去 24 個月年增金額的平均）÷ 過去 24 個月年增金額的標準差。
// 用法：選股腳本；請用「還原日線」盤後執行；股池選「上市」；換股日＝每月第一個「日期 >= RebalDay」的交易日（預設 11）。
// 研究中心回測（官方日 K、只上市、排除創新板、100 萬、10 檔；嚴成本＝手續費 2.8 折＋證交稅＋每邊半個跳動點＋每邊再加 0.3% 滑價）：
//   E3'（空位現金）：2019 年起年化 48.60%（同期 0050 28.91%）、2015-07 起 37.53%（0050 21.14%）、2024 年起 48.91%（0050 54.48%，輸）、最大回撤 -45.3%。
//     成本加倍 36.09%／27.31%；禁買它自己最賺的 5 筆後 43.11%／34.00%；空位放 0050 是 49.03%／37.28%。
//     20 條路徑（起始資金 60～200 萬 × 換股日晚 0～3 天）的 2019 年起年化：最小 38.50%、中位數 46.45%、最大 50.66%。
//     換股日晚 5 天 38.33%、晚 10 天 34.17%（2019 年起）：對換股日很敏感，請在 XQ 把 RebalDay 改 11、13、15 各跑一次。
// 研究中心事前登記的五個改法，機械判定「相對 E3' 改善」的一個都沒有（所以預設值維持 E3'）：
//   F1 只買前 10 名中已在漲的、不遞補（空位 0050）：53.63%／37.70%，判定「沒差」（2019 年起較好、前半較差，逐月比較每月 -0.14 個百分點，t -0.33）。
//      想試就把 MinSurge 設 2.5：研究中心 129 個月裡「季線上股票第 10 名的 SURGE」中位數 2.52（四分位 2.23～3.01，最小 1.09、最大 5.02）。
//      固定門檻只是近似名次：門檻偏高的月份會少買、偏低的月份會買到 11 名以後；建議 MinSurge 設 2.2、2.5、3.0 各跑一次。
//      參考：第 15 名中位數 2.27（四分位 1.93～2.57）；第 20 名中位數 2.01（四分位 1.72～2.37）。
//   F3 追高上限（MaxRet20 = 30）：37.66%／29.98%，成本加倍後輸 0050，判定「沒差」（前半少 14.49 個百分點）。
//   F5（F1＋F3＋F4，空位 0050）：48.54%／34.20%，判定「沒差」。F2（遞補只到前 20 名）「變差」；F4 見出場腳本。
//   鄰域 N15（只買前 10 名中前 20 日漲幅 > 15% 的、空位 0050；MinRet20 = 15、MinSurge = 2.5）：56.84%／41.61%，機械判定「改善」，
//      但它是事前講好只當穩健性的鄰域格，平均只持 3.65 檔；不拿它取代 E3'。
// 未在 XQ 編譯，也未在 XQ 回測。語法只用了 xq_scripts 既有腳本出現過的寫法。
// 需在 XQ 確認：(1) 欄位名「月營收」；(2) 回測時營收公布前不會用到新月份（不偷看）；(3) 研究中心的營收檔沒有 KY 股與已下市公司；
//   (4) 2016-02 以前研究中心沒有訊號；(5) 「最多 10 檔、滿了不買」與「空位放 0050」在 XQ 回測能不能設。
Input: MinRet20(10), MaxRet20(1000), RetLen(20), UseMA60(1), MALen(60), MinAvgValue(1000), MinPrice(10), LiqDays(60), OnlyRebalDay(1), RebalDay(11), MinSurge(0);
// MinRet20：前 RetLen 日漲幅下限（%），要「大於」才入選；10 = E3'（預設）；15 = 鄰域 N15；-100 = 不設（D1）
// MaxRet20：前 RetLen 日漲幅上限（%），要「小於或等於」才入選；1000 = 不設（E3'，預設）；30 = F3／F5 的追高上限
// UseMA60：1 = 收盤要高於 MALen 日均線才入選；0 = 不看均線
// MinAvgValue：近 60 日平均成交值下限（萬元）；1000 = 1,000 萬
// OnlyRebalDay：1 = 只在換股日選股；0 = 每天都算（看清單用）
// MinSurge：SURGE 下限；0 = 不設（E3'，預設，只靠排序取前 10）；2.5 = 近似 F1「只買 L_MA 前 10 名、不遞補」（見開頭註解）
Var: MissingValue(-999999999), Idx(0), Idx2(0), RevA(0), RevB(0), RevOK(true),
     D0(0), Dk(0), SumD(0), SumD2(0), MeanD(0), VarD(0), Surge(-999),
     AvgValue(0), IsRebal(true), DayNow(0), DayPrev(0), MAval(0), MAOK(true), Bias(0),
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
Bias = 0;
if MAval > 0 then Bias = (Close / MAval - 1) * 100;
MAOK = true;
if UseMA60 = 1 then begin
    MAOK = false;
    if CurrentBar >= MALen and Close > MAval then MAOK = true;
end;
// 進場前已在漲：收盤 ÷ RetLen 個交易日前的收盤 − 1（%）要大於 MinRet20
Ret20 = -999;
if Close[RetLen] > 0 then Ret20 = (Close / Close[RetLen] - 1) * 100;
RetOK = false;
if Ret20 > MinRet20 and Ret20 <= MaxRet20 then RetOK = true;
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
OutputField(3, Bias, 1, "距季線%");
OutputField(4, Ret20, 1, "前20日漲幅%");
OutputField(5, D0, 0, "本月營收年增金額");
