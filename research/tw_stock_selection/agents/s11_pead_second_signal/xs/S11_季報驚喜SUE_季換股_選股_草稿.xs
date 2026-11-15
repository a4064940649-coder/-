// ===== S11 季報驚喜 SUE（季換股，選股腳本）草稿 v0.1 =====
// ★ 研究中心判定：不建議加進現行策略（季報驚喜單獨衛星 2019 年起年化 23.35%，輸 0050 的 28.91%；和 SURGE 各半合併後每年少約 10 個百分點，回撤沒有變淺）。
//   這支只給 XQ 交叉驗證研究中心的結果、或紙上追蹤用，不要拿真錢跑。細節見 research/tw_stock_selection/agents/s11_pead_second_signal/FINDINGS.md。
// 規則（研究中心 Q 格）：每季財報法定期限後第一個交易日換股一次。期限：Q1 5/15、Q2 8/14、Q3 11/14、年報 3/31（證券交易法第 36 條）。
//   在上市、非金融保險、非創新板、60 日均成交值 >= 1,000 萬、收盤 >= 10 的股票裡，挑「季報驚喜」最大的 10 檔，隔天開盤買；
//   下一季換股日不在新的前 10 名就賣，仍在前 10 名的續抱；季中不出場。資金：0050 底倉 30%、衛星 70%、每檔＝衛星 ÷ 10。
// SUE＝（本季 EPS − 去年同季 EPS）÷ 過去 8 季「EPS 年增金額」的標準差。
// REVSUE＝同法用單季營收。研究中心的主定義 SUE⊥＝pct(SUE) −（a＋b·pct(REVSUE)）要在全體股票之間排名、迴歸，XS 單檔腳本做不到。
//   這裡輸出三個欄位：SUE、REVSUE、近似分數 SUE − B × REVSUE（B 預設 0.5，研究中心各季 pct 迴歸斜率約 0.3～0.66）。
//   要和研究中心逐筆對齊，請把當天選股結果（SUE、REVSUE）匯出到 Excel，自己算百分位與迴歸殘差，再取前 10 名。
//   研究中心另跑了「原始 SUE 排名」（QR 格，等於 B 設 0）：2019 年起 30.80%、最大回撤 −42.09%，成本加倍與禁買最賺 5 筆後都輸 0050。
// 研究中心回測（官方日 K、只上市、排除創新板與金融保險、100 萬、底倉 30%＋衛星 70%、嚴成本）：
//   Q（SUE⊥ 前 10 名）：2019 年起 23.35%、2015-07 起 18.48%、2024 年起 22.75%、最大回撤 −35.77%；20 條路徑 2019 年起中位數 23.12%（0／20 條贏 0050）。
//   QF（再加「收盤 > 季線且前 20 日漲幅 > 10%」）：26.20%／19.84%／36.63%／−29.39%。
//   同期 0050：28.91%／21.14%／54.48%／−33.96%。
// 用法：選股腳本；請用「還原日線」盤後執行；股池選「上市」，並手動排除金融保險類與股名帶「-創」的股票。
// 未在 XQ 編譯，也未在 XQ 回測。語法只用了 xq_scripts 既有腳本出現過的寫法，以下幾處需在 XQ 確認：
//   (1) 欄位名「每股稅後淨利(元)」（02 號腳本用過）與單季營收欄位名「營業收入淨額」（需在 XQ 確認，可能叫「營業收入」或「營收淨額」）；
//   (2) 回測時 XQ 的季資料是在「公布當天」才更新，還是在季底就出現（後者會偷看，研究中心的數字是公布後才用）；
//   (3) 函數 Month(Date) 是否可用（需在 XQ 確認；既有腳本只用過 DayOfMonth）；
//   (4) -KY 公司第二季期限是 8/31，研究中心在 8/14 換股時把它們排除（還沒公布）；XQ 若在 8/14 已看得到它們的舊季資料，會拿上一季的 SUE 排名，請在 Q2 換股日把 -KY 排除；
//   (5) 法定期限遇假日時，研究中心是在「順延後的期限日」的下一個交易日換股，這支腳本是在「期限日之後第一個交易日」，兩者在期限遇假日的季別差一天。
Input: B(0.5), MinAvgValue(1000), MinPrice(10), LiqDays(60), OnlyRebalDay(1), UseMA60(0), MALen(60), MinRet20(-1000), RetLen(20);
// B：近似分數 SUE − B × REVSUE 的係數；0 = 原始 SUE（QR 格）；0.5 = 預設
// UseMA60、MinRet20：QF 格設 UseMA60 = 1、MinRet20 = 10（收盤 > 季線且前 20 日漲幅 > 10%）；預設不設（Q 格）
// OnlyRebalDay：1 = 只在季換股日選股；0 = 每天都算（看清單用）
Var: MissingValue(-999999999), Idx(0), EA(0), EB(0), RA(0), RB(0), EpsOK(true), RevOK(true),
     DE0(0), DEk(0), SumE(0), SumE2(0), MeanE(0), VarE(0), Sue(-999),
     DR0(0), DRk(0), SumR(0), SumR2(0), MeanR(0), VarR(0), RevSue(-999), Score(-999),
     AvgValue(0), IsRebal(true), CodeNow(0), CodePrev(0), MAval(0), MAOK(true), Ret20(-999), RetOK(true);
SetTotalBar(100);
SetBackBar(100);
SetBackBar(14, "Q");
if BarFreq <> "D" and BarFreq <> "AD" then RaiseRunTimeError("請使用日線盤後選股");
Ret = 0;
// EPS 年增金額 DE(k) = EPS[k] − EPS[k+4]；k = 0 是最新一季，k = 1..8 是過去 8 季（研究中心至少要 6 季，這裡要 8 季都有）
EpsOK = true; RevOK = true; DE0 = 0; SumE = 0; SumE2 = 0; DR0 = 0; SumR = 0; SumR2 = 0;
for Idx = 0 to 8 begin
    EA = GetField("每股稅後淨利(元)", "Q", Default:=-999999999)[Idx];
    EB = GetField("每股稅後淨利(元)", "Q", Default:=-999999999)[Idx + 4];
    RA = GetField("營業收入淨額", "Q", Default:=-999999999)[Idx];
    RB = GetField("營業收入淨額", "Q", Default:=-999999999)[Idx + 4];
    if EA = MissingValue or EB = MissingValue then EpsOK = false;
    if RA = MissingValue or RB = MissingValue then RevOK = false;
    DEk = EA - EB;
    DRk = RA - RB;
    if Idx = 0 then begin
        DE0 = DEk;
        DR0 = DRk;
    end;
    if Idx >= 1 then begin
        SumE = SumE + DEk;
        SumE2 = SumE2 + DEk * DEk;
        SumR = SumR + DRk;
        SumR2 = SumR2 + DRk * DRk;
    end;
end;
// 研究中心的 SUE 分子是 EPS 年增金額本身（不減平均），分母是過去 8 季年增金額的標準差（ddof = 1）
Sue = -999;
if EpsOK = true then begin
    MeanE = SumE / 8;
    VarE = (SumE2 - 8 * MeanE * MeanE) / 7;
    if VarE > 0 then Sue = DE0 / Power(VarE, 0.5);
end;
RevSue = -999;
if RevOK = true then begin
    MeanR = SumR / 8;
    VarR = (SumR2 - 8 * MeanR * MeanR) / 7;
    if VarR > 0 then RevSue = DR0 / Power(VarR, 0.5);
end;
Score = -999;
if Sue > -999 and (RevSue > -999 or B = 0) then begin
    if B = 0 then Score = Sue else Score = Sue - B * RevSue;
end;
// 近 60 日平均成交值（萬元）：成交量(張) × 收盤(元) = 千元，再 ÷ 10 = 萬元
AvgValue = Average(Close * Volume, LiqDays) / 10;
MAval = Average(Close, MALen);
MAOK = true;
if UseMA60 = 1 then begin
    MAOK = false;
    if CurrentBar >= MALen and Close > MAval then MAOK = true;
end;
Ret20 = -999;
if Close[RetLen] > 0 then Ret20 = (Close / Close[RetLen] - 1) * 100;
RetOK = false;
if Ret20 > MinRet20 then RetOK = true;
// 季換股日：日期（月×100＋日）越過 331、515、814、1114 的第一個交易日
IsRebal = true;
if OnlyRebalDay = 1 then begin
    CodeNow = Month(Date) * 100 + DayOfMonth(Date);
    CodePrev = Month(Date[1]) * 100 + DayOfMonth(Date[1]);
    IsRebal = false;
    if CodeNow > 331 and CodePrev <= 331 and CodeNow < 515 then IsRebal = true;
    if CodeNow > 515 and CodePrev <= 515 and CodeNow < 814 then IsRebal = true;
    if CodeNow > 814 and CodePrev <= 814 and CodeNow < 1114 then IsRebal = true;
    if CodeNow > 1114 and CodePrev <= 1114 then IsRebal = true;
end;
// 選出後依 Score 由大到小取前 10 檔（在 XQ 選股結果排序；回測設「最多 10 檔、滿了不買」）
// 出場：下一個季換股日，持股不在新的前 10 名就隔天開盤賣（需在 XQ 確認回測能否設「換股日重新選股、不在名單就賣」；不行就用固定持有到下一季換股日近似）
if IsRebal and Close >= MinPrice
   and AvgValue >= MinAvgValue
   and MAOK
   and RetOK
   and Score > -999 then Ret = 1;
OutputField(1, Score, 3, "季報驚喜近似分數");
OutputField(2, Sue, 2, "SUE");
OutputField(3, RevSue, 2, "REVSUE");
OutputField(4, AvgValue, 0, "60日均成交值(萬)");
OutputField(5, Ret20, 1, "前20日漲幅%");
