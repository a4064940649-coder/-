// ===== 15 月營收驚奇 SURGE：下次營收公布時未達標就賣（出場腳本）v1.0 =====
// 搭配 14。持股的新月份營收公布那天，SURGE < ExitSurge（或算不出來）就出場，隔天開盤賣；達標就續抱。
// 月中不出場、不停損。未在 XQ 編譯，也未在 XQ 回測。
Input: ExitSurge(2.8);
Var: MissingValue(-999999999), Idx(0), Idx2(0), RevA(0), RevB(0), RevOK(true),
     D0(0), Dk(0), SumD(0), SumD2(0), MeanD(0), VarD(0), Surge(-999),
     RevNow(0), NewRev(false);
SetTotalBar(100);
SetBackBar(100);
SetBackBar(40, "M");
if BarFreq <> "D" and BarFreq <> "AD" then RaiseRunTimeError("請使用日線");
Ret = 0;
RevNow = GetField("月營收", "M", Default:=-999999999);
NewRev = false;
if CurrentBar > 1 and RevNow <> MissingValue and RevNow <> RevNow[1] then NewRev = true;
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
if NewRev and Surge < ExitSurge then Ret = 1;
