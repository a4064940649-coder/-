// ===== 06 VCP 突破（選股腳本，進場）=====
// 把貼文的 VCP 警示腳本改成選股腳本，條件與預設值照原文，方便回測。
// 趨勢樣板 + 三段波動逐段收縮 + 量縮 + 收盤向上穿越樞紐且放量。
// 用法：選股腳本；日線（原始價）盤後執行；符合就隔天開盤買；同時最多 10 檔；
//       訊號多於空位時依輸出欄位 3「量比」由大到小。出場搭配 07，另外在回測設定裡加停損 8%。
// 研究中心回測：22 年只有 491 個訊號，資金多半閒置，不建議單獨使用。
input: SegLen(25); setinputname(1, "每段天數（三段合計為基底長度）");
input: MaxLastDepth(10); setinputname(2, "末段收縮上限(%)");
input: MinFirstDepth(10); setinputname(3, "首段收縮下限(%)");
input: VolDryRatio(0.7); setinputname(4, "量縮比(10日/50日均量)");
input: VolBreak(1.5); setinputname(5, "突破量能倍數(相對50日均量)");
input: En_RS(1); setinputname(6, "啟用相對強度濾網，1=啟用");
input: RSMinDiff(0); setinputname(7, "相對大盤250日漲幅差(%)");
input: MinVol(500); setinputname(8, "最低成交量(張)，0為不限");
input: LowMult(1.3); setinputname(9, "收盤相對52週低點最低倍數");
input: HighPct(25); setinputname(10, "收盤距52週高點最大幅度(%)");
input: MA200UpBars(20); setinputname(11, "MA200上揚比較天數");
variable: _ma50(0), _ma150(0), _ma200(0), _low52(0), _high52(0);
variable: _h1(0), _l1(0), _h2(0), _l2(0), _h3(0), _l3(0);
variable: _d1(0), _d2(0), _d3(0);
variable: _avgVol10(0), _avgVol50(0), _volRatio(0), _volMult(0);
variable: _pivot(0);
variable: _idxClose(0), _idxROC(0), _stkROC(0);
variable: _trend(false), _contract(false), _volDry(false), _rsOK(true);
settotalbar(400);
setbackbar(400);
if BarFreq <> "D" then RaiseRunTimeError("VCP 突破請使用日線");
ret = 0;
if CurrentBar >= 260 and CurrentBar > 3 * SegLen + 1 then begin
 _ma50 = Average(Close, 50);
 _ma150 = Average(Close, 150);
 _ma200 = Average(Close, 200);
 _low52 = Lowest(Low, 250);
 _high52 = Highest(High, 250);
 _trend = Close > _ma150 and Close > _ma200
 and _ma150 > _ma200
 and _ma200 > _ma200[MA200UpBars]
 and _ma50 > _ma150 and _ma50 > _ma200
 and Close > _ma50
 and _low52 > 0 and Close >= _low52 * LowMult
 and _high52 > 0 and Close >= _high52 * (1 - HighPct / 100);
 _h3 = Highest(High[1], SegLen);
 _l3 = Lowest(Low[1], SegLen);
 _h2 = Highest(High[SegLen + 1], SegLen);
 _l2 = Lowest(Low[SegLen + 1], SegLen);
 _h1 = Highest(High[2 * SegLen + 1], SegLen);
 _l1 = Lowest(Low[2 * SegLen + 1], SegLen);
 if _h1 > 0 then _d1 = (_h1 - _l1) / _h1 * 100;
 if _h2 > 0 then _d2 = (_h2 - _l2) / _h2 * 100;
 if _h3 > 0 then _d3 = (_h3 - _l3) / _h3 * 100;
 _contract = _d1 >= MinFirstDepth and _d1 > _d2 and _d2 > _d3 and _d3 <= MaxLastDepth;
 _avgVol10 = Average(Volume[1], 10);
 _avgVol50 = Average(Volume[1], 50);
 if _avgVol50 > 0 then _volRatio = _avgVol10 / _avgVol50;
 _volDry = _avgVol50 > 0 and _volRatio < VolDryRatio;
 if _avgVol50 > 0 then _volMult = Volume / _avgVol50;
 _pivot = _h3;
 _rsOK = true;
 if En_RS = 1 then begin
  if CheckSymbolField("TSE.TW", "收盤價", "D") then begin
   _idxClose = GetSymbolField("TSE.TW", "收盤價", "D");
   _idxROC = RateOfChange(_idxClose, 250);
   _stkROC = RateOfChange(Close, 250);
   if _stkROC < _idxROC + RSMinDiff then _rsOK = false;
  end;
 end;
 if _trend and _contract and _volDry and _rsOK
 and _pivot > 0 and Close crosses over _pivot
 and Volume > _avgVol50 * VolBreak
 and (MinVol = 0 or Volume >= MinVol) then ret = 1;
end;
OutputField(1, _pivot, 2, "樞紐價");
OutputField(2, _d3, 2, "末段收縮%");
OutputField(3, _volMult, 2, "量比");
