{@type:sensor}
{VCP 突破警示：趨勢樣板 + 波動逐段收縮後，收盤向上穿越樞紐點且放量
  樞紐點 = 末段（不含當日）最高價。量縮看突破前一日為止的 10/50 日均量，避免當日放量汙染。
  相對強度：官方文件沒有 IBD 式 RS Rank 欄位；啟用時改以個股 250 日漲幅對比加權指數 TSE.TW。
  Threads 影片中 J Law 的額外條件無法取得，本腳本未納入。
  請用日線執行。}
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
variable: _avgVol10(0), _avgVol50(0), _volRatio(0);
variable: _pivot(0);
variable: _idxClose(0), _idxROC(0), _stkROC(0);
variable: _trend(false), _contract(false), _volDry(false), _rsOK(true);
settotalbar(400);
setbackbar(400);
if BarFreq <> "D" then
 RaiseRunTimeError("VCP 突破警示請使用日線");
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
 {量縮以上一筆為止計算，突破當日放量不計入}
 _avgVol10 = Average(Volume[1], 10);
 _avgVol50 = Average(Volume[1], 50);
 if _avgVol50 > 0 then _volRatio = _avgVol10 / _avgVol50;
 _volDry = _avgVol50 > 0 and _volRatio < VolDryRatio;
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
 and (MinVol = 0 or Volume >= MinVol) then begin
 ret = 1;
 retmsg = "VCP放量突破 樞紐:" + NumToStr(_pivot, 2)
 + " 末段收縮:" + NumToStr(_d3, 2) + "%"
 + " 量比:" + NumToStr(Volume / _avgVol50, 2);
 end;
end;
