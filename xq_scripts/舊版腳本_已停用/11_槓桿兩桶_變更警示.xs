// ===== 11 槓桿兩桶：目標比例變更警示（警示腳本）v1.0 =====
// 請套用在「加權指數」的日線上。目標比例和前一天不同時跳警示。規則同 10。未在 XQ 編譯。
input: HighWin(252);  setinputname(1, "回落計算的高點天數");
input: MALen(200);    setinputname(2, "均線天數");
input: Buf(3);        setinputname(3, "均線上下緩衝(%)");
input: DD1(20);       setinputname(4, "第一階回落(%)");
input: W1(60);        setinputname(5, "第一階逆勢桶換成00631L的比例(%)");
input: DD2(30);       setinputname(6, "第二階回落(%)");
input: W2(100);       setinputname(7, "第二階逆勢桶換成00631L的比例(%)");
input: Bucket(50);    setinputname(8, "逆勢桶占總資金(%)");
variable: _hi(0), _dd(0), _lvl(0), _on(1), _ma(0), _aL(0), _wL(0), _w50(0), _cash(0);
settotalbar(1500);
if BarFreq <> "D" then RaiseRunTimeError("請使用日線");
_hi = Highest(Close, HighWin);
if _hi > 0 then _dd = (Close / _hi - 1) * 100;
if _dd >= 0 then _lvl = 0;
if _dd <= -DD1 and _lvl < 1 then _lvl = 1;
if _dd <= -DD2 and _lvl < 2 then _lvl = 2;
if CurrentBar >= MALen then begin
 _ma = Average(Close, MALen);
 if Close > _ma * (1 + Buf / 100) then _on = 1;
 if Close < _ma * (1 - Buf / 100) then _on = 0;
end;
_aL = 0;
if _lvl = 1 then _aL = W1;
if _lvl = 2 then _aL = W2;
_wL = Bucket / 100 * _aL + (1 - Bucket / 100) * _on * 100;
_w50 = Bucket / 100 * (100 - _aL);
_cash = 100 - _wL - _w50;
if CurrentBar > MALen + 1 and (_wL <> _wL[1] or _w50 <> _w50[1]) then begin
 ret = 1;
 retmsg = "目標比例變更 00631L:" + NumToStr(_wL, 0) + "% 0050:" + NumToStr(_w50, 0)
 + "% 現金:" + NumToStr(_cash, 0) + "%";
end;
