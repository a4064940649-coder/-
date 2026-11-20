// ===== 10 槓桿兩桶：00631L／0050 目標比例（指標腳本）v1.0 =====
// 請套用在「加權指數」的日線上。畫出今天收盤後的目標持有比例：00631L、0050、現金。
// 規則（來源：公開專案 zheng2-signal，研究中心已獨立重算過）：
//   資金分兩桶，各一半。
//   逆勢桶：平常持有 0050。加權指數從近 252 日最高收盤回落 20% → 這一桶的 60% 換成 00631L；
//           回落 30% → 這一桶全部換成 00631L；指數再創 252 日新高 → 換回 0050。
//   順勢桶：加權指數收盤高於 200 日均線 3% 以上 → 持有 00631L；低於 200 日均線 3% 以上 → 全部現金；
//           在上下 3% 之間維持原狀。
// 執行：今天收盤出訊號，下一個交易日收盤前調整到目標比例；平常比例偏離目標超過 10 個百分點才調整。
// 未在 XQ 編譯。
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
Plot1(_wL, "00631L目標%");
Plot2(_w50, "0050目標%");
Plot3(_cash, "現金目標%");
Plot4(_dd, "距252日高%");
