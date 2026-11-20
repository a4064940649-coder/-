// ===== 07 VCP 出場：收盤跌破 50 日均線（出場腳本）=====
// 搭配 06 使用。另外請在回測設定裡加「停損 8%」（以進場價計）。隔天開盤賣出。
input: MALen(50); setinputname(1, "均線天數");
settotalbar(MALen + 3);
if Close < Average(Close, MALen) then ret = 1;
