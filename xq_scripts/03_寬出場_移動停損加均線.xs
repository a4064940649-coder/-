// ===== 03 寬出場：移動停損 15% 或 8 日線跌破 20 日線（出場腳本）=====
// 搭配 02（CAN SLIM 進場）或 05（強勢股池）使用。隔天開盤賣出。
// 原貼文的出場是回落 6%，太緊：平均只抱 8 天、31% 的交易隔天就賣。這裡把 6% 放寬成 15%，其他不變。
// 均線剛好相等不算跌破（用 crosses under）。
input: TrailPct(15);  setinputname(1, "移動停損%");
input: TrailDays(20); setinputname(2, "最高收盤觀察天數");
input: FastMA(8);     setinputname(3, "短均線");
input: SlowMA(20);    setinputname(4, "長均線");
input: UseMA(1);      setinputname(5, "啟用均線跌破出場，1=啟用");
settotalbar(SlowMA + TrailDays + 3);
if Close <= Highest(Close, TrailDays) * (1 - TrailPct / 100) then ret = 1;
if UseMA = 1 and Average(Close, FastMA) crosses under Average(Close, SlowMA) then ret = 1;
