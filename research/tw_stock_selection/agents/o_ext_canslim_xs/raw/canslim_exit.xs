// CAN SLIM 出場：收盤從近期最高收盤回落6%，或8MA跌破20MA
input: TrailPct(6);   setinputname(1, "移動停損%");
input: TrailDays(20); setinputname(2, "最高收盤觀察天數");
input: FastMA(8);     setinputname(3, "短均線");
input: SlowMA(20);    setinputname(4, "長均線");
settotalbar(SlowMA + TrailDays + 3);
if Close <= Highest(Close, TrailDays) * (1 - TrailPct / 100) then ret = 1;
if Average(Close, FastMA) crosses under Average(Close, SlowMA) then ret = 1;
