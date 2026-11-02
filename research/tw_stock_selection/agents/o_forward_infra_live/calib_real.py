#!/usr/bin/env python3
"""V7X κ 前瞻校準（o_exec_engine_rewrite_v7x/s8_forward_calib.py 的 update()，原檔只讀）只用實際成交：trades.csv 的 kind＝real。
紙上成交（kind＝paper_*）的價格就是 K5T 模型本身，放進去會把後驗拉回先驗，所以一律排除。
使用者真錢下單時，把每筆成交寫進 live_tracking/main_forward/fills_input.csv（欄位 date,code,side,order_time,limit_px,fill_px,shares,is_odd,list），
下一次 make_forward.py 會自動補上開盤價、距開盤跳數、占開盤量比例（事前 OA＝share_dec×V60_{t−1}）。"""
import importlib.util, json
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
spec = importlib.util.spec_from_file_location('s8', AG / 'o_exec_engine_rewrite_v7x/s8_forward_calib.py')
s8 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s8)
T = pd.read_csv(AG / 'live_tracking/main_forward/trades.csv', dtype={'code': str})
real = T[T['kind'] == 'real'].copy()
real['is_odd'] = real['is_odd'].astype(str)
out = s8.update(real) if len(real) else dict(status='no_real_fills', n_nonodd=0)
print(json.dumps(out, ensure_ascii=False, indent=1))
