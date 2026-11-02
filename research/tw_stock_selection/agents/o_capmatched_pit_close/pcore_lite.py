#!/usr/bin/env python3
"""輕量常數（不載入面板數值）：股票代碼順序與產業。"""
from pathlib import Path
import pandas as pd
O7 = Path(__file__).resolve().parent.parent / 'o_capmatched_selection'
COLS = list(pd.read_parquet(O7 / 'raw' / 'panel' / 'close_raw.parquet').columns)
CI = {c: j for j, c in enumerate(COLS)}
_SL = pd.read_csv(O7 / 'raw' / 'screener' / 'stock_list.csv', dtype=str).drop_duplicates('code').set_index('code')
IND = _SL['industry'].to_dict()
NAME = _SL['name'].to_dict()
