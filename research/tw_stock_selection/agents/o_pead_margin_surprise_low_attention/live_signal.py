#!/usr/bin/env python3
"""規則書的操作範例（紙上追蹤用；不是推薦）：列出最近一季已可得（A 日已過）的 MAIN 候選（SUE⊥ ≥ 前期第 80 百分位、非 0050、股池內），
以及每檔的建議進場日（A 的下一個交易日開盤）與到期出場日（第 60 個交易日開盤）。
用法：python3 -I -B live_signal.py [YYYY-MM-DD 截止日，預設資料最後一天]"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from pcore import DATA, load_panel

P = load_panel(); IDX = pd.DatetimeIndex(P['IDX'])
asof = pd.Timestamp(sys.argv[1]) if len(sys.argv) > 1 else IDX[-1]
E = pd.read_parquet(DATA / 'events.parquet')
E = E[(E['A'] <= asof) & E['SUEP'].notna()]
k = E['k'].max()
c = E[(E['k'] == k) & E['top_SUEP'] & ~E['in0050']].sort_values('A')
c['entry'] = [IDX[min(a + 1, len(IDX) - 1)] for a in c['ai']]
c['exit_due'] = [IDX[a + 61] if a + 61 < len(IDX) else pd.NaT for a in c['ai']]
c['FY_Q'] = [f'{x // 4}Q{x % 4 + 1}' for x in c['k']]
print(f'截止 {asof.date()}，財報季 {k // 4}Q{k % 4 + 1}，候選 {len(c)} 檔（20 槽依公告先後、同日依 SUE⊥ 大到小）')
print(c[['code', 'mkt', 'FY_Q', 'A', 'A_method', 'entry', 'exit_due', 'SUE', 'REVSUE', 'SUEP', 'V60']].head(40).to_string(index=False))
