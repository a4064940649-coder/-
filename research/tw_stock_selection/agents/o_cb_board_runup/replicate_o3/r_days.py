#!/usr/bin/env python3
"""複製員：董事會→掛牌交易日數（tpex_cb_listings.csv 的掛牌日；以 FSC 對上的事件為準）。"""
import json
from pathlib import Path
import pandas as pd, numpy as np
HERE = Path(__file__).resolve().parent
x = pd.read_csv(HERE / 'my_fsc_match.csv', dtype={'code': str}, parse_dates=['recv', 'board', 'eff'])
T = pd.read_csv(HERE.parent.parent / 'o_issuance_avoid/tpex_cb_listings.csv', dtype={'issuer': str, 'cb_code': str}, parse_dates=['list_date'])
cal = pd.read_parquet('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel/close_raw.parquet', columns=['2330']).index
x = x[x.board.notna()]
def lst(r):
    g = T[(T.issuer == r.code) & (T.list_date > r.recv) & (T.list_date <= r.recv + pd.Timedelta(days=200))]
    return g.list_date.min() if len(g) else pd.NaT
x['list'] = [lst(r) for r in x.itertuples()]
y = x[x['list'].notna()]
td = [int(((cal > b) & (cal <= l)).sum()) for b, l in zip(y.board, y['list'])]
out = {'board_to_list_td_median': float(np.median(td)), 'n': len(td), 'board_to_recv_days_median': float((x.recv - x.board).dt.days.median()), 'n_b2r': len(x)}
print(json.dumps(out))
