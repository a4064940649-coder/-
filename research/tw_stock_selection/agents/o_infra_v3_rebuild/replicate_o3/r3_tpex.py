#!/usr/bin/env python3
"""上櫃：henry753951 原始檔中，最後成交日在 2019-06～2022-11、而且不在上市面板的代碼（＝上櫃下市候選），是否已在 tpex_main 面板內、面板最後一筆是否相同。"""
import json
from pathlib import Path
import pandas as pd, numpy as np
R = Path('/home/user/-/research/tw_stock_selection/agents/o_infra_v3_rebuild/raw/henry753951_finance/data/stocks')
TP = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/tpex_panel')
C0 = pd.read_parquet('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel/close_raw.parquet')
TC = pd.read_parquet(TP / 'close_raw.parquet')
SL = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv', dtype=str).drop_duplicates('code').set_index('code')
rows = []
for f in sorted(R.glob('*.csv')):
    c = f.stem
    if len(c) != 4 or c in C0.columns: continue
    r = pd.read_csv(f, usecols=['date', 'Trading_Volume'], parse_dates=['date'])
    r = r[r['Trading_Volume'] > 0]
    if r.empty: continue
    last = r['date'].max()
    if pd.Timestamp('2019-06-01') <= last <= pd.Timestamp('2022-11-30'):
        inp = c in TC.columns
        rows.append({'code': c, 'raw_last': str(last.date()), 'in_tpex_panel': inp,
                     'tpex_last': str(TC[c].last_valid_index().date()) if inp and TC[c].notna().any() else None,
                     'still_listed': c in SL.index})
df = pd.DataFrame(rows)
print(df.to_string()); print('n', len(df), 'in tpex', int(df['in_tpex_panel'].sum()) if len(df) else 0)
Path(__file__).with_name('my_tpex.json').write_text(df.to_json(orient='records', force_ascii=False))
