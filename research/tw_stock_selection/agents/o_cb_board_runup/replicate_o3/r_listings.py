#!/usr/bin/env python3
"""複製員：面板每檔的初次上市分類（自己的規則）。
transfer：上市前在 tw_revenue.csv 出現過 market=otc，或 FSC 案件表在上市前有 ctype=上櫃 的紀錄。
reorg：控股轉換／股份轉換新設（上市日前後 10 日內有其他代碼下市，且名稱含 控/金/投控，或 FSC 有「股份轉換」紀錄）。
dr：代碼 91xx。"""
import json
from pathlib import Path
import pandas as pd, numpy as np
HERE = Path(__file__).resolve().parent
P = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
C = pd.read_parquet(P / 'close_raw.parquet')
L = pd.DataFrame({'first': C.apply(lambda s: s.first_valid_index()), 'last': C.apply(lambda s: s.last_valid_index())})
L.index.name = 'code'
R = pd.read_csv('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv', dtype=str)
R['month'] = pd.to_datetime(R['month'])
names = R.sort_values('month').drop_duplicates('code', keep='last').set_index('code')['name']
SL = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv', dtype=str).drop_duplicates('code').set_index('code')
F = pd.read_csv(HERE.parent.parent / 'o_issuance_avoid/fsc_cases_all.csv', dtype=str)
F['recv'] = pd.to_datetime(F['recv'], errors='coerce')
L['name'] = [SL['name'].get(c) if c in SL.index else names.get(c, '') for c in L.index]
otc_first = R[R.market == 'otc'].groupby('code')['month'].min()
otc_last = R[R.market == 'otc'].groupby('code')['month'].max()
fsc_otc = F[F.ctype == '上櫃'].groupby('code')['recv'].min()
fsc_emg = F[F.ctype == '興櫃'].groupby('code')['recv'].min()
L['new'] = L['first'] >= pd.Timestamp('2004-06-01')
L['otc_rev_first'] = otc_first.reindex(L.index); L['fsc_otc_first'] = fsc_otc.reindex(L.index); L['fsc_emg_first'] = fsc_emg.reindex(L.index)
L['transfer'] = L['new'] & ((L['otc_rev_first'] < L['first']) | (L['fsc_otc_first'] < L['first']))
# 控股／股份轉換新設
item = F['item'].fillna('').str.replace(r'\s+', '', regex=True)
swap_codes = set(F.loc[item.str.contains('股份轉換|轉換他公司股份|受讓他公司股份'), 'code'])
reorg = {}
for c, r in L[L['new']].iterrows():
    nm = str(r['name'])
    if 'KY' in nm: reorg[c] = False; continue
    near = L[(L.index != c) & (L['last'] <= r['first']) & (L['last'] >= r['first'] - pd.Timedelta(days=30))]
    hold = ('控' in nm) or nm.endswith('金') or c.startswith('37')
    reorg[c] = bool(hold and (len(near) > 0 or '控' in nm or c.startswith('37')))
L['reorg'] = pd.Series(reorg).reindex(L.index).fillna(False).astype(bool)
L['dr'] = L.index.str.startswith('91')
L['kind'] = np.select([~L['new'], L['dr'], L['reorg'], L['transfer']], ['old', 'dr', 'reorg', 'transfer'], 'ipo')
L.to_csv(HERE / 'my_listings.csv')
print(L['kind'].value_counts().to_dict())
print(L[L.kind == 'reorg'][['first', 'name']].to_string())
