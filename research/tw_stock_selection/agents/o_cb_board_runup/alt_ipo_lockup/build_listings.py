# -*- coding: utf-8 -*-
"""替代題資料：上市日、初次上市／上櫃轉上市標記、控股公司轉換標記（不計算任何報酬）。
上市日＝面板第一個交易日（2004-06-01 以後的才算新上市；之前的視為 2004 年以前上市的老股）。
轉上市：tw_revenue 在上市月之前（含 ≥ 2 個月前）有 market=otc 的營收列，或 FSC 申報案件中上市日之前有 ctype=上櫃 的案件。
控股公司轉換（非新股供給）：代碼 3701～3719、名稱含「控」、或 o_infra_v3_rebuild/merger_consideration.csv 的存續公司 → reorg。
DR（91xx）另標，分析時排除。"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
AG = HERE.parent.parent
P = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
C = pd.read_parquet(P / 'close_raw.parquet')
first = C.apply(lambda s: s.first_valid_index())
last = C.apply(lambda s: s.last_valid_index())
SL = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv', dtype=str).drop_duplicates('code').set_index('code')
REV = pd.read_csv('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv', dtype={'code': str}, parse_dates=['month'])
otc = REV[REV.market == 'otc'].groupby('code').month.min()
FSC = pd.read_csv(AG / 'o_issuance_avoid' / 'fsc_cases_all.csv', dtype={'code': str}, parse_dates=['recv'])
fsc_otc = FSC[FSC.ctype == '上櫃'].groupby('code').recv.min()
fsc_em = FSC[FSC.ctype.isin(['興櫃', '未上市櫃', '公發', '補辦公發'])].groupby('code').recv.min()
M = pd.read_csv(AG / 'o_infra_v3_rebuild' / 'merger_consideration.csv', dtype=str)
succ = set(M['successor'].dropna())
# 面板中名稱（stock_list 只有現存股；下市股用 delist_official 名稱）
DL = pd.read_csv(AG / 'o_infra_v3_rebuild' / 'delist_official.csv', dtype=str).drop_duplicates('code', keep='last').set_index('code')
rows = []
for c in C.columns:
    f = first[c]
    if f is None or pd.isna(f): continue
    name = SL['name'].get(c) if c in SL.index else (DL['name'].get(c) if c in DL.index else None)
    new = f >= pd.Timestamp('2004-06-01')
    o = otc.get(c); fo = fsc_otc.get(c)
    transfer = bool(new and ((o is not None and pd.notna(o) and o <= f - pd.Timedelta(days=45)) or (fo is not None and pd.notna(fo) and fo < f)))
    reorg = bool(new and ((c[:2] == '37' and '3701' <= c <= '3719') or (isinstance(name, str) and '控' in name) or c in succ))
    dr = c[:2] == '91'
    rows.append(dict(code=c, name=name, first=f, last=last[c], new=bool(new), transfer=transfer, reorg=reorg, dr=dr,
                     industry=SL['industry'].get(c) if c in SL.index else None,
                     otc_rev_first=o, fsc_otc_first=fo, fsc_emerging_first=fsc_em.get(c)))
D = pd.DataFrame(rows)
D['kind'] = np.where(~D.new, 'old', np.where(D.dr, 'dr', np.where(D.reorg, 'reorg', np.where(D.transfer, 'transfer', 'ipo'))))
D.to_csv(HERE / 'listings.csv', index=False)
nw = D[D.new]
summ = dict(n_codes=len(D), kind=D.kind.value_counts().to_dict(),
            new_by_year=nw.groupby([nw['first'].dt.year, 'kind']).size().unstack(fill_value=0).to_dict('index'),
            delisted_new_by_year=nw[nw['last'] < pd.Timestamp('2026-10-01')].groupby(nw['first'].dt.year).size().to_dict(),
            ipo_with_emerging_record=int(((D.kind == 'ipo') & D.fsc_emerging_first.notna()).sum()))
(HERE / 'listings_summary.json').write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps(summ, ensure_ascii=False, indent=1, default=str))
