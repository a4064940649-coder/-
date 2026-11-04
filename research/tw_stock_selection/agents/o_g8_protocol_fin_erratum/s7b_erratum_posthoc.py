#!/usr/bin/env python3
"""題 B 事後補充（ERRATUM_SPEC 沒有列，標「事後」）：C2 多判為金融、C0 沒排除的股票（1219、5854、5871、6004、6012）
在 C0 名單（2006～2013）的股池月數、名次與進前 10 名月份；附各股上市／下市日（listing_master）。只計數，不算報酬。
用法：python3 -I -B s7b_erratum_posthoc.py"""
import sys, json, csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import g8lib as G
G.check_reg('ERRATUM_SPEC', files=False)
AG = G.AG
sys.path.insert(0, str(AG / 'main_v2'))
import numpy as np, pandas as pd
import v2_core as V
sys.path.pop(0)
K = V.K
COLS = list(K.COLS)
E = json.loads((G.HERE / 'ERRATUM.json').read_text(encoding='utf-8'))
codes = E['C2_extra_vs_C0']
YL, Y5, _ = V.load_revenue()
S = V.V2('gapfix', YL, Y5, exfin=True)
pl = S.plan(10, 'official')
LM = {r['code']: r for r in csv.DictReader(open(AG / 'o_v7_listing_exec_engine/listing_master.csv', encoding='utf-8'))}
out = {}
for c in codes:
    rows = []
    for e, pk, el, d in pl:
        if not (pd.Timestamp('2006-01-01') <= d <= pd.Timestamp('2013-12-31')): continue
        cs = S.cross_section(d); od = [COLS[j] for j in S.order(cs, 'official')[:int(cs['el'].sum())]]
        if c in od:
            rows.append(dict(signal=str(d.date()), rank=od.index(c) + 1, in_top10=c in [COLS[j] for j in pk]))
    rk = [r['rank'] for r in rows]
    out[c] = dict(listed=LM.get(c, {}).get('listed_date'), delisted=LM.get(c, {}).get('delist_date'), n_in_pool=len(rows),
                  rank_median=float(np.median(rk)) if rk else None, rank_min=min(rk) if rk else None,
                  top10=[(r['signal'], r['rank']) for r in rows if r['in_top10']])
res = dict(utc=G.now_utc(), label='事後補充（未事前登記）', stocks=out,
           note='6012 金鼎證、6004 元京證為證券商（當時證交所產業別應屬金融保險業，但兩者在最早 dated 快照 2011-07-28 前已下市，無 PIT 依據）；5871 中租-KY 為租賃業（證交所其他業）；1219 福壽為食品業（C2 名稱規則「壽」結尾誤配）')
E['posthoc_c2_only'] = res
(G.HERE / 'ERRATUM.json').write_text(json.dumps(E, ensure_ascii=False, indent=1, default=G._js), encoding='utf-8')
r = G.load_res(); r['ERRATUM']['posthoc_c2_only'] = res
G.save_key('ERRATUM', r['ERRATUM'])
print(json.dumps(res, ensure_ascii=False, default=G._js))
