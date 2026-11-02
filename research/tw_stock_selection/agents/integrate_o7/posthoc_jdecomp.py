#!/usr/bin/env python3
"""integrate_o7 事後補充（非事前登記，不參與任何判定）：把「官方日零衝擊」與「可執行 j 中位數」在 2019 年後的落差拆成
(a) 官方日零衝擊 → (b) j＝0～12 零衝擊（ideal base）中位數 → (c) j 零衝擊＋處置期賣出＋併購延遲（ideal both）中位數 → (d) j 可執行（30 萬 FIX，K5T／TICK）中位數。
只讀 V7 的 navs_ideal7.parquet 與本資料夾 data/ 副本；寫進 results.json 的 POSTHOC_jdecomp 鍵。"""
import sys, json, hashlib
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent; V7D = AG / 'o_v7_listing_exec_engine'
sys.path.insert(0, str(V7D)); import v7stats as ST; sys.path.pop(0)
RJ = HERE / 'results.json'; RES = json.loads(RJ.read_text(encoding='utf-8'))
assert 'RESULTS' in RES and 'POSTHOC_jdecomp' not in RES
NI = pd.read_parquet(V7D / 'navs_ideal7.parquet'); NE = pd.read_parquet(HERE / 'data/navs_exec7.parquet'); NV = pd.read_parquet(HERE / 'data/navs_v7.parquet')
E = pd.read_csv(HERE / 'data/etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()
PER = {'ALL': ('2006-04-01', '2026-10-22'), 'SEL': ('2006-04-01', '2018-12-31'), 'Y19on': ('2019-01-01', '2026-10-22'), 'P19_22': ('2019-01-01', '2022-12-31'),
       'P23_26': ('2023-01-01', '2026-10-22'), 'HOLD': ('2024-01-01', '2026-10-22')}
def cg(nav, p):
    a, b = PER[p]; b = min(pd.Timestamp(b), nav.dropna().index[-1]); return ST.cagr_mdd(nav.dropna(), a, b)[0]
def e50(nav, p):
    ix = nav.dropna().index; return cg(E.reindex(ix.union(E.index)).ffill().reindex(ix), p)
out = {'note': '事後補充，非事前登記；不參與任何判定。', 'sha_navs_ideal7': hashlib.sha256((V7D / 'navs_ideal7.parquet').read_bytes()).hexdigest()}
for st, nvc in (('AB', 'AB_V7|normal'), ('B10', 'AB_V7_B10|normal')):
    o = {}
    for p in PER:
        row = {'official_zero_impact': cg(NV[nvc], p), 'official_0050': e50(NV[nvc], p)}
        for lab, fmt in (('j_ideal_base', '{st}|J{j}|base'), ('j_ideal_both', '{st}|J{j}|both'), ('j_exec_K5T_30萬FIX', '{st}|J{j}|K5T|normal|30萬|FIX'), ('j_exec_TICK_30萬FIX', '{st}|J{j}|TICK|normal|30萬|FIX')):
            src = NI if 'ideal' in lab else NE
            v = [cg(src[fmt.format(st=st, j=j)], p) for j in range(13)]
            row[lab + '_median'] = float(np.median(v)); row[lab + '_min'] = float(np.min(v)); row[lab + '_max'] = float(np.max(v))
            if lab == 'j_ideal_base': row['j_ideal_base_by_j'] = v
        row['j_0050_median'] = float(np.median([e50(NE[f'{st}|J{j}|K5T|normal|30萬|FIX'], p) for j in range(13)]))
        o[p] = row
    out[st] = o
RES['POSTHOC_jdecomp'] = out
RJ.write_text(json.dumps(RES, ensure_ascii=False, indent=1), encoding='utf-8')
for st in ('AB', 'B10'):
    for p, r in out[st].items():
        print(st, p, {k: (round(v, 2) if isinstance(v, float) else [round(x, 1) for x in v]) for k, v in r.items()})
