#!/usr/bin/env python3
"""SPEC v2 小型驗證（只讀）：外國股名稱來源逐檔核對。python3 -I -B check_names.py"""
import csv, json
from pathlib import Path
AG = Path('/home/user/-/research/tw_stock_selection/agents')
codes = ['3697', '3638', '4144', '2311', '6286', '3561', '1262', '1592', '6422', '6452', '2325', '1704']
tw = {}
with open('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv', encoding='utf-8') as f:
    for r in csv.DictReader(f): tw.setdefault(r['code'], set()).add(r['name'])
sl = {}
with open('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv', encoding='utf-8') as f:
    for r in csv.DictReader(f): sl.setdefault(r['code'], r['name'])
snap = {}
for fn in ('grs_568e6685_tools_twse_list.csv', 'grs_893c8772_grs_stock_no.csv', 'goristock_c3c718ab_grs_stock_no.csv'):
    p = AG / 'o_pit_shares_industry_2006_2013/raw/industry_snap' / fn
    for r in csv.reader(open(p, encoding='utf-8', errors='replace')):
        r = [x.strip().strip('"') for x in r]
        if len(r) >= 2 and r[0].isdigit() and len(r[0]) == 4: snap.setdefault(r[0], set()).add(r[1])
dl = {}
with open(AG / 'o_infra_v3_rebuild/delist_official.csv', encoding='utf-8') as f:
    for r in csv.DictReader(f): dl.setdefault(r['code'], r['name'])
fc = set(json.loads((AG / 'o_rev_survivorship/data_freeze.json').read_text(encoding='utf-8'))['foreign_codes'])
def isf(n): return ('KY' in n.upper()) or n.endswith('DR') or n.startswith('F-')
out = {}
for c in codes:
    nm_pclib = ({sorted(tw[c])[0]} if False else set())  # placeholder
    # pclib.NAME：tw_revenue 第一次出現的名稱（去重後檔案順序），沒有才用 stock_list
    out[c] = dict(tw_revenue_names=sorted(tw.get(c, [])), stock_list=sl.get(c), snap=sorted(snap.get(c, [])), delist_official=dl.get(c),
                  in_o_rev_survivorship_foreign_codes=c in fc,
                  any_name_foreign_excl_delist=any(isf(n) for n in list(tw.get(c, [])) + [sl.get(c) or ''] + list(snap.get(c, []))),
                  delist_name_foreign=bool(dl.get(c)) and (dl[c].startswith('F-') or 'KY' in dl[c].upper()))
print(json.dumps(out, ensure_ascii=False, indent=1))
print('n foreign_codes', len(fc), 'n tw codes', len(tw), 'n snap', len(snap), 'n stock_list', len(sl))
