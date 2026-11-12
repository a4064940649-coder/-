#!/usr/bin/env python3
"""SPEC v2 小型驗證（只讀，不算報酬）：(1) 兩種外國股名稱來源讀法在面板欄位上的差異；(2) 紙上正式名單是否出現 KY 股；(3) 名單檔欄位。python3 -I -B check_foreign.py"""
import csv, json, gzip
from pathlib import Path
import pyarrow.parquet as pq
AG = Path('/home/user/-/research/tw_stock_selection/agents')
SCR = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad')
cols_paper = [c for c in pq.read_schema(SCR / 'panel/close_raw.parquet').names if c[:1].isdigit()]
cols_v7 = [c for c in pq.read_schema(AG / 'o_v7_listing_exec_engine/data/close_raw.parquet').names if c[:1].isdigit()]
# pclib.NAME：tw_revenue 依 (月, 代碼, sii 優先) 排序去重後，第一次出現的名稱；tw_revenue 沒有的代碼才用 stock_list
rows = []
with open('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv', encoding='utf-8') as f:
    for r in csv.DictReader(f): rows.append((r['month'][:7], r['code'], 0 if r['market'] == 'sii' else 1, r['name']))
rows.sort(key=lambda x: (x[0], x[1], x[2]))
NAME = {}; seen = set()
for ym, c, pri, nm in rows:
    if (ym, c) in seen: continue
    seen.add((ym, c)); NAME.setdefault(c, nm)
tw_all = {}
for ym, c, pri, nm in rows: tw_all.setdefault(c, set()).add(nm)
sl = {}
with open('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv', encoding='utf-8') as f:
    for r in csv.DictReader(f): sl.setdefault(r['code'], r['name'])
for c, nm in sl.items(): NAME.setdefault(c, nm)
SNAP = {}
for fn in ('grs_568e6685_tools_twse_list.csv', 'grs_893c8772_grs_stock_no.csv', 'goristock_c3c718ab_grs_stock_no.csv'):
    for r in csv.reader(open(AG / 'o_pit_shares_industry_2006_2013/raw/industry_snap' / fn, encoding='utf-8', errors='replace')):
        r = [x.strip().strip('"') for x in r]
        if len(r) >= 2 and r[0].isdigit() and len(r[0]) == 4: SNAP.setdefault(r[0], set()).add(r[1])
def nf(n): return ('KY' in n.upper()) or n.endswith('DR') or n.startswith('F-')
def pclib_foreign(c): return c[:2] == '91' or any(nf(n) for n in ({NAME.get(c, '') or ''} | SNAP.get(c, set())))
dl = list(csv.DictReader(open(AG / 'o_infra_v3_rebuild/delist_official.csv', encoding='utf-8')))
fc = set(json.loads((AG / 'o_rev_survivorship/data_freeze.json').read_text(encoding='utf-8'))['foreign_codes'])
V4F = set(fc) | {c for c in cols_v7 if c.startswith('91')} | {r['code'] for r in dl if r['name'].startswith('F-') or 'KY' in r['name'].upper()}
out = {}
P = {c for c in cols_paper if pclib_foreign(c)}; P7 = {c for c in cols_v7 if pclib_foreign(c)}
out['n_cols_paper'] = len(cols_paper); out['n_cols_v7'] = len(cols_v7); out['cols_equal'] = cols_paper == cols_v7
out['n_pclib_foreign_in_paper_cols'] = len(P)
out['n_v4_FOREIGN_in_v7_cols'] = len(V4F & set(cols_v7))
out['v4_FOREIGN_not_pclib_foreign'] = sorted((V4F & set(cols_v7)) - P7)
out['pclib_foreign_not_v4_FOREIGN'] = sorted(P7 - V4F)
# 「任一名稱」另一讀法：加入 delist_official 名稱
PD = {c for c in cols_paper if pclib_foreign(c) or any((r['code'] == c) and (r['name'].startswith('F-') or 'KY' in r['name'].upper()) for r in dl)}
out['delist_reading_extra_foreign_vs_pclib'] = sorted(PD - P)
# 另一讀法：tw_revenue 全部歷史名稱＋stock_list（不論 tw_revenue 是否已有）
PA = {c for c in cols_paper if c[:2] == '91' or any(nf(n) for n in (tw_all.get(c, set()) | {sl.get(c, '') or ''} | SNAP.get(c, set())))}
out['allnames_reading_extra_foreign_vs_pclib'] = sorted(PA - P)
# 紙上正式名單（k_paper_lists）是否含 pclib 判外國或 V4 FOREIGN 的股票
import io
L = list(csv.DictReader(io.TextIOWrapper(gzip.open(AG / 'o_g6_allpit_list_trace_stage2/out/k_paper_lists.csv.gz'), encoding='utf-8')))
out['k_paper_lists_keys'] = sorted({(r['mask'], r['variant']) for r in L}); out['k_paper_lists_n'] = len(L)
hit = {}
for r in L:
    if r['mask'] != 'M1': continue
    for c in r['picks'].split(','):
        if c in V4F or pclib_foreign(c): hit.setdefault(r['variant'], {}).setdefault(c, []).append(r['signal'])
out['M1_paper_picks_foreign_by_either_rule'] = {v: {c: [d[0], d[-1], len(d)] for c, d in h.items()} for v, h in hit.items()}
out['n_rev5y_codes_v4foreign'] = None
r5 = set()
with gzip.open('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras/rev_5y.csv.gz', 'rt', encoding='utf-8') as f:
    for r in csv.DictReader(f): r5.add(r['code'])
out['n_rev5y_codes'] = len(r5); out['n_rev5y_codes_v4foreign'] = len(r5 & V4F); out['n_rev5y_codes_pclib_foreign'] = len({c for c in r5 if pclib_foreign(c)})
out['n_tw_revenue_codes_any_foreign_name'] = len({c for c, s in tw_all.items() if any(nf(n) for n in s)})
Path(__file__).with_name('check_foreign.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(out, ensure_ascii=False, indent=1))
