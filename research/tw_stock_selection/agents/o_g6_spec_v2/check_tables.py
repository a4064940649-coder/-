#!/usr/bin/env python3
"""SPEC v2 小型驗證（只讀，不算報酬）：補回表、處置清單、事件表的列數與日期範圍；V7X M2 名單差異檔是否含 KY 六檔。python3 -I -B check_tables.py"""
import json, csv, collections
from pathlib import Path
import pandas as pd
AG = Path('/home/user/-/research/tw_stock_selection/agents')
X = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras')
o = {}
d = pd.read_csv(AG / 'integrate_o20/data/v7x_plan_diff_M2.csv.gz', dtype=str); allp = set()
for col in ('picks_b', 'picks_m'):
    for s in d[col].dropna(): allp |= set(s.split(','))
o['v7x_plan_diff_M2'] = dict(rows=len(d), ky6_in_picks=sorted(allp & {'1262', '1592', '2936', '6422', '6452', '8480'}), signal_min=d.signal.min(), signal_max=d.signal.max())
rb = pd.read_csv(AG / 'o_delist_rev_backfill_allpit/data/rev_backfill_B1.csv', dtype={'code': str})
o['rev_backfill_B1'] = dict(rows=len(rb), yoy_finite=int(rb.yoy_pct.notna().sum()), codes=int(rb.code.nunique()), ym_min=rb.ym.min(), ym_max=rb.ym.max())
u = pd.read_csv(AG / 'o_disposal_list_pit_audit_pkl18/data/universe_announce.csv', dtype={'code': str})
o['universe_announce'] = dict(rows=len(u), announce_min=u.announce.min(), announce_max=u.announce.max(), announce_na=int(u.announce.isna().sum()))
du = pd.read_csv(AG / 'o_rev_avail_ds_pit_count/raw/disposal_official_universe.csv', dtype={'code': str}); o['disposal_official_universe'] = dict(rows=len(du), columns=list(du.columns))
h = pd.read_csv(AG / 'o_disposal_rejudge/disposal_hist_rejudge.csv', dtype={'code': str}); o['disposal_hist_rejudge'] = dict(rows=len(h), in_panel_true=int((h.in_panel == True).sum()))
d5 = pd.read_csv(X / 'disposal_5y.csv.gz', dtype={'code': str}); o['disposal_5y'] = dict(rows=len(d5), date_min=d5.date.min(), date_max=d5.date.max())
ev = list(csv.DictReader(open(AG / 'o_engine_delist_consideration/events_v2.csv', encoding='utf-8')))
o['events_v2'] = dict(rows=len(ev), types=dict(collections.Counter(x['type'] for x in ev)), no_last_trade=sum(1 for x in ev if not x['last_trade']), no_terminate=sum(1 for x in ev if not x['terminate_date']))
R = json.loads((AG / 'o_delist_rev_backfill_allpit/results.json').read_text(encoding='utf-8'))
s = R['RESULTS_V7X']['settle_info_sample']; o['o18_settle_info_sample'] = dict(n=len(s), n_fallback=dict(collections.Counter(str(v.get('n_fallback')) for v in s.values() if v)))
o['o18_real_table_info'] = R['RESULTS_V7X_PLANS']['real_table_info']; o['o18_plan_counts'] = {k: v['n_rows'] for k, v in R['RESULTS_V7X_PLANS']['counts'].items()}
Path(__file__).with_name('check_tables.json').write_text(json.dumps(o, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(o, ensure_ascii=False))
