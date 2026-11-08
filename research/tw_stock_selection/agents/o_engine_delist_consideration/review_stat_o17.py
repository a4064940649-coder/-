#!/usr/bin/env python3
"""反方審查（資料與統計）：o_engine_delist_consideration。只讀，不改研究員檔案。python3 -I -B review_stat_o17.py"""
import json, csv, pickle
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
AG = HERE.parent
SCR = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad')
E = {r['code']: r for r in csv.DictReader(open(HERE / 'events_v2.csv', encoding='utf-8'))}
X = json.loads((HERE / 'explore.json').read_text(encoding='utf-8'))
R = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
out = {}
# 1. ia 面板存活者缺口：32 檔在 2019-05-17 後 ia 面板是否完全沒價
C = pd.read_parquet(SCR / 'panel' / 'close_raw.parquet')
gap = [r['code'] for r in X['price_survivor_gap_ia_panel']['rows']]
cut = pd.Timestamp('2019-05-17')
res = {}
for c in gap:
    if c not in C.columns: res[c] = 'not_in_ia_panel'; continue
    s = C[c].dropna()
    res[c] = dict(n_before=int((s.index < cut).sum()), n_after=int((s.index >= cut).sum()), last=str(s.index.max().date()) if len(s) else None)
out['gap_ia'] = res
out['gap_n_zero_after'] = sum(1 for v in res.values() if v == 'not_in_ia_panel' or v['n_after'] == 0)
# 一般：ia 面板 2019-05-16 最後有價的檔數 vs 前後日期
last = C.apply(lambda s: s.last_valid_index())
out['ia_codes_last_2019_05_16'] = int((last == pd.Timestamp('2019-05-16')).sum())
# 2. C0 在 2019-05-17 後能否選到這 32 檔：名單裡出現次數
P = pickle.loads((SCR / 'o17_delist' / 'plans_ia.pkl').read_bytes()) if (SCR / 'o17_delist' / 'plans_ia.pkl').exists() else None
out['plans_ia_available'] = P is not None
# 3. 事件核實度：U_opt 6 起、V7X 引擎結算事件
uopt = [e[0] for e in X['registered_sets']['U_opt（FM_COUNTS C0|U_opt，只含 2006～2013）']['events_U']]
out['U_opt_events'] = {c: (E[c]['type'], E[c]['verified'], bool(E[c]['raw_files'])) for c in uopt}
si = R['RESULTS_V7X']['settle_info']
ev_types = {}
for k, v in si.items():
    for eng in v:
        for e in v[eng]['ev']:
            ev_types.setdefault(e[0], e[1])
out['V7X_engine_events'] = {c: (t, E.get(c, {}).get('verified'), E.get(c, {}).get('held'), bool(E.get(c, {}).get('raw_files'))) for c, t in sorted(ev_types.items())}
out['V7X_engine_n'] = len(ev_types)
out['V7X_engine_never_searched'] = [c for c, t in ev_types.items() if E.get(c, {}).get('held') == 'False']
out['V7X_engine_terms_verified'] = [c for c, t in ev_types.items() if E.get(c, {}).get('verified') == '已核實（新聞）']
# 4. lo 寫銷的來源：cons_lo writeoff 價值合計，依事件（依 held 與否）
wo = defaultdict(float); n_paths = defaultdict(int)
for k, v in si.items():
    if 'cons_lo' not in v: continue
    for e in v['cons_lo']['ev']:
        for p in e[4]:
            if p[0] == 'writeoff':
                wo[e[0]] += e[3]; n_paths[e[0]] += 1
out['lo_writeoff_value_by_event'] = {c: [round(wo[c]), n_paths[c], E.get(c, {}).get('held')] for c in sorted(wo, key=lambda c: -wo[c])}
tot = sum(wo.values()); ns = sum(wo[c] for c in wo if E.get(c, {}).get('held') == 'False')
out['lo_writeoff_share_from_never_searched'] = ns / tot if tot else None
out['n_settle_paths'] = len(si)
# 5. 2809 / 2888 價格延續：面板是否真的停止
for c in ('2809', '2888'):
    s = C[c].dropna() if c in C.columns else pd.Series(dtype=float)
    out['last_ia_' + c] = str(s.index.max().date()) if len(s) else None
out['ia_panel_last_date'] = str(C.index.max().date())
n_alive_last = int(C.iloc[-30:].notna().any().sum())
out['ia_codes_alive_last30'] = n_alive_last
# 6. 事件總數：28 held + 5 執行層
out['held_table'] = Counter(r['verified'] for r in E.values() if r['held'] == 'True')
out['held_plus_exec_n'] = sum(1 for r in E.values() if r['held'] == 'True') + len(out['V7X_engine_never_searched'])
# 7. 複製 asis（不對齊）差
rep = R['REPLICATION']['summary']
out['replication_asis_fail'] = {k: v for k, v in rep.items() if not v['pass_']}
# 8. 年度集中度：V7X lo 年度差絕對值前 3 年占比
yd = R['RESULTS_V7X']['yearly_delta_med_100m_normal']
for eng in ('cons_main', 'cons_lo'):
    a = pd.Series(yd[eng]).abs().sort_values(ascending=False)
    out['yearly_top3_share_' + eng] = dict(top3=a.index[:3].tolist(), share=float(a.iloc[:3].sum() / a.sum()), sum_signed=float(pd.Series(yd[eng]).sum()))
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
