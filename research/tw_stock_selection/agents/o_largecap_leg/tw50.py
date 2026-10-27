#!/usr/bin/env python3
"""輔助股池：臺灣50 成分倒推（只讀 etf_rebalance/events.csv 與 extras/etf0050_w.csv），寫 tw50_members.json。
從 2026-10-09 成分往回，每個生效日之前＝之後 − 納入 ＋ 刪除；每一步檢查（納入者在之後名單、刪除者不在、數量＝50），
第一個不一致的時點以前不用。另加 etf_rebalance 註明的非定期遞補（2016-12 中壽替華亞科、2018-04 國巨替矽品），
以及 2018-04-30 日月光投控（3711）承接日月光（2311）的控股轉換（指數規則：控股公司轉換直接承接）。"""
import json
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
e = pd.read_csv(HERE.parent / 'etf_rebalance' / 'events.csv', dtype=str)
e = e[e['etf'] == 'TW50']
w = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras/etf0050_w.csv', dtype=str)
cur = set(w.loc[w['asof'] == '2026-10-09', 'code'])
assert len(cur) == 50
ev = {}
for _, r in e.iterrows():
    ev.setdefault(r['eff'], {'add': set(), 'del': set()})[r['action']].add(r['code'])
EXTRA = [('2016-12-07', {'2823'}, {'3474'}, '中壽替華亞科（下市遞補，日期取華亞科終止上市前後）'),
         ('2018-04-30', {'2327'}, {'2325'}, '國巨替矽品（下市遞補）'),
         ('2018-04-30', {'3711'}, {'2311'}, '日月光投控承接日月光（控股轉換）')]
for dt, a, d_, _ in EXTRA:
    x = ev.setdefault(dt, {'add': set(), 'del': set()}); x['add'] |= a; x['del'] |= d_
steps = []; S = set(cur); ok_from = None; issues = []
for dt in sorted(ev, reverse=True):
    a, d_ = ev[dt]['add'], ev[dt]['del']
    bad = [c for c in a if c not in S] + [c for c in d_ if c in S]
    if bad or len(a) != len(d_):
        issues.append({'eff': dt, 'bad': bad, 'n_add': len(a), 'n_del': len(d_)}); ok_from = dt; break
    S = (S - a) | d_
    assert len(S) == 50
    steps.append({'eff': dt, 'members_before': sorted(S)})
out = {'current_2026-10-09': sorted(cur), 'steps_backward': steps, 'issues': issues, 'first_bad_eff': ok_from,
       'earliest_valid_eff': steps[-1]['eff'] if steps else None, 'extra_events': [list(map(lambda z: sorted(z) if isinstance(z, set) else z, x)) for x in EXTRA],
       'note': '成分在 eff 日收盤生效（隔一交易日起新成分）。訊號日 d 的成分＝最後一個 eff < d 之後的名單；最早可用訊號日＝earliest_valid_eff 之後。events.csv 從 2015-03 審核開始，2015-03 之前一律不用。'}
(HERE / 'tw50_members.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
print('steps', len(steps), 'earliest', out['earliest_valid_eff'], 'issues', issues)
