#!/usr/bin/env python3
"""步驟 3：市場證據初篩（判準見 SPEC_draft.json；SPEC_draft 寫於本程式之前）。對 22 起事件＋o17 已核實 11 起（校準用）計算。
輸出 market_screen.json、raw/market_<code>.csv（只存數值）。python3 -I -B s3_market.py"""
import json, hashlib, sys
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); import mkt as M; sys.path.pop(0)
SPEC = json.loads((HERE / 'SPEC_draft.json').read_text(encoding='utf-8'))
E = pd.read_csv(HERE.parent / 'o_engine_delist_consideration' / 'events_v2.csv', dtype=str)
L = pd.read_csv(HERE / 'event_list_frozen.csv', dtype=str)
VER = list(E.code[E.verified == '已核實（新聞）'])
out = {}
for c in list(L.code) + VER:
    r = E[E.code == c].iloc[0]
    succ = r.successor if isinstance(r.successor, str) else None
    res = M.screen(c, r.last_trade, r.terminate_date, succ)
    if isinstance(res, tuple):
        o, rows = res; rows.to_csv(HERE / 'raw' / f'market_{c}.csv', index=False)
    else:
        o = res
    o['group'] = 'event22' if c in set(L.code) else 'verified_o17'; o['type_v2'] = r.type; o['successor'] = succ
    out[c] = o
(HERE / 'market_screen.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
for c, o in out.items():
    print(c, o.get('group'), o.get('type_v2'), o.get('market'), 'lockf', round(o.get('lock_frac', -1), 2), 'w', round(o.get('lock_width', -1), 4), 'R2', o.get('track_R2') and round(o['track_R2'], 2), 'ratio', o.get('ratio_med10') and round(o['ratio_med10'], 4), 'gap', o.get('gap'), o.get('lt_panel'))
