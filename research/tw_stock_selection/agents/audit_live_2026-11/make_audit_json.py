#!/usr/bin/env python3
"""合併 audit_list.json、audit_backtest.json、audit_0050.json 成 audit.json（含逐項判定與總結論）。python3 -I -B make_audit_json.py"""
import json, collections
from pathlib import Path
H = Path(__file__).resolve().parent
L = json.load(open(H / 'audit_list.json', encoding='utf-8'))
B = json.load(open(H / 'audit_backtest.json', encoding='utf-8'))
E = json.load(open(H / 'audit_0050.json', encoding='utf-8'))
README12 = dict(cagr_since2019=0.4813, cagr_full=0.3408, maxdd=-0.334, n_trades=379)
bt_checks = [
    dict(item='回測 2019 起年化', site=48.13, recomputed=round(B['cagr_since2019'] * 100, 2)),
    dict(item='回測 2015-07 起年化', site=34.08, recomputed=round(B['cagr_full'] * 100, 2)),
    dict(item='回測 最大回撤', site=-33.4, recomputed=round(B['maxdd'] * 100, 2)),
]
for c in bt_checks:
    c['diff'] = round(c['recomputed'] - c['site'], 3); c['verdict'] = '通過' if abs(c['diff']) <= 0.1 else '不通過'; c['source'] = 'xq_scripts/README.md 第 12 節'
bt_checks.append(dict(item='回測 已平倉交易筆數', site=379, recomputed=B['n_trades_closed'], diff=B['n_trades_closed'] - 379,
                      verdict='通過' if B['n_trades_closed'] == 379 else '不通過', source='xq_scripts/README.md 第 12 節', note=f"另有期末未平倉 {B['n_open_end']} 筆"))
bt_checks.append(dict(item='回測淨值 對 s19 navs.csv.gz（逐日最大相對差）', site=0, recomputed=B['vs_s19_navs']['max_abs_rel_diff'], diff=B['vs_s19_navs']['max_abs_rel_diff'],
                      verdict='通過' if B['vs_s19_navs']['max_abs_rel_diff'] < 1e-5 else '不通過', source='s19 navs.csv.gz E0|W10_70'))
checks = L['checks'] + bt_checks
cnt = collections.Counter(c['verdict'] for c in checks)
out = dict(conclusion='有條件通過' if cnt.get('不通過', 0) and all(('yearly' in c['item'] and '0050' in c['item']) for c in checks if c['verdict'] == '不通過') else ('通過' if not cnt.get('不通過') else '不通過'),
           counts=dict(cnt), checks=checks, list_audit={k: v for k, v in L.items() if k != 'checks'}, backtest=B, e50_official=E)
json.dump(out, open(H / 'audit.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(out['conclusion'], dict(cnt))
