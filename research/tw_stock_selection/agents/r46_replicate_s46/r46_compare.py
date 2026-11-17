#!/usr/bin/env python3
"""R46 對照：把 results.json 與 S46 宣稱值（任務單＋S46 FINDINGS.md 熱圖，手抄）逐項比對，容許 ±0.05 個百分點。輸出 compare.csv。"""
import json, csv
from pathlib import Path
H = Path(__file__).resolve().parent
r = json.loads((H / 'results.json').read_text()); s = r['summary']; c = r['cells']
claims = [('現行主跑 2019 起年化', 48.76, s['current_main_since2019'] * 100), ('現行已出場筆數', 381, s['current_n_closed']),
          ('現行 20 條中位數', 46.40, s['current_med20'] * 100), ('家族 45 格中位數', 40.15, s['family_median_of_med20'] * 100),
          ('現行名次（/45）', 1, s['current_rank']), ('現行−中位數（百分點）', 6.25, s['P3_pp']),
          ('900 條路徑中位數', 40.42, s['all900_median'] * 100), ('900 條路徑 P10', 35.94, s['all900_p10'] * 100),
          ('回看期線 3 格中位數', 40.93, s['lookback_line_median'] * 100), ('配對勝過現行格數', 0, s['n_better']),
          ('配對比現行差格數', 41, s['n_worse'])]
heat = {8: {10: [39.90, 38.99, 36.86, 39.51, 37.67], 20: [38.91, 41.84, 42.61, 42.40, 39.25], 40: [36.77, 36.59, 37.07, 36.61, 37.02]},
        10: {10: [42.90, 43.21, 40.43, 41.10, 38.61], 20: [40.32, 43.45, 46.40, 45.16, 42.89], 40: [39.64, 38.30, 40.93, 40.74, 41.09]},
        12: {10: [41.06, 40.58, 39.72, 40.15, 37.96], 20: [39.82, 42.46, 45.18, 45.18, 42.50], 40: [39.65, 38.34, 39.59, 39.54, 40.22]}}
for D, row in heat.items():
    for L, vals in row.items():
        for g, v in zip((0, 5, 10, 15, 20), vals):
            claims.append((f'熱圖 {g}%/{L}日/前{D}', v, c[f'g{g}_L{L}_D{D}']['med20'] * 100))
rows = []
for name, cl, mine in claims:
    d = mine - cl
    rows.append(dict(item=name, s46=cl, r46=round(mine, 4), diff=round(d, 4), ok=abs(d) <= 0.05 + 1e-9))
with open(H / 'compare.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(sum(x['ok'] for x in rows), '/', len(rows), '通過；最大 |差|', max(abs(x['diff']) for x in rows))
for x in rows:
    if not x['ok'] or not x['item'].startswith('熱圖'):
        print(x)
