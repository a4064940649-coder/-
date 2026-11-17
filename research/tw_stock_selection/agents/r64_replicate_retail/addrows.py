#!/usr/bin/env python3
"""讀 results.json，在 compare.csv 後面補「相對照做」另一種統計口徑（每次亂數先取 20 條配對差中位數，再取 200 次中位數）與附帶格子。不重跑模擬。"""
import json, csv
from pathlib import Path
H = Path(__file__).resolve().parent
g = json.load(open(H / 'results.json'))
rows = list(g['compare'])
def add(no, name, claim, mine, tol):
    d = mine - claim
    rows.append(dict(編號=no, 項目=name, 原宣稱=claim, 重現值=round(mine, 4), 差=round(d, 4), 容許=tol, 符合=bool(abs(d) <= tol + 1e-9)))
add('1b', 'S64 隨機只買一半：相對照做（先每次取中位數，百分點）', -6.00, g['s64_rand_half']['diff_median_of_draw_medians_pp'], 0.6)
add('3b', 'S64 漏做 25%：相對照做（先每次取中位數，百分點）', -2.12, g['s64_miss']['p25']['diff_median_of_draw_medians_pp'], 0.6)
add('附', 'S64 完全照做 20 條中位數（%）', 46.40, g['step0']['base_median_c19'] * 100, 0.3)
add('附', 'S64 回落 20% 停手等 6 個月：全期最大回撤逐條差中位數（百分點，正＝變淺）', 0.0, g['s64_stop']['X20_M6']['ddfull_diff_median_pp'], 2.0)
add('附', 'S64 漏做 10%：2019 起（%）', 45.46, g['s64_miss']['p10']['c19_median_of_draw_medians'] * 100, 0.6)
add('附', 'S64 漏做 50%：2019 起（%）', 39.88, g['s64_miss']['p50']['c19_median_of_draw_medians'] * 100, 0.6)
add('附', 'S64 開盤漲超過 5% 不買：相對照做（百分點）', -0.24, g['s64_gap']['gap5']['diff_median_pp'], 0.3)
add('附', '0050 2019 起年化（%）', 28.91, g['meta']['e50_since2019'] * 100, 0.3)
g['compare_full'] = rows; g['all_pass_full'] = bool(all(r['符合'] for r in rows))
json.dump(g, open(H / 'results.json', 'w'), ensure_ascii=False, indent=1)
with open(H / 'compare.csv', 'w', newline='', encoding='utf-8-sig') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows[15:]:
    print(r)
print(g['all_pass_full'])
