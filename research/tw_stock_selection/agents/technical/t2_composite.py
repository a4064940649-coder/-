#!/usr/bin/env python3
"""第 6 步：候選規則（挑選期淨 CAGR > 0 且月報酬對上市等權 t ≥ 2）合成 → 驗證期。
挑選期沒有任何規則通過時，正式合成為空集合；另列兩個「探索用」合成（不構成候選，只說明若硬做會怎樣）：
  X1＝挑選期對等權 t > 0 的規則；X2＝挑選期組合淨 CAGR > 0 的規則。
合成方式（事前固定）：各規則事件放在同一池，強度改為「同日同規則內的百分位」，各自沿用原規則的出場；上限 30 檔等權。
另：每條規則挑選期再切 7% 漲跌幅（2005-01～2015-05）與 10%（2015-06～2019-04）兩段的 t 值，看是否只在某段有效。
"""
import sys, json, pickle
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tlib as L


def main():
    D = L.load()
    res_rules = json.loads((L.OUT / 'results_rules.json').read_text())
    with open(L.PKL, 'rb') as f:
        pk = pickle.load(f)
    evs, daily = pk['events'], pk['daily']
    R = res_rules['rules']
    cand = [k for k, v in R.items() if v['pass_S']]
    x1 = [k for k, v in R.items() if (v['S']['portfolio']['t_vs_EW'] or -9) > 0]
    x2 = [k for k, v in R.items() if (v['S']['portfolio']['net'].get('CAGR') or -9) > 0]
    out = {'candidates': cand, 'formal_composite': 'EMPTY（挑選期沒有規則通過）' if not cand else cand}
    sets = {'X1_S_tEW>0': x1, 'X2_S_netCAGR>0': x2}
    if cand: sets = {'FORMAL': cand, **sets}
    rows = {w: [] for w in L.WIN}
    for nm, ks in sets.items():
        parts = []
        for k in ks:
            e = evs[k].copy()
            e['strength'] = e.groupby('e')['strength'].rank(pct=True)
            e['src_rule'] = k
            parts.append(e)
        ev = pd.concat(parts, ignore_index=True)
        o, pf, _ = L.evaluate(D, ev, nm)
        o['members'] = ks
        out[nm] = o
        for w in L.WIN: rows[w].append(L.row(o, w))
    out['tables'] = rows
    # 挑選期兩段
    sub = {}
    for k, pf in daily.items():
        net = pf['gross'] - pf['cost']
        r = {}
        for lab, a, b in [('S_7pct_2005_2015-05', '2005-01-01', '2015-05-31'), ('S_10pct_2015-06_2019-04', '2015-06-01', '2019-04-30')]:
            m = (net.index >= a) & (net.index <= b)
            n_, b_, e_ = L.monthly(net[m]), L.monthly(D['r0050'][m]), L.monthly(D['r_ew'][m])
            r[lab] = {'CAGR': L.perf(net[m]).get('CAGR'), 'EW_CAGR': L.perf(D['r_ew'][m]).get('CAGR'), 't_vs_EW': L.tstat(n_ - e_), 't_vs_0050': L.tstat(n_ - b_)}
        sub[k] = r
    out['S_subperiods'] = sub
    L.save('results_composite.json', out)
    for w in L.WIN:
        print('==', w); print(pd.DataFrame(rows[w]).to_string(index=False))
    print(pd.DataFrame({k: {f'{lab}_{m}': v[lab][m] for lab in v for m in ('CAGR', 't_vs_EW')} for k, v in sub.items()}).T.to_string())


if __name__ == '__main__':
    main()
