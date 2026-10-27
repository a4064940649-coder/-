"""迭代：只用挑選期刪除事件，8 個事前列舉的版本；全部記錄到 iterations.json。
選擇規則（事前寫死）：高賣壓組扣 0.5% 後淨平均的月分群 t ≥ 1.5，且高減低 > 0（劑量方向正確）；
多個版本通過時取高賣壓組淨 t 最大者，同分取持有期較短者。沒有版本通過 → 不可用，不打開驗證期。"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import iflib as L
T = pd.read_csv(os.path.join(L.HERE, 'explore_events_select.csv'), parse_dates=['eff'], dtype={'code': str})
T = T[T.is_del & (T.eff <= L.SEL_END)].copy()
assert T.eff.max() <= L.SEL_END
VERS = [dict(v=f'V{i+1}', entry=en, hold=h, cut=c) for i, (en, h, c) in enumerate(
    [('closeE', 5, 'T3'), ('closeE', 10, 'T3'), ('closeE', 20, 'T3'), ('closeE', 5, 'T2'), ('closeE', 10, 'T2'), ('closeE', 20, 'T2'),
     ('openE1', 5, 'T3'), ('openE1', 20, 'T3')])]
q3 = T.pressure.quantile([1 / 3, 2 / 3]).tolist(); q2 = float(T.pressure.median())
g = T.eff.dt.to_period('M').astype(str)
rows = []
for V in VERS:
    x = T[f'x_{V["entry"]}_{V["hold"]}']
    if V['cut'] == 'T3':
        grp = np.where(T.pressure >= q3[1], 'high', np.where(T.pressure < q3[0], 'low', 'mid'))
    else:
        grp = np.where(T.pressure >= q2, 'high', 'low')
    res = {k: L.stats(x[grp == k], g[grp == k], cost=0.005) for k in ['low', 'mid', 'high'] if (grp == k).any()}
    res['all'] = L.stats(x, g, cost=0.005)
    res['high_stress1pct'] = L.stats(x[grp == 'high'], g[grp == 'high'], cost=0.01)
    # 高減低：按事件月配對（同月兩組平均之差）
    hi = pd.Series(x[grp == 'high'].values, index=g[grp == 'high'].values).groupby(level=0).mean()
    lo = pd.Series(x[grp == 'low'].values, index=g[grp == 'low'].values).groupby(level=0).mean()
    d = (hi - lo).dropna()
    diff = {'mean': round(float(d.mean()) * 100, 2) if len(d) else None, 'n_months': int(len(d)),
            't': round(float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))), 2) if len(d) > 2 else None,
            'pooled_mean_diff': round(float(x[grp == 'high'].mean() - x[grp == 'low'].mean()) * 100, 2)}
    passed = bool(res['high'].get('t_month', -9) >= 1.5 and (diff['pooled_mean_diff'] or -1) > 0)
    rows.append({**V, 'cuts': q3 if V['cut'] == 'T3' else [q2], 'groups_net': res, 'high_minus_low': diff, 'pass': passed})
    print(V['v'], V['entry'], V['hold'], V['cut'], 'high', res['high'], '| low', res['low'].get('mean'), '| H-L', diff, 'PASS' if passed else '')
ok = [r for r in rows if r['pass']]
sel = sorted(ok, key=lambda r: (-r['groups_net']['high']['t_month'], r['hold']))[0] if ok else None
json.dump({'n_versions': len(VERS), 'select_rule': __doc__, 'versions': rows, 'selected': sel['v'] if sel else None},
          open(os.path.join(L.HERE, 'iterations.json'), 'w'), ensure_ascii=False, indent=1, default=float)
print('SELECTED', sel['v'] if sel else None)
