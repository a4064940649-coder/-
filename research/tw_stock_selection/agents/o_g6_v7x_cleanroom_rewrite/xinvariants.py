# 只做不變量檢查（不輸出任何報酬數字）
import numpy as np, pandas as pd, json, time, math, sys
import xdata, xengine, xlists
W = xengine.World(); L = xlists.Lists()
od = L.official_days()
res = {}
# 1 CPI
c = xdata.cpi(); res['cpi_ratio_2006_2026'] = c[2006] / c[2026]
# 2 跳動點漲停價為跳動點整數倍
X = xdata.load(); up = X['UP']; ok = np.isfinite(up)
tk = xdata.tick_of(np.where(ok, up, 1)); frac = np.abs(up / tk - np.round(up / tk))
res['limit_tick_multiple_maxdev'] = float(np.nanmax(np.where(ok, frac, 0)))
# 3 模擬：現金不為負、整股股數、名單規模
plan = L.plan(od[:40], seed=None)
for cfg in [dict(), dict(cost='stress'), dict(tier=300_000, whole_pre2020=True), dict(tier=10_000_000)]:
    t0 = time.time()
    sim = xengine.Sim(W, plan, plan[-1][0], dict(cfg, record=True))
    class Spy(xengine.Sim): pass
    nav = sim.run()
    tr = sim.trades
    lots_ok = True; neg = 0
    res[str(cfg)] = {'n_trades': len(tr), 'secs': round(time.time() - t0, 2), 'cash_end_nonneg': bool(sim.cash >= -1e-6),
                     'all_holdings_nonneg': bool(all(v >= -1e-9 for v in sim.S.values())),
                     'nav_finite': bool(np.isfinite(nav[plan[0][0]:plan[-1][0] + 1]).all())}
    if cfg.get('whole_pre2020'):
        res[str(cfg)]['odd_trades_pre2020'] = int(sum(1 for (t, j, n, cash) in tr if abs(n) % 1000 != 0 and W.dates[t] < pd.Timestamp('2020-10-26')))
json.dump(res, open('logs/invariants_v1.json', 'w'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(res, ensure_ascii=False, default=str))
