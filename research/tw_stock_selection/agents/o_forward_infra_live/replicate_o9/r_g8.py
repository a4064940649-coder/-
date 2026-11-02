"""G8 檢定力：月超額 ~ N(α, σ)，12 個月加總 > 0 的機率＝Φ(α√12/σ)；另以模擬覆核。資料過期規則檢查。"""
import sys, json, math
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_forward_infra_live/replicate_o9')
import numpy as np, pandas as pd
import rlib as L
sig = 6.14
Phi = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
rng = np.random.default_rng(0)
out = {}
for a in [-0.5, -0.3, 0.0, 0.5]:
    sim = float((rng.normal(a, sig, (400000, 12)).sum(1) > 0).mean())
    out[str(a)] = dict(analytic=round(Phi(a * math.sqrt(12) / sig), 4), sim=round(sim, 4))
# 資料過期：asof 是交易日且晚於資料最後交易日，或相差 > 4 日
st = {}
for a in ['2026-11-10', '2026-11-11', '2026-11-02']:
    a = pd.Timestamp(a)
    stale = (a > L.DATA_END and a in L.CAL) or (a - L.DATA_END).days > 4
    st[str(a.date())] = dict(stale=bool(stale), rc=2 if stale else 0)
print(json.dumps(dict(g8=out, stale=st, data_end=str(L.DATA_END.date())), indent=1))
(L.HERE / 'out_g8.json').write_text(json.dumps(dict(g8=out, stale=st), indent=1))
