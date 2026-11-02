# P3 成本：每次額外換股的來回成本（K5T 近似：OA_pred＝0.03×V60_{t−1}，壓力＝衝擊×2＋每邊 0.4%＋手續費 6 折）
import pandas as pd, numpy as np, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
P = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel/'
cr = pd.read_parquet(P + 'close_raw.parquet'); op = pd.read_parquet(P + 'open.parquet'); val = pd.read_parquet(P + 'value.parquet'); ret = pd.read_parquet(P + 'ret.parquet')
good = cr.index[cr.notna().sum(axis=1) >= 300]
cr, op, val, ret = [x.loc[good] for x in (cr, op, val, ret)]
V60 = val.rolling(60, min_periods=40).mean().shift(1); S60 = ret.rolling(60, min_periods=40).std().shift(1)
Tr = pd.read_csv(os.path.join(HERE, 'p3_trades_V2.csv'), dtype={'code': str}, parse_dates=['e'])
M = pd.read_csv(os.path.join(HERE, 'p3_monthly_V2.csv'))
ex = pd.read_json(os.path.join(HERE, 'p3_rep.json'))
def tick(p): return 0.01 if p < 10 else 0.05 if p < 50 else 0.1 if p < 100 else 0.5 if p < 500 else 1.0 if p < 1000 else 5.0
out = {}
import sys
STRESS_K = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
win = Tr.copy()
win['o'] = [op.at[e, c] for e, c in zip(win.e, win.code)]
win['v60'] = [V60.at[e, c] for e, c in zip(win.e, win.code)]; win['s60'] = [S60.at[e, c] for e, c in zip(win.e, win.code)]
for tier in [1e6, 3e6, 1e7]:
    Q = tier / 10
    def side(o, v, s, stress):
        if not (np.isfinite(v) and np.isfinite(s) and v > 0): imp = 0.003
        else: imp = max(0.001 + 0.5 * s * np.sqrt(Q / (0.03 * v)), tick(o) / o)
        fee = 0.001425 * (0.6 if stress else 0.28)
        return fee + (STRESS_K * imp + 0.004 if stress else imp)
    g = np.array([2 * side(o, v, s, False) + 0.003 for o, v, s in zip(win.o, win.v60, win.s60) if np.isfinite(v) and np.isfinite(s)])
    st = np.array([2 * side(o, v, s, True) + 0.003 for o, v, s in zip(win.o, win.v60, win.s60) if np.isfinite(v) and np.isfinite(s)])
    out[f'{int(tier/1e4)}萬'] = dict(n=len(g), gen_median_pct=np.median(g) * 100, gen_p90_pct=np.quantile(g, .9) * 100, stress_median_pct=np.median(st) * 100)
extra_half = ex.loc['extra', 'V2'] / 2
for k, v in out.items():
    v['port_gen_pp'] = extra_half * v['gen_median_pct'] / 10; v['port_stress_pp'] = extra_half * v['stress_median_pct'] / 10
    v['port_gen_pp_at_res_1.89'] = 1.89 * v['gen_median_pct'] / 10; v['port_stress_pp_at_res_1.89'] = 1.89 * v['stress_median_pct'] / 10
print(json.dumps(out, ensure_ascii=False, indent=1, default=float))
json.dump(out, open(os.path.join(HERE, 'p3_cost_k%s.json' % STRESS_K + ''), 'w'), ensure_ascii=False, indent=1, default=float)
