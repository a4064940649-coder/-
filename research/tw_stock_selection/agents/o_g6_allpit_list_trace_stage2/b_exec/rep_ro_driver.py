# B 角色：在 scratchpad 唯讀副本中執行複製員引擎（不修改），只加 trace；輸出 json.gz 到本題 out/。
import sys, json, gzip
sys.path.insert(0, '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/o_g6s2/rep_ro')
import numpy as np, pandas as pd, rengine as RE
j, seed, cell = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
CELLS = {'N100': (100e4, False, False), 'N300': (300e4, False, False), 'N1000': (1000e4, False, False), 'S100': (100e4, True, False), 'O30': (30e4, False, False)}
D = RE.Data('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/rep_o19/rdata_strict.npz')
p = pd.read_csv(RE.AG + '/o_v7_listing_exec_engine/data/plans_j7.csv', dtype={'picks': str})
g = p[(p.strat == 'AB_V7_B10') & (p.j == j) & (p.seed == seed)].sort_values('signal')
pl = [(D.dpos[np.datetime64(a)], D.dpos[np.datetime64(b)], [D.col[c] for c in s.split(',')]) for a, b, s in zip(g.signal, g['exec'], g.picks)]
tier, stress, w30 = CELLS[cell]
tr = []
s0, eq = RE.run_sim(D, pl, RE.Cfg(tier, stress, w30), trace=tr)
rec = [dict(t=int(t), date=str(D.dates[t]), Tm=float(Tm), S={str(D.codes[n]): float(v) for n, v in S.items()}, cash=float(c)) for t, Tm, S, c in tr]
out = dict(j=j, seed=seed, cell=cell, s0=int(s0), rec=rec, eq={str(D.dates[i]): float(eq[i]) for i in range(s0, len(eq)) if np.isfinite(eq[i])})
fn = f'/home/user/-/research/tw_stock_selection/agents/o_g6_allpit_list_trace_stage2/out/rep_J{j}_s{seed}_{cell}.json.gz'
with gzip.open(fn, 'wt') as f: json.dump(out, f)
print('ok', fn, len(rec))
