"""J0 在 13 個 j 年化超額（月窗 ann − 0050 ann，同 integrate_o11 gex 口徑）的排序位置。"""
import importlib.util, numpy as np, pandas as pd
AG = '/home/user/-/research/tw_stock_selection/agents/'
sp = importlib.util.spec_from_file_location('fc', AG + 'o_main_spa_breaks/fam_common.py'); FC = importlib.util.module_from_spec(sp); sp.loader.exec_module(FC)
WIN = FC.WIN; r50 = FC.win_ret(FC.e50_nav()).to_numpy()
EX = pd.read_parquet(AG + 'o_exec_engine_rewrite_v7x/out/navs_oos_R4.parquet')
def ann(r): return float(np.prod(1 + r) ** (12 / len(r)) - 1) * 100
R = []
for j in range(13):
    r = FC.win_ret(EX[f'B10|J{j}|K5T|normal|100萬|GROW']).to_numpy(); R.append(np.where(np.isfinite(r), r, 0.0))
for nm, a, b in (('ALL', '2006-04-01', '2026-10-12'), ('2019_', '2019-01-01', '2026-10-12')):
    mk = (WIN >= pd.Timestamp(a)) & (WIN <= pd.Timestamp(b))
    g = pd.Series({j: ann(R[j][mk]) - ann(r50[mk]) for j in range(13)})
    asc = g.sort_values(); rank_asc = list(asc.index).index(0) + 1
    p10 = np.percentile(g.values, 10)
    print(nm, f'J0 asc rank {rank_asc}/13 gex {g[0]:+.2f}  median J{asc.index[6]} {asc.iloc[6]:+.2f}  p10 {p10:+.2f} between J{asc.index[1]} & J{asc.index[2]}  drop-J0 median {np.median(g.drop(0)):+.2f}')
    print('  ', {f'J{k}': round(v, 2) for k, v in asc.items()})
