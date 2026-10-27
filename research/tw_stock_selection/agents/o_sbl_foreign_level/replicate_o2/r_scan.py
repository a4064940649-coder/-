import pandas as pd, numpy as np
N = pd.read_parquet('/home/user/-/research/tw_stock_selection/agents/o_sbl_foreign_level/replicate_o2/my_navs.parquet')
def cg(s, a, b):
    x = s.loc[a:b]; yrs = (x.index[-1] - x.index[0]).days / 365.25; return round((x.iloc[-1] / x.iloc[0]) ** (1 / yrs) * 100 - 100, 2)
for a in ('2021-10-12', '2021-10-13'):
    for b in ('2023-12-11', '2023-12-29', '2024-02-15', '2024-02-16', '2024-03-11'):
        print(a, b, {c: cg(N[c], a, b) for c in ('main', 'C2', 'B1_normal', 'EW', '0050')})
for a, b in (('2024-02-15', '2025-10-13'), ('2024-03-11', '2025-10-13'), ('2024-02-16', '2025-10-14'), ('2025-10-13', '2026-10-12'), ('2025-10-14', '2026-10-12'), ('2025-10-01','2026-10-12'),('2024-02-15', '2026-10-12'), ('2024-03-11', '2026-10-12'), ('2021-10-12', '2026-10-12')):
    x = N.loc[a:b]; dd = (x / x.cummax() - 1).min() * 100
    print(a, b, {c: cg(N[c], a, b) for c in ('main', 'C2', 'B1_normal', 'C1', 'EW', '0050')}, 'MDD', dd[['main', 'C2', 'B1_normal', '0050']].round(1).to_dict())
