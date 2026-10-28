import pandas as pd, numpy as np
B = '/home/user/-/research/tw_stock_selection/agents/o_v5_pit_revenue/data/'
f = pd.read_csv(B + 'fv_cells.csv', dtype={'code': str, 'ym': str}); print(len(f), f.ym.min(), f.agree.value_counts().to_dict())
t = pd.read_csv(B + 'top30_cells.csv', dtype={'code': str, 'ym': str}); print(t.role.value_counts().to_dict())
for name, inc in [('abs>=.5%', f.rel_first.abs() >= 0.005), ('abs>=.5%&<=50%', (f.rel_first.abs() >= 0.005) & (f.rel_first.abs() <= 0.5)), ('~agree', ~f.agree.astype(bool))]:
    cor = inc & (f.consensus | f.internal_ok) & (f.rel_first.abs() <= 0.5)
    for role in ['cur', None]:
        tt = t if role is None else t[t.role == role]
        k = set(zip(tt.code, tt.ym)); m = np.array([(c, y) in k for c, y in zip(f.code, f.ym)])
        print(name, role, 'all %.2f%% cor %.2f%% (%d) | top n %d raw %.2f%% cor %d/%d ratio %.2f %.2f' % (inc.mean()*100, cor.mean()*100, cor.sum(), m.sum(), inc[m].mean()*100, cor[m].sum(), m.sum(), inc[m].mean()/inc.mean(), cor[m].mean()/cor.mean()))
