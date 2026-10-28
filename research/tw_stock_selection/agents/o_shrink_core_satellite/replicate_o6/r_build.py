"""Independent replication o6: build windows, focus series and families from raw inputs."""
import json
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

A = '/home/user/-/research/tw_stock_selection/agents/'
OUT = A + 'o_shrink_core_satellite/replicate_o6/'

p = pd.read_csv(A + 'o_v4_revenue_patch_official/plan_V4.csv')
G = pd.to_datetime(list(p.signal) + ['2026-10-12'])
r50 = pd.read_parquet(A + 'o_main_spa_breaks/data/bench.parquet').r0050.values


def win(s):
    s = s.reindex(G)
    return s.values[1:] / s.values[:-1] - 1


v4 = pd.read_parquet(A + 'o_v4_revenue_patch_official/navs_v4.parquet')
abv = pd.read_parquet(A + 'o_exec_capacity_tiers/navs_abavg.parquet')
eb = pd.read_parquet(A + 'o_exec_capacity_tiers/data/bench.parquet')
v5 = pd.read_parquet(A + 'o_v5_pit_revenue/navs_v5.parquet')
js = pd.read_parquet(A + 'o_v5_pit_revenue/navs_jscan_v5.parquet')

e = {}
e['AB'] = (win(v4['AB_V4|normal']) - r50) * 100
e['B10'] = (win(v4['AB_V4_B10|normal']) - r50) * 100
drag = (win(abv['main|100萬|FIX|A']) - win(eb['AB_V4|normal'])) * 100
e['EXEC'] = e['B10'] + drag
mj = [np.nanmean(win(js[c]) - r50) for c in js.columns]
dj = (np.median(mj) - np.mean(win(v5['AB_V5|normal']) - r50)) * 100
e['EXEC_J'] = e['EXEC'] + dj
e['AB_S'] = (win(v4['AB_V4|stress']) - r50) * 100
e['B10_S'] = (win(v4['AB_V4_B10|stress']) - r50) * 100
e['EXEC_S'] = e['B10_S'] + (win(abv['stress|100萬|FIX|B']) - win(eb['AB_V4|normal'])) * 100
I = ((win(abv['d5_impact|100萬|FIX|A']) - win(abv['d4_fees|100萬|FIX|A'])) * 100).mean()
sd_cost = (1.2965 - 0.2) * abs(I) / np.sqrt(12)
foc = pd.DataFrame(e, index=G[:-1])
foc['r0050'] = r50
foc.to_parquet(OUT + 'focus.parquet')
print('drag', drag.mean(), 'dj', dj, 'I', I, 'sd_cost', sd_cost)
print(foc.mean())

# families
fe = pd.read_parquet(A + 'o_main_spa_breaks/data/family_excess.parquet')
fm = pd.read_csv(A + 'o_main_spa_breaks/data/family_members.csv')
ids = fm.member_id[fm.F_long.astype(bool)].tolist()
X = fe[ids].fillna(0.0)
X.to_parquet(OUT + 'flong.parquet')
C = np.corrcoef(X.values.T)
C = np.nan_to_num(C)
np.fill_diagonal(C, 1.0)
lam = np.linalg.eigvalsh(C)
lam = np.abs(lam)
meff = float(np.sum((lam >= 1).astype(float) + (lam - np.floor(lam))))
print('Li-Ji Meff', meff)
D = 1 - C
D = (D + D.T) / 2
np.fill_diagonal(D, 0)
D[D < 0] = 0
Z = linkage(squareform(D, checks=False), method='average')
lab = fcluster(Z, t=161, criterion='maxclust')
print('n clusters', len(set(lab)))
med = []
for k in sorted(set(lab)):
    idx = np.where(lab == k)[0]
    sub = C[np.ix_(idx, idx)]
    med.append(ids[idx[np.argmax(np.round(sub.mean(1), 10))]])
iab = ids.index('o_v4_revenue_patch_official/navs_v4.parquet::AB_V4|normal')
kab = lab[iab]
print('AB cluster', kab, 'size', (lab == kab).sum(), 'AB is medoid', ids[iab] in med)
ref = json.load(open(A + 'o_shrink_core_satellite/data/families.json'))
print('MAIN medoids overlap with ref', len(set(med) & set(ref['MAIN'])), 'of', len(med))
medab = med[sorted(set(lab)).index(kab)]
loco = [m for m in med if m != medab]
fam = {'MAIN': med, 'LOCO': loco, 'ALL': ids, 'THEORY': ref['THEORY'], 'meff': meff, 'kab': int(kab)}
json.dump(fam, open(OUT + 'fam.json', 'w'))
