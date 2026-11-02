import sys
sys.argv = ['x']
exec(open('rp_stats.py').read().split("o, bh, bl, I = block")[0])
bh, bl = slopes('main', 'A_hi'), slopes('main', 'A_lo')
I = bh - bl
p = sel(I, 'post'); h = sel(bh, 'post'); l = sel(bl, 'post')
print(p.index.min(), p.index.max(), len(p))
best = []
for m in p.index:
    k = p.index != m
    best.append((abs(l[k].mean() + 0.0838) + abs(h[k].mean() + 0.0215) + abs(p[k].mean() - 0.0623), str(m), round(l[k].mean(), 4), round(h[k].mean(), 4), round(p[k].mean(), 4), round(tst(p[k]), 2), round(tst(l[k]),2)))
for b in sorted(best)[:5]: print(b)
x = A[(A.leg=='main')&(A.grp=='A_lo')]
print(x[x.mp >= P('2026-08')].groupby('m').j.apply(list))
print(x[(x.mp >= P('2023-12'))&(x.mp <= P('2024-03'))].groupby('m').j.apply(list))
print('--- FO probes')
bh, bl = slopes('main','FO_hi'), slopes('main','FO_lo'); I2=(bh-bl).dropna()
print(I2.index.min(), I2.index.max(), len(I2))
x = A[(A.grp=='FO_hi')&(A.leg=='main')]; print(x.groupby('mp').j.count().head(3))
# population
y = A[(A.leg=='main')&A.grp.isin(['A_lo','A_mid','A_hi'])&(A.mp>=P('2019-05'))]
el = y.groupby(['mp','j']).N.sum().unstack()
print(el.sum(axis=0))
