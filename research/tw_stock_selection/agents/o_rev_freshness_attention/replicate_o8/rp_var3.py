base = '/home/user/-/research/tw_stock_selection/agents/o_rev_freshness_attention/replicate_o8/'
src = open(base + 'rp_stats.py').read()
exec(src.split("o, bh, bl, I = block")[0])
I = slopes('main','A_hi') - slopes('main','A_lo')
for st in ('2021-10','2021-11'):
    k = (I.index >= P(st)) & (I.index <= P('2026-08')); print(st, summ(I[k]))
# FO 族群成員數
x = A[(A.leg=='main')&(A.grp.isin(['FO_lo','FO_mid','FO_hi']))]
print(x.groupby('grp').N.mean())
