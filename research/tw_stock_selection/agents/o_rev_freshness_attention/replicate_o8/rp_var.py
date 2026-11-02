import sys
exec(open('/home/user/-/research/tw_stock_selection/agents/o_rev_freshness_attention/replicate_o8/rp_stats.py').read().split("A = D.groupby")[0].replace("f'rp_alpha_s{s}.parquet'", "f'rp_alpha_s{s}sig.parquet'"))
exec("A = D.groupby" + open('/home/user/-/research/tw_stock_selection/agents/o_rev_freshness_attention/replicate_o8/rp_stats.py').read().split("A = D.groupby")[1].split("o, bh, bl, I = block")[0])
print('M50', block('main', 'M50_in', 'M50_out', keys=('post','pre'))[0])
print('FO', block('main', 'FO_hi', 'FO_lo', keys=('post',))[0])
print('FOhi b', summ(sel(slopes('main','FO_hi'),'post')))
