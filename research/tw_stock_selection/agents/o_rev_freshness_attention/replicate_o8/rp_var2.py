import sys
base = '/home/user/-/research/tw_stock_selection/agents/o_rev_freshness_attention/replicate_o8/'
src = open(base + 'rp_stats.py').read()
for tag in ('', 'sig'):
    exec(src.split("A = D.groupby")[0].replace("f'rp_alpha_s{s}.parquet'", "f'rp_alpha_s{s}" + tag + ".parquet'"))
    exec("A = D.groupby" + src.split("A = D.groupby")[1].split("o, bh, bl, I = block")[0])
    for st in ('2021-10', '2021-11'):
        bh, bl = slopes('main','FO_hi'), slopes('main','FO_lo'); I2 = (bh-bl)
        k = (I2.index >= P(st)) & (I2.index <= P('2026-08'))
        print(tag, st, summ(I2[k]), 'b_hi', summ(bh[k]), 'b_lo', summ(bl[k]))
