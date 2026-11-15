# -*- coding: utf-8 -*-
"""凍結後才執行：my_trades.csv.gz 對研究員 trades.csv.gz（只讀這個檔）逐筆比對 → compare_vs_researcher.json。
用法：python3 -I -B rep_s6_compare.py"""
import os, json
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
t = pd.read_csv(os.path.join(HERE, "..", "trades.csv.gz"), dtype={"code": str})
m = pd.read_csv(os.path.join(HERE, "my_trades.csv.gz"), dtype={"code": str})
out = {}
for cell in ["W10_70", "W7_70", "WN_70", "WN_100", "RW10_70", "RWN_70"]:
    a = t[(t.run == cell + "|strict") & (t.why == "rule")]
    b = m[m.run == cell]
    ka = set(zip(a.code, a.ent_date, a.exit_date)); kb = set(zip(b.code, b.buy_date, b.sell_date))
    j = a.merge(b, left_on=["code", "ent_date", "exit_date"], right_on=["code", "buy_date", "sell_date"], suffixes=("_t", "_m"))
    dr = (j.ret_t - j.ret_m).abs()
    out[cell] = dict(n_theirs=len(a), n_mine=len(b), n_common=len(ka & kb),
                     only_theirs=sorted(ka - kb, key=lambda x: x[1]), only_mine=sorted(kb - ka, key=lambda x: x[1]),
                     pos_ma_mismatch=int((j.pos_ma_t != j.pos_ma_m).sum()),
                     max_abs_ret_diff=float(dr.max()), n_ret_diff_gt_1bp=int((dr > 1e-4).sum()),
                     n_ret_diff_gt_1bp_reduced=int(j[dr > 1e-4].reduced.sum()),
                     max_abs_ret_diff_not_reduced=float(dr[j.reduced == 0].max()))
json.dump(out, open(os.path.join(HERE, "compare_vs_researcher.json"), "w"), ensure_ascii=False, indent=1)
for k, v in out.items():
    print(k, v)
