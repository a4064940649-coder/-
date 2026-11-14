# -*- coding: utf-8 -*-
"""S1 C1 獨立複製：凍結後與研究員輸出比對。用法：python3 -I -B rep_s1_compare.py
讀 my_numbers.json（已凍結）、my_trades_c1.csv.gz、my_navs.csv.gz 與研究員的 results.json、trades.csv.gz、navs.csv.gz。
輸出 compare_vs_researcher.json。"""
import os, json, hashlib
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); UP = os.path.dirname(HERE)
fn = os.path.join(HERE, "my_numbers.json")
assert hashlib.sha256(open(fn, "rb").read()).hexdigest() == open(os.path.join(HERE, "my_numbers.sha256")).read().split()[0]
my = json.load(open(fn)); rs = json.load(open(os.path.join(UP, "results.json")))
R = rs["runs"]["C1_REVSURGE|twse|cash"]
out = {}
print(json.dumps({k: R[k] for k in ("windows", "cost_x2", "drop_top5", "top5_trades", "trades_full", "trades_cost_x2", "engine", "yearly")}, ensure_ascii=False)[:5000])
# 逐筆
rt = pd.read_csv(os.path.join(UP, "trades.csv.gz"), dtype={"code": str})
rt = rt[rt.run == "C1_REVSURGE|twse|cash"].copy()
mt = pd.read_csv(os.path.join(HERE, "my_trades_c1.csv.gz"), dtype={"code": str})
print(rt.why.value_counts().to_dict())
rc = rt[rt.why != "open"] if "open" in set(rt.why) else rt
m = rt.merge(mt, left_on=["code", "ent_date"], right_on=["code", "buy_date"], how="outer", indicator=True, suffixes=("_r", "_m"))
print(m._merge.value_counts().to_dict())
both = m[m._merge == "both"]
out["n_r_rows"] = int(len(rt)); out["n_m_closed"] = int(len(mt)); out["n_matched"] = int(len(both))
out["only_r"] = m[m._merge == "left_only"][["code", "sig_date", "ent_date", "exit_date", "why", "ret_r"]].to_dict("records")
out["only_m"] = m[m._merge == "right_only"][["code", "buy_date", "sell_date", "ret_m"]].to_dict("records")
out["exit_date_diff"] = int((both.exit_date != both.sell_date).sum())
d = (both.ret_r - both.ret_m)
out["ret_diff"] = dict(max_abs=float(d.abs().max()), mean=float(d.mean()), n_gt_10bp=int((d.abs() > 0.001).sum()))
dc = (both.cost_r / both.cost_m - 1)
out["cost_diff"] = dict(max_abs=float(dc.abs().max()), n_gt_1pct=int((dc.abs() > 0.01).sum()))
x = both[(d.abs() > 0.001) | (both.exit_date != both.sell_date) | (dc.abs() > 0.01)].sort_values("ent_date")
print(x[["code", "ent_date", "exit_date", "sell_date", "shares", "cost_r", "cost_m", "ret_r", "ret_m", "why"]].head(25).to_string())
print(out)
# 淨值
rn = pd.read_csv(os.path.join(UP, "navs.csv.gz")); mn = pd.read_csv(os.path.join(HERE, "my_navs.csv.gz"))
z = rn.merge(mn, on="date")
for a, b in (("C1_REVSURGE|twse|cash", "c1"), ("C1_REVSURGE|twse|cash|cost_x2", "c1_x2"), ("C1_REVSURGE|twse|cash|drop5", "c1_drop5"), ("0050", "e50")):
    q = z[a] / z[a].iloc[0] / (z[b] / z[b].iloc[0]) - 1
    k = np.where(q.abs() > 1e-4)[0]
    print(a, "final ratio-1 = %.4f" % q.iloc[-1], "first >1bp:", z.date.iloc[k[0]] if len(k) else None, "max abs %.4f" % q.abs().max())
    out["nav_" + b] = dict(final_ratio_m1=float(q.iloc[-1]), first_gt_1bp=z.date.iloc[k[0]] if len(k) else None)
# 5 個換股日的名單
tg = json.load(open(os.path.join(HERE, "my_targets_c1.json")))
lists = []
for sd in ("2016-02-16", "2018-09-11", "2020-03-11", "2023-06-13", "2026-08-11"):
    rb = set(rt[rt.sig_date == sd].code)
    mine = [c for c, _ in tg.get(sd, [])]
    mb = set(mt[mt.buy_date > sd].sort_values("buy_date").groupby("code").head(0).code)  # placeholder
    lists.append(dict(signal=sd, my_top10=mine, researcher_buys=sorted(rb), researcher_buys_in_my_top10=len(rb & set(mine)),
                      researcher_buys_not_in_my_top10=sorted(rb - set(mine))))
    print(sd, mine, sorted(rb), len(rb & set(mine)), sorted(rb - set(mine)))
out["lists"] = lists
# 全部訊號日：研究員的買進是否都在我的前 10 名
bad = 0; tot = 0
for sd, g in rt.groupby("sig_date"):
    s = set(c for c, _ in tg.get(sd, []))
    tot += len(g); bad += len(set(g.code) - s)
out["all_signals"] = dict(n_sig_r=int(rt.sig_date.nunique()), n_sig_m=len(tg), r_buys=tot, r_buys_not_in_my_top10=bad,
                          sig_days_same=sorted(set(rt.sig_date)) == sorted(set(tg) & set(rt.sig_date)))
print(out["all_signals"])
json.dump(out, open(os.path.join(HERE, "compare_vs_researcher.json"), "w"), ensure_ascii=False, indent=1)
