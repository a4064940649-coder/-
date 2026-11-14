# -*- coding: utf-8 -*-
"""S1 C1 獨立複製：差異追查（凍結後）。用法：python3 -I -B rep_s1_diag.py <scratch_dir>"""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); UP = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import rep_s1_lib as L
P = L.build(sys.argv[1]); R, mi0 = L.load_rev(P); S, D = L.surge(R, mi0); plans, info = L.c1_plans(P, S, mi0)
rs = json.load(open(os.path.join(UP, "results.json")))["runs"]["C1_REVSURGE|twse|cash"]
print("engine", rs["engine"]); print("trades_cost_x2", rs["trades_cost_x2"]); print("trades_full", rs["trades_full"]); print("checks", rs["checks"])
rt = pd.read_csv(os.path.join(UP, "trades.csv.gz"), dtype={"code": str}); rt = rt[rt.run == "C1_REVSURGE|twse|cash"]
byS = {x["signal"]: x for x in info}
def look(code, sig):
    x = byS[sig]; m = x["m"]; s = x["s"]; i = P.cidx.get(code)
    if i is None: print(code, sig, "not in my panel"); return
    order = [int(q) for q in plans[s]]
    rk = order.index(i) if i in order else None
    print(code, sig, "rank", rk, "surge", S[m, i], "univ", bool(L.universe(P, s)[i]), "C", P.C[s, i], "adv60", P.adv60[s, i], "traded", P.traded[s, i],
          "M", P.M[s, i], "barno", P.barno[s, i], "O[t+1]", P.O[s + 1, i], "ref", P.ref[s + 1, i], "lim", L.limit_up(P.ref[s + 1, i]), "AMT", P.AMT[s + 1, i])
for c, sg in (("2376", "2017-06-13"), ("2609", "2017-06-13"), ("3055", "2018-11-13"), ("2419", "2018-11-13"), ("2603", "2019-01-11"), ("6533", "2019-01-11"),
              ("2329", "2019-01-11"), ("2404", "2019-01-11"), ("3557", "2020-02-11"), ("6177", "2024-01-11"), ("3025", "2024-01-11"),
              ("6155", "2024-10-14"), ("2363", "2024-10-14"), ("2363", "2024-11-12"), ("3563", "2024-11-12"), ("3563", "2024-12-11"), ("3653", "2025-01-13"), ("5515", "2025-01-13")):
    look(c, sg)
for sg in ("2017-06-13", "2018-11-13", "2019-01-11", "2020-02-11"):
    x = byS[sg]
    print(sg, "my top10", [P.codes[int(i)] for i in plans[x["s"]][:10]], "r buys", sorted(rt[rt.sig_date == sg].code))
# 整張取整方式
base = L.simulate(P, plans)
b = base["buys"]; b["code"] = b.code.astype(str)
m = rt.merge(b, left_on=["code", "ent_date"], right_on=["code", "buy_date"])
pre = m[m.ent_date < "2020-10-26"]
print("pre-odd matched", len(pre), "same shares", int((pre.shares == pre.sh).sum()), "r more", int((pre.shares > pre.sh).sum()), "r fewer", int((pre.shares < pre.sh).sum()))
print(pre[pre.shares != pre.sh][["code", "ent_date", "shares", "sh", "open", "cost_x", "cost_y"]].head(12).to_string())
post = m[m.ent_date >= "2020-10-26"]
print("post matched", len(post), "share ratio quantiles", (post.shares / post.sh).quantile([0, .05, .5, .95, 1]).round(4).tolist())
# 成本加倍：分項
for nm, kw in (("x2 all", dict(cmult=2.0)),):
    r = L.simulate(P, plans, **kw); s = L.summary(P, r)
    print(nm, round(s["ann_2019"] * 100, 2), round(s["ann_full"] * 100, 2), {k: round(v) for k, v in r["cnt"].items() if k in ("fee", "tax", "slip", "buy", "lot_skip")})
c = base["cnt"]; print("base cost", round(c["fee"]), round(c["tax"]), round(c["slip"]), "total/start", (c["fee"] + c["tax"] + c["slip"]) / 1e6)
rn = pd.read_csv(os.path.join(UP, "navs.csv.gz")); mn = pd.read_csv(os.path.join(HERE, "my_navs.csv.gz")); z = rn.merge(mn, on="date")
q = z["C1_REVSURGE|twse|cash|cost_x2"] / z["c1_x2"]
print(q.iloc[::180].round(4).tolist())
q = z["C1_REVSURGE|twse|cash"] / z["c1"]
print(q.iloc[::180].round(4).tolist())
# ---- 成本加倍：找淨值比值跳動的日子
q = (z["C1_REVSURGE|twse|cash|cost_x2"] / z["c1_x2"]).values
dq = q[1:] / q[:-1] - 1
k = np.where(np.abs(dq) > 0.002)[0]
print([(z.date.iloc[i + 1], round(float(dq[i]) * 100, 2)) for i in k][:40])
x2 = L.simulate(P, plans, cmult=2.0)
rt2 = pd.read_csv(os.path.join(UP, "trades.csv.gz"), dtype={"code": str})
print(sorted(set(rt2.run))[:3])
b2 = x2["buys"]; b0 = base["buys"]
k0 = set(zip(b0.code, b0.buy_date)); k2 = set(zip(b2.code, b2.buy_date))
print("base only", sorted(k0 - k2, key=lambda x: x[1])); print("x2 only", sorted(k2 - k0, key=lambda x: x[1]))
# 停牌假說：訊號日前 60 根內缺列的天數
z5 = np.load(os.path.join(sys.argv[1], "panel.npz")); pres = z5["P"].astype(bool)
for c, sg in (("2609", "2017-06-13"), ("2419", "2018-11-13"), ("2404", "2019-01-11"), ("2329", "2019-01-11"), ("3557", "2020-02-11"), ("3025", "2024-01-11"), ("2363", "2024-10-14"), ("5515", "2025-01-13"), ("3563", "2024-11-12"),
              ("2376", "2017-06-13"), ("6155", "2024-10-14")):
    i = P.cidx[c]; s = L.didx(P, sg)
    print(c, sg, "rows missing in last 60:", int((~pres[s - 59:s + 1, i]).sum()), "no-trade days:", int((~P.traded[s - 59:s + 1, i]).sum()))
