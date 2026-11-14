# -*- coding: utf-8 -*-
"""凍結後才執行：把我方逐筆與研究員的 signals.csv.gz／trades.csv.gz 逐筆比對。"""
import sys, os, json, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import rep_lib as L
HERE = os.path.dirname(os.path.abspath(__file__)); UP = os.path.dirname(HERE)
assert hashlib.sha256(open(os.path.join(HERE, "my_numbers.json"), "rb").read()).hexdigest() == open(os.path.join(HERE, "my_numbers.sha256")).read().split()[0], "my_numbers 未凍結或已變動"
S = sys.argv[1]
P = L.load_panel(S); K = L.tech(P)
out = {}
# 訊號
FP = L.fundamentals(P, True)
sig = L.signals(P, FP, K, "both")
tt, ii = np.where(sig)
mine = pd.DataFrame({"date": [P.dstr[t] for t in tt], "code": [P.codes[i] for i in ii], "EpsQYoY": FP[3][tt, ii], "EpsCAGR": FP[4][tt, ii],
                     "RevYoY3": FP[5][tt, ii], "FromHigh": K["fromhigh"][tt, ii], "RS": K["excess"][tt, ii], "AvgQty": K["avgq"][tt, ii]})
his = pd.read_csv(os.path.join(UP, "signals.csv.gz"), dtype={"code": str})
m = mine.merge(his, on=["date", "code"], how="outer", suffixes=("_m", "_h"), indicator=True)
out["signals"] = {"mine": len(mine), "his": len(his), "both": int((m._merge == "both").sum()), "only_mine": int((m._merge == "left_only").sum()),
                  "only_his": int((m._merge == "right_only").sum())}
b = m[m._merge == "both"]
for c in ["EpsQYoY", "EpsCAGR", "RevYoY3", "FromHigh", "RS", "AvgQty"]:
    d = (b[c + "_m"] - b[c + "_h"]).abs()
    out["signals"]["maxabsdiff_" + c] = float(d.max())
# 逐筆
ht = pd.read_csv(os.path.join(UP, "trades.csv.gz"), dtype={"code": str})
def cmp(mt, hv, name, mcol=("buy_date", "sell_date"), rcols=()):
    h = ht[ht.variant == hv].copy()
    a = mt.rename(columns={mcol[0]: "ent_date", mcol[1]: "exit_date"})
    j = a.merge(h, on=["code", "ent_date", "exit_date"], how="outer", indicator=True, suffixes=("_m", "_h"))
    r = {"mine": len(a), "his": len(h), "both": int((j._merge == "both").sum()), "only_mine": int((j._merge == "left_only").sum()),
         "only_his": int((j._merge == "right_only").sum())}
    bb = j[j._merge == "both"]
    for mc, hc in rcols:
        r["maxabsdiff_%s_vs_%s" % (mc, hc)] = float((bb[mc] - bb[hc]).abs().max())
    d = j[j._merge != "both"].sort_values(["ent_date", "code"])
    r["first_diff"] = d.head(3)[["code", "ent_date", "exit_date", "_merge"]].astype(str).to_dict("records")
    out[name] = r
for u, p, hv in [("twse", True, "POST_TWSE_PIT"), ("both", True, "POST_ALL_PIT"), ("twse", False, "POST_TWSE_NONPIT"), ("both", False, "POST_ALL_NONPIT")]:
    F = FP if p else L.fundamentals(P, False)
    tr = L.post_trades(P, L.signals(P, F, K, u), K)
    cmp(tr, hv, hv, rcols=(("ret_raw", "ret_px"), ("ret_div", "ret_tr")))
r = L.run_exec(P, L.signals(P, FP, K, "twse"), K)
et = r["trades"].copy()
# 期末未平倉補進去（研究員的 EXEC_MAIN 1,464 筆含 open_end 10 筆）
h = ht[ht.variant == "EXEC_MAIN"].copy()
j = et.rename(columns={"buy_date": "ent_date", "sell_date": "exit_date"}).merge(h[h.why != "open_end"], on=["code", "ent_date", "exit_date"], how="outer", indicator=True, suffixes=("_m", "_h"))
bb = j[j._merge == "both"]
out["EXEC_MAIN"] = {"mine_closed": len(et), "his_closed": int((h.why != "open_end").sum()), "his_open_end": int((h.why == "open_end").sum()), "mine_open": int(r["open_pos"]),
                    "both": int(len(bb)), "only_mine": int((j._merge == "left_only").sum()), "only_his": int((j._merge == "right_only").sum()),
                    "maxabsdiff_cost": float((bb.cost_m - bb.cost_h).abs().max()), "maxabsdiff_proceeds": float((bb.proceeds_m - bb.proceeds_h).abs().max()),
                    "maxreldiff_proceeds": float(((bb.proceeds_m - bb.proceeds_h).abs() / bb.proceeds_h).max()),
                    "sum_cost_m": float(bb.cost_m.sum()), "sum_cost_h": float(bb.cost_h.sum()), "sum_proc_m": float(bb.proceeds_m.sum()), "sum_proc_h": float(bb.proceeds_h.sum()),
                    "end_nav_mine": float(r["nav"][-1]),
                    "first_diff": j[j._merge != "both"].sort_values(["ent_date", "code"]).head(3)[["code", "ent_date", "exit_date", "_merge"]].astype(str).to_dict("records")}
d = (bb.proceeds_m - bb.proceeds_h).abs()
k = d.sort_values(ascending=False).head(5).index
out["EXEC_MAIN"]["largest_proceeds_diffs"] = bb.loc[k, ["code", "ent_date", "exit_date", "cost_m", "cost_h", "proceeds_m", "proceeds_h", "why"]].astype(str).to_dict("records")
json.dump(out, open(os.path.join(HERE, "compare_vs_researcher.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
