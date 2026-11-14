# -*- coding: utf-8 -*-
"""凍結後審查加跑之二：移動停損幅度的敏感度（貼文口徑逐筆＋可執行逐年），輸出 review_trail.json。"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import rep_lib as L
HERE = os.path.dirname(os.path.abspath(__file__)); S = sys.argv[1]
P = L.load_panel(S); T = P.T; t0 = P.t0; b50 = P.e50.values
i19 = max(i for i, d in enumerate(P.dstr) if d < "2019-01-01")
FP = L.fundamentals(P, True); K = L.tech(P)
sig = {"twse": L.signals(P, FP, K, "twse"), "both": L.signals(P, FP, K, "both")}
out = {}
ye = [t0] + [max(i for i, d in enumerate(P.dstr) if d <= "%d-12-31" % y) for y in range(2017, 2027)]
for tr in (6, 8, 10, 11, 12, 13, 15, 20):
    Kt = L.tech(P, trail=float(tr))
    for u in ("twse", "both"):
        t = L.post_trades(P, sig[u], Kt); cl = t[t.closed == 1]
        nav, n = L.post_tw_curve(P, t, 0.0)
        d = {"n": int(len(t)), "win": float((t.ret_raw > 0).mean()), "mean_div": float(t.ret_div.mean()), "hold": float(cl.hold.mean()),
             "mean_div_per_day_bp": float(t.ret_div.sum() / max(1, (t.b - t.a).sum()) * 1e4), "hold1": float((cl.hold == 1).mean()),
             "tw_ann_nocost": L.ann(1.0, nav[-1], P.dstr[t0], P.dstr[-1])[0], "avg_concurrent": float(n[1:].mean()),
             "top5_share_of_sum": float(t.ret_div.sort_values(ascending=False).head(5).sum() / t.ret_div.sum())}
        r = L.run_exec(P, sig[u], Kt); nv = r["nav"]
        d["exec_full"] = L.ann(nv[t0], nv[-1], P.dstr[t0], P.dstr[-1])[0]; d["exec_2019"] = L.ann(nv[i19], nv[-1], P.dstr[i19], P.dstr[-1])[0]
        d["exec_yearly"] = {str(2017 + k): float(nv[ye[k + 1]] / nv[ye[k]] - 1) for k in range(10)}
        et = r["trades"]; pr = et.proceeds / et.cost - 1
        d["exec_n"] = int(len(et)); d["exec_top5_pnl_share"] = float((et.proceeds - et.cost).sort_values(ascending=False).head(5).sum() / (nv[-1] - nv[t0]))
        # 引擎慣例擾動：不重用當日空位、20 檔
        r2 = L.run_exec(P, sig[u], Kt, same_day_slot=False); d["exec_full_no_same_day_slot"] = L.ann(r2["nav"][t0], r2["nav"][-1], P.dstr[t0], P.dstr[-1])[0]
        r3 = L.run_exec(P, sig[u], Kt, capital=3e6); d["exec_full_3m"] = L.ann(r3["nav"][t0], r3["nav"][-1], P.dstr[t0], P.dstr[-1])[0]
        r4 = L.run_exec(P, sig[u], Kt, cost="stress"); d["exec_full_stress"] = L.ann(r4["nav"][t0], r4["nav"][-1], P.dstr[t0], P.dstr[-1])[0]
        d["exec_2019_stress"] = L.ann(r4["nav"][i19], r4["nav"][-1], P.dstr[i19], P.dstr[-1])[0]
        out["%s_trail%d" % (u, tr)] = d
out["e0050_yearly"] = {str(2017 + k): float(b50[ye[k + 1]] / b50[ye[k]] - 1) for k in range(10)}
out["e0050_full"] = L.ann(b50[t0], b50[-1], P.dstr[t0], P.dstr[-1])[0]; out["e0050_2019"] = L.ann(b50[i19], b50[-1], P.dstr[i19], P.dstr[-1])[0]
json.dump(out, open(os.path.join(HERE, "review_trail.json"), "w"), ensure_ascii=False, indent=1, default=float)
for k, d in out.items():
    if not isinstance(d, dict) or "n" not in d: continue
    print("%-13s n %4d win %.3f mean %.2f%% hold %4.1f bp/d %5.1f TW %.1f%% | exec full %.2f%% 2019+ %.2f%% | noSDS %.2f%% 3m %.2f%% stress %.2f%%/%.2f%% | top5pnl %.0f%%" % (
        k, d["n"], d["win"], d["mean_div"] * 100, d["hold"], d["mean_div_per_day_bp"], d["tw_ann_nocost"] * 100, d["exec_full"] * 100, d["exec_2019"] * 100,
        d["exec_full_no_same_day_slot"] * 100, d["exec_full_3m"] * 100, d["exec_full_stress"] * 100, d["exec_2019_stress"] * 100, d["exec_top5_pnl_share"] * 100))
    print("     yearly", {y: round(v * 100, 1) for y, v in d["exec_yearly"].items()})
print("0050", {y: round(v * 100, 1) for y, v in out["e0050_yearly"].items()}, out["e0050_full"], out["e0050_2019"])
