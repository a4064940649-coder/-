# -*- coding: utf-8 -*-
"""凍結後才執行：比對我的 my_trades.csv.gz／my_navs.csv.gz 與研究員的 trades.csv.gz／navs.csv.gz。
用法：python3 -I -B rep_s5_compare.py   輸出：compare_vs_researcher.json
"""
import os, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
UP = os.path.dirname(HERE)


def main():
    mt = pd.read_csv(os.path.join(HERE, "my_trades.csv.gz"), dtype={"code": str})
    mn = pd.read_csv(os.path.join(HERE, "my_navs.csv.gz"))
    rt = pd.read_csv(os.path.join(UP, "trades.csv.gz"), dtype={"code": str})
    rn = pd.read_csv(os.path.join(UP, "navs.csv.gz"))
    out = {}
    nv = mn.merge(rn, on="date", how="inner")
    out["nav_dates"] = dict(mine=len(mn), researcher=len(rn), common=len(nv))
    e = (nv["e50"] / nv["e50"].iloc[0]) / (nv["0050"] / nv["0050"].iloc[0]) - 1.0
    out["e50_max_rel_diff"] = float(np.abs(e).max())
    for k in ("W70_T10", "W50_T10", "W70_T15"):
        a = nv[k].values; b = nv[k + "|strict"].values
        rel = a / b - 1.0
        bad = np.where(np.abs(rel) > 1e-6)[0]
        d = dict(final_rel_diff=float(rel[-1]), max_abs_rel_diff=float(np.abs(rel).max()),
                 first_diverge_date=nv.date.iloc[bad[0]] if len(bad) else None,
                 first_diverge_rel=float(rel[bad[0]]) if len(bad) else None)
        # 底倉 B 與個股部位
        Bm = nv[k + "_B"].values; Br = nv[k + "|strict|B"].values
        d["B_max_abs_rel_diff"] = float(np.nanmax(np.abs(Bm / Br - 1.0)))
        sm = nv[k + "_stkw"].values; sr = nv[k + "|strict|stockfrac"].values
        d["stockfrac_max_abs_diff"] = float(np.nanmax(np.abs(sm - sr)))
        # 研究員的 B 被夾住天數（由 B ÷ 0050 的變化反推：B/0050 下降的日子，排除拉回日）
        ratio = Br / nv["0050"].values
        ch = ratio[1:] / ratio[:-1] - 1.0
        d["researcher_B_down_days"] = int((ch < -1e-9).sum())
        d["researcher_B_up_days"] = int((ch > 1e-9).sum())
        rm = Bm / nv["e50"].values
        chm = rm[1:] / rm[:-1] - 1.0
        d["mine_B_down_days"] = int((chm < -1e-9).sum()); d["mine_B_up_days"] = int((chm > 1e-9).sum())
        d["researcher_B_down_dates"] = nv.date.iloc[1:][ch < -1e-9].tolist()[:60]
        # 交易逐筆
        x = mt[mt.run == k][["code", "buy_date", "sell_date", "ret", "cost"]].rename(
            columns={"buy_date": "ent_date", "sell_date": "exit_date"})
        y = rt[rt.run == k + "|strict"][["code", "ent_date", "exit_date", "ret", "cost"]]
        mg = x.merge(y, on=["code", "ent_date", "exit_date"], how="outer", suffixes=("_m", "_r"), indicator=True)
        d["trades"] = dict(mine=len(x), researcher=len(y), matched=int((mg._merge == "both").sum()),
                           only_mine=mg[mg._merge == "left_only"][["code", "ent_date", "exit_date"]].values.tolist()[:10],
                           only_researcher=mg[mg._merge == "right_only"][["code", "ent_date", "exit_date"]].values.tolist()[:10])
        both = mg[mg._merge == "both"]
        d["trades"]["max_abs_ret_diff"] = float(np.abs(both.ret_m - both.ret_r).max())
        d["trades"]["max_abs_cost_rel_diff"] = float(np.abs(both.cost_m / both.cost_r - 1.0).max())
        if d["trades"]["max_abs_ret_diff"] > 1e-4:
            w = both.assign(dd=np.abs(both.ret_m - both.ret_r)).sort_values("ent_date")
            w = w[w.dd > 1e-4]
            d["trades"]["first_ret_diff"] = w.head(3)[["code", "ent_date", "exit_date", "ret_m", "ret_r", "cost_m", "cost_r"]].values.tolist()
        out[k] = d
    json.dump(out, open(os.path.join(HERE, "compare_vs_researcher.json"), "w"), ensure_ascii=False, indent=1, default=str)
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
