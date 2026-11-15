# -*- coding: utf-8 -*-
"""凍結後比對：我的逐筆交易（my_trades.csv.gz）對研究員 trades.csv.gz。
用法：python3 -I -B rep_s4_compare.py    輸出：compare_vs_researcher.json（本資料夾）
"""
import os, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
THEIR = os.path.join(os.path.dirname(HERE), "trades.csv.gz")

PAIRS = {"E3p_cash": "E3'|cash|strict", "E3p_e50": "E3'|e50|strict", "F1_e50": "F1|e50|strict"}


def main():
    me = pd.read_csv(os.path.join(HERE, "my_trades.csv.gz"), dtype={"code": str})
    th = pd.read_csv(THEIR, dtype={"code": str})
    out = {"why_values": th.why.value_counts().to_dict()}
    for mine, theirs in PAIRS.items():
        a = me[me.run == mine].copy()
        b = th[th.run == theirs].copy()
        a["k"] = a.code + "_" + a.buy_date; b["k"] = b.code + "_" + b.ent_date
        bc = b[b.why == "rule"].copy()          # 研究員檔案含期末未平倉（why=open_end），只比已平倉
        m = a.merge(bc, on="k", how="outer", suffixes=("_me", "_th"), indicator=True)
        only_me = m[m._merge == "left_only"].sort_values("buy_date")
        only_th = m[m._merge == "right_only"].sort_values("ent_date")
        both = m[m._merge == "both"]
        same_exit = (both.sell_date == both.exit_date)
        dret = (both.ret_me - both.ret_th).abs()
        dsh = (both.sh != both.shares)
        first_diff = None
        cand = []
        if len(only_me): cand.append((only_me.buy_date.iloc[0], "only_me", only_me.iloc[0][["code_me", "buy_date", "sell_date", "ret_me", "cost_me"]].to_dict()))
        if len(only_th): cand.append((only_th.ent_date.iloc[0], "only_th", only_th.iloc[0][["code_th", "ent_date", "exit_date", "ret_th", "cost_th"]].to_dict()))
        bad = both[~same_exit]
        if len(bad): cand.append((bad.buy_date.min(), "exit_diff", bad.sort_values("buy_date").iloc[0][["k", "sell_date", "exit_date"]].to_dict()))
        shd = both[dsh]
        first_share = shd.sort_values("buy_date").iloc[0][["k", "sh", "shares", "cost_me", "cost_th"]].to_dict() if len(shd) else None
        if cand:
            first_diff = sorted(cand, key=lambda x: x[0])[0]
        out[mine] = dict(n_me=int(len(a)), n_th_rows=int(len(b)), n_th_closed=int(len(bc)),
                         th_why=b.why.value_counts().to_dict(), matched=int(len(both)),
                         only_me=only_me[["code_me", "buy_date", "sell_date", "ret_me", "cost_me"]].to_dict("records"),
                         only_th=only_th[["code_th", "ent_date", "exit_date", "ret_th", "cost_th", "why"]].to_dict("records"),
                         exit_mismatch=int((~same_exit).sum()), max_abs_ret_diff=float(dret.max()) if len(dret) else None,
                         n_ret_diff_gt_1bp=int((dret > 1e-4).sum()), n_share_diff=int(dsh.sum()), first_share_diff=first_share,
                         first_diff=first_diff,
                         th_open=b[b.why != "rule"][["code", "ent_date", "ret"]].to_dict("records"))
    json.dump(out, open(os.path.join(HERE, "compare_vs_researcher.json"), "w"), ensure_ascii=False, indent=1, default=str)
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
