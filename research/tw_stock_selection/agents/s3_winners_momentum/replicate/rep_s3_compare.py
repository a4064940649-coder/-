# -*- coding: utf-8 -*-
"""凍結後才執行：逐筆比對 my_trades.csv.gz 與研究員 trades.csv.gz（只讀這一個研究員輸出檔）。
用法：python3 -I -B rep_s3_compare.py  → compare_vs_researcher.json"""
import os, json
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
t = pd.read_csv(os.path.join(HERE, "..", "trades.csv.gz"), dtype={"code": str})
m = pd.read_csv(os.path.join(HERE, "my_trades.csv.gz"), dtype={"code": str})
out = {}
for c in ("D1", "E1", "E2", "E3"):
    a = t[(t.run == c + "|cash|strict") & (t.why != "open_end")]
    b = m[m.cell == c]
    ka = set(zip(a.code, a.ent_date, a.exit_date)); kb = set(zip(b.code, b.buy_date, b.sell_date))
    j = a.rename(columns={"ent_date": "buy_date", "exit_date": "sell_date"}).merge(b, on=["code", "buy_date", "sell_date"])
    dsh = j[j.shares != j.sh].sort_values("buy_date")
    out[c] = dict(n_res=len(a), n_me=len(b), only_res=sorted(ka - kb, key=lambda x: x[1]),
                  only_me=sorted(kb - ka, key=lambda x: x[1]),
                  first_share_diff=dsh[["code", "buy_date", "shares", "sh"]].head(1).to_dict("records"),
                  n_share_diff=int(len(dsh)),
                  dust_lt1000_res=int((a.cost < 1000).sum()), dust_lt1000_me=int((b.cost < 1000).sum()),
                  dust_res=a[a.cost < 1000][["code", "ent_date", "exit_date", "shares", "cost", "ret"]].to_dict("records"))
json.dump(out, open(os.path.join(HERE, "compare_vs_researcher.json"), "w"), ensure_ascii=False, indent=1, default=str)
for c, v in out.items():
    print(c, v["n_res"], v["n_me"], v["only_res"], v["only_me"], v["first_share_diff"], v["n_share_diff"])
