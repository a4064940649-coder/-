# -*- coding: utf-8 -*-
"""S1 C1 獨立複製：第一部分（凍結用）。用法：python3 -I -B rep_s1_run.py <scratch_dir>
輸出（本資料夾）：my_numbers.json、my_numbers.sha256、my_trades_c1.csv.gz、my_targets_c1.json、my_navs.csv.gz
"""
import sys, os, json, hashlib, resource
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import rep_s1_lib as L


def main(scratch):
    P = L.build(scratch)
    R, mi0 = L.load_rev(P)
    S, D = L.surge(R, mi0, null2013=True)
    plans, info = L.c1_plans(P, S, mi0)
    out = dict(meta=dict(T=P.T, N=P.N, first=P.dstr[0], last=P.dstr[-1], t0=P.dstr[P.t0], n_bad=P.n_bad,
                         e50_missing=P.e50_missing, n_signals=len(plans), first_signal=info[0]["signal"],
                         last_signal=info[-1]["signal"]))
    out["bench_0050"] = L.bench(P)
    base = L.simulate(P, plans)
    out["c1_base"] = L.summary(P, base)
    x2 = L.simulate(P, plans, cmult=2.0)
    out["c1_cost_x2"] = L.summary(P, x2)
    tr = base["trades"].sort_values("pnl", ascending=False)
    top5 = tr.head(5)
    out["top5"] = [dict(code=a, buy=b, sell=c, pnl=float(d), ret=float(e)) for a, b, c, d, e in
                   zip(top5.code, top5.buy_date, top5.sell_date, top5.pnl, top5.ret)]
    blk = set(zip(top5.code, top5.buy_date))
    d5 = L.simulate(P, plans, block=blk)
    out["c1_drop5_entry_only"] = L.summary(P, d5)
    span = {}
    for c, a, b in zip(top5.code, top5.tb, top5.ts):
        span.setdefault(c, []).append((int(a), int(b)))
    d5b = L.simulate(P, plans, block_span=span)
    out["c1_drop5_whole_span"] = L.summary(P, d5b)
    # 股池統計
    out["universe"] = dict(avg_n=float(np.mean([x["n_univ"] for x in info])),
                           surge_cov=[(x["ym"], x["n_univ"], x["n_surge"]) for x in info])
    out["signals"] = [dict(ym=x["ym"], d10=x["d10"], deadline=x["deadline"], signal=x["signal"], buy=x["buy"]) for x in info]
    # 目標名單
    tg = {}
    for x in info:
        m = x["m"]; s = x["s"]
        tg[x["signal"]] = [(P.codes[int(i)], round(float(S[m, int(i)]), 4)) for i in plans[s][:10]]
    json.dump(tg, open(os.path.join(HERE, "my_targets_c1.json"), "w"), ensure_ascii=False)
    # 手算 2 檔 SURGE（列出原始營收，供人工驗算）
    hand = []
    for sig, k in (("2020-03-11", 0), ("2023-06-13", 0)):
        x = [y for y in info if y["signal"] == sig]
        if not x:
            x = [info[len(info) // 2]]
        x = x[0]; m = x["m"]; i = int(plans[x["s"]][k])
        revs = {("%04d-%02d" % ((mi0 + q) // 12, (mi0 + q) % 12 + 1)): (None if not R[q, i] == R[q, i] else float(R[q, i]))
                for q in range(m - 36, m + 1)}
        Dw = [float(D[q, i]) for q in range(m - 24, m)]
        mu = sum(Dw) / 24; sd = (sum((v - mu) ** 2 for v in Dw) / 23) ** 0.5
        hand.append(dict(signal=x["signal"], ym=x["ym"], code=P.codes[i], rev=revs, D_m=float(D[m, i]), mu=mu, sd=sd,
                         surge_hand=(float(D[m, i]) - mu) / sd, surge_lib=float(S[m, i])))
    out["hand_surge"] = hand
    out["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    base["trades"].drop(columns=["i"]).to_csv(os.path.join(HERE, "my_trades_c1.csv.gz"), index=False)
    pd.DataFrame(dict(date=P.dstr[P.t0:], c1=base["nav"][P.t0:], c1_x2=x2["nav"][P.t0:], c1_drop5=d5["nav"][P.t0:],
                      e50=P.e50.values[P.t0:])).to_csv(os.path.join(HERE, "my_navs.csv.gz"), index=False)
    fn = os.path.join(HERE, "my_numbers.json")
    json.dump(out, open(fn, "w"), ensure_ascii=False, indent=1)
    h = hashlib.sha256(open(fn, "rb").read()).hexdigest()
    open(os.path.join(HERE, "my_numbers.sha256"), "w").write(h + "  my_numbers.json\n")
    b = out["c1_base"]; z = out["bench_0050"]
    pr = lambda v: "%.2f%%" % (v * 100)
    print("0050", pr(z["ann_2019"]), pr(z["ann_full"]), pr(z["ann_h1"]), pr(z["ann_h2"]))
    for k in ("c1_base", "c1_cost_x2", "c1_drop5_entry_only", "c1_drop5_whole_span"):
        b = out[k]
        print(k, pr(b["ann_2019"]), pr(b["ann_full"]), pr(b["ann_h1"]), pr(b["ann_h2"]), b["mdd_2019"], b["trades"])
    print(out["c1_base"]["cnt"]); print(out["c1_base"]["yearly"]); print(out["top5"]); print(out["meta"], out["peak_rss_mb"], h)


if __name__ == "__main__":
    main(sys.argv[1])
