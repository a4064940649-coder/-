# -*- coding: utf-8 -*-
"""S13 獨立複製：凍結後（my_numbers.sha256 已寫）才跑的比對。
讀研究員的 FINDINGS.md 全文數字（手抄於下）與 cand_train.csv.gz、cand_valid.csv.gz（不讀 results*.json）。
用法：python3 -I -B rep_s13_compare.py
輸出：compare_vs_researcher.json
"""
import os, json, hashlib
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
UP = os.path.dirname(HERE)
FEATS = ["ACC", "REVHI12", "GMYOY", "NEAR52", "RET60", "ADVCHG"]


def main():
    js = open(os.path.join(HERE, "my_numbers.json")).read()
    h = hashlib.sha256(js.encode("utf-8")).hexdigest()
    assert open(os.path.join(HERE, "my_numbers.sha256")).read().split()[0] == h
    m = json.loads(js)
    mine = pd.read_csv(os.path.join(HERE, "my_cands.csv.gz"), dtype={"code": str})
    rs = pd.concat([pd.read_csv(os.path.join(UP, f), dtype={"code": str}).assign(period=p)
                    for f, p in (("cand_train.csv.gz", "train"), ("cand_valid.csv.gz", "valid"))], ignore_index=True)
    out = dict(my_numbers_sha256=h)

    # ---------- 逐筆
    a = mine.merge(rs, left_on=["sig", "code"], right_on=["sig_date", "code"], how="outer", suffixes=("_m", "_r"),
                   indicator=True)
    out["rows"] = dict(mine=int(len(mine)), researcher=int(len(rs)), both=int((a._merge == "both").sum()),
                       only_mine=a.loc[a._merge == "left_only", ["sig", "code", "kind"]].values.tolist(),
                       only_researcher=a.loc[a._merge == "right_only", ["sig_date", "code"]].values.tolist())
    b = a[a._merge == "both"].copy()
    f1r = b.f1_r.astype(str).str.lower().eq("true")
    out["f1_same"] = int((f1r.astype(int) == b.f1_m.astype(int)).sum())
    out["rank_same"] = int((b.rank_m == b.rank_r).sum())
    feat = {}
    for f in FEATS:
        x = b[f + "_m"].values; y = b[f + "_r"].values
        nanm = ~np.isfinite(x); nanr = ~np.isfinite(y)
        ok = ~nanm & ~nanr
        feat[f] = dict(nan_mismatch=int((nanm != nanr).sum()), max_abs_diff=float(np.max(np.abs(x[ok] - y[ok]))),
                       mismatch_rows=b.loc[nanm != nanr, ["sig", "code", f + "_m", f + "_r"]].values.tolist())
    out["features"] = feat
    out["gm_q_same"] = int((b.gm_q_m.fillna("") == b.gm_q_r.fillna("")).sum())
    d = b.exc.values - b.excess.values
    ok = np.isfinite(d)
    out["outcome"] = dict(max_abs_excess_diff=float(np.max(np.abs(d[ok]))), n_compared=int(ok.sum()),
                          n_diff_gt_1e9=int((np.abs(d[ok]) > 1e-9).sum()),
                          hold_same=int((b.hold == b.hold_days).sum()),
                          bw_same=int((b.bw_m.astype(str).str.lower() == b.bw_r.astype(str).str.lower()).sum()),
                          how_vs_kind=pd.crosstab(b.kind, b.how).to_dict(),
                          limup_same=int((b.lim_up.astype(str).str.lower() == b.limup_open.astype(str).str.lower()).sum()))
    bv = b[b.period_m == "valid"]
    out["valid_keep_flag_same"] = int(((bv.ADVCHG_m >= 0.2963874236280218) ==
                                       bv.keep.astype(str).str.lower().eq("true")).sum())

    # ---------- 摘要數字對照（研究員數字抄自 FINDINGS.md）
    R = dict(
        n_train=1419, n_valid=1160, months=(71, 58), f1=(279, 245), bw_thr=0.1953, n_bw=(142, 133),
        auc_train=dict(ACC=0.516, REVHI12=0.538, GMYOY=0.517, NEAR52=0.451, RET60=0.625, ADVCHG=0.602),
        auc_valid=dict(ACC=0.448, REVHI12=0.499, GMYOY=0.496, NEAR52=0.475, RET60=0.626, ADVCHG=0.524),
        rule_t=dict(ACC=[1.60, 2.75, 1.14], REVHI12=[1.29, 1.16, 0.47], GMYOY=[-1.61, -2.06, -1.32],
                    NEAR52=[1.22, -1.31, -1.15], RET60=[0.95, 0.06, 0.86], ADVCHG=[0.77, 1.25, 1.15]),
        frozen=("ADVCHG", 1, 0.2964), V=dict(auc=0.524, bw_keep=0.119, bw_all=0.115, d=-0.0056, t=-0.95, v4=-0.0098,
                                             keep_frac=0.484, mean_keep=0.0128, mean_all=0.0176),
        yearly=dict(y2022=(0.0092, 0.78), y2023=(-0.0028, -0.23), y2024=(-0.0121, -1.05), y2025=(0.0057, 0.35),
                    y2026=(-0.0326, -2.81)),
        f1_valid=dict(n_keep=164, n_drop=81, mean_keep=0.0145, mean_drop=0.1447, t=-2.59, bw_keep=0.140, bw_drop=0.247),
        loser=dict(thr=-0.214, RET60=(0.747, 0.638), ADVCHG=(0.620, 0.566)),
        ret60_q5_sd=(0.365, 0.268), n_trunc=(29, 28), n_limup=(6, 2), train_mean=0.0244, train_median=-0.0144,
        A2=(178, 178))
    tm = m["train"]["univariate"]; vm = m["valid"]["univariate"]; v = m["valid"]["frozen_file"]
    rl = {f: [r["t"] for r in m["train"]["rules"] if r["feature"] == f] for f in FEATS}
    cmp = dict(
        auc_train={f: [round(tm[f]["auc"], 4), R["auc_train"][f], round(tm[f]["auc"] - R["auc_train"][f], 4)] for f in FEATS},
        auc_valid={f: [round(vm[f]["auc"], 4), R["auc_valid"][f], round(vm[f]["auc"] - R["auc_valid"][f], 4)] for f in FEATS},
        rule_t={f: [[round(x, 3) for x in rl[f]], R["rule_t"][f]] for f in FEATS},
        rule_t_max_abs_diff=float(max(abs(x - y) for f in FEATS for x, y in zip(rl[f], R["rule_t"][f]))),
        frozen=dict(mine=m["train"]["frozen"], file=json.load(open(os.path.join(UP, "FROZEN.json")))),
        valid=dict(mine={k: v[k] for k in ("auc_dir", "bw_share_keep", "bw_share_all", "d_mean", "t", "V4_diff",
                                           "keep_frac", "mean_keep", "mean_all", "V1", "V2", "V3", "V4", "pass")},
                   researcher=R["V"]),
        yearly=dict(mine={k: [v["yearly_dm"][k]["mean"], v["yearly_dm"][k]["t"]] for k in v["yearly_dm"]},
                    researcher=R["yearly"]),
        f1_valid=dict(mine=m["f1_subset"]["valid"], researcher=R["f1_valid"]),
        review_A=dict(mine=dict(thr=m["review_A"]["big_loser_threshold"],
                                RET60=(m["review_A"]["auc_loser"]["train"]["RET60"], m["review_A"]["auc_loser"]["valid"]["RET60"]),
                                ADVCHG=(m["review_A"]["auc_loser"]["train"]["ADVCHG"], m["review_A"]["auc_loser"]["valid"]["ADVCHG"]),
                                ret60_q5_sd=(m["review_A"]["ret60_sd_train"]["sd_top"], m["review_A"]["ret60_sd_valid"]["sd_top"])),
                      researcher=R["loser"] | dict(ret60_q5_sd=R["ret60_q5_sd"])),
        profile=dict(mine=dict(n=(m["profile"]["train"]["n_outcome"], m["profile"]["valid"]["n_outcome"]),
                               bw_thr=m["profile"]["big_winner_threshold"],
                               n_bw=(m["profile"]["n_bw_train"], m["profile"]["n_bw_valid"]),
                               n_trunc=(m["profile"]["train"]["n_trunc"], m["profile"]["valid"]["n_trunc"]),
                               n_limup=(m["profile"]["train"]["n_limup_entry"], m["profile"]["valid"]["n_limup_entry"]),
                               train_mean=m["profile"]["train"]["mean_exc"], train_median=m["profile"]["train"]["median_exc"],
                               A2=(m["acceptance"]["A2"]["n"], m["acceptance"]["A2"]["same"])),
                     researcher=dict(n=(R["n_train"], R["n_valid"]), bw_thr=R["bw_thr"], n_bw=R["n_bw"],
                                     n_trunc=R["n_trunc"], n_limup=R["n_limup"], train_mean=R["train_mean"],
                                     train_median=R["train_median"], A2=R["A2"])))
    out["summary"] = cmp
    # 研究員候選檔重算（確認差異只來自 RET60 那一格）
    rt = rs[rs.period == "train"]
    thr = float(np.quantile(rt.excess.values, 0.9))
    bwr = rt.excess >= thr

    def auc(y, x):
        y = np.asarray(y, bool); x = np.asarray(x, float); ok = np.isfinite(x); y = y[ok]; x = x[ok]
        r = pd.Series(x).rank().values; n1 = y.sum(); n0 = (~y).sum()
        return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    out["recomputed_from_researcher_cands"] = dict(bw_thr=thr, auc_train={f: auc(bwr, rt[f]) for f in FEATS})
    # 超額差 > 1e-5 的列：全部是進場或出場落在 2025-06-12（0050 2025-06-11～06-17 分割停牌）
    dd = b.loc[np.abs(b.exc - b.excess) > 1e-5]
    out["excess_diff_rows"] = dict(n=int(len(dd)), ret_max_abs_diff=float(np.max(np.abs(b.ret_m - b.ret_r))),
                                   entry_or_exit_on_20250612=int(((dd.entry == "2025-06-12") | (dd.exit == "2025-06-12")).sum()),
                                   note="0050 停牌日：複製員沿用 rep_s4 attach_e50（開盤÷收盤比沿用 06-10 → 還原開盤＝06-10 開盤），"
                                        "研究員用 06-10 收盤；差 2.1～3.1 個百分點，只影響驗證期 35 筆")
    q10 = float(np.quantile(rt.excess.values, 0.1))
    bl = rt.excess <= q10
    rv = rs[rs.period == "valid"]
    out["review_A_loser_threshold_check"] = dict(
        q10_from_researcher_cands=q10, findings_text="-21.4%",
        auc_RET60_at_q10=(auc(bl, rt.RET60), auc(rv.excess <= q10, rv.RET60)),
        auc_RET60_at_minus_0214=(auc(rt.excess <= -0.214, rt.RET60), auc(rv.excess <= -0.214, rv.RET60)),
        note="FINDINGS 4.1 的 0.747／0.638 與 25.7%／28.4% 對應第 10 百分位數 −13.2%，不是文字寫的 −21.4%")
    json.dump(out, open(os.path.join(HERE, "compare_vs_researcher.json"), "w"), ensure_ascii=False, indent=1,
              default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
    print(json.dumps({k: out[k] for k in ("rows", "f1_same", "rank_same", "gm_q_same", "outcome", "valid_keep_flag_same")},
                     ensure_ascii=False, default=str))
    print(json.dumps(out["features"], ensure_ascii=False, default=str))
    print(json.dumps(cmp["auc_train"]), json.dumps(cmp["auc_valid"]), cmp["rule_t_max_abs_diff"])
    print(json.dumps(out["recomputed_from_researcher_cands"]))


if __name__ == "__main__":
    main()
