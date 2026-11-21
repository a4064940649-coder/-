#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 產生網站「環境溫度計」資料檔 thermo.json（JSON：{rows, meta}，格式同 build_site_data.py 的其他資料檔）。2026-11-21 加。
# 定義與門檻照 xq_scripts/README.md 第 32.2 節（S83，R83 已獨立重現）。**只描述當下，不能拿來進出場。**
#
# 用法：python3 -I -B xq_scripts/site/make_thermo.py <輸出 thermo.json> <官方日K倉庫> [選項]
#   --raw <倉庫>      研究中心的官方日 K 封存倉庫（預設 research/tw_stock_selection/raw/jb7k6295r7-bot_tw-stock-data）。
#                     同一天兩個倉庫都有時用 <官方日K倉庫> 的（讀法同 make_list.py：只讀 git HEAD、不連網；本機沒有的 blob 改用封存）。
#   --twt49u-dir <資料夾>  除權息封存（預設 research/tw_stock_selection/raw/twt49u/twse-exright）。含息漲幅要用它做除權息調整。
#   --hist <csv>      研究存檔 thermometer_monthly.csv（預設 agents/s83_environment_thermometer/）。官方日 K 2015-01-05 才開始，
#                     2015-04 以前的月份本程式算不出來（股池要 60 天、飆股密度要再往前 63 天），直接沿用這份存檔並在「來源」欄標明。
#   --asof YYYY-MM-DD 只用到這一天為止的資料（驗證用，例如 2026-10-22＝研究存檔的最後一天）。
#   --check <輸出.json>  把本程式算出來的月份和研究存檔逐月比對（三個讀數差 < 1e-9、冷溫熱相同），結果寫到這個檔；不符時結束碼 3。
#   --reduce-fix      減資／面額變更恢復買賣日用事件參考價（候選修正，預設關；同 cslib.REDUCE_FIX）。
#
# 三個溫度計（股池＝上市、四碼普通股、非 91 開頭、非創新板、當天有成交、收盤 ≥ 10 元、近 60 個交易日都在日 K 內且 60 日平均成交值 ≥ 1,000 萬；
# 每個日曆月最後一個交易日收盤後算；最新月份還沒結束時算到資料最後一天，「完整月」欄＝否）：
#   飆股密度＝股池內近 63 個交易日含息漲幅 > 50% 的檔數比例。
#   大漲家數比例＝近 21 個交易日，每天股池內單日含息漲幅 ≥ 6.5% 檔數比例的平均。
#   右尾厚度＝股池內近 21 個交易日含息報酬的第 95 百分位 − 中位數。
# 門檻（只用 2008～2015 訂；agents/s83_environment_thermometer/results_posthoc.json 的 thermometer_def）：低於「冷」門檻＝冷、高於「熱」門檻＝熱、其餘＝溫。
# 合成：三項中 ≥ 2 項高於熱門檻＝熱；≥ 2 項低於冷門檻＝冷；其餘＝溫。
# 含息日報酬的算法同回測引擎 cslib.Panel（除權息日用 TWT49U 參考價；直接呼叫 cslib.exright_fix_chg，不另外寫一份）。需要 numpy／pandas。
import calendar, csv, datetime, importlib.util, io, json, os, re, subprocess, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "research", "tw_stock_selection")
RAW_DEFAULT = os.path.join(RES, "raw", "jb7k6295r7-bot_tw-stock-data")
TWT49U_DEFAULT = os.path.join(RES, "raw", "twt49u", "twse-exright")
HIST_DEFAULT = os.path.join(RES, "agents", "s83_environment_thermometer", "thermometer_monthly.csv")
CSLIB = os.path.join(RES, "agents", "o_ext_canslim_xs", "cslib.py")
ENV = dict(os.environ, GIT_NO_LAZY_FETCH="1")
# 門檻（比例，不是百分比）：S83 訓練段（2008～2015）的下、上三分位；寫死，不隨資料變動
THR = {"SPURT3": (0.011768219832735898, 0.027973994025654494),
       "BIGUP21": (0.015926705352397565, 0.025147846757934165),
       "TAIL21": (0.15294895455956783, 0.18877802941765767)}
NAME = {"SPURT3": "飆股密度", "BIGUP21": "大漲家數比例", "TAIL21": "右尾厚度"}
DEFN = {"SPURT3": "股池內近 63 個交易日含息漲幅 > 50% 的檔數比例",
        "BIGUP21": "近 21 個交易日，每天股池內單日漲幅 ≥ 6.5% 檔數比例的平均",
        "TAIL21": "股池內近 21 個交易日個股報酬的第 95 百分位減中位數"}
KEYS = ["SPURT3", "BIGUP21", "TAIL21"]
INNOV_PAT = re.compile(r"-(?:KY)?創$")
MIN_K = 79            # 第 k 列（從 0 算）要 ≥ 79 才算：大漲家數比例要 k−20 那天已有股池（60 日均成交值，第 59 列起），飆股密度要 k−63
LIMIT = "只描述當下，不能拿來進出場：三個讀數只說明「現在市場上飆股多不多」，不能預告下個月（15 個指標 × 3 個期間的領先檢定，校正後 0 個通過；策略超額不延續）。冷的時候只下修期望，不是停掉策略（場外等待代價更大）。"


def opt(name, default=None, flag=False):
    if name in sys.argv:
        i = sys.argv.index(name)
        if flag:
            del sys.argv[i]; return True
        v = sys.argv[i + 1]; del sys.argv[i:i + 2]; return v
    return False if flag else default


def git(repo, *a):
    return subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True, check=True, env=ENV).stdout


def ls_daily(repo):
    out = {}
    for ln in git(repo, "ls-tree", "HEAD", "data/universe/daily/").splitlines():
        meta, path = ln.split("\t"); f = path.rsplit("/", 1)[1]
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}\.csv", f):
            out[f[:10]] = meta.split()[2]
    return out


def local_objects(repo):
    return set(git(repo, "cat-file", "--batch-all-objects", "--batch-check=%(objectname)", "--unordered").split())


def read_blobs(repo, shas):
    p = subprocess.Popen(["git", "-C", repo, "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=ENV)
    for s in shas:
        p.stdin.write((s + "\n").encode()); p.stdin.flush()
        hd = p.stdout.readline().decode().split()
        if len(hd) != 3 or hd[1] != "blob":
            raise SystemExit(f"讀不到日 K 檔（{repo} {s}）")
        b = p.stdout.read(int(hd[2])); p.stdout.read(1)
        yield b
    p.stdin.close(); p.wait()


def load_panel(repos, asof=None):
    """repos：由舊到新。同一天後面的倉庫覆蓋前面的（那個 blob 本機有才換）。回傳 dict（dates、codes、C、AMT、CHG、MK、INNOV）與來源統計。"""
    plan = {}; diff = []
    for rp in repos:
        have = local_objects(rp)
        for d, sha in ls_daily(rp).items():
            if sha not in have or (asof and d > asof):
                continue
            if d in plan and plan[d][1] != sha:
                diff.append(d)
            if d not in plan or plan[d][1] != sha:
                plan[d] = (rp, sha)
    dates = sorted(plan)
    days = {}
    for rp in repos:
        want = [d for d in dates if plan[d][0] == rp]
        for d, b in zip(want, read_blobs(rp, [plan[d][1] for d in want])):
            x = pd.read_csv(io.BytesIO(b), dtype={"stock_id": str}, usecols=["stock_id", "name", "market", "close", "amount", "change"])
            x = x[x["stock_id"].str.fullmatch(r"[1-9]\d{3}") & x["market"].isin(["twse", "tpex"])].drop_duplicates("stock_id", keep="last")
            days[d] = (x["stock_id"].to_numpy(), np.where(x["market"].to_numpy() == "twse", 1, 2).astype(np.int8),
                       x[["close", "amount", "change"]].apply(pd.to_numeric, errors="coerce").to_numpy(float),
                       x["name"].astype(str).str.strip().str.contains(INNOV_PAT).to_numpy())
    codes = np.array(sorted(set(np.concatenate([days[d][0] for d in dates]))))
    ci = {c: j for j, c in enumerate(codes)}
    T, N = len(dates), len(codes)
    C = np.full((T, N), np.nan); AMT = C.copy(); CHG = C.copy(); MK = np.zeros((T, N), np.int8); INNOV = np.zeros((T, N), bool)
    for t, d in enumerate(dates):
        cs, mk, v, inn = days[d]
        j = np.array([ci[c] for c in cs])
        C[t, j], AMT[t, j], CHG[t, j] = v[:, 0], v[:, 1], v[:, 2]
        MK[t, j] = mk; INNOV[t, j] = inn
    src = {os.path.abspath(rp): sum(1 for d in dates if plan[d][0] == rp) for rp in repos}
    return dict(dates=dates, codes=codes, C=C, AMT=AMT, CHG=CHG, MK=MK, INNOV=INNOV), dict(days_by_repo=src, blob_diff_days=sorted(set(diff)))


def indicators(P, cslib, twt49u_dir, reduce_fix=False):
    """回傳（每月一列的 list、資訊 dict）。含息日報酬 R、還原指數 G、股池的算法逐行同 cslib.Panel／stlib.load／stlib.universe。"""
    dates = P["dates"]; IDX = pd.DatetimeIndex(dates); C, MK = P["C"], P["MK"]
    ev = cslib.load_twt49u(twt49u_dir)
    CHG, xinfo = cslib.exright_fix_chg(IDX, P["codes"], C, P["CHG"], MK, ev)
    rinfo = None
    if reduce_fix:
        CHG, rinfo = cslib.reduce_fix_chg(IDX, P["codes"], C, CHG, MK, cslib.load_reduce())
    T, N = C.shape
    has = np.isfinite(C) & (C > 0)
    listed = MK > 0
    ref = C - CHG
    ref[~(ref > 0)] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        R = C / ref - 1
        R[np.abs(R) > 0.5] = np.nan
        CF = pd.DataFrame(C).ffill().to_numpy()
        prevC = np.vstack([np.full((1, N), np.nan), CF[:-1]])
        R = np.where(np.isfinite(R), R, np.where(has, C / prevC - 1, 0.0))
    R = np.where(np.isfinite(R), R, 0.0)
    G = np.cumprod(1 + R, axis=0)
    amt = np.where(listed, np.nan_to_num(P["AMT"]), np.nan)
    V60 = pd.DataFrame(amt).rolling(60, min_periods=60).mean().to_numpy()
    not91 = np.array([not c.startswith("91") for c in P["codes"]])
    with np.errstate(invalid="ignore"):
        U = (V60 >= 1e7) & (C >= 10) & has & not91[None, :] & (MK == 1) & ~P["INNOV"]
        big = (R >= 0.065) & U
    bigshare = big.sum(1) / np.maximum(U.sum(1), 1)
    me = pd.Series(np.arange(T), index=IDX).groupby(IDX.to_period("M")).last()
    rows = []
    for m, k in me.items():
        k = int(k)
        if k < MIN_K:
            continue
        u = U[k]
        with np.errstate(invalid="ignore", divide="ignore"):
            r3 = G[k] / G[k - 63] - 1; r1 = G[k] / G[k - 21] - 1
        v3 = u & np.isfinite(r3); v1 = u & np.isfinite(r1)
        rows.append(dict(month=str(m), date=dates[k], n_univ=int(u.sum()), SPURT3=float((r3[v3] > 0.5).mean()), BIGUP21=float(bigshare[k - 20:k + 1].mean()),
                         TAIL21=float(np.quantile(r1[v1], 0.95) - np.median(r1[v1]))))
    return rows, dict(exright=xinfo, reduce=rinfo, twt49u_last=str(ev["date"].max().date()))


def state_of(x, k):
    lo, hi = THR[k]
    return "熱" if x > hi else ("冷" if x < lo else "溫")


def classify(r):
    hi = sum(r[k] > THR[k][1] for k in KEYS); lo = sum(r[k] < THR[k][0] for k in KEYS)
    return hi, lo, ("熱" if hi >= 2 else ("冷" if lo >= 2 else "溫"))


def month_complete(month, last_day):
    """最新月份算不算完整：資料最後一天之後，這個月已經沒有週一到週五（遇到月底連假會保守地判成「否」）。"""
    y, m = int(month[:4]), int(month[5:7]); d0 = int(last_day[8:10])
    return not any(datetime.date(y, m, d).weekday() < 5 for d in range(d0 + 1, calendar.monthrange(y, m)[1] + 1))


def main():
    raw = opt("--raw", RAW_DEFAULT); twt = opt("--twt49u-dir", TWT49U_DEFAULT); hist_csv = opt("--hist", HIST_DEFAULT)
    asof = opt("--asof"); check = opt("--check"); reduce_fix = opt("--reduce-fix", flag=True)
    if len(sys.argv) != 3:
        raise SystemExit("用法：make_thermo.py <輸出 thermo.json> <官方日K倉庫> [--raw 倉庫] [--twt49u-dir 資料夾] [--hist csv] [--asof 日期] [--check 輸出.json] [--reduce-fix]")
    out_path, tsd = sys.argv[1], sys.argv[2]
    sp = importlib.util.spec_from_file_location("cslib", CSLIB); cslib = importlib.util.module_from_spec(sp); sp.loader.exec_module(cslib)
    repos = [raw] + ([tsd] if os.path.abspath(tsd) != os.path.abspath(raw) else [])
    P, src = load_panel(repos, asof)
    calc, info = indicators(P, cslib, twt, reduce_fix)
    last_day = P["dates"][-1]
    if not calc:
        raise SystemExit("官方日 K 天數不夠，算不出任何月份")
    # 研究存檔（只拿本程式算不出來的早期月份；另供 --check 比對）
    hist = {}
    with open(hist_csv, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            hist[r[""]] = dict(month=r[""], date=r["date"], SPURT3=float(r["SPURT3"]), BIGUP21=float(r["BIGUP21"]), TAIL21=float(r["TAIL21"]), state=r["state"],
                               n_high=int(r["n_high"]), n_low=int(r["n_low"]))
    first_calc = calc[0]["month"]
    for h in hist.values():                                # 門檻寫死在本檔：確認用它重新分類研究存檔，冷溫熱完全相同
        if classify(h) != (h["n_high"], h["n_low"], h["state"]):
            raise SystemExit(f"門檻對不上研究存檔：{h['month']}")
    if check:
        cmp_rows = [(c, hist[c["month"]]) for c in calc if c["month"] in hist and c["date"] == hist[c["month"]]["date"]]
        skipped = [c["month"] for c in calc if c["month"] in hist and c["date"] != hist[c["month"]]["date"]]
        md = {k: max(abs(c[k] - h[k]) for c, h in cmp_rows) for k in KEYS}
        st_diff = [c["month"] for c, h in cmp_rows if classify(c)[2] != h["state"]]
        bad = [dict(month=c["month"], **{k: c[k] - h[k] for k in KEYS}) for c, h in cmp_rows if any(abs(c[k] - h[k]) >= 1e-9 for k in KEYS)]
        res = dict(asof=last_day, hist_csv=os.path.abspath(hist_csv), months_compared=len(cmp_rows), first=cmp_rows[0][0]["month"], last=cmp_rows[-1][0]["month"],
                   max_abs_diff=md, n_months_diff_ge_1e9=len(bad), months_diff=bad[:20], state_diff_months=st_diff,
                   months_same_month_but_other_day=skipped, months_only_in_hist_not_recomputed=[m for m in hist if m < first_calc][:1] + [m for m in hist if m < first_calc][-1:],
                   n_months_only_in_hist=sum(1 for m in hist if m < first_calc), reduce_fix=bool(reduce_fix),
                   **{"pass": bool(not bad and not st_diff and len(cmp_rows) > 0)})
        with open(check, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=1)
        print("比對", json.dumps({k: v for k, v in res.items() if k != "months_diff"}, ensure_ascii=False))
    rows = []
    for m in sorted(hist):
        if m < first_calc:
            rows.append(dict(hist[m], n_univ=None, src="研究存檔（S83）"))
    rows += [dict(c, src="官方日K計算") for c in calc]
    out = []
    for i, r in enumerate(rows):
        hi, lo, st = classify(r)
        full = True if i < len(rows) - 1 else month_complete(r["month"], r["date"])
        o = {"月份": r["month"], "資料日": r["date"], "完整月": "是" if full else "否"}
        # 讀數存到小數 2 位＝頁面顯示的精度（冷溫熱用沒有四捨五入的原值判斷，所以讀數剛好貼著門檻時，看起來相等但狀態不同是正常的）
        for k in KEYS:
            o[NAME[k] + "%"] = round(r[k] * 100, 2); o[NAME[k] + "_狀態"] = state_of(r[k], k)
            o[NAME[k] + "_冷門檻%"] = round(THR[k][0] * 100, 2); o[NAME[k] + "_熱門檻%"] = round(THR[k][1] * 100, 2)      # 每列都帶門檻（頁面不靠 meta 也畫得出來）
        o.update({"高於熱門檻項數": hi, "低於冷門檻項數": lo, "合成狀態": st, "股池檔數": r["n_univ"], "來源": r["src"]})
        out.append(o)
    run = 1
    while run < len(out) and out[-1 - run]["合成狀態"] == out[-1]["合成狀態"]:
        run += 1
    cur = out[-1]
    for o in out:
        o["連續月數"] = None
    cur["連續月數"] = run
    pc = lambda x: f"{x * 100:.2f}%"
    meta = [{"label": "產生程式", "value": "make_thermo.py"}, {"label": "資料最後一天", "value": last_day},
            {"label": "最新月份", "value": cur["月份"]}, {"label": "最新月份是否完整月", "value": cur["完整月"]},
            {"label": "最新月份說明", "value": (f"{cur['月份']} 已是完整月（月底 {cur['資料日']}）" if cur["完整月"] == "是" else f"{cur['月份']} 還沒結束，只算到 {cur['資料日']}；月底的讀數可能不同")},
            {"label": "用途限制", "value": LIMIT},
            {"label": "股池", "value": "上市、四碼普通股（不含 91 開頭）、非創新板、當天有成交、收盤 ≥ 10 元、近 60 個交易日都在日 K 內且 60 日平均成交值 ≥ 1,000 萬；每個日曆月最後一個交易日收盤後算"}]
    for k in KEYS:
        meta.append({"label": NAME[k] + "_定義", "value": DEFN[k]})
        meta.append({"label": NAME[k] + "_冷門檻%", "value": round(THR[k][0] * 100, 2)})
        meta.append({"label": NAME[k] + "_熱門檻%", "value": round(THR[k][1] * 100, 2)})
    meta += [{"label": "門檻", "value": "；".join(f"{NAME[k]}：冷 < {pc(THR[k][0])}、熱 > {pc(THR[k][1])}" for k in KEYS) + "（只用 2008～2015 訂，之後不再調）"},
             {"label": "合成規則", "value": "三項中 ≥ 2 項高於熱門檻＝熱；≥ 2 項低於冷門檻＝冷；其餘＝溫（合成規則是事後訂的）"},
             {"label": "歷史對照", "value": "2016-01～2026-10（門檻只用 2008～2015 訂）當月策略對 0050 的超額年化：冷 −11.7／溫 +2.2／熱 +36.1 個百分點（25／52／53 個月）。這是同一個月的同步關係，不是預測"},
             {"label": "含息調整", "value": f"含息日報酬同回測引擎（cslib.Panel 開除權息修正）：除權息日用證交所 TWT49U 參考價，封存到 {info['twt49u_last']}"
                                       + ("；減資／面額變更恢復買賣日用事件參考價（候選修正，已開）" if reduce_fix else "")},
             {"label": "資料來源", "value": f"{first_calc} 起由本程式用官方日 K（git HEAD）計算；{rows[0]['month']}～{[r['month'] for r in rows if r['src'] != '官方日K計算'][-1] if rows[0]['src'] != '官方日K計算' else '—'} 沿用研究存檔 thermometer_monthly.csv"
                                       "（官方日 K 2015-01-05 才開始，之前的月份本程式無法重算）"},
             {"label": "日K來源天數", "value": "；".join(f"{os.path.basename(k.rstrip('/'))}：{v} 天" for k, v in src["days_by_repo"].items())},
             {"label": "出處", "value": "xq_scripts/README.md 第 32.2 節；research/tw_stock_selection/agents/s83_environment_thermometer/（S83）、agents/r82_r83_replicate/（R83 已獨立重現三條月序列、門檻與分組結果）"},
             {"label": "make_thermo.py", "value": open(os.path.abspath(__file__), encoding="utf-8").read(), "type": "code", "lang": "python"}]
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"rows": out, "meta": meta}, fh, ensure_ascii=False)
    print(f"thermo.json：{len(out)} 個月（{out[0]['月份']}～{cur['月份']}），資料最後一天 {last_day}，最新月份完整月＝{cur['完整月']}，"
          f"最新讀數 {cur['飆股密度%']}%／{cur['大漲家數比例%']}%／{cur['右尾厚度%']}% → {cur['合成狀態']}（連續 {run} 個月）")
    if check and not res["pass"]:
        sys.exit(3)


if __name__ == "__main__":
    main()
