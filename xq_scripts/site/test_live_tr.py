# -*- coding: utf-8 -*-
# live_tr.py 的歷史假想測試：含息後帳戶報酬＝含息還原報酬。
# 用法：python3 -I -B xq_scripts/site/test_live_tr.py <官方日K倉庫> [<結果 json>]
# 測試（都只持有 0050、起始日開盤買進）：
#   A 除息：起始日＝0050 除息日的前一個交易日（2026-07-20、2026-01-21、2025-07-18），持有到除息日後第 5 個交易日
#   B 分割：起始日 2025-06-09（0050 在 2025-06-11～06-17 停止買賣、2025-06-18 一股分四股），持有到 2025-06-25
#   C 長一點：2026-07-20 開盤～2026-10-22 收盤（對 data/adj）
#   C' 2026-07-20 收盤～2026-08-11 收盤，和 navs_official.csv.gz 的 0050 欄（研究引擎的含息還原淨值）比。
#      只比到 2026-08-11：navs 的 0050 欄從 2026-08-12 起和官方日 K 收盤的日報酬每天差 ±0.05～0.19 個百分點（價格來源不同，和除息無關）。
#   A'、C'' 同 A、C，但現金股利在除息日收盤買回（simulate(reinvest=True)，0050 對照帳戶的另一種做法），原始差應 < 0.05 個百分點
#   D 把倉庫事件檔當成沒有、只用研究中心 TWT49U 封存重算 C，結果應相同
#   E（說明用，不設門檻）2015-07-01～2026-10-22 只買 0050：現金留著 對 還原（股利再投入），量化「現金不再投入」的長期差
# 注意：官方日 K 倉庫若是 blob:none 的部分複製，讀本機沒有的舊日檔會向遠端抓（要連網）。不能連網時先設
#   GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=protocol.allow GIT_CONFIG_VALUE_0=never，缺檔會直接報錯而不是去抓。
# 對照的「含息還原報酬」獨立來源＝官方日 K 倉庫 data/adj/0050.csv 的還原因子（不是 live_tr 讀的事件檔）：
#   還原報酬＝期末收盤 ÷ 期初價 ÷ 期間各事件因子的連乘 − 1（等同股利在除息日以參考價再投入）。
# 帳戶把現金股利留在現金（不再投入），所以和還原報酬的差有理論值：Σ 現金股利 ÷ 期初總值 ×（1 − 期末收盤 ÷ 除息參考價）（單一除息事件時精確）。
# 驗收兩層：① 扣掉這個「現金不再投入」的理論差之後，殘差 < 0.001 個百分點（程式沒有算錯）；② 原始差 < 0.05 個百分點（短窗、除息後漲跌不大時成立）。
import csv, gzip, json, os, subprocess, sys, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
tsd = sys.argv[1]
sp = importlib.util.spec_from_file_location("live_tr", os.path.join(HERE, "live_tr.py")); LT = importlib.util.module_from_spec(sp); sp.loader.exec_module(LT)
git = lambda *a: subprocess.run(["git", "-C", tsd] + list(a), capture_output=True, text=True, check=True).stdout
DAYS = sorted(x.rsplit("/", 1)[1][:-4] for x in git("ls-tree", "--name-only", "HEAD", "data/universe/daily/").split())
_c = {}
def px_of(d):
    if d not in _c:
        o = {}
        for r in csv.DictReader(git("show", f"HEAD:data/universe/daily/{d}.csv").splitlines()):
            if r["stock_id"] == "0050": o["0050"] = (float(r["open"]), float(r["close"]))
        _c[d] = o
    return _c[d]
ADJ = list(csv.DictReader(git("show", "HEAD:data/adj/0050.csv").splitlines()))
ARCH = os.path.join(HERE, "..", "..", "research", "tw_stock_selection", "raw", "twt49u", "twse-exright")
FD = LT.list_event_days(tsd)

def case(name, start, end, tol_pp, reinvest=False):
    days = [d for d in DAYS if start <= d <= end]
    ev, cov = LT.load_events(tsd, start, end, {"0050"}, ARCH, FD)
    A = LT.simulate({"0050": 1000.0}, 0.0, days, px_of, ev, start, reinvest=reinvest)
    N = LT.simulate({"0050": 1000.0}, 0.0, days, px_of, {}, start)          # 未調整（舊算法）
    acct = A["nav"] / A["base"] - 1; raw = N["nav"] / N["base"] - 1
    fac = 1.0; used = []
    for r in ADJ:
        if start < r["date"] <= end: fac *= float(r["factor"]); used.append(f"{r['date']} {r['kind'] or r['event']} {r['factor']}")
    adj = px_of(days[-1])["0050"][1] / px_of(days[0])["0050"][0] / fac - 1
    diff = (acct - adj) * 100
    theory = None
    cashev = [l for l in A["log"] if l["現金股利"]]
    if len(A["log"]) == 1 and len(cashev) == 1 and not reinvest:      # 單一除息事件：理論差＝股利 ÷ 期初總值 ×（1 − 期末收盤 ÷ 未四捨五入參考價）
        l = cashev[0]; theory = l["現金股利"] / A["base"] * (1 - px_of(days[-1])["0050"][1] / (l["除權息前收盤"] - l["權值＋息值"])) * 100
    out = {"測試": name, "起始日": start, "結束日": end, "事件（live_tr）": [f"{l['日期']} {l['種類']} 現金 {l['現金股利']:.2f} 股數×{l['股數倍數']:.6f}" for l in A["log"]],
           "事件（data/adj）": used, "未調整報酬%": round(raw * 100, 4), "含息帳戶報酬%": round(acct * 100, 4), "含息還原報酬%": round(adj * 100, 4),
           "差_個百分點": round(diff, 4), "現金不再投入的理論差_個百分點": (round(theory, 4) if theory is not None else None),
           "殘差_個百分點": (round(diff - theory, 6) if theory is not None else None), "容許_個百分點": tol_pp,
           "原始差小於容許": bool(abs(diff) < tol_pp),
           "通過": bool(len(A["log"]) == len(used) and len(used) > 0 and (abs(diff - theory) < 0.001 if theory is not None else abs(diff) < tol_pp))}
    return out, A

res = []
i = DAYS.index
for exd in ("2026-07-21", "2026-01-22", "2025-07-21"):
    res.append(case(f"A 除息 {exd}", DAYS[i(exd) - 1], DAYS[i(exd) + 5], 0.05)[0])
for exd in ("2026-07-21", "2026-01-22", "2025-07-21"):      # A'：0050 對照帳戶的另一種做法（除息日收盤再投入），原始差應 < 0.05 個百分點
    res.append(case(f"A' 除息 {exd}（除息日收盤再投入）", DAYS[i(exd) - 1], DAYS[i(exd) + 5], 0.05, reinvest=True)[0])
res.append(case("C'' 2026-07-20～2026-10-22（除息日收盤再投入，對 data/adj）", "2026-07-20", "2026-10-22", 0.05, reinvest=True)[0])
res.append(case("B 分割 2025-06-18", "2025-06-09", "2025-06-25", 0.05)[0])
c, A = case("C 2026-07-20～2026-10-22（對 data/adj）", "2026-07-20", "2026-10-22", 0.05)
res.append(c)
# C'：對 navs_official 的 0050 欄（收盤對收盤：2026-07-20 收盤 → 2026-10-22 收盤）
nav = {}
with gzip.open(os.path.join(HERE, "navs_official.csv.gz"), "rt") as f:
    for r in csv.DictReader(f): nav[r["date"]] = float(r["0050"])
rows = dict(A["rows"])
a = rows["2026-08-11"] / rows["2026-07-20"] - 1; b = nav["2026-08-11"] / nav["2026-07-20"] - 1
res.append({"測試": "C' 2026-07-20 收盤～2026-08-11 收盤（對 navs_official.csv.gz 的 0050 欄）", "含息帳戶報酬%": round(a * 100, 4), "含息還原報酬%": round(b * 100, 4),
            "差_個百分點": round((a - b) * 100, 4), "容許_個百分點": 0.05, "通過": bool(abs(a - b) * 100 < 0.05)})
# D：封存交叉核對——同一段事件改只用研究中心 TWT49U 封存（把倉庫事件檔當成沒有），結果應相同
ev2, _ = LT.load_events(tsd, "2026-07-20", "2026-10-22", {"0050"}, ARCH, {})
D = LT.simulate({"0050": 1000.0}, 0.0, [d for d in DAYS if "2026-07-20" <= d <= "2026-10-22"], px_of, ev2, "2026-07-20")
res.append({"測試": "D 只用 TWT49U 封存重算 C", "含息帳戶報酬%": round((D["nav"] / D["base"] - 1) * 100, 4), "差_個百分點": round((D["nav"] / D["base"] - A["nav"] / A["base"]) * 100, 6),
            "容許_個百分點": 1e-6, "通過": bool(abs(D["nav"] / D["base"] - A["nav"] / A["base"]) < 1e-8 and len(D["log"]) == 1)})
# E：長期「現金不再投入」的差（說明用）。只讀起訖兩天的日 K：事件一次套在期末（現金股利不再投入，所以結果和逐日算相同）
dl = ["2015-07-01", "2026-10-22"]
evL, _ = LT.load_events(tsd, dl[0], dl[-1], {"0050"}, ARCH, FD)
EC = LT.simulate({"0050": 1000.0}, 0.0, dl, px_of, evL, dl[0])
fac = 1.0
for r in ADJ:
    if dl[0] < r["date"] <= dl[-1]: fac *= float(r["factor"])
adjL = px_of(dl[-1])["0050"][1] / px_of(dl[0])["0050"][0] / fac
yrs = 11 + 113 / 365.0; ann = lambda x: (x ** (1 / yrs) - 1) * 100
info = {"測試": "E（說明用）2015-07-01 開盤～2026-10-22 收盤只買 0050", "事件數": len(EC["log"]), "事件數（data/adj）": sum(1 for r in ADJ if dl[0] < r["date"] <= dl[-1]),
        "期末倍數_現金留著（live_tr 口徑）": round(EC["nav"] / EC["base"], 4), "期末倍數_還原（data/adj，股利再投入）": round(adjL, 4),
        "期末倍數_navs_official": round(nav["2026-10-22"] / nav["2015-07-01"] * px_of("2015-07-01")["0050"][1] / px_of("2015-07-01")["0050"][0], 4),
        "年化%_現金留著": round(ann(EC["nav"] / EC["base"]), 2), "年化%_還原": round(ann(adjL), 2),
        "現金留著比還原每年少_個百分點": round(ann(EC["nav"] / EC["base"]) - ann(adjL), 2), "期末現金占比%": round(EC["cash"] / EC["nav"] * 100, 2)}
for r in res: print(json.dumps(r, ensure_ascii=False))
print(json.dumps(info, ensure_ascii=False))
ok = all(r["通過"] for r in res)
print("全部通過" if ok else "有測試沒過")
if len(sys.argv) > 2: json.dump({"ok": ok, "tests": res, "info": info}, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
sys.exit(0 if ok else 1)
