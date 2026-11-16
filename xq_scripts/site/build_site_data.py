# 產生「月營收驚奇操盤台」網站的資料檔（JSON：{rows, meta}）。
# 用法：python3 -I -B xq_scripts/site/build_site_data.py <lists/YYYY-MM.csv> <lists/YYYY-MM_backtest_holdings.csv> <list.json> <profit.json> <今天YYYY-MM-DD> <下次訊號日> <下次執行日> <輸出資料夾> [<官方日K倉庫> <換股後持股csv> <上線起始日>]
# 後三個參數有給時，另外產生 live.json（策略上線後每日淨值，起始日開盤＝100 萬，對照 0050）並在持股、換股清單加上最新收盤；持股另加股數、起始價與 0050 一列。
#   換股後持股 csv 旁邊若有同月份的現金檔（holdings_after_YYYY-MM.csv → cash_after_YYYY-MM.txt，內容是一個數字，# 開頭為註解），
#   現金會計入 live.json 的策略淨值（起點與每日都加上現金）。
# <profit.json> 由 make_profit.py 產生（0050 用精確含息還原價，週標籤＝當週最後一個交易日）。
# 另產生 risk.json（README 的回測風險數字，標明出處）。
import csv, json, sys, os, re
lst_csv, hold_csv, lst_json, profit_json, today, next_sig, next_exec, outdir = sys.argv[1:9]
tsd = sys.argv[9] if len(sys.argv) > 9 else None
state_csv = sys.argv[10] if len(sys.argv) > 10 else None
live_start = sys.argv[11] if len(sys.argv) > 11 else None
cash_file = re.sub(r'holdings_after_([^/]+)\.csv$', r'cash_after_\1.txt', state_csv) if state_csv else None
def read_cash(p):
    if not p or p == state_csv or not os.path.exists(p): return 0.0, None
    for line in open(p, encoding='utf-8-sig'):
        line = line.split('#', 1)[0].strip().replace(',', '')
        if line: return float(line), p
    return 0.0, p
cash, cash_src = read_cash(cash_file)
import subprocess
def day_prices(date):
    try:
        txt = subprocess.run(["git", "-C", tsd, "show", f"HEAD:data/universe/daily/{date}.csv"], capture_output=True, text=True, check=True).stdout
    except Exception:
        return None
    out = {}
    for r in csv.DictReader(txt.splitlines()):
        try: out[r["stock_id"]] = (float(r["open"]) if r["open"] else None, float(r["close"]) if r["close"] else None)
        except ValueError: pass
    return out
def trade_days():
    txt = subprocess.run(["git", "-C", tsd, "ls-tree", "--name-only", "HEAD", "data/universe/daily/"], capture_output=True, text=True, check=True).stdout
    return sorted(x.rsplit("/", 1)[1][:-4] for x in txt.split())
os.makedirs(outdir, exist_ok=True)
L = json.load(open(lst_json, encoding='utf-8'))
P = json.load(open(profit_json, encoding='utf-8'))
src = os.path.basename(__file__)
code_txt = open(__file__, encoding='utf-8').read()
def dump(name, rows, extra):
    meta = [{"label": "產生程式", "value": src}, {"label": "資料最後一天", "value": L["data_last_day"]}] + extra + [{"label": src, "value": code_txt, "type": "code", "lang": "python"}]
    json.dump({"rows": rows, "meta": meta}, open(os.path.join(outdir, name + ".json"), "w", encoding="utf-8"), ensure_ascii=False)
def rd(p):
    with open(p, encoding='utf-8-sig') as f: return list(csv.DictReader(f))
H = rd(hold_csv)
num = lambda s: (float(s) if s not in ("", None) else None)
# 狀態（階段：idle 不用買賣、sig 今天收盤後換股、exec 今天開盤執行；*_pending＝換股日到了但清單還是上個月的）
if today < L["sig_date"]:
    stage = "idle"; act = "今天不用買賣"; detail = f"本月換股日是 {L['sig_date']}，收盤後產生名單。"
elif today == L["sig_date"]:
    stage = "sig"; act = "今天收盤後換股"; detail = f"今天收盤後看名單，明天（{L['exec_date']}）開盤先賣後買。"
elif today == L["exec_date"]:
    stage = "exec"; act = "今天開盤執行換股"; detail = "照下面的賣出與買進清單，開盤先賣後買。"
elif today == next_sig:
    stage = "sig_pending"; act = "今天收盤後換股"; detail = f"今天收盤後（建議 22 點後）產生新名單，明天（{next_exec}）開盤先賣後買。新名單產生後本頁更新。"
elif today == next_exec:
    stage = "exec_pending"; act = "今天開盤執行換股"; detail = "今天開盤要換股，但本頁的名單還沒更新到這個月，請用 XQ 20、21 號的結果下單。"
else:
    stage = "idle"; act = "今天不用買賣"; detail = f"本月已在 {L['exec_date']} 換股完成。下次換股日 {next_sig} 收盤後產生名單，{next_exec} 開盤執行。"
dump("status", [{"今天": today, "階段": stage, "動作": act, "說明": detail, "本月訊號日": L["sig_date"], "本月執行日": L["exec_date"],
                 "下次訊號日": next_sig, "下次執行日": next_exec, "營收月份": L["rev_month"], "資料最後一天": L["data_last_day"]}],
     [{"label": "規則", "value": "換股日＝每月 10 日（含，遇假日順延）之後第一個交易日的下一個交易日；換股日收盤後產生名單，隔天開盤先賣後買（2026-11-16 統一，同回測與 20／21／22 號腳本）"}])
# 本月動作（策略持股延續）
acts = []
for r in H:
    blk = r["區塊"]
    if blk.startswith("持股_賣出"): a = "賣出"
    elif blk.startswith("持股_續抱"): a = "續抱"
    elif blk.startswith("F1"): a = "買進"
    else: continue
    acts.append({"動作": a, "代號": r["代號"], "名稱": r["名稱"], "名次": num(r["名次"]), "SURGE": num(r["SURGE"]),
                 "前20日漲幅%": num(r["RET20pct"]), "訊號日收盤": num(r["收盤"]), "原因": r["說明"] or ("前 10 名且前 20 日漲幅 > 10%" if a == "買進" else "")})
last_px = {}; last_day = None
if tsd:
    days = trade_days(); last_day = days[-1]; last_px = day_prices(last_day) or {}
    for x in acts:
        c = last_px.get(x["代號"], (None, None))[1]
        x["最新收盤"] = c
        x["訊號日以來%"] = round((c / x["訊號日收盤"] - 1) * 100, 2) if (c and x["訊號日收盤"]) else None
dump("actions", acts, [{"label": "來源", "value": os.path.basename(hold_csv) + "（策略回測持股延續到本月的換股結果）"},
                       {"label": "訊號日以來%", "value": "最新收盤 ÷ 訊號日收盤 − 1（基準是訊號日收盤，不是成交日開盤；持股表的「本月以來%」才是對起始價）"}])
# 目前持股（有換股後持股 csv 時加「股數」與「起始價」＝上線起始日開盤價，並加一列 0050）
hold = [ {k: x.get(k) for k in ("代號", "名稱", "名次", "SURGE", "訊號日收盤", "最新收盤")} | {"狀態": "新買" if x["動作"] == "買進" else "續抱"} for x in acts if x["動作"] in ("續抱", "買進")]
hold_meta = [{"label": "說明", "value": "另有 0050（底倉 30%＋衛星沒用完的資金）"}]
if tsd and state_csv:
    sh = {r["代號"]: float(r["股數"]) for r in rd(state_csv)}
    p0 = day_prices(live_start) if live_start else None
    ps = day_prices(L["sig_date"]) or {}
    if "0050" in sh:
        c0 = last_px.get("0050", (None, None))[1]; s0 = ps.get("0050", (None, None))[1]
        hold.append({"代號": "0050", "名稱": "元大台灣50", "名次": None, "SURGE": None, "訊號日收盤": s0, "最新收盤": c0, "狀態": "0050"})
    for x in hold:
        x["股數"] = sh.get(x["代號"])
        x["起始價"] = (p0 or {}).get(x["代號"], (None, None))[0]
    hold_meta += [{"label": "股數", "value": os.path.basename(state_csv) + "（100 萬比例的策略持股）"},
                  {"label": "起始價", "value": f"{live_start} 開盤價（上線起始）"}]
    if cash_src:
        hold_meta.append({"label": "現金", "value": f"另有現金 {cash:,.0f} 元（{os.path.basename(cash_src)}），已計入 live.json 淨值"})
# 本月以來%：有起始價時＝最新收盤 ÷ 起始價 − 1（和起始價欄、live.json 同一個起點）；沒有起始價時才用訊號日收盤
for x in hold:
    b = x.get("起始價") or None; c = x.get("最新收盤")
    if b is None: b = x.get("訊號日收盤")
    x["本月以來%"] = round((c / b - 1) * 100, 2) if (c and b) else None
hold_meta.append({"label": "本月以來%", "value": "最新收盤 ÷ 起始價（上線起始日開盤）− 1；沒有起始價時改用訊號日收盤當基準"})
dump("holdings", hold, hold_meta)
# 前 20 名
top = [{"名次": t["rank_ma"], "代號": t["code"], "名稱": t["name"], "SURGE": round(t["surge"], 3), "前20日漲幅%": round(t["ret20"] * 100, 1),
        "距季線%": round(t["dist_ma60"] * 100, 1), "收盤": t["close"], "入選": "是" if (t["rank_ma"] <= 10 and t["ret20"] > 0.10) else ""} for t in L["top20"]]
dump("top20", top, [{"label": "定義", "value": "收盤在季線上的上市股票，依 SURGE 由大到小；前 10 名中前 20 日漲幅 > 10% 的入選"}])
# 風險提示：README 的回測數字（寫成常數，標明出處）
RISK = [
    ("回撤_單一路徑", -32.1, "%", "最大回撤（100 萬、標準換股日那一條，2024-07 到 2025-04；除權息修正後，修正前 −33.4）", "xq_scripts/README.md 第 12 節「2026-11-16 除權息修正後」"),
    ("回撤_20條中位數", -33.5, "%", "最大回撤，20 個起始日的中位數（除權息修正後；修正前 −34.0，第 15 節「修正後怎麼選」表）", "xq_scripts/README.md 第 12 節「2026-11-16 除權息修正後」"),
    ("回撤_壞的時候", -36.0, "%", "最大回撤，100 條路徑裡差的 10%（除權息修正前的數字，未重算）", "xq_scripts/README.md 第 16 節"),
    ("0050_回撤", -34.0, "%", "0050 同期最大回撤（2022）", "xq_scripts/README.md 第 12 節"),
    ("落後0050最多", 15.3, "個百分點", "2025 年 0050 大漲時，滾動 12 個月最多落後 0050（到 2025-10-07；除權息修正後，修正前 17.6）", "xq_scripts/README.md 第 12 節「2026-11-16 除權息修正後」"),
    ("落後時間比例", 19.5, "%", "滾動 12 個月落後 0050 的時間比例（除權息修正後，修正前 19.8）", "xq_scripts/README.md 第 12 節「2026-11-16 除權息修正後」"),
    ("最賺5筆占比", 36, "%", "最賺 5 筆占已平倉損益（除權息修正後，修正前 37）", "xq_scripts/README.md 第 12 節「2026-11-16 除權息修正後」"),
    ("年化_全部", 48.99, "%", "2019 年起每年（單一路徑；除權息修正後，修正前 48.13）", "xq_scripts/README.md 第 12 節「2026-11-16 除權息修正後」"),
    ("年化_不買最賺5檔", 45.07, "%", "完全不買最賺 5 檔，2019 年起每年（除權息修正後，修正前 43.42）", "xq_scripts/README.md 第 12 節「2026-11-16 除權息修正後」"),
    ("勝率", 51.7, "%", "賺錢的交易比例（381 筆；除權息修正後，修正前 379 筆 51.2）", "xq_scripts/README.md 第 12 節「2026-11-16 除權息修正後」"),
    ("賺的平均", 26.75, "%", "賺錢的交易平均賺（除權息修正後，修正前 27.16）", "xq_scripts/README.md 第 12 節「2026-11-16 除權息修正後」"),
    ("賠的平均", -9.26, "%", "賠錢的交易平均賠（除權息修正後，修正前 −9.24）", "xq_scripts/README.md 第 12 節「2026-11-16 除權息修正後」"),
    ("晚1天少", 3.6, "個百分點", "晚 1 天下單，2019 年起每年少（第 14 節「每檔＝衛星 ÷ 10，底倉不能動」那列）；除權息修正前的數字，未重算", "xq_scripts/README.md 第 14 節「晚下單很貴」表"),
    ("晚2天少", 5.3, "個百分點", "晚 2 天下單，2019 年起每年少（第 14 節「每檔＝衛星 ÷ 10，底倉不能動」那列）；除權息修正前的數字，未重算", "xq_scripts/README.md 第 14 節「晚下單很貴」表"),
    ("晚2天回撤", -36.4, "%", "晚 2 天下單的最大回撤（第 14 節「每檔＝衛星 ÷ 10，底倉不能動」那列）；除權息修正前的數字，未重算", "xq_scripts/README.md 第 14 節「晚下單很貴」表"),
    ("準時可預期", 46.5, "%", "每月都準時開盤下單的可預期年報酬（2019 年起）；除權息修正前的數字，未重算", "xq_scripts/README.md 第 16 節"),
    ("有時晚可預期", 43, "%", "有時晚 0～4 天下單的可預期年報酬（100 條路徑中位數）；除權息修正前的數字，未重算", "xq_scripts/README.md 第 16 節"),
]
dump("risk", [{"項目": a, "數值": b, "單位": c, "說明": d, "出處": e} for a, b, c, d, e in RISK],
     [{"label": "說明", "value": "研究中心回測（嚴成本、只上市、排除創新板、衛星 70%＋每檔＝衛星 ÷ 10）；2026-11-16 起除權息參考價修正後（標「修正前」的是舊口徑、未重算）；回測不保證未來"}])
# 回測淨值（週）與逐年
nav = [{"日期": d, "現行策略": c, "XQ版": x, "0050": e} for d, c, x, e in zip(P["dates"], P["cur"], P["xq"], P["etf"])]
dump("nav", nav, [{"label": "說明", "value": "100 萬起始、嚴成本回測、除權息參考價修正後（2026-11-16）；現行＝研究中心名次版 W10_70；XQ版＝20＋21 號固定門檻、換股日同回測（A10）；0050 含息"},
                  {"label": "來源", "value": "現行、XQ版：xq_scripts/site/navs_exright.csv.gz（research/tw_stock_selection/agents/exright_engine_fix/run_engine.py 1；修正前是 s19_xq_threshold/navs.csv.gz）；0050：精確含息還原價（研究引擎同一來源，不用 navs 的整數欄）；由 make_profit.py 產生 profit.json"},
                  {"label": "日期", "value": "每週取最後一個交易日，日期就是那一天（不超過回測最後一天）"}])
yrs = sorted(P["years"]["cur"].keys())
dump("yearly", [{"年": int(y), "現行策略": P["years"]["cur"][y], "XQ版": P["years"]["xq"][y], "0050": P["years"]["etf"][y]} for y in yrs],
     [{"label": "說明", "value": "2015 為 7 月起；2026 到 10/22"},
      {"label": "0050", "value": "用精確含息還原價算（make_profit.py），不用 navs.csv.gz 的整數欄"}])
# 上線後每日淨值
if tsd and state_csv and live_start:
    sh = {r["代號"]: float(r["股數"]) for r in rd(state_csv)}
    days = [d for d in trade_days() if d >= live_start]
    rows = []; base = None; e_base = None
    for d in days:
        px = day_prices(d)
        if not px: continue
        if base is None:
            base = sum(sh[k] * px[k][0] for k in sh if k in px and px[k][0]) + cash; e_base = px["0050"][0]
        nav = sum(sh[k] * px[k][1] for k in sh if k in px and px[k][1]) + cash
        rows.append({"日期": d, "策略": round(nav / base * 1e6), "0050": round(px["0050"][1] / e_base * 1e6)})
    dump("live", rows, [{"label": "起始", "value": f"{live_start} 開盤＝100 萬（策略換股後持股，含 0050 底倉" + (f"與現金 {cash:,.0f} 元）" if cash_src else "）")},
                        {"label": "持股檔", "value": os.path.basename(state_csv)},
                        {"label": "現金", "value": (f"{cash:,.0f} 元（{os.path.basename(cash_src)}，起點與每日都計入）" if cash_src else "無現金檔，以 0 計")},
                        {"label": "價格", "value": "官方日 K 收盤（未含除息調整）"}])
# 0050 最新收盤寫進 status（給頁面下單計算用）
if last_px:
    stp = os.path.join(outdir, "status.json"); S = json.load(open(stp, encoding="utf-8"))
    S["rows"][0]["0050最新收盤"] = last_px.get("0050", (None, None))[1]; S["rows"][0]["價格日期"] = last_day
    json.dump(S, open(stp, "w", encoding="utf-8"), ensure_ascii=False)
print("ok", outdir, last_day)
