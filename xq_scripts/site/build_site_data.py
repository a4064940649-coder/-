# 產生「月營收驚奇操盤台」網站的資料檔（JSON：{rows, meta}）。
# 用法：python3 -I -B xq_scripts/site/build_site_data.py <lists/YYYY-MM.csv> <lists/YYYY-MM_backtest_holdings.csv> <list.json> <profit.json> <今天YYYY-MM-DD> <下次訊號日> <下次執行日> <輸出資料夾>
import csv, json, sys, os
lst_csv, hold_csv, lst_json, profit_json, today, next_sig, next_exec, outdir = sys.argv[1:9]
os.makedirs(outdir, exist_ok=True)
L = json.load(open(lst_json, encoding='utf-8'))
P = json.load(open(profit_json, encoding='utf-8'))
src = os.path.basename(__file__)
code_txt = open(__file__, encoding='utf-8').read()
def dump(name, rows, extra):
    meta = [{"label": "產生程式", "value": src}, {"label": "資料最後一天", "value": L["data_last_day"]}] + extra
    json.dump({"rows": rows, "meta": meta}, open(os.path.join(outdir, name + ".json"), "w", encoding="utf-8"), ensure_ascii=False)
def rd(p):
    with open(p, encoding='utf-8-sig') as f: return list(csv.DictReader(f))
H = rd(hold_csv)
num = lambda s: (float(s) if s not in ("", None) else None)
# 狀態
in_window = today <= L["exec_date"]
if today < L["sig_date"]:
    act = "今天不用買賣"; detail = f"本月換股日是 {L['sig_date']}，收盤後產生名單。"
elif today == L["sig_date"]:
    act = "今天收盤後換股"; detail = f"今天收盤後看名單，明天（{L['exec_date']}）開盤先賣後買。"
elif today == L["exec_date"]:
    act = "今天開盤執行換股"; detail = "照下面的賣出與買進清單，開盤先賣後買。"
else:
    act = "今天不用買賣"; detail = f"本月已在 {L['exec_date']} 換股完成。下次換股日 {next_sig} 收盤後產生名單，{next_exec} 開盤執行。"
dump("status", [{"今天": today, "動作": act, "說明": detail, "本月訊號日": L["sig_date"], "本月執行日": L["exec_date"],
                 "下次訊號日": next_sig, "下次執行日": next_exec, "營收月份": L["rev_month"], "資料最後一天": L["data_last_day"]}],
     [{"label": "規則", "value": "每月第一個日期 >= 11 的交易日收盤後產生名單，隔天開盤先賣後買"}])
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
dump("actions", acts, [{"label": "來源", "value": os.path.basename(hold_csv) + "（策略回測持股延續到本月的換股結果）"}])
# 目前持股
hold = [ {k: x[k] for k in ("代號", "名稱", "名次", "SURGE", "訊號日收盤")} | {"狀態": "新買" if x["動作"] == "買進" else "續抱"} for x in acts if x["動作"] in ("續抱", "買進")]
dump("holdings", hold, [{"label": "說明", "value": "另有 0050 底倉 30%，衛星沒用完的資金也放 0050"}])
# 前 20 名
top = [{"名次": t["rank_ma"], "代號": t["code"], "名稱": t["name"], "SURGE": round(t["surge"], 3), "前20日漲幅%": round(t["ret20"] * 100, 1),
        "距季線%": round(t["dist_ma60"] * 100, 1), "收盤": t["close"], "入選": "是" if (t["rank_ma"] <= 10 and t["ret20"] > 0.10) else ""} for t in L["top20"]]
dump("top20", top, [{"label": "定義", "value": "收盤在季線上的上市股票，依 SURGE 由大到小；前 10 名中前 20 日漲幅 > 10% 的入選"}])
# 回測淨值（週）與逐年
nav = [{"日期": d, "現行策略": c, "XQ版": x, "0050": e} for d, c, x, e in zip(P["dates"], P["cur"], P["xq"], P["etf"])]
dump("nav", nav, [{"label": "說明", "value": "100 萬起始、嚴成本回測；現行＝研究中心名次版 W10_70；XQ版＝20＋21 號固定門檻；0050 含息"},
                  {"label": "來源", "value": "research/tw_stock_selection/agents/s19_xq_threshold/navs.csv.gz"}])
yrs = sorted(P["years"]["cur"].keys())
dump("yearly", [{"年": int(y), "現行策略": P["years"]["cur"][y], "XQ版": P["years"]["xq"][y], "0050": P["years"]["etf"][y]} for y in yrs],
     [{"label": "說明", "value": "2015 為 7 月起；2026 到 10/22"}])
print("ok", outdir)
