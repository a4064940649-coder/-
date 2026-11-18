# 產生「月營收驚奇操盤台」網站的資料檔（JSON：{rows, meta}）。
# 用法：python3 -I -B xq_scripts/site/build_site_data.py <lists/YYYY-MM.csv> <lists/YYYY-MM_backtest_holdings.csv> <list.json> <profit.json> <今天YYYY-MM-DD> <下次訊號日> <下次執行日> <輸出資料夾> [<官方日K倉庫> <換股後持股csv> <上線起始日>]
# 後三個參數有給時，另外產生 live.json（策略上線後每日淨值，起始日開盤＝100 萬，對照 0050）並在持股、換股清單加上最新收盤；持股另加股數、起始價與 0050 一列。
#   換股後持股 csv 旁邊若有同月份的現金檔（holdings_after_YYYY-MM.csv → cash_after_YYYY-MM.txt，內容是一個數字，# 開頭為註解），
#   現金會計入 live.json 的策略淨值（起點與每日都加上現金）。
# <profit.json> 由 make_profit.py 產生（0050 用精確含息還原價，週標籤＝當週最後一個交易日）。
# 另產生 risk.json（README 的回測風險數字，標明出處）。
# 2026-11-18 起（後三個參數有給時）持股頁列出完整交易資訊：
#   holdings 每列加 帳戶股數、成本金額、最新市值、損益金額、損益%、占總資產%、季線價位、距季線%、試算名次、目前狀態、出場條件、本月交易、備註，
#   並加「現金」「合計」兩列；status 加帳戶起始日、起始資金、目前總資產、累計損益、同期 0050、下次檢查條件；actions 加股數、成交價、金額。
#   金額一律是「100 萬帳戶口徑」：持股檔股數在起始日開盤連現金不是剛好 100 萬，live.json 一直把它正規化成 100 萬；
#   這裡用同一個換算倍數（100 萬 ÷ 起始日開盤總值），所以合計列總資產＝live.json 最後一筆策略淨值（不一致就報錯、不寫檔）。
#   季線（60 日均線）與試算名次直接呼叫名單產生器 make_list.py 的 compute()／held_status()（還原收盤口徑同策略 E0），
#   先在訊號日重算一次、和名單的前 20 名與距季線逐檔比對（不同就報錯），再把日期換成最新一天試算。
#   make_list.py 預設讀 ../../research/tw_stock_selection/agents/s18_live_list/live_surge/make_list.py，可用第 12 個參數另外指定（需要 numpy／pandas）。
# 2026-11-18 起 live.json 是「含息總報酬口徑」（同資料夾的 live_tr.py；策略帳戶與 0050 對照帳戶同一段程式）：
#   起始日之後遇到持股除息日，現金股利（息值 × 股數）加進帳戶現金；除權、權息、ETF 分割、面額變更、減資調整股數。
#   事件資料＝官方日 K 倉庫的 data/universe/exright 等日期軸事件檔（git HEAD），倉庫沒有的日子才用研究中心 TWT49U 封存
#   （預設 ../../research/tw_stock_selection/raw/twt49u/twse-exright，可用第 13 個參數指定）。
#   除權息資料最後一天早於最新股價日時不猜：live、holdings、status 的 meta「除權息警告」會寫「只涵蓋到 X 日，之後未調整」（warning＝true）。
#   沒有任何事件時，數字和舊算法（股數 × 收盤＋現金）逐元相同。持股表：現金股利記在該股的損益金額，錢放在現金列。
# 2026-11-19 起可以接多個月份（跨月銜接，同資料夾的 ledger.py；規則見 state/README.md「跨月銜接」）：
#   第 10 個參數可以給「段落表」（state/segments.csv：月份,生效日,持股檔,現金檔,底倉股數,備註），第 11 個參數給第一段的生效日或「-」。
#   每一段＝一次換股後的持股檔＋現金檔，生效日＝執行日（開盤換股）。live.json 逐段接成連續淨值：換股日先把當天除權息套到舊持股，
#   再在開盤把舊持股換成新持股（成交價＝官方日 K 開盤；交易成本＝換股前後開盤總值的差，已反映在現金檔，研究中心嚴成本）。
#   帳戶口徑不變：金額＝持股檔口徑 × 換算倍數（100 萬 ÷ 第一段起始日開盤總值），各月份同一個倍數。
#   只有一段時（舊用法：一份持股檔＋上線起始日，或段落表只有一列）8 個資料檔的 rows 和以前逐位相同。
#   兩段以上時：actions 的賣出列加 買進成本、成本金額、已實現損益、已實現損益%、持有起日；holdings 加一列「已實現」；status 加換股累計欄位。
#   另外一律產生 trades.json（帳戶起始＋歷次換股逐筆＋每次換股的交易成本；頁面還沒接）。
#   名單還沒結算成持股檔時（訊號日晚上到執行日結算前：段落表最後一段的生效日 ≤ 名單訊號日）：actions 是「預計」（沒有成交價、金額），
#   holdings 仍是換股前的持股，狀態寫「續抱／將賣出」。
import csv, json, sys, os, re
lst_csv, hold_csv, lst_json, profit_json, today, next_sig, next_exec, outdir = sys.argv[1:9]
tsd = sys.argv[9] if len(sys.argv) > 9 else None
state_csv = sys.argv[10] if len(sys.argv) > 10 else None
live_start = sys.argv[11] if len(sys.argv) > 11 else None
ml_path = sys.argv[12] if len(sys.argv) > 12 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "research", "tw_stock_selection", "agents", "s18_live_list", "live_surge", "make_list.py")
twt49u_dir = sys.argv[13] if len(sys.argv) > 13 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "research", "tw_stock_selection", "raw", "twt49u", "twse-exright")
START_CAPITAL = 1000000
# 0050 對照帳戶領到的現金股利：False＝留在帳戶現金、不再投入（2026-11-18 任務指定的做法）；True＝除息日收盤買回 0050（接近含息還原報酬）。
# 長期影響很大（test_live_tr.py 的 E 項：2015-07～2026-10 現金留著比含息還原每年少約 2.8 個百分點，等於讓策略相對 0050 佔便宜），
# 要不要改成 True 由產品擁有者決定；0050 下次除息（約 2027-01）之前兩種做法數字相同。改了要在 state/README.md 註記變更日。
BENCH_REINVEST = True
def load_live_tr():
    import importlib.util
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_tr.py")
    if not os.path.exists(p): raise SystemExit(f"找不到 {p}（live.json 含息口徑要用它）")
    sp = importlib.util.spec_from_file_location("live_tr", p); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return m, open(p, encoding="utf-8").read()
def load_ledger():
    import importlib.util
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ledger.py")
    if not os.path.exists(p): raise SystemExit(f"找不到 {p}（跨月銜接的帳本要用它）")
    sp = importlib.util.spec_from_file_location("ledger", p); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return m, open(p, encoding="utf-8").read()
LG, lg_code = load_ledger()
SEGS = LG.segments_from_arg(state_csv, live_start) if (state_csv and (live_start or LG.is_manifest(state_csv))) else None      # 各段換股後持股（依生效日）；只有一段＝舊用法
if SEGS:
    live_start = SEGS[0]["生效日"]
MULTI = bool(SEGS) and len(SEGS) > 1
cash, cash_src = (SEGS[0]["cash"], SEGS[0]["cash_src"]) if SEGS else (0.0, None)      # 第一段（帳戶起始）的現金
seg_names = "、".join(os.path.basename(s["持股檔"]) for s in SEGS) if SEGS else None
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
md = lambda d: f"{int(d[5:7])}/{int(d[8:10])}"          # 2026-12-11 → 12/11
def alloc(vals, total):
    """把一組金額四捨五入成整數元，而且加總剛好等於 total（最大餘數法）；逐列金額與合計才會逐元一致。"""
    fl = [int(v // 1) for v in vals]; left = int(total) - sum(fl)
    if not 0 <= left <= len(vals): raise SystemExit(f"金額分配檢查失敗：各列合計 {sum(vals):.2f} 和總數 {total} 對不上")
    for i in sorted(range(len(vals)), key=lambda i: -(vals[i] - fl[i]))[:left]: fl[i] += 1
    return fl
def live_rows_calc(segs):
    """上線後每日淨值（起始日開盤＝100 萬；含息總報酬口徑，見 live_tr.py；多段持股檔在各段生效日開盤換股，見 ledger.py）。
    回傳（每日列, 策略帳戶模擬結果, 0050 對照帳戶模擬結果, 除權息資訊）。"""
    LT, lt_code = load_live_tr()
    days = [d for d in trade_days() if d >= live_start]
    cache = {}
    def px_of(d):
        if d not in cache: cache[d] = day_prices(d)
        return cache[d]
    sh = set().union(*[set(s["sh"]) for s in segs])                   # 各段持有過的代號
    codes = set(sh) | {"0050"}
    ev, cov = LT.load_events(tsd, live_start, days[-1], codes, twt49u_dir)
    A = LG.simulate_account(LT, segs, days, px_of, ev)                 # 策略帳戶（只有一段時＝LT.simulate(持股, 現金, …)）
    E = LT.simulate({"0050": 1.0}, 0.0, days, px_of, ev, live_start, reinvest=BENCH_REINVEST)   # 0050 對照帳戶（每 1 股）
    if [d for d, _ in A["rows"]] != [d for d, _ in E["rows"]]: raise SystemExit("策略與 0050 對照帳戶的日期不同")
    rows = [{"日期": d, "策略": round(v / A["base"] * 1e6), "0050": round(e / E["base"] * 1e6)} for (d, v), (_, e) in zip(A["rows"], E["rows"])]
    last = A["rows"][-1][0] if A["rows"] else live_start
    cov_last, warns = LT.coverage_warnings(cov, last, LT.markets(tsd, last, set(sh)))
    sus = [c for c in LT.suspicious_cells(tsd, days, codes, ev, live_start)
           if c.split()[0] == "0050" or c.split()[0] in LG.held_before(segs, c.split()[1])]     # 只看那天開盤前真的持有的（0050 對照帳戶一直持有）
    if sus: warns.append("漲跌價差＝0 但收盤有變、又不在除權息資料的日子（可能漏掉除權息，請人工確認；未調整）：" + "、".join(sus))
    for lg in A["log"] + E["log"]:
        w = (f"{lg['代號']} {lg['日期']} {lg['種類']}：" + (lg["error"] if lg.get("error") else "近似處理，請人工確認（" + "；".join(lg["注意"]) + "）")) if (lg.get("error") or lg["近似"]) else None
        if w and w not in warns: warns.append(w)
    X = {"cov": cov, "cov_last": cov_last, "warns": warns, "code": lt_code, "f0050": 1e6 / E["base"]}
    return rows, A, E, X
def ma_trial(codes):
    """季線與「若今天就是訊號日」的試算：呼叫名單產生器 make_list.py（同策略 E0 的還原收盤口徑）。
    回傳 {代號: {季線價位, 距季線%, 試算名次, 目前狀態, 原因}}, 說明 dict。"""
    import importlib.util
    if not os.path.exists(ml_path): raise SystemExit(f"找不到名單產生器 {ml_path}（季線與試算名次要用它）；可用第 12 個參數指定")
    sp = importlib.util.spec_from_file_location("make_list", ml_path); ml = importlib.util.module_from_spec(sp); sp.loader.exec_module(ml)
    M = L.get("meta") or {}
    cut = (((M.get("rev") or {}).get("supplement") or {}).get(L["rev_month"]) or {}).get("cutoff")
    exr_on = bool((M.get("exright") or {}).get("on")); dl_on = bool((M.get("deadline") or {}).get("on"))
    D = ml.Data([ml.RAW_DEFAULT, tsd], rev_cutoff=cut)
    if D.dates[-1] != last_day: raise SystemExit(f"名單產生器讀到的日 K 最後一天 {D.dates[-1]} 和網站資料 {last_day} 不同")
    # 1) 訊號日重算，和名單逐檔比對（確認口徑相同）
    r1 = ml.compute(D, L["signal_month"], exright=exr_on, extra_codes=codes, deadline=dl_on)
    if r1.get("status") != "有訊號" or r1.get("sig_date") != L["sig_date"]: raise SystemExit("季線口徑檢查失敗：訊號日重算沒有訊號或日期不同")
    if [t["code"] for t in r1["top20"]] != [t["code"] for t in L["top20"]]: raise SystemExit("季線口徑檢查失敗：訊號日重算的前 20 名和名單不同")
    dmax = max(abs(a["dist_ma60"] - b["dist_ma60"]) for a, b in zip(r1["top20"], L["top20"]))
    if dmax > 1e-9: raise SystemExit(f"季線口徑檢查失敗：訊號日重算的距季線和名單最大差 {dmax}")
    # 2) 把「訊號日」換成最新一天再算一次（營收仍是名單那一份；只有股價、季線、股池是最新的）
    ml.signal_day = lambda D_, month, deadline=False: (len(D_.dates) - 1, None, "")
    r2 = ml.compute(D, L["signal_month"], exright=exr_on, extra_codes=codes, deadline=dl_on)
    if r2.get("status") != "有訊號" or r2.get("sig_date") != last_day: raise SystemExit("試算失敗：最新一天沒有訊號或日期不同")
    out = {}
    for c in codes:
        st, r = ml.held_status(r2, c)
        if r is None or r.get("dist_ma60") is None or not r.get("close"):
            out[c] = {"季線價位": None, "距季線%": None, "試算名次": None, "目前狀態": "會賣：" + st["reason"], "原因": st["reason"]}; continue
        if st["action"] == "續抱": cur = "續抱"
        elif "跌破季線" in st["reason"]: cur = "會賣：跌破季線"
        elif "前 20 名" in st["reason"]: cur = "會賣：跌出前 20 名"
        else: cur = "會賣：" + st["reason"]
        out[c] = {"季線價位": round(r["close"] / (1 + r["dist_ma60"]), 2), "距季線%": round(r["dist_ma60"] * 100, 2), "試算名次": r["rank_ma"],
                  "目前狀態": cur, "原因": st["reason"], "_close": r["close"]}
    x = r2.get("exright") or {}
    info = {"訊號日比對": f"訊號日 {L['sig_date']} 重算：前 20 名代號順序相同、距季線最大差 {dmax:.1e}", "營收截止": cut, "除權息修正": exr_on,
            "未確認格子": [f"{u['code']} {u['date']}（{u['fixed']}）" for u in (x.get("unverified") or []) if u["code"] in codes],
            "封存之後的日子": len(x.get("days_after_twt49u_in_window") or []), "封存最後一天": x.get("twt49u_last"), "試算季線上檔數": r2.get("n_ma")}
    return out, info
os.makedirs(outdir, exist_ok=True)
L = json.load(open(lst_json, encoding='utf-8'))
P = json.load(open(profit_json, encoding='utf-8'))
src = os.path.basename(__file__)
code_txt = open(__file__, encoding='utf-8').read()
OUT = {}
def dump(name, rows, extra):
    meta = [{"label": "產生程式", "value": src}, {"label": "資料最後一天", "value": L["data_last_day"]}] + extra + [{"label": src, "value": code_txt, "type": "code", "lang": "python"}]
    OUT[name] = {"rows": rows, "meta": meta}      # 全部算完、檢查都過了才一起寫檔（見檔尾）
def rd(p):
    with open(p, encoding='utf-8-sig') as f: return list(csv.DictReader(f))
H = rd(hold_csv)
num = lambda s: (float(s) if s not in ("", None) else None)
def next_weekday(d):
    import datetime
    x = datetime.date.fromisoformat(d) + datetime.timedelta(days=1)
    while x.weekday() >= 5: x += datetime.timedelta(days=1)
    return x.isoformat()
EXEC_EST = not L.get("exec_date")                 # 訊號日晚上產生的名單還沒有執行日：先用下一個平日預估（只扣週末，不扣國定假日）
EXEC_D = L.get("exec_date") or next_weekday(L["sig_date"])
# 狀態（階段：idle 不用買賣、sig 今天收盤後換股、exec 今天開盤執行；*_pending＝換股日到了但清單還是上個月的）
if today < L["sig_date"]:
    stage = "idle"; act = "今天不用買賣"; detail = f"本月換股日是 {L['sig_date']}，收盤後產生名單。"
elif today == L["sig_date"]:
    stage = "sig"; act = "今天收盤後換股"; detail = f"今天收盤後看名單，明天（{L['exec_date']}）開盤先賣後買。" if not EXEC_EST else f"今天收盤後看名單，下一個交易日（預估 {EXEC_D}）開盤先賣後買。"
elif today == EXEC_D:
    stage = "exec"; act = "今天開盤執行換股"; detail = "照下面的賣出與買進清單，開盤先賣後買。"
elif today == next_sig:
    stage = "sig_pending"; act = "今天收盤後換股"; detail = f"今天收盤後（建議 22 點後）產生新名單，明天（{next_exec}）開盤先賣後買。新名單產生後本頁更新。"
elif today == next_exec:
    stage = "exec_pending"; act = "今天開盤執行換股"; detail = "今天開盤要換股，但本頁的名單還沒更新到這個月，請用 XQ 20、21 號的結果下單。"
else:
    stage = "idle"; act = "今天不用買賣"; detail = f"本月已在 {EXEC_D} 換股完成。下次換股日 {next_sig} 收盤後產生名單，{next_exec} 開盤執行。"
dump("status", [{"今天": today, "階段": stage, "動作": act, "說明": detail, "本月訊號日": L["sig_date"], "本月執行日": EXEC_D,
                 "下次訊號日": next_sig, "下次執行日": next_exec, "營收月份": L["rev_month"], "資料最後一天": L["data_last_day"]}],
     [{"label": "規則", "value": "換股日＝每月 10 日（含，遇假日順延）之後第一個交易日的下一個交易日；換股日收盤後產生名單，隔天開盤先賣後買（2026-11-16 統一，同回測與 20／21／22 號腳本）"}])
# 本月動作（策略持股延續）
acts = []
for r in H:
    blk = r["區塊"]
    if blk.startswith("持股_賣出"): a = "賣出"
    elif blk.startswith("持股_續抱"): a = "續抱"
    elif blk.startswith("F1"):
        if str(r.get("動作") or "").startswith("續抱"): continue          # 買進名單裡已經持有的那一檔，已列在「續抱」
        a = "未買進" if str(r.get("動作") or "").startswith("不買") else "買進"
    else: continue
    acts.append({"動作": a, "代號": r["代號"], "名稱": r["名稱"], "名次": num(r["名次"]), "SURGE": num(r["SURGE"]),
                 "前20日漲幅%": num(r["RET20pct"]), "訊號日收盤": num(r["收盤"]), "原因": (r["動作"] if a == "未買進" else r["說明"] or ("前 10 名且前 20 日漲幅 > 10%" if a == "買進" else ""))})
Hacts = [r for r in H if r["區塊"].startswith(("持股_賣出", "持股_續抱")) or (r["區塊"].startswith("F1") and not str(r.get("動作") or "").startswith("續抱"))]
last_px = {}; last_day = None
if tsd:
    days = trade_days(); last_day = days[-1]; last_px = day_prices(last_day) or {}
    if last_day and last_day > L["data_last_day"]: L["data_last_day"] = last_day  # 資料最後一天跟著最新股價日（名單日只用在訊號日欄位）
    for x in acts:
        c = last_px.get(x["代號"], (None, None))[1]
        x["最新收盤"] = c
        x["訊號日以來%"] = round((c / x["訊號日收盤"] - 1) * 100, 2) if (c and x["訊號日收盤"]) else None
# 策略帳戶（100 萬帳戶口徑）：持股檔股數 × 換算倍數；換算倍數＝100 萬 ÷ 起始日開盤總值（含現金），和 live.json 的正規化相同
ACC = None; act_meta = []
if tsd and state_csv and live_start:
    if SEGS[-1]["生效日"] > last_day: raise SystemExit(f"段落表最後一段的生效日 {SEGS[-1]['生效日']} 晚於官方日 K 最後一天 {last_day}")
    sh_state = dict(SEGS[-1]["sh"])                                    # 目前這一段（最近一次換股後）的持股檔股數
    px_start = day_prices(live_start) or {}
    for k in SEGS[0]["sh"]:
        if not px_start.get(k, (None, None))[0]: raise SystemExit(f"持股 {k} 缺起始日開盤價，無法算帳戶金額")
    for k in sh_state:
        if not last_px.get(k, (None, None))[1]: raise SystemExit(f"持股 {k} 缺最新收盤，無法算帳戶金額")
    live_rows, SIM, SIM_E, XR = live_rows_calc(SEGS)
    LED = LG.build_ledger(SEGS, SIM, px_start)                         # 帳本：目前部位成本、歷次買賣、每次換股的交易成本（持股檔口徑）
    _tot, _real, _unr, _res = LG.check_identity(SIM, LED, {k: last_px[k][1] for k in LED["pos"]})
    if abs(_res) > 1e-6 * max(1.0, abs(SIM["base"])): raise SystemExit(f"帳本檢查失敗：總損益 {_tot:.4f} ≠ 已實現 {_real:.4f}＋未實現 {_unr:.4f}")
    for s_ in LED["switches"]:
        if s_["交易成本"] < -1e-6 or s_["交易成本"] > 0.03 * s_["換股前開盤總值"]:
            raise SystemExit(f"{s_['日期']} 換股前後開盤總值差 {s_['交易成本']:.2f} 元不像交易成本（應 ≥ 0 且不到總值 3%）；請檢查 {s_['月份']} 的持股檔與現金檔")
    base_raw, nav_raw = SIM["base"], SIM["nav"]
    if not live_rows or live_rows[0]["日期"] != live_start or live_rows[-1]["日期"] != last_day: raise SystemExit("live 每日淨值的起訖日和起始日／最新股價日不同")
    ACC = {"sh": sh_state, "f": START_CAPITAL / base_raw, "base_raw": base_raw, "nav_raw": nav_raw, "live": live_rows,
           "sh_now": SIM["sh"], "cash_now": SIM["cash"], "div": SIM["div"], "log": SIM["log"], "log_e": SIM_E["log"], "xr": XR, "led": LED}
    # 含息口徑的 meta（live、holdings、status 共用）：除權息資料涵蓋到哪一天、期間事件、警告（warning＝true 表示有未調整或需人工確認的地方）
    def ev_txt(lg, per_share=False):
        if lg.get("error"): return f"{lg['代號']} {lg['日期']} {lg['種類']}：{lg['error']}"
        t = (f"{lg['代號']} {lg['日期']} {lg['種類']}：" + (f"現金股利每股 {lg['權值＋息值']:g} 元" + ("" if per_share else f" × {lg['事件前股數']:,.4g} 股＝{lg['現金股利']:,.2f} 元（持股檔口徑）") + "，加進現金"
             if lg["現金股利"] else f"股數 × {lg['股數倍數']:.6f}（前收 {lg['除權息前收盤']:g} → 參考價 {lg['參考價']:g}）"))
        return t + ("；" + "；".join(lg["注意"]) if lg["注意"] else "") + f"（{lg['來源']}）"
    cv = XR["cov"]
    XR["meta"] = [
        {"label": "除權息資料涵蓋", "value": f"上市除權息資料涵蓋到 {XR['cov_last'] or '（找不到）'}（官方日 K 倉庫 data/universe/exright 最後一個事件檔 {cv['repo_exright_last'] or '無'}；研究中心 TWT49U 封存到 {cv['archive_last'] or '無'}）。"
                                         f"最新股價日 {last_day}。沒有事件的日子倉庫不會有檔，所以「涵蓋到」＝最後一個有事件檔的日子", "covered_through": XR["cov_last"]},
        {"label": "期間除權息事件", "value": ("策略帳戶：" + ("；".join(ev_txt(l) for l in SIM["log"]) or "無") + "。0050 對照帳戶：" + ("；".join(ev_txt(l, True) for l in SIM_E["log"]) or "無")
                                         + f"。（{live_start} 之後到 {last_day} 的持股與 0050 事件；起始日當天的不算，因為開盤價已是除權息後）"),
         "n_events": len(SIM["log"]), "n_events_0050": len(SIM_E["log"])},
        {"label": "除權息警告", "value": ("；".join(XR["warns"]) if XR["warns"] else f"無（除權息資料涵蓋到 {XR['cov_last']}，不早於最新股價日 {last_day}）"), "warning": bool(XR["warns"]), "covered_through": XR["cov_last"]}]
    if XR["warns"]: print("警告：" + "；".join(XR["warns"]), file=sys.stderr)
    ACC["total"] = nav_raw * ACC["f"]
    if round(ACC["total"]) != live_rows[-1]["策略"]: raise SystemExit(f"帳戶總資產 {ACC['total']:.2f} 和 live 最後一天 {live_rows[-1]['策略']} 不一致")
# 歷次買賣（trades.json）：帳戶起始＋每次換股逐筆＋每次換股一列交易成本；金額是 100 萬帳戶口徑（× 換算倍數，四捨五入到元）
TROWS = []; rg_i = cd_i = 0
if ACC:
    _nm = {}
    def name_of(code, d):
        if d not in _nm: _nm[d] = LG.day_table(tsd, d) or {}
        return (_nm[d].get(code) or {}).get("name") or ""
    f_ = ACC["f"]
    for i in range(len(LED["switches"]) + 1):
        for t in [t for t in LED["trades"] if t["換股序"] == i]:
            row = {"換股序": i, "月份": t["月份"], "成交日": t["成交日"], "動作": t["動作"], "代號": t["代號"], "名稱": name_of(t["代號"], t["成交日"]),
                   "類別": "0050" if t["代號"] == "0050" else "個股", "股數": t["股數"], "帳戶股數": round(t["股數"] * f_), "成交價": t["成交價"], "金額": round(t["金額"] * f_),
                   "買進成本": None, "成本金額": None, "已實現損益": None, "已實現損益%": None, "持有起日": None, "期間現金股利": None, "備註": None}
            note = []
            if t["動作"] == "起始": note.append("帳戶起始持股，以起始日開盤價當成本")
            if t["動作"] == "賣出":
                row["買進成本"] = round(t["平均成本"], 4); row["成本金額"] = round(t["成本金額"] * f_); row["已實現損益"] = row["金額"] - row["成本金額"]
                row["已實現損益%"] = round((t["金額"] / t["成本金額"] - 1) * 100, 2); row["持有起日"] = t["持有起日"]
                row["期間現金股利"] = round(t["期間現金股利"] * f_) if t["全部出清"] else None
                rg_i += row["已實現損益"]; cd_i += row["期間現金股利"] or 0
                if not t["全部出清"]: note.append("只賣出一部分；持有期間的現金股利留在這檔的部位上，全部出清時才列")
            if t["動作"] == "買進" and t.get("加碼"): note.append("加碼：和原有部位合併算移動平均成本")
            if t.get("沒有開盤價"): note.append("當天沒有開盤價，用前一個收盤（需人工確認）")
            row["備註"] = "；".join(note) if note else None; TROWS.append(row)
        if i:
            sw_ = LED["switches"][i - 1]; c_ = round(sw_["交易成本"] * f_)
            TROWS.append({"換股序": i, "月份": sw_["月份"], "成交日": sw_["日期"], "動作": "交易成本", "代號": "成本", "名稱": "交易成本", "類別": "成本", "股數": None, "帳戶股數": None, "成交價": None,
                          "金額": c_, "買進成本": None, "成本金額": None, "已實現損益": -c_, "已實現損益%": None, "持有起日": None, "期間現金股利": None,
                          "備註": f"換股前開盤總值 {round(sw_['換股前開盤總值'] * f_):,} 元 − 換股後開盤總值 {round(sw_['換股後開盤總值'] * f_):,} 元（嚴成本：半個跳動點＋每邊 0.3% 滑價＋手續費＋證交稅；已反映在現金檔）"
                                  f"；這次賣出 {round(sw_['賣出金額'] * f_):,} 元、買進 {round(sw_['買進金額'] * f_):,} 元；換股前現金含期間現金股利 {round(sw_['期間現金股利'] * f_):,} 元"})
# 名單結算了沒：最後一段持股檔的生效日晚於名單訊號日＝這份名單已經換股完成（SETTLED）；否則名單是「預計」，持股還是換股前的
SETTLED = True; SW = None; TRL = {}
if ACC:
    SETTLED = SEGS[-1]["生效日"] > L["sig_date"]
    if SETTLED:
        if L.get("exec_date") and L["exec_date"] != SEGS[-1]["生效日"]: raise SystemExit(f"名單的執行日 {L['exec_date']} 和最後一段持股檔的生效日 {SEGS[-1]['生效日']} 不同")
        EXEC_D = SEGS[-1]["生效日"]; EXEC_EST = False
        if MULTI:
            SW = LED["switches"][-1]; TRL = {t["代號"]: t for t in LED["trades"] if t["換股序"] == SW["換股序"]}
    POS = LED["pos"]
# 本月換股紀錄：每筆補 股數、帳戶股數、成交日、成交價（執行日開盤）、金額、備註
if tsd:
    px_exec = day_prices(L["exec_date"]) if L.get("exec_date") else None
    for x, r in zip(acts, Hacts):
        lst_sh = num(r.get("建議股數")); note = []; code = x["代號"]
        st_sh = ACC["sh"].get(x["代號"]) if ACC else None
        if x["動作"] == "未買進":
            x.update({"股數": None, "帳戶股數": None, "成交日": None, "成交價": None, "金額": None, "備註": "名單上有、但沒有買（" + (x.get("原因") or "") + "）"}); continue
        if ACC and (not SETTLED or SW):                    # 有帳本的兩種情況：名單還沒結算（預計）／有上一份持股檔的換股（完整明細）
            f_ = ACC["f"]; x.update({"買進成本": None, "成本金額": None, "已實現損益": None, "已實現損益%": None, "持有起日": None})
            if not SETTLED:
                x["股數"] = lst_sh; x["帳戶股數"] = round(lst_sh * f_) if lst_sh is not None else None; x["成交價"] = None; x["金額"] = None
                if x["動作"] == "續抱": x["成交日"] = None; note.append("續抱，本月沒有買賣")
                else:
                    x["成交日"] = EXEC_D
                    note.append(f"預計 {EXEC_D}{'（預估日期，只扣週末）' if EXEC_EST else ''} 開盤{x['動作']}，還沒成交"
                                + (f"；以訊號日收盤估約 {round(lst_sh * f_ * x['訊號日收盤']):,} 元" if (lst_sh and x.get("訊號日收盤")) else ""))
                if x["動作"] != "買進":
                    if code not in POS: raise SystemExit(f"名單的持股 {code} 不在目前帳戶持股裡（名單要用目前帳戶持股產生）")
                    if lst_sh is None or abs(lst_sh - POS[code]["sh_now"]) > 1e-6: note.append(f"名單股數 {lst_sh} 和帳戶目前股數 {POS[code]['sh_now']:g} 不同")
                    x["買進成本"] = round(POS[code]["cb"] / POS[code]["sh_now"], 4); x["成本金額"] = round(POS[code]["cb"] * f_); x["持有起日"] = POS[code]["since"]
            else:
                t = TRL.get(code); aft = SW["sh_after"]; x["成交日"] = None; x["成交價"] = None; x["金額"] = None
                if x["動作"] == "賣出" and t and t["動作"] == "賣出":
                    x["股數"] = t["股數"]; x["成交日"] = EXEC_D; x["成交價"] = t["成交價"]; x["金額"] = round(t["金額"] * f_)
                    x["買進成本"] = round(t["平均成本"], 4); x["成本金額"] = round(t["成本金額"] * f_); x["已實現損益"] = x["金額"] - x["成本金額"]
                    x["已實現損益%"] = round((t["金額"] / t["成本金額"] - 1) * 100, 2); x["持有起日"] = t["持有起日"]
                    if t["期間現金股利"]: note.append(f"持有期間另領現金股利 {round(t['期間現金股利'] * f_):,} 元（不含在已實現損益裡）")
                    if not t["全部出清"]: note.append(f"只賣出一部分，還持有 {aft.get(code, 0):g} 股")
                    if t["沒有開盤價"]: note.append("執行日沒有開盤價，成交價是前一個收盤（需人工確認）")
                elif x["動作"] == "賣出":
                    if code not in aft: raise SystemExit(f"名單要賣 {code}，但換股前後的持股檔都沒有這檔")
                    x["動作"] = "續抱"; x["股數"] = aft[code]; note.append("名單要賣，但執行日開盤跌停或沒有開盤價，沒有賣出、仍持有（隔天開盤再掛，需人工追蹤）")
                elif x["動作"] == "買進" and t and t["動作"] == "買進":
                    x["股數"] = t["股數"]; x["成交日"] = EXEC_D; x["成交價"] = t["成交價"]; x["金額"] = round(t["金額"] * f_)
                    if lst_sh is not None and abs(lst_sh - t["股數"]) > 1e-6: note.append(f"名單以訊號日收盤估 {lst_sh:,.0f} 股；照執行日開盤價重算後買 {t['股數']:,.0f} 股")
                    if t["沒有開盤價"]: note.append("執行日沒有開盤價，成交價是前一個收盤（需人工確認）")
                elif x["動作"] == "買進":
                    x["動作"] = "未買進"; x["股數"] = None; note.append("名單要買，但執行日開盤漲停、沒有開盤價或買得到的金額不到目標一半，沒有買進、不遞補")
                else:
                    if code not in aft: raise SystemExit(f"名單續抱 {code}，但換股後持股檔沒有這檔")
                    x["股數"] = aft[code]; note.append("續抱，本月沒有買賣" if not t else f"續抱；另{t['動作']} {t['股數']:g} 股（成交價 {t['成交價']:g}）")
                x["帳戶股數"] = round(x["股數"] * f_) if x["股數"] is not None else None
            x["備註"] = "；".join(note) if note else None
            continue
        if x["動作"] == "賣出":
            x["股數"] = lst_sh
            if lst_sh is None: note.append("股數資料不足（名單沒有這檔的股數）")
        else:
            x["股數"] = st_sh if st_sh is not None else lst_sh
            if st_sh is not None and lst_sh is not None and st_sh != lst_sh:
                note.append(f"名單建議 {lst_sh:,.0f} 股（以訊號日收盤估）；帳戶沿用已發布的 {st_sh:,.0f} 股（見 state/README.md）")
            if x["股數"] is None: note.append("股數資料不足")
        x["帳戶股數"] = round(x["股數"] * ACC["f"]) if (ACC and x["股數"] is not None) else None
        if x["動作"] == "續抱":
            x["成交日"] = None; x["成交價"] = None; x["金額"] = None; note.append("續抱，本月沒有買賣")
        else:
            o = (px_exec or {}).get(x["代號"], (None, None))[0]
            x["成交日"] = L.get("exec_date"); x["成交價"] = o
            x["金額"] = round(x["股數"] * ACC["f"] * o) if (ACC and o and x["股數"] is not None) else None
            if o is None: note.append("成交價資料不足（官方日 K 沒有執行日開盤價）")
            elif x["金額"] is None: note.append("金額資料不足")
        x["備註"] = "；".join(note) if note else None
    act_meta = [{"label": "股數", "value": "持股檔口徑：賣出＝名單的持有股數（" + os.path.basename(hold_csv) + " 的建議股數欄）；續抱、買進＝換股後持股檔（" + (os.path.basename(state_csv) if state_csv else "無") + "）的股數"},
                {"label": "帳戶股數", "value": "100 萬帳戶口徑＝股數 × 換算倍數" + (f" {ACC['f']:.6f}" if ACC else "") + "，四捨五入到整數股（只供閱讀；金額用未四捨五入的股數算）"},
                {"label": "成交價", "value": f"執行日 {L.get('exec_date')} 的官方日 K 開盤價（模擬帳戶假設開盤成交；不是實際券商成交回報）。續抱沒有買賣，成交日、成交價、金額是空的"},
                {"label": "金額", "value": "100 萬帳戶口徑＝股數 × 換算倍數 × 成交價，四捨五入到元；沒有扣手續費、證交稅、滑價"},
                {"label": "備註", "value": "空值＝不適用；資料來源沒有的欄位會在這裡寫「資料不足」，不用猜的"},
                {"label": "沒有的資料", "value": "賣出各檔當初的買進成本與已實現損益：資料不足（state 與 lists 都沒有）。0050 本月的買賣寫在持股表 0050 那一列的「本月交易」"}]
    if ACC and (not SETTLED or SW):
        f_ = ACC["f"]; act_meta = [m for m in act_meta if m["label"] != "沒有的資料"]
        act_meta[0] = {"label": "股數", "value": ("持股檔口徑。名單還沒結算：賣出、續抱＝帳戶目前股數，買進＝名單以訊號日收盤估的股數（執行日會照開盤價重算）" if not SETTLED else
                                                   f"持股檔口徑：賣出＝換股前持股（上一份持股檔，期間除權／分割已調整）減換股後持股；買進＝換股後持股檔（{os.path.basename(SEGS[-1]['持股檔'])}）多出來的股數；續抱＝換股後股數")}
        act_meta[2] = {"label": "成交價", "value": (f"還沒成交（預計 {EXEC_D}{'，預估日期' if EXEC_EST else ''} 開盤）；成交日是預計日期，成交價、金額是空的" if not SETTLED else
                                                     f"執行日 {EXEC_D} 的官方日 K 開盤價（模擬帳戶假設開盤成交；不是實際券商成交回報）。續抱沒有買賣，成交日、成交價、金額是空的")}
        act_meta += [{"label": "買進成本／成本金額", "value": "賣出列（名單還沒結算時也包含續抱列）：買進成本＝這檔的每股平均成本（移動平均；帳戶起始持股用起始日開盤價、之後買的用買進那次執行日開盤價，未含費用；期間除權／分割後按新股數攤算）；成本金額＝賣出股數 × 平均成本 × 換算倍數"},
                     {"label": "已實現損益", "value": "賣出列：金額 − 成本金額（100 萬帳戶口徑，整數元）；已實現損益%＝開盤成交價 ÷ 平均成本 − 1。不含手續費、證交稅、滑價（這些合計列在「本次換股摘要」的交易成本與 trades.json），也不含持有期間領到的現金股利（寫在備註）。還沒成交時是空的"},
                     {"label": "動作", "value": "賣出／續抱／買進；「未買進」＝名單上有但沒有買（已滿 10 檔、買得到的金額不到目標一半、開盤漲停）。0050 的買賣不在這張表（頁面的下單試算只認個股），寫在「本次換股摘要」、持股表 0050 列與 trades.json"}]
        if SW:
            t50 = TRL.get("0050")
            act_meta.append({"label": "本次換股摘要", "value": (
                f"{SW['日期']} 開盤換股（100 萬帳戶口徑）：賣出 {round(SW['賣出金額'] * f_):,} 元、買進 {round(SW['買進金額'] * f_):,} 元（都含 0050）；"
                + (f"0050 {t50['動作']} {round(t50['股數'] * f_):,} 股（持股檔口徑 {t50['股數']:,.0f} 股），成交價 {t50['成交價']:g}，金額 {round(t50['金額'] * f_):,} 元"
                   + (f"，已實現損益 {round(t50['金額'] * f_) - round(t50['成本金額'] * f_):+,} 元" if t50["動作"] == "賣出" else "") + "；" if t50 else "0050 沒有買賣；")
                + f"賣出個股與 0050 的已實現價差合計 {round(SW['已實現價差'] * f_):+,} 元；交易成本（嚴成本：半個跳動點＋每邊 0.3% 滑價＋手續費＋證交稅）{round(SW['交易成本'] * f_):,} 元"
                + f"＝換股前開盤總值 {round(SW['換股前開盤總值'] * f_):,} 元 − 換股後開盤總值 {round(SW['換股後開盤總值'] * f_):,} 元；"
                + f"換股前現金 {round(SW['換股前現金'] * f_):,} 元（含這一段期間領到的現金股利 {round(SW['期間現金股利'] * f_):,} 元）→ 換股後現金 {round(SW['換股後現金'] * f_):,} 元"),
                "switch_date": SW["日期"], "cost": round(SW["交易成本"] * f_), "realized_gross": round(SW["已實現價差"] * f_)})
dump("actions", acts, [{"label": "來源", "value": os.path.basename(hold_csv) + "（策略回測持股延續到本月的換股結果）"},
                       {"label": "訊號日以來%", "value": "最新收盤 ÷ 訊號日收盤 − 1（基準是訊號日收盤，不是成交日開盤；持股表的「本月以來%」才是對起始價）"}] + act_meta)
# 目前持股（有換股後持股 csv 時加「股數」與「起始價」＝上線起始日開盤價，並加一列 0050）
hold = [ {k: x.get(k) for k in ("代號", "名稱", "名次", "SURGE", "訊號日收盤", "最新收盤")} | {"狀態": "新買" if x["動作"] == "買進" else "續抱"} for x in acts if x["動作"] in ("續抱", "買進")]
if ACC and not SETTLED:        # 名單還沒結算：持股表是換股前的持股（名單的賣出＋續抱），買進的還沒進來
    hold = [ {k: x.get(k) for k in ("代號", "名稱", "名次", "SURGE", "訊號日收盤", "最新收盤")} | {"狀態": "將賣出" if x["動作"] == "賣出" else "續抱"} for x in acts if x["動作"] in ("續抱", "賣出")]
hold_meta = [{"label": "說明", "value": "另有 0050（底倉 30%＋衛星沒用完的資金）"}]
if tsd and state_csv:
    sh = dict(ACC["sh"]) if ACC else {r["代號"]: float(r["股數"]) for r in rd(state_csv)}
    p0 = day_prices(live_start) if live_start else None
    ps = day_prices(L["sig_date"]) or {}
    if "0050" in sh:
        c0 = last_px.get("0050", (None, None))[1]; s0 = ps.get("0050", (None, None))[1]
        hold.append({"代號": "0050", "名稱": "元大台灣50", "名次": None, "SURGE": None, "訊號日收盤": s0, "最新收盤": c0, "狀態": "0050"})
    for x in hold:
        x["股數"] = sh.get(x["代號"])
        x["起始價"] = (p0 or {}).get(x["代號"], (None, None))[0]
        if ACC and x["代號"] in POS:                           # 有帳本：起始價＝這檔的每股平均成本（只有一筆買進、沒有部分賣出時就是那次的開盤價）
            x["起始價"] = POS[x["代號"]]["px0"] if POS[x["代號"]]["lots"] == 1 else round(POS[x["代號"]]["px0"], 4)
    hold_meta += [{"label": "股數", "value": os.path.basename(SEGS[-1]["持股檔"] if SEGS else state_csv) + "（持股檔原始股數：回測持股按比例縮放＋本月新買；金額口徑見「帳戶口徑」）"},
                  {"label": "起始價", "value": f"{live_start} 開盤價（上線起始）"}]
    if cash_src:
        hold_meta.append({"label": "現金", "value": f"另有現金 {cash:,.0f} 元（{os.path.basename(cash_src)}；持股檔口徑），已計入 live.json 淨值"})
# 本月以來%：有起始價時＝最新收盤 ÷ 起始價 − 1（和起始價欄、live.json 同一個起點）；沒有起始價時才用訊號日收盤
for x in hold:
    b = x.get("起始價") or None; c = x.get("最新收盤")
    if MULTI and ACC: b = LED["seg_open"].get(x["代號"]) or b       # 兩段以上：本月以來% 的基準是最近一次換股日的開盤價
    if b is None: b = x.get("訊號日收盤")
    x["本月以來%"] = round((c / b - 1) * 100, 2) if (c and b) else None
hold_meta.append({"label": "本月以來%", "value": "最新收盤 ÷ 起始價（上線起始日開盤）− 1；沒有起始價時改用訊號日收盤當基準"})
# 完整交易資訊：金額（100 萬帳戶口徑）、換股點位（季線、試算名次、出場條件）、現金與合計兩列
SUM = None
if ACC:
    chk_sig, chk_exec = (next_sig, next_exec) if SETTLED else (L["sig_date"], EXEC_D)        # 名單還沒結算時，「檢查日」就是名單的訊號日、賣出在名單的執行日
    f = ACC["f"]; ns, ne, ld = md(chk_sig), md(chk_exec), md(last_day)
    rev_now = int(L["rev_month"][5:7]); rev_next = (int(chk_sig[5:7]) + 10) % 12 + 1      # 下次訊號日用「上一個月」的營收
    stock_codes = [x["代號"] for x in hold if x["狀態"] != "0050"]
    if sorted(x["代號"] for x in hold) != sorted(ACC["sh"]): raise SystemExit("持股表的代號和換股後持股檔不同：" + str(sorted(ACC["sh"])))
    TR, tinfo = ma_trial(stock_codes)
    # 金額：逐列四捨五入到元，而且加總＝合計（成本合計＝起始資金；市值合計＝live 最後一天）
    # 含息口徑：成本用起始股數與起始現金；市值用目前股數（除權／分割調整後）與目前現金（含已領現金股利）。
    # 現金股利的損益記在該股那一列（損益金額＝最新市值 − 成本金額＋已領現金股利），錢放在現金列；沒有事件時和舊算法逐元相同。
    # 兩段以上（跨月銜接）：成本＝帳本的移動平均成本；現金列成本＝目前現金 − 目前持股領過的股利；多一列「已實現」＝歷次換股的已實現價差＋已出清持股的股利 − 交易成本，
    # 所以 個股與 0050 成本＋現金成本＝起始資金＋已實現、各列損益加總（含已實現列）＝累計損益。
    segsh = ACC["sh"]; shn = ACC["sh_now"]; cash_now = ACC["cash_now"]; DIV = {k: p["div"] for k, p in POS.items()}
    sh0 = {k: p["sh0"] for k, p in POS.items()}; seg_d = SEGS[-1]["生效日"]; seg_cash = SEGS[-1]["cash"]; seg_cash_src = SEGS[-1]["cash_src"]
    ev_by = {}
    for lg in ACC["log"]:
        if lg["代號"] in POS and lg.get("套用日", lg["日期"]) > POS[lg["代號"]]["since"]: ev_by.setdefault(lg["代號"], []).append(lg)   # 只算這個部位持有期間的事件
    real_raw = LED["real_gross"] + LED["closed_div"] - LED["costs"]
    REAL_i = round(real_raw * f) if MULTI else 0
    cash_cost = cash if not MULTI else cash_now - sum(DIV.values())
    cost_i = alloc([POS[x["代號"]]["cb"] * f for x in hold] + [cash_cost * f], START_CAPITAL + REAL_i)
    total_i = ACC["live"][-1]["策略"]
    mv_i = alloc([shn[x["代號"]] * x["最新收盤"] * f for x in hold] + [cash_now * f], total_i)
    div_tot = mv_i[-1] - cost_i[-1]                       # 現金列增加的金額＝已領現金股利（100 萬帳戶口徑，整數元）
    div_i = [0] * len(hold)
    if not ACC["log"]:
        if div_tot != 0: raise SystemExit("現金在成本與市值兩邊四捨五入後不同")
    else:
        dv = [DIV.get(x["代號"], 0.0) * f for x in hold]
        div_i = [round(v) for v in dv]; left = div_tot - sum(div_i)
        if abs(left) > len(hold) + 2: raise SystemExit(f"現金股利分配檢查失敗：現金列增加 {div_tot} 元，各股股利合計 {sum(dv):.2f} 元")
        if left:                                           # 四捨五入的尾差放在股利最大（沒有股利時市值最大）的一列
            k = max(range(len(hold)), key=lambda i: (dv[i], mv_i[i])); div_i[k] += left
    e50_plan = None; base_sh = None
    pj = re.sub(r"\.csv$", ".json", hold_csv)
    if pj != hold_csv and os.path.exists(pj):
        try:
            PL = (json.load(open(pj, encoding="utf-8")).get("plan") or {}); e50_plan = PL.get("e50"); base_sh = (PL.get("band") or {}).get("base_shares_new")
        except Exception: e50_plan = None
    e50_note = next((r["說明"] for r in H if r["區塊"] == "0050"), "")
    for i, x in enumerate(hold):
        x["列別"] = "0050" if x["狀態"] == "0050" else "個股"
        code = x["代號"]; acc0 = round(sh0[code] * f); evs = ev_by.get(code, [])
        if shn[code] != segsh[code]:                        # 除權／分割調整過股數：股數欄改成目前股數，本月以來% 用調整後的價格比
            x["股數"] = shn[code]; x["本月以來%"] = round((x["最新收盤"] * shn[code] / segsh[code] / LED["seg_open"][code] - 1) * 100, 2)
        x["帳戶股數"] = round(x["股數"] * f)
        x["成本金額"] = cost_i[i]; x["最新市值"] = mv_i[i]; x["損益金額"] = mv_i[i] - cost_i[i] + div_i[i]
        x["損益%"] = (round((x["最新收盤"] / x["起始價"] - 1) * 100, 2) if (not evs and POS[code]["lots"] == 1 and not DIV.get(code))
                      else round(((shn[code] * x["最新收盤"] + DIV.get(code, 0.0)) / POS[code]["cb"] - 1) * 100, 2))
        tl = TRL.get(code); chk_w = "下次" if SETTLED else "本次"
        ev_note = None
        if evs:
            ev_note = "期間除權息（含息口徑）：" + "；".join(
                (f"{md(l['日期'])} 除息每股 {l['權值＋息值']:g} 元" if l["現金股利"] else f"{md(l['日期'])} {l['種類']}，股數 × {l['股數倍數']:.6f}" + ("（近似）" if l["近似"] else "")) if not l.get("error")
                else f"{md(l['日期'])} {l['種類']}：{l['error']}" for l in evs) + (f"。已領現金股利 {div_i[i]:,} 元，已計入本列損益金額與損益%，錢放在現金列" if div_i[i] else "")
        x["占總資產%"] = round(x["股數"] * x["最新收盤"] * f / ACC["total"] * 100, 2)
        if x["列別"] == "個股":
            t = TR[x["代號"]]
            if t.get("_close") is not None and abs(t["_close"] - x["最新收盤"]) > 1e-9: raise SystemExit(f"{x['代號']} 試算用的收盤 {t['_close']} 和最新收盤 {x['最新收盤']} 不同")
            x["季線價位"] = t["季線價位"]; x["距季線%"] = t["距季線%"]; x["試算名次"] = t["試算名次"]; x["試算基準日"] = last_day; x["目前狀態"] = t["目前狀態"]
            rk0 = f"訊號日（{md(L['sig_date'])}）第 {x['名次']:.0f} 名" if x.get("名次") else f"訊號日（{md(L['sig_date'])}）名次資料不足"
            rk1 = (f"第 {t['試算名次']} 名" if t["試算名次"] else "已不在排名內（季線下的股票不列入排名）" if t["目前狀態"] == "會賣：跌破季線" else "不在排名內（" + t["原因"] + "）")
            if t["季線價位"] is None:
                x["出場條件"] = f"只在換股訊號日檢查，平常跌破不賣。{chk_w} {ns} 收盤檢查，不合格就在 {ne} 開盤賣出。季線資料不足（{t['原因']}）。"
            else:
                x["出場條件"] = (f"只在換股訊號日檢查，平常跌破不賣。{chk_w} {ns} 收盤檢查：收盤（還原）≤ 季線（以 {ld} 資料估計約 {t['季線價位']:.1f} 元；季線每天會變），"
                                 f"或跌出前 20 名（{ns} 會用 {rev_next} 月營收重新排名），就在 {ne} 開盤賣出。"
                                 f"目前：{ld} 收盤 {x['最新收盤']:g} 元，距季線 {t['距季線%']:+.1f}%；名次：{rk0}，用同一份 {rev_now} 月營收、{ld} 股價試算{rk1}。"
                                 f"若今天就是訊號日，試算結果是「{t['目前狀態']}」。")
            x["本月交易"] = (f"{md(live_start)} 開盤買進 {acc0:,} 股，成交價 {x['起始價']:g} 元，金額 {x['成本金額']:,} 元" if x["狀態"] == "新買"
                             else f"續抱，本月沒有買賣；帳戶以 {md(live_start)} 開盤價 {x['起始價']:g} 元起算")
            if not SETTLED:
                x["本月交易"] = (f"名單（{md(L['sig_date'])} 收盤）判定賣出，預計 {ne} 開盤賣出，還沒成交" if x["狀態"] == "將賣出" else f"名單（{md(L['sig_date'])} 收盤）判定續抱，沒有買賣") + f"；平均成本 {x['起始價']:g} 元（{POS[code]['since']} 起持有）"
            elif SW:
                x["本月交易"] = (f"{md(seg_d)} 開盤買進 {round(tl['股數'] * f):,} 股，成交價 {tl['成交價']:g} 元，金額 {round(tl['金額'] * f):,} 元" if (tl and tl["動作"] == "買進" and not tl.get("加碼"))
                                 else f"續抱，本月沒有買賣；平均成本 {x['起始價']:g} 元（{POS[code]['since']} 起持有），{md(seg_d)} 開盤價 {LED['seg_open'][code]:g} 元")
            x["備註"] = ev_note
        else:
            x["季線價位"] = None; x["距季線%"] = None; x["試算名次"] = None; x["試算基準日"] = last_day; x["目前狀態"] = "不適用（底倉＋閒置資金）"
            x["出場條件"] = ("不適用：0050 是底倉＋還沒用到的資金，不看季線、不看名次，換股日不會因為跌而賣出。"
                             f"只有 {ns} 名單要買的股票錢不夠（現金用完）時才賣一部分 0050；衛星占總資產超出 60%～80% 時只改記底倉股數。")
            if e50_plan and e50_plan.get("shares_before") is not None and e50_plan.get("shares_after") is not None:
                b0, b1 = e50_plan["shares_before"], e50_plan["shares_after"]
                x["本月交易"] = (f"持股檔口徑：換股前 {b0:,.0f} 股，{'買進' if b1 >= b0 else '賣出'} {abs(b1 - b0):,.0f} 股，換股後 {b1:,.0f} 股（名單以訊號日收盤估算的股數）；"
                                 f"帳戶以 {md(live_start)} 開盤價 {x['起始價']:g} 元起算，換股後帳戶股數 {acc0:,} 股")
                if b1 != x["股數"]: x["本月交易"] += f"；注意：名單換股後 {b1:,.0f} 股和持股檔 {x['股數']:,.0f} 股不同"
            else:
                x["本月交易"] = "0050 本月買賣股數：資料不足（找不到名單的 json）"
            if not SETTLED:
                x["本月交易"] = ((f"名單（{md(L['sig_date'])} 收盤）預計{'買進' if e50_plan['shares_after'] >= e50_plan['shares_before'] else '賣出'} {abs(e50_plan['shares_after'] - e50_plan['shares_before']):,.0f} 股（持股檔口徑，以訊號日收盤估），{ne} 開盤執行，還沒成交"
                                  if (e50_plan and e50_plan.get("shares_after") is not None) else "名單的 0050 買賣股數：資料不足") + f"；平均成本 {x['起始價']:g} 元")
                base_sh = SEGS[-1]["base_shares"] if SEGS[-1]["base_shares"] is not None else None      # 還沒換股：底倉是上一段的
            elif SW:
                x["本月交易"] = ((f"{md(seg_d)} 開盤{tl['動作']} {round(tl['股數'] * f):,} 股（持股檔口徑 {tl['股數']:,.0f} 股），成交價 {tl['成交價']:g} 元，金額 {round(tl['金額'] * f):,} 元"
                                  + (f"，已實現損益 {round(tl['金額'] * f) - round(tl['成本金額'] * f):+,} 元" if tl["動作"] == "賣出" else "")) if tl else f"{md(seg_d)} 換股沒有買賣 0050")
                x["本月交易"] += f"；換股後帳戶股數 {round(segsh[code] * f):,} 股，平均成本 {x['起始價']:g} 元"
                if SEGS[-1]["base_shares"] is not None: base_sh = SEGS[-1]["base_shares"]
            x["備註"] = ("其中底倉 " + (f"{base_sh:,.0f}" if base_sh is not None else "（資料不足）") + " 股（持股檔口徑；名單說明：" + (e50_note or "無") + "），其餘是衛星沒用完的資金")
            if ev_note: x["備註"] += "；" + ev_note
    cash_i = mv_i[-1]; cash_cost_i = cost_i[-1]; n_st = sum(1 for x in hold if x["列別"] == "個股")
    st_mv = sum(x["最新市值"] for x in hold if x["列別"] == "個股"); e_mv = sum(x["最新市值"] for x in hold if x["列別"] == "0050")
    st_cost = sum(x["成本金額"] for x in hold if x["列別"] == "個股"); e_cost = sum(x["成本金額"] for x in hold if x["列別"] == "0050")
    e_nav = ACC["live"][-1]["0050"]; pnl = total_i - START_CAPITAL
    unreal_i = sum(x["損益金額"] for x in hold)                  # 目前持股的損益（市值 − 成本＋目前持股領過的現金股利）
    # 檢查（不過就報錯、不寫檔）
    if st_mv + e_mv + cash_i != total_i: raise SystemExit("檢查失敗：個股市值＋0050 市值＋現金 ≠ 總資產")
    if st_cost + e_cost + cash_cost_i != START_CAPITAL + REAL_i: raise SystemExit("檢查失敗：各列成本＋現金 ≠ 起始資金＋已實現")
    if unreal_i + REAL_i != pnl: raise SystemExit("檢查失敗：各列損益加總（含已實現）≠ 累計損益")
    if abs(sum(x["股數"] * x["最新收盤"] for x in hold) + cash_now - ACC["nav_raw"]) > 1e-6: raise SystemExit("檢查失敗：持股表和 live 用的股數或價格不同")
    blank = {"名次": None, "SURGE": None, "訊號日收盤": None, "最新收盤": None, "股數": None, "起始價": None, "本月以來%": None, "帳戶股數": None,
             "季線價位": None, "距季線%": None, "試算名次": None, "試算基準日": last_day, "目前狀態": "不適用", "出場條件": "不適用"}
    hold.append({"代號": "現金", "名稱": "現金", "狀態": "現金", "列別": "現金"} | blank | {
        "成本金額": cash_cost_i, "最新市值": cash_i, "損益金額": 0, "損益%": 0.0, "占總資產%": round(cash_now * f / ACC["total"] * 100, 2), "本月交易": "沒有買賣",
        "備註": (f"持股檔現金 {cash:,.0f} 元（{os.path.basename(cash_src) if cash_src else '無現金檔，以 0 計'}）× 換算倍數 {f:.6f}；不計利息"
                 + (f"；另含期間已領現金股利 {div_tot:,} 元（持股檔口徑 {cash_now - cash:,.2f} 元 × 換算倍數），股利的損益記在各股那一列，所以本列損益是 0" if div_tot else "")) if not MULTI else
                (f"{seg_d} 換股後現金 {seg_cash:,.0f} 元（{os.path.basename(seg_cash_src) if seg_cash_src else '無現金檔，以 0 計'}；已含之前各段領到的現金股利）× 換算倍數 {f:.6f}"
                 + (f"＋{seg_d} 之後領到的現金股利（持股檔口徑 {cash_now - seg_cash:,.2f} 元）" if abs(cash_now - seg_cash) > 1e-9 else "") + "；不計利息。"
                 + f"本列成本金額＝目前現金扣掉目前持股領過的現金股利 {div_tot:,} 元（股利的損益記在各股那一列，所以本列損益是 0）；下次換股時這些現金（含股利）全部算進可用現金")})
    if MULTI:                                                # 已實現一列（兩段以上才有）：歷次換股的已實現價差＋已出清持股領過的股利 − 交易成本
        tc_i = rg_i + cd_i - REAL_i                           # rg_i、cd_i＝trades 逐筆（整數元）加總；交易成本取差額，尾差放在最後一次換股的交易成本列
        _cr = [r for r in TROWS if r["動作"] == "交易成本"]; _adj = tc_i - sum(r["金額"] for r in _cr)
        if abs(_adj) > len(TROWS) + 2: raise SystemExit(f"檢查失敗：交易成本尾差 {_adj} 元太大")
        if _adj: _cr[-1]["金額"] += _adj; _cr[-1]["已實現損益"] -= _adj; _cr[-1]["備註"] += f"；含四捨五入尾差 {_adj:+,} 元（讓已實現合計和持股表逐元相同）"
        if sum(r["已實現損益"] or 0 for r in TROWS) + cd_i != REAL_i: raise SystemExit("檢查失敗：trades 的已實現損益加總 ≠ 持股表已實現列")
        hold.append({"代號": "已實現", "名稱": "已實現損益", "狀態": "已實現", "列別": "已實現"} | blank | {
            "成本金額": None, "最新市值": None, "損益金額": REAL_i, "損益%": round(REAL_i / START_CAPITAL * 100, 2), "占總資產%": None,
            "本月交易": f"{md(SW['日期'])} 換股：已實現價差 {round(SW['已實現價差'] * f):+,} 元、交易成本 {round(SW['交易成本'] * f):,} 元" if SW else f"名單還沒結算；最近一次換股 {md(seg_d)}",
            "備註": (f"{live_start} 以來 {len(LED['switches'])} 次換股累計：賣出的已實現價差 {rg_i:+,} 元＋已出清持股持有期間領的現金股利 {cd_i:,} 元 − 交易成本 {tc_i:,} 元＝{REAL_i:+,} 元。"
                     "價差＝賣出股數 ×（執行日開盤價 − 平均成本）；交易成本＝每次換股前後開盤總值的差（嚴成本，含四捨五入尾差）。逐筆見 trades.json。"
                     "本列沒有市值；損益%＝本列損益 ÷ 起始資金")})
    diff_pp = round((total_i - e_nav) / START_CAPITAL * 100, 2)
    hold.append({"代號": "合計", "名稱": "合計", "狀態": "合計", "列別": "合計"} | blank | {
        "成本金額": START_CAPITAL, "最新市值": total_i, "損益金額": pnl, "損益%": round(pnl / START_CAPITAL * 100, 2), "占總資產%": 100.0,
        "本月交易": f"{md(live_start)} 開盤以 {START_CAPITAL:,} 元起算",
        "備註": (f"起始資金 {START_CAPITAL:,} 元（{live_start} 開盤）→ {last_day} 收盤總資產 {total_i:,} 元：個股 {n_st} 檔 {st_mv:,} 元＋0050 {e_mv:,} 元＋現金 {cash_i:,} 元；"
                 f"累計損益 {pnl:+,} 元（{pnl / START_CAPITAL * 100:+.2f}%）" + (f"＝已實現 {REAL_i:+,} 元（歷次換股）＋目前持股損益 {unreal_i:+,} 元" if MULTI else "") + f"。同期只買 0050：{e_nav:,} 元（{(e_nav / START_CAPITAL - 1) * 100:+.2f}%）；"
                 f"策略{'領先' if diff_pp >= 0 else '落後'} {abs(diff_pp):.2f} 個百分點。沒有扣手續費、證交稅；" + ("0050 與個股都沒有另外加股利" if not (ACC["log"] or ACC["log_e"]) else
                    f"含息口徑：策略帳戶已領現金股利 {div_tot:,} 元（在現金列），同期只買 0050 也含 0050 的現金股利（" + ("除息日收盤再投入" if BENCH_REINVEST else "留在現金、不再投入") + "；除權息事件見 meta）"))})
    n_sell = sum(1 for x in hold if str(x.get("目前狀態", "")).startswith("會賣"))
    SUM = {"帳戶起始日": live_start, "起始資金": START_CAPITAL, "目前總資產": total_i, "累計損益金額": pnl, "累計損益%": round(pnl / START_CAPITAL * 100, 2),
           "個股檔數": n_st, "個股市值合計": st_mv, "0050市值": e_mv, "現金": cash_i, "同期只買0050總資產": e_nav, "同期只買0050損益%": round((e_nav / START_CAPITAL - 1) * 100, 2),
           "領先0050_個百分點": diff_pp, "下次換股檢查日": chk_sig, "下次換股執行日": chk_exec,
           "下次檢查條件": (f"只在 {ns} 收盤檢查一次（平常跌破不賣）：個股收盤（還原）≤ 季線（60 日均線），或還在季線上但月營收驚奇排名（改用 {rev_next} 月營收）掉出前 20 名，"
                            f"就在 {ne} 開盤賣出；0050 不適用。"),
           "試算基準日": last_day, "試算會賣檔數": n_sell,
           "試算說明": (f"用 {ld} 的股價與季線、名單那一份 {rev_now} 月營收試算「若今天就是訊號日」：{n_st} 檔個股中 {n_sell} 檔會賣、{n_st - n_sell} 檔續抱。"
                        f"只是試算，真正只看 {ns} 收盤，而且那天會改用 {rev_next} 月營收重新排名。"),
           "帳戶換算倍數": round(f, 6), "持股檔起始日開盤總值": round(ACC["base_raw"], 2)}
    if MULTI:
        SUM.update({"換股次數": len(LED["switches"]), "最近換股日": seg_d, "累計已實現損益": REAL_i, "累計已實現價差": rg_i, "已出清持股現金股利": cd_i, "累計交易成本": tc_i,
                    "目前持股損益": unreal_i, "本月以來損益金額": total_i - round(LED["switches"][-1]["換股後開盤總值"] * f)})
    if not SETTLED:
        SUM.update({"名單狀態": "預計", "名單說明": f"{L['sig_date']} 收盤後的名單還沒執行（預計 {EXEC_D}{'，預估日期' if EXEC_EST else ''} 開盤換股）；持股表是換股前的持股"})
    hold_meta[0] = {"label": "說明", "value": "策略模擬帳戶（不是任何人的實際持股）：個股各一列、0050 一列（底倉＋衛星沒用完的資金）、現金一列、合計一列（看「列別」欄）"}
    hold_meta += [
        {"label": "帳戶口徑", "value": f"{live_start} 開盤以 {START_CAPITAL:,} 元起算。持股檔股數在那天開盤連現金共 {ACC['base_raw']:,.0f} 元，不是剛好 100 萬，所以金額一律乘上換算倍數 {f:.6f}（＝100 萬 ÷ {ACC['base_raw']:,.1f}，和 live.json 的正規化相同）；百分比不受影響"},
        {"label": "股數／帳戶股數", "value": "股數＝持股檔原始股數；帳戶股數＝股數 × 換算倍數，四捨五入到整數股（只供閱讀）。金額用未四捨五入的股數算，所以帳戶股數 × 價格和金額欄可能差幾十元"},
        {"label": "成本金額／最新市值／損益金額", "value": "成本金額＝股數 × 換算倍數 × 起始價；最新市值＝股數 × 換算倍數 × 最新收盤；損益金額＝最新市值 − 成本金額。逐列四捨五入到元並調整到加總剛好等於合計列；沒有扣手續費與證交稅"},
        {"label": "含息口徑", "value": "2026-11-18 起是含息總報酬口徑（live_tr.py）：持有期間遇除息日，現金股利（息值 × 股數）加進現金列，並記在該股的損益金額與損益%（該股 損益金額＝最新市值 − 成本金額＋已領現金股利；現金列損益記 0）；"
                                       "除權、權息、分割、減資調整股數（股數欄＝調整後股數，成本仍用起始股數 × 起始價）。期間沒有事件時，各欄和舊算法逐元相同。事件與資料涵蓋範圍見下面三列"},
        {"label": "損益%／占總資產%", "value": "損益%＝最新收盤 ÷ 起始價 − 1（個股與 0050 列同「本月以來%」；期間有除權息的股票改成（目前股數 × 最新收盤＋已領現金股利）÷（起始股數 × 起始價）− 1；合計列＝累計損益 ÷ 起始資金）；占總資產%＝最新市值 ÷ 合計列總資產"},
        {"label": "合計列檢查", "value": f"合計列最新市值 {total_i:,} 元＝live.json 最後一天（{last_day}）策略淨值；個股＋0050＋現金＝合計、各列損益加總＝累計損益，程式逐元檢查，不一致就報錯不寫檔"},
        {"label": "季線價位／距季線%", "value": f"季線＝60 日均線，用策略 E0 的還原收盤口徑（名單產生器 make_list.py 的 compute()，和前 20 名表的「距季線%」同一段程式）。距季線%＝{last_day} 還原收盤 ÷ 季線 − 1；季線價位＝最新收盤 ÷（1＋距季線），也就是把季線換算成今天的股價。以最新資料估計，每天會變。{tinfo['訊號日比對']}"},
        {"label": "名次／試算名次", "value": f"名次＝訊號日 {L['sig_date']} 的名次（{rev_now} 月營收，月內不變）。試算名次＝同一份 {rev_now} 月營收分數（名單的截止版），只把股價、季線、股池換成 {last_day} 重排（季線上 {tinfo['試算季線上檔數']} 檔）；季線下的股票不列入排名所以是空的。{next_sig} 真正檢查時會改用 {rev_next} 月營收，名次會再變"},
        {"label": "目前狀態", "value": f"以 {last_day} 資料試算「若今天就是訊號日會不會被賣」（make_list.py 的 held_status()）：續抱／會賣：跌破季線／會賣：跌出前 20 名／會賣：其他原因。只是試算，不是賣出訊號"},
        {"label": "出場條件", "value": f"只在換股訊號日（下次 {next_sig}）收盤檢查，隔天（{next_exec}）開盤賣出；平常跌破季線不賣。0050、現金、合計列不適用"},
        {"label": "本月交易／備註", "value": "本月交易＝這一列在本月換股做了什麼；0050 列的買賣股數取自名單 json（以訊號日收盤估算）。資料來源沒有的寫「資料不足」"}] + ACC["xr"]["meta"]
    if tinfo["未確認格子"]:
        hold_meta.append({"label": "除權息未確認", "value": "近 60 日有漲跌價差＝0 但收盤有變、又不在除權息封存的日子，季線可能少算股利，請人工確認：" + "、".join(tinfo["未確認格子"])})
    if tinfo["除權息修正"] and tinfo["封存之後的日子"]:
        hold_meta.append({"label": "除權息封存", "value": f"除權息參考價封存到 {tinfo['封存最後一天']}，之後有 {tinfo['封存之後的日子']} 個交易日還沒涵蓋；這段期間持股沒有偵測到可疑的除權息日" if not tinfo["未確認格子"] else f"除權息參考價封存到 {tinfo['封存最後一天']}"})
if ACC and MULTI:
    hold_meta.append({"label": "跨月銜接", "value": (
        f"帳戶由 {len(SEGS)} 份換股後持股檔接成（{'、'.join(s['生效日'] + ' ' + os.path.basename(s['持股檔']) for s in SEGS)}），每份在生效日開盤換股；換算倍數各月份相同，所以淨值連續。"
        "「起始價」＝每股平均成本（移動平均：帳戶起始持股用起始日開盤價，之後買的用買進那次執行日開盤價，未含費用）；「成本金額」＝平均成本 × 目前部位；"
        "「本月以來%」的基準是最近一次換股日的開盤價，「損益%」的基準是平均成本（含持有期間現金股利）。"
        "多一列「已實現」（列別＝已實現）：歷次換股賣出的已實現價差＋已出清持股領過的現金股利 − 交易成本，所以 個股＋0050 成本＋現金成本＝起始資金＋已實現、各列損益加總（含已實現列）＝累計損益；"
        "現金列成本金額＝目前現金扣掉目前持股領過的現金股利。逐筆買賣在 trades.json"), "segments": [{"月份": s["月份"], "生效日": s["生效日"]} for s in SEGS]})
if ACC and not SETTLED:
    hold_meta.append({"label": "名單還沒結算", "value": f"{L['sig_date']} 收盤後的名單預計 {EXEC_D}{'（預估日期，只扣週末）' if EXEC_EST else ''} 開盤執行，還沒寫成換股後持股檔；本表是換股前的持股，「狀態」＝續抱／將賣出，"
                                                         "「目前狀態」是用最新一天股價的試算（訊號日當晚就等於名單的判定）；要買進的股票還不在表上（見換股清單）"})
dump("holdings", hold, hold_meta)
# 前 20 名
top = [{"名次": t["rank_ma"], "代號": t["code"], "名稱": t["name"], "SURGE": round(t["surge"], 3), "前20日漲幅%": round(t["ret20"] * 100, 1),
        "距季線%": round(t["dist_ma60"] * 100, 1), "收盤": t["close"], "入選": "是" if (t["rank_ma"] <= 10 and t["ret20"] > 0.10) else ""} for t in L["top20"]]
dump("top20", top, [{"label": "定義", "value": "收盤在季線上的上市股票，依 SURGE 由大到小；前 10 名中前 20 日漲幅 > 10% 的入選"}])
# 風險提示：README 的回測數字（寫成常數，標明出處）
RISK = [
    ("回撤_單一路徑", -32.1, "%", "最大回撤（100 萬、標準換股日那一條，2024-07 到 2025-04；除權息＋期限修正後，與只修除權息相同；修正前 −33.4）", "xq_scripts/README.md 第 12 節「2026-11-17 起正式數字＝除權息＋期限修正」"),
    ("回撤_20條中位數", -33.4, "%", "最大回撤，20 個起始日的中位數（除權息＋期限修正後；只修除權息 −33.5，修正前 −34.0）", "xq_scripts/README.md 第 12 節「2026-11-17 起正式數字＝除權息＋期限修正」"),
    ("回撤_壞的時候", -36.4, "%", "最大回撤，100 條路徑裡差的 10%（除權息＋期限修正後，2026-11-17 重算；期限修正前（只修除權息）−36.1）", "agents/official_nav/resync/resync_results.json（除權息＋期限修正後，2026-11-17 重算）；方法同 agents/s26_exright_robustness（S9 A10～A15 × 20 條）"),
    ("0050_回撤", -34.0, "%", "0050 同期最大回撤（2022）", "xq_scripts/README.md 第 12 節"),
    ("落後0050最多", 15.3, "個百分點", "2025 年 0050 大漲時，滾動 12 個月最多落後 0050（到 2025-10-07；除權息＋期限修正後，與只修除權息相同；修正前 17.6）", "xq_scripts/README.md 第 12 節「2026-11-17 起正式數字＝除權息＋期限修正」"),
    ("落後時間比例", 20.2, "%", "滾動 12 個月落後 0050 的時間比例（除權息＋期限修正後；只修除權息 19.5，修正前 19.8）", "xq_scripts/README.md 第 12 節「2026-11-17 起正式數字＝除權息＋期限修正」"),
    ("最賺5筆占比", 36, "%", "最賺 5 筆占已平倉損益（除權息＋期限修正後，與只修除權息相同；修正前 37）", "xq_scripts/README.md 第 12 節「2026-11-17 起正式數字＝除權息＋期限修正」"),
    ("年化_全部", 48.76, "%", "2019 年起每年（單一路徑；除權息＋期限修正後；只修除權息 48.99，修正前 48.13）", "xq_scripts/README.md 第 12 節「2026-11-17 起正式數字＝除權息＋期限修正」"),
    ("年化_不買最賺5檔", 44.88, "%", "完全不買最賺 5 檔，2019 年起每年（除權息＋期限修正後；只修除權息 45.07，修正前 43.42）", "xq_scripts/README.md 第 12 節「2026-11-17 起正式數字＝除權息＋期限修正」"),
    ("勝率", 50.4, "%", "賺錢的交易比例（381 筆；除權息＋期限修正後；只修除權息 381 筆 51.7，修正前 379 筆 51.2）", "xq_scripts/README.md 第 12 節「2026-11-17 起正式數字＝除權息＋期限修正」"),
    ("賺的平均", 27.43, "%", "賺錢的交易平均賺（除權息＋期限修正後；只修除權息 26.75，修正前 27.16）", "xq_scripts/README.md 第 12 節「2026-11-17 起正式數字＝除權息＋期限修正」"),
    ("賠的平均", -9.11, "%", "賠錢的交易平均賠（除權息＋期限修正後；只修除權息 −9.26，修正前 −9.24）", "xq_scripts/README.md 第 12 節「2026-11-17 起正式數字＝除權息＋期限修正」"),
    ("晚1天少", 3.7, "個百分點", "晚 1 天下單，2019 年起每年少（每檔＝衛星 ÷ 10、底倉不動，5 種資金中位數；除權息＋期限修正後，2026-11-17 重算；期限修正前 3.5）", "agents/official_nav/resync/resync_results.json（除權息＋期限修正後，2026-11-17 重算）；方法同 agents/s26_exright_robustness（S7 W10_70L）"),
    ("晚2天少", 6.2, "個百分點", "晚 2 天下單，2019 年起每年少（同上口徑；除權息＋期限修正後，2026-11-17 重算；期限修正前 5.5）", "agents/official_nav/resync/resync_results.json（除權息＋期限修正後，2026-11-17 重算）；方法同 agents/s26_exright_robustness（S7 W10_70L）"),
    ("晚2天回撤", -35.3, "%", "晚 2 天下單的最大回撤（100 萬單一路徑；除權息＋期限修正後，2026-11-17 重算；期限修正前 −34.1）", "agents/official_nav/resync/resync_results.json（除權息＋期限修正後，2026-11-17 重算）；方法同 agents/s26_exright_robustness（S7 W10_70L）"),
    ("準時可預期", 46.6, "%", "每月都準時開盤下單的可預期年報酬（2019 年起；除權息＋期限修正後，2026-11-17 重算；期限修正前 46.7）", "agents/official_nav/resync/resync_results.json（除權息＋期限修正後，2026-11-17 重算）；方法同 agents/s26_exright_robustness（S9 100 條迴歸截距）"),
    ("有時晚可預期", 43, "%", "有時晚 0～4 天下單的可預期年報酬（100 條路徑中位數 42.9；差的 10% 為 40.2；除權息＋期限修正後，2026-11-17 重算；期限修正前 43.2／40.4）", "agents/official_nav/resync/resync_results.json（除權息＋期限修正後，2026-11-17 重算）；方法同 agents/s26_exright_robustness（S9 A10～A15 × 20 條）"),
    ("警戒A_3年超額", 15.0, "個百分點", "現值（2026-10）：滾動 36 個月對 0050 年化超額。警戒線：低於 −8（策略有效時 5 年誤報約 8%）。除權息＋期限修正後，2026-11-17 重算（期限修正前 15.3；警戒線不變）", "agents/s27_expectation_ranges（已獨立複製）；agents/official_nav/resync/resync_results.json（除權息＋期限修正後，2026-11-17 重算）"),
    ("警戒E_相對前高", -7.1, "%", "現值（2026-10）：策略÷0050 對前高的落後。警戒線：深於 −24%（有效時 5 年誤報約 9%）。除權息＋期限修正後，2026-11-17 重算（期限修正前 −6.6、誤報約 10%；警戒線不變）", "agents/s27_expectation_ranges（已獨立複製）；agents/official_nav/resync/resync_results.json（除權息＋期限修正後，2026-11-17 重算）"),
    ("警戒B_1年超額", 33.5, "個百分點", "現值（2026-10）：滾動 12 個月對 0050 超額。警戒線：低於 −26；某 12 個月輸 0050 本身很常見（5 年內 97%）。除權息＋期限修正後，2026-11-17 重算（期限修正前 34.5；警戒線不變）", "agents/s27_expectation_ranges（已獨立複製）；agents/official_nav/resync/resync_results.json（除權息＋期限修正後，2026-11-17 重算）"),
    ("警戒C_月底回撤", -5.4, "%", "現值（2026-10，月底口徑；日內 −10.6）。警戒線：月底深於 −30%（約日內 −37%）。除權息＋期限修正後，2026-11-17 重算（期限修正前 −5.3；日內與警戒線不變）", "agents/s27_expectation_ranges（已獨立複製）；agents/official_nav/resync/resync_results.json（除權息＋期限修正後，2026-11-17 重算）"),
    ("紅燈規則", 2, "條", "同一個月 2 條以上警戒線觸發＝正式檢討（策略有效時 5 年誤報約 7%）；目前 0 條觸發", "agents/s27_expectation_ranges（已獨立複製）"),
    ("實盤可預期_中性", 41.8, "%", "2019 起每年：回測再扣首次公布營收、期限前視、有時晚 0～4 天、每邊滑價 0.4%（100 條中位數）；最大回撤約 −35%（壞時 −37.5%）。除權息＋期限修正後，2026-11-17 重算：數字不變（S42 原本就已含期限修正）", "agents/s42_realistic_expectation（依賴 S39、S40，皆已獨立複製）；agents/official_nav/resync/resync_results.json（除權息＋期限修正後，2026-11-17 重算）"),
    ("實盤可預期_保守", 36.1, "%", "中性再把對 0050 的超額打一半；最差 10% 加滑價 0.5% 約 34.0%、回撤約 −40%。網站上的 48.76% 是單一路徑＋理想執行，不要當預期（除權息＋期限修正後，2026-11-17 重算：數字不變；期限修正前寫 48.99%）", "agents/s42_realistic_expectation；agents/official_nav/resync/resync_results.json（除權息＋期限修正後，2026-11-17 重算）"),
    ("營收更正高估", -0.8, "個百分點", "回測用事後更正的月營收，實盤只看得到首次公布值：2024-07～2026-10 年化約低 0.8（20 條中位數 −0.93，最多約 −5），回撤不變；2024-06 以前無資料", "agents/s39_first_print_revenue（已獨立複製）"),
    ("期限前視高估", -0.24, "個百分點", "已併入正式數字（2026-11-17 起）：2019-01、2023-03 期限延後月份與 2026 起 15 日期限保險／金控股已在回測裡修正，上面的年化 48.76% 已扣掉這 0.24 個百分點（20 條中位數 −0.35），回撤不變；不要再另外扣", "agents/s40_revenue_deadline_pit（已獨立複製）；agents/deadline_engine_fix、agents/official_nav"),
    ("正常範圍_1年", 30, "%", "未來 1 年報酬中位數約 30%，常見範圍約 0～1% 到 99%（P10～P90）；輸 0050 機率約四分之一。除權息＋期限修正後，2026-11-17 重算：中位數 30.0%（期限修正前 30.5%），四捨五入不變", "agents/s27_expectation_ranges（已獨立複製）；agents/official_nav/resync/resync_results.json（除權息＋期限修正後，2026-11-17 重算）"),
    ("參數挑選後可預期", 40.4, "%", "進場回看 20 日在 45 格家族排第 1、高出家族中位數 6.25 個百分點（S46，R46 重現）；回測參考預期改用家族 900 條路徑中位數 40.4%，差的情況 35.9%。這是參數挑選的打折，和上面 S42 的執行打折不一定能直接相加", "agents/s46_momentum_gate_family（R46 已獨立重現）"),
    ("參數挑選_差的情況", 35.9, "%", "同上 S46 家族 900 條路徑的差的情況；和 S42 的執行打折不一定能直接相加", "agents/s46_momentum_gate_family（R46 已獨立重現）"),
    ("照做_只買一半少", 6.0, "個百分點", "名單上的股票每個月只隨機買一半，每年少賺約 6.0 個百分點（46.4% → 40.1%；獨立重現時換一組隨機抽法是 5.8）。名單要全買，不要自己挑。回測 2019 起、20 條路徑中位數；R64 已重現", "agents/s64_retail_behavior_tax（R64 已獨立重現：agents/r64_replicate_retail/compare.csv）"),
    ("照做_回落20%停手少", 4.8, "個百分點", "帳戶從高點回落 20% 就停手、改放 0050 等 6 個月再回來，每年少賺約 4.8 個百分點（46.4% → 41.5%），而且最大回撤沒有變淺（重現值還略深 0.4 個百分點）。大跌時停手換不到保護。回測 2019 起、20 條路徑中位數；R64 已重現", "agents/s64_retail_behavior_tax（R64 已獨立重現：agents/r64_replicate_retail/compare.csv）"),
    ("照做_漏做四分之一少", 2.1, "個百分點", "每四個月漏做一個月（隨機漏 25% 的月份），每年少賺約 2.1 個百分點（46.4% → 43.8%）；漏一成約少 0.7，漏一半約少 6.1。回測 2019 起、20 條路徑中位數；R64 已重現", "agents/s64_retail_behavior_tax（R64 已獨立重現：agents/r64_replicate_retail/compare.csv）"),
    ("照做_開盤漲3%不買少", 1.6, "個百分點", "開盤比前一天收盤漲超過 3% 就不買，每年少賺約 1.6 個百分點（46.4% → 45.6%）；門檻放到 5% 只少約 0.2。開盤漲多也照買。回測 2019 起、20 條路徑中位數；R64 已重現", "agents/s64_retail_behavior_tax（R64 已獨立重現：agents/r64_replicate_retail/compare.csv）"),
    ("小資金_10萬零股超額", 14.7, "個百分點", "本金 10 萬、用零股照做，扣掉散戶成本後每年仍比 0050 多約 14.7 個百分點（43.6% 對 0050 的 28.9%）。成本假設：手續費 6 折、零股低消 1 元、零股每邊多 0.3% 滑價（疊在原本的嚴成本上）；2020-10 以前用現行零股制度回推。回測 2019 起、20 條路徑中位數；R64 已重現", "agents/s62_small_capital_retail（R64 已獨立重現：agents/r64_replicate_retail/compare.csv）"),
    ("小資金_10萬只買整張超額", 0.5, "個百分點", "本金 10 萬但堅持只買整張（買不起就放棄），九成多的名單買不到，每年只比 0050 多約 0.5 個百分點（29.4% 對 28.9%），等於只買 0050。小資金一定要用零股。回測 2019 起、20 條路徑中位數；R64 已重現", "agents/s62_small_capital_retail（R64 已獨立重現：agents/r64_replicate_retail/compare.csv）"),
]
dump("risk", [{"項目": a, "數值": b, "單位": c, "說明": d, "出處": e} for a, b, c, d, e in RISK],
     [{"label": "說明", "value": "研究中心回測（嚴成本、只上市、排除創新板、衛星 70%＋每檔＝衛星 ÷ 10）；2026-11-17 起正式數字＝除權息＋期限修正後（標出處為第 12 節「2026-11-17 起正式數字」的各列已重算；原本只修除權息的 S26、S27、S42 各列，2026-11-17 已用正式引擎重算，出處為 agents/official_nav/resync/resync_results.json，舊數字寫在各列說明的「期限修正前」）；警戒線與正常範圍出自 S27 模擬，假設未來像過去 10 年；「照做_」「小資金_」各列出自 S64、S62（散戶照做須知，README 第 32 節），是 2018 年底新起跑的 20 條路徑中位數，R64 已獨立重現；回測不保證未來"}])
# 回測淨值（週）與逐年
nav = [{"日期": d, "現行策略": c, "XQ版": x, "0050": e} for d, c, x, e in zip(P["dates"], P["cur"], P["xq"], P["etf"])]
dump("nav", nav, [{"label": "說明", "value": "100 萬起始、嚴成本回測、除權息＋期限修正後（2026-11-17 起正式數字；除權息參考價修正＋營收期限例外表＋2026 起 15 日期限保險／金控股遮蔽）；現行＝研究中心名次版 W10_70；XQ版＝20＋21 號固定門檻、換股日同回測（A10），期限例外表有套用、15 日期限遮蔽不適用（XQ 用自己的營收資料時點）；0050 含息"},
                  {"label": "來源", "value": "現行、XQ版：xq_scripts/site/navs_official.csv.gz（research/tw_stock_selection/agents/official_nav/run_official.py；只修除權息的是 navs_exright.csv.gz，修正前是 s19_xq_threshold/navs.csv.gz）；0050：精確含息還原價（研究引擎同一來源，不用 navs 的整數欄）；由 make_profit.py 產生 profit.json"},
                  {"label": "日期", "value": "每週取最後一個交易日，日期就是那一天（不超過回測最後一天）"}])
yrs = sorted(P["years"]["cur"].keys())
dump("yearly", [{"年": int(y), "現行策略": P["years"]["cur"][y], "XQ版": P["years"]["xq"][y], "0050": P["years"]["etf"][y]} for y in yrs],
     [{"label": "說明", "value": "2015 為 7 月起；2026 到 10/22"},
      {"label": "0050", "value": "用精確含息還原價算（make_profit.py），不用 navs.csv.gz 的整數欄"}])
# 上線後每日淨值（每日列在前面的 live_rows_calc 算好；持股表合計列已和最後一天逐元核對）
if ACC:
    rows = ACC["live"]
    dump("live", rows, [{"label": "起始", "value": f"{live_start} 開盤＝100 萬（策略換股後持股，含 0050 底倉" + (f"與現金 {cash:,.0f} 元）" if cash_src else "）")},
                        {"label": "持股檔", "value": seg_names},
                        {"label": "現金", "value": (f"{cash:,.0f} 元（{os.path.basename(cash_src)}，起點與每日都計入）" if cash_src else "無現金檔，以 0 計")},
                        {"label": "價格", "value": "官方日 K 收盤；含息總報酬口徑（2026-11-18 起，live_tr.py）：策略帳戶與 0050 對照帳戶在起始日之後遇到持股除息日，現金股利（息值 × 股數）加進帳戶現金（策略帳戶不再投入；0050 對照帳戶見「0050 欄」）；"
                                                    "除權、權息、ETF 分割、面額變更、減資調整股數（＝前收 ÷ 參考價；權息合併、現金增資、退還股款無法拆分現金，以股數調整並標近似）。沒有事件時和未調整的算法逐元相同"},
                        {"label": "0050 欄", "value": "起始日開盤把 100 萬全部買 0050；0050 分割時股數跟著調整。之後領到的現金股利" + ("在除息日收盤買回 0050（BENCH_REINVEST＝True）" if BENCH_REINVEST else
                                                       "留在帳戶現金、不再買進 0050（BENCH_REINVEST＝False）。注意：這樣長期會比含息還原（股利再投入）的 0050 低（歷史試算 2015-07～2026-10 每年約少 2.8 個百分點），"
                                                       "策略相對 0050 會顯得比較好；0050 下次除息前要決定是否改成再投入"), "bench_reinvest": BENCH_REINVEST}]
         + ([{"label": "跨月銜接", "value": "帳戶由多份換股後持股檔接成，各段在生效日開盤換股：" + "；".join(
                f"{w['日期']}（{w['月份']}）換股前開盤總值 {round(w['換股前開盤總值'] * ACC['f']):,} 元 → 換股後 {round(w['換股後開盤總值'] * ACC['f']):,} 元，交易成本 {round(w['交易成本'] * ACC['f']):,} 元"
                f"（換股前現金含期間現金股利 {round(w['期間現金股利'] * ACC['f']):,} 元）" for w in LED["switches"])
              + "。換股日當天的淨值＝新持股收盤市值＋新現金；和前一天的差＝舊持股隔夜到開盤的漲跌＋交易成本＋新持股開盤到收盤的漲跌。換股日先把當天的除權息套到舊持股再換股。"
                "各月份用同一個換算倍數（100 萬 ÷ 第一段起始日開盤總值），不重新正規化。交易成本是研究中心嚴成本（半個跳動點＋每邊 0.3% 滑價＋手續費＋證交稅），在產生持股檔與現金檔時就扣掉了",
              "switches": [{"日期": w["日期"], "月份": w["月份"], "換股前": round(w["換股前開盤總值"] * ACC["f"]), "換股後": round(w["換股後開盤總值"] * ACC["f"])} for w in LED["switches"]]}] if MULTI else [])
         + ACC["xr"]["meta"] + [{"label": "live_tr.py", "value": ACC["xr"]["code"], "type": "code", "lang": "python"}, {"label": "ledger.py", "value": lg_code, "type": "code", "lang": "python"}])
    dump("trades", TROWS, [
        {"label": "說明", "value": "策略模擬帳戶的歷次買賣（100 萬帳戶口徑）：換股序 0＝帳戶起始持股；之後每次換股列出賣出、買進（含 0050）與一列「交易成本」。頁面還沒接這個資料集"},
        {"label": "成交價／金額", "value": "成交價＝執行日官方日 K 開盤價（模擬帳戶假設開盤成交，不是券商成交回報）；金額＝股數 × 換算倍數 × 成交價，四捨五入到元，未含費用。股數是持股檔口徑，帳戶股數＝股數 × 換算倍數"},
        {"label": "買進成本／已實現損益", "value": "賣出列：買進成本＝每股平均成本（移動平均，未含費用）；成本金額＝賣出股數 × 平均成本；已實現損益＝金額 − 成本金額；全部出清時「期間現金股利」＝這個部位持有期間領到的現金股利（不含在已實現損益）"},
        {"label": "交易成本", "value": "每次換股一列：金額＝換股前開盤總值 − 換股後開盤總值（嚴成本：半個跳動點＋每邊 0.3% 滑價＋手續費＋證交稅，已反映在現金檔），已實現損益欄記負數"},
        {"label": "恆等式", "value": "已實現損益欄加總＋期間現金股利欄加總＝持股表「已實現」列＝status 的累計已實現損益；再加目前持股損益（持股表個股與 0050 各列損益加總）＝累計損益＝live.json 最後一天 − 起始資金。程式逐元檢查，不一致就不寫檔"
                                     + (f"。目前：已實現 {REAL_i:+,} 元＋目前持股損益 {unreal_i:+,} 元＝累計損益 {pnl:+,} 元" if MULTI else "。目前只有帳戶起始，沒有換股"),
         "realized": (REAL_i if MULTI else 0), "unrealized": unreal_i, "total_pnl": pnl},
        {"label": "持股檔", "value": seg_names}, {"label": "換算倍數", "value": f"{ACC['f']:.6f}"},
        {"label": "ledger.py", "value": lg_code, "type": "code", "lang": "python"}])
# 0050 最新收盤寫進 status（給頁面下單計算用）
if last_px:
    S = OUT["status"]
    S["rows"][0]["0050最新收盤"] = last_px.get("0050", (None, None))[1]; S["rows"][0]["價格日期"] = last_day; S["rows"][0]["資料最後一天"] = L["data_last_day"]
    for m in S["meta"]:
        if m.get("label") == "資料最後一天": m["value"] = L["data_last_day"]
    if SUM:
        S["rows"][0].update(SUM)
        S["meta"][-1:-1] = [
            {"label": "帳戶", "value": f"策略模擬帳戶：{live_start} 開盤以 {START_CAPITAL:,} 元起算（不是任何人的實際帳戶）。目前總資產＝live.json 最後一天策略淨值＝持股表合計列；個股市值合計＋0050市值＋現金＝目前總資產"},
            {"label": "累計損益", "value": "累計損益金額＝目前總資產 − 起始資金；累計損益%＝累計損益金額 ÷ 起始資金。沒有扣手續費與證交稅；含息總報酬口徑（期間現金股利加進現金、除權與分割調整股數，見 live.json 的 meta）"},
            {"label": "同期只買 0050", "value": "同期只買0050總資產＝起始日開盤把 100 萬全部買 0050 到最新收盤，含期間 0050 現金股利（live.json 的 0050 欄）；領先0050_個百分點＝（目前總資產 − 同期只買0050總資產）÷ 起始資金 × 100，負數＝落後"},
            {"label": "下次換股", "value": "下次換股檢查日＝下次訊號日（收盤後檢查）；下次換股執行日＝下次執行日（開盤先賣後買）。下次檢查條件＝一句白話的規則"},
            {"label": "試算", "value": "試算會賣檔數、試算說明＝用試算基準日的股價與季線、本月名單同一份營收試算「若今天就是訊號日」；只供參考，真正只看下次訊號日收盤"},
            {"label": "帳戶換算倍數", "value": "＝起始資金 ÷ 持股檔起始日開盤總值（持股檔股數 × 起始日開盤價＋現金）；持股表與換股紀錄的金額都乘這個倍數"}] + (
            [{"label": "換股累計", "value": "兩段以上才有：換股次數、最近換股日、累計已實現損益（＝累計已實現價差＋已出清持股現金股利 − 累計交易成本＝持股表「已實現」列）、目前持股損益（個股與 0050 各列損益加總）、"
                                           "本月以來損益金額（目前總資產 − 最近一次換股後的開盤總值）。累計已實現損益＋目前持股損益＝累計損益金額"}] if MULTI else []) + (
            [{"label": "名單狀態", "value": "名單狀態＝預計：訊號日收盤後的名單還沒執行，持股與金額是換股前的；下次換股檢查日／執行日這時是名單自己的訊號日／執行日"}] if not SETTLED else []) + ACC["xr"]["meta"]
for name, doc in OUT.items():
    json.dump(doc, open(os.path.join(outdir, name + ".json"), "w", encoding="utf-8"), ensure_ascii=False)
print("ok", outdir, last_day)
