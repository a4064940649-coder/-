# -*- coding: utf-8 -*-
# 跨月銜接的帳本（2026-11-19 加）：多份「換股後持股檔」接成一個連續的策略模擬帳戶。
# build_site_data.py（live.json、holdings、actions、trades.json）與 monthly_rebalance.py（訊號日／執行日流程）共用。
#
# 帳戶口徑（理由見 state/README.md「跨月銜接」）：
#   - 帳本用「持股檔口徑」＝持股檔股數（整數股；除權後可能有小數）＋現金檔現金，是一個自己會平衡的帳戶；
#     每個月的持股檔由上一份持股檔＋期間現金股利、照執行日開盤價與研究中心嚴成本換股算出來（monthly_rebalance.py settle），
#     所以網站的「100 萬帳戶口徑」只要乘上同一個換算倍數（100 萬 ÷ 第一段起始日開盤總值），各月份不用重新正規化，淨值自然連續。
#   - 換股日：先把當天的除權息事件套到舊持股，再在開盤把舊持股換成新持股。
#     換股前開盤總值（舊持股 × 開盤＋舊現金，含期間領到的現金股利）− 換股後開盤總值（新持股 × 開盤＋新現金）＝這次換股的交易成本
#     （嚴成本：半個跳動點＋每邊 0.3% 滑價＋手續費＋證交稅，已經反映在現金檔裡；帳本不另外算費用，只量差額）。
#   - 成本：移動平均成本，用開盤成交價（未含費用）。賣出的已實現損益＝賣出股數 ×（開盤價 − 平均成本）；費用另列「交易成本」。
#     恆等式：總損益＝已實現價差＋已出清持股領過的現金股利 − 交易成本＋（目前持股市值 − 成本＋目前持股領過的現金股利）。
# 段落表（segments.csv）：月份,生效日,持股檔,現金檔,底倉股數,備註；路徑相對於段落表所在資料夾；生效日＝執行日（開盤換股）。
import csv, os, re, subprocess

EPS = 1e-9


def read_cash(p):
    """現金檔：第一個非註解行的數字（# 之後是註解）。找不到檔＝（0, None）。"""
    if not p or not os.path.exists(p): return 0.0, None
    for line in open(p, encoding="utf-8-sig"):
        line = line.split("#", 1)[0].strip().replace(",", "")
        if line: return float(line), p
    return 0.0, p


def read_holdings(p):
    """持股檔（代號,股數）。"""
    out = {}
    with open(p, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            c = (r.get("代號") or r.get("code") or "").strip()
            if not c: continue
            out[c] = out.get(c, 0.0) + float(str(r.get("股數") or r.get("shares")).replace(",", ""))
    return out


def is_manifest(p):
    """是不是段落表（表頭有「生效日」）。"""
    try:
        with open(p, encoding="utf-8-sig") as f: return "生效日" in (f.readline() or "")
    except OSError:
        return False


def read_segments(manifest):
    """讀段落表，依生效日排序。每段：月份、生效日、sh、cash、cash_src、base_shares、持股檔、現金檔。"""
    d = os.path.dirname(os.path.abspath(manifest)); segs = []
    with open(manifest, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if not (r.get("月份") or "").strip() or r["月份"].lstrip().startswith("#"): continue
            hp = os.path.join(d, r["持股檔"].strip()); cp = os.path.join(d, r["現金檔"].strip()) if (r.get("現金檔") or "").strip() else None
            if not os.path.exists(hp): raise SystemExit(f"段落表 {manifest} 的持股檔不存在：{hp}")
            if cp and not os.path.exists(cp): raise SystemExit(f"段落表 {manifest} 的現金檔不存在：{cp}")
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", r["生效日"].strip()): raise SystemExit(f"段落表的生效日格式不對：{r['生效日']}")
            cash, src = read_cash(cp)
            b = (r.get("底倉股數") or "").strip()
            segs.append({"月份": r["月份"].strip(), "生效日": r["生效日"].strip(), "sh": read_holdings(hp), "cash": cash, "cash_src": src,
                         "base_shares": float(b) if b else None, "持股檔": hp, "現金檔": cp})
    segs.sort(key=lambda s: s["生效日"])
    ds = [s["生效日"] for s in segs]
    if len(set(ds)) != len(ds): raise SystemExit("段落表有重複的生效日：" + "、".join(ds))
    return segs


def segments_from_arg(state_arg, live_start=None):
    """build_site_data.py 的第 10、11 個參數 → 段落。第 10 個參數是段落表時用段落表；否則是舊用法（一份持股檔＋上線起始日）。"""
    if is_manifest(state_arg):
        segs = read_segments(state_arg)
        if not segs: raise SystemExit(f"段落表 {state_arg} 沒有任何一段")
        if live_start and live_start not in ("-", segs[0]["生效日"]): raise SystemExit(f"上線起始日 {live_start} 和段落表第一段的生效日 {segs[0]['生效日']} 不同")
        return segs
    if not live_start: raise SystemExit("只給一份持股檔時要給上線起始日（第 11 個參數）")
    cp = re.sub(r"holdings_after_([^/]+)\.csv$", r"cash_after_\1.txt", state_arg)
    cash, src = read_cash(cp if cp != state_arg else None)
    m = re.search(r"holdings_after_([^/]+)\.csv$", state_arg)
    return [{"月份": m.group(1) if m else live_start[:7], "生效日": live_start, "sh": read_holdings(state_arg), "cash": cash, "cash_src": src,
             "base_shares": None, "持股檔": state_arg, "現金檔": src}]


def write_segments(manifest, segs_rows):
    """寫段落表（rows：月份、生效日、持股檔、現金檔、底倉股數、備註；檔名相對於段落表）。"""
    with open(manifest, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["月份", "生效日", "持股檔", "現金檔", "底倉股數", "備註"]); w.writeheader()
        for r in sorted(segs_rows, key=lambda r: r["生效日"]): w.writerow({k: r.get(k, "") for k in w.fieldnames})


def manifest_rows(manifest):
    if not os.path.exists(manifest): return []
    with open(manifest, encoding="utf-8-sig") as f: return [r for r in csv.DictReader(f) if (r.get("月份") or "").strip()]


# ---------------------------------------------------------------- 官方日 K（git HEAD）
def _git(tsd, *a):
    return subprocess.run(["git", "-C", tsd] + list(a), capture_output=True, text=True, check=True).stdout


def trade_days(tsd):
    return sorted(x.rsplit("/", 1)[1][:-4] for x in _git(tsd, "ls-tree", "--name-only", "HEAD", "data/universe/daily/").split())


def day_table(tsd, date):
    """某天的日 K：{代號: {open, close, name}}；讀不到＝None。"""
    try:
        txt = _git(tsd, "show", f"HEAD:data/universe/daily/{date}.csv")
    except Exception:
        return None
    out = {}
    for r in csv.DictReader(txt.splitlines()):
        try: out[r["stock_id"]] = {"open": float(r["open"]) if r["open"] else None, "close": float(r["close"]) if r["close"] else None, "name": (r.get("name") or "").strip()}
        except ValueError: pass
    return out


def make_px_of(tsd):
    cache = {}
    def px_of(d):
        if d not in cache:
            t = day_table(tsd, d); cache[d] = None if t is None else {k: (v["open"], v["close"]) for k, v in t.items()}
        return cache[d]
    return px_of


def held_before(segs, d):
    """d 那天開盤前（前一個收盤）持有的代號＝最後一個生效日 < d 的那一段。"""
    cur = set()
    for s in segs:
        if s["生效日"] < d: cur = set(s["sh"])
    return cur


def simulate_account(LT, segs, days, px_of, events):
    """多段持股檔接成一個帳戶（live_tr.simulate 的 switches）。days 要從第一段生效日開始。"""
    sw = {s["生效日"]: {"sh": s["sh"], "cash": s["cash"]} for s in segs[1:]}
    return LT.simulate(segs[0]["sh"], segs[0]["cash"], days, px_of, events, segs[0]["生效日"], switches=sw)


def account_state(LT, segs, tsd, upto, twt49u_dir=None):
    """帳戶到 upto 那天（含）的狀態：除權息事件已套用、段落表裡生效日 ≤ upto 的換股都已執行。
    回傳（模擬結果, 帳本, 日期清單, px_of）。monthly_rebalance.py 用它拿「目前持股、目前現金（含期間現金股利）」。"""
    days = [d for d in trade_days(tsd) if segs[0]["生效日"] <= d <= upto]
    if not days or days[0] != segs[0]["生效日"]: raise SystemExit(f"官方日 K 沒有第一段生效日 {segs[0]['生效日']}")
    use = [s for s in segs if s["生效日"] <= upto]
    codes = set().union(*[set(s["sh"]) for s in use])
    px_of = make_px_of(tsd)
    ev, cov = LT.load_events(tsd, segs[0]["生效日"], days[-1], codes, twt49u_dir)
    A = simulate_account(LT, use, days, px_of, ev)
    A["cov"] = cov
    return A, build_ledger(use, A, px_of(use[0]["生效日"])), days, px_of


def build_ledger(segs, A, px_start):
    """由模擬結果（switch_log）做帳本：目前部位的成本、每筆買賣、每次換股的交易成本。金額都是持股檔口徑（還沒乘換算倍數）。
    回傳 dict：pos（目前部位）、trades（逐筆，含起始）、switches（每次換股摘要）、real_gross、closed_div、costs、seg_open。"""
    pos = {}; trades = []; sws = []
    s0 = segs[0]
    for k in s0["sh"]:
        n = s0["sh"][k]; o = (px_start.get(k) or (None, None))[0]
        if not n: continue
        if not o: raise SystemExit(f"持股 {k} 缺起始日 {s0['生效日']} 開盤價，無法算成本")
        pos[k] = {"sh": n, "sh0": n, "px0": o, "cb": n * o, "since": s0["生效日"], "div0": 0.0, "lots": 1}
        trades.append({"換股序": 0, "月份": s0["月份"], "成交日": s0["生效日"], "動作": "起始", "代號": k, "股數": n, "成交價": o, "金額": n * o})
    seg_open = {k: p["px0"] for k, p in pos.items()}; prev_div = {}
    for i, sw in enumerate(A.get("switch_log") or [], 1):
        d = sw["日期"]; b = sw["sh_before"]; a = sw["sh_after"]; opn = sw["open"]; dv = sw["div_before"]; seg = segs[i]
        if seg["生效日"] != d: raise SystemExit(f"換股紀錄 {d} 和段落表第 {i + 1} 段的生效日 {seg['生效日']} 不同")
        gross = 0.0; cdiv = 0.0; sells = buys = 0.0
        for k in sorted(set(b) | set(a)):
            n0 = b.get(k, 0.0); n1 = a.get(k, 0.0); o = opn[k]
            if k in pos and abs(pos[k]["sh"] - n0) > EPS: pos[k]["sh"] = n0          # 期間除權／分割調整過股數（成本不變）
            if n1 < n0 - EPS:
                p = pos[k]; sold = n0 - n1; cost = p["cb"] * sold / n0; amt = sold * o; allout = n1 <= EPS
                dk = (dv.get(k, 0.0) - p["div0"]) if allout else 0.0
                trades.append({"換股序": i, "月份": seg["月份"], "成交日": d, "動作": "賣出", "代號": k, "股數": sold, "成交價": o, "金額": amt,
                               "平均成本": cost / sold, "成本金額": cost, "已實現損益": amt - cost, "持有起日": p["since"], "全部出清": allout,
                               "期間現金股利": dk, "沒有開盤價": k in sw["noopen"]})
                gross += amt - cost; cdiv += dk; sells += amt
                if allout: del pos[k]
                else: p["cb"] -= cost; p["sh"] = n1; p["sh0"] = n1; p["px0"] = p["cb"] / n1; p["lots"] += 1
            elif n1 > n0 + EPS:
                bought = n1 - n0; amt = bought * o; buys += amt
                trades.append({"換股序": i, "月份": seg["月份"], "成交日": d, "動作": "買進", "代號": k, "股數": bought, "成交價": o, "金額": amt,
                               "加碼": k in pos, "沒有開盤價": k in sw["noopen"]})
                if k in pos:
                    p = pos[k]; p["cb"] += amt; p["sh"] = n1; p["sh0"] = n1; p["px0"] = p["cb"] / n1; p["lots"] += 1
                else:
                    pos[k] = {"sh": n1, "sh0": n1, "px0": o, "cb": amt, "since": d, "div0": dv.get(k, 0.0), "lots": 1}
        seg_open = {k: opn[k] for k in a}
        sws.append({"換股序": i, "月份": seg["月份"], "日期": d, "換股前開盤總值": sw["value_before"], "換股後開盤總值": sw["value_after"],
                    "交易成本": sw["value_before"] - sw["value_after"], "賣出金額": sells, "買進金額": buys, "已實現價差": gross, "出清持股現金股利": cdiv,
                    "換股前現金": sw["cash_before"], "換股後現金": sw["cash_after"], "期間現金股利": sum(dv.values()) - sum(prev_div.values()),
                    "沒有開盤價": list(sw["noopen"]), "sh_before": dict(b), "sh_after": dict(a), "open": dict(opn)})
        prev_div = dict(dv)
    for k, p in pos.items():
        if k not in A["sh"]: raise SystemExit(f"帳本部位 {k} 不在模擬結果的持股裡")
        p["sh_now"] = A["sh"][k]; p["div"] = A["div"].get(k, 0.0) - p["div0"]
    if sorted(pos) != sorted(k for k, v in A["sh"].items() if v): raise SystemExit("帳本部位和模擬結果的持股不同")
    return {"pos": pos, "trades": trades, "switches": sws, "real_gross": sum(s["已實現價差"] for s in sws),
            "closed_div": sum(s["出清持股現金股利"] for s in sws), "costs": sum(s["交易成本"] for s in sws), "seg_open": seg_open}


def check_identity(A, LED, last_close):
    """總損益＝已實現價差＋出清持股股利 − 交易成本＋未實現（含目前持股已領股利）。回傳（總損益, 已實現合計, 未實現, 殘差），都是持股檔口徑。"""
    unreal = sum(p["sh_now"] * last_close[k] - p["cb"] + p["div"] for k, p in LED["pos"].items())
    real = LED["real_gross"] + LED["closed_div"] - LED["costs"]
    tot = A["nav"] - A["base"]
    return tot, real, unreal, tot - real - unreal
