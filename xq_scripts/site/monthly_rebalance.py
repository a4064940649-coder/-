#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 每月換股流程（策略模擬帳戶；只產生檔案，不連券商、不下單）。2026-11-19 加。規則見 state/README.md「跨月銜接」「每月換股流程」。
#
# 訊號日收盤後（建議 22 點後、官方日 K 倉庫已更新到訊號日）：
#   python3 -I -B xq_scripts/site/monthly_rebalance.py plan --month 2026-12 --tsd <官方日K倉庫> [--rev-cutoff 2026-12-11T14:00:00Z]
#   1. 跑名單產生器 make_list.py 兩次：空倉 100 萬版（<名單資料夾>/<月>.csv/.json/.md）與策略帳戶版（<月>_strategy.*）。
#      策略帳戶版的 --holdings＝帳戶到訊號日的持股（上一份持股檔，期間除權／分割已調整）、--cash＝上一份現金檔＋期間領到的現金股利、
#      --base-shares＝段落表的底倉股數（0050 分割時跟著調整）。
#   2. 產生「預計下單清單」<狀態資料夾>/orders/orders_<月>_strategy.csv、orders_<月>_new1m.csv 與 orders_<月>.md：
#      先賣後買；每檔股數（整張＋零股）、以訊號日收盤估的約略金額、掛單價（買＝漲停價下一檔、賣＝跌停價上一檔，ROD 限價，08:30～09:00 送出）、
#      開盤沒成交立刻刪單。
# 執行日收盤後（官方日 K 倉庫已更新到執行日）：
#   python3 -I -B xq_scripts/site/monthly_rebalance.py settle --month 2026-12 --tsd <官方日K倉庫> [--rev-cutoff 同上]
#   用執行日實際開盤價重算一次 make_list.plan()（＝回測口徑：開盤成交、研究中心嚴成本、開盤漲停不買、開盤跌停或沒有開盤價不賣），
#   寫 holdings_after_<月>.csv、cash_after_<月>.txt、rebalance_<月>.json（結算明細），並在段落表 segments.csv 加一列。
#   之後跑 build_site_data.py（第 10 個參數給段落表）就會接成連續淨值。
# 第一次（沒有段落表，從現金開始）：plan／settle 加 --init-total 1000000。
# 需要 numpy／pandas（make_list.py）。make_list.py 的月營收、日 K 封存、TWT49U 封存路徑用它自己的預設。
import argparse, csv, importlib.util, json, math, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ML_DEFAULT = os.path.join(HERE, "..", "..", "research", "tw_stock_selection", "agents", "s18_live_list", "live_surge", "make_list.py")
LISTS_DEFAULT = os.path.join(HERE, "..", "..", "research", "tw_stock_selection", "agents", "s18_live_list", "lists")
STATE_DEFAULT = os.path.join(HERE, "state", "segments.csv")
TWT49U_DEFAULT = os.path.join(HERE, "..", "..", "research", "tw_stock_selection", "raw", "twt49u", "twse-exright")


def load(name, path=None):
    p = path or os.path.join(HERE, name + ".py")
    sp = importlib.util.spec_from_file_location(name, p); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return m


def tick_of(p, etf):
    if etf: return 0.01 if p < 50 else 0.05
    return 0.01 if p < 10 else 0.05 if p < 50 else 0.1 if p < 100 else 0.5 if p < 500 else 1.0 if p < 1000 else 5.0


def limit_prices(ref, etf=False):
    """參考價 → （漲停價, 跌停價, 買進掛單價＝漲停價下一檔, 賣出掛單價＝跌停價上一檔）。漲跌幅 10%，漲停往下、跌停往上取到跳動點。"""
    t = tick_of(ref * 1.10, etf); lu = round(math.floor(ref * 1.10 / t + 1e-9) * t, 2)
    t = tick_of(ref * 0.90, etf); ld = round(math.ceil(ref * 0.90 / t - 1e-9) * t, 2)
    return lu, ld, round(lu - tick_of(lu - 1e-9, etf), 2), round(ld + tick_of(ld, etf), 2)


def fmt_sh(n):
    return int(n) if float(n).is_integer() else round(float(n), 4)


def run_make_list(a, extra, prefix):
    cmd = [sys.executable, "-I", "-B", a.make_list, "list", "--month", a.month, "--out-prefix", prefix, "--quiet", "--twt49u-dir", a.twt49u_dir]
    for r in a.repos: cmd += ["--repo", r]
    if a.rev_cutoff: cmd += ["--rev-cutoff", a.rev_cutoff]
    r = subprocess.run(cmd + extra, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(prefix + ".json"):
        raise SystemExit("名單產生器沒有產生名單（資料不足、無訊號或出錯），流程停止：\n" + (r.stdout or "")[-3000:] + (r.stderr or "")[-3000:])
    J = json.load(open(prefix + ".json", encoding="utf-8"))
    if J.get("status") != "有訊號": raise SystemExit(f"名單狀態是「{J.get('status')}」：{J.get('reason')}")
    return J


def current_state(a, LG, LT, upto):
    """帳戶到 upto（含）的狀態。回傳（段落, 模擬結果, 帳本, 目前持股, 目前現金, 底倉股數, 換算倍數）；沒有段落表＝None。"""
    if not os.path.exists(a.state):
        if a.init_total is None: raise SystemExit(f"找不到段落表 {a.state}；第一次從現金開始請加 --init-total 1000000")
        return None
    segs = [s for s in LG.read_segments(a.state) if s["月份"] < a.month]
    if not segs:
        if a.init_total is None: raise SystemExit(f"段落表 {a.state} 沒有 {a.month} 之前的段落；第一次從現金開始請加 --init-total")
        return None
    if a.init_total is not None: raise SystemExit("段落表已經有之前的持股，不能再給 --init-total")
    A, LED, days, _ = LG.account_state(LT, segs, a.tsd, upto, a.twt49u_dir)
    if days[-1] != upto: raise SystemExit(f"官方日 K 沒有 {upto}（最後一天 {days[-1]}）")
    sh = {k: v for k, v in A["sh"].items() if v}
    base = segs[-1]["base_shares"]
    if base is None: raise SystemExit(f"段落表 {segs[-1]['月份']} 那一列沒有底倉股數，請補上（見 state/README.md）")
    f0 = segs[-1]["sh"].get("0050")
    if f0 and abs(sh.get("0050", 0.0) - f0) > 1e-9: base = base * sh.get("0050", 0.0) / f0          # 0050 分割：底倉股數跟著調整
    return {"segs": segs, "A": A, "LED": LED, "sh": sh, "cash": A["cash"], "base": base, "f": 1e6 / A["base"],
            "div_period": A["cash"] - segs[-1]["cash"], "last": segs[-1]}


def order_rows(J, version, f=None):
    """名單 json 的 plan → 預計下單清單（先賣後買）。"""
    pl = J["plan"]; rows = []; exd = J.get("exec_date") or "訊號日的下一個交易日"
    def add(side, code, name, sh, ref, etf, why):
        lu, ld, bpx, spx = limit_prices(ref, etf)
        sh_i = int(math.floor(sh + 1e-9)); frac = sh - sh_i
        r = {"版本": version, "順序": len(rows) + 1, "方向": side, "代號": code, "名稱": name, "股數": fmt_sh(sh), "整張": sh_i // 1000, "零股": sh_i % 1000,
             "訊號日收盤": ref, "約略金額": round(sh * ref), "漲停價": lu, "跌停價": ld, "掛單價": bpx if side == "買" else spx,
             "掛單說明": ("漲停價下一檔" if side == "買" else "跌停價上一檔") + "，整股 ROD 限價，08:30～09:00 送出參加開盤集合競價",
             "沒成交處理": ("09:00 開盤沒成交（開盤漲停）立刻刪單，當天放棄、不追不補" if side == "買" else "09:00 開盤沒成交（開盤跌停）立刻刪單，隔天開盤再掛"),
             "預計成交": f"{exd} 開盤", "原因": why,
             "備註": "；".join(x for x in [("零股不參加整股開盤集合競價，另外用零股委託（盤中零股 09:00 起撮合）" if sh_i % 1000 else ""),
                                          (f"另有不足 1 股的 {frac:.4f} 股（除權產生的畸零股，實際帳戶會由公司折現金）" if frac > 1e-9 else ""),
                                          "漲跌停價以訊號日收盤當參考價估算；執行日若除權息，以交易所公告的參考價為準，下單前在券商軟體確認"] if x)}
        if f: r["帳戶股數"] = round(sh * f); r["帳戶約略金額"] = round(sh * f * ref)
        rows.append(r)
    for st in pl["sells"]:
        if st.get("close") and st["shares"] > 0: add("賣", st["code"], st.get("name", ""), st["shares"], st["close"], False, st["reason"])
    d50 = pl["e50"]["shares_after"] - pl["e50"]["shares_before"]; p50 = pl["e50"]["price"]
    if d50 <= -1: add("賣", "0050", "元大台灣50", -d50, p50, True, pl["e50"]["action"])
    for b in pl["buys"]:
        if b["action"] == "買進": add("買", b["code"], b["name"], b["shares"], b["close"], False, f"L_MA 第 {b['rank_ma']} 名，目標 {b['target']:,.0f} 元" + ("；" + b["warn"] if b.get("warn") else ""))
    if d50 >= 1: add("買", "0050", "元大台灣50", d50, p50, True, pl["e50"]["action"])
    return rows


def cmd_plan(a):
    LG = load("ledger"); LT = load("live_tr")
    os.makedirs(a.lists_dir, exist_ok=True); os.makedirs(a.orders_dir, exist_ok=True)
    # 1) 空倉 100 萬版（順便拿到訊號日）
    J1 = run_make_list(a, ["--total", "1000000"], os.path.join(a.lists_dir, a.month))
    sig = J1["sig_date"]
    st = current_state(a, LG, LT, sig)
    # 2) 策略帳戶版
    pre = os.path.join(a.lists_dir, a.month + "_strategy")
    if st is None:
        J2 = run_make_list(a, ["--total", f"{a.init_total:.2f}"], pre); f = 1e6 / a.init_total; hb = None
        acct = f"帳戶還沒有持股：以 --init-total {a.init_total:,.0f} 元從現金開始"
    else:
        hb = os.path.join(a.orders_dir, f"holdings_before_{a.month}.csv")
        with open(hb, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh); w.writerow(["代號", "股數"])
            for k, v in st["sh"].items(): w.writerow([k, fmt_sh(v)])
        J2 = run_make_list(a, ["--holdings", hb, "--cash", repr(st["cash"]), "--base-shares", repr(st["base"])], pre); f = st["f"]
        acct = (f"帳戶到 {sig} 收盤（持股檔口徑）：持股 {'、'.join(f'{k} {fmt_sh(v)} 股' for k, v in st['sh'].items())}；現金 {st['cash']:,.2f} 元"
                f"＝{os.path.basename(st['last']['現金檔'] or '（無現金檔）')} 的 {st['last']['cash']:,.2f} 元＋{st['last']['生效日']} 之後領到的現金股利 {st['div_period']:,.2f} 元；底倉 {st['base']:,.0f} 股；"
                f"換算倍數 {f:.6f}（× 之後＝100 萬帳戶口徑）")
    if J2["sig_date"] != sig or [t["code"] for t in J2["top20"]] != [t["code"] for t in J1["top20"]]: raise SystemExit("兩個版本的名單訊號日或前 20 名不同，請檢查")
    r2 = order_rows(J2, "策略帳戶", f); r1 = order_rows(J1, "空倉100萬")
    for rows, nm in ((r2, "strategy"), (r1, "new1m")):
        cols = list(rows[0].keys()) if rows else ["版本", "順序", "方向", "代號", "名稱", "股數"]
        with open(os.path.join(a.orders_dir, f"orders_{a.month}_{nm}.csv"), "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
    exd = J2.get("exec_date") or "訊號日的下一個交易日"
    M = [f"# {a.month} 預計下單清單（訊號日 {sig} 收盤後產生；預計 {exd} 開盤執行）", "",
         "- 只是清單，程式不下單。先賣後買；整股 ROD 限價，08:30～09:00 送出參加 09:00 開盤集合競價（單一價格成交，掛寬不會成交更差）。",
         "- 掛單價：買進掛「漲停價下一檔」、賣出掛「跌停價上一檔」。09:00 開盤沒成交就**立刻刪單**：買不到當天放棄、不追不補；賣不掉隔天開盤再掛。",
         "- 股數與約略金額用訊號日收盤估；模擬帳戶在執行日收盤後會照實際開盤價重算（`monthly_rebalance.py settle`），股數會和這裡略有不同。",
         "- 漲跌停價以訊號日收盤當參考價估算；執行日若除權息，以交易所公告的參考價為準。零股不參加整股開盤集合競價，另外用零股委託。", f"- {acct}", ""]
    for title, rows, J in ((f"策略帳戶版（持股檔口徑；帳戶股數＝× {f:.6f}）", r2, J2), ("空倉 100 萬版（從現金開始照名單買）", r1, J1)):
        M += [f"## {title}", "", "| 順序 | 方向 | 代號 | 名稱 | 股數 | 整張＋零股 | 訊號日收盤 | 約略金額 | 掛單價 | 漲停價 | 跌停價 |" + (" 帳戶股數 | 帳戶約略金額 |" if "帳戶股數" in (rows[0] if rows else {}) else ""),
              "|---|---|---|---|---|---|---|---|---|---|---|" + ("---|---|" if "帳戶股數" in (rows[0] if rows else {}) else "")]
        for r in rows:
            M.append(f"| {r['順序']} | {r['方向']} | {r['代號']} | {r['名稱']} | {r['股數']:,} | {r['整張']} 張 {r['零股']} 股 | {r['訊號日收盤']:g} | {r['約略金額']:,} | {r['掛單價']:g} | {r['漲停價']:g} | {r['跌停價']:g} |"
                     + (f" {r['帳戶股數']:,} | {r['帳戶約略金額']:,} |" if "帳戶股數" in r else ""))
        pl = J["plan"]
        M += ["", f"- 續抱（不用動）：{'、'.join(s['code'] + ' ' + s.get('name', '') for s in pl['keeps']) or '無'}",
              f"- 名單上沒買的：{'、'.join(b['code'] + '（' + b['action'] + '）' for b in pl['buys'] if b['action'] != '買進') or '無'}",
              f"- 0050：{pl['e50']['action']}；換股後共 {pl['e50']['shares_after']:,.0f} 股，其中底倉 {pl['band']['base_shares_new']:,.0f} 股；估計剩餘現金 {pl['cash_left']:,.0f} 元"]
        M += [f"- 注意：{n}" for n in pl["notes"]] + [""]
    open(os.path.join(a.orders_dir, f"orders_{a.month}.md"), "w", encoding="utf-8").write("\n".join(M) + "\n")
    print("\n".join(M))
    print(f"已寫：{os.path.join(a.lists_dir, a.month)}.*、{pre}.*、{os.path.join(a.orders_dir, 'orders_' + a.month)}_strategy.csv／_new1m.csv／.md")


def cmd_settle(a):
    LG = load("ledger"); LT = load("live_tr"); ml = load("make_list", a.make_list)
    import numpy as np
    sdir = os.path.dirname(os.path.abspath(a.state)); os.makedirs(sdir, exist_ok=True)
    hp = os.path.join(sdir, f"holdings_after_{a.month}.csv"); cp = os.path.join(sdir, f"cash_after_{a.month}.txt"); jp = os.path.join(sdir, f"rebalance_{a.month}.json")
    if any(r["月份"] == a.month for r in LG.manifest_rows(a.state)) or os.path.exists(hp) or os.path.exists(cp):
        if not a.force: raise SystemExit(f"{a.month} 已經結算過（段落表或 {hp}／{cp} 已存在）；確定要重算請加 --force")
    # 訊號日晚上的名單（策略帳戶版）：沿用它的營收截止時點，並核對名單沒有變
    pre = os.path.join(a.lists_dir, a.month + "_strategy.json"); J0 = json.load(open(pre, encoding="utf-8")) if os.path.exists(pre) else None
    cut = a.rev_cutoff
    if not cut and J0:
        cut = ((((J0.get("meta") or {}).get("rev") or {}).get("supplement") or {}).get(J0["rev_month"]) or {}).get("cutoff")
        if cut and not str(cut)[:4].isdigit(): cut = None
    D = ml.Data(a.repos, rev_cutoff=cut)
    codes_prev = []
    if os.path.exists(a.state): codes_prev = sorted(set().union(*[set(s["sh"]) for s in LG.read_segments(a.state) if s["月份"] < a.month] or [set()]) - {"0050"})
    res = ml.compute(D, a.month, exright=True, twt49u_dir=a.twt49u_dir, extra_codes=codes_prev, deadline=True, reduce=True)   # reduce：2026-11-21 起，同正式回測 REDUCE_FIX
    if res.get("status") != "有訊號": raise SystemExit(f"名單狀態是「{res.get('status')}」：{res.get('reason')}")
    if not res.get("exec_date") or res.get("_ex") is None: raise SystemExit(f"官方日 K 還沒有執行日（訊號日 {res['sig_date']} 的下一個交易日）的資料，不能結算")
    sig, exd, ex, ci = res["sig_date"], res["exec_date"], res["_ex"], res["_ci"]
    warn = []
    if J0:
        if J0["sig_date"] != sig: raise SystemExit(f"訊號日晚上的名單訊號日 {J0['sig_date']} 和現在重算的 {sig} 不同")
        if [t["code"] for t in J0["top20"]] != [t["code"] for t in res["top20"]] or [t["code"] for t in J0["f1"]] != [t["code"] for t in res["f1"]]:
            msg = "現在重算的前 20 名或買進名單和訊號日晚上的名單不同（營收補充資料變了？請用同一個 --rev-cutoff）"
            if not a.force: raise SystemExit(msg + "；確定要用現在的名單請加 --force")
            warn.append(msg)
    else:
        warn.append(f"找不到訊號日晚上的名單 {pre}，直接用現在重算的名單")
    st = current_state(a, LG, LT, exd)             # 到執行日為止：執行日當天的除權息已套到舊持股（開盤價已是除權息後）
    opn = lambda c: (lambda o: o if (np.isfinite(o) and o > 0) else None)(float(ex["open"][ci[c]]) if c in ci else float("nan"))
    lim = lambda c, k: float(ex[k][ci[c]])
    # 買不進：開盤漲停或沒有開盤價（不買、不遞補）
    f1 = []; nobuy = []
    for r in res["f1"]:
        o = opn(r["code"])
        if o is None or o >= lim(r["code"], "limup") - 1e-9: nobuy.append({"code": r["code"], "name": r["name"], "open": o, "limup": lim(r["code"], "limup"), "why": "沒有開盤價" if o is None else "開盤漲停"})
        else: f1.append(dict(r, close=o))
    # 賣不掉：名單判定賣出但開盤跌停或沒有開盤價（順延，仍持有）
    stuck = []; stuck_out = {}
    hold = {} if st is None else {k: v for k, v in st["sh"].items()}
    for c in [c for c in hold if c != "0050"]:
        s0, _ = ml.held_status(res, c)
        if s0["action"] != "賣出": continue
        o = opn(c)
        if c not in ci: stuck_out[c] = hold.pop(c); stuck.append({"code": c, "why": "訊號日不在官方日 K（停止交易或下市），無法賣出，仍記在持股（需人工處理）"})
        elif o is None or o <= lim(c, "limdn") + 1e-9: stuck.append({"code": c, "open": o, "limdn": lim(c, "limdn"), "why": ("沒有開盤價" if o is None else "開盤跌停") + "，沒有賣出、仍持有（隔天開盤再掛，需人工追蹤）"})
    row0 = res["_row"]
    def row2(j):
        r = row0(j); o = opn(r["code"])
        return dict(r, close=o) if o is not None else r                      # 沒有開盤價的用訊號日收盤估值
    res["_row"] = row2
    res["top20"] = [row2(ci[r["code"]]) for r in res["top20"]] + [row2(ci[s["code"]]) for s in stuck if s["code"] in ci]   # 賣不掉的當作續抱（佔一個名額）
    res["f1"] = f1
    e50o = float(ex["e50_open"])
    if not (np.isfinite(e50o) and e50o > 0): raise SystemExit("執行日 0050 沒有開盤價，不能結算")
    res["e50_close"] = e50o
    if st is None: pl = ml.plan(res, None, total=a.init_total)
    else: pl = ml.plan(res, hold, cash=st["cash"], base_shares=st["base"])
    new = {s["code"]: s["shares"] for s in pl["keeps"]}
    for b in pl["buys"]:
        if b["action"] == "買進": new[b["code"]] = new.get(b["code"], 0) + b["shares"]
    new.update(stuck_out)
    e_after = pl["e50"]["shares_after"]
    if e_after < -1e-9: raise SystemExit(f"照開盤價重算後 0050 要賣 {pl['e50']['shares_before'] - e_after:,.0f} 股，超過持有的 {pl['e50']['shares_before']:,.0f} 股；請人工檢查")
    if e_after > 0: new["0050"] = e_after
    cash_new = pl["cash_left"]; base_new = pl["band"]["base_shares_new"]
    if cash_new < -1e-6: raise SystemExit(f"結算後現金是負的（{cash_new:.2f}），請人工檢查")
    if pl["band"].get("trim_value"): warn.append(f"衛星占比超過 80% 而且現金＋0050 不足：規則要依市值等比例減碼續抱股約 {pl['band']['trim_value']:,.0f} 元，本程式沒有自動做，請人工處理後改持股檔")
    warn += list(pl["notes"])
    # 寫檔
    with open(hp, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh); w.writerow(["代號", "股數"])
        for k, v in new.items(): w.writerow([k, fmt_sh(v)])
    div_p = 0.0 if st is None else st["div_period"]
    with open(cp, "w", encoding="utf-8") as fh:
        fh.write(f"# {a.month} 換股（{exd} 開盤）後的策略現金（元，持股檔口徑）。build_site_data.py 會計入 live.json 淨值。\n"
                 f"# 由 monthly_rebalance.py settle 產生：照 {exd} 開盤價、研究中心嚴成本重算 make_list.plan()；結算明細見 rebalance_{a.month}.json。\n"
                 + (f"# 換股前可用現金 {st['cash']:.2f}＝上一份現金檔 {st['last']['cash']:.2f}＋{st['last']['生效日']} 之後領到的現金股利 {div_p:.2f}。\n" if st else f"# 帳戶從現金 {a.init_total:.2f} 元開始。\n")
                 + f"{round(cash_new, 2)}\n")
    rows = [r for r in LG.manifest_rows(a.state) if r["月份"] != a.month]
    rows.append({"月份": a.month, "生效日": exd, "持股檔": os.path.basename(hp), "現金檔": os.path.basename(cp), "底倉股數": repr(round(base_new, 6)),
                 "備註": f"訊號日 {sig}；monthly_rebalance.py settle"})
    LG.write_segments(a.state, rows)
    # 用和網站同一段程式核對：換股前後開盤總值只差交易成本
    segs = [s for s in LG.read_segments(a.state) if s["月份"] <= a.month]
    A, LED, _, px_of = LG.account_state(LT, segs, a.tsd, exd, a.twt49u_dir)
    if st is None:
        before = a.init_total; after = A["base"]; turn = after - cash_new
    else:
        w = LED["switches"][-1]; before, after = w["換股前開盤總值"], w["換股後開盤總值"]; turn = w["賣出金額"] + w["買進金額"]
    cost = before - after
    out = {"月份": a.month, "訊號日": sig, "執行日": exd, "營收截止": cut, "換股前": None if st is None else {"持股": st["sh"], "現金": st["cash"], "其中期間現金股利": div_p, "底倉股數": st["base"]},
           "換股後": {"持股": new, "現金": cash_new, "底倉股數": base_new}, "換股前開盤總值": before, "換股後開盤總值": after, "交易成本": cost,
           "成交金額合計": turn, "交易成本占成交金額%": round(cost / turn * 100, 4) if turn else None, "沒買進": nobuy, "沒賣出": stuck, "注意": warn,
           "成本口徑": "研究中心嚴成本（make_list.plan）：成交價＝開盤 ±（半個跳動點＋0.3% 滑價）；手續費 0.1425% × 0.28（最低 20 元）；證交稅個股 0.3%、0050 0.1%",
           "plan": pl}
    json.dump(out, open(jp, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    if cost < -1e-6 or cost > 0.03 * before:
        raise SystemExit(f"檢查失敗：換股前開盤總值 {before:,.2f} − 換股後 {after:,.2f}＝{cost:,.2f} 元不像交易成本。檔案已寫出（{hp} 等），請人工檢查後修正或刪除")
    print(f"{a.month} 結算完成：訊號日 {sig}、執行日 {exd}（開盤成交）")
    print(f"  賣出：{'、'.join(s['code'] + ' ' + format(s['shares'], ',.0f') + ' 股' for s in pl['sells']) or '無'}")
    print(f"  買進：{'、'.join(b['code'] + ' ' + format(b['shares'], ',.0f') + ' 股' for b in pl['buys'] if b['action'] == '買進') or '無'}；0050：{pl['e50']['action']}（照開盤價重算）")
    print(f"  換股後持股：{'、'.join(k + ' ' + str(fmt_sh(v)) for k, v in new.items())}；現金 {cash_new:,.2f} 元；底倉 {base_new:,.0f} 股")
    print(f"  換股前開盤總值 {before:,.2f} 元 → 換股後 {after:,.2f} 元；交易成本 {cost:,.2f} 元" + (f"（成交金額的 {cost / turn * 100:.3f}%）" if turn else "")
          + ("" if st is None else f"；換股前現金 {st['cash']:,.2f} 元（含期間現金股利 {div_p:,.2f} 元）"))
    for x in nobuy: print(f"  沒買進：{x['code']} {x['name']}（{x['why']}；不遞補）")
    for x in stuck: print(f"  沒賣出：{x['code']}（{x['why']}）")
    for x in warn: print(f"  注意：{x}")
    print(f"已寫：{hp}、{cp}、{jp}、{a.state}")


def main():
    ap = argparse.ArgumentParser(description="每月換股流程：plan（訊號日收盤後：名單＋預計下單清單）／settle（執行日收盤後：用實際開盤價結算持股檔與現金檔）。只產生檔案，不下單")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "settle"):
        p = sub.add_parser(name)
        p.add_argument("--month", required=True, help="訊號月份，例如 2026-12")
        p.add_argument("--tsd", required=True, help="官方日 K 倉庫（git）")
        p.add_argument("--state", default=STATE_DEFAULT, help="段落表 segments.csv（持股檔、現金檔放在同一個資料夾）")
        p.add_argument("--lists-dir", default=LISTS_DEFAULT, help="名單輸出資料夾")
        p.add_argument("--orders-dir", help="預計下單清單輸出資料夾（預設＝段落表資料夾/orders）")
        p.add_argument("--rev-cutoff", help="補充營收的時點截止（UTC ISO，例如 2026-12-11T14:00:00Z＝台北 22:00）；settle 沒給時沿用訊號日晚上名單的截止")
        p.add_argument("--init-total", type=float, help="第一次、帳戶還沒有持股時：從多少現金開始")
        p.add_argument("--raw", help="研究中心的官方日 K 封存倉庫（預設 make_list.py 的 RAW_DEFAULT）")
        p.add_argument("--extra-repo", action="append", default=[], help="額外的營收備援小倉庫（官方快照或人工檔轉的；排在最後、同月份覆蓋前面的），可多次")
        p.add_argument("--make-list", default=ML_DEFAULT); p.add_argument("--twt49u-dir", default=TWT49U_DEFAULT)
        p.add_argument("--force", action="store_true", help="settle：已結算過或名單有變時仍然重算並覆蓋")
    a = ap.parse_args()
    a.make_list = os.path.abspath(a.make_list); a.twt49u_dir = os.path.abspath(a.twt49u_dir); a.state = os.path.abspath(a.state); a.lists_dir = os.path.abspath(a.lists_dir)
    a.orders_dir = os.path.abspath(a.orders_dir) if a.orders_dir else os.path.join(os.path.dirname(a.state), "orders")
    raw = a.raw or load("make_list", a.make_list).RAW_DEFAULT
    a.repos = [raw, a.tsd] + [os.path.abspath(r) for r in a.extra_repo]
    return cmd_plan(a) if a.cmd == "plan" else cmd_settle(a)


if __name__ == "__main__":
    main()
