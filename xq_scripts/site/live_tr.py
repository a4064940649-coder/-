# -*- coding: utf-8 -*-
# 含息總報酬口徑的帳戶逐日淨值（build_site_data.py 的 live.json 用；test_live_tr.py 做歷史假想測試）。
# 2026-11-18 加（S69 FINDINGS「需處理 1」：live.json 淨值原本未含除息調整）。
#
# 做法（策略帳戶與 0050 對照帳戶同一段程式）：
#   - 起始日開盤買進（起始日當天才除權息的不算，因為開盤價已經是除權息後的價）。
#   - 起始日之後遇到持股的除權息／分割事件日：
#       息（現金股利）        → 現金 += 息值 × 股數（留在帳戶現金，不再投入；simulate(reinvest=True) 可改成除息日收盤買回，只給對照用）
#       權、權息、ETF 分割、面額變更、減資 → 股數 × 調整倍數（＝除權息前收盤 ÷ 參考價）
#     權息合併、現金增資除權、退還股款的減資：官方表沒有把現金和股數分開，這裡一律用股數調整
#     （等同把那筆現金在事件日以參考價再投入），事件紀錄會標「近似」，不另外猜拆分比例。
#   - 事件資料只讀官方日 K 倉庫的日期軸事件檔（data/universe/exright 等，git HEAD）與研究中心的 TWT49U 封存；
#     讀不到的日子不調整、不猜，由 coverage()／warnings 明確列出「涵蓋到哪一天」。
# 沒有任何事件時，淨值與舊算法（股數 × 收盤＋現金）逐位相同。
# 注意：官方日 K 倉庫若是 blob:none 的部分複製，git show 讀本機沒有的檔會向遠端抓（要連網）；抓不到時 load_events 把檔名記在 unreadable、
#       coverage_warnings 會警告，不會當成「沒有事件」。
import csv, gzip, glob, os, subprocess

# 官方日 K 倉庫內的事件資料夾：（資料夾, 市場, 類別）
EVENT_DIRS = [("exright", "twse", "exright"), ("otcexright", "tpex", "exright"),
              ("etfsplit", "twse", "split"), ("parvalue", "twse", "split"), ("otcparvalue", "tpex", "split"),
              ("reduce", "twse", "reduce"), ("otcreduce", "tpex", "reduce")]
KIND_MAP = {"息": "息", "權": "權", "權息": "權息", "除息": "息", "除權": "權", "除權息": "權息"}


def _git(tsd, *a):
    return subprocess.run(["git", "-C", tsd] + list(a), capture_output=True, text=True, check=True).stdout


def _num(s):
    try:
        return float(s) if s not in ("", None) else None
    except ValueError:
        return None


def list_event_days(tsd):
    """各事件資料夾有檔案的日子：{資料夾: 排序後的日期清單}。資料夾不存在＝空清單。"""
    out = {}
    for d, _, _ in EVENT_DIRS:
        try:
            txt = _git(tsd, "ls-tree", "--name-only", "HEAD", f"data/universe/{d}/")
        except Exception:
            txt = ""
        out[d] = sorted(x.rsplit("/", 1)[1][:-4] for x in txt.split() if x.endswith(".csv") and not x.rsplit("/", 1)[1].startswith("_"))
    return out


def load_archive(twt49u_dir, lo, hi):
    """研究中心 TWT49U 封存（YYYY/YYYY-MM.csv.gz；date,code,prev_close,ref_price,value,type）。回傳（事件清單, 封存最後一天）。"""
    evs = []; last = None
    if not twt49u_dir or not os.path.isdir(twt49u_dir): return evs, last
    for p in sorted(glob.glob(os.path.join(twt49u_dir, "*", "*.csv.gz"))):
        ym = os.path.basename(p)[:7]
        with gzip.open(p, "rt", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if last is None or r["date"] > last: last = r["date"]
                if ym < lo[:7] or ym > hi[:7] or not (lo < r["date"] <= hi): continue
                evs.append({"date": r["date"], "code": r["code"], "src": "TWT49U封存", "cls": "exright", "kind": KIND_MAP.get(r["type"], r["type"]),
                            "pre_close": _num(r["prev_close"]), "ref_price": _num(r["ref_price"]), "value": _num(r["value"]), "ex_div_ref": None})
    return evs, last


def load_events(tsd, lo, hi, codes, twt49u_dir=None, file_days=None):
    """讀 lo（不含）～hi（含）之間、代號在 codes 裡的事件。回傳（{日期: [事件]}, 涵蓋資訊）。
    來源優先序：官方日 K 倉庫的日期軸事件檔；倉庫沒有那一天的除權息檔而封存有 → 用封存。"""
    codes = set(codes)
    file_days = file_days if file_days is not None else list_event_days(tsd)
    by = {}; seen = set(); unreadable = []
    for d, market, cls in EVENT_DIRS:
        for day in file_days.get(d, []):
            if not (lo < day <= hi): continue
            try:
                txt = _git(tsd, "show", f"HEAD:data/universe/{d}/{day}.csv")
            except Exception:
                unreadable.append(f"data/universe/{d}/{day}.csv"); continue      # 倉庫列得出檔名但讀不到內容：不猜，交給警告
            for r in csv.DictReader(txt.splitlines()):
                if r.get("stock_id") not in codes: continue
                kind = KIND_MAP.get(r.get("kind", ""), None) if cls == "exright" else (r.get("reason") or ("分割" if cls == "split" else "減資"))
                ev = {"date": r.get("date") or day, "code": r["stock_id"], "src": f"data/universe/{d}", "cls": cls, "kind": kind or r.get("kind") or "?",
                      "pre_close": _num(r.get("pre_close")), "ref_price": _num(r.get("ref_price")), "value": _num(r.get("value")),
                      "ex_div_ref": _num(r.get("ex_div_ref"))}
                by.setdefault(ev["date"], []).append(ev); seen.add((ev["date"], ev["code"], cls))
    arch, arch_last = load_archive(twt49u_dir, lo, hi)
    n_arch = 0
    repo_ex_days = set(file_days.get("exright", []))
    for ev in arch:
        if ev["code"] not in codes or ev["date"] in repo_ex_days or (ev["date"], ev["code"], "exright") in seen: continue
        by.setdefault(ev["date"], []).append(ev); n_arch += 1
    last = lambda k: (file_days.get(k) or [None])[-1]
    cov = {"twse_exright_last": max([x for x in (last("exright"), arch_last) if x] or [None], default=None),
           "repo_exright_last": last("exright"), "archive_last": arch_last, "tpex_exright_last": last("otcexright"),
           "dirs_last": {d: last(d) for d, _, _ in EVENT_DIRS}, "n_from_archive": n_arch, "unreadable": unreadable}
    return by, cov


def _snap(fac):
    """分割／面額變更的倍數：參考價四捨五入到檔位，倍數離整數（或整數分之一）不到 0.2% 就取整數。"""
    for x in (fac, 1 / fac):
        n = round(x)
        if n >= 2 and abs(x / n - 1) < 0.002: return float(n) if x is fac else 1.0 / n
    return fac


def apply_event(ev, shares, prev_close=None):
    """一筆事件套到持有 shares 股上。回傳（現金增加, 新股數, 紀錄 dict）。資料不足＝不調整並在紀錄標 error。"""
    pre, ref, val = ev["pre_close"], ev["ref_price"], ev["value"]
    log = {"日期": ev["date"], "代號": ev["code"], "種類": ev["kind"], "來源": ev["src"], "除權息前收盤": pre, "參考價": ref, "權值＋息值": val,
           "事件前股數": shares, "現金股利": 0.0, "股數倍數": 1.0, "近似": False, "注意": []}
    if prev_close is not None and pre is not None and abs(prev_close - pre) > 1e-6:
        log["注意"].append(f"事件表的前收 {pre:g} 和日 K 前一交易日收盤 {prev_close:g} 不同")
    if ev["cls"] == "exright" and ev["kind"] == "息":
        if val is None or val < 0:
            log["error"] = "息值缺漏，未調整"; return 0.0, shares, log
        if pre and ref and abs((pre - ref) - val) > max(0.005 * pre, 0.011):
            log["注意"].append(f"前收 − 參考價 {pre - ref:.4f} 和息值 {val:g} 差超過 0.5%")
        log["現金股利"] = val * shares
        return val * shares, shares, log
    # 其餘：調整股數
    if not pre or not ref or pre <= 0 or ref <= 0:
        log["error"] = "前收或參考價缺漏，未調整"; return 0.0, shares, log
    if ev["cls"] == "exright":
        fac = pre / (pre - val) if (val is not None and 0 < val < pre) else pre / ref     # 權值＋息值未四捨五入，比參考價準
        if ev["kind"] == "權息":
            log["近似"] = True; log["注意"].append("權息合併：官方表沒有分開現金股利與股票股利，全部以股數調整（等同現金股利在除權息日以參考價再投入）")
        elif ev["kind"] == "權":
            x = ev.get("ex_div_ref")
            if x is not None and abs(x - ref) > 1e-9:
                log["近似"] = True; log["注意"].append("含現金增資除權：以參考價調整股數（等同認股權以理論價值換成股票）")
        else:
            log["近似"] = True; log["注意"].append(f"種類「{ev['kind']}」不認得，以前收 ÷ 參考價調整股數")
    elif ev["cls"] == "split":
        fac = _snap(pre / ref)
    else:
        fac = pre / ref
        if "彌補虧損" not in str(ev["kind"]):
            log["近似"] = True; log["注意"].append("減資（非彌補虧損）：官方表沒有分開退還股款與股數，以參考價調整股數")
    log["股數倍數"] = fac
    return 0.0, shares * fac, log


def simulate(sh0, cash0, days, px_of, events, start, reinvest=False):
    """帳戶逐日總值。sh0＝{代號: 股數}；px_of(日)＝{代號: (開盤, 收盤)}；events＝{日期: [事件]}。
    reinvest＝False（預設）：現金股利留在帳戶現金；True：現金股利在除息日以當天收盤買回同一檔（只給對照與測試用，零股、不計費用）。
    回傳 dict：rows＝[(日, 收盤總值)]、base＝起始日開盤總值（含現金）、nav＝最後一天總值、sh＝最後股數、cash＝最後現金、
    div＝{代號: 累計現金股利}、log＝已套用事件紀錄。"""
    sh = dict(sh0); cash = cash0; base = None; nav = None; rows = []; log = []; div = {}
    prev_day = None; prev_px = {}
    ev_days = sorted(events)
    for d in [d for d in days if d >= start]:
        px = px_of(d)
        if not px: continue
        if base is None:
            base = sum(sh[k] * px[k][0] for k in sh if k in px and px[k][0]) + cash
        else:
            for ed in [e for e in ev_days if prev_day < e <= d]:
                for ev in events[ed]:
                    k = ev["code"]
                    if k not in sh or not sh[k]: continue
                    c, s, lg = apply_event(ev, sh[k], (prev_px.get(k) or (None, None))[1])
                    sh[k] = s; div[k] = div.get(k, 0.0) + c; lg["套用日"] = d; log.append(lg)
                    if c and reinvest and k in px and px[k][1]:
                        sh[k] += c / px[k][1]; lg["再投入"] = f"以 {d} 收盤 {px[k][1]:g} 買回 {c / px[k][1]:.6f} 股"
                    else:
                        cash += c
        nav = sum(sh[k] * px[k][1] for k in sh if k in px and px[k][1]) + cash
        rows.append((d, nav)); prev_day = d
        for k in sh:
            if k in px and px[k][1]: prev_px[k] = px[k]
    return {"rows": rows, "base": base, "nav": nav, "sh": sh, "cash": cash, "div": div, "log": log}


def markets(tsd, day, codes):
    """代號 → 市場（twse／tpex…），讀官方日 K 當日檔的 market 欄。"""
    out = {}
    try:
        txt = _git(tsd, "show", f"HEAD:data/universe/daily/{day}.csv")
    except Exception:
        return out
    for r in csv.DictReader(txt.splitlines()):
        if r.get("stock_id") in codes: out[r["stock_id"]] = r.get("market") or ""
    return out


def suspicious_cells(tsd, days, codes, events, start):
    """可疑格子：日 K 的漲跌價差＝0、收盤和前一交易日不同、那天又沒有事件（官方日 K 在除權息日的漲跌價差是 0）。只提示、不調整。"""
    out = []; prev = {}
    for d in [d for d in days if d >= start]:
        try:
            txt = _git(tsd, "show", f"HEAD:data/universe/daily/{d}.csv")
        except Exception:
            continue
        evc = {e["code"] for e in events.get(d, [])}
        for r in csv.DictReader(txt.splitlines()):
            k = r.get("stock_id")
            if k not in codes: continue
            c = _num(r.get("close")); ch = _num(r.get("change"))
            if c is None: continue
            if d > start and k in prev and ch == 0 and abs(c - prev[k]) > 1e-9 and k not in evc: out.append(f"{k} {d}")
            prev[k] = c
    return out


def coverage_warnings(cov, last_day, mk):
    """回傳（涵蓋到哪一天的說明, 警告清單）。警告＝最新股價日比除權息資料最後一天還晚，或持股有上市以外的市場而該市場資料不足。"""
    warns = []
    tw = cov.get("twse_exright_last")
    if cov.get("unreadable"):
        u = cov["unreadable"]
        warns.append(f"有 {len(u)} 個除權息事件檔讀不到內容（{'、'.join(u[:5])}{'…' if len(u) > 5 else ''}），那幾天的事件未調整")
    if any(m != "tpex" for m in mk.values()) or not mk:
        if not tw: warns.append("找不到上市除權息資料，全部期間未調整")
        elif tw < last_day: warns.append(f"上市除權息資料只涵蓋到 {tw}，之後（到 {last_day}）未調整")
    otc = sorted(k for k, m in mk.items() if m == "tpex")
    if otc:
        tp = cov.get("tpex_exright_last")
        if not tp: warns.append(f"持股含上櫃（{'、'.join(otc)}）但找不到上櫃除權息資料，未調整")
        elif tp < last_day: warns.append(f"上櫃除權息資料只涵蓋到 {tp}，之後（到 {last_day}）上櫃持股（{'、'.join(otc)}）未調整")
    other = sorted(k for k, m in mk.items() if m not in ("twse", "tpex"))
    if other: warns.append(f"持股 {'、'.join(other)} 的市場別不是上市或上櫃，沒有除權息資料來源，未調整")
    return tw, warns
