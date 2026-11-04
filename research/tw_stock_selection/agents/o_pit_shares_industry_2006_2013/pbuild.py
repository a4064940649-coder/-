#!/usr/bin/env python3
"""股數錨點載入與 PIT 股數建構（路徑 A：錨點法）＋ 產業快照載入。只用 raw/ 與面板 close_raw/open/flag；不讀 adj、ret。

錨點來源（src 代碼；都是「該日普通股股數」）：
  tejm ：TEJ 月底流通在外股數（JunJie-Chang/ff5_replication），存活到 2025-12 者，2000～2025 每月
  tejq ：TEJ 期末市值 ÷ 面板期末收盤（JackWang0102；mv＝期末股本×收盤），存活到 2021-01 者，2005～2007 半年、2008 起每季
  gq   ：goodinfo 季股本(億)×1e8÷面額（yhzbambu；460 檔 1xxx/2xxx 存活股），2006～2007 半年、2008 起每季
  ga   ：goodinfo 年底股本(億)×1e8÷面額（henry753951；含部分下市股），每年 12-31
  twq  ：twsec 每季股本(千元)×1e3÷面額（上市非金融，2013Q1 起）
面額：2013 年以前上市普通股一律 10 元（2014-01 起才開放彈性面額）；本題窗 2006～2013 不做面額偵測，只在 explore 報 tejm 與帳面股本比例異常的股票。
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
import plib as P

PAR = 10.0
SRC_PRIO = {'tejm': 6, 'tejq': 5, 'twq': 4, 'gq': 3, 'ga': 2, 'o8': 1}     # 同交易日多來源時的「真值」優先序（事前定）

def _qend(y, q): return pd.Timestamp(y, 3 * q, 1) + pd.offsets.MonthEnd(0)

def load_anchor_rows(t_lo='2004-06-01', t_hi='2021-12-31'):
    """回傳長表 code, date（曆日）, t（面板索引：≤ date 的最後交易日）, shares, src。"""
    rows = []
    # tejm
    X = pd.read_csv(P.RAW / 'tej_ff5' / 'shares_monthly.csv', dtype={'code': str})
    X = X[X.market.isin(['TSE', 'REG']) & X.shares_k.notna() & (X.shares_k > 0)]
    X['date'] = pd.to_datetime(X.month_end)
    for r in X.itertuples(): rows.append((r.code, r.date, float(r.shares_k) * 1e3, 'tejm'))
    # tejq
    M = pd.read_excel(P.RAW / 'tej_jackwang' / '市值.xlsx'); M['code'] = M.ticker.astype(str)
    for r in M.itertuples():
        if r.code not in P.CI or not (r.mv > 0): continue
        t = P.ti(r.date)
        if P.IDX[t] != pd.Timestamp(r.date): continue
        c = P.Cn[t, P.CI[r.code]]
        if np.isfinite(c) and c > 0: rows.append((r.code, pd.Timestamp(r.date), float(r.mv) * 1e3 / c, 'tejq'))
    # gq
    for f in sorted((P.RAW / 'goodinfo_q').glob('*_BzPerformance.csv')):
        d = pd.read_csv(f, dtype=str)
        for q, cap in zip(d['季度'], d['股本(億)']):
            try: v = float(str(cap).replace(',', ''))
            except ValueError: continue
            if not (v > 0) or 'Q' not in str(q): continue
            y, k = int(q[:4]), int(q[-1])
            rows.append((f.name.split('_')[0], _qend(y, k), v * 1e8 / PAR, 'gq'))
    # ga
    G = pd.read_csv(P.RAW / 'goodinfo_a' / 'equity_year.csv', dtype={'code': str})
    for r in G.itertuples():
        if r.equity_e8 > 0 and r.year <= 2022: rows.append((r.code, pd.Timestamp(int(r.year), 12, 31), float(r.equity_e8) * 1e8 / PAR, 'ga'))
    # twq
    for f in sorted((P.RAW / 'twsec_q').glob('*.csv')):
        code = f.stem.replace('out_', '')
        try: d = pd.read_csv(f, skiprows=1)
        except Exception: continue
        if '股本' not in d.columns: continue
        for q, cap in zip(d['季度'], d['股本']):
            try: v = float(cap)
            except (TypeError, ValueError): continue
            q = str(q).strip()
            if v > 0 and ' Q' in q: rows.append((code, _qend(int(q[:4]), int(q[-1])), v * 1e3 / PAR, 'twq'))
    # o8（2013-01 起 PIT 股數；只取季末，最低優先，給 2013～2014 有、但其他來源沒有的股票當錨點）
    S8 = pd.read_parquet(P.RAW / 'center' / 'o8_shares_pit.parquet')
    for q in pd.date_range('2013-03-31', '2014-12-31', freq='QE'):
        t = P.ti(q); d = P.IDX[t]
        if d not in S8.index: continue
        r = S8.loc[d]
        for c, v in r.dropna().items():
            if v > 0: rows.append((c, d, float(v), 'o8'))
    A = pd.DataFrame(rows, columns=['code', 'date', 'shares', 'src'])
    A = A[A.code.isin(P.CI) & (A.date >= pd.Timestamp(t_lo)) & (A.date <= pd.Timestamp(t_hi))]
    A['t'] = [P.ti(d) for d in A.date]
    A = A[A.t >= 0].drop_duplicates(['code', 'date', 'src'], keep='last').reset_index(drop=True)
    return A

def merge_anchors(A, prio=SRC_PRIO):
    """同股同一交易日（t）多來源 → 取優先序最高者（tejm＞tejq＞twq＞gq＞ga＞o8）。回傳 code, date, t, shares, src。"""
    B = A.copy(); B['p'] = B.src.map(prio)
    B = B.sort_values(['code', 't', 'p']).drop_duplicates(['code', 't'], keep='last')
    return B.drop(columns='p').reset_index(drop=True)

# ---------------- 事件（除權等股數變動日；只用 open/flag/前收） ----------------
TAU_CASH = 0.08; DELTA = 0.02
def event_multipliers(t_lo, t_hi, F=None):
    """與 o12 share_multipliers 同規則，但因子 f 一律用開盤跳空估計（plib.open_factor，不讀 ret）。
    flag 1/2：ŷ＝本股 ±3 年 f−1∈(0.5%,8%] 事件中位數（否則當年市場中位數）；m＝f/(1+ŷ)，m ≤ 1.02 → 1。
    flag 3（參考價重設：減資、現增除權等）：|f−1| > 2% → m＝f。m 夾在 [0.2, 3]。回傳 dict j -> list[(t, m)]（只含 m≠1）。"""
    if F is None: F = P.open_factor()
    yr = P.IDX.year.values; Fn = P.Fn
    sub = F[t_lo:t_hi + 1]
    ev = np.argwhere(np.isfinite(sub) & (np.abs(np.nan_to_num(sub, nan=1.0) - 1) > 0.005)); ev[:, 0] += t_lo
    cash, mkt = {}, {}
    for t, j in ev:
        x = F[t, j] - 1
        if Fn[t, j] in (1, 2) and 0.005 < x <= TAU_CASH:
            cash.setdefault(j, []).append((yr[t], x)); mkt.setdefault(yr[t], []).append(x)
    mmed = {y: float(np.median(v)) for y, v in mkt.items()}
    out = {}
    for t, j in ev:
        fx = F[t, j]
        if Fn[t, j] == 3:
            m = fx if abs(fx - 1) > 0.02 else 1.0
        else:
            own = [x for (y, x) in cash.get(j, []) if abs(y - yr[t]) <= 3]
            yh = float(np.median(own)) if own else mmed.get(yr[t], 0.03)
            m = fx / (1 + yh)
            if m <= 1 + DELTA: m = 1.0
        m = float(min(max(m, 0.2), 3.0))
        if m != 1.0: out.setdefault(int(j), []).append((int(t), m))
    return out

def flag_events(t_lo, t_hi):
    """所有 flag 1/2/3 事件日（不論估計 m），供「區間內是否有事件」判斷。dict j -> sorted list[t]。"""
    sub = P.Fn[t_lo:t_hi + 1]
    ev = np.argwhere(np.isin(sub, (1, 2, 3)) & P.TRADED[t_lo:t_hi + 1]); ev[:, 0] += t_lo
    out = {}
    for t, j in ev: out.setdefault(int(j), []).append(int(t))
    return out

# ---------------- 建構 ----------------
def build(anc, t_lo, t_hi, mult, version='V1'):
    """anc：merge 後錨點表（可已移除留一錨點）。回傳 S（T×N；t_lo..t_hi 以外為 NaN）、SRC（0 無、1 錨點日、2 錨點間、3 外推）。
    股數事件＝mult 中估計乘數 m≠1 的事件（配股、參考價重設）。
    V1（階梯，o8 規則）：兩錨點間有股數事件 → 整段對數變動放在第一個事件日；無事件 → 依交易日對數線性內插。
    V2（事件估計＋內插）：事件日先套估計 m（若 m 與總變動 r 異號或 |log m| > |log r|，改成整段放在第一個事件日），剩餘對數變動依交易日線性內插；無事件 → 對數內插。
    V3（純內插，對照組）：不看事件，一律依交易日對數線性內插。
    錨點外：用 mult 乘數外推（往前除以 m、往後乘以 m）；三版相同。"""
    T = len(P.IDX); S = np.full((T, P.N), np.nan); SRC = np.zeros((T, P.N), np.int8)
    for code, g in anc.groupby('code'):
        j = P.CI[code]
        pts = sorted((int(t), float(v)) for t, v in zip(g.t, g.shares) if np.isfinite(v) and v > 0)
        if not pts: continue
        ev_m = sorted(mult.get(j, []))
        for (t0, v0), (t1, v1) in zip(pts[:-1], pts[1:]):
            if t1 <= t0: continue
            lr = np.log(v1 / v0); seg = np.arange(t0, t1 + 1)
            em = [(t, m) for (t, m) in ev_m if t0 < t <= t1]
            if version == 'V3' or not em:
                S[t0:t1 + 1, j] = v0 * np.exp(lr * (seg - t0) / (t1 - t0))
            elif version == 'V1':
                S[t0:t1 + 1, j] = np.where(seg >= em[0][0], v1, v0)
            else:
                lm = float(np.sum(np.log([m for _, m in em])))
                if np.sign(lr) != np.sign(lm) or abs(lm) > abs(lr):
                    S[t0:t1 + 1, j] = np.where(seg >= em[0][0], v1, v0)
                else:
                    st = np.zeros(len(seg))
                    for t, m in em: st += np.where(seg >= t, np.log(m), 0.0)
                    S[t0:t1 + 1, j] = v0 * np.exp(st + (lr - lm) * (seg - t0) / (t1 - t0))
            SRC[t0 + 1:t1, j] = 2
        for t, v in pts:
            S[t, j] = v; SRC[t, j] = 1
        ta, va = pts[0]; tb, vb = pts[-1]
        lo = max(t_lo, 0)
        if ta > lo:
            cur = va; hi = ta - 1
            for t, m in sorted([(t, m) for t, m in ev_m if lo < t <= ta], reverse=True):
                if t <= hi: S[t:hi + 1, j] = cur
                cur = cur / m; hi = t - 1
            S[lo:hi + 1, j] = cur; SRC[lo:ta, j] = 3
        if tb < t_hi:
            cur = vb; lo2 = tb + 1
            for t, m in sorted([(t, m) for t, m in ev_m if tb < t <= t_hi]):
                S[lo2:t, j] = cur; cur = cur * m; lo2 = t
            S[lo2:t_hi + 1, j] = cur; SRC[tb + 1:t_hi + 1, j] = 3
    S[:t_lo] = np.nan; S[t_hi + 1:] = np.nan; SRC[:t_lo] = 0; SRC[t_hi + 1:] = 0
    return S, SRC

def impute_mcap(t, mc, alive, ADV):
    """o12 同規則：無股數的存活股 市值＝ADV120 × 同期有股數股（ADV 名次 30～200）市值/ADV 中位數。"""
    a = ADV[t]
    have = np.isfinite(mc) & (mc > 0) & alive & np.isfinite(a) & (a > 0)
    need = alive & ~np.isfinite(mc) & np.isfinite(a) & (a > 0)
    out = mc.copy(); imp = np.zeros(P.N, bool)
    if need.any() and have.sum() > 50:
        o = np.argsort(-np.where(have, a, -np.inf)); sel = o[29:200]; sel = sel[have[sel]]
        k = float(np.median(mc[sel] / a[sel]))
        out[need] = a[need] * k; imp[need] = True
    return out, imp

# ---------------- 產業快照 ----------------
TSE_CODE = {1: '水泥工業', 2: '食品工業', 3: '塑膠工業', 4: '紡織纖維', 5: '電機機械', 6: '電器電纜', 8: '玻璃陶瓷', 9: '造紙工業', 10: '鋼鐵工業',
            11: '橡膠工業', 12: '汽車工業', 13: '電子工業', 14: '建材營造', 15: '航運業', 16: '觀光事業', 17: '金融保險業', 18: '貿易百貨', 19: '綜合',
            20: '其他業', 21: '化學工業', 22: '生技醫療業', 23: '油電燃氣業', 24: '半導體業', 25: '電腦及週邊設備業', 26: '光電業', 27: '通信網路業',
            28: '電子零組件業', 29: '電子通路業', 30: '資訊服務業', 31: '其他電子業', 7: '化學生技醫療'}
def norm_ind(s):
    s = str(s).strip()
    m = {'金融保險': '金融保險業', '其他': '其他業', '觀光事業': '觀光事業', '觀光餐旅': '觀光事業'}
    return m.get(s, s)

def load_snapshots():
    """回傳 list of dict(date, src, evidence, map: code->industry)。只收上市普通股（4 碼數字、非 0 開頭）。"""
    snaps = []
    man = json.loads((P.RAW / 'manifest.json').read_text(encoding='utf-8'))['files']
    for rel, meta in man.items():
        if not rel.startswith('industry_snap/') or 'industry_code' in rel: continue
        txt = (P.RAW / rel).read_text(encoding='utf-8', errors='replace').splitlines()
        mp = {}
        if 'twse_list' in rel:
            for line in txt[1:]:
                p = line.split(',')
                if len(p) >= 4 and p[0].isdigit() and len(p[0]) == 4 and p[2].isdigit():
                    mp[p[0]] = norm_ind(p[3])
            date = pd.Timestamp('2013-12-18')
        else:
            for line in txt[1:]:
                p = [x.strip().strip('"') for x in line.split(',')]
                if len(p) >= 3 and p[0].isdigit() and len(p[0]) == 4:
                    try: mp[p[0]] = TSE_CODE[int(p[2])]
                    except (ValueError, KeyError): pass
            date = pd.Timestamp(meta['update_line']) if meta.get('update_line') else pd.Timestamp(meta['commit_time'][:10])
        snaps.append(dict(date=date, src=rel, evidence=meta.get('evidence', ''), map=mp))
    # weihautin 2015/2016/2017、ISIN 2023、screener 2026
    def stocklist(rel, d):
        T = pd.read_csv(P.RAW / rel, dtype=str, encoding='utf-8', on_bad_lines='skip')
        cc = [c for c in T.columns if '代號' in c or c.lower() in ('code', 'stock_id')][0]
        ic = [c for c in T.columns if '產業' in c or c.lower() == 'industry'][0]
        mp = {str(a).strip(): norm_ind(b) for a, b in zip(T[cc], T[ic]) if str(a).strip().isdigit() and len(str(a).strip()) == 4 and isinstance(b, str)}
        snaps.append(dict(date=pd.Timestamp(d), src=rel, evidence='A（o12 以收盤價比對定日／檔名日期）', map=mp))
    for rel, d in [('center/StockList_20150522.csv', '2015-05-22'), ('center/StockList_20160401.csv', '2016-04-01'), ('center/StockList_20170706.csv', '2017-07-06')]:
        try: stocklist(rel, d)
        except Exception as e: print('snap fail', rel, e)
    try:
        T = pd.read_csv(P.RAW / 'center/isin_20230116_listed.csv', dtype=str)
        cc = [c for c in T.columns if 'code' in c.lower() or '代號' in c][0]; ic = [c for c in T.columns if 'industry' in c.lower() or '產業' in c][0]
        snaps.append(dict(date=pd.Timestamp('2023-01-16'), src='center/isin_20230116_listed.csv', evidence='A', map={str(a): norm_ind(b) for a, b in zip(T[cc], T[ic]) if isinstance(b, str)}))
    except Exception as e: print('isin fail', e)
    snaps.sort(key=lambda s: s['date'])
    return snaps

# ---------------- 留一錨點檢核（D1a 協定；s2 校準窗與 s3 閘門共用） ----------------
def top_k(mc, k=100):
    o = np.argsort(-np.nan_to_num(mc, nan=-1.0)); return o[:k]

def loo_eval(M, cps, mult, version, S_full, ADV, drop_days=20, win_td=520, anc_filter=None, topk=100):
    """M：merge 後全部錨點；cps：檢核日（面板索引）清單。對每個檢核日 tq：
    真值＝tq 當日錨點（M 中 t==tq）；拿掉所有 |日期−tq| ≤ drop_days 曆日的錨點（全部股票、全部來源）後，用 ±win_td 交易日內剩餘錨點重建 → 留一值。
    沒有真值錨點的存活股：兩邊都用全錨點建構值 S_full（無值則 ADV 插補）。
    anc_filter：可選，函式(DataFrame)→DataFrame，用來模擬稀疏錨點（校準診斷用）。
    回傳每檢核日：真值前 k 與留一前 k 的重疊、真值前 k（有真值者）股數 |對數誤差| 中位數／p90、真值前 k 內無真值檔數、留一插補檔數，與誤差明細。"""
    rows, det = [], []
    for tq in cps:
        d = P.IDX[tq]
        W = M[(M.t >= tq - win_td) & (M.t <= tq + win_td)]
        tru = W[W.t == tq].set_index('code')['shares']
        keep = W[(W.date - d).abs() > pd.Timedelta(days=drop_days)]
        if anc_filter is not None: keep = anc_filter(keep)
        S_loo, _ = build(keep, tq - win_td, tq + win_td, mult, version)
        al = alive_at_cached(tq); px = P.CFF[tq]
        has = np.zeros(P.N, bool); tv = np.full(P.N, np.nan)
        for c, v in tru.items():
            j = P.CI[c]; has[j] = True; tv[j] = v
        base = np.where(has, tv, S_full[tq])
        mt = np.where(al, base * px, np.nan); mt, imp_t = impute_mcap(tq, mt, al, ADV)
        lv = np.where(has, S_loo[tq], S_full[tq])
        ml = np.where(al, lv * px, np.nan); ml, imp_l = impute_mcap(tq, ml, al, ADV)
        T = top_k(mt, topk); H = top_k(ml, topk)
        Th = [j for j in T if has[j]]
        e = np.abs(np.log(ml[Th] / mt[Th]))
        rows.append(dict(d=str(d.date()), overlap=len(set(T) & set(H)), n_true=len(Th), med_abs_log=float(np.median(e)) if len(e) else np.nan,
                         p90_abs_log=float(np.percentile(e, 90)) if len(e) else np.nan, n_truthless_in_top=int(topk - len(Th)),
                         n_imputed_true_top=int(imp_t[T].sum()), n_imputed_loo_in_true_top=int(imp_l[T].sum()),
                         missed=[P.COLS[j] for j in set(T) - set(H)], entered=[P.COLS[j] for j in set(H) - set(T)]))
        rk = {j: r + 1 for r, j in enumerate(T)}
        for j, x in zip(Th, e): det.append(dict(d=str(d.date()), code=P.COLS[j], rank=rk[j], abs_log=float(x), imputed_loo=bool(imp_l[j])))
    return rows, det

_ALIVE = {}
def alive_at_cached(t):
    if t not in _ALIVE: _ALIVE[t] = P.alive_at(t)
    return _ALIVE[t]
