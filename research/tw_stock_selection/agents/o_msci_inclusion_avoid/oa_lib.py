#!/usr/bin/env python3
"""o_msci_inclusion_avoid 共用：審核日期表、MSCI 納入事件、事件 CAR、主策略改版（A／B／C）日頻引擎。

只讀 import：main_v2/v2_core.py（→ independent_audit/ia_core.py）、o_index_deletion_flow/iflib.py（事件表 sha256 檢查）。
不修改任何其他資料夾。
"""
import sys, os, json, math, hashlib
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
sys.path.insert(0, str(AG / 'main_v2'))
import v2_core as V          # noqa: E402
sys.path.pop(0)
K = V.K
sys.path.insert(0, str(AG / 'o_index_deletion_flow'))
import iflib as L            # noqa: E402
sys.path.pop(0)

IDX, COLS, N = K.IDX, K.COLS, K.N
CI = {c: j for j, c in enumerate(COLS)}
SRC3 = (pd.Timestamp('2024-02-15'), pd.Timestamp('2025-09-23'))
SEL_END = pd.Timestamp('2018-12-31')
VAL_START = pd.Timestamp('2019-01-01')

# ---------------- 審核公告日（台灣時間，MSCI 於台灣清晨約 05:00～06:00 公布，當天開盤前已知） ----------------
# 來源：o1 lists_transcribed.tsv 每次審核的新聞日期，逐筆核對新聞用語（「今天凌晨」「昨（13）日」等）；
# 2025-08／2025-11／2026-08 以 WebSearch 補（見 ANN_NOTE）。
ANN_TW = {'2010-11': '2010-11-11', '2011-11': '2011-11-16', '2012-11': '2012-11-15', '2013-05': '2013-05-16', '2013-08': '2013-08-15',
          '2013-11': '2013-11-08', '2014-02': '2014-02-13', '2014-05': '2014-05-15', '2014-11': '2014-11-07', '2015-02': '2015-02-12',
          '2015-05': '2015-05-13', '2015-11': '2015-11-13', '2016-05': '2016-05-13', '2016-11': '2016-11-15', '2017-05': '2017-05-16',
          '2017-08': '2017-08-11', '2017-11': '2017-11-14', '2018-05': '2018-05-15', '2018-11': '2018-11-14', '2019-05': '2019-05-14',
          '2019-11': '2019-11-08', '2020-05': '2020-05-13', '2020-08': '2020-08-13', '2020-11': '2020-11-11', '2021-02': '2021-02-10',
          '2021-05': '2021-05-12', '2021-11': '2021-11-12', '2022-05': '2022-05-13', '2022-11': '2022-11-11', '2023-02': '2023-02-10',
          '2023-05': '2023-05-12', '2023-08': '2023-08-11', '2023-11': '2023-11-15', '2024-02': '2024-02-13', '2024-05': '2024-05-15',
          '2024-08': '2024-08-13', '2024-11': '2024-11-07', '2025-02': '2025-02-12', '2025-05': '2025-05-14', '2025-08': '2025-08-08',
          '2025-11': '2025-11-06', '2026-02': '2026-02-11', '2026-05': '2026-05-13', '2026-08': '2026-08-13'}
ANN_NOTE = {
    '2010-11': 'bofrank 2542/20101112 自由時報：「MSCI…昨天公布」→ 台灣 11-11',
    '2011-11': 'bofrank 6176/20111116：「MSCI 今天公佈…和泰車盤中亮燈漲停」→ 11-16',
    '2012-11': 'bofrank 2368/20121115：「MSCI 今天凌晨公布」→ 11-15',
    '2015-05': 'ctbc 20150514_189：「半年度調整結果昨（13）日出爐」→ 05-13（20150513 MoneyDJ 為當日稿）',
    '2017-05': 'ctbc 20170516：「明晟今晨公布成分股半年度調整」→ 05-16',
    '2020-08': 'ctbc 202008：「MSCI 昨 (13) 日宣布」「將於台灣時間 13 日公布」→ 08-13',
    '2025-08': 'WebSearch：chinatimes 20250804「8 月 8 日凌晨公布結果，8 月 26 日尾盤生效」、technews 2025/08/08 → 08-08',
    '2025-11': 'WebSearch：udn 9121024／uanalyze「MSCI 今 (6) 日凌晨公布…11/24 收盤後生效」→ 11-06',
    '2026-08': 'WebSearch：technews 2026/08/13、pocket.tw 7774「8 月 13 日台灣時間清晨公布，8 月 31 日收盤後生效」→ 08-13',
    'others': '新聞稿日期即台灣公布日（ctbc／agora0／nick 的當日稿），並與 MSCI 美國時間公布日 +1 日的慣例一致',
}

def cal_on_or_after(d):
    i = IDX.searchsorted(pd.Timestamp(d), side='left'); return i

def review_table():
    """每次審核：台灣公布日、該月換股訊號日、成交日、生效日 E、公告是否在訊號日收盤前／成交日開盤前已知。"""
    ev = L.load_events()
    eff = ev[ev.family == 'MSCI'].groupby('review').eff.first()
    rows = []
    for rv, a in ANN_TW.items():
        a = pd.Timestamp(a); E = pd.Timestamp(eff[rv])
        y, m = map(int, rv.split('-'))
        sd = K.rebalance_days(f'{y}-{m:02d}-01', f'{y}-{m:02d}-28')
        sd = [d for d in sd if d.month == m][0]
        ex = IDX[IDX.get_loc(sd) + 1]
        iA = cal_on_or_after(a)
        rows.append(dict(review=rv, ann_tw=a, ann_trade_day=IDX[iA], signal=sd, exec=ex, eff=E,
                         known_at_signal=bool(a <= sd), known_at_exec=bool(a <= ex),
                         tdays_ann_to_E=int(IDX.get_loc(E) - iA), tdays_exec_to_E=int(IDX.get_loc(E) - IDX.get_loc(ex)),
                         src3=bool(SRC3[0] <= E <= SRC3[1]), period='select' if E <= SEL_END else 'oos',
                         ann_note=ANN_NOTE.get(rv, ANN_NOTE['others'])))
    return pd.DataFrame(rows)

# ---------------- 事件 ----------------
K_ADD = {'STD': 5e8, 'SC': 5e7, 'TW50': 3e8, 'HDETF': 3e8}     # 類別典型被動買額（NT$），沿用 o1 的刪除端常數

def inclusion_events():
    """MSCI 標準納入（含小型升級）＋小型納入（排除同次審核由標準降級者）＋安慰劑（臺灣50、高股息 ETF 納入）。"""
    ev = L.load_events()
    ev = ev.copy()
    std_del = set(zip(ev[ev.type == 'MSCI_STD_DEL'].review, ev[ev.type == 'MSCI_STD_DEL'].code))
    out = []
    for r in ev.itertuples():
        if r.type == 'MSCI_STD_ADD': c = 'STD'
        elif r.type == 'MSCI_SC_ADD':
            if (r.review, r.code) in std_del: continue          # 標準降級到小型＝淨賣出，不算納入
            c = 'SC'
        elif r.type == 'TW50_ADD': c = 'TW50'
        elif r.type.endswith('_ADD') and r.index in L.HD: c = 'HDETF'
        else: continue
        out.append(dict(event_id=r.event_id, review=r.review if r.family == 'MSCI' else f'{r.index}:{r.eff.date()}', cls=c, code=r.code,
                        name=r.name, eff=r.eff, in_panel=bool(r.in_panel), src=r.src))
    X = pd.DataFrame(out)
    X = X[X.code.isin(CI.keys())].copy()
    X = X[X.eff.isin(IDX)].copy()
    X['iE'] = [IDX.get_loc(e) for e in X.eff]
    RT = review_table().set_index('review')
    X['ann_tw'] = [RT.ann_tw.get(rv, pd.NaT) for rv in X.review]
    X['iA'] = [cal_on_or_after(a) if pd.notna(a) else -1 for a in X.ann_tw]
    X['src3'] = (X.eff >= SRC3[0]) & (X.eff <= SRC3[1])
    X['period'] = np.where(X.eff <= SEL_END, 'select', 'oos')
    # 同一檔同一次審核只留一筆（STD 優先）
    X['pri'] = X.cls.map({'STD': 0, 'SC': 1, 'TW50': 2, 'HDETF': 3})
    X = X.sort_values(['review', 'code', 'pri']).drop_duplicates(['review', 'code']).drop(columns='pri')
    return X.reset_index(drop=True)

# ---------------- 事件報酬 ----------------
_RN = np.clip(np.nan_to_num(K.Rn, nan=0.0), -0.5, 0.5)
_RNm = np.where(np.isfinite(K.Rn), np.clip(K.Rn, -0.5, 0.5), np.nan)
EW_R = np.nanmean(_RNm, axis=1)                       # 上市等權日報酬（同 iflib：ret clip ±0.5）
V20 = K.V.rolling(20, min_periods=15).mean().to_numpy()
V60 = K.V.rolling(60, min_periods=40).mean().to_numpy()

def ar(j, a, b):
    """日異常報酬加總 a..b（含），停牌日異常＝−等權。"""
    if a > b: return 0.0
    return float(np.sum(_RN[a:b + 1, j] - EW_R[a:b + 1]))

def limit_dn_locked(t, j):
    """收盤是否鎖跌停（無 low／量資料：收盤 ≤ 跌停價視為賣不出，保守）。"""
    if not K.TRADED[t, j]: return True
    lim = 0.07 if IDX[t] < pd.Timestamp('2015-06-01') else 0.10
    ref = K.ref_price(t)[j]
    if not np.isfinite(ref): return False
    raw = ref * (1 - lim); tk = K.tick(raw); ld = np.ceil(raw / tk - 1e-9) * tk
    return bool(K.Cn[t, j] <= ld + 1e-6)

def event_metrics(X, next_reb=None):
    """每筆事件的描述性指標（日異常加總，對上市等權）。"""
    rows = []
    for r in X.itertuples():
        j = CI[r.code]; i = r.iE
        rec = dict(event_id=r.event_id)
        if i + 21 >= len(IDX) or i - 25 < 0: rec['ok'] = False; rows.append(rec); continue
        v60, v20, px = V60[i - 1, j], V20[i - 1, j], K.Cn[i - 1, j]
        rec['v60'] = v60; rec['v20'] = v20; rec['px'] = px
        rec['ok'] = bool(np.isfinite(v60) and v60 >= 1e7 and np.isfinite(px) and px >= 10 and K.TRADED[i, j])
        rec['pressure'] = K_ADD[r.cls] / v20 if np.isfinite(v20) and v20 > 0 else np.nan
        rec['vol_ratio_E'] = float(K.V.iat[i, j] / np.nanmedian(K.V.iloc[i - 25:i - 5, j].to_numpy())) if K.TRADED[i, j] else np.nan
        rec['car_pre10'] = ar(j, i - 10, i - 1)          # E−10..E−1
        rec['ar_E'] = ar(j, i, i)
        rec['car_runup'] = ar(j, i - 10, i)               # E−10..E（P2 的漲幅）
        rec['car_p1_5'] = ar(j, i + 1, i + 5)
        rec['car_p1_10'] = ar(j, i + 1, i + 10)
        rec['car_p1_20'] = ar(j, i + 1, i + 20)
        rec['ar_p1'] = ar(j, i + 1, i + 1)
        iA = r.iA
        if iA is not None and iA >= 0:
            rec['car_preA10'] = ar(j, iA - 10, iA - 1)    # 公告前 10 日
            rec['ar_A'] = ar(j, iA, iA)                   # 公告日（開盤前已知）
            rec['car_A_E'] = ar(j, iA, i)                 # 公告日到生效日（含）
            rec['tdays_A_E'] = int(i - iA)
        # 買進持有異常（E 收盤 → E+5 收盤），對上市等權買進持有
        A = K.MYADJ.to_numpy()
        a0, a1 = A[i, j], A[i + 5, j]
        if np.isfinite(a0) and np.isfinite(a1):
            br = A[i + 5] / A[i] - 1
            rec['bhar_p1_5'] = float(a1 / a0 - 1 - np.nanmean(np.clip(br[np.isfinite(br)], -0.9, 5)))
        # 52 週高點比（E−1；說明主策略的入選傾向）
        rows.append(rec)
    T = pd.DataFrame(rows)
    return X.merge(T, on='event_id', how='left')

def clus(x, g):
    """按群（審核日）平均後的 t。回傳 dict。"""
    s = pd.Series(np.asarray(x, float), index=np.asarray(g)).dropna()
    if len(s) == 0: return {'n': 0}
    m = s.groupby(level=0).mean()
    t = float(m.mean() / (m.std(ddof=1) / math.sqrt(len(m)))) if len(m) >= 3 and m.std(ddof=1) > 0 else float('nan')
    return {'n': int(len(s)), 'n_clusters': int(len(m)), 'mean_clustered%': round(float(m.mean()) * 100, 2),
            'pooled_mean%': round(float(s.mean()) * 100, 2), 'median%': round(float(s.median()) * 100, 2),
            't_clustered': round(t, 2) if np.isfinite(t) else None}

# ---------------- 主策略改版：選股計畫 ----------------
YL, Y5, Y5f = V.load_revenue()
RT_CACHE = {}

def strategy(exfin=True):
    key = exfin
    if key not in RT_CACHE:
        RT_CACHE[key] = V.V2('gapfix', YL, Y5, exfin=exfin)
    return RT_CACHE[key]

def pending_inclusions(INC, exec_t, classes, gate=None):
    """成交日開盤前已公告、尚未生效（E ≥ 成交日）的 MSCI 納入代碼集合。gate：可選 event_id 集合（壓力門檻）。"""
    d = IDX[exec_t]
    m = INC.cls.isin(classes) & (INC.ann_tw <= d) & (INC.eff >= d)
    if gate is not None: m &= INC.event_id.isin(gate)
    return set(INC[m].code)

def plan_mod(S, n, INC=None, a_classes=(), gate=None):
    """複製 V2.plan（official 同分），另外排除「已知、待生效」的 MSCI 納入股（A 版）。回傳 [(e, picks, pool, d, ranked)]。"""
    out = []; held = set(); nblk = 0; nskip = []
    for d in S.sig_days:
        cs = S.cross_section(d)
        el = cs['el']
        if el.sum() < 50: continue
        e = cs['e']; gu = K.gap_up_open(e)
        cand = S.order(cs, 'official')[:int(el.sum())]
        excl = set()
        if a_classes and INC is not None:
            excl = {CI[c] for c in pending_inclusions(INC, e, a_classes, gate)}
        pick = []; skipped = []
        for j in cand:
            if j in excl:
                skipped.append(j); continue
            if gu[j] and j not in held: nblk += 1; continue
            pick.append(j)
            if len(pick) == n: break
        # 只記錄「若無 A 規則本來會入選」的被跳過者
        out.append((e, np.array(pick, int), np.where(el)[0], d, cand))
        if skipped:
            base_pick = []
            for j in cand:
                if gu[j] and j not in held: continue
                base_pick.append(j)
                if len(base_pick) == n: break
            sk = [COLS[j] for j in skipped if j in base_pick]
            if sk: nskip.append((str(d.date()), sk))
        held = set(pick)
    return out, nskip

# ---------------- 日頻引擎（K.simulate 的擴充：月中 E 日收盤賣出、E+1 開盤遞補或買 0050） ----------------
E50 = K.E50.to_numpy()

def simulate_mod(plan, buy, sell, t0, t1, actions=None, b50=(K.FEE, K.FEE + 0.001), INC_codes_by_review=None):
    """plan：[(e, picks, pool, d, ranked)]。actions：dict t_E → list of (j, mode, ranked_list, excl_set)。
    mode 'repl'：E 收盤賣出、E+1 開盤用名單下一名遞補；'e50'：E 收盤賣出、E+1 開盤起持有 0050（以 E 收盤價近似），下一換股日開盤賣出（以前一日收盤近似）。
    跌停鎖死（收盤 ≤ 跌停價）或 E 日無成交：延後到下一日收盤再試，最多到下一換股日前一日。
    無 actions 時與 K.simulate（freeze=True）逐日相同。"""
    actions = actions or {}
    reb = {p[0]: p for p in plan}
    reb_days = sorted(reb)
    val = np.zeros(N); cash = 1.0; e50v = 0.0
    nav = np.empty(t1 - t0 + 1); log = []
    pend_sell = {}     # t → list of (j, mode, ranked, excl)
    pend_buy = []      # list of (cash_amount, mode, ranked, excl)，下一日開盤各自買進
    for k, v in actions.items():
        pend_sell.setdefault(k, []).extend(v)
    for t in range(t0, t1 + 1):
        if t in reb:
            pk = reb[t][1]
            # 0050 部位與待買現金先回到現金（0050 以前一日收盤近似賣出）
            if e50v > 0: cash += e50v * (1 - b50[1]); e50v = 0.0
            if pend_buy: cash += sum(x[0] for x in pend_buy); pend_buy = []
            tr = K.TRADED[t] & np.isfinite(K.On[t])
            gap = np.where(tr, K.GROW[t] * K.On[t] / np.where(tr, K.Cn[t], 1.0), 1.0)
            h = val * gap
            frz = (h > 0) & ~tr
            Vfz = h[frz].sum()
            htr = np.where(frz, 0.0, h)
            Vt = htr.sum() + cash
            kk = len(pk)
            def cost_of(T):
                tgt = np.zeros(N); tgt[pk] = T
                return buy * np.clip(tgt - htr, 0, None).sum() + sell * np.clip(htr - tgt, 0, None).sum()
            lo, hi = 0.0, Vt / kk
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if mid * kk + cost_of(mid) > Vt: hi = mid
                else: lo = mid
            T = lo
            newval = np.where(frz, h, 0.0)
            newval[pk] += T
            oc = np.where(tr, K.Cn[t] / np.where(tr, K.On[t], 1.0), 1.0)
            val = np.where(frz, newval, newval * oc)
            cash = 0.0
            cur_plan = reb[t]
        else:
            if t > t0:
                val = val * K.GROW[t]
                if e50v > 0: e50v *= E50[t] / E50[t - 1]
            if pend_buy:
                gu = K.gap_up_open(t)
                for (amt, mode, ranked, excl) in pend_buy:
                    done = False
                    if mode == 'repl':
                        for j in ranked:
                            if val[j] > 0 or j in excl: continue
                            if not (K.TRADED[t, j] and np.isfinite(K.On[t, j])) or gu[j]: continue
                            val[j] += amt / (1 + buy) * K.Cn[t, j] / K.On[t, j]
                            log.append(('buy_repl', str(IDX[t].date()), COLS[j])); done = True; break
                    if not done:   # 'e50' 或找不到遞補 → 0050
                        e50v += amt / (1 + b50[0]) * (E50[t] / E50[t - 1])
                        log.append(('buy_0050', str(IDX[t].date()), mode))
                pend_buy = []
        # 收盤：E 日賣出
        if t in pend_sell:
            nxt = [d for d in reb_days if d > t]
            nxt = nxt[0] if nxt else t1 + 1
            for (j, mode, ranked, excl) in pend_sell.pop(t):
                if val[j] <= 0: continue
                if limit_dn_locked(t, j):
                    if t + 1 < nxt:
                        pend_sell.setdefault(t + 1, []).append((j, mode, ranked, excl)); log.append(('postpone', str(IDX[t].date()), COLS[j]))
                    continue
                amt = val[j] * (1 - sell); val[j] = 0.0
                log.append(('sell_E', str(IDX[t].date()), COLS[j], mode))
                if t + 1 < nxt and t + 1 <= t1:
                    pend_buy.append((amt, mode, ranked, excl))
                else:
                    cash += amt
        nav[t - t0] = val.sum() + cash + e50v + sum(x[0] for x in pend_buy)
    return pd.Series(nav, index=IDX[t0:t1 + 1]), log

def build_actions(plan, INC, b_classes, mode, gate=None):
    """B 版：持有期內（成交日 e_k ≤ E ≤ e_{k+1}−1）遇到 b_classes 納入生效日 → E 收盤賣出。"""
    acts = {}; hits = []
    ex = [p[0] for p in plan]
    sub = INC[INC.cls.isin(b_classes)]
    if gate is not None: sub = sub[sub.event_id.isin(gate)]
    rev_codes = INC[INC.cls.isin(['STD', 'SC'])].groupby('review').code.apply(set).to_dict()
    for k, (e, pk, pool, d, ranked) in enumerate(plan):
        e1 = ex[k + 1] if k + 1 < len(ex) else 10 ** 9
        held = {COLS[j] for j in pk}
        h = sub[(sub.iE >= e) & (sub.iE <= e1 - 1) & sub.code.isin(held)]
        for r in h.itertuples():
            j = CI[r.code]
            excl = {CI[c] for c in rev_codes.get(r.review, set()) if c in CI} | set(pk.tolist())
            rk = [x for x in ranked if x not in excl]
            acts.setdefault(r.iE, []).append((j, mode, rk, excl))
            hits.append(dict(sig=d, exec=IDX[e], E=r.eff, code=r.code, name=r.name, cls=r.cls, review=r.review, event_id=r.event_id))
    return acts, pd.DataFrame(hits)

def sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()
