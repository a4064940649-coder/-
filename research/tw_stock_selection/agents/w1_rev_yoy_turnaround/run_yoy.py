#!/usr/bin/env python3
"""營收年增率由負轉正／連 3 月遞增（營收轉機股，CMoney 教學）長歷史 2006～2026 可成交回測。
輸出 results.json（PREREG 先落地，才開始計算）、nav.parquet、_info_*.pkl。執行：python3 -I run_yoy.py
成交、成本、漲停判定、60 日均值修補沿用 momentum_family/mlib.py → exec_rebuild/exlib.py（只讀取）。
權重版引擎（入選 < 5 檔時其餘放 0050；3 個月持有的三批部位）與 exlib.sim_portfolio 邏輯相同，並以等權情形逐日比對驗證。"""
import json, math, re, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'momentum_family'))
import mlib as M  # noqa: E402
L = M.L
T0 = time.time()
LONG = Path('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv')
XR = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras')
SL = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv')

PREREG = {
    '寫於看結果前': True,
    '來源規則': 'CMoney 教學（nid=116293、177417、66097）：營收年增率由負轉正且連 3 個月為正／年增率連 3 個月遞增（營收轉機股）；CMoney 組合版另加年增率 > 20% 與站上季線。',
    '年增率': '長期月營收 tw_revenue.csv（千元；sii 優先去重；只用面板內代碼）自算 YoY(t) = Rev(t)/Rev(t−12) − 1（Rev(t−12) > 0 才有值，單位 %）。',
    '訊號月份 t 與換股日 d': '每月第一個日期 ≥ 11 日的交易日 d（= 營收 10 日截止後第一個交易日；mlib.LDATES），t = d 前一個月。長期檔無公告日，假設 10 日前已公告（最後一期用 rev_hist ann_date 檢查）。d 收盤算訊號，d+1 開盤成交。',
    '版本 A（轉正）': 'YoY(t−3) < 0，且 YoY(t−2)、YoY(t−1)、YoY(t) 都 > 0。',
    '版本 B（加速）': 'YoY(t−2) < YoY(t−1) < YoY(t)，且 YoY(t) > 0。',
    '版本 C（CMoney 組合）': '版本 B 的「連 3 增」＋ YoY(t) > 20% ＋ d 收盤還原價 > 60 日均線（還原價，2018-01-13 等有值檔數 < 300 的日子剔除後滾動）。',
    '股池（三版本共用，同池等權也用它）': '面板上市股，排除 91xx 存託憑證；剔除金融保險（stock_list industry=金融保險業；不在名單者代碼 28xx 或名稱含 金控/銀行/商銀/證券/保險/人壽/產險/票券/期貨）與控股公司（任何時期名稱含 控股/投控/金控，或代碼 3701～3719 投資控股）；d 收盤 60 日均成交值 ≥ NT$1,000 萬（mlib.V60）、收盤 ≥ 10 元；YoY(t−3..t) 四個月都有值；60 日均線有值；d+1 有開盤價。不設前 300 名、不設處置濾網（教學沒有）。',
    'IFRS 處理': '2013 年營收為 IFRS 合併口徑、2012 年為個體口徑，2013 年各月 YoY 失真。只要訊號用到的 YoY 月份落在 2013-01～2013-12（A 用 t−3..t → t 到 2014-03 為止），就不產生訊號：t = 2013-01～2014-03（換股日 2013-02～2014-04，共 15 期）三版本一律 100% 0050，同池等權也放 0050，並從挑選期統計剔除（3 個月持有版的剔除窗相同）。',
    '進場': 'd+1 開盤等權買進全部入選股；d+1 開盤漲停（還原開盤／前收 − 1 ≥ 漲停幅 − 0.6%，漲停幅 2015-06-01 前 7%、後 10%；mlib.GAP_UP）且目前未持有者不買、不遞補（規則本來就全買，沒有排名可遞補）；已持有者續抱。',
    '權重（持有 1 個月，主設定）': '入選 k ≥ 5：每檔 1/k；k < 5：每檔 1/5，其餘 (5−k)/5 放 0050；k = 0：100% 0050。0050 以前一日收盤（含息）成交，成本與個股相同（偏保守）。',
    '權重（持有 3 個月，另報）': '每月新成立一批（1/3 資金），批內規則同上（含 <5 檔補 0050）；組合 = 最近 3 批平均，每月換股日把三批調回各 1/3（Jegadeesh–Titman 重疊組合）。起始前兩月不足的批次放 0050。',
    '出場': '下一個換股日 d+1 開盤調到新目標（續選者只調差額）；最後一期持有到面板最後一日 2026-10-22。',
    '成本': 'exlib.cost_rates：買 0.0855%+滑價、賣 0.0855%+稅 0.3%+滑價；主 = 滑價 0.1%（來回約 0.67%）、壓力 = 買 0.5%+賣 0.5%（來回 1%）；另報未扣成本。',
    '對手': '0050 含息（日頻）；同池等權 = 上述共用股池全部等權、同引擎、d+1 開盤換股、不扣成本（對策略較嚴格；3 個月版也對同一條月換股等權線）。',
    '統計': '日頻 NAV 算年化、日頻最大回撤、逐年；t = 換股日收盤→下一換股日收盤的月窗超額（每月一個觀察值 = 按進場月份分群）。另報事件層級：每筆（股票, 月）d+1 開盤→下一 d+1 開盤（3 個月版：→ 第 3 個 d+1 開盤）報酬扣主成本、減同池等權同期報酬，先按進場月份平均再算 t（3 個月版月份重疊，t 偏高，僅供參考）。',
    '期間': {'挑選期': '2006-05（A 需 YoY(t−3)，最早 t = 2006-04）～2019-04，剔除 IFRS 15 期', '驗證期': '2019-05～2026-10',
             '另報': ['2024-01～2025-12', 'A 乾淨段 2006-05～2013-01（無存活者偏誤、口徑一致）', 'B 存活者段 2014-05～2019-04', '2019-05～2023-12（價格面板只有存活者）', '2021-03～2023-12／2024-01～2026-10（對照 2021 起資料切分）', '官方價格來源段（排除 src3 2024-02-15～2025-09-23）']},
    '判定': '每個版本（A、B、C；持有 1 個月；滑價 0.1%）各自判定：挑選期與驗證期扣成本後皆「對同池等權月窗 t ≥ 2 且年化 ≥ 0050 年化」→ 可用；只有一段通過 → 待觀察；否則不可用。整體結論取三版本中最好的判定，但因同時測 3 個版本，若該版本任一段 t < 2.4（Bonferroni 3 檢定）要註明多重檢定警示。3 個月持有、壓力成本、事件層級只報告、不改判定。',
    '資料檢查（事前列出）': ['IFRS 期間若不剔除的訊號數', '每月入選檔數、<5 檔月數、跳過的開盤漲停數', '入選股持有期公司行動 flag 3/4/5、單月 >100% 報酬', '最後一期（營收 2026-09）用 rev_hist ann_date 檢查公告是否晚於訊號日', 'rev_hist 2024-10 起營收與長期檔比對', '剔除金融／控股檔數'],
}
res = {'PREREG': PREREG}
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')

IDX, COLS, N, pos = L.IDX, L.COLS, L.N, L.pos

# ---------------- 營收／年增率 ----------------
d = pd.read_csv(LONG, dtype={'code': str, 'market': str, 'name': str})
d['p'] = pd.PeriodIndex(pd.to_datetime(d['month']), freq='M')
d = d.sort_values(['p', 'market']).drop_duplicates(['p', 'code'], keep='first')
names_all = d.groupby('code')['name'].apply(lambda s: '|'.join(sorted(set(s.dropna()))))
d = d[d['code'].isin(COLS)]
REV = d.pivot_table(index='p', columns='code', values='revenue_k_twd', aggfunc='last').sort_index()
REV = REV.reindex(pd.period_range(REV.index.min(), REV.index.max(), freq='M')).reindex(columns=COLS)
prev = REV.shift(12)
YOY = ((REV / prev - 1) * 100).where(prev > 0)
Y0, Y1, Y2, Y3 = YOY, YOY.shift(1), YOY.shift(2), YOY.shift(3)
AVAIL = Y0.notna() & Y1.notna() & Y2.notna() & Y3.notna()
SIG_A = (Y3 < 0) & (Y2 > 0) & (Y1 > 0) & (Y0 > 0)
SIG_B = (Y2 < Y1) & (Y1 < Y0) & (Y0 > 0)
SIG_C0 = SIG_B & (Y0 > 20)
IFRS = (YOY.index >= pd.Period('2013-01', 'M')) & (YOY.index <= pd.Period('2014-03', 'M'))

# ---------------- 金融／控股 ----------------
sl = pd.read_csv(SL, dtype={'code': str}).drop_duplicates('code').set_index('code')
FIN_RE = re.compile('金控|銀行|商銀|證券|保險|人壽|產險|票券|期貨')
HOLD_RE = re.compile('控股|投控|金控')
excl_fin, excl_hold = set(), set()
for c in COLS:
    nm = '|'.join([str(sl['name'].get(c, '')), str(names_all.get(c, ''))])
    ind = sl['industry'].get(c, None) if c in sl.index else None
    if (isinstance(ind, str) and ind == '金融保險業') or (c not in sl.index and (c.startswith('28') or FIN_RE.search(nm))):
        excl_fin.add(c)
    if HOLD_RE.search(nm) or (c.isdigit() and 3701 <= int(c) <= 3719):
        excl_hold.add(c)
EXCL = np.array([c in excl_fin or c in excl_hold or c.startswith('91') for c in COLS])
res['excluded'] = {'financial': len(excl_fin), 'holding': len(excl_hold), 'holding_codes': sorted(excl_hold), 'dr_91xx': int(sum(c.startswith('91') for c in COLS))}

# ---------------- 日曆與橫截面 ----------------
first_t = AVAIL.index[AVAIL.sum(axis=1) > 50].min()
D = [x for x in M.LDATES if first_t <= (pd.Period(x, 'M') - 1) <= YOY.index.max()]
D = D + [IDX[-1]] if IDX[-1] > D[-1] else D
DATES = pd.DatetimeIndex(D)
mm = [pd.Period(x, 'M') - 1 for x in D]
GOOD = L.adj.notna().sum(axis=1) >= 300
_ag = L.adj.loc[GOOD]
MA60 = _ag.rolling(60).mean().reindex(IDX)
v60 = M.V60.reindex(DATES); px = L.C.reindex(DATES)
liq = ((v60 >= 1e7) & (px >= 10)).to_numpy() & ~EXCL[None, :]
ma = MA60.reindex(DATES).to_numpy(); ac_d = L.adj.reindex(DATES).to_numpy()
above = np.isfinite(ma) & (ac_d > ma)
def at_m(X):
    a = X.reindex(mm).to_numpy(); return np.where(pd.isna(a), False, a).astype(bool)
avail = at_m(AVAIL); sigs = {'A': at_m(SIG_A), 'B': at_m(SIG_B), 'C': at_m(SIG_C0) & above}
ifrs_m = np.array([(p >= pd.Period('2013-01', 'M')) and (p <= pd.Period('2014-03', 'M')) for p in mm])

E50 = L.E50.ffill()
AC2 = np.hstack([L.AC, E50.to_numpy()[:, None]])
AO2 = np.hstack([L.AO, E50.shift(1).bfill().to_numpy()[:, None]])
J50 = N

def tranche(pick):
    kk = len(pick)
    if kk >= 5: return np.array(pick, dtype=int), np.full(kk, 1.0 / kk)
    return np.array(list(pick) + [J50], dtype=int), np.r_[np.full(kk, 0.2), (5 - kk) * 0.2]

def combine(trs):
    w = np.zeros(N + 1)
    for idx, ww in trs: np.add.at(w, idx, ww / len(trs))
    nz = np.where(w > 0)[0]; return nz, w[nz]

def build(ver, hold=1):
    events, pools, info = {}, {}, []
    held = set(); nblk = 0; trs = [(np.array([J50]), np.array([1.0]))] * (hold - 1)
    for k, dd in enumerate(D[:-1]):
        i = pos(dd); ex = i + 1
        trad = np.isfinite(L.OPEN_RAW[ex]) & np.isfinite(L.ADJ_RAW[ex])
        pool = liq[k] & avail[k] & np.isfinite(ma[k]) & trad
        if ifrs_m[k]:
            cand = np.array([], dtype=int); pick = []
        else:
            cand = np.where(pool & sigs[ver][k])[0]
            pick = [j for j in cand if not (M.GAP_UP[ex, j] and j not in held)]
        nblk += len(cand) - len(pick)
        tr = tranche(pick)
        trs = (trs + [tr])[-hold:]
        events[ex] = combine(trs) if hold > 1 else tr
        pp = np.where(pool)[0]
        pools[ex] = (pp, np.full(len(pp), 1.0 / len(pp))) if (len(pp) and not ifrs_m[k]) else (np.array([J50]), np.array([1.0]))
        info.append({'d': dd, 'ex': ex, 'n_sig': int(len(cand)), 'n_pick': len(pick), 'n_blocked': int(len(cand) - len(pick)), 'pool': int(pool.sum()), 'picks': [COLS[j] for j in pick]})
        held = set(int(j) for j in events[ex][0] if j != J50)
    return events, pools, pd.DataFrame(info), nblk

def sim_w(events, buy, sell, t0, t1):
    """exlib.sim_portfolio 的權重版：換股日以開盤價把漂移後持股調到目標權重，買賣金額分別乘費率（6 次固定點迭代）。"""
    NN = N + 1; u = np.zeros(NN); cash = 1.0; nav = np.empty(t1 - t0 + 1); tov = []
    for t in range(t0, t1 + 1):
        if t in events:
            pk, w = events[t]
            Pz = AO2[t]
            h = np.where(u > 0, u * np.nan_to_num(Pz), 0.0)
            Vt = h.sum() + cash; Vn = Vt; tgt = np.zeros(NN)
            for _ in range(6):
                tgt[:] = 0; tgt[pk] = w * Vn
                cost = buy * np.clip(tgt - h, 0, None).sum() + sell * np.clip(h - tgt, 0, None).sum()
                Vn = Vt - cost
            tov.append(np.clip(h - tgt, 0, None).sum() / Vt if Vt > 0 else 0)
            u = np.zeros(NN); u[pk] = tgt[pk] / Pz[pk]; cash = 0.0
        nav[t - t0] = np.nansum(u * AC2[t]) + cash
    return pd.Series(nav, index=IDX[t0:t1 + 1]), float(np.mean(tov[1:])) if len(tov) > 1 else float('nan')

def daily_stats(nav, segs, bench_nav=None, ew_nav=None):
    """同 w1_rev_record_high/run_rrh.daily_stats。"""
    r = nav.pct_change(); mask = np.zeros(len(r), bool); span = 0.0
    for a, b in segs:
        m = (r.index > pd.Timestamp(a)) & (r.index <= pd.Timestamp(b)) & r.notna().to_numpy()
        if m.sum() == 0: continue
        first = np.argmax(m); last = len(m) - 1 - np.argmax(m[::-1])
        span += (r.index[last] - r.index[first - 1]).days / 365.25
        mask |= m
    rr = r[mask]
    if len(rr) < 20: return None
    eq = (1 + rr).cumprod()
    o = {'CAGR': round(float(eq.iloc[-1] ** (1 / span) - 1), 4), 'vol': round(float(rr.std() * math.sqrt(252)), 4),
         'maxDD': round(float((eq / eq.cummax() - 1).min()), 4), 'years': round(span, 2)}
    nd = nav.reindex(DATES); pr = (nd.shift(-1) / nd - 1).iloc[:-1].dropna()
    inw = np.zeros(len(pr), bool)
    for a, b in segs: inw |= (pr.index >= pd.Timestamp(a)) & (pr.index < pd.Timestamp(b))
    pr = pr[inw]; o['n_months'] = int(len(pr))
    for nm, bn in (('0050', bench_nav), ('EW', ew_nav)):
        if bn is None: continue
        bd = bn.reindex(DATES); bp = (bd.shift(-1) / bd - 1).iloc[:-1].reindex(pr.index); ex = pr - bp
        o[f'excess_vs_{nm}_pct_m'] = round(float(ex.mean() * 100), 2); o[f't_vs_{nm}_m'] = L.tstat(ex)
        o[f'hit_vs_{nm}_m'] = round(float((ex > 0).mean()), 3)
        b = bn.pct_change().reindex(rr.index); eb = (1 + b.fillna(0)).cumprod()
        o[f'{nm}_CAGR'] = round(float(eb.iloc[-1] ** (1 / span) - 1), 4); o[f'{nm}_maxDD'] = round(float((eb / eb.cummax() - 1).min()), 4)
    return o

def yearly(nav):
    r = nav.pct_change().dropna(); return {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in r.groupby(r.index.year)}

# ---- 引擎驗證 ----
_ev, _pl, _, _ = build('A')
_t0 = pos(D[0]); _t1 = pos(D[-1])
_pe = {t: ('open', v[0]) for t, v in _pl.items() if J50 not in v[0]}
_pw = {t: v for t, v in _pl.items() if J50 not in v[0]}
b1, s1 = L.cost_rates(0.001)
na, _ = L.sim_portfolio(_pe, b1, s1, _t0, _t1); nb_, _ = sim_w(_pw, b1, s1, _t0, _t1)
res['engine_check_max_abs_nav_diff'] = float((na - nb_).abs().max())
print('engine check', res['engine_check_max_abs_nav_diff'], flush=True)

def fdate(s): return DATES[DATES >= pd.Timestamp(s)][0]
d13 = fdate('2013-02-01'); d14 = fdate('2014-05-01')
a0 = str(D[0].date())
W = {
    'select_ex_IFRS': [(a0, str(d13.date())), (str(d14.date()), '2019-04-30')],
    'valid_2019-05_2026-10': [('2019-05-01', '2026-10-31')],
    'y2024-01_2025-12': [('2024-01-01', '2025-12-31')],
    'A_clean_to_2013-01sig': [(a0, str(d13.date()))],
    'B_survivor_2014-05_2019-04': [(str(d14.date()), '2019-04-30')],
    'IFRS_block_0050': [(str(d13.date()), str(d14.date()))],
    'valid_survivorPanel_2019-05_2023-12': [('2019-05-01', '2023-12-31')],
    'r21_select_2021-03_2023-12': [('2021-03-01', '2023-12-31')],
    'r21_valid_2024-01_2026-10': [('2024-01-01', '2026-10-31')],
    'valid_officialPx_only': [('2019-05-01', '2024-02-14'), ('2025-09-24', '2026-10-31')],
    'full_ex_IFRS': [(a0, str(d13.date())), (str(d14.date()), '2026-10-31')],
}
res['windows'] = W
t0 = pos(D[0]); t1 = pos(D[-1])
navs = {}
_, pools1, info0, _ = build('A')
EW, _ = sim_w(pools1, 0.0, 0.0, t0, t1); navs['EW'] = EW
res['EW'] = {'windows': {k: daily_stats(EW, s, E50) for k, s in W.items()}, 'yearly': yearly(EW),
             'pool_by_year_mean': {int(y): round(float(g.pool.mean()), 1) for y, g in info0.groupby(pd.DatetimeIndex(info0.d).year)}}
res['0050_yearly'] = yearly(E50[(E50.index >= D[0]) & (E50.index <= D[-1])])
FL = pd.read_parquet(L.P / 'flag.parquet').reindex(index=IDX, columns=COLS).to_numpy()
exs_all = None

def event_study(info, pools, hold):
    b, s = L.cost_rates(0.001)
    exs = list(info.ex) + [pos(D[-1])]
    rows = []
    for k in range(len(info)):
        if k + hold >= len(exs): continue
        e0, e1 = exs[k], exs[k + hold]
        p0 = L.AO[e0]; p1 = L.AC[e1] if e1 == pos(D[-1]) else L.AO[e1]
        rr = p1 / p0 - 1
        pl = pools[e0][0]; pl = pl[pl != J50]
        if len(pl) == 0 or not info.picks[k]: continue
        ewr = np.nanmean(rr[pl])
        for c in info.picks[k]:
            j = L.CI[c]
            if not np.isfinite(rr[j]): continue
            net = (1 + rr[j]) * (1 - s) / (1 + b) - 1
            rows.append({'d': info.d[k], 'code': c, 'ret': rr[j], 'net': net, 'abn': net - ewr})
    evd = pd.DataFrame(rows); out = {}
    if not len(evd): return out, evd
    for kname, segs in W.items():
        m = np.zeros(len(evd), bool)
        for a, b_ in segs: m |= (evd.d >= pd.Timestamp(a)) & (evd.d < pd.Timestamp(b_))
        x = evd[m]
        if len(x) < 10: continue
        g = x.groupby('d').abn.mean()
        out[kname] = {'n_events': int(len(x)), 'n_months': int(len(g)), 'mean_net_ret_pct': round(float(x.net.mean() * 100), 2),
                      'mean_abn_vs_poolEW_pct': round(float(x.abn.mean() * 100), 2), 'month_cluster_mean_abn_pct': round(float(g.mean() * 100), 2),
                      't_month_cluster': L.tstat(g), 'hit_abn': round(float((x.abn > 0).mean()), 3)}
    return out, evd

def verdict(w):
    def ok(x): return x is not None and x.get('t_vs_EW_m', -9) >= 2 and x['CAGR'] >= x['0050_CAGR']
    a, b = ok(w['select_ex_IFRS']), ok(w['valid_2019-05_2026-10'])
    return {'select_pass': a, 'valid_pass': b, 'verdict': '可用' if a and b else ('待觀察' if a or b else '不可用')}

for ver in ('A', 'B', 'C'):
    for hold in (1, 3):
        key = f'{ver}_H{hold}'
        ev, pools, info, nblk = build(ver, hold)
        o = {'skipped_limit_up_total': int(nblk),
             'n_pick_by_year_mean': {int(y): round(float(g.n_pick.mean()), 1) for y, g in info.groupby(pd.DatetimeIndex(info.d).year)},
             'months_lt5_picks_select': int(((info.n_pick < 5) & ~ifrs_m[:len(info)] & (info.d < pd.Timestamp('2019-05-01'))).sum()),
             'months_lt5_picks_valid': int(((info.n_pick < 5) & (info.d >= pd.Timestamp('2019-05-01'))).sum()),
             'months_zero_picks_ex_IFRS': int(((info.n_pick == 0) & ~ifrs_m[:len(info)]).sum())}
        for cost in ('gross', 0.001, 'stress'):
            b, s = (0.0, 0.0) if cost == 'gross' else L.cost_rates(cost)
            nav, tov = sim_w(ev, b, s, t0, t1); navs[f'{key}_{cost}'] = nav
            o[f'cost_{cost}'] = {'turnover_per_rebalance': round(tov, 3), 'windows': {k: daily_stats(nav, s_, E50, EW) for k, s_ in W.items()}, 'yearly_pct': yearly(nav)}
        o['verdict_0.001'] = verdict(o['cost_0.001']['windows'])
        o['verdict_stress'] = verdict(o['cost_stress']['windows'])
        o['verdict_gross'] = verdict(o['cost_gross']['windows'])
        es, evd = event_study(info, pools, hold); o['event_study'] = es
        if hold == 1:
            big = evd[evd.ret > 1.0]
            o['events_ret_gt_100pct'] = big.assign(d=big.d.astype(str)).round(3).to_dict('records')[:20]
            o['n_events_ret_gt_100pct'] = int(len(big))
            exs = list(info.ex) + [pos(D[-1])]; nfl = 0; ntot = 0
            for k in range(len(info)):
                e0, e1 = exs[k], min(exs[k + 1], len(IDX) - 1)
                for c in info.picks[k]:
                    j = L.CI[c]; ntot += 1
                    if np.isin(FL[e0:e1 + 1, j], [3, 4, 5]).any(): nfl += 1
            o['picks_with_flag345_in_hold'] = {'n': nfl, 'of': ntot}
            o['last_signal'] = {'d': str(info.d.iloc[-1].date()), 'n': len(info.picks.iloc[-1]), 'picks': info.picks.iloc[-1]}
            # 剔除 IFRS 時若不剔除的訊號數
            sraw = {'A': SIG_A, 'B': SIG_B, 'C': SIG_C0}[ver]
            o['raw_signal_count_by_t_IFRS_window'] = {str(p): int((sraw.loc[p] & AVAIL.loc[p]).sum()) for p in pd.period_range('2012-07', '2014-08', freq='M')}
            o['raw_signal_share_by_year'] = {int(y): round(float((sraw.loc[g] & AVAIL.loc[g]).sum().sum() / max(AVAIL.loc[g].sum().sum(), 1)), 3)
                                             for y, g in pd.Series(sraw.index, index=sraw.index).groupby(sraw.index.year) if AVAIL.loc[g].sum().sum() > 0}
            pd.to_pickle(info, HERE / f'_info_{ver}.pkl')
            res[f'{key}_info_tail'] = info.tail(6).assign(d=info.d.tail(6).astype(str)).drop(columns='picks').to_dict('records')
        res[key] = o
        w = o['cost_0.001']['windows']
        print(key, {k: (w[k]['CAGR'], w[k]['maxDD'], w[k].get('t_vs_0050_m'), w[k].get('t_vs_EW_m'), w[k].get('0050_CAGR'), w[k].get('EW_CAGR')) for k in ('select_ex_IFRS', 'valid_2019-05_2026-10', 'y2024-01_2025-12', 'A_clean_to_2013-01sig') if w.get(k)}, o['verdict_0.001']['verdict'], round(time.time() - T0), flush=True)

# 整體判定
rank = {'可用': 2, '待觀察': 1, '不可用': 0}
vs = {v: res[f'{v}_H1']['verdict_0.001']['verdict'] for v in ('A', 'B', 'C')}
best = max(vs, key=lambda v: (rank[vs[v]], res[f'{v}_H1']['cost_0.001']['windows']['select_ex_IFRS']['t_vs_EW_m']))
bw = res[f'{best}_H1']['cost_0.001']['windows']
res['overall'] = {'per_version': vs, 'best_version': best, 'overall_verdict': vs[best],
                  'multiple_testing_flag': bool(vs[best] != '不可用' and min(bw['select_ex_IFRS']['t_vs_EW_m'], bw['valid_2019-05_2026-10']['t_vs_EW_m']) < 2.4)}

# ---- ann_date 檢查與 rev_hist 比對 ----
rh = pd.read_csv(XR / 'rev_hist.csv.gz', dtype={'code': str})
ad = rh[(rh.ym == '2026-09') & rh.ann_date.notna()].set_index('code')['ann_date']
chk = {}
for ver in ('A', 'B', 'C'):
    info = pd.read_pickle(HERE / f'_info_{ver}.pkl'); last = info.iloc[-1]
    dsig = str(last.d.date()); pk = last.picks; a = ad.reindex(pk)
    chk[ver] = {'signal_date': dsig, 'n_picks': len(pk), 'n_with_ann_date': int(a.notna().sum()), 'n_ann_after_signal_date': int((a > dsig).sum()), 'late_codes': list(a[a > dsig].index)}
res['ann_date_check_2026-09'] = chk
rh['p'] = pd.PeriodIndex(rh['ym'], freq='M')
mrg = rh.merge(d[['p', 'code', 'revenue_k_twd']], on=['p', 'code'], how='inner')
rel = (mrg.revenue / mrg.revenue_k_twd - 1).abs()
res['rev_hist_vs_long'] = {'n': int(len(mrg)), 'share_within_0.1pct': round(float((rel < 1e-3).mean()), 4)}
# 版本間重疊（入選股相同比例）
ov = {}
iA, iB, iC = (pd.read_pickle(HERE / f'_info_{v}.pkl') for v in ('A', 'B', 'C'))
for (na_, x), (nb2, y) in ((('A', iA), ('B', iB)), (('B', iB), ('C', iC)), (('A', iA), ('C', iC))):
    sh = [len(set(p) & set(q)) / len(p) for p, q in zip(x.picks, y.picks) if len(p)]
    ov[f'{na_}_in_{nb2}'] = round(float(np.mean(sh)), 3)
res['overlap_share'] = ov
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
pd.DataFrame(navs).to_parquet(HERE / 'nav.parquet')
print('overall', res['overall'], 'done', round(time.time() - T0))
