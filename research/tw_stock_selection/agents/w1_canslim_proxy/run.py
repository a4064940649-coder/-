#!/usr/bin/env python3
"""CAN SLIM 代理版（C、A 盈餘成長＋N 接近新高＋L 強勢＋I 法人＋M 大盤）— 可成交回測（上市，2022-09～2026-10）。
執行：python3 -I run.py。PREREG 先寫入 results.json 才開始計算。
成交、成本、漲停判定、60 日均值（剔除稀疏日）沿用 momentum_family/mlib.py → exec_rebuild/exlib.py（只讀取）。
引擎 sim() 是 exlib.sim_portfolio 的延伸（支援現金、個股停損），以 exlib 逐日比對驗證（無停損、無現金時應完全相同）。"""
import json, math, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'momentum_family'))
import mlib as M  # noqa: E402
L = M.L
T0 = time.time()
XR = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras')

PREREG = {
    '寫於看結果前': True,
    '來源規則': 'stockfeel《笑傲股市》CANSLIM 起手式、工商時報 20220111700417-431001、moomoo CANSLIM 教學：C 當季盈餘成長、A 年度盈餘成長、N 新高、S（不測）、L 領導股、I 法人認同、M 大盤向上；個股從買價跌 7～8% 停損。',
    '代理定義（照任務逐字）': {
        'EPS_ttm': 'EPS_ttm(m) = 原始收盤 close_raw(d_m) ÷ PE(m)，PE 取 pe_5y（TWSE 每月第一個交易日、以當日收盤計算的本益比，近四季 EPS）。PE 空值（虧損）→ EPS 視為非正。',
        'C': 'EPS_ttm(m)/EPS_ttm(m−3) − 1 ≥ 5%，且 EPS_ttm(m) > 0、EPS_ttm(m−3) > 0（基期為負無法算成長率，最常見作法是要求正值）',
        'A': 'EPS_ttm(m)/EPS_ttm(m−12) − 1 ≥ 25%，且 EPS_ttm(m−12) > 0',
        'N': 'adj(d) ≥ 0.85 × d 之前 365 個日曆日（含 d）的最高含息還原收盤；剔除有值檔數 < 300 的日子後計算，窗口至少 200 個有值交易日',
        'L': '12 個月報酬 = adj(d)/adj(d−365 日曆日，取當日或之前最近有值) − 1；在當月流動性股池（見下）內取百分位（pct rank），≥ 0.80',
        'I': 'd（含）往前 20 個法人交易日（inst_5y＋inst_hist 自己的日曆）外資＋投信淨買超合計 > 0；當日無紀錄視為 0（法人資料 d 盤後公布，d+1 開盤可用）',
        'M': '加權指數收盤(d) > 加權指數 200 日簡單均線(d)（指數自己的交易日曆）；否則當月全部空手（現金，報酬 0）',
    },
    '訊號日 d 與成交': '每月 d = pe_5y 該月日期（=每月第一個交易日；唯一例外 2026-10 資料日期為 10-08，照資料用）。d 收盤後算訊號 → d+1 開盤成交（BRIEF：盤後資料最早隔日開盤）。',
    '股池（同池等權也用它）': '價格面板上市股、排除 91xx 存託憑證；d 收盤 60 日均成交值 ≥ NT$1,000 萬（mlib.V60，剔除稀疏日後滾動）且原始收盤 ≥ 10 元；d 當月 pe_5y 有該股紀錄（含 PE 空值者）；d+1 有開盤價與收盤。不設前 300 名、不設處置濾網（教學沒有）。',
    '上櫃': '價格面板只有上市股；上櫃日 K 只有 2025-09-24 起（history.csv.gz），無法涵蓋 2022-09 起的訊號 → 只測上市，上櫃從缺（偏離規則，FINDINGS 註明）。',
    '入選與權重': 'C、A、N、L、I、M 全部成立才入選；d+1 開盤把整個組合調成所有入選股等權（漂移後權重計周轉）；入選 0 檔或 M 不成立 → 全部現金。每月換股：上月持股若本月未入選即賣出。',
    '停損': '每個持有股以「進場價」= 進場當天含息還原開盤價為基準（連續入選續抱者保留原進場價，不因月度調權重而重設）；任一交易日含息還原收盤 ≤ 0.92 × 進場價 → 下一個交易日開盤賣出（該日須有開盤價；開盤跌停（跳空 ≤ −9.4%）延到下一個非跌停開盤日）。停損賣出的資金留現金到下次換股日。d 收盤觸發停損的股票在 d+1 換股時一併賣出、該月不再買入；停損後下個月若再入選可重新買入（新進場價）。',
    '買不到': 'd+1 開盤漲停（mlib.GAP_UP，跳空 ≥ 漲停 −0.6%）且未持有 → 視為買不到、不遞補（規則全買，無排名可遞補）；已持有者續抱並調權重。',
    '缺漏月': '2024-01-02、2024-02-01 不在價格面板（2024-01-02～2024-02-14 缺漏）→ 2024-01、2024-02 不換股（持股延到 2024-03-04 換股，停損於 2024-02-15 起恢復檢查）。EPS 推算：這兩個月沒有收盤，改用前一個有收盤的月份（2023-12-01）的 EPS（TWSE PE 在 2023-11-15～2024-03-31 都用同一組至 2023Q3 四季 EPS；當月 PE 須 > 0；同 w1_pe_river_low 的修補）；影響 2024-04、2024-05 的 C 與 2025-01、2025-02 的 A。',
    '期間': {'說明': 'pe_5y 從 2021-09 起，A 要 12 個月 → 第一個訊號 2022-09-01。依 BRIEF 2021 起資料：挑選 2021～2023（實際 2022-09～2023-12，16 個換股月）、驗證 2024-01～2026-10；另報 2024-01～2025-12、全期、官方價格段（2025-09-24 起）。',
             '挑選期': ['2022-09-01', '2023-12-31'], '驗證期': ['2024-01-01', '2026-10-31'], '另報_2024_2025': ['2024-01-01', '2025-12-31'], '全期': ['2022-09-01', '2026-10-31'],
             '官方段': ['2025-09-24', '2026-10-31']},
    '成本': 'exlib.cost_rates：買 0.0855%+滑價、賣 0.0855%+稅 0.3%+滑價；主 = 滑價 0.1%（來回約 0.67%）；壓力 = 買 0.5%+賣 0.5%（來回 1%）；另報未扣成本。停損賣出同樣扣賣出成本。',
    '對手': '0050 含息（日頻）；同池等權 = 同股池全部等權、同引擎、d+1 開盤換股、不扣成本、永遠滿倉（判定用）；另報「同池等權＋M 擇時」（M 不成立時現金，只作拆解，不用於判定）。',
    '統計': '日頻 NAV：年化、日頻最大回撤、逐年；t = 訊號日 d 收盤→下一個 d 收盤的月窗超額（每月一個觀察值，月份即分群單位）對 0050 與同池等權（mlib.stats）。另報選股力診斷：入選月份中，入選股 d+1 開盤→下一換股日開盤的平均報酬（未扣成本、不含停損）減同池等權同期報酬，按月 t。',
    '判定': '主版本（滑價 0.1%、含停損與 M）：挑選期與驗證期都「對同池等權月窗 t ≥ 2 且年化 ≥ 0050 年化」→ 可用；只一段過 → 待觀察；否則不可用。驗證期有持股的月份 < 12 → 無法測試。因挑選期只有約 16 個月、可測期約 4 年，最佳結論上限為「待觀察」（任務資料備註）。敏感度（無停損、無 M、壓力成本、未扣成本）只報告、不改判定。',
    '敏感度（只報告）': ['no_stop：不停損', 'no_M：不看大盤（永遠依 C/A/N/L/I 選股）', 'no_stop_no_M'],
    '資料檢查（事前列出）': ['每月股池檔數、各條件通過檔數、入選檔數、開盤漲停擋下數、停損次數', '持有期 flag 3/4/5 與單月 > 50% 報酬的持股', 'src3 段（13:20 近似收盤）會讓 EPS 推算有 ~1% 雜訊：報官方段統計', '引擎與 exlib.sim_portfolio 比對', 'EPS 推算抽樣（月間比值）'],
}
res = {'PREREG': PREREG}
OUT = HERE / 'results.json'
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')

IDX, COLS, N, pos = L.IDX, L.COLS, L.N, L.pos
CI = {c: j for j, c in enumerate(COLS)}
IDXSET = set(IDX)

# ---------------- PE / EPS 月表 ----------------
pe = pd.read_csv(XR / 'pe_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
pe = pe[(pe['market'] == 'TWSE') & pe['code'].isin(COLS)]
PE = pe.pivot_table(index='date', columns='code', values='pe', aggfunc='last').sort_index().reindex(columns=COLS)
INPE = pe.assign(one=1).pivot_table(index='date', columns='code', values='one', aggfunc='last').reindex(index=PE.index, columns=COLS).notna().to_numpy()
MD = PE.index; Mn = len(MD)
Pn = PE.where(PE > 0).to_numpy()
Cd = L.C.reindex(MD).reindex(columns=COLS).to_numpy()
EPS = Cd / Pn
_miss = list(np.where(np.isnan(Cd).all(1))[0])
EPSf = EPS.copy()
for k in _miss:
    kk = k - 1
    while kk >= 0 and kk in _miss: kk -= 1
    EPSf[k] = np.where(np.isfinite(Pn[k]) & np.isfinite(EPS[kk]), EPS[kk], np.nan)
res['eps_fill_months'] = [str(MD[k].date()) for k in _miss]
def lag(a, n): return np.vstack([np.full((n, N), np.nan), a[:-n]])
E0 = EPSf; E3 = lag(EPSf, 3); E12 = lag(EPSf, 12)
with np.errstate(all='ignore'):
    COND_C = (E0 > 0) & (E3 > 0) & (E0 / E3 - 1 >= 0.05)
    COND_A = (E0 > 0) & (E12 > 0) & (E0 / E12 - 1 >= 0.25)
# EPS 推算檢查：同一季內（例如 6/1 vs 7/1，都用到 Q1 的四季 EPS）比值應接近 1
_chk = {}
for a_, b_ in (('2022-06-01', '2022-07-01'), ('2023-06-01', '2023-07-03'), ('2025-10-01', '2025-11-03')):
    ka, kb = MD.get_loc(pd.Timestamp(a_)), MD.get_loc(pd.Timestamp(b_))
    r = pd.Series(EPSf[kb] / EPSf[ka]).replace([np.inf, -np.inf], np.nan).dropna()
    _chk[f'{a_}->{b_}'] = {'n': int(len(r)), 'median': round(float(r.median()), 4), 'share_within_2pct': round(float(((r - 1).abs() < 0.02).mean()), 3)}
res['eps_ratio_check_same_quarter'] = _chk

# ---------------- 價格訊號（剔除稀疏日） ----------------
adj = L.adj
cnt = adj.notna().sum(axis=1)
DENSE = adj[cnt >= 300]
res['sparse_days_dropped'] = [str(d.date()) for d in adj.index[cnt < 300]]
HI = DENSE.rolling('365D', min_periods=200).max()
NOBS = DENSE.notna().rolling('365D').sum()
HI = HI.where(NOBS >= 200)
DA = DENSE.ffill()

def r12_at(d):
    p0 = DA.loc[:d - pd.Timedelta(days=365)]
    if len(p0) == 0: return np.full(N, np.nan)
    return (DENSE.loc[d] / p0.iloc[-1]).to_numpy() - 1

# 法人
i5 = pd.read_csv(XR / 'inst_5y.csv.gz', dtype={'code': str}, usecols=['date', 'code', 'foreign', 'trust'])
ih = pd.read_csv(XR / 'inst_hist.csv.gz', dtype={'code': str}, usecols=['date', 'code', 'foreign', 'trust'])
ih = ih[ih['date'] > i5['date'].max()]
inst = pd.concat([i5, ih]); inst = inst[inst['code'].isin(COLS)]
inst['ft'] = inst['foreign'].fillna(0) + inst['trust'].fillna(0)
FT = inst.pivot_table(index='date', columns='code', values='ft', aggfunc='sum').sort_index()
FT.index = pd.to_datetime(FT.index); FT = FT.reindex(columns=COLS).fillna(0.0)
FT20 = FT.rolling(20, min_periods=20).sum()
res['inst_range'] = [str(FT.index.min().date()), str(FT.index.max().date()), int(len(FT))]

# 大盤
tx = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date']).set_index('date')['close'].sort_index()
MA200 = tx.rolling(200).mean()

# ---------------- 每月條件 ----------------
FLAGN = pd.read_parquet(L.P / 'flag.parquet').reindex(index=IDX, columns=COLS).to_numpy()
SRCN = L.SRC.reindex(index=IDX, columns=COLS).to_numpy()
is_dr = np.array([c.startswith('91') for c in COLS])
K0 = 12
rows = []; QUAL = {}; QUAL_NOM = {}; POOL = {}; MOK = {}; EXI = {}
for k in range(K0, Mn):
    d = MD[k]
    if d not in IDXSET: continue
    i = pos(d)
    if i + 1 >= len(IDX): continue
    ex = i + 1
    v60 = M.V60.iloc[i].to_numpy(); px = L.C.iloc[i].to_numpy()
    pool = (v60 >= 1e7) & (px >= 10) & ~is_dr & INPE[k] & np.isfinite(L.OPEN_RAW[ex]) & np.isfinite(L.ADJ_RAW[ex])
    a_d = DENSE.loc[d].to_numpy(); hi = HI.loc[d].to_numpy()
    with np.errstate(all='ignore'):
        cN = np.isfinite(a_d) & np.isfinite(hi) & (a_d >= 0.85 * hi)
    r12 = r12_at(d)
    rk = pd.Series(np.where(pool & np.isfinite(r12), r12, np.nan)).rank(pct=True).to_numpy()
    cL = np.nan_to_num(rk) >= 0.80
    fi = FT20.loc[:d]
    cI = (fi.iloc[-1].to_numpy() > 0) if len(fi) and fi.index[-1] >= d - pd.Timedelta(days=5) else np.zeros(N, bool)
    txd = tx.loc[:d]; mad = MA200.loc[:d]
    mok = bool(txd.iloc[-1] > mad.iloc[-1])
    cC = COND_C[k]; cA = COND_A[k]
    q_nom = pool & cC & cA & cN & cL & cI
    QUAL_NOM[ex] = np.where(q_nom)[0]; QUAL[ex] = np.where(q_nom)[0] if mok else np.array([], int)
    POOL[ex] = np.where(pool)[0]; MOK[ex] = mok; EXI[ex] = d
    rows.append({'d': str(d.date()), 'pool': int(pool.sum()), 'C': int((pool & cC).sum()), 'A': int((pool & cA).sum()), 'N': int((pool & cN).sum()),
                 'L': int((pool & cL).sum()), 'I': int((pool & cI).sum()), 'CA': int((pool & cC & cA).sum()), 'CANLI': int(q_nom.sum()),
                 'M_ok': mok, 'taiex': round(float(txd.iloc[-1]), 1), 'ma200': round(float(mad.iloc[-1]), 1),
                 'picks': [COLS[j] for j in np.where(q_nom)[0]] if mok else []})
res['monthly_conditions'] = rows
SIGD = pd.DatetimeIndex([EXI[e] for e in sorted(EXI)])
EXS = sorted(EXI)
print('months', len(EXS), 'first', SIGD[0].date(), 'invested-eligible months', sum(MOK.values()))

GAPUP = M.GAP_UP
thr_dn = np.where(IDX < pd.Timestamp('2015-06-01'), 0.07, 0.10)[:, None] - L.LTOL
GAPDN = (L.AO_raw / L.AC_prev - 1) <= -thr_dn

def sim(events, buy, sell, t0, t1, stop=0.08, gapblock=True):
    """events: {exec_day: idx array}。回傳收盤 NAV、日誌。"""
    u = np.zeros(N); cash = 1.0; entry = np.full(N, np.nan); pending = set()
    nav = np.empty(t1 - t0 + 1); tov = []; nstop = 0; nblk = 0; nhold = []; stops = []
    for t in range(t0, t1 + 1):
        if t in events:
            q = events[t]; Pz = L.AO[t]
            held = set(np.where(u > 0)[0])
            S = []
            for j in q:
                if j in pending: continue
                if gapblock and GAPUP[t, j] and j not in held: nblk += 1; continue
                S.append(j)
            S = np.array(S, dtype=int)
            h = np.where(u > 0, u * np.nan_to_num(Pz), 0.0); Vt = h.sum() + cash
            if len(S) == 0:
                cost = sell * h.sum(); tov.append(h.sum() / Vt if Vt > 0 else 0)
                u = np.zeros(N); cash = Vt - cost; entry[:] = np.nan
            else:
                tgt = np.zeros(N); w = 1.0 / len(S); Vn = Vt
                for _ in range(6):
                    tgt[:] = 0; tgt[S] = w * Vn
                    cost = buy * np.clip(tgt - h, 0, None).sum() + sell * np.clip(h - tgt, 0, None).sum()
                    Vn = Vt - cost
                tov.append(np.clip(h - tgt, 0, None).sum() / Vt if Vt > 0 else 0)
                newe = np.full(N, np.nan)
                for j in S: newe[j] = entry[j] if (j in held and np.isfinite(entry[j])) else Pz[j]
                entry = newe
                u = np.zeros(N); u[S] = tgt[S] / Pz[S]; cash = 0.0
            pending = set(); nhold.append(len(S))
        elif pending:
            for j in list(pending):
                if u[j] > 0 and np.isfinite(L.OPEN_RAW[t, j]) and not GAPDN[t, j]:
                    cash += u[j] * L.AO[t, j] * (1 - sell); stops.append((COLS[j], str(IDX[t].date()), float(L.AO[t, j] / entry[j] - 1)))
                    u[j] = 0; entry[j] = np.nan; pending.discard(j); nstop += 1
                elif u[j] <= 0: pending.discard(j)
        nav[t - t0] = np.nansum(u * L.AC[t]) + cash
        if stop is not None:
            hj = np.where(u > 0)[0]
            trig = hj[np.isfinite(L.ADJ_RAW[t, hj]) & (L.AC[t, hj] <= (1 - stop) * entry[hj])]
            pending.update(int(j) for j in trig)
    log = {'turnover_per_rebalance': round(float(np.mean(tov[1:])), 3) if len(tov) > 1 else None, 'n_stops': nstop, 'gap_blocked': nblk,
           'avg_holdings_all_months': round(float(np.mean(nhold)), 2), 'months_invested': int(sum(1 for x in nhold if x > 0)),
           'avg_holdings_invested_months': round(float(np.mean([x for x in nhold if x > 0])), 2) if any(nhold) else 0,
           'stop_exit_ret_mean': round(float(np.mean([s[2] for s in stops])), 4) if stops else None}
    return pd.Series(nav, index=IDX[t0:t1 + 1]), log, stops

T_START = EXS[0] - 1; T_END = len(IDX) - 1
# 引擎驗證：無停損、無漲停阻擋、全池 → 應等於 exlib.sim_portfolio
ev_chk = {e: POOL[e] for e in EXS}
nA, _, _ = sim(ev_chk, *L.cost_rates(0.001), T_START, T_END, stop=None, gapblock=False)
nB, _ = L.sim_portfolio({e: ('open', POOL[e]) for e in EXS}, *L.cost_rates(0.001), T_START, T_END)
res['engine_check_max_abs_diff'] = float((nA - nB).abs().max())
print('engine diff', res['engine_check_max_abs_diff'])

EW, _, _ = sim({e: POOL[e] for e in EXS}, 0, 0, T_START, T_END, stop=None, gapblock=False)
EWM, _, _ = sim({e: (POOL[e] if MOK[e] else np.array([], int)) for e in EXS}, 0, 0, T_START, T_END, stop=None, gapblock=False)
E50 = L.E50
WIN = {'select_2022-09_2023-12': ('2022-09-01', '2023-12-31'), 'valid_2024-01_2026-10': ('2024-01-01', '2026-10-31'),
       'y2024_2025': ('2024-01-01', '2025-12-31'), 'full_2022-09_2026-10': ('2022-09-01', '2026-10-31'),
       'official_2025-09-24_2026-10': ('2025-09-24', '2026-10-31')}
res['same_pool_EW'] = {'yearly_pct': M.yearly(EW), 'windows': {w: M.stats(EW, a, b, SIGD, bench=E50) for w, (a, b) in WIN.items()}}
res['same_pool_EW_with_M_timing'] = {'yearly_pct': M.yearly(EWM), 'windows': {w: M.stats(EWM, a, b, SIGD, bench=E50, ew=EW) for w, (a, b) in WIN.items()}}
res['0050_yearly_pct'] = M.yearly(E50.loc[EW.index])

navs = {'EW': EW, 'EW_Mtiming': EWM}
VERS = {'main': dict(useM=True, stop=0.08), 'no_stop': dict(useM=True, stop=None), 'no_M': dict(useM=False, stop=0.08), 'no_stop_no_M': dict(useM=False, stop=None)}
for ver, cfg in VERS.items():
    ev = QUAL if cfg['useM'] else QUAL_NOM
    R = {}
    for cost in ('gross', 0.001, 'stress'):
        b, s = (0.0, 0.0) if cost == 'gross' else L.cost_rates(cost)
        nav, log, stops = sim(ev, b, s, T_START, T_END, stop=cfg['stop'])
        navs[f'{ver}_{cost}'] = nav
        R[f'cost_{cost}'] = {'log': log, 'yearly_pct': M.yearly(nav), 'windows': {w: M.stats(nav, a, bb, SIGD, bench=E50, ew=EW) for w, (a, bb) in WIN.items()}}
        if ver == 'main' and cost == 0.001: R['stops_list'] = stops
    res[ver] = R

# 選股力診斷（不含擇時、不含停損、未扣成本）：入選股 d+1 開盤 → 下一換股日開盤
def pick_power(qual):
    out = []
    for a_, b_ in zip(EXS[:-1], EXS[1:]):
        q = qual[a_]
        if len(q) == 0: continue
        p = POOL[a_]
        rq = L.AO[b_, q] / L.AO[a_, q] - 1; rp = L.AO[b_, p] / L.AO[a_, p] - 1
        out.append({'d': str(EXI[a_].date()), 'n': int(len(q)), 'pick': float(np.nanmean(rq)), 'pool': float(np.nanmean(rp)), 'ex': float(np.nanmean(rq) - np.nanmean(rp)),
                    'max_pick_ret': float(np.nanmax(rq)), 'flag345': int(sum(np.isin(FLAGN[a_:b_ + 1, j], [3, 4, 5]).any() for j in q))})
    return pd.DataFrame(out)
pp = pick_power(QUAL_NOM); pp.to_csv(HERE / 'pick_power_monthly.csv', index=False)
def pp_stats(df, a, b):
    e = df[(df['d'] >= a) & (df['d'] <= b)]
    if len(e) == 0: return None
    return {'months': int(len(e)), 'avg_n': round(e['n'].mean(), 2), 'pick_pct_m': round(e['pick'].mean() * 100, 2), 'pool_pct_m': round(e['pool'].mean() * 100, 2),
            'excess_pct_m': round(e['ex'].mean() * 100, 2), 't_month': L.tstat(e['ex']), 'hit': round(float((e['ex'] > 0).mean()), 3)}
res['pick_power_noM'] = {w: pp_stats(pp, a, b) for w, (a, b) in WIN.items()}
ppm = pp[pp['d'].isin([str(EXI[e].date()) for e in EXS if MOK[e]])]
res['pick_power_Mok_months'] = {w: pp_stats(ppm, a, b) for w, (a, b) in WIN.items()}
res['pick_power_extremes'] = {'n_pick_ret_gt_50pct_months': int((pp['max_pick_ret'] > 0.5).sum()), 'flag345_holdings': int(pp['flag345'].sum())}

# 判定
def passes(w):
    o = res['main']['cost_0.001']['windows'][w]
    if o is None: return None
    return bool((o['t_vs_EW_m'] >= 2) and (o['CAGR'] >= o['0050_CAGR']))
inv_valid = sum(1 for e in EXS if MOK[e] and len(QUAL[e]) > 0 and EXI[e] >= pd.Timestamp('2024-01-01'))
ps, pv = passes('select_2022-09_2023-12'), passes('valid_2024-01_2026-10')
if inv_valid < 12: verdict = '無法測試'
elif ps and pv: verdict = '待觀察（兩段都過，但依資料備註上限為待觀察）'
elif ps or pv: verdict = '待觀察'
else: verdict = '不可用'
res['verdict'] = {'select_pass': ps, 'valid_pass': pv, 'valid_months_with_picks': inv_valid, 'verdict': verdict}
pd.DataFrame(navs).to_parquet(HERE / 'nav.parquet')
res['runtime_s'] = round(time.time() - T0, 1)

def conv(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.bool_): return bool(o)
    return str(o)
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=conv), encoding='utf-8')
print('done', res['runtime_s'], res['verdict'])
