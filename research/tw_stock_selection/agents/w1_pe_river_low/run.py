#!/usr/bin/env python3
"""本益比河流圖下緣買進（自身 24 個月本益比第 20 百分位）— 可成交回測（上市，2023-09～2026-10）。
執行：python3 -I run.py。PREREG 先寫入 results.json 才開始計算。
成交、成本、漲停判定、60 日均值修補沿用 momentum_family/mlib.py → exec_rebuild/exlib.py（只讀取）。
引擎 sim() 與 exlib.sim_portfolio 同邏輯，另支援「無持股＝現金」並記錄換股日開盤淨值（事件基準用），以 exlib 逐日比對驗證。"""
import json, math, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'momentum_family'))
import mlib as M  # noqa: E402
L = M.L
T0 = time.time()
XR = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras')
SL = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv')

MEMORY = ['2408', '2344', '2337', '3006', '8271', '4967', '2451', '3260', '8299', '5351', '6531', '4973']
PANEL = ['2409', '3481', '6116', '8105', '3049', '2475']
PREREG = {
    '寫於看結果前': True,
    '來源規則': '玩股網 blog/203664/post/162、CMoney note 872197、鉅亨 hao post/200578：本益比河流圖跌到下緣（低本益比區）且獲利沒有衰退時買進，回到中軌（中位數）賣出。',
    '資料': 'pe_5y.csv.gz（每月第一個交易日的 TWSE/TPEx 本益比，2021-09～2026-10，共 62 個月；PE 以當日收盤計算——已用 close_raw/PE 檢查，當日收盤版本月間比值穩定）。虧損股 PE 為空值，視為非正。',
    '可測範圍': '價格面板只有上市股；上櫃日 K 只有 2025-09-24 起（history.csv.gz），無法涵蓋 2023-09 起的訊號 → 只測上市，上櫃從缺（偏離規則，FINDINGS 註明）。',
    '訊號月 m 與訊號日 d': 'd = pe_5y 該月日期（每月第一個交易日），d 收盤後才知道 PE → d+1 開盤成交。',
    '股池（同池等權也用它）': '面板上市股、排除 91xx 存託憑證；d 收盤 60 日均成交值 ≥ NT$1,000 萬（mlib.V60，剔除稀疏日後滾動）且收盤 ≥ 10 元；PE(m−11..m) 12 個月都 > 0；PE(m−24..m−1) 24 個月都 > 0（百分位窗口完整）；d+1 有開盤價。不設前 300 名、不設處置濾網（教學沒有）。',
    '剔除景氣循環（主版本）': {'航運': 'stock_list industry = 航運業（不在名單者代碼 26xx）', '鋼鐵': 'industry = 鋼鐵工業（不在名單者代碼 20xx）',
                     '記憶體（固定名單）': MEMORY, '面板（固定名單）': PANEL, '另報': '不剔除版（股池與同池等權都不剔除）'},
    '進場條件（月 m）': ['PE(m) ≤ PE(m−24..m−1) 的第 20 百分位（numpy linear 內插）', 'EPS_ttm(m) = close(d_m)/PE(m) ≥ EPS_ttm(m−12) = close(d_{m−12})/PE(m−12)', '在股池內'],
    '出場條件（月 m 檢查，d+1 開盤賣）': ['PE(m) ≥ PE(m−24..m−1) 中位數（窗口內正 PE 的 nanmedian，至少 12 個值；PE 為空＝虧損時不觸發此條）', '或持有滿 12 個月（進場月 m0，m ≥ m0+12）'],
    '其他細節': ['出場與進場同月判斷：先出場，若當月又符合進場條件則視為重新進場（不交易、12 個月計時重來）；持有中再次符合進場條件不重設計時。',
              '持有股掉出股池（流動性、PE 轉負）不強制出場，只依規則出場。',
              'd+1 開盤漲停（mlib.GAP_UP，含息跳空 ≥ 漲停 −0.6%）的新進場股視為買不到、不遞補（規則全買，無排名可遞補），訊號作廢；已持有者續抱。',
              '2024-01、2024-02 的 d+1 不在價格面板（2024-01-02～2024-02-14 缺漏）→ 這兩個月不換股（訊號作廢、出場延到 2024-03），持有計時照算。',
              '停牌／下市：沿用 exlib 以最後成交價計，下次換股時以該價賣出。'],
    '權重': '每個換股日 d+1 開盤把組合調回「所有在場持股等權」（漂移後權重計周轉）；無持股時放現金（報酬 0）。',
    '成本': 'exlib.cost_rates：買 0.0855%+滑價、賣 0.0855%+稅 0.3%+滑價；主 = 滑價 0.1%（來回約 0.67%）；壓力 = 買 0.5%+賣 0.5%（來回 1%）；另報未扣成本。',
    '對手': '0050 含息（日頻）；同池等權 = 同股池全部等權、同引擎、d+1 開盤換股、不扣成本；事件基準「上市等權」= 面板上市股（排除 91xx）中 d 收盤 60 日均值 ≥ 1,000 萬且 ≥ 10 元、d+1 有開盤的全部股票等權（不要求 PE），每月 d+1 開盤調整。',
    '統計（組合）': '日頻 NAV：年化、日頻最大回撤、逐年；t = 訊號日 d 收盤→下一 d 收盤月窗超額（每月一個觀察值）對 0050 與同池等權。',
    '統計（事件）': '每筆（股票, 進場月）d+1 開盤→出場 d+1 開盤（未出場者算到 2026-10-22 收盤，標記未結束）報酬扣主成本，減同期上市等權（換股日開盤淨值）報酬；按進場月份平均後算 t（持有期重疊，t 偏樂觀，僅供參考）。',
    '期間': {'說明': '24 個月窗口 → 第一個訊號 2023-09；依 BRIEF 2021 起資料切分 2021～2023 挑選、2024～2026 驗證，挑選期實際只剩 2023-09～2023-12（約 4 個月），不具判斷力；驗證期 2024-01～2026-10；另報 2024-01～2025-12、全期 2023-09～2026-10、官方價格段（排除 src3 2024-02-15～2025-09-23）。',
             '挑選期': ['2023-09-01', '2023-12-31'], '驗證期': ['2024-01-01', '2026-10-31'], '另報_2024_2025': ['2024-01-01', '2025-12-31'], '全期': ['2023-09-01', '2026-10-31']},
    '判定': '主版本（剔除循環、滑價 0.1%）：挑選期與驗證期都「對同池等權月窗 t ≥ 2 且年化 ≥ 0050」→ 可用；只一段過 → 待觀察；否則不可用。因挑選期僅約 4 個月、可測期約 3 年，最佳結論上限為「待觀察」。不剔除版、壓力成本、事件層級只報告、不改判定。',
    '資料修補（第一次執行後補記，非調參）': '第一次執行發現 2025-01、2025-02 訊號數為 0：EPS(m−12) 需要 2024-01-02、2024-02-01 收盤，但價格面板缺 2024-01-02～2024-02-14。改用前一個有收盤的月份（2023-12-01）推算的 EPS 代替（TWSE PE 在 2023-11-15～2024-03-31 都用同一組至 2023Q3 的四季 EPS，故推算值相同；當月 PE 須 > 0）。其他參數未動。修補前結果另存於 results.json 的 pre_fix_summary。',
    '資料檢查（事前列出）': ['每月股池檔數、訊號檔數、持有檔數、開盤漲停擋下數', '持有期 flag 3/4/5 與單筆 > 100% 報酬', 'src3 段（13:20 近似收盤）在 EPS 推算與 NAV 的影響：報官方段統計', '剔除循環股檔數', '引擎與 exlib.sim_portfolio 比對'],
}
res = {'PREREG': PREREG}
if (HERE / '_prefix_summary.json').exists():
    res['pre_fix_summary_slip0.1'] = json.loads((HERE / '_prefix_summary.json').read_text())
OUT = HERE / 'results.json'
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')

IDX, COLS, N, pos = L.IDX, L.COLS, L.N, L.pos
CI = {c: j for j, c in enumerate(COLS)}

# ---------------- PE 月表 ----------------
pe = pd.read_csv(XR / 'pe_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
pe = pe[(pe['market'] == 'TWSE') & pe['code'].isin(COLS)]
PE = pe.pivot_table(index='date', columns='code', values='pe', aggfunc='last').sort_index().reindex(columns=COLS)
MD = PE.index                                   # 62 個訊號日
PEp = PE.where(PE > 0)
Pn = PEp.to_numpy(); Mn = len(MD)
P20 = np.full((Mn, N), np.nan); MED = np.full((Mn, N), np.nan); FULL24 = np.zeros((Mn, N), bool)
for k in range(24, Mn):
    w = Pn[k - 24:k]
    full = np.isfinite(w).all(0); FULL24[k] = full
    with np.errstate(all='ignore'):
        P20[k, full] = np.percentile(w[:, full], 20, axis=0)
        cntw = np.isfinite(w).sum(0)
        med = np.nanmedian(np.where(cntw >= 12, w, np.nan), axis=0)
    med[cntw < 12] = np.nan; MED[k] = med
POS12 = np.zeros((Mn, N), bool)
for k in range(11, Mn): POS12[k] = np.isfinite(Pn[k - 11:k + 1]).all(0)
Cd = L.C.reindex(MD).reindex(columns=COLS).to_numpy()   # d 在面板缺漏時為 NaN
EPS = Cd / Pn
# 資料修補：2024-01-02、2024-02-01 不在價格面板 → 這兩月 EPS 無法推算，改用前一個有收盤的月份（2023-12-01）。
# 依據：TWSE 本益比用最近四季 EPS，2023-11-15～2024-03-31 間都是同一組（至 2023Q3）四季，推算 EPS 相同。
_miss = np.where(np.isnan(Cd).all(1))[0]
EPSf = EPS.copy()
for k in _miss:
    kk = k - 1
    while kk >= 0 and kk in set(_miss): kk -= 1
    EPSf[k] = np.where(np.isfinite(Pn[k]) & np.isfinite(EPS[kk]), EPS[kk], np.nan)
EPS12 = np.vstack([np.full((12, N), np.nan), EPSf[:-12]])
SIG = (Pn <= P20) & (EPS >= EPS12) & np.isfinite(EPS) & np.isfinite(EPS12)
EXITPE = np.isfinite(Pn) & np.isfinite(MED) & (Pn >= MED)

# ---------------- 股池 ----------------
sl = pd.read_csv(SL, dtype={'code': str}).drop_duplicates('code').set_index('code')
indu = pd.Series({c: (sl.at[c, 'industry'] if c in sl.index else None) for c in COLS})
is_ship = np.array([(indu[c] == '航運業') or (indu[c] is None and c.startswith('26')) for c in COLS])
is_steel = np.array([(indu[c] == '鋼鐵工業') or (indu[c] is None and c.startswith('20')) for c in COLS])
is_cyc = is_ship | is_steel | np.isin(np.array(COLS), MEMORY + PANEL)
is_dr = np.array([c.startswith('91') for c in COLS])
inpanel = np.array([d in set(IDX) for d in MD])
LIQ = np.zeros((Mn, N), bool); TRAD = np.zeros((Mn, N), bool); EXI = np.full(Mn, -1)
for k, d in enumerate(MD):
    if not inpanel[k]: continue
    i = pos(d)
    if i + 1 >= len(IDX): continue
    EXI[k] = i + 1
    v60 = M.V60.iloc[i].to_numpy(); px = L.C.iloc[i].to_numpy()
    LIQ[k] = (v60 >= 1e7) & (px >= 10) & ~is_dr
    TRAD[k] = np.isfinite(L.OPEN_RAW[i + 1]) & np.isfinite(L.ADJ_RAW[i + 1])
MKT = LIQ & TRAD                                  # 上市等權基準股池
POOL_ALL = MKT & POS12 & FULL24
POOL_EX = POOL_ALL & ~is_cyc[None, :]
GAP = M.GAP_UP
K0 = 24                                           # 第一個可算訊號的月份（2023-09）
print('first signal month', MD[K0].date(), 'skipped months', [str(MD[k].date()) for k in range(K0, Mn) if EXI[k] < 0])

def build(pool):
    """回傳 events {exec_day: ('open', idx)}、事件清單、診斷。"""
    held = {}                                      # j -> entry month k
    events, trades, diag = {}, [], []
    for k in range(K0, Mn):
        if EXI[k] < 0: continue
        ex = EXI[k]
        sig = SIG[k] & pool[k]
        exits = [j for j, k0 in held.items() if EXITPE[k, j] or (k - k0 >= 12)]
        nblk = 0; new = []
        for j in exits:
            k0 = held.pop(j)
            if sig[j] and k - k0 >= 12 and not EXITPE[k, j]:
                held[j] = k; trades.append((j, k0, k, 'reentry12')); new.append(j); continue
            trades.append((j, k0, k, 'pe_med' if EXITPE[k, j] else 'hold12'))
        for j in np.where(sig)[0]:
            if j in held: continue
            if GAP[ex, j]: nblk += 1; continue
            held[j] = k; new.append(j)
        events[ex] = ('open', np.array(sorted(held), dtype=int))
        diag.append({'d': str(MD[k].date()), 'pool': int(pool[k].sum()), 'signals': int(sig.sum()), 'new': len(new), 'exits': len(exits),
                     'held': len(held), 'gap_blocked': nblk})
    for j, k0 in held.items(): trades.append((j, k0, None, 'open_end'))
    return events, trades, diag

def sim(events, buy, sell, t0, t1):
    """exlib.sim_portfolio 同邏輯；空持股 = 現金；回傳收盤 NAV、換股日開盤淨值（調整前、未扣成本）、平均周轉。"""
    u = np.zeros(N); cash = 1.0; nav = np.empty(t1 - t0 + 1); tov = []; vopen = {}
    for t in range(t0, t1 + 1):
        if t in events:
            mode, pk = events[t]
            Pz = L.AO[t]
            h = np.where(u > 0, u * np.nan_to_num(Pz), 0.0)
            Vt = h.sum() + cash; vopen[t] = Vt
            if len(pk) == 0:
                cost = sell * h.sum(); tov.append(h.sum() / Vt if Vt > 0 else 0)
                u = np.zeros(N); cash = Vt - cost
            else:
                tgt = np.zeros(N); w = 1.0 / len(pk); Vn = Vt
                for _ in range(6):
                    tgt[:] = 0; tgt[pk] = w * Vn
                    cost = buy * np.clip(tgt - h, 0, None).sum() + sell * np.clip(h - tgt, 0, None).sum()
                    Vn = Vt - cost
                tov.append(np.clip(h - tgt, 0, None).sum() / Vt if Vt > 0 else 0)
                u = np.zeros(N); u[pk] = tgt[pk] / Pz[pk]; cash = 0.0
        nav[t - t0] = np.nansum(u * L.AC[t]) + cash
    return pd.Series(nav, index=IDX[t0:t1 + 1]), vopen, float(np.mean(tov[1:])) if len(tov) > 1 else float('nan')

T_START = EXI[K0] - 1; T_END = len(IDX) - 1
SIGD = pd.DatetimeIndex([MD[k] for k in range(K0, Mn) if EXI[k] >= 0])   # 月窗端點（d 收盤）

def pool_events(pool):
    return {EXI[k]: ('open', np.where(pool[k])[0]) for k in range(K0, Mn) if EXI[k] >= 0}

# 引擎驗證
ev_chk = pool_events(POOL_EX)
nA, _, _ = sim(ev_chk, *L.cost_rates(0.001), T_START, T_END)
nB, _ = L.sim_portfolio(ev_chk, *L.cost_rates(0.001), T_START, T_END)
res['engine_check_max_abs_diff'] = float((nA - nB).abs().max())
print('engine diff', res['engine_check_max_abs_diff'])

MKT_NAV, MKT_OPEN, _ = sim(pool_events(MKT), 0, 0, T_START, T_END)
E50 = L.E50

WIN = {'select_2023-09_2023-12': ('2023-09-01', '2023-12-31'), 'valid_2024-01_2026-10': ('2024-01-01', '2026-10-31'),
       'y2024_2025': ('2024-01-01', '2025-12-31'), 'full_2023-09_2026-10': ('2023-09-01', '2026-10-31'),
       'official_2025-09-24_2026-10': ('2025-09-24', '2026-10-31')}

def stats(nav, a, b, ew):
    o = M.stats(nav, a, b, SIGD, bench=E50, ew=ew)
    return o

FLAG = pd.read_parquet(L.P / 'flag.parquet').reindex(index=IDX, columns=COLS).to_numpy()
SRCN = L.SRC.reindex(index=IDX, columns=COLS).to_numpy()
b_main, s_main = L.cost_rates(0.001)

def event_table(trades):
    rows = []
    for j, k0, k1, why in trades:
        ti = EXI[k0]; to = EXI[k1] if k1 is not None else T_END
        p0 = L.AO[ti, j]; p1 = L.AO[to, j] if k1 is not None else L.AC[to, j]
        if not (np.isfinite(p0) and np.isfinite(p1)): continue
        g = p1 / p0 - 1; net = p1 * (1 - s_main) / (p0 * (1 + b_main)) - 1
        mo = MKT_OPEN[ti]; m1 = MKT_OPEN[to] if k1 is not None else MKT_NAV.iloc[-1]
        bm = m1 / mo - 1
        fl = FLAG[ti:to + 1, j]
        rows.append({'code': COLS[j], 'entry_d': str(MD[k0].date()), 'exit_d': str(MD[k1].date()) if k1 is not None else None, 'why': why,
                     'gross': g, 'net': net, 'mkt': bm, 'abn': net - bm, 'days': int(to - ti),
                     'flag345': bool(np.isin(fl, [3, 4, 5]).any()), 'src3_share': float(np.mean(SRCN[ti:to + 1, j] == 3))})
    return pd.DataFrame(rows)

def event_stats(et, a=None, b=None):
    e = et.copy()
    if a: e = e[(e['entry_d'] >= a) & (e['entry_d'] <= b)]
    if len(e) == 0: return None
    g = e.groupby('entry_d')['abn'].mean()
    return {'n_events': int(len(e)), 'n_entry_months': int(len(g)), 'mean_net_pct': round(e['net'].mean() * 100, 2), 'median_net_pct': round(e['net'].median() * 100, 2),
            'mean_abn_vs_TWSE_EW_pct': round(e['abn'].mean() * 100, 2), 'hit_abn': round(float((e['abn'] > 0).mean()), 3),
            'mean_abn_by_month_pct': round(g.mean() * 100, 2), 't_abn_month_cluster': L.tstat(g), 'mean_hold_days': round(e['days'].mean(), 1),
            'exit_reasons': e['why'].value_counts().to_dict(), 'open_end': int((e['why'] == 'open_end').sum())}

navs = {}
for ver, pool in (('ex_cyclical', POOL_EX), ('incl_cyclical', POOL_ALL)):
    events, trades, diag = build(pool)
    ew_nav, _, ew_tov = sim(pool_events(pool), 0, 0, T_START, T_END)
    navs[f'{ver}_EW'] = ew_nav
    R = {'diag_monthly': diag}
    for cost in ('gross', 0.001, 'stress'):
        b, s = (0.0, 0.0) if cost == 'gross' else L.cost_rates(cost)
        nav, _, tov = sim(events, b, s, T_START, T_END)
        navs[f'{ver}_{cost}'] = nav
        R[f'cost_{cost}'] = {'turnover_per_rebalance': round(tov, 3), 'windows': {w: stats(nav, a, bb, ew_nav) for w, (a, bb) in WIN.items()},
                             'yearly_pct': M.yearly(nav)}
    R['same_pool_EW'] = {'yearly_pct': M.yearly(ew_nav), 'windows': {w: M.stats(ew_nav, a, bb, SIGD, bench=E50) for w, (a, bb) in WIN.items()}}
    et = event_table(trades)
    et.to_csv(HERE / f'events_{ver}.csv.gz', index=False)
    R['events'] = {'all': event_stats(et), 'entry_2023': event_stats(et, '2023-01-01', '2023-12-31'), 'entry_2024_2026': event_stats(et, '2024-01-01', '2026-12-31'),
                   'entry_2024_2025': event_stats(et, '2024-01-01', '2025-12-31'),
                   'n_flag345': int(et['flag345'].sum()), 'n_gross_gt_100pct': int((et['gross'] > 1).sum()),
                   'top5_gross': et.nlargest(5, 'gross')[['code', 'entry_d', 'exit_d', 'gross', 'flag345']].round(3).to_dict('records'),
                   'industry_mix_top10': indu.reindex(et['code']).fillna('未列').value_counts().head(10).to_dict()}
    res[ver] = R
res['TWSE_EW_market_yearly_pct'] = M.yearly(MKT_NAV)
res['0050_yearly_pct'] = M.yearly(E50.loc[MKT_NAV.index])
res['n_cyclical_codes_in_panel'] = {'ship': int(is_ship.sum()), 'steel': int(is_steel.sum()), 'memory_panel': int(np.isin(np.array(COLS), MEMORY + PANEL).sum())}
pd.DataFrame(navs).to_parquet(HERE / 'nav.parquet')
res['runtime_s'] = round(time.time() - T0, 1)

def conv(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    return str(o)
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=conv), encoding='utf-8')
print('done', res['runtime_s'])
