#!/usr/bin/env python3
"""網路教學「PEG < 0.75 低估成長股」（以營收成長率代替 EPS 成長率）— 可成交回測（上市，2021-09～2026-10）。
執行：python3 -I run.py。PREREG 先寫入 results.json 才開始計算。
成交、成本、開盤漲停判定、60 日均值修補沿用 momentum_family/mlib.py → exec_rebuild/exlib.py（只讀取）。
sim() 另支援「盤中提前賣出放現金」與「無持股＝現金」，並以 exlib.sim_portfolio 逐日比對驗證（純換股事件）。"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'momentum_family'))
import mlib as M  # noqa: E402
L = M.L
T0 = time.time()
XR = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras')
SL = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv')
LONG = Path('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv')

VARIANTS = {  # 名稱: (PEG 進場上限, 成長率視窗 'ttm' 或 '3m', 提前賣出門檻或 None)
    'peg0.75_ttm': (0.75, 'ttm', 1.2), 'peg1.0_ttm': (1.0, 'ttm', 1.2),
    'peg0.75_3m': (0.75, '3m', 1.2), 'peg1.0_3m': (1.0, '3m', 1.2),
    'peg0.75_ttm_noearly': (0.75, 'ttm', None),   # 診斷：無提前賣出
}
MAIN = 'peg0.75_ttm'
NMAX = 30
PREREG = {
    '寫於看結果前': True,
    '來源規則': 'stockfeel 本益成長比 PEG 選股、Fugle blog peg-ratio、roo.cash peg-ratio-article、CMoney note 203668、samuel6788 substack peg：PEG = PE / 盈餘成長率(%)；< 0.75 低估、約 1 合理、> 1.2 高估。',
    '偏離（必須註明）': '教學原版用 EPS 成長率；季報 EPS 不在資料裡 → 以營收成長率代替。這不是原版 PEG，營收成長不等於盈餘成長（毛利率、業外、一次性損益都不在內）。',
    '資料': {'PE': 'pe_5y.csv.gz（TWSE 每月第一個交易日公布之 PE，2021-09-01～2026-10-08）。PE 空值＝虧損或無資料 → 不符合 PE > 0。',
             '營收': '長期月營收 tw_revenue.csv（NiklavsD/bottleneck-signals，上市 sii 優先去重，千元）算水準加總；rev_5y 的單月 yoy 只用於「控制單月營收年增率」與一致性檢查。'},
    '訊號日 d': '每月 12 日（含）之後的第一個交易日（月營收公告截止日 10 日之後；面板日曆）。2021-09～2026-10。',
    '訊號（月 m 的 d）': {'PE': '當月 pe_5y 月初 PE（pe 日期在 m 月、≤ d）',
                       'g_ttm（主）': '近 12 個月（m−12～m−1）營收合計 ÷ 前 12 個月（m−24～m−13）合計 − 1（%）；24 個月都要有值、前期合計 > 0',
                       'g_3m（變體）': '近 3 個月（m−3～m−1）合計 ÷ 去年同 3 個月（m−15～m−13）合計 − 1（%）',
                       '條件': 'PE > 0、g ≥ 10%、PEG = PE / g < 0.75（主）；變體 < 1'},
    '股池（同池等權也用它）': '面板上市股、排除 91xx 存託憑證、排除金融保險業（營收對金融業無意義；stock_list industry=金融保險業，不在名單者以 28xx 判定）；d 當月 pe_5y 有該股紀錄；d 收盤 60 日均成交值 ≥ NT$1,000 萬（mlib.V60，剔除稀疏日後滾動）且收盤 ≥ 10 元；d+1 有開盤價。不設前 300 名、不設處置濾網（教學沒有）。',
    '進出場': 'd 收盤後 → d+1 開盤成交。符合者依 PEG 由低到高取前 30 檔等權；不足 30 檔就全買；續抱者也調回等權（漂移後權重計周轉）。下個月 d 不再入選者 d+1 開盤賣出。無入選者放現金（報酬 0）。',
    '提前賣出': '持有期間每個交易日收盤算 PEG_t = PE_t / g，PE_t = 最近一次 pe_5y 的 PE × 收盤原始價_t ÷ 該 pe 日收盤原始價（即近四季 EPS 不變、用當日股價），g 用最近一次換股時的值（月營收一個月才更新一次）。PEG_t > 1.2 → t+1 開盤賣出，資金放現金到下次換股。換股日 d 當天不檢查（隔日本來就換股）。PE_t 無法計算（例如轉虧）不觸發。',
    '買不到': 'd+1 開盤漲停（mlib.GAP_UP，跳空 ≥ 漲停 −0.6%）且未持有的新股視為買不到，由 PEG 次低的符合者遞補（至多 30 檔）；已持有者續抱。提前賣出不模擬跌停賣不掉（exlib 沒有跌停判定）。',
    '面板缺漏': '2024-01 整月不在價格面板（2024-01-02～2024-02-14 缺漏）→ 該月不換股、也無法逐日檢查提前賣出，持股續抱到 2024-02-15。訊號日若營收 m−1 月覆蓋 < 80% 中位數則該月不換股（事前規則）。',
    '停牌／下市': '沿用 exlib：以最後成交價計，下次換股時以該價賣出。',
    '成本': 'exlib.cost_rates：買 0.0855%+滑價、賣 0.0855%+稅 0.3%+滑價；主 = 滑價 0.1%（來回約 0.67%）；壓力 = 買 0.5%+賣 0.5%（來回 1%）；另報未扣成本。提前賣出也收賣出成本。',
    '對手': '0050 含息（日頻）；同池等權 = 同股池全部等權、同引擎、每月 d+1 開盤換股、不扣成本（與 w4_buffett_roe_pe_screen 同口徑）。',
    '統計': '日頻 NAV：年化、日頻最大回撤、逐年；t = 訊號日 d 收盤 → 下一個 d 收盤的月窗超額（每月一個觀察值＝按月份分群）對 0050 與同池等權。',
    '期間': {'挑選期': ['2021-09-01', '2023-12-31'], '驗證期': ['2024-01-01', '2026-10-31'], '另報_2024_2025': ['2024-01-01', '2025-12-31'],
             '全期': ['2021-09-01', '2026-10-31'], '官方價格段': ['2025-09-24', '2026-10-31'],
             '說明': 'pe_5y 從 2021-09 起，價格類長期間無法做；依 BRIEF 2021 起資料切分。挑選期實際只有 2021-09～2023-12（約 28 個月窗），檢定力低。面板 2019–2023 只有存活股。'},
    '版本': {'主': 'PEG < 0.75、g = 近 12 月營收年增率、PEG > 1.2 提前賣、最多 30 檔、每月換股、滑價 0.1%。',
             '變體（全部報告）': 'PEG < 1（TTM）；PEG < 0.75（3 月）；PEG < 1（3 月）；診斷：主版無提前賣出。各版 × {未扣成本, 滑價 0.1%, 壓力}。'},
    '增量檢定（控制單月營收年增率）': {
        'FM 迴歸': '每月在股池內做橫斷面迴歸：股票 d+1 開盤 → 下次 d+1 開盤的未扣成本報酬 ~ 入選虛擬變數（主版前 30 檔，不扣漲停遞補） + 單月營收 yoy（rev_5y，m−1 月）百分位；另一式再加 E/P 百分位（虧損 E/P=0）。係數以月份為觀察值算 t，挑選期／驗證期分開。',
        '配對比較': '每月把股池依單月 yoy 分 10 組，入選股報酬減去「同 yoy 組股池等權報酬」，按月平均後算 t。'},
    '判定': '主版本、滑價 0.1%：挑選期與驗證期都「對同池等權月窗 t ≥ 2 且年化 ≥ 0050」→ 可用；只一段過 → 待觀察；否則不可用。變體、壓力成本只報告、不改判定、不用來挑參數。',
    '資料檢查（事前列出）': ['每月股池檔數、符合檔數、入選檔數、開盤漲停擋下數、提前賣出次數', '長期月營收與 rev_5y 營收水準一致性', 'g 極端值（基期過低）與 PE 極端值', '持有期 flag 3/4/5 檔次、單期毛報酬 > 50% 檔次', '產業分布、持股集中度', 'src3 段影響：報官方價格段', '引擎與 exlib.sim_portfolio 比對'],
}
res = {'PREREG': PREREG}
OUT = HERE / 'results.json'
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')

IDX, COLS, N, pos = L.IDX, L.COLS, L.N, L.pos
IDXSET = set(IDX)

# ---------------- 換股日曆 ----------------
_s = pd.Series(IDX, index=IDX)
SD = [g[g.dt.day >= 12].iloc[0] for _, g in _s.groupby([IDX.year, IDX.month]) if (g.dt.day >= 12).any()]
SD = [d for d in SD if pd.Timestamp('2021-09-01') <= d <= pd.Timestamp('2026-10-31')]
SD = [d for d in SD if pos(d) + 1 < len(IDX)]
SIGD = pd.DatetimeIndex(SD); Mn = len(SIGD)

# ---------------- pe_5y ----------------
pe = pd.read_csv(XR / 'pe_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
pe = pe[(pe['market'] == 'TWSE') & pe['code'].isin(COLS)]
PEw = pe.pivot_table(index='date', columns='code', values='pe', aggfunc='last').sort_index().reindex(columns=COLS)
HASw = pe.assign(one=1).pivot_table(index='date', columns='code', values='one', aggfunc='last').reindex(index=PEw.index, columns=COLS).notna()
PED = PEw.index
pe_of_month = {(d.year, d.month): d for d in PED}

# ---------------- 月營收 ----------------
lg = pd.read_csv(LONG, dtype={'code': str, 'market': str, 'name': str})
lg['p'] = pd.PeriodIndex(pd.to_datetime(lg['month']), freq='M')
lg = lg.sort_values(['p', 'market']).drop_duplicates(['p', 'code'], keep='first')
lg = lg[lg['code'].isin(COLS)]
REV = lg.pivot_table(index='p', columns='code', values='revenue_k_twd', aggfunc='last').sort_index()
REV = REV.reindex(pd.period_range('2018-01', REV.index.max(), freq='M'), columns=COLS)
S12 = REV.rolling(12, min_periods=12).sum(); S3 = REV.rolling(3, min_periods=3).sum()
G_TTM = ((S12 / S12.shift(12) - 1) * 100).where(S12.shift(12) > 0)
G_3M = ((S3 / S3.shift(12) - 1) * 100).where(S3.shift(12) > 0)
rv = pd.read_csv(XR / 'rev_5y.csv.gz', dtype={'code': str}); rv = rv[rv['code'].isin(COLS)]; rv['p'] = pd.PeriodIndex(rv['ym'], freq='M')
YOY1 = rv.pivot_table(index='p', columns='code', values='yoy', aggfunc='last').sort_index().reindex(columns=COLS)
RV5 = rv.pivot_table(index='p', columns='code', values='revenue', aggfunc='last').sort_index().reindex(columns=COLS)
_c = REV.reindex(RV5.index); _m = _c.notna() & RV5.notna()
_ratio = (RV5 / _c)[_m]
res['revenue_consistency_long_vs_rev5y'] = {'n_pairs': int(_m.to_numpy().sum()),
                                            'share_within_0.1pct': round(float((np.abs(_ratio - 1) < 0.001).to_numpy()[_m.to_numpy()].mean()), 4)}
rev_cnt = REV.notna().sum(axis=1)

# ---------------- 股池與訊號（月 × 股） ----------------
sl = pd.read_csv(SL, dtype={'code': str}).drop_duplicates('code').set_index('code')
indu = pd.Series({c: (sl.at[c, 'industry'] if c in sl.index else None) for c in COLS})
is_fin = np.array([(indu[c] == '金融保險業') or (indu[c] is None and c.startswith('28')) for c in COLS])
is_dr = np.array([c.startswith('91') for c in COLS])
CN_ff = L.C.ffill().to_numpy()

POOL = np.zeros((Mn, N), bool); EXI = np.full(Mn, -1); SIGI = np.array([pos(d) for d in SIGD])
PE_M = np.full((Mn, N), np.nan); GT = np.full((Mn, N), np.nan); G3 = np.full((Mn, N), np.nan); Y1 = np.full((Mn, N), np.nan)
PEDATE_I = np.full(Mn, -1); skip = []
for k, d in enumerate(SIGD):
    i = pos(d); pm = pd.Period(d, 'M') - 1
    pdte = pe_of_month.get((d.year, d.month))
    if pdte is None or pdte > d:
        skip.append((str(d.date()), 'no pe')); continue
    if rev_cnt.get(pm, 0) < 0.8 * rev_cnt.loc[pd.Period('2021-01', 'M'):].median():
        skip.append((str(d.date()), 'rev coverage')); continue
    EXI[k] = i + 1
    PEDATE_I[k] = int(np.searchsorted(IDX, pdte, side='right') - 1)   # pe 日（或之前最近交易日）在面板的位置
    PE_M[k] = PEw.loc[pdte].to_numpy(); GT[k] = G_TTM.loc[pm].to_numpy(); G3[k] = G_3M.loc[pm].to_numpy()
    if pm in YOY1.index: Y1[k] = YOY1.loc[pm].to_numpy()
    v60 = M.V60.iloc[i].to_numpy(); px = L.C.iloc[i].to_numpy()
    trad = np.isfinite(L.OPEN_RAW[i + 1]) & np.isfinite(L.ADJ_RAW[i + 1])
    POOL[k] = (v60 >= 1e7) & (px >= 10) & ~is_dr & ~is_fin & HASw.loc[pdte].to_numpy() & trad
res['skipped_months'] = skip
GAP = M.GAP_UP
with np.errstate(invalid='ignore', divide='ignore'):
    PEG_TTM = np.where((PE_M > 0) & (GT >= 10), PE_M / GT, np.nan)
    PEG_3M = np.where((PE_M > 0) & (G3 >= 10), PE_M / G3, np.nan)
# 每個交易日可用的「最近一次 pe」：EPS 隱含 = pe 日收盤 / PE（PE_t = 收盤_t / EPS）
EPS_IMPL = np.full((Mn, N), np.nan)
for k in range(Mn):
    if EXI[k] < 0: continue
    with np.errstate(invalid='ignore', divide='ignore'):
        EPS_IMPL[k] = np.where(PE_M[k] > 0, CN_ff[PEDATE_I[k]] / PE_M[k], np.nan)
# 所有 pe 日（含換股日之間的月初）的 EPS 隱含值，供逐日提前賣出用
PE_ALL_I = np.array([int(np.searchsorted(IDX, p, side='right') - 1) for p in PED])
with np.errstate(invalid='ignore', divide='ignore'):
    PEn_all = PEw.to_numpy(); EPS_ALL = np.where(PEn_all > 0, CN_ff[PE_ALL_I] / PEn_all, np.nan)


def build(var, nmax=None, excl=None):
    NMAX_ = NMAX if nmax is None else nmax
    thr, gw, early = VARIANTS[var]
    PEG = PEG_TTM if gw == 'ttm' else PEG_3M
    G = GT if gw == 'ttm' else G3
    held = set(); events, diag, picks_raw = {}, [], {}
    valid_k = [k for k in range(Mn) if EXI[k] >= 0]
    n_early = 0
    for kk, k in enumerate(valid_k):
        ex = EXI[k]
        peg = np.where(POOL[k] & (True if excl is None else ~excl), PEG[k], np.nan)
        q = np.where(np.isfinite(peg) & (peg < thr))[0]
        q = q[np.argsort(peg[q], kind='stable')]
        picks_raw[k] = q[:NMAX_]
        pick, blk = [], 0
        for j in q:
            if GAP[ex, j] and j not in held:
                blk += 1; continue
            pick.append(j)
            if len(pick) == NMAX_: break
        pick = np.array(sorted(pick), dtype=int)
        events[ex] = ('rebal', pick)
        cur = set(pick.tolist())
        n_e = 0
        # 提前賣出：ex 收盤起逐日檢查到下一個訊號日的前一日
        if early is not None and len(pick):
            nxt_sig = SIGI[valid_k[kk + 1]] if kk + 1 < len(valid_k) else len(IDX) - 1
            gk = G[k]
            for t in range(ex, nxt_sig):
                if t + 1 >= len(IDX): break
                pi = np.searchsorted(PE_ALL_I, t, side='right') - 1
                eps = EPS_ALL[pi]
                sells = []
                for j in list(cur):
                    if np.isfinite(L.ADJ_RAW[t, j]) and np.isfinite(eps[j]) and eps[j] > 0 and gk[j] > 0:
                        if (L.C.iat[t, j] / eps[j]) / gk[j] > early:
                            sells.append(j)
                if sells:
                    events.setdefault(t + 1, ('sell', []))
                    events[t + 1] = ('sell', list(events[t + 1][1]) + sells)
                    cur -= set(sells); n_e += len(sells)
        n_early += n_e
        with np.errstate(invalid='ignore'):
            diag.append({'d': str(SIGD[k].date()), 'pool': int(POOL[k].sum()), 'pe_pos_g10': int((POOL[k] & (PE_M[k] > 0) & (G[k] >= 10)).sum()),
                         'qualify': int(len(q)), 'held': int(len(pick)), 'new': int(len(set(pick.tolist()) - held)),
                         'sold': int(len(held - set(pick.tolist()))), 'gap_blocked': blk, 'early_sold': n_e,
                         'median_peg_held': round(float(np.median(peg[pick])), 3) if len(pick) else None})
        held = set(pick.tolist())
    return events, diag, picks_raw, n_early


def sim(events, buy, sell, t0, t1):
    u = np.zeros(N); cash = 1.0; nav = np.empty(t1 - t0 + 1); tov = []
    for t in range(t0, t1 + 1):
        if t in events:
            mode, pk = events[t]
            Pz = L.AO[t]
            if mode == 'sell':
                for j in pk:
                    if u[j] > 0:
                        v = u[j] * Pz[j]
                        if np.isfinite(v): cash += v * (1 - sell)
                        u[j] = 0.0
            else:
                h = np.where(u > 0, u * np.nan_to_num(Pz), 0.0)
                Vt = h.sum() + cash
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
    return pd.Series(nav, index=IDX[t0:t1 + 1]), float(np.mean(tov[1:])) if len(tov) > 1 else float('nan')


K0 = int(np.where(EXI >= 0)[0][0])
T_START = EXI[K0] - 1; T_END = len(IDX) - 1
SIGV = pd.DatetimeIndex([SIGD[k] for k in range(Mn) if EXI[k] >= 0])
pool_events = {EXI[k]: ('rebal', np.where(POOL[k])[0]) for k in range(Mn) if EXI[k] >= 0}
nA, _ = sim(pool_events, *L.cost_rates(0.001), T_START, T_END)
nB, _ = L.sim_portfolio({t: ('open', v[1]) for t, v in pool_events.items()}, *L.cost_rates(0.001), T_START, T_END)
res['engine_check_max_abs_diff'] = float((nA - nB).abs().max())
EW_NAV, _ = sim(pool_events, 0, 0, T_START, T_END)
E50 = L.E50
WIN = {'select_2021-09_2023-12': ('2021-09-01', '2023-12-31'), 'valid_2024-01_2026-10': ('2024-01-01', '2026-10-31'),
       'y2024_2025': ('2024-01-01', '2025-12-31'), 'full_2021-09_2026-10': ('2021-09-01', '2026-10-31'),
       'official_2025-09-24_2026-10': ('2025-09-24', '2026-10-31')}
FL = pd.read_parquet(L.P / 'flag.parquet').reindex(index=IDX, columns=COLS).to_numpy()
res['pool_size'] = {'mean': round(float(POOL[EXI >= 0].sum(1).mean()), 1), 'min': int(POOL[EXI >= 0].sum(1).min()), 'max': int(POOL[EXI >= 0].sum(1).max())}
res['n_rebalances'] = int((EXI >= 0).sum())

navs = {'same_pool_EW': EW_NAV}
PICKS = {}
for v in VARIANTS:
    events, diag, picks_raw, n_early = build(v)
    PICKS[v] = picks_raw
    R = {}
    for cost in ('gross', 0.001, 'stress'):
        b, s = (0.0, 0.0) if cost == 'gross' else L.cost_rates(cost)
        nav, tov = sim(events, b, s, T_START, T_END)
        navs[f'{v}|{cost}'] = nav
        R[f'cost_{cost}'] = {'turnover_per_rebalance': round(tov, 3), 'windows': {w: M.stats(nav, a, bb, SIGV, bench=E50, ew=EW_NAV) for w, (a, bb) in WIN.items()},
                             'yearly_pct': M.yearly(nav)}
    hc = np.array([dd['held'] for dd in diag]); qc = np.array([dd['qualify'] for dd in diag])
    R['holdings'] = {'mean': round(float(hc.mean()), 1), 'min': int(hc.min()), 'max': int(hc.max()), 'months_empty': int((hc == 0).sum()),
                     'qualify_mean': round(float(qc.mean()), 1), 'months_qualify_gt30': int((qc > NMAX).sum()),
                     'gap_blocked_total': int(sum(dd['gap_blocked'] for dd in diag)), 'early_sold_total': int(n_early),
                     'early_sold_share_of_positions': round(n_early / max(hc.sum(), 1), 3)}
    rebal = sorted(t for t, e in events.items() if e[0] == 'rebal')
    rows = []
    for a_, b_ in zip(rebal, rebal[1:] + [T_END]):
        for j in events[a_][1]:
            p0, p1 = L.AO[a_, j], (L.AO[b_, j] if b_ != T_END else L.AC[b_, j])
            rows.append({'code': COLS[j], 'exec': str(IDX[a_].date()), 'gross_hold_to_next': p1 / p0 - 1 if np.isfinite(p0) and np.isfinite(p1) else np.nan,
                         'flag345': bool(np.isin(FL[a_:b_ + 1, j], [3, 4, 5]).any())})
    hd = pd.DataFrame(rows)
    if len(hd):
        R['holding_detail'] = {'n_stock_periods': int(len(hd)), 'n_unique_codes': int(hd['code'].nunique()), 'n_flag345': int(hd['flag345'].sum()),
                               'n_gross_gt_50pct': int((hd['gross_hold_to_next'] > 0.5).sum()),
                               'industry_mix_top8': indu.reindex(hd['code']).fillna('未列').value_counts().head(8).to_dict(),
                               'most_held_top10': hd['code'].value_counts().head(10).to_dict()}
    if v == MAIN:
        R['diag_monthly'] = diag
        hd.to_csv(HERE / 'holdings_main.csv.gz', index=False)
    res[v] = R
res['same_pool_EW'] = {'yearly_pct': M.yearly(EW_NAV), 'windows': {w: M.stats(EW_NAV, a, bb, SIGV, bench=E50) for w, (a, bb) in WIN.items()}}
res['0050_yearly_pct'] = M.yearly(E50.loc[EW_NAV.index])
pd.DataFrame(navs).to_parquet(HERE / 'nav.parquet')

# ---------------- 訊號特性：PE、g 極端值 ----------------
mk = np.zeros((Mn, N), bool)
for k, q in PICKS[MAIN].items(): mk[k, q] = True
res['main_pick_characteristics'] = {'n': int(mk.sum()), 'median_pe': round(float(np.nanmedian(PE_M[mk])), 2), 'median_g_ttm': round(float(np.nanmedian(GT[mk])), 1),
                                    'p90_g_ttm': round(float(np.nanpercentile(GT[mk], 90)), 1), 'n_g_gt_100': int((GT[mk] > 100).sum()),
                                    'median_peg': round(float(np.nanmedian(PEG_TTM[mk])), 3), 'median_yoy1': round(float(np.nanmedian(Y1[mk])), 1),
                                    'pool_median_yoy1': round(float(np.nanmedian(Y1[POOL])), 1), 'pool_median_pe': round(float(np.nanmedian(np.where(PE_M > 0, PE_M, np.nan)[POOL])), 2)}

# ---------------- 增量檢定：控制單月營收年增率 ----------------
valid_k = [k for k in range(Mn) if EXI[k] >= 0]
FWD = np.full((Mn, N), np.nan)
for kk, k in enumerate(valid_k):
    a_ = EXI[k]; b_ = EXI[valid_k[kk + 1]] if kk + 1 < len(valid_k) else T_END
    p1 = L.AO[b_] if kk + 1 < len(valid_k) else L.AC[b_]
    with np.errstate(invalid='ignore', divide='ignore'):
        FWD[k] = p1 / L.AO[a_] - 1


def pct_rank(x, m):
    s = pd.Series(np.where(m, x, np.nan)); return s.rank(pct=True).to_numpy()


def incr(var):
    out = {}
    rows = []
    for k in valid_k:
        m = POOL[k] & np.isfinite(FWD[k]) & np.isfinite(Y1[k])
        if m.sum() < 50: continue
        dmy = np.zeros(N); dmy[PICKS[var][k]] = 1.0
        ry = pct_rank(Y1[k], m)
        ep = np.where(PE_M[k] > 0, 1 / PE_M[k], 0.0); rep = pct_rank(ep, m)
        y = FWD[k][m] - np.nanmean(FWD[k][m])
        rec = {'d': SIGD[k], 'n_pick': int(dmy[m].sum())}
        for nm, Xs in (('dummy_only', [dmy[m]]), ('dummy+yoy1', [dmy[m], ry[m]]), ('dummy+yoy1+ep', [dmy[m], ry[m], rep[m]])):
            if dmy[m].sum() == 0: rec[nm] = np.nan; continue
            X = np.column_stack([np.ones(m.sum())] + Xs)
            beta = np.linalg.lstsq(X, y, rcond=None)[0]; rec[nm] = beta[1]
        # 配對：單月 yoy 10 組
        dec = np.full(N, -1); dec[m] = np.minimum((ry[m] * 10).astype(int), 9)
        pk = [j for j in PICKS[var][k] if m[j]]
        if pk:
            grp_mean = {g: np.nanmean(FWD[k][m & (dec == g)]) for g in range(10)}
            rec['matched_diff'] = float(np.mean([FWD[k][j] - grp_mean[dec[j]] for j in pk]))
            rec['pick_minus_pool'] = float(np.mean(FWD[k][pk]) - np.nanmean(FWD[k][m]))
        else:
            rec['matched_diff'] = np.nan; rec['pick_minus_pool'] = np.nan
        rows.append(rec)
    df = pd.DataFrame(rows).set_index('d')
    for w, (a, b) in WIN.items():
        if w.startswith('official'): continue
        x = df[(df.index >= pd.Timestamp(a)) & (df.index <= pd.Timestamp(b))]
        o = {'n_months': int(len(x))}
        for c in ('pick_minus_pool', 'matched_diff', 'dummy_only', 'dummy+yoy1', 'dummy+yoy1+ep'):
            z = x[c].dropna(); o[c] = {'mean_pct_m': round(float(z.mean() * 100), 2), 't': L.tstat(z), 'n': int(len(z))}
        out[w] = o
    return out


res['incremental_vs_single_month_yoy'] = {'說明': '未扣成本、d+1 開盤 → 下次 d+1 開盤；入選＝前 30 檔（不含漲停遞補）；t 以月份為觀察值。dummy 係數＝控制後入選股的月超額。',
                                          **{v: incr(v) for v in ('peg0.75_ttm', 'peg1.0_ttm', 'peg0.75_3m')}}


# ---------------- 事後診斷（看完主結果後才加入；不改判定、不用來挑參數） ----------------
is_con = np.array([indu[c] == '建材營造業' for c in COLS])
post = {'說明': '看完主結果後加入：主版「取 PEG 最低 30 檔」使 PEG 門檻不起作用（每月符合檔數平均約 100 檔、入選者 PEG 中位數約 0.16），且偏向營收基期極低的營建、航運股。以下只為解釋原因，不改判定。'}
for nm, (v, nmax, excl) in {'peg0.75_ttm_all_qualifiers': ('peg0.75_ttm', 10**6, None), 'peg1.0_ttm_all_qualifiers': ('peg1.0_ttm', 10**6, None),
                            'peg0.75_3m_all_qualifiers': ('peg0.75_3m', 10**6, None), 'peg0.75_ttm_ex_construction': ('peg0.75_ttm', NMAX, is_con)}.items():
    events, diag, _, n_early = build(v, nmax, excl)
    nav, tov = sim(events, *L.cost_rates(0.001), T_START, T_END)
    hc = np.array([dd['held'] for dd in diag])
    post[nm] = {'cost': 0.001, 'holdings_mean': round(float(hc.mean()), 1), 'holdings_min': int(hc.min()), 'early_sold_total': int(n_early), 'turnover_per_rebalance': round(tov, 3),
                'windows': {w: M.stats(nav, a, bb, SIGV, bench=E50, ew=EW_NAV) for w, (a, bb) in WIN.items()}, 'yearly_pct': M.yearly(nav)}
res['post_hoc_diagnostics'] = post

# ---------------- 判定（主版本） ----------------
mw = res[MAIN]['cost_0.001']['windows']
def ok(w): return bool((w is not None) and (w['t_vs_EW_m'] >= 2) and (w['CAGR'] >= w['0050_CAGR']))
p1, p2 = ok(mw['select_2021-09_2023-12']), ok(mw['valid_2024-01_2026-10'])
res['verdict_main'] = {'select_pass': p1, 'valid_pass': p2, 'verdict': '可用' if p1 and p2 else ('待觀察' if p1 or p2 else '不可用')}
res['runtime_s'] = round(time.time() - T0, 1)


def conv(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    return str(o)
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=conv), encoding='utf-8')
print('done', res['runtime_s'], res['verdict_main'], 'engine diff', res['engine_check_max_abs_diff'])
