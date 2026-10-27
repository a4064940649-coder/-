#!/usr/bin/env python3
"""存股族「巴菲特條件」：ROE ≥ 15%、0 < PE < 15、殖利率 > 3%、近 3 年每年都有現金除息（另版加 PB < 1.5）— 可成交回測（上市，2021-09～2026-10）。
執行：python3 -I run.py。PREREG 先寫入 results.json 才開始計算。
成交、成本、漲停判定、60 日均值修補沿用 momentum_family/mlib.py → exec_rebuild/exlib.py（只讀取）。
sim() 與 exlib.sim_portfolio 同邏輯、另支援「無持股＝現金」（複製自 w1_pe_river_low/run.py 的寫法），並以 exlib 逐日比對驗證。"""
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

VARIANTS = {   # 名稱: (ROE 門檻, PE 上限, PB 上限或 None)
    'roe15_pe15': (0.15, 15, None), 'roe10_pe15': (0.10, 15, None), 'roe15_pe16': (0.15, 16, None), 'roe10_pe16': (0.10, 16, None),
    'roe15_pe15_pb1.5': (0.15, 15, 1.5), 'roe10_pe15_pb1.5': (0.10, 15, 1.5), 'roe15_pe16_pb1.5': (0.15, 16, 1.5), 'roe10_pe16_pb1.5': (0.10, 16, 1.5),
}
MAIN = ('roe15_pe15', 'monthly')
PREREG = {
    '寫於看結果前': True,
    '來源規則': 'Smart 智富 36455／6013066、今周刊 201804090041、CMoney note 166600、nstock 207：存股族的巴菲特條件——ROE ≥ 15%、本益比 < 15、現金殖利率 > 3%、連年配息；Smart 2014 四條件版另加 PB < 1.5。',
    '資料': 'pe_5y.csv.gz（每月第一個交易日 TWSE 公布之 PE、PB、殖利率，2021-09-01～2026-10-08，共 62 個月）。PE 為空值＝虧損或無資料，視為不符合 0 < PE。',
    'ROE 代理': 'ROE_ttm ≈ PB / PE = EPS_ttm / 每股淨值（TWSE 的 PE 用近四季 EPS、PB 用最近一季每股淨值）。只用當期 TTM，不要求 5 年平均（pe_5y 從 2021-09 起，做不出）。',
    '條件（月 m 的訊號日 d）': ['ROE 代理 ≥ 15%（主）；另測 ≥ 10%', '0 < PE < 15（主）；另測 < 16', '殖利率 > 3%', '近 3 年每年都有現金除息紀錄', '四條件版另加 PB < 1.5'],
    '近 3 年連年配息的定義': {
        '年份': '訊號日 d 所在年 Y 的前三個完整日曆年 Y−1、Y−2、Y−3，每一年都至少一筆現金除息（除息日在該年）。不用 Y 年當年已發生的除息（保守、無前視）。',
        '資料來源': 'exdiv_5y（kind ∈ {息, 權息}）只涵蓋 2021-10-07 以後；之前的年份（挑選期需要 2018～2021）用價格面板推算：［原寫 flag == 2 且，見資料修補］含息報酬隱含配發 (1+ret)×前收/收 − 1 ≥ 0.2%。面板版無法分辨現金股利與純股票股利（exdiv_5y 中純「權」約占 6%），列為已知誤差。2021 年＝2021-10-06 以前用面板、之後用 exdiv_5y。',
        '偏離': '教學原本要求 5～10 年，資料（exdiv_5y）不夠，依任務縮短為 3 年；面板其實可推到更早，但不改任務給定的 3 年。'},
    '可測範圍': '價格面板只有上市股；上櫃日 K 只有 2025-09-24 起（history.csv.gz），涵蓋不了 2021-09 起的訊號 → 只測上市，上櫃從缺（偏離規則，FINDINGS 註明）。價格類長期間（2005-01～2019-04／2019-05～2026-10）因 pe_5y 只從 2021-09 起，無法做，改用 2021 起資料的切分。',
    '股池（同池等權也用它）': '面板上市股、排除 91xx 存託憑證；排除金融保險業（stock_list industry = 金融保險業；不在名單的已下市股以代碼 28xx 判定）；d 當日 pe_5y 有該股紀錄；d 收盤 60 日均成交值 ≥ NT$1,000 萬（mlib.V60，剔除稀疏日後滾動）且收盤 ≥ 10 元；d+1 有開盤價。不設前 300 名、不設處置濾網（教學沒有）。',
    '進出場': 'd（月初第一個交易日，即 pe_5y 日期）收盤後才知道 PE/PB/殖利率 → d+1 開盤成交。全部符合者等權買進；每次重篩日把組合調為「本次符合者」等權（續抱者也調回等權，漂移後權重計周轉）；不再符合者 d+1 開盤賣出；無符合者放現金（報酬 0）。',
    '重篩頻率': {'主': '每月重篩（持有 1 個月）', '另報': '每季重篩：1、4、7、10 月的訊號日；該月無法換股（價格面板缺漏）時順延到下一個可換股月份，下一季仍回到原季月。'},
    '買不到': 'd+1 開盤漲停（mlib.GAP_UP，含息跳空 ≥ 漲停 −0.6%）且未持有的新股視為買不到、不遞補（規則是全買，沒有排名可遞補）；已持有者若仍符合就續抱。',
    '面板缺漏': '2024-01-02、2024-02-01 兩個訊號日不在價格面板（2024-01-02～2024-02-14 缺漏）→ 這兩個月不換股，持股續抱到 2024-03；季版 2024 Q1 換股順延到 2024-03。',
    '停牌／下市': '沿用 exlib：以最後成交價計，下次換股時以該價賣出。',
    '成本': 'exlib.cost_rates：買 0.0855%+滑價、賣 0.0855%+稅 0.3%+滑價；主 = 滑價 0.1%（來回約 0.67%）；壓力 = 買 0.5%+賣 0.5%（來回 1%）；另報未扣成本。',
    '對手': '0050 含息（日頻）；同池等權 = 同股池全部等權、同引擎、每月 d+1 開盤換股、不扣成本（與 w1_pe_river_low 相同口徑）。',
    '統計': '日頻 NAV：年化、日頻最大回撤、逐年；t = 訊號日 d 收盤 → 下一個 d 收盤的月窗超額（每月一個觀察值＝按月份分群）對 0050 與同池等權。季版也用月窗。',
    '期間': {'挑選期': ['2021-09-01', '2023-12-31'], '驗證期': ['2024-01-01', '2026-10-31'], '另報_2024_2025': ['2024-01-01', '2025-12-31'],
             '全期': ['2021-09-01', '2026-10-31'], '官方價格段': ['2025-09-24', '2026-10-31'],
             '說明': '依 BRIEF 2021 起資料切分；挑選期實際 2021-09～2023-12（28 個月窗）。面板 2019–2023 只有存活股（倖存者偏差，挑選期同時影響策略與同池等權）。'},
    '版本': {'參數格': 'ROE {15%, 10%} × PE 上限 {15, 16} × {無 PB, PB < 1.5} = 8 組 × {月, 季} = 16 組，全部報告。', '主版本': 'ROE ≥ 15%、0 < PE < 15、殖利率 > 3%、近 3 年連年配息、無 PB 條件、每月重篩、滑價 0.1%。'},
    '判定': '主版本：挑選期與驗證期都「對同池等權月窗 t ≥ 2 且年化 ≥ 0050」→ 可用；只一段過 → 待觀察；否則不可用。其他 15 組、壓力成本只報告、不改判定、不用來挑參數。',
    '資料修補（第一次執行後補記，非調參）': '第一次執行發現 2021～2023 所有月份符合檔數為 0：面板 flag == 2（除權息日）只從 2021-10 起才有標記，2018～2020 年推算不到任何除息 → 「近 3 年連年配息」全部不成立。改為只用隱含配發 (1+ret)×前收/收 − 1 ≥ 0.2%（不再要求 flag == 2）。在 2022～2023 與 exdiv_5y 現金除息比對（股票×年）：2022 面板 803 檔／exdiv 799 檔，不一致 12＋8 檔；2023 818／803，不一致 19＋4 檔（約 98% 一致）。門檻、其他參數未動。修補前結果存於 results.json 的 pre_fix_summary。',
    '資料檢查（事前列出）': ['每月股池檔數、符合檔數、各條件通過率、開盤漲停擋下數', '面板推算除息與 exdiv_5y 在 2022～2023 的一致性', 'ROE 代理極端值（> 50%）與一次性收益疑慮', '持有期 flag 3/4/5 檔次', '產業分布、持股集中度', 'src3 段影響：報官方價格段', '引擎與 exlib.sim_portfolio 比對'],
}
res = {'PREREG': PREREG}
if (HERE / '_prefix_summary.json').exists():
    res['pre_fix_summary'] = json.loads((HERE / '_prefix_summary.json').read_text())
OUT = HERE / 'results.json'
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')

IDX, COLS, N, pos = L.IDX, L.COLS, L.N, L.pos
CA = np.array(COLS)

# ---------------- pe_5y 月表 ----------------
pe = pd.read_csv(XR / 'pe_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
pe = pe[(pe['market'] == 'TWSE') & pe['code'].isin(COLS)]
piv = lambda v: pe.pivot_table(index='date', columns='code', values=v, aggfunc='last').sort_index().reindex(columns=COLS)
PE, PB, YL = piv('pe'), piv('pb'), piv('yield')
HAS = pe.assign(one=1).pivot_table(index='date', columns='code', values='one', aggfunc='last').reindex(index=PE.index, columns=COLS).notna().to_numpy()
MD = PE.index; Mn = len(MD)
PEn, PBn, YLn = PE.to_numpy(), PB.to_numpy(), YL.to_numpy()
with np.errstate(all='ignore'):
    ROE = np.where((PEn > 0) & (PBn > 0), PBn / PEn, np.nan)

# ---------------- 連年配息 ----------------
FLAG = pd.read_parquet(L.P / 'flag.parquet').reindex(index=IDX, columns=COLS)
Dimp = (1 + L.RET) * L.C.shift(1) / L.C - 1
EXCUT = pd.Timestamp('2021-10-07')
pan_ev = (Dimp >= 0.002).loc[:EXCUT - pd.Timedelta(days=1)]          # 修補：flag 2 只從 2021-10 起才有，改只用隱含配發
pan_years = pan_ev.groupby(pan_ev.index.year).any()                        # year × code
ex = pd.read_csv(XR / 'exdiv_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
exT = ex[(ex['market'] == 'TWSE') & ex['code'].isin(COLS)]
excash = exT[exT['kind'].isin(['息', '權息'])]
ex_years = pd.crosstab(excash['date'].dt.year, excash['code']).reindex(columns=COLS, fill_value=0) > 0
yrs = range(2004, 2027)
DIVY = (pan_years.reindex(index=yrs, columns=COLS, fill_value=False).astype(bool) |
        ex_years.reindex(index=yrs, columns=COLS, fill_value=False).astype(bool))
DIV3 = np.zeros((Mn, N), bool)
for k, d in enumerate(MD):
    Y = d.year; DIV3[k] = DIVY.loc[[Y - 1, Y - 2, Y - 3]].all(axis=0).to_numpy()
# 一致性檢查：2022-2023 面板推算 vs exdiv_5y
pan_full = (Dimp >= 0.002)
pchk = pan_full.loc['2022-01-01':'2023-12-31']
pany = pchk.groupby(pchk.index.year).any()
exany_all = pd.crosstab(exT['date'].dt.year, exT['code']).reindex(index=[2022, 2023], columns=COLS, fill_value=0) > 0
exany_cash = ex_years.reindex(index=[2022, 2023], columns=COLS, fill_value=False)
res['div_proxy_check_2022_2023'] = {
    str(y): {'panel_years_with_event': int(pany.loc[y].sum()), 'exdiv_any_kind': int(exany_all.loc[y].sum()), 'exdiv_cash': int(exany_cash.loc[y].sum()),
             'panel_yes_exdiv_cash_no': int((pany.loc[y] & ~exany_cash.loc[y]).sum()), 'panel_no_exdiv_cash_yes': int((~pany.loc[y] & exany_cash.loc[y]).sum())}
    for y in (2022, 2023)}

# ---------------- 股池 ----------------
sl = pd.read_csv(SL, dtype={'code': str}).drop_duplicates('code').set_index('code')
indu = pd.Series({c: (sl.at[c, 'industry'] if c in sl.index else None) for c in COLS})
is_fin = np.array([(indu[c] == '金融保險業') or (indu[c] is None and c.startswith('28')) for c in COLS])
is_dr = np.array([c.startswith('91') for c in COLS])
IDXSET = set(IDX)
POOL = np.zeros((Mn, N), bool); EXI = np.full(Mn, -1)
for k, d in enumerate(MD):
    if d not in IDXSET: continue
    i = pos(d)
    if i + 1 >= len(IDX): continue
    EXI[k] = i + 1
    v60 = M.V60.iloc[i].to_numpy(); px = L.C.iloc[i].to_numpy()
    trad = np.isfinite(L.OPEN_RAW[i + 1]) & np.isfinite(L.ADJ_RAW[i + 1])
    POOL[k] = (v60 >= 1e7) & (px >= 10) & ~is_dr & ~is_fin & HAS[k] & trad
print('skipped months', [str(MD[k].date()) for k in range(Mn) if EXI[k] < 0])
GAP = M.GAP_UP

def cond(v):
    r, pmax, pb = VARIANTS[v]
    with np.errstate(invalid='ignore'):
        c = POOL & (ROE >= r) & (PEn > 0) & (PEn < pmax) & (YLn > 3) & DIV3
        if pb is not None: c &= (PBn < pb)
    return c

def build(sig, freq):
    held = set(); events, diag = {}, []
    qpending = False
    for k in range(Mn):
        if freq == 'quarterly':
            if MD[k].month in (1, 4, 7, 10) or k == 0: qpending = True
            if not qpending: continue
        if EXI[k] < 0: continue
        qpending = False
        ex = EXI[k]
        want = set(np.where(sig[k])[0])
        blk = [j for j in want if (j not in held) and GAP[ex, j]]
        new_held = want - set(blk)
        events[ex] = ('open', np.array(sorted(new_held), dtype=int))
        diag.append({'d': str(MD[k].date()), 'pool': int(POOL[k].sum()), 'pass': len(want), 'held': len(new_held),
                     'new': len(new_held - held), 'sold': len(held - new_held), 'gap_blocked': len(blk)})
        held = new_held
    return events, diag

def sim(events, buy, sell, t0, t1):
    u = np.zeros(N); cash = 1.0; nav = np.empty(t1 - t0 + 1); tov = []
    for t in range(t0, t1 + 1):
        if t in events:
            mode, pk = events[t]
            Pz = L.AO[t]
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
SIGD = pd.DatetimeIndex([MD[k] for k in range(Mn) if EXI[k] >= 0])
pool_events = {EXI[k]: ('open', np.where(POOL[k])[0]) for k in range(Mn) if EXI[k] >= 0}

nA, _ = sim(pool_events, *L.cost_rates(0.001), T_START, T_END)
nB, _ = L.sim_portfolio(pool_events, *L.cost_rates(0.001), T_START, T_END)
res['engine_check_max_abs_diff'] = float((nA - nB).abs().max())
EW_NAV, _ = sim(pool_events, 0, 0, T_START, T_END)
E50 = L.E50
WIN = {'select_2021-09_2023-12': ('2021-09-01', '2023-12-31'), 'valid_2024-01_2026-10': ('2024-01-01', '2026-10-31'),
       'y2024_2025': ('2024-01-01', '2025-12-31'), 'full_2021-09_2026-10': ('2021-09-01', '2026-10-31'),
       'official_2025-09-24_2026-10': ('2025-09-24', '2026-10-31')}
FL = FLAG.to_numpy()

# 條件通過率（股池內）
def rate(m): return round(float(m.sum() / max(POOL.sum(), 1)), 3)
with np.errstate(invalid='ignore'):
    res['condition_pass_rate_in_pool'] = {'roe>=15': rate(POOL & (ROE >= .15)), 'roe>=10': rate(POOL & (ROE >= .10)), '0<pe<15': rate(POOL & (PEn > 0) & (PEn < 15)),
                                          '0<pe<16': rate(POOL & (PEn > 0) & (PEn < 16)), 'yield>3': rate(POOL & (YLn > 3)), 'div3y': rate(POOL & DIV3), 'pb<1.5': rate(POOL & (PBn < 1.5))}
res['pool_size'] = {'mean': round(float(POOL[EXI >= 0].sum(1).mean()), 1), 'min': int(POOL[EXI >= 0].sum(1).min()), 'max': int(POOL[EXI >= 0].sum(1).max())}
res['n_financial_codes_excluded_in_panel'] = int(is_fin.sum())

navs = {'same_pool_EW': EW_NAV}
b_main, s_main = L.cost_rates(0.001)
for v in VARIANTS:
    sig = cond(v)
    for freq in ('monthly', 'quarterly'):
        events, diag = build(sig, freq)
        name = f'{v}|{freq}'; R = {}
        for cost in ('gross', 0.001, 'stress'):
            b, s = (0.0, 0.0) if cost == 'gross' else L.cost_rates(cost)
            nav, tov = sim(events, b, s, T_START, T_END)
            navs[f'{name}|{cost}'] = nav
            R[f'cost_{cost}'] = {'turnover_per_rebalance': round(tov, 3), 'windows': {w: M.stats(nav, a, bb, SIGD, bench=E50, ew=EW_NAV) for w, (a, bb) in WIN.items()},
                                 'yearly_pct': M.yearly(nav)}
        hc = np.array([dd['held'] for dd in diag])
        R['holdings'] = {'mean': round(float(hc.mean()), 1), 'min': int(hc.min()), 'max': int(hc.max()), 'months_empty': int((hc == 0).sum()),
                         'gap_blocked_total': int(sum(dd['gap_blocked'] for dd in diag))}
        # 持股明細（月份 × 檔）：產業、flag、集中度
        rows = []
        exs = sorted(events)
        for a_, b_ in zip(exs, exs[1:] + [T_END]):
            for j in events[a_][1]:
                p0, p1 = L.AO[a_, j], (L.AO[b_, j] if b_ != T_END else L.AC[b_, j])
                rows.append({'code': COLS[j], 'exec': str(IDX[a_].date()), 'gross': p1 / p0 - 1 if np.isfinite(p0) and np.isfinite(p1) else np.nan,
                             'flag345': bool(np.isin(FL[a_:b_ + 1, j], [3, 4, 5]).any())})
        hd = pd.DataFrame(rows)
        if len(hd):
            R['holding_detail'] = {'n_stock_periods': int(len(hd)), 'n_unique_codes': int(hd['code'].nunique()), 'n_flag345': int(hd['flag345'].sum()),
                                   'n_gross_gt_50pct': int((hd['gross'] > 0.5).sum()),
                                   'industry_mix_top8': indu.reindex(hd['code']).fillna('未列').value_counts().head(8).to_dict(),
                                   'most_held_top10': hd['code'].value_counts().head(10).to_dict()}
        if (v, freq) == MAIN:
            R['diag_monthly'] = diag
            hd.to_csv(HERE / 'holdings_main.csv.gz', index=False)
        res[name] = R
res['same_pool_EW'] = {'yearly_pct': M.yearly(EW_NAV), 'windows': {w: M.stats(EW_NAV, a, bb, SIGD, bench=E50) for w, (a, bb) in WIN.items()}}
res['0050_yearly_pct'] = M.yearly(E50.loc[EW_NAV.index])
# ROE 代理極端值
with np.errstate(invalid='ignore'):
    m = cond('roe15_pe15')
    res['roe_proxy_in_main_picks'] = {'n': int(m.sum()), 'median': round(float(np.nanmedian(ROE[m])), 3), 'p90': round(float(np.nanpercentile(ROE[m], 90)), 3),
                                      'n_gt_50pct': int((ROE[m] > 0.5).sum()), 'median_pe': round(float(np.nanmedian(PEn[m])), 2), 'median_pb': round(float(np.nanmedian(PBn[m])), 2),
                                      'median_yield': round(float(np.nanmedian(YLn[m])), 2)}
pd.DataFrame(navs).to_parquet(HERE / 'nav.parquet')

# 判定（主版本）
mw = res[f'{MAIN[0]}|{MAIN[1]}']['cost_0.001']['windows']
def ok(w): return (w['t_vs_EW_m'] >= 2) and (w['CAGR'] >= w['0050_CAGR'])
p1, p2 = ok(mw['select_2021-09_2023-12']), ok(mw['valid_2024-01_2026-10'])
res['verdict_main'] = {'select_pass': bool(p1), 'valid_pass': bool(p2), 'verdict': '可用' if p1 and p2 else ('待觀察' if p1 or p2 else '不可用')}
res['runtime_s'] = round(time.time() - T0, 1)

def conv(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    return str(o)
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=conv), encoding='utf-8')
print('done', res['runtime_s'], res['verdict_main'], 'engine diff', res['engine_check_max_abs_diff'])
