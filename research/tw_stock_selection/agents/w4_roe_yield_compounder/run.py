#!/usr/bin/env python3
"""存股 ROE＋殖利率＋本益比篩選（隱含 ROE = PB/PE）— 可成交回測（上市，pe_5y 2021-09～2026-10）。
執行：python3 -I run.py。PREREG 先寫入 results.json 才開始計算。
成交、成本、漲停判定、60 日均值修補沿用 momentum_family/mlib.py → exec_rebuild/exlib.py（只讀取）。
sim() 與 exlib.sim_portfolio 同邏輯、另支援「無持股＝現金」，並以 exlib 逐日比對驗證。"""
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

PREREG = {
    '寫於看結果前': True,
    '來源規則': 'Smart 智富 6004355／6007312／6009146、財報狗「長期存股」、CMoney note 166600：存股挑 ROE 高（≥15%，台股太少時放寬到 10%）、殖利率 ≥ 5%、本益比不高（≤ 20）的公司，長期持有、不停損；ROE 要多年穩定。',
    '資料': 'pe_5y.csv.gz（每月第一個交易日 TWSE 公布之 PE、PB、殖利率，2021-09-01～2026-10-08，62 個月；market == TWSE）。PE 空值＝虧損或無資料，視為不符合 0 < PE。',
    '當時可知性': '事前檢查：以 close_raw(d)/PE(d) 推 EPS，相鄰月份 EPS 不變的比例 vs 用 close_raw(d−1)，若前者明顯較高則 PE 用的是 d 當日收盤 → 訊號只能在 d 收盤後取得，d+1 開盤成交（不能在 d 開盤成交）。殖利率為 TWSE 當日公布的落後殖利率（最近年度現金股利／d 收盤），在 d 收盤後已知。PB 用最近已公布季報每股淨值、PE 用近四季 EPS（TWSE 公布值，當時已知）。',
    'ROE 代理': 'ROE_ttm = PB / PE（= 近四季 EPS／最近季每股淨值）；PE ≤ 0 或空值、PB ≤ 0 → ROE 為空值（不符合）。非教學的多年平均 ROE。',
    '穩定性代理': '訊號月 k 及其前 11 個月（共 12 個月初，月份 k−11..k）中，該股有 pe_5y 紀錄的每一個月初 ROE_ttm 都 ≥ 10%（該月有紀錄但 PE 空值＝不符合），且有紀錄月數 ≥ 11。pe_5y 從 2021-09 起 → 第一個可算訊號月為 2022-07（2021-09～2022-07 共 11 個月）。',
    '入選條件（訊號日 d）': ['股池內', '0 < PE ≤ 20', '殖利率 ≥ 5%', '穩定性代理成立', 'ROE_ttm ≥ 15%；若符合檔數 < 10，整批改用 ROE_ttm ≥ 10%（其餘條件不變）；放寬後仍不足 10 檔就有幾檔買幾檔，0 檔放現金'],
    '股池（同池等權也用它）': '面板上市股（4 碼數字代碼）、排除 91xx 存託憑證；不排除金融股（教學沒排除）；d 當日 pe_5y 有該股紀錄；d 收盤 mlib.V60（剔除 2018-01-13 等稀疏日後滾動）≥ NT$1,000 萬且 close_raw ≥ 10 元；d+1 有開盤價與還原價。',
    '進出場': 'd（pe_5y 日期＝月初第一個交易日）收盤後才知道 → d+1 開盤成交。換股日把組合調成「本次入選者」等權（續抱者也調回等權，漂移後權重計周轉）；未入選者 d+1 開盤賣出。換股日之間不交易、不停損。',
    '換股頻率': {'主規格': '每年一次：7 月的訊號日（2022-07、2023-07、2024-07、2025-07、2026-07），持有 12 個月到下一個 7 月。2022-07 之前放現金、統計從 2022-07 換股後開始。',
               '變體': ['每季（1、4、7、10 月；換股月價格面板缺漏時順延到下一可換股月）', '每月', '年度換股起始月敏感度：1～12 月各一版（只報告、不改判定）']},
    '買不到': 'd+1 開盤漲停（mlib.GAP_UP，含息跳空 ≥ 漲停 −0.6%）且未持有的新股視為買不到、不遞補（規則是符合者全買，沒有排名）；已持有者仍入選則續抱。',
    '面板缺漏': '2024-01-02、2024-02-01 兩個訊號日不在價格面板 → 月版這兩個月不換股；季版 2024 Q1 順延到 2024-03。年版不受影響。',
    '停牌／下市': '沿用 exlib：以最後成交價計，下次換股時以該價賣出。',
    '成本': 'exlib.cost_rates：買 0.0855%+滑價、賣 0.0855%+稅 0.3%+滑價；主 = 滑價 0.1%；壓力 = 買 0.5%+賣 0.5%；另報未扣成本。',
    '對手': '0050 含息（日頻）；同池等權 = 同股池全部等權、同引擎、每月 d+1 開盤換股、不扣成本（與 w1_pe_river_low、w4_buffett_roe_pe_screen 同口徑）。',
    '統計': '日頻 NAV：年化、日頻最大回撤、逐年；t = 訊號日 d 收盤 → 下一個 d 收盤的月窗超額（每月一個觀察值＝按月份分群），對 0050 與同池等權，扣成本後。',
    '期間': {'挑選期': ['2022-07-01', '2023-12-31'], '驗證期': ['2024-01-01', '2026-10-31'], '另報_2024_2025': ['2024-01-01', '2025-12-31'],
             '全期': ['2022-07-01', '2026-10-31'], '官方價格段': ['2025-09-24', '2026-10-31'],
             '說明': 'BRIEF 2021 起資料：2021～2023 挑選、2024～2026 驗證；因穩定性代理要 11 個月歷史，挑選期實際從 2022-07 開始（約 17 個月窗、年版只有 2 次換股）。價格類長期間（2005～2019／2019～2026）pe_5y 沒有，無法做。面板 2019–2023 只有存活股。'},
    '診斷版本（只報告、不改判定）': ['no_roe：拿掉 ROE 與穩定性條件（只留 0<PE≤20、殖利率≥5%），年 7 月與月版 → 檢驗 ROE 品質條件是否有增益', 'no_stab：拿掉穩定性代理，年 7 月版', 'roe10_only：直接用 ROE ≥ 10%（不先試 15%），年 7 月版'],
    '判定': '主規格（年 7 月、滑價 0.1%）：挑選期與驗證期都「對同池等權月窗 t ≥ 2 且年化 ≥ 0050」→ 可用；只一段過 → 待觀察；否則不可用。其他版本與壓力成本只報告。',
    '資料檢查（事前列出）': ['PE 用 d 收盤的驗證', '每次換股股池檔數、15%／10% 各自通過數、是否放寬、開盤漲停擋下數', '各條件通過率', 'ROE 代理極端值（> 50%）', '持有期 flag 3/4/5', '產業分布（含金融股比重）、持股名單', '官方價格段（排除 src3）', '引擎與 exlib.sim_portfolio 比對'],
}
res = {'PREREG': PREREG}
OUT = HERE / 'results.json'
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')

IDX, COLS, N, pos = L.IDX, L.COLS, L.N, L.pos
IDXSET = set(IDX)

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

# 當時可知性檢查
Cr = L.C
md_in = [d for d in MD if d in IDXSET]
s0, s1 = [], []
for a, b in zip(md_in[:-1], md_in[1:]):
    ia, ib = pos(a), pos(b)
    E0a = Cr.loc[a] / PE.loc[a]; E0b = Cr.loc[b] / PE.loc[b]
    E1a = Cr.iloc[ia - 1] / PE.loc[a]; E1b = Cr.iloc[ib - 1] / PE.loc[b]
    s0.append(float(((E0b / E0a - 1).abs() < 0.002).mean())); s1.append(float(((E1b / E1a - 1).abs() < 0.002).mean()))
res['timing_check'] = {'share_eps_unchanged_using_close_d': round(np.mean(s0), 3), 'share_eps_unchanged_using_close_d-1': round(np.mean(s1), 3),
                       'conclusion': 'PE 用 d 當日收盤 → d+1 開盤成交' if np.mean(s0) > 2 * np.mean(s1) else '不明確'}

# 穩定性代理
STAB = np.zeros((Mn, N), bool); NOBS = np.zeros((Mn, N), int)
with np.errstate(invalid='ignore'):
    good = HAS & (ROE >= 0.10)
    bad = HAS & ~(ROE >= 0.10)
for k in range(Mn):
    lo = max(0, k - 11)
    NOBS[k] = HAS[lo:k + 1].sum(0)
    STAB[k] = (NOBS[k] >= 11) & (bad[lo:k + 1].sum(0) == 0)

# ---------------- 股池 ----------------
is_common = np.array([(len(c) == 4 and c.isdigit() and not c.startswith('91')) for c in COLS])
IND = M.IND.reindex(COLS).fillna('未分類')
is_fin = (IND == '金融保險業').to_numpy()
POOL = np.zeros((Mn, N), bool); EXI = np.full(Mn, -1)
for k, d in enumerate(MD):
    if d not in IDXSET: continue
    i = pos(d)
    if i + 1 >= len(IDX): continue
    EXI[k] = i + 1
    v60 = M.V60.iloc[i].to_numpy(); px = L.C.iloc[i].to_numpy()
    trad = np.isfinite(L.OPEN_RAW[i + 1]) & np.isfinite(L.ADJ_RAW[i + 1])
    POOL[k] = (v60 >= 1e7) & (px >= 10) & is_common & HAS[k] & trad
K_FIRST = int(np.where(MD >= pd.Timestamp('2022-07-01'))[0][0])
print('skipped months', [str(MD[k].date()) for k in range(Mn) if EXI[k] < 0], 'first signal', MD[K_FIRST].date())
GAP = M.GAP_UP

with np.errstate(invalid='ignore'):
    BASE = POOL & (PEn > 0) & (PEn <= 20) & (YLn >= 5)

def signal(kind):
    """回傳 (sig[Mn,N], info[k])"""
    sig = np.zeros((Mn, N), bool); info = {}
    with np.errstate(invalid='ignore'):
        for k in range(Mn):
            if k < K_FIRST: continue
            if kind == 'main':
                a = BASE[k] & STAB[k] & (ROE[k] >= 0.15); b = BASE[k] & STAB[k] & (ROE[k] >= 0.10)
                relaxed = a.sum() < 10; sig[k] = b if relaxed else a
                info[k] = {'n15': int(a.sum()), 'n10': int(b.sum()), 'relaxed': bool(relaxed)}
            elif kind == 'no_stab':
                a = BASE[k] & (ROE[k] >= 0.15); b = BASE[k] & (ROE[k] >= 0.10)
                relaxed = a.sum() < 10; sig[k] = b if relaxed else a
                info[k] = {'n15': int(a.sum()), 'n10': int(b.sum()), 'relaxed': bool(relaxed)}
            elif kind == 'roe10_only':
                sig[k] = BASE[k] & STAB[k] & (ROE[k] >= 0.10); info[k] = {'n': int(sig[k].sum())}
            elif kind == 'no_roe':
                sig[k] = BASE[k]; info[k] = {'n': int(sig[k].sum())}
    return sig, info

def rebal_months(freq):
    if freq == 'monthly': return None
    if freq == 'quarterly': return (1, 4, 7, 10)
    if freq.startswith('annual'): return (int(freq.split('_')[1]),)

def build(sig, freq, info):
    held = set(); events, diag = {}, []
    months = rebal_months(freq); pending = False
    for k in range(K_FIRST, Mn):
        if months is not None:
            if MD[k].month in months: pending = True
            if not pending: continue
        if EXI[k] < 0: continue
        pending = False
        ex = EXI[k]
        want = set(np.where(sig[k])[0])
        blk = [j for j in want if (j not in held) and GAP[ex, j]]
        new_held = want - set(blk)
        events[ex] = ('open', np.array(sorted(new_held), dtype=int))
        dd = {'d': str(MD[k].date()), 'pool': int(POOL[k].sum()), 'pass': len(want), 'held': len(new_held),
              'new': len(new_held - held), 'sold': len(held - new_held), 'gap_blocked': len(blk)}
        dd.update(info.get(k, {}))
        diag.append(dd)
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

T_START = EXI[K_FIRST] - 1; T_END = len(IDX) - 1
SIGD = pd.DatetimeIndex([MD[k] for k in range(K_FIRST, Mn) if EXI[k] >= 0])
pool_events = {EXI[k]: ('open', np.where(POOL[k])[0]) for k in range(K_FIRST, Mn) if EXI[k] >= 0}
nA, _ = sim(pool_events, *L.cost_rates(0.001), T_START, T_END)
nB, _ = L.sim_portfolio(pool_events, *L.cost_rates(0.001), T_START, T_END)
res['engine_check_max_abs_diff'] = float((nA - nB).abs().max())
EW_NAV, _ = sim(pool_events, 0, 0, T_START, T_END)
E50 = L.E50
WIN = {'select_2022-07_2023-12': ('2022-07-01', '2023-12-31'), 'valid_2024-01_2026-10': ('2024-01-01', '2026-10-31'),
       'y2024_2025': ('2024-01-01', '2025-12-31'), 'full_2022-07_2026-10': ('2022-07-01', '2026-10-31'),
       'official_2025-09-24_2026-10': ('2025-09-24', '2026-10-31')}
FL = pd.read_parquet(L.P / 'flag.parquet').reindex(index=IDX, columns=COLS).to_numpy()

def stats_from(nav, first_exec_date):
    out = {}
    for w, (a, b) in WIN.items():
        a2 = max(pd.Timestamp(a), first_exec_date)
        out[w] = M.stats(nav, a2, b, SIGD, bench=E50, ew=EW_NAV) if a2 < pd.Timestamp(b) else None
    return out

with np.errstate(invalid='ignore'):
    P = POOL[K_FIRST:]; Pn = P.sum()
    rt = lambda m: round(float((m[K_FIRST:] & POOL[K_FIRST:]).sum() / Pn), 3)
    res['condition_pass_rate_in_pool'] = {'0<pe<=20': rt((PEn > 0) & (PEn <= 20)), 'yield>=5': rt(YLn >= 5), 'roe>=15': rt(ROE >= .15), 'roe>=10': rt(ROE >= .10),
                                          'stab12m_roe>=10': rt(STAB), 'base(pe&yield)': rt(BASE), 'base&stab&roe15': rt(BASE & STAB & (ROE >= .15)),
                                          'base&stab&roe10': rt(BASE & STAB & (ROE >= .10))}
_ps = POOL[K_FIRST:][EXI[K_FIRST:] >= 0].sum(1)
res['pool_size'] = {'mean': round(float(_ps.mean()), 1), 'min': int(_ps.min()), 'max': int(_ps.max()), 'note': '只計可換股月份（排除 2024-01、2024-02 面板缺漏）'}

sl = pd.read_csv(SL, dtype={'code': str}).drop_duplicates('code').set_index('code')['name']
navs = {'same_pool_EW': EW_NAV}
RUNS = [('main', 'annual_7'), ('main', 'quarterly'), ('main', 'monthly')] + [('main', f'annual_{m}') for m in range(1, 13) if m != 7] + \
       [('no_roe', 'annual_7'), ('no_roe', 'monthly'), ('no_stab', 'annual_7'), ('roe10_only', 'annual_7')]
SIGS = {}
for kind, freq in RUNS:
    if kind not in SIGS: SIGS[kind] = signal(kind)
    sig, info = SIGS[kind]
    events, diag = build(sig, freq, info)
    first_exec = IDX[min(events)]
    name = f'{kind}|{freq}'; R = {'first_exec': str(first_exec.date())}
    for cost in ('gross', 0.001, 'stress'):
        b, s = (0.0, 0.0) if cost == 'gross' else L.cost_rates(cost)
        nav, tov = sim(events, b, s, T_START, T_END)
        navs[f'{name}|{cost}'] = nav
        R[f'cost_{cost}'] = {'turnover_per_rebalance': round(tov, 3), 'windows': stats_from(nav, first_exec), 'yearly_pct': M.yearly(nav.loc[first_exec:])}
    hc = np.array([dd['held'] for dd in diag])
    R['holdings'] = {'mean': round(float(hc.mean()), 1), 'min': int(hc.min()), 'max': int(hc.max()), 'n_rebalances': len(diag),
                     'months_empty': int((hc == 0).sum()), 'gap_blocked_total': int(sum(dd['gap_blocked'] for dd in diag)),
                     'n_relaxed': int(sum(dd.get('relaxed', False) for dd in diag))}
    rows = []; exs = sorted(events)
    for a_, b_ in zip(exs, exs[1:] + [T_END]):
        for j in events[a_][1]:
            p0, p1 = L.AO[a_, j], (L.AO[b_, j] if b_ != T_END else L.AC[b_, j])
            k_ = int(np.where(EXI == a_)[0][0])
            rows.append({'code': COLS[j], 'name': sl.get(COLS[j], ''), 'industry': IND[COLS[j]], 'exec': str(IDX[a_].date()), 'exit': str(IDX[b_].date()),
                         'gross': p1 / p0 - 1 if np.isfinite(p0) and np.isfinite(p1) else np.nan,
                         'roe': ROE[k_, j], 'pe': PEn[k_, j], 'yield': YLn[k_, j], 'flag345': bool(np.isin(FL[a_:b_ + 1, j], [3, 4, 5]).any())})
    hd = pd.DataFrame(rows)
    if len(hd):
        R['holding_detail'] = {'n_stock_periods': int(len(hd)), 'n_unique_codes': int(hd['code'].nunique()), 'n_flag345': int(hd['flag345'].sum()),
                               'n_gross_gt_50pct': int((hd['gross'] > 0.5).sum()), 'share_financial': round(float((hd['industry'] == '金融保險業').mean()), 3),
                               'industry_mix_top8': hd['industry'].value_counts().head(8).to_dict(), 'most_held_top10': hd['code'].value_counts().head(10).to_dict(),
                               'median_roe': round(float(hd['roe'].median()), 3), 'n_roe_gt_50pct': int((hd['roe'] > 0.5).sum()),
                               'median_pe': round(float(hd['pe'].median()), 2), 'median_yield': round(float(hd['yield'].median()), 2)}
    if freq in ('annual_7', 'monthly', 'quarterly') and kind == 'main':
        R['diag'] = diag
    if (kind, freq) == ('main', 'annual_7'):
        hd.to_csv(HERE / 'holdings_main.csv.gz', index=False)
        R['per_cohort'] = [{'exec': e, 'n': int(len(g)), 'mean_gross_pct': round(float(g['gross'].mean() * 100), 1),
                            'codes': ' '.join(f"{c}{n}" for c, n in zip(g['code'], g['name']))} for e, g in hd.groupby('exec')]
    res[name] = R
    w = R['cost_0.001']['windows']
    print(name, {k: (v['CAGR'], v['t_vs_EW_m'], v['t_vs_0050_m']) if v else None for k, v in w.items()}, R['holdings'])
res['same_pool_EW'] = {'yearly_pct': M.yearly(EW_NAV), 'windows': {w: M.stats(EW_NAV, a, b, SIGD, bench=E50) for w, (a, b) in WIN.items()}}
res['0050_yearly_pct'] = M.yearly(E50.loc[EW_NAV.index])
pd.DataFrame(navs).to_parquet(HERE / 'nav.parquet')

mw = res['main|annual_7']['cost_0.001']['windows']
def ok(w): return bool(w is not None and (w['t_vs_EW_m'] >= 2) and (w['CAGR'] >= w['0050_CAGR']))
p1, p2 = ok(mw['select_2022-07_2023-12']), ok(mw['valid_2024-01_2026-10'])
res['verdict_main'] = {'select_pass': p1, 'valid_pass': p2, 'verdict': '可用' if p1 and p2 else ('待觀察' if p1 or p2 else '不可用')}
res['runtime_s'] = round(time.time() - T0, 1)

def conv(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    return str(o)
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=conv), encoding='utf-8')
print('done', res['runtime_s'], res['verdict_main'], 'engine diff', res['engine_check_max_abs_diff'], res['timing_check'])
