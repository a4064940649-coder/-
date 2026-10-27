#!/usr/bin/env python3
"""紅利低波（中證紅利低波 H30269／00713 選股法，簡化版）— 可成交回測（上市，2007-01～2026-10）。
執行：python3 -I run.py。PREREG 先寫入 results.json 才開始計算任何策略結果。
成交、成本、漲停判定沿用 momentum_family/mlib.py → exec_rebuild/exlib.py（只讀取、不修改）。
股利推估邏輯複製自 w4_sun_7pct_dividend_bible/run.py（E1／E3 與 exdiv_5y 合併；不 import 它）。
sim() 是 exlib.sim_portfolio 的非等權延伸，並以 exlib 對同一等權組合逐日比對驗證。"""
import json, sys, time, math
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
    '簡化版聲明': '00713（臺灣指數公司特選高股息低波動指數）另有 ROE、盈餘穩定度等財報條件，中證紅利低波有市值條件；本研究沒有長期財報與股本資料，全部省略，市值前 80% 以 252 日日均成交值前 80% 代替。這是「殖利率＋低波動」兩段式選股的簡化版，不是指數的複製。',
    '來源規則': '中證紅利低波（H30269）：年度調樣，過去 3 年連續現金分紅、過去 3 年平均現金股息率排序取前 75 檔，再取過去一年波動率最低 50 檔，股息率加權、單檔上限 10%。00713：6 月與 12 月調整、另加近一年股價報酬門檻。',
    '股利資料': {
        '2004-01～2021-10-06（面板推估）': '除權息日（panel flag 1 或 2，且前一交易日有收盤，排除「不比價」X 日）。面板把這些日子的日報酬設為 0，adj/close_raw 比值的跳動＝前收／當日價。主規格 E3：每股股利 D = 前收 − 當日開盤／(1＋當日全市場開盤跳空中位數)（w4_sun_7pct_dividend_bible 以 exdiv_5y 真值驗證 MAE 約 1.0% 股價，E1 收盤比值法約 1.8%；此選擇只依資料誤差，不依策略結果）；D<0 記 0。',
        '2021-10-07 起（exdiv_5y，TWSE）': 'kind=息 → 現金股利＝value；kind=權（純股票股利）→ 不算；kind=權息 → value 全額當現金（上限）。',
        'flag 3/4/5': '不當作股利。',
        '已知限制': ['2021-10 前無法區分現金與股票股利（股票股利、現增除權會被算成現金股利，2005～2012 配股盛行時殖利率偏高）', '估計誤差 MAE 約 1% 股價', '股利是原始每股金額'],
        '敏感度（事前固定，只報告）': ['E1：D = 前收 − 當日收盤（題目所說的比值法原樣）', 'cap15：E3 下單一事件推估殖利率 > 15% 視為非現金（股票股利／公司行動），不計入'],
    },
    '殖利率定義': '年度 k 的現金殖利率 y_k = 年度 k 內所有除息事件 D 的總和 ÷ 年度 k 最後一個交易日 close_raw（該日無收盤則用年度內最後一筆收盤）。3 年平均殖利率 yld3 = (y_{k1}+y_{k2}+y_{k3})/3。',
    '股利年度': {'12 月訊號（年 Y）': '用 Y−2、Y−1、Y 三個年度（Y 年除息已全部發生，年底收盤＝訊號日收盤）', '6 月訊號（年 Y）': '用 Y−3、Y−1 之間三個完整年度 Y−3、Y−2、Y−1'},
    '連續配息條件': '三個年度每年 D 總和 > 0（E3 推估 ≤ 0 視為該年未配息）。',
    '股池（訊號日 d 收盤可知）': [
        '面板上市普通股：4 碼數字、排除 91xx（存託憑證）',
        'd 有收盤；252 日日均成交值 ADV252（先剔除有值檔數 < 300 的稀疏日再滾動，min_periods 200）在 d 當天所有有 ADV252 的普通股中排前 80%（代替市值前 80%）',
        'BRIEF 流動性下限：mlib.V60 ≥ NT$1,000 萬、close_raw ≥ 10 元',
        'd+1 有開盤價與還原價（可成交）',
        '連續 3 年配息條件 → 「配息股池」（＝同池等權的股池）'],
    '選股步驟': {
        '動能門檻（00713）': '近一年含息報酬 R252 = adj(d)/adj(d 前 252 個非稀疏交易日) − 1；R252 < −30% 剔除；R252 無值剔除。主規格在排序前剔除（先剔除再取殖利率前 75）。',
        '步驟一': '配息股池中（通過動能門檻、252 日波動有值）依 yld3 由高到低取前 75 檔。',
        '步驟二': '75 檔中依過去 252 日（剔除稀疏日、min_periods 200）日報酬標準差由小到大取 50 檔。d+1 開盤漲停（mlib.GAP_UP，2015-06 前 7%、之後 10%）且未持有者買不到，跳過、由 75 檔中下一低波動者遞補（75 檔用完則少於 50 檔）。',
        '敏感度 F2（事前固定，只報告）': '先依 yld3 取前 75，再剔除 R252 < −30%，剩下的取最低波動 50（剔除後不補足 75）。'},
    '權重': {'(a) 等權': '每檔 1/n', '(b) 殖利率加權（中證版）': 'w ∝ yld3，單檔上限 10%（超出部分依比例分給未達上限者，反覆至收斂）'},
    '換股時點': {'年度版（中證）': '每年 12 月最後一個交易日收盤算訊號 → 隔一交易日開盤成交（M2）', '半年版（00713）': '每年 6 月與 12 月最後一個交易日收盤算訊號 → 隔一交易日開盤成交',
               '持有期': '持有到下一次換股，期間不調整（權重隨價格漂移）；持有期間停止交易／下市者以最後成交價計（exlib 規則），下次換股時以該價賣出。',
               '面板缺漏': '2023-12-29 訊號的下一個面板交易日是 2024-02-15（2024-01-02～02-14 無資料），就在 2024-02-15 開盤成交；缺漏期間持有舊組合（所有對手同樣跨過缺口）。',
               '起點': '第一個訊號 2006-12-29（需要 2004～2006 三年股利），2007-01-02 開盤第一次成交。'},
    '主規格（判定用）': '年度版、(a) 等權、E3 股利、動能門檻先剔除、滑價 0.1%。其餘 3 個（年度殖利率加權、半年等權、半年殖利率加權）與敏感度（E1、cap15、F2）全部報告但不進判定。',
    '成本': 'exlib.cost_rates：買 0.0855%＋滑價、賣 0.0855%＋證交稅 0.3%＋滑價；主 = 滑價 0.1%（來回約 0.67%），壓力 = 買 0.5%＋賣 0.5%（來回 1%）；另報未扣成本。',
    '對手': {'0050': '含息日頻（finlab etf_0050.csv）', '0056': '含息日頻（finlab etf_0056.csv，2007-12-26 起）', '00713': '含息月底總報酬指數（tw-stock-screener extras/etf_tr_m.csv.gz，2020-08 起；只能比月頻）',
             '同池等權（判定用）': '配息股池全部等權、同一換股日、同引擎、不扣成本', '流動性池等權（另報）': '流動性池（未加配息條件）全部等權、同換股日、不扣成本'},
    '統計': '日頻 NAV：年化（實際天數）、日頻最大回撤、逐年。t＝扣成本後「月底收盤→下月底收盤」月超額報酬（每個月一個觀察值＝按月份分群）對 0050、同池等權、0056；00713 用月底總報酬指數。',
    '期間': {'價格類挑選期': ['2005-01-01', '2019-04-30'], '價格類驗證期': ['2019-05-01', '2026-10-31'], '2021 起資料挑選期': ['2021-01-01', '2023-12-31'],
             '2021 起資料驗證期': ['2024-01-01', '2026-10-31'], '另報': ['2024-01-01', '2025-12-31'], '全期': ['2005-01-01', '2026-10-31'],
             '說明': '價格類挑選期實際從 2007-01 起（先累積 3 年股利）。面板 2019～2023 只有存活股（偏多）。'},
    '判定': '主規格、滑價 0.1%：價格類挑選期與驗證期都「對同池等權月 t ≥ 2 且年化 ≥ 0050 年化」→ 可用；只有一段過 → 待觀察；都沒過 → 不可用。2021 起切分（2021～2023／2024～2026）也報告兩段是否通過，作為附註。',
    '資料檢查（事前列出）': ['每次換股：股池、配息股池、動能剔除數、75／50 實選數、漲停擋下數', '持股產業分布（金融股比重）、最常入選名單、入選股 yld3 與波動中位數 vs 配息股池', '最近年度殖利率 vs pe_5y 官方殖利率（2021 起）相關', '持有期 flag 3/4/5 與單期報酬 > 100%', '自寫 sim 與 exlib.sim_portfolio 等權比對'],
}
res = {'PREREG': PREREG}
OUT = HERE / 'results.json'
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')

IDX, COLS, N, pos = L.IDX, L.COLS, L.N, L.pos
C = L.C; Cn = L.CN; On = L.OPEN_RAW; adj = L.adj; V = L.V; RET = L.RET
FLAG = pd.read_parquet(L.P / 'flag.parquet').reindex(index=IDX, columns=COLS)
FL = FLAG.to_numpy()
CUT = pd.Timestamp('2021-10-07')

# ---------------- 股利事件（複製 w4_sun_7pct_dividend_bible 的邏輯） ----------------
Cprev = np.vstack([np.full((1, N), np.nan), Cn[:-1]])
normal = (FL == 0) & np.isfinite(Cprev) & np.isfinite(Cn)
with np.errstate(all='ignore'):
    mgap = np.nanmedian(np.where(normal & np.isfinite(On), On / Cprev - 1, np.nan), axis=1)
isX = np.isin(FL, [1, 2])
xr = np.where(isX & (IDX < CUT)[:, None] & np.isfinite(Cn))
prevok = np.isfinite(Cprev[xr])
ii, jj = xr[0][prevok], xr[1][prevok]
cp, cc, oo = Cprev[ii, jj], Cn[ii, jj], On[ii, jj]
E1 = cp - cc
E3 = np.where(np.isfinite(oo), cp - oo / (1 + mgap[ii]), E1)
ev_pre = pd.DataFrame({'date': IDX[ii], 'j': jj, 'D_E1': np.maximum(E1, 0), 'D_E3': np.maximum(E3, 0), 'yE3': E3 / cp})
exd = pd.read_csv(XR / 'exdiv_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
exd = exd[(exd['market'] == 'TWSE') & exd['code'].isin(COLS) & (exd['date'] >= CUT)]
ep = exd[exd['kind'].isin(['息', '權息'])]
ev_post = pd.DataFrame({'date': ep['date'].values, 'j': [L.CI[c] for c in ep['code']], 'D_E1': ep['value'].values, 'D_E3': ep['value'].values,
                        'yE3': (ep['value'] / ep['prev_close']).values})
EV = pd.concat([ev_pre, ev_post], ignore_index=True).sort_values('date').reset_index(drop=True)
EV['year'] = EV['date'].dt.year
res['div_events'] = {'n_pre_cut': int(len(ev_pre)), 'n_post_cut_cash_or_mixed': int(len(ev_post)),
                     'pre_cut_share_E3_yield_gt15pct': round(float((ev_pre['yE3'] > .15).mean()), 4),
                     'pre_cut_share_E3_nonpositive': round(float((ev_pre['yE3'] <= 0).mean()), 4)}

YEARS = np.arange(2004, 2027); YI = {y: k for k, y in enumerate(YEARS)}
# 年底收盤
Cff20 = C.ffill(limit=20)
YE = np.full((len(YEARS), N), np.nan)
for y in YEARS:
    m = IDX.year == y
    if m.any(): YE[YI[y]] = Cff20.loc[IDX[m][-1]].to_numpy()

def year_yield(col, drop_gt=None):
    e = EV if drop_gt is None else EV[~(EV['yE3'] > drop_gt)]
    DY = np.zeros((len(YEARS), N))
    for (y, j), v in e.groupby(['year', 'j'])[col].sum().items():
        DY[YI[y], j] = v
    with np.errstate(all='ignore'):
        Y = DY / YE
    return DY, Y

DIV = {'E3': year_yield('D_E3'), 'E1': year_yield('D_E1'), 'cap15': year_yield('D_E3', 0.15)}

# ---------------- 滾動特徵（剔除稀疏日） ----------------
GOOD = adj.notna().sum(axis=1) >= 300
res['sparse_days_removed'] = [str(d.date()) for d in IDX[~GOOD]]
ag = adj.loc[GOOD].ffill(limit=10); vg = V.loc[GOOD]; rg = RET.loc[GOOD]
ADV252 = vg.rolling(252, min_periods=200).mean().reindex(IDX)
VOL252 = rg.rolling(252, min_periods=200).std().reindex(IDX)
R252 = (ag / ag.shift(252) - 1).reindex(IDX)
V60 = M.V60.reindex(IDX)
is_common = np.array([(len(c) == 4 and c.isdigit() and not c.startswith('91')) for c in COLS])

# ---------------- 換股日 ----------------
_s = pd.Series(IDX, index=IDX)
ME = pd.DatetimeIndex([g.iloc[-1] for _, g in _s.groupby([IDX.year, IDX.month])])
SIG_ANNUAL = [d for d in ME if d.month == 12 and pd.Timestamp('2006-12-01') <= d < IDX[-1]]
SIG_SEMI = [d for d in ME if d.month in (6, 12) and pd.Timestamp('2006-12-01') <= d < IDX[-1]]
assert SIG_ANNUAL[0] == pd.Timestamp('2006-12-29'), SIG_ANNUAL[0]
GAP = M.GAP_UP

def pools_at(d, est):
    i = pos(d); ex = i + 1; Y = d.year
    yrs = [Y - 2, Y - 1, Y] if d.month == 12 else [Y - 3, Y - 2, Y - 1]
    DY, YL = DIV[est]
    k = [YI[y] for y in yrs]
    paid = (DY[k] > 0).all(0)
    yld3 = YL[k].mean(0)
    adv = ADV252.loc[d].to_numpy(); hasc = np.isfinite(Cn[i])
    base = is_common & hasc & np.isfinite(adv)
    thr = np.nanquantile(adv[base], 0.20)                      # 前 80%
    liq = base & (adv >= thr) & (V60.loc[d].to_numpy() >= 1e7) & (Cn[i] >= 10)
    trad = np.isfinite(L.OPEN_RAW[ex]) & np.isfinite(L.ADJ_RAW[ex])
    liqp = liq & trad
    divp = liqp & paid & np.isfinite(yld3) & (yld3 > 0)
    return {'ex': ex, 'liq': liqp, 'div': divp, 'yld3': yld3, 'vol': VOL252.loc[d].to_numpy(), 'r252': R252.loc[d].to_numpy(), 'n_base': int(base.sum())}

def cap_weights(x, cap=0.10):
    w = x / x.sum()
    for _ in range(100):
        over = w > cap + 1e-12
        if not over.any(): break
        exc = (w[over] - cap).sum(); w[over] = cap
        free = ~over & (w < cap)
        w[free] += exc * w[free] / w[free].sum()
    return w

def build(sig_dates, est='E3', weight='ew', filt='pre'):
    events = {}; ew_div = {}; ew_liq = {}; diag = []; held = set(); picks_log = []
    for d in sig_dates:
        p = pools_at(d, est); ex = p['ex']
        divp, yld3, vol, r252 = p['div'], p['yld3'], p['vol'], p['r252']
        okv = np.isfinite(vol)
        momok = np.isfinite(r252) & (r252 >= -0.30)
        if filt == 'pre':
            cand = np.where(divp & okv & momok)[0]
            top75 = cand[np.argsort(-yld3[cand], kind='stable')][:75]
            step2 = top75
        else:
            cand = np.where(divp & okv)[0]
            top75 = cand[np.argsort(-yld3[cand], kind='stable')][:75]
            step2 = top75[momok[top75]]
        order = step2[np.argsort(vol[step2], kind='stable')]
        pick = []; nb = 0
        for j in order:
            if GAP[ex, j] and j not in held: nb += 1; continue
            pick.append(j)
            if len(pick) == 50: break
        pick = np.array(pick, int)
        w = np.zeros(N)
        if len(pick):
            w[pick] = 1.0 / len(pick) if weight == 'ew' else cap_weights(yld3[pick].astype(float))
        events[ex] = w
        wd = np.zeros(N); wd[divp] = 1.0 / divp.sum(); ew_div[ex] = wd
        wl = np.zeros(N); wl[p['liq']] = 1.0 / p['liq'].sum(); ew_liq[ex] = wl
        held = set(pick.tolist())
        diag.append({'d': str(d.date()), 'exec': str(IDX[ex].date()), 'n_base': p['n_base'], 'liq_pool': int(p['liq'].sum()), 'div_pool': int(divp.sum()),
                     'mom_excluded_in_divpool': int((divp & ~momok).sum()), 'n_step1': int(len(top75)), 'n_step2_cand': int(len(step2)),
                     'n_pick': int(len(pick)), 'gap_blocked': nb, 'max_w': round(float(w.max()), 4),
                     'med_yld3_pick': round(float(np.median(yld3[pick])), 4) if len(pick) else None, 'med_yld3_divpool': round(float(np.nanmedian(yld3[divp])), 4),
                     'med_vol_pick_ann': round(float(np.median(vol[pick]) * math.sqrt(252)), 4) if len(pick) else None,
                     'med_vol_divpool_ann': round(float(np.nanmedian(vol[divp]) * math.sqrt(252)), 4)})
        picks_log.append((d, ex, pick, w[pick], yld3[pick], vol[pick]))
    return events, ew_div, ew_liq, diag, picks_log

def sim(events, cost, t0, t1):
    """非等權 exlib 延伸：換股日 t 開盤（L.AO）調到目標權重，買賣金額分別乘費率；其餘照 exlib。"""
    b, s = (0.0, 0.0) if cost == 'gross' else L.cost_rates(cost)
    u = np.zeros(N); cash = 1.0; nav = np.empty(t1 - t0 + 1); tov = []
    for t in range(t0, t1 + 1):
        if t in events:
            w = events[t]; Pz = L.AO[t]
            h = np.where(u > 0, u * np.nan_to_num(Pz), 0.0); Vt = h.sum() + cash; Vn = Vt
            for _ in range(8):
                tgt = w * Vn
                cst = b * np.clip(tgt - h, 0, None).sum() + s * np.clip(h - tgt, 0, None).sum()
                Vn = Vt - cst
            tov.append(np.clip(h - tgt, 0, None).sum() / Vt if Vt > 0 else 0)
            u = np.zeros(N); nz = w > 0; u[nz] = tgt[nz] / Pz[nz]; cash = max(1.0 - w.sum(), 0.0) * Vn
        nav[t - t0] = np.nansum(u * L.AC[t]) + cash
    return pd.Series(nav, index=IDX[t0:t1 + 1]), float(np.mean(tov[1:])) if len(tov) > 1 else float('nan')

T_START = pos(SIG_ANNUAL[0]); T_END = len(IDX) - 1

# ---------------- 對手 ----------------
E50 = L.E50.loc[IDX[T_START]:]
e56 = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0056.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()
E56 = e56.reindex(IDX, method='ffill').loc[IDX[T_START]:]
E56[E56.index < e56.index[0]] = np.nan
trm = pd.read_csv(XR / 'etf_tr_m.csv.gz', dtype={'code': str})
TR713 = trm[trm['code'] == '00713'].set_index('ym')['tr']; TR713.index = pd.PeriodIndex(TR713.index, freq='M')
TR56m = trm[trm['code'] == '0056'].set_index('ym')['tr']; TR56m.index = pd.PeriodIndex(TR56m.index, freq='M')

MEC = pd.DatetimeIndex([d for d in ME if d >= IDX[T_START]])   # 月底日曆（t 用）

def mrets(nav):
    x = nav.reindex(MEC); r = x / x.shift(1) - 1
    r.index = r.index.to_period('M'); return r.iloc[1:]

def mrets_tr(tr):
    r = tr / tr.shift(1) - 1
    return r.iloc[1:]

WIN = {'select_price_2005-01_2019-04': ('2005-01-01', '2019-04-30'), 'valid_price_2019-05_2026-10': ('2019-05-01', '2026-10-31'),
       'select_2021_2023': ('2021-01-01', '2023-12-31'), 'valid_2024_2026-10': ('2024-01-01', '2026-10-31'),
       'y2024_2025': ('2024-01-01', '2025-12-31'), 'full_2007_2026-10': ('2005-01-01', '2026-10-31')}

def dstats(nav, a, b):
    r = nav.pct_change(); m = (r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b)) & r.notna().to_numpy()
    if m.sum() < 20: return None
    rr = r[m]; first = np.argmax(m)
    span = (rr.index[-1] - r.index[max(first - 1, 0)]).days / 365.25
    eq = (1 + rr).cumprod()
    return {'CAGR': round(float(eq.iloc[-1] ** (1 / span) - 1), 4), 'vol': round(float(rr.std() * math.sqrt(252)), 4),
            'maxDD': round(float((eq / eq.cummax() - 1).min()), 4), 'years': round(span, 2)}

def wstats(nav, a, b, benches):
    o = dstats(nav, a, b)
    if o is None: return None
    pr = mrets(nav); pa, pb = pd.Period(a, 'M'), pd.Period(b, 'M')
    pr = pr[(pr.index >= pa) & (pr.index <= pb)].dropna()
    o['n_months'] = int(len(pr))
    for nm, bn in benches.items():
        bp = (mrets_tr(bn) if isinstance(bn.index, pd.PeriodIndex) else mrets(bn)).reindex(pr.index)
        ex = (pr - bp).dropna()
        if len(ex) < 6: continue
        o[f'excess_vs_{nm}_pct_m'] = round(float(ex.mean() * 100), 2); o[f't_vs_{nm}_m'] = L.tstat(ex); o[f'n_m_{nm}'] = int(len(ex))
        if isinstance(bn.index, pd.PeriodIndex):
            g = (1 + bp.loc[ex.index]).prod(); s_ = (1 + pr.loc[ex.index]).prod(); yrs = len(ex) / 12
            o[f'{nm}_CAGR_monthly'] = round(float(g ** (1 / yrs) - 1), 4); o[f'self_CAGR_monthly_same_months_{nm}'] = round(float(s_ ** (1 / yrs) - 1), 4)
        else:
            d2 = dstats(bn.dropna(), max(pd.Timestamp(a), bn.dropna().index[0]), b)
            if d2: o[f'{nm}_CAGR'] = d2['CAGR']; o[f'{nm}_maxDD'] = d2['maxDD']
    return o

def yearly(nav):
    r = nav.pct_change().dropna()
    return {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in r.groupby(r.index.year)}

# ---------------- 引擎檢查 ----------------
ev0, ewd0, ewl0, _, _ = build(SIG_ANNUAL)
ex_ev = {t: ('open', np.where(w > 0)[0]) for t, w in ewd0.items()}
nA, _ = sim(ewd0, 0.001, T_START, T_END)
nB, _ = L.sim_portfolio(ex_ev, *L.cost_rates(0.001), T_START, T_END)
res['engine_check_max_abs_diff'] = float((nA - nB).abs().max())

# ---------------- 執行 ----------------
sl = pd.read_csv(SL, dtype={'code': str}).drop_duplicates('code').set_index('code')['name']
IND = M.IND.reindex(COLS).fillna('未分類')
RUNS = [('annual', 'ew', 'E3', 'pre'), ('annual', 'yld', 'E3', 'pre'), ('semi', 'ew', 'E3', 'pre'), ('semi', 'yld', 'E3', 'pre'),
        ('annual', 'ew', 'E1', 'pre'), ('annual', 'ew', 'cap15', 'pre'), ('annual', 'ew', 'E3', 'post')]
MAIN = 'annual|ew|E3|pre'
navs = {}; EWN = {}
for sched, wt, est, filt in RUNS:
    sd = SIG_ANNUAL if sched == 'annual' else SIG_SEMI
    events, ew_div, ew_liq, diag, plog = build(sd, est, wt, filt)
    name = f'{sched}|{wt}|{est}|{filt}'
    key = (sched, est)
    if key not in EWN:
        EWN[key] = {'EW_divpool': sim(ew_div, 'gross', T_START, T_END)[0], 'EW_liqpool': sim(ew_liq, 'gross', T_START, T_END)[0]}
        navs[f'EW_divpool|{sched}|{est}'] = EWN[key]['EW_divpool']; navs[f'EW_liqpool|{sched}|{est}'] = EWN[key]['EW_liqpool']
    benches = {'0050': E50, 'EW': EWN[key]['EW_divpool'], 'EWliq': EWN[key]['EW_liqpool'], '0056': E56, '00713': TR713}
    R = {}
    for cost in ('gross', 0.001, 'stress'):
        nav, tov = sim(events, cost, T_START, T_END)
        navs[f'{name}|{cost}'] = nav
        R[f'cost_{cost}'] = {'turnover_per_rebalance': round(tov, 3), 'windows': {w: wstats(nav, a, b, benches) for w, (a, b) in WIN.items()}, 'yearly_pct': yearly(nav)}
    dg = pd.DataFrame(diag)
    R['diag_summary'] = {'n_rebalances': int(len(dg)), 'mean_liq_pool': round(float(dg['liq_pool'].mean()), 1), 'mean_div_pool': round(float(dg['div_pool'].mean()), 1),
                         'mean_mom_excluded': round(float(dg['mom_excluded_in_divpool'].mean()), 1), 'min_pick': int(dg['n_pick'].min()), 'mean_pick': round(float(dg['n_pick'].mean()), 1),
                         'gap_blocked_total': int(dg['gap_blocked'].sum()), 'max_weight_seen': float(dg['max_w'].max())}
    if name == MAIN or name == 'annual|yld|E3|pre':
        R['diag'] = diag
        rows = []
        exs = [x[1] for x in plog] + [T_END]
        for k, (d, ex, pk, ww, yy, vv) in enumerate(plog):
            nx = exs[k + 1]
            for j, w_, y_, v_ in zip(pk, ww, yy, vv):
                p0 = L.AO[ex, j]; p1 = L.AO[nx, j] if nx != T_END else L.AC[nx, j]
                rows.append({'signal': str(d.date()), 'code': COLS[j], 'name': sl.get(COLS[j], ''), 'industry': IND[COLS[j]], 'w': round(float(w_), 4),
                             'yld3': round(float(y_), 4), 'vol_ann': round(float(v_ * math.sqrt(252)), 4), 'gross': p1 / p0 - 1 if np.isfinite(p0) and np.isfinite(p1) else np.nan,
                             'flag345': bool(np.isin(FL[ex:nx + 1, j], [3, 4, 5]).any())})
        hd = pd.DataFrame(rows)
        if name == MAIN: hd.to_csv(HERE / 'holdings_main.csv.gz', index=False)
        R['holding_detail'] = {'n_stock_periods': int(len(hd)), 'n_unique': int(hd['code'].nunique()), 'n_flag345': int(hd['flag345'].sum()),
                               'n_gross_gt_100pct': int((hd['gross'] > 1).sum()), 'n_gross_lt_m50pct': int((hd['gross'] < -0.5).sum()),
                               'share_financial': round(float((hd['industry'] == '金融保險業').mean()), 3),
                               'share_financial_by_signal_year': hd.assign(f=hd['industry'] == '金融保險業').groupby(hd['signal'].str[:4])['f'].mean().round(2).to_dict(),
                               'industry_top10': hd['industry'].value_counts().head(10).to_dict(),
                               'most_held_top20': {f"{c}{sl.get(c, '')}": int(n) for c, n in hd['code'].value_counts().head(20).items()},
                               'yld3_gt15pct_picks': int((hd['yld3'] > 0.15).sum()),
                               'latest_picks': hd[hd['signal'] == hd['signal'].max()][['code', 'name', 'industry', 'w', 'yld3', 'vol_ann']].to_dict('records')}
    res[name] = R
    w = R['cost_0.001']['windows']
    print(name, {k: (v['CAGR'], v.get('t_vs_EW_m'), v.get('t_vs_0050_m'), v.get('0050_CAGR'), v.get('EW_CAGR')) if v else None for k, v in w.items()}, flush=True)

# 對手本身
for key, d_ in EWN.items():
    res[f'bench_EW|{key[0]}|{key[1]}'] = {nm: {'yearly_pct': yearly(n), 'windows': {w: wstats(n, a, b, {'0050': E50}) for w, (a, b) in WIN.items()}} for nm, n in d_.items()}
res['bench_0050_yearly_pct'] = yearly(E50); res['bench_0056_yearly_pct'] = yearly(E56.dropna())
res['bench_0056_windows'] = {w: dstats(E56.dropna(), max(pd.Timestamp(a), E56.dropna().index[0]), b) for w, (a, b) in WIN.items()}
tr = TR713; res['bench_00713_yearly_pct_from_month_end_TR'] = {int(y): round(float(g.iloc[-1] / tr[tr.index < pd.Period(f'{y}-01', 'M')].iloc[-1] - 1) * 100, 1)
                                                              for y, g in tr.groupby(tr.index.year) if (tr.index < pd.Period(f'{y}-01', 'M')).any()}
# 0056 日頻 vs 月頻 TR 一致性
x = mrets(E56.dropna()).dropna(); y_ = mrets_tr(TR56m).reindex(x.index).dropna()
res['check_0056_daily_vs_monthlyTR_max_abs_diff'] = round(float((x.reindex(y_.index) - y_).abs().max()), 5)

# pe_5y 交叉檢查：Dec 訊號年 Y 的 y_Y（E3）vs 下一年 1～3 月初 pe_5y TWSE 殖利率（官方最近年度股利／價格；只比排名）
pe = pd.read_csv(XR / 'pe_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
pe = pe[(pe['market'] == 'TWSE') & pe['code'].isin(COLS)]
xc = {}
DY3, YL3 = DIV['E3']
for Y in range(2021, 2026):
    g = pe[(pe['date'] >= f'{Y+1}-01-01') & (pe['date'] < f'{Y+1}-04-01')].sort_values('date').drop_duplicates('code')
    j = np.array([L.CI[c] for c in g['code']]); ours = YL3[YI[Y], j]; tw = g['yield'].to_numpy() / 100
    ok = np.isfinite(ours) & np.isfinite(tw) & (ours > 0) & (tw > 0)
    if ok.sum() > 30:
        xc[Y] = {'n': int(ok.sum()), 'spearman': round(float(pd.Series(ours[ok]).rank().corr(pd.Series(tw[ok]).rank())), 3),
                 'median_ratio_ours_over_twse': round(float(np.median(ours[ok] / tw[ok])), 3)}
res['pe5y_crosscheck_yearly_yield'] = xc

# ---------------- 事後診斷（看過主結果後加；只報告、不進判定） ----------------
def build_alt(sig_dates, mode):
    events = {}; held = set()
    for d in sig_dates:
        p = pools_at(d, 'E3'); ex = p['ex']; divp, yld3, vol, r252 = p['div'], p['yld3'], p['vol'], p['r252']
        cand = np.where(divp & np.isfinite(vol) & np.isfinite(r252) & (r252 >= -0.30))[0]
        if mode == 'yield50': order = cand[np.argsort(-yld3[cand], kind='stable')]; lim = 50
        elif mode == 'lowvol50': order = cand[np.argsort(vol[cand], kind='stable')]; lim = 50
        elif mode == 'top75': order = cand[np.argsort(-yld3[cand], kind='stable')][:75]; lim = 75
        elif mode == 'lowyield_lowvol50':   # 反向對照：殖利率後 75 檔裡最低波動 50
            o = cand[np.argsort(yld3[cand], kind='stable')][:75]; order = o[np.argsort(vol[o], kind='stable')]; lim = 50
        pick = []
        for j in order:
            if GAP[ex, j] and j not in held: continue
            pick.append(j)
            if len(pick) == lim: break
        w = np.zeros(N); w[pick] = 1.0 / len(pick); events[ex] = w; held = set(pick)
    return events
WIN_PH = dict(WIN, **{'sub_2007_2012': ('2005-01-01', '2012-12-31'), 'sub_2013_2019-04': ('2013-01-01', '2019-04-30')})
ph = {'說明': '看過主結果後加的事後診斷：拆解兩段式選股哪一段在起作用（只看殖利率、只看低波動、殖利率前 75 等權、反向：低殖利率裡的低波動），以及主規格的子期間；年度換股、E3、滑價 0.1%，對手同主規格（年度配息股池等權）。不進判定。'}
bn = {'0050': E50, 'EW': EWN[('annual', 'E3')]['EW_divpool'], '0056': E56, '00713': TR713}
for mode in ['yield50', 'lowvol50', 'top75', 'lowyield_lowvol50']:
    nav, tov = sim(build_alt(SIG_ANNUAL, mode), 0.001, T_START, T_END); navs[f'posthoc_{mode}|0.001'] = nav
    ph[mode] = {'turnover': round(tov, 3), 'windows': {w: wstats(nav, a, b, bn) for w, (a, b) in WIN_PH.items()}, 'yearly_pct': yearly(nav)}
for nm in [MAIN, 'semi|ew|E3|pre']:
    nav = navs[f'{nm}|0.001']
    ph[f'subwindows|{nm}'] = {w: wstats(nav, a, b, bn) for w, (a, b) in WIN_PH.items() if w.startswith('sub_')}
    a_ = mrets(nav); b_ = mrets_tr(TR713).reindex(a_.index); m_ = a_.notna() & b_.notna()
    ph[f'corr_monthly_vs_00713|{nm}'] = {'n': int(m_.sum()), 'corr': round(float(a_[m_].corr(b_[m_])), 3),
                                         'tracking_err_ann': round(float((a_[m_] - b_[m_]).std() * math.sqrt(12)), 4)}
    c56 = mrets(E56.dropna()).reindex(a_.index); m2 = a_.notna() & c56.notna()
    ph[f'corr_monthly_vs_0056|{nm}'] = {'n': int(m2.sum()), 'corr': round(float(a_[m2].corr(c56[m2])), 3)}
res['posthoc'] = ph

pd.DataFrame(navs).to_parquet(HERE / 'nav.parquet')

def ok(w): return bool(w is not None and w.get('t_vs_EW_m', -9) >= 2 and w['CAGR'] >= w['0050_CAGR'])
verd = {}
for sched, wt, est, filt in RUNS:
    name = f'{sched}|{wt}|{est}|{filt}'; mw = res[name]['cost_0.001']['windows']
    p1, p2 = ok(mw['select_price_2005-01_2019-04']), ok(mw['valid_price_2019-05_2026-10'])
    q1, q2 = ok(mw['select_2021_2023']), ok(mw['valid_2024_2026-10'])
    verd[name] = {'price_select_pass': p1, 'price_valid_pass': p2, 'split2021_select_pass': q1, 'split2021_valid_pass': q2,
                  'verdict_price_split': '可用' if p1 and p2 else ('待觀察' if p1 or p2 else '不可用')}
res['verdicts_all_variants'] = verd
res['verdict_main'] = {'spec': MAIN, **verd[MAIN]}
res['runtime_s'] = round(time.time() - T0, 1)

def conv(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    return str(o)
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=conv), encoding='utf-8')
print('done', res['runtime_s'], res['verdict_main'], 'engine diff', res['engine_check_max_abs_diff'])
