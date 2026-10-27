#!/usr/bin/env python3
"""孫慶龍「7% 存股聖經」：5 年平均現金股利殖利率 ≥ 7% 買、≤ 5% 賣半、≤ 3% 全賣（遲滯帶）— 可成交回測（上市，2010-01～2026-10）。
執行：python3 -I run.py。PREREG 先寫入 results.json 才開始計算策略結果。
成交、成本、漲停判定、60 日均值修補沿用 momentum_family/mlib.py → exec_rebuild/exlib.py（只讀取、不修改）。
sim() 是 exlib.sim_portfolio 的延伸（支援非等權目標、閒置部位放現金或 0050），並以 exlib 逐日比對驗證。"""
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
    '來源規則': '孫慶龍「7% 存股聖經」（Smart 智富 6013374、6009146、6008849；nstock 789）：挑長期穩定配息的公司，用「近 5 年平均現金股利」算便宜價＝平均股利／7%、合理價＝／5%、昂貴價＝／3%；股價 ≤ 便宜價買進，≥ 合理價開始減碼（賣一半），≥ 昂貴價全部出清；配息中斷就出場。',
    '股利資料（D 的來源）': {
        '2004-01～2021-10-06（面板推估）': '除權息日（panel flag 1 或 2）的 adj／close_raw 比值變化反推：面板在這段把除權息日報酬設 0，所以比值變化＝前收／當日收盤 → 每股股利估計 D = 前一交易日收盤 − 當日收盤（主規格 E1，即題目指定的比值法）；D < 0 記 0。',
        '不比價濾網': '證交所「X」同時代表除權息與「不比價」（前一日無成交等）。2021-10 起用 exdiv_5y 對照：前一交易日無收盤的 X 日 1,077 筆中幾乎都不是除權息、真除息 3,815 筆只有 2 筆前一日無成交 → 只採用「前一交易日有收盤」的 X 日。此濾網是資料檢查決定的，不是看策略結果決定的。',
        '公司行動': 'flag 3（參考價重設：減資等）、4（資料缺口）、5（錯誤）不當作股利。',
        '2021-10-07 起（exdiv_5y）': 'market=TWSE 且代碼在面板中：kind=息 → 現金股利＝value（前收−參考價）；kind=權（純股票股利）→ 不算現金股利；kind=權息 → 無法拆分，value 全額當現金股利（上限）。',
        '已知限制（事前列出）': ['2021-10 前無法區分現金股利與股票股利／現增除權：股票股利會被當成現金股利，2010～2015 年配股盛行，挑選期 D5 偏高、名單會混入配股公司', '估計誤差：E1 對 exdiv_5y 真值（官方價格段）MAE 約 1.8% 股價，E3（開盤、扣市場跳空）約 1.0%', '股利是原始每股金額（未按之後的配股、減資調整），和教學用 Goodinfo 原始數字一致'],
        '敏感度（事前固定）': ['E3：D = 前收 − 當日開盤／(1+當日全市場開盤跳空中位數)，以 exdiv_5y 真值驗證誤差較小', 'cap15（診斷）：E1 下單一事件推估殖利率 > 15% 視為非現金（股票股利／公司行動），不計入'],
    },
    '股利年度與 D5': '股利年度＝除息日所在曆年。訊號日 t（年 Y）：若該股在 Y 年 1/1～t 已有現金除息 → 最近股利年度 Y*=Y（只計已除息部分），否則 Y*=Y−1。D5 = Y*−4..Y* 五個年度現金股利總額的平均（沒有發放的年度算 0）。5 年均殖利率 y5 = D5 / close_raw(t)。',
    '股池（買進資格，也是同池等權的股池）': ['面板上市股（4 碼數字、排除 91xx）', 't 收盤 mlib.V60（剔除 2018-01-13 等稀疏日後滾動的 60 日均成交值）≥ NT$1,000 萬、close_raw ≥ 10 元', 'Y−5..Y−1 五個完整曆年每年都有現金除息', '未配息中斷', 't+1 有開盤價與還原價'],
    '配息中斷（強制全賣）': 'Y−1 年沒有現金除息，或最近一次現金除息距 t 超過 456 天（約 15 個月，容許除息日前後移）。',
    '訊號與成交時點': '每月最後一個交易日 t 收盤後計算 → t+1 開盤成交（M2）。t+1 開盤漲停（mlib.GAP_UP，2015-06 前 7%、之後 10%，含息跳空 ≥ 漲停 −0.6%）時不能買進／加碼，由下一名遞補。若 t+1 距 t 超過 10 個日曆日（2024-01-02～2024-02-14 面板缺漏）該月不換股。',
    '主規格（遲滯帶）': {'買進': '股池內且 y5 ≥ 7%；新買進依 y5 由高到低補到最多 30 檔（已持有者優先保留）',
                       '持有中': 'y5 ≤ 3% → 全賣；3% < y5 ≤ 5% → 賣一半（已是半倉就維持）；半倉股若 y5 回到 ≥ 7% 且符合股池 → 加回全倉；5%～7% 之間維持原部位',
                       '權重': '每檔一個名額：全倉權重 1/n、半倉 1/(2n)，n＝持有檔數；半倉釋出的部位與無持股時放現金（主）或 0050（另報）。每月換股日把持股調回目標權重。',
                       '下市／長期停牌': '持股若最近 20 個交易日都無收盤 → 以最後成交價賣出（exlib 規則：以最後成交價計）。',
                       '持有期': '沒有固定持有期，只依賣出條件出場（每月檢查）。'},
    '簡化版（另報）': 'y5 ≤ 5% 就全賣，其餘同主規格。',
    '閒置部位': '主＝現金（0 報酬）；另報＝0050 含息（以 t 收盤價成交近似，因無 0050 開盤價）。兩者都報、判定用主規格（現金）。',
    '成本': 'exlib.cost_rates：股票買 0.0855%+滑價、賣 0.0855%+稅 0.3%+滑價；主 = 滑價 0.1%，壓力 = 買 0.5%+賣 0.5%；另報未扣成本。0050：買 0.0855%+滑價、賣 0.0855%+ETF 稅 0.1%+滑價（壓力同股票壓力）。',
    '對手': '0050 含息（日頻）；同池等權＝上述股池全部等權、同引擎、每月 t+1 開盤換股、不扣成本（與 w1_pe_river_low、w4_roe_yield_compounder 同口徑）。',
    '統計': '日頻 NAV：年化、日頻最大回撤、逐年；t＝訊號日 t 收盤 → 下一訊號日收盤的月窗超額（每月一個觀察值＝按月份分群），扣成本後，對 0050 與同池等權。',
    '期間': {'價格類挑選期': ['2010-01-01', '2019-04-30'], '價格類驗證期': ['2019-05-01', '2026-10-31'],
             '2021 起資料挑選期': ['2021-01-01', '2023-12-31'], '2021 起資料驗證期': ['2024-01-01', '2026-10-31'], '另報': ['2024-01-01', '2025-12-31'], '全期': ['2010-01-01', '2026-10-31'],
             '說明': 'BRIEF 價格類挑選期從 2005-01 起，但要先累積 2005～2009 五年股利，第一個訊號是 2010-01 月底。面板 2019–2023 只有存活股（偏誤偏多）。'},
    '判定': '主規格（遲滯帶、E1、閒置現金、滑價 0.1%）：價格類挑選期與驗證期都「對同池等權月窗 t ≥ 2 且年化 ≥ 0050」→ 可用；只一段過 → 待觀察；否則不可用。其餘版本、2021 起切分、壓力成本只報告。',
    '資料檢查（事前列出）': ['股利推估 vs exdiv_5y 真值（E1～E4 誤差，只用官方價格段）', '不比價 X 日比例', '推估「最近年度股利 / 收盤」vs pe_5y 的 TWSE 殖利率（2021-09 起）相關與中位比值', '每月股池檔數、y5 ≥ 7% 檔數、持有檔數、半倉檔數、空手月數、開盤漲停擋下數', '持有期 flag 3/4/5、單期報酬 > 50%、y5 極端值（> 20%）', '產業分布（含金融股比重）、最常持有名單', '引擎與 exlib.sim_portfolio 比對'],
}
res = {'PREREG': PREREG}
OUT = HERE / 'results.json'
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')

IDX, COLS, N, pos = L.IDX, L.COLS, L.N, L.pos
C = L.C; Cn = L.CN; On = L.OPEN_RAW
FLAG = pd.read_parquet(L.P / 'flag.parquet').reindex(index=IDX, columns=COLS)
FL = FLAG.to_numpy()
SRC = L.SRC.reindex(index=IDX, columns=COLS).to_numpy()
CUT = pd.Timestamp('2021-10-07')

# ---------------- 股利事件 ----------------
Cprev = np.vstack([np.full((1, N), np.nan), Cn[:-1]])
normal = (FL == 0) & np.isfinite(Cprev) & np.isfinite(Cn)
with np.errstate(all='ignore'):
    mret = np.nanmedian(np.where(normal, Cn / Cprev - 1, np.nan), axis=1)
    mgap = np.nanmedian(np.where(normal & np.isfinite(On), On / Cprev - 1, np.nan), axis=1)
isX = np.isin(FL, [1, 2])
pre_mask = (IDX < CUT)[:, None]
xr = np.where(isX & pre_mask & np.isfinite(Cn))
prevok = np.isfinite(Cprev[xr])
res['x_days_pre_cut'] = {'n_X': int(len(xr[0])), 'share_prev_day_traded': round(float(prevok.mean()), 3)}
ii, jj = xr[0][prevok], xr[1][prevok]
cp, cc, oo = Cprev[ii, jj], Cn[ii, jj], On[ii, jj]
E1 = cp - cc
E3 = np.where(np.isfinite(oo), cp - oo / (1 + mgap[ii]), E1)
ev_pre = pd.DataFrame({'date': IDX[ii], 'j': jj, 'D_E1': np.maximum(E1, 0), 'D_E3': np.maximum(E3, 0), 'yld_E1': E1 / cp})
exd = pd.read_csv(XR / 'exdiv_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
exd = exd[(exd['market'] == 'TWSE') & exd['code'].isin(COLS) & (exd['date'] >= CUT)]
ev_post = exd[exd['kind'].isin(['息', '權息'])]
ev_post = pd.DataFrame({'date': ev_post['date'].values, 'j': [L.CI[c] for c in ev_post['code']], 'D_E1': ev_post['value'].values,
                        'D_E3': ev_post['value'].values, 'yld_E1': (ev_post['value'] / ev_post['prev_close']).values})
EV = pd.concat([ev_pre, ev_post], ignore_index=True).sort_values('date').reset_index(drop=True)
EV['year'] = EV['date'].dt.year
res['div_events'] = {'n_pre_cut': int(len(ev_pre)), 'n_post_cut_cash': int(len(ev_post)), 'n_post_cut_stock_only_ignored': int((exd['kind'] == '權').sum()),
                     'n_post_cut_mixed_counted_full': int((exd['kind'] == '權息').sum()),
                     'pre_cut_E1_yield_quantiles': {q: round(float(np.nanquantile(ev_pre['yld_E1'], q)), 4) for q in (.05, .25, .5, .75, .95, .99)},
                     'pre_cut_share_E1_yield_gt15pct': round(float((ev_pre['yld_E1'] > .15).mean()), 4),
                     'pre_cut_share_E1_negative': round(float((ev_pre['yld_E1'] < 0).mean()), 4)}

# 估計誤差驗證（2021-10 起、官方價格段，exdiv_5y 為真值）
val = {}
exv = pd.read_csv(XR / 'exdiv_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
exv = exv[exv['code'].isin(COLS) & exv['date'].isin(IDX)]
for kind in ['息', '權', '權息']:
    e = exv[exv['kind'] == kind]
    ti = np.array([pos(d) for d in e['date']]); tj = np.array([L.CI[c] for c in e['code']])
    ok = np.isfinite(Cprev[ti, tj]) & np.isin(SRC[ti, tj], [1, 2, 4]) & np.isin(SRC[ti - 1, tj], [1, 2, 4]) & \
         ((e['date'] < '2024-01-01') | (e['date'] > '2025-09-30')).to_numpy()
    ti, tj, e = ti[ok], tj[ok], e[ok]
    cp_, cc_, oo_ = Cprev[ti, tj], Cn[ti, tj], On[ti, tj]
    tr = e['value'].to_numpy() / e['prev_close'].to_numpy()
    est = {'E1_close(指定比值法)': (cp_ - cc_) / cp_, 'E2_close_mkt': (cp_ - cc_ / (1 + mret[ti])) / cp_,
           'E3_open_mkt': (cp_ - oo_ / (1 + mgap[ti])) / cp_, 'E4_open_raw': (cp_ - oo_) / cp_}
    val[kind] = {'n': int(len(e)), 'true_yield_mean': round(float(tr.mean()), 4),
                 **{k: {'bias': round(float(np.nanmean(v - tr)), 4), 'MAE': round(float(np.nanmean(np.abs(v - tr))), 4),
                        'p95_abs': round(float(np.nanquantile(np.abs(v - tr), .95)), 4)} for k, v in est.items()}}
# 不比價：2021-10 起 X 日對 exdiv_5y
xm = np.where(isX & ~pre_mask)
xd = pd.DataFrame({'date': IDX[xm[0]], 'code': COLS[xm[1]], 'prev': np.isfinite(Cprev[xm])})
xd = xd.merge(exv[['date', 'code', 'kind']], on=['date', 'code'], how='left')
val['X_days_post_cut_vs_exdiv'] = {f'{k}|prev_traded={p}': int(n) for (k, p), n in xd.groupby([xd['kind'].fillna('不在exdiv'), 'prev']).size().items()}
res['estimator_validation'] = val

# ---------------- 年度表 ----------------
YEARS = np.arange(2004, 2027); YI = {y: k for k, y in enumerate(YEARS)}
def year_tables_ev(e, col):
    DY = np.zeros((len(YEARS), N)); PAID = np.zeros((len(YEARS), N), bool)
    for (y, j), v in e.groupby(['year', 'j'])[col].sum().items():
        DY[YI[y], j] = v; PAID[YI[y], j] = True
    return DY, PAID, e

def year_tables(col, drop_gt=None):
    e = EV if drop_gt is None else EV[~(EV['yld_E1'] > drop_gt)]
    DY = np.zeros((len(YEARS), N)); PAID = np.zeros((len(YEARS), N), bool)
    g = e.groupby(['year', 'j'])[col].sum()
    for (y, j), v in g.items():
        DY[YI[y], j] = v; PAID[YI[y], j] = True
    return DY, PAID, e

# 訊號日：每月最後一個交易日（2010-01 起）
_s = pd.Series(IDX, index=IDX)
ME = [g.iloc[-1] for _, g in _s.groupby([IDX.year, IDX.month])]
ME = [d for d in ME if d >= pd.Timestamp('2010-01-01')]
SIGD = pd.DatetimeIndex(ME)
EXI = {}
for d in SIGD:
    i = pos(d)
    if i + 1 < len(IDX) and (IDX[i + 1] - d).days <= 10: EXI[d] = i + 1
is_common = np.array([(len(c) == 4 and c.isdigit() and not c.startswith('91')) for c in COLS])
Cff = C.ffill().to_numpy()
last_trade_age = np.zeros((len(IDX), N), int)
_age = np.full(N, 10**6)
for t in range(len(IDX)):
    _age = np.where(np.isfinite(Cn[t]), 0, _age + 1); last_trade_age[t] = _age
V60 = M.V60.reindex(IDX).to_numpy()

def signals(col, drop_gt=None, ev=None, share=None):
    if ev is None: DY, PAID, e = year_tables(col, drop_gt)
    else: DY, PAID, e = year_tables_ev(ev, col)
    ed = e['date'].to_numpy(); ej = e['j'].to_numpy(); eD = e[col].to_numpy()
    out = {}
    for d in SIGD:
        Y = d.year; i = pos(d)
        m_cur = (ed >= np.datetime64(f'{Y}-01-01')) & (ed <= np.datetime64(d))
        curD = np.zeros(N); np.add.at(curD, ej[m_cur], eD[m_cur])
        curP = np.zeros(N, bool); curP[ej[m_cur]] = True
        k = YI[Y]
        prev5 = DY[k - 5:k]; paid5 = PAID[k - 5:k].all(0)
        # Y*=Y：Y-4..Y-1 + 當年已除息；Y*=Y-1：Y-5..Y-1
        D5 = np.where(curP, (DY[k - 4:k].sum(0) + curD) / 5, prev5.sum(0) / 5)
        m_last = ed <= np.datetime64(d)
        lastd = pd.Series(ed[m_last]).groupby(ej[m_last]).max()
        days_since = np.full(N, 10**6.0)
        days_since[lastd.index.to_numpy()] = (d - pd.DatetimeIndex(lastd.to_numpy())).days.to_numpy()
        interrupted = (~PAID[k - 1]) | (days_since > 456)
        px = Cff[i]
        if share is not None: D5 = D5 * share[i]
        with np.errstate(all='ignore'):
            y5 = D5 / px
        liq = (V60[i] >= 1e7) & (Cn[i] >= 10) & is_common
        trad = np.zeros(N, bool)
        if d in EXI:
            ex = EXI[d]; trad = np.isfinite(L.OPEN_RAW[ex]) & np.isfinite(L.ADJ_RAW[ex])
        pool = liq & paid5 & ~interrupted & (D5 > 0) & np.isfinite(Cn[i]) & trad
        out[d] = {'y5': y5, 'pool': pool, 'interrupted': interrupted, 'stale': last_trade_age[i] >= 20, 'D5': D5}
    return out

GAP = M.GAP_UP

def build(sg, mode, cap=30):
    """mode: 'hyst' 主規格（≤5% 賣半、≤3% 全賣）或 'simple'（≤5% 全賣）。回傳 events={ex: 權重陣列}、diag。"""
    held = {}; events = {}; diag = []
    for d in SIGD:
        if d not in EXI: continue
        ex = EXI[d]; s = sg[d]; y5 = s['y5']; pool = s['pool']
        nb = 0; sold_all = sold_half = readd = 0
        new = {}
        for j, u in held.items():
            if s['interrupted'][j] or s['stale'][j] or not np.isfinite(y5[j]):
                sold_all += 1; continue
            if mode == 'hyst':
                if y5[j] <= 0.03: sold_all += 1; continue
                if y5[j] <= 0.05:
                    if u == 1.0: sold_half += 1
                    new[j] = 0.5; continue
                if u == 0.5 and y5[j] >= 0.07 and pool[j] and not GAP[ex, j]:
                    new[j] = 1.0; readd += 1; continue
                new[j] = u
            else:
                if y5[j] <= 0.05: sold_all += 1; continue
                new[j] = u
        cand = np.where(pool & (y5 >= 0.07))[0]
        cand = cand[np.argsort(-y5[cand])]
        nbuy = 0
        for j in cand:
            if cap is not None and len(new) >= cap: break
            if j in new: continue
            if GAP[ex, j]: nb += 1; continue
            new[j] = 1.0; nbuy += 1
        held = new
        w = np.zeros(N)
        if held:
            n = len(held)
            for j, u in held.items(): w[j] = u / n
        events[ex] = w
        diag.append({'d': str(d.date()), 'pool': int(pool.sum()), 'n_y5_ge7': int((pool & (y5 >= 0.07)).sum()), 'held': len(held),
                     'half': int(sum(u == 0.5 for u in held.values())), 'new_buys': nbuy, 'sold_all': sold_all, 'sold_half': sold_half,
                     'readd': readd, 'gap_blocked': nb, 'invested': round(float(w.sum()), 3)})
    return events, diag

E50n = L.E50.to_numpy()
def etf_rates(cost):
    if cost == 'gross': return 0.0, 0.0
    if cost == 'stress': return 0.005, 0.005
    return L.FEE + cost, L.FEE + 0.001 + cost

def sim(events, cost, idle, t0, t1):
    b, s = (0.0, 0.0) if cost == 'gross' else L.cost_rates(cost)
    eb, es = etf_rates(cost)
    u = np.zeros(N); z = 0.0; cash = 1.0; nav = np.empty(t1 - t0 + 1); tov = []
    for t in range(t0, t1 + 1):
        if t in events:
            w = events[t]; Pz = L.AO[t]; Pe = E50n[t - 1]
            h = np.where(u > 0, u * np.nan_to_num(Pz), 0.0); he = z * Pe
            Vt = h.sum() + he + cash; widle = max(1.0 - w.sum(), 0.0); Vn = Vt
            for _ in range(8):
                tgt = w * Vn; te = widle * Vn if idle == '0050' else 0.0
                cost_ = b * np.clip(tgt - h, 0, None).sum() + s * np.clip(h - tgt, 0, None).sum() + eb * max(te - he, 0) + es * max(he - te, 0)
                Vn = Vt - cost_
            tov.append(np.clip(h - tgt, 0, None).sum() / Vt if Vt > 0 else 0)
            u = np.zeros(N); nz = w > 0; u[nz] = tgt[nz] / Pz[nz]
            z = te / Pe if idle == '0050' else 0.0
            cash = widle * Vn if idle == 'cash' else 0.0
        nav[t - t0] = np.nansum(u * L.AC[t]) + z * E50n[t] + cash
    return pd.Series(nav, index=IDX[t0:t1 + 1]), float(np.mean(tov[1:])) if len(tov) > 1 else float('nan')

T_START = pos(SIGD[0]); T_END = len(IDX) - 1
SG = {'E1': signals('D_E1'), 'E3': signals('D_E3'), 'cap15': signals('D_E1', drop_gt=0.15)}

# ---------------- 事後診斷：clean 版股利（看過主結果後加，只報告、不進判定） ----------------
# 1) 2021-10 前：E3 估計，單一事件推估殖利率 > 15% 視為股票股利（不計現金、改當股數調整）
# 2) 2021-10 起：息＝value；權息的現金部分改用其後 60 天內第一個 pe_5y 的 TWSE 殖利率 × 當日收盤（找不到就不計）；權＝股數調整
# 3) 股數基礎調整：flag 3（減資／面額變更等，因子＝參考價／前收）與上述股票股利，之前的每股股利乘上之後的因子
RETn = L.RETN
pe_all = pd.read_csv(XR / 'pe_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
pe_all = pe_all[(pe_all['market'] == 'TWSE') & pe_all['code'].isin(COLS) & pe_all['date'].isin(IDX)].sort_values('date')
fac = np.ones((len(IDX), N))
with np.errstate(all='ignore'):
    f3 = np.where((FL == 3) & np.isfinite(RETn) & np.isfinite(Cprev) & np.isfinite(Cn), (Cn / (1 + RETn)) / Cprev, 1.0)
f3 = np.where(np.abs(f3 - 1) > 0.02, f3, 1.0); fac *= np.where(np.isfinite(f3), f3, 1.0)
yE3 = np.where(np.isfinite(oo), (cp - oo / (1 + mgap[ii])) / cp, E1 / cp)
stk = yE3 > 0.15
np.multiply.at(fac, (ii[stk], jj[stk]), np.clip(1 - yE3[stk], 0.05, 1))
ev_c = [pd.DataFrame({'date': IDX[ii[~stk]], 'j': jj[~stk], 'D': np.maximum(cp[~stk] * yE3[~stk], 0)})]
xk = exd.copy(); xk['j'] = [L.CI[c] for c in xk['code']]; xk['ti'] = IDX.searchsorted(xk['date'])
pe_y = {c: g[['date', 'yield']].to_numpy() for c, g in pe_all.groupby('code')}
rows_c = []; n_mixed_ok = n_mixed_miss = 0
for r in xk.itertuples():
    ti = min(r.ti, len(IDX) - 1)
    if r.kind == '息':
        rows_c.append((r.date, r.j, r.value)); continue
    if r.kind == '權':
        fac[ti, r.j] *= r.ref_price / r.prev_close; continue
    arr = pe_y.get(r.code); cash = np.nan
    if arr is not None:
        k = np.searchsorted(arr[:, 0].astype('datetime64[ns]'), np.datetime64(r.date))
        if k < len(arr) and (pd.Timestamp(arr[k, 0]) - r.date).days <= 60 and np.isfinite(arr[k, 1]):
            ci = pos(pd.Timestamp(arr[k, 0])); cash = arr[k, 1] / 100 * Cn[ci, r.j]
    if np.isfinite(cash) and 0 < cash < r.prev_close - r.ref_price:
        n_mixed_ok += 1; rows_c.append((r.date, r.j, cash))
        fac[ti, r.j] *= r.ref_price / (r.prev_close - cash)
    else:
        n_mixed_miss += 1; fac[ti, r.j] *= r.ref_price / r.prev_close
ev_c.append(pd.DataFrame(rows_c, columns=['date', 'j', 'D']))
EVC = pd.concat(ev_c, ignore_index=True).sort_values('date').reset_index(drop=True)
SHARE = np.cumprod(fac, axis=0)
ei = IDX.searchsorted(EVC['date']); ei = np.clip(ei - 1, 0, len(IDX) - 1)
EVC['Dc'] = EVC['D'].to_numpy() / SHARE[ei, EVC['j'].to_numpy()]
EVC['year'] = EVC['date'].dt.year
res['clean_posthoc_div'] = {'n_events': int(len(EVC)), 'n_pre_cut_treated_as_stock(E3>15%)': int(stk.sum()), 'n_mixed_cash_from_pe5y': n_mixed_ok,
                            'n_mixed_unresolved(cash=0)': n_mixed_miss, 'n_flag3_share_factors': int((f3 != 1).sum())}
SG['clean'] = signals('Dc', ev=EVC, share=SHARE)
sg = SG['E1']
pool_events = {}
for d in SIGD:
    if d not in EXI: continue
    p = sg[d]['pool']
    if p.sum() > 0: pool_events[EXI[d]] = ('open', np.where(p)[0])
# 引擎檢查：同池等權用 exlib 與本 sim 比對
w_events = {}
for ex, (_, pk) in pool_events.items():
    w = np.zeros(N); w[pk] = 1.0 / len(pk); w_events[ex] = w
nA, _ = sim(w_events, 0.001, 'cash', min(pool_events) - 1, T_END)
nB, _ = L.sim_portfolio(pool_events, *L.cost_rates(0.001), min(pool_events) - 1, T_END)
res['engine_check_max_abs_diff'] = float((nA - nB).abs().max())
EW_NAV, _ = sim(w_events, 'gross', 'cash', T_START, T_END)
E50 = L.E50
WIN = {'select_price_2010-01_2019-04': ('2010-01-01', '2019-04-30'), 'valid_price_2019-05_2026-10': ('2019-05-01', '2026-10-31'),
       'select_2021_2023': ('2021-01-01', '2023-12-31'), 'valid_2024_2026-10': ('2024-01-01', '2026-10-31'),
       'y2024_2025': ('2024-01-01', '2025-12-31'), 'full_2010_2026-10': ('2010-01-01', '2026-10-31')}

def stats_from(nav, first_exec_date):
    out = {}
    for w, (a, b) in WIN.items():
        a2 = max(pd.Timestamp(a), first_exec_date - pd.Timedelta(days=1))
        out[w] = M.stats(nav, a2, b, SIGD, bench=E50, ew=EW_NAV) if a2 < pd.Timestamp(b) else None
    return out

# pe_5y 交叉檢查：最近股利年度的股利 / 收盤 vs TWSE 公布殖利率
pe = pd.read_csv(XR / 'pe_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
pe = pe[(pe['market'] == 'TWSE') & pe['code'].isin(COLS) & pe['date'].isin(IDX)]
DY1, PAID1, _ = year_tables('D_E1')
xc = []
for d, g in pe.groupby('date'):
    i = pos(d); Y = d.year; k = YI[Y]
    m_cur = (EV['date'] >= f'{Y}-01-01') & (EV['date'] <= d)
    cur = EV[m_cur].groupby('j')['D_E1'].sum()
    for c, yl in zip(g['code'], g['yield']):
        j = L.CI[c]
        if not np.isfinite(yl) or not np.isfinite(Cn[i, j]) or Cn[i, j] <= 0: continue
        if j in cur.index: dlast, ylast = cur[j], Y
        else: dlast, ylast = DY1[k - 1, j], Y - 1
        xc.append({'date': d, 'y_twse': yl / 100, 'y_ours': dlast / Cn[i, j], 'year': ylast})
xc = pd.DataFrame(xc)
xco = {}
for nm, g in {'last_div_year<=2020(面板推估)': xc[xc['year'] <= 2020], 'last_div_year=2021(混合)': xc[xc['year'] == 2021], 'last_div_year>=2022(exdiv_5y)': xc[xc['year'] >= 2022]}.items():
    gg = g[(g['y_twse'] > 0) & (g['y_ours'] > 0)]
    if len(gg) < 10: continue
    xco[nm] = {'n': int(len(g)), 'corr': round(float(g['y_twse'].corr(g['y_ours'])), 3), 'spearman': round(float(g['y_twse'].rank().corr(g['y_ours'].rank())), 3),
               'median_ratio_ours_over_twse': round(float((gg['y_ours'] / gg['y_twse']).median()), 3),
               'share_abs_diff_gt_1pct': round(float((g['y_ours'] - g['y_twse']).abs().gt(0.01).mean()), 3),
               'share_twse0_ours_gt0': round(float(((g['y_twse'] == 0) & (g['y_ours'] > 0)).mean()), 3)}
res['pe5y_crosscheck'] = xco

sl = pd.read_csv(SL, dtype={'code': str}).drop_duplicates('code').set_index('code')['name']
IND = M.IND.reindex(COLS).fillna('未分類')
navs = {'same_pool_EW': EW_NAV}
RUNS = [('E1', 'hyst', 'cash'), ('E1', 'hyst', '0050'), ('E1', 'simple', 'cash'), ('E1', 'simple', '0050'),
        ('E3', 'hyst', 'cash'), ('E3', 'hyst', '0050'), ('E3', 'simple', 'cash'), ('cap15', 'hyst', 'cash'),
        ('E1', 'hyst_nocap', 'cash'), ('E3', 'hyst_nocap', 'cash'), ('clean', 'hyst_posthoc', 'cash'), ('clean', 'hyst_posthoc', '0050'), ('clean', 'simple_posthoc', 'cash')]   # 最後兩個是事後診斷（看過主結果後加）：不設 30 檔上限、所有符合者等權，只報告、不進判定
for est, mode, idle in RUNS:
    events, diag = build(SG[est], mode.replace('_nocap', '').replace('_posthoc', ''), cap=None if mode.endswith('_nocap') else 30)
    first_exec = IDX[min(t for t, w in events.items() if w.sum() > 0)]
    name = f'{est}|{mode}|idle_{idle}'; R = {'first_exec': str(first_exec.date())}
    for cost in ('gross', 0.001, 'stress'):
        nav, tov = sim(events, cost, idle, T_START, T_END)
        navs[f'{name}|{cost}'] = nav
        R[f'cost_{cost}'] = {'turnover_per_rebalance': round(tov, 3), 'windows': stats_from(nav, first_exec), 'yearly_pct': M.yearly(nav.loc[first_exec - pd.Timedelta(days=1):])}
    dg = pd.DataFrame(diag)
    R['holdings'] = {'mean_held': round(float(dg['held'].mean()), 1), 'max_held': int(dg['held'].max()), 'months_at_30': int((dg['held'] == 30).sum()),
                     'months_empty': int((dg['held'] == 0).sum()), 'n_months': int(len(dg)), 'mean_invested': round(float(dg['invested'].mean()), 3),
                     'mean_half': round(float(dg['half'].mean()), 2), 'gap_blocked_total': int(dg['gap_blocked'].sum()),
                     'mean_pool': round(float(dg['pool'].mean()), 1), 'mean_n_y5_ge7': round(float(dg['n_y5_ge7'].mean()), 1),
                     'by_year_mean_held': dg.groupby(dg['d'].str[:4])['held'].mean().round(1).to_dict(),
                     'by_year_mean_invested': dg.groupby(dg['d'].str[:4])['invested'].mean().round(2).to_dict(),
                     'by_year_mean_pool': dg.groupby(dg['d'].str[:4])['pool'].mean().round(0).to_dict()}
    if (est, mode, idle) in (('E1', 'hyst', 'cash'), ('clean', 'hyst_posthoc', 'cash')):
        if est == 'E1': R['diag'] = diag
        rows = []; exs = sorted(events)
        for a_, b_ in zip(exs, exs[1:] + [T_END]):
            w = events[a_]
            dsig = IDX[a_ - 1]
            for j in np.where(w > 0)[0]:
                p0, p1 = L.AO[a_, j], (L.AO[b_, j] if b_ != T_END else L.AC[b_, j])
                rows.append({'code': COLS[j], 'name': sl.get(COLS[j], ''), 'industry': IND[COLS[j]], 'exec': str(IDX[a_].date()), 'w': w[j],
                             'gross': p1 / p0 - 1 if np.isfinite(p0) and np.isfinite(p1) else np.nan, 'y5': SG[est][dsig]['y5'][j] if dsig in SG[est] else np.nan,
                             'flag345': bool(np.isin(FL[a_:b_ + 1, j], [3, 4, 5]).any())})
        hd = pd.DataFrame(rows); hd.to_csv(HERE / ('holdings_main.csv.gz' if est == 'E1' else 'holdings_clean_posthoc.csv.gz'), index=False)
        R['holding_detail'] = {'n_stock_months': int(len(hd)), 'n_unique_codes': int(hd['code'].nunique()), 'n_flag345': int(hd['flag345'].sum()),
                               'n_gross_gt_50pct': int((hd['gross'] > 0.5).sum()), 'n_gross_lt_m50pct': int((hd['gross'] < -0.5).sum()),
                               'share_financial': round(float((hd['industry'] == '金融保險業').mean()), 3),
                               'industry_mix_top10': hd['industry'].value_counts().head(10).to_dict(),
                               'most_held_top15': {f"{c}{sl.get(c, '')}": int(n) for c, n in hd['code'].value_counts().head(15).items()},
                               'median_y5_at_signal': round(float(hd['y5'].median()), 4), 'n_y5_gt20pct': int((hd['y5'] > 0.2).sum()),
                               'by_period_unique': {p: int(hd[(hd['exec'] >= a) & (hd['exec'] <= b)]['code'].nunique()) for p, (a, b) in WIN.items()},
                               'by_period_share_y5_gt15pct': {p: round(float((hd[(hd['exec'] >= a) & (hd['exec'] <= b)]['y5'] > 0.15).mean()), 3) for p, (a, b) in WIN.items()},
                               'by_period_median_y5': {p: round(float(hd[(hd['exec'] >= a) & (hd['exec'] <= b)]['y5'].median()), 4) for p, (a, b) in WIN.items()}}
        bad = hd[hd['y5'] > 0.2].drop_duplicates('code')[['code', 'name', 'exec', 'y5']].head(15)
        R['holding_detail']['y5_gt20pct_examples'] = bad.to_dict('records')
    res[name] = R
    w = R['cost_0.001']['windows']
    print(name, {k: (v['CAGR'], v['t_vs_EW_m'], v['t_vs_0050_m'], v['0050_CAGR'], v['EW_CAGR']) if v else None for k, v in w.items()}, R['holdings']['mean_held'], R['holdings']['months_empty'])
res['same_pool_EW'] = {'yearly_pct': M.yearly(EW_NAV), 'windows': {w: M.stats(EW_NAV, a, b, SIGD, bench=E50) for w, (a, b) in WIN.items()}}
res['0050_yearly_pct'] = M.yearly(E50.loc[EW_NAV.index])
pd.DataFrame(navs).to_parquet(HERE / 'nav.parquet')

def ok(w): return bool(w is not None and (w['t_vs_EW_m'] >= 2) and (w['CAGR'] >= w['0050_CAGR']))
verd = {}
for name in [f'{e}|{m}|idle_{i}' for e, m, i in RUNS]:
    mw = res[name]['cost_0.001']['windows']
    p1, p2 = ok(mw['select_price_2010-01_2019-04']), ok(mw['valid_price_2019-05_2026-10'])
    q1, q2 = ok(mw['select_2021_2023']), ok(mw['valid_2024_2026-10'])
    verd[name] = {'price_select_pass': p1, 'price_valid_pass': p2, 'split2021_select_pass': q1, 'split2021_valid_pass': q2,
                  'verdict_price_split': '可用' if p1 and p2 else ('待觀察' if p1 or p2 else '不可用')}
res['verdicts_all_variants'] = verd
res['verdicts_note'] = '只有 E1|hyst|idle_cash 是判定用主規格；*_nocap、clean|*_posthoc 是看過主結果後加的事後診斷'
res['verdict_main'] = verd['E1|hyst|idle_cash']
res['runtime_s'] = round(time.time() - T0, 1)

def conv(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    return str(o)
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=conv), encoding='utf-8')
print('done', res['runtime_s'], res['verdict_main'], 'engine diff', res['engine_check_max_abs_diff'])
