#!/usr/bin/env python3
"""存股估價：歷年股價法（高低價法，版本 A）／股利法（版本 B）— 便宜價買、昂貴價賣 — 可成交回測（上市，2014-01～2026-10；5 年版 2009-01 起）。
執行：python3 -I run.py。PREREG 先寫入 results.json 才開始計算策略結果。
成交、成本、漲停判定、60 日均值修補沿用 momentum_family/mlib.py → exec_rebuild/exlib.py（只讀取、不修改）。
股利推估（E1 比值法＋exdiv_5y）複製自 w4_sun_7pct_dividend_bible/run.py（不改原檔），兩題共用同一套股利資料。"""
import json, sys, time, warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
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
    '來源規則': '存股族估價（CMoney 筆記 183445／113435／24725、uptogo 1893421、App「定存股合理價」）：'
              '歷年股價法＝近 N 年「每年最低價」平均為便宜價、「每年平均價」平均為合理價、「每年最高價」平均為昂貴價；'
              '股利法＝近 10 年平均現金股利 ×15 便宜價、×20 合理價、×30 昂貴價。股價 ≤ 便宜價買進，≥ 昂貴價賣出，介於中間續抱不加碼。',
    '股池（買進資格；也是同池等權的股池）': ['面板上市普通股（4 碼數字、排除 91xx）',
        '訊號日 t：mlib.V60（剔除 2018-01-13 等稀疏日後滾動的 60 日均成交值）≥ NT$1,000 萬、close_raw ≥ 10 元（BRIEF 流動性門檻）',
        '過去 K=10 個完整曆年（Y−10..Y−1）每年都有現金除息且年度股利合計 > 0（存股族條件）',
        't+1 有開盤價與還原價'],
    '股利資料': {
        '2021-10-07 前': '面板除權息日（flag 1/2，且前一交易日有收盤＝排除「不比價」）D = 前收 − 當日收盤（E1，adj/close_raw 比值法；面板把除息日報酬設 0），D<0 記 0。flag 3/4/5 不當股利。',
        '2021-10-07 起': 'exdiv_5y（TWSE）：息 → value；權 → 不算；權息 → value 全額（上限）。',
        '已知限制（事前列出）': ['2021-10 前無法分開現金股利與股票股利／現增除權 → 股票股利被當成現金股利，配股公司 D 偏高、版本 B 便宜價偏高、股池條件偏寬',
                         'E1 對 exdiv_5y 真值 MAE 約 1.8% 股價（見 sun 題 estimator_validation）',
                         '股利為原始每股金額，與教學用 Goodinfo 原始數字一致'],
        '股利年度': '除息日所在曆年。t（年 Y）若 Y 年 1/1～t 已有現金除息 → 最近股利年度 Y*=Y（只計已除息部分），否則 Y*=Y−1；D10＝Y*−9..Y* 十個年度現金股利平均（未發年度算 0，但股池已要求 Y−10..Y−1 每年都發）。'},
    '版本 A：歷年股價法（主：N=7、原始收盤）': {
        '價格': '面板沒有最低價 → 用 close_raw（原始收盤，教學用原始股價）每個完整曆年的最低收盤／平均收盤／最高收盤；該年需 ≥ 100 個有收盤的交易日，N 年任一年不足 → 當月無價帶（不可新買；持有中則續抱）。',
        '便宜價': 'Y−N..Y−1 各年最低收盤的平均（因用收盤而非盤中最低，便宜價略高 → 較容易觸發買進）',
        '合理價': '各年平均收盤的平均（只報告，不參與交易）',
        '昂貴價': '各年最高收盤的平均',
        '敏感度（事前固定）': ['N=10', '還原價版：各年用 adj（含息還原）算最低／平均／最高，再乘 close_raw(t)/adj(t) 換回今日價位（過去價位因股利還原而較低 → 便宜價更嚴）'],
    },
    '版本 B：股利法（主：×15 買、×30 賣）': {
        '便宜價': 'D10 × 15', '合理價': 'D10 × 20（只報告）', '昂貴價': 'D10 × 30',
        '倍數敏感度（事前固定，全部報告）': '便宜倍數 ∈ {15,16,20} × 昂貴倍數 ∈ {20,30,32}，便宜 < 昂貴：15/20、15/30（主）、15/32、16/20、16/30、16/32、20/30、20/32',
        'cap15 診斷': '單一事件推估殖利率 > 15% 視為非現金（股票股利／公司行動）不計入，主倍數 15/30',
    },
    '共同交易規則': {
        '訊號與成交': '每月最後一個交易日 t 收盤後計算 → t+1 開盤成交（M2）。t+1 開盤漲停（mlib.GAP_UP，2015-06 前 7%、之後 10%）時不能新買，由下一名遞補。t+1 距 t > 10 個日曆日（2024-01～02 面板缺漏）該月不換股。',
        '買進': '股池內、未持有、close_raw(t) ≤ 便宜價。候選依 close/便宜價 由低到高（折價最深優先）補到最多 30 檔；已持有者優先保留。',
        '賣出': '持有中 close_raw(t) ≥ 昂貴價 → 全賣。持有中不因離開股池、配息中斷而賣（教學只有昂貴價賣出）；最近 20 個交易日都無收盤（下市／長期停牌）→ 以最後成交價賣出。',
        '續抱': '便宜價 < 收盤 < 昂貴價：續抱不加碼。',
        '權重（主）': '持有名單等權（每月換股日調回 1/n，n ≤ 30）；完全沒有持股時 100% 持有 0050 含息（以 t 收盤近似成交，因無 0050 開盤價）。',
        '權重敏感度 drift（事前固定）': '「不加碼」的字面版：已持有者只會被按比例減碼、從不加碼；新買每檔 1/n（n＝換股後檔數），資金先用賣股所得與 0050 閒置部位，不足時按比例減碼舊持股；剩餘閒置資金放 0050。',
        '持有期': '無固定持有期，只依昂貴價或下市出場（每月檢查）。',
    },
    '5 年版（事前固定的延長樣本敏感度，題目建議）': '股池改為 Y−5..Y−1 每年都有現金股利、股利法用 D5（同 Y* 規則）；版本 A 用 N=5（另報 N=7）。訊號 2009-01 起（A N=7 為 2011-01 起），可涵蓋 2009～2019-04 挑選期。只報告，不取代主判定。',
    '成本': 'exlib.cost_rates：買 0.0855%+滑價、賣 0.0855%+稅 0.3%+滑價；主＝滑價 0.1%、壓力＝買 0.5%+賣 0.5%；另報未扣成本。0050：買 0.0855%+滑價、賣 0.0855%+ETF 稅 0.1%+滑價。',
    '對手': '0050 含息（日頻）；同池等權＝同一股池全部等權、同引擎、每月 t+1 開盤換股、不扣成本（與 w4_sun、w1_pe_river_low 同口徑）。',
    '統計': '日頻 NAV：年化、日頻最大回撤、逐年；t＝訊號日收盤 → 下一訊號日收盤的月窗超額（每月一個觀察值＝按月份分群），扣成本後，對 0050 與同池等權。',
    '期間': {'價格類挑選期': ['2005-01-01', '2019-04-30'], '價格類驗證期': ['2019-05-01', '2026-10-31'],
             '2021 起資料挑選期': ['2021-01-01', '2023-12-31'], '2021 起資料驗證期': ['2024-01-01', '2026-10-31'], '另報': ['2024-01-01', '2025-12-31'],
             '說明': '主規格要 10 年股利＋價格歷史（面板 2004 起）→ 第一個訊號 2014-01 月底，價格類挑選期實際只有 2014-02～2019-04（約 5.3 年）。面板 2019–2023 只有存活股（偏多）。'},
    '判定': '各版本主規格（A：N=7 原始收盤、等權；B：15/30、等權；滑價 0.1%）：價格類挑選期與驗證期都「對同池等權月窗 t ≥ 2 且年化 ≥ 0050」→ 可用；只一段過 → 待觀察；否則不可用。研究整體結論取兩個主規格中較好者（並註明做了 2 個主檢定）。其餘版本、2021 起切分、壓力成本只報告。',
    '資料檢查（事前列出）': ['每月股池檔數、低於便宜價檔數、持有檔數、全 0050 月數、開盤漲停擋下數、平均持有月數',
                       '持有期 flag 3/4/5、單期報酬 > 50%、close/便宜價 < 0.4 的極端訊號（原始價受減資／配股影響）',
                       '原始價 vs 還原價價帶差異', '產業分布（金融股比重）、最常持有名單', '版本 B 主規格與 sun 題主規格月報酬相關', '引擎與 exlib.sim_portfolio 比對'],
}
res = {'PREREG': PREREG}
OUT = HERE / 'results.json'
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')

IDX, COLS, N, pos = L.IDX, L.COLS, L.N, L.pos
C = L.C; Cn = L.CN; On = L.OPEN_RAW
FL = pd.read_parquet(L.P / 'flag.parquet').reindex(index=IDX, columns=COLS).to_numpy()
CUT = pd.Timestamp('2021-10-07')

# ---------------- 股利事件（同 sun 題 E1） ----------------
Cprev = np.vstack([np.full((1, N), np.nan), Cn[:-1]])
isX = np.isin(FL, [1, 2]); pre_mask = (IDX < CUT)[:, None]
xr = np.where(isX & pre_mask & np.isfinite(Cn)); prevok = np.isfinite(Cprev[xr])
ii, jj = xr[0][prevok], xr[1][prevok]
E1 = Cprev[ii, jj] - Cn[ii, jj]
ev_pre = pd.DataFrame({'date': IDX[ii], 'j': jj, 'D': np.maximum(E1, 0), 'yld': E1 / Cprev[ii, jj]})
exd = pd.read_csv(XR / 'exdiv_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
exd = exd[(exd['market'] == 'TWSE') & exd['code'].isin(COLS) & (exd['date'] >= CUT)]
ep = exd[exd['kind'].isin(['息', '權息'])]
ev_post = pd.DataFrame({'date': ep['date'].values, 'j': [L.CI[c] for c in ep['code']], 'D': ep['value'].values, 'yld': (ep['value'] / ep['prev_close']).values})
EV = pd.concat([ev_pre, ev_post], ignore_index=True).sort_values('date').reset_index(drop=True); EV['year'] = EV['date'].dt.year
res['div_events'] = {'n_pre_cut': int(len(ev_pre)), 'n_post_cut_cash': int(len(ev_post))}
YEARS = np.arange(2004, 2027); YI = {y: k for k, y in enumerate(YEARS)}

def year_tables(e):
    DY = np.zeros((len(YEARS), N))
    for (y, j), v in e.groupby(['year', 'j'])['D'].sum().items(): DY[YI[y], j] = v
    return DY

# ---------------- 年度價格表 ----------------
yr = IDX.year
def ystats(df):
    g = df.groupby(yr)
    return {k: g.agg(k).reindex(YEARS).reindex(columns=COLS).to_numpy() for k in ('min', 'mean', 'max', 'count')}
YS_RAW = ystats(C); YS_ADJ = ystats(L.adj)
Cff = C.ffill().to_numpy(); Aff = L.adj.ffill().to_numpy()

# ---------------- 訊號日 ----------------
_s = pd.Series(IDX, index=IDX)
ME = [g.iloc[-1] for _, g in _s.groupby([IDX.year, IDX.month])]
ME = [d for d in ME if d >= pd.Timestamp('2009-01-01')]
SIGD = pd.DatetimeIndex(ME)
EXI = {}
for d in SIGD:
    i = pos(d)
    if i + 1 < len(IDX) and (IDX[i + 1] - d).days <= 10: EXI[d] = i + 1
is_common = np.array([(len(c) == 4 and c.isdigit() and not c.startswith('91')) for c in COLS])
last_trade_age = np.zeros((len(IDX), N), int); _age = np.full(N, 10**6)
for t in range(len(IDX)):
    _age = np.where(np.isfinite(Cn[t]), 0, _age + 1); last_trade_age[t] = _age
V60 = M.V60.reindex(IDX).to_numpy()

def base_signals(K, e):
    """回傳每個訊號日：pool（K 年連續配息＋流動性）、DK（K 年平均股利）、px、stale。"""
    DY = year_tables(e); ed = e['date'].to_numpy(); ej = e['j'].to_numpy(); eD = e['D'].to_numpy()
    out = {}
    for d in SIGD:
        Y = d.year; i = pos(d); k = YI[Y]
        if k - K < 0: continue
        m_cur = (ed >= np.datetime64(f'{Y}-01-01')) & (ed <= np.datetime64(d))
        curD = np.zeros(N); np.add.at(curD, ej[m_cur], eD[m_cur])
        curP = np.zeros(N, bool); curP[ej[m_cur][eD[m_cur] > 0]] = True
        DK = np.where(curP, (DY[k - K + 1:k].sum(0) + curD) / K, DY[k - K:k].sum(0) / K)
        paidK = (DY[k - K:k] > 0).all(0)
        liq = (V60[i] >= 1e7) & (Cn[i] >= 10) & is_common
        trad = np.zeros(N, bool)
        if d in EXI:
            ex = EXI[d]; trad = np.isfinite(L.OPEN_RAW[ex]) & np.isfinite(L.ADJ_RAW[ex])
        out[d] = {'pool': liq & paidK & (DK > 0) & np.isfinite(Cn[i]) & trad, 'DK': DK, 'px': Cff[i], 'stale': last_trade_age[i] >= 20,
                  'n_liq': int((liq & np.isfinite(Cn[i])).sum())}
    return out

def band_price(d, Nyr, kind):
    Y = d.year; k = YI[Y]; i = pos(d)
    if k - Nyr < 0: return None
    ys = YS_RAW if kind == 'raw' else YS_ADJ
    ok = (ys['count'][k - Nyr:k] >= 100).all(0)
    f = 1.0 if kind == 'raw' else Cff[i] / Aff[i]
    with np.errstate(all='ignore'):
        cheap = np.where(ok, np.nanmean(ys['min'][k - Nyr:k], 0) * f, np.nan)
        fair = np.where(ok, np.nanmean(ys['mean'][k - Nyr:k], 0) * f, np.nan)
        exp = np.where(ok, np.nanmean(ys['max'][k - Nyr:k], 0) * f, np.nan)
    return cheap, fair, exp

GAP = M.GAP_UP
def build(bs, bandf, cap=30):
    """bandf(d, s) -> (cheap, fair, exp)。回傳 events={ex: (kept idx, new idx)}、diag、持有段。"""
    held = []; events = {}; diag = []; entry = {}; spells = []
    for d in SIGD:
        if d not in EXI or d not in bs: continue
        b = bandf(d, bs[d])
        if b is None: continue
        ex = EXI[d]; s = bs[d]; cheap, fair, exp = b; px = s['px']
        kept = []; sold = 0
        for j in held:
            if s['stale'][j] or (np.isfinite(exp[j]) and px[j] >= exp[j]):
                sold += 1; spells.append((entry.pop(j), d, 'stale' if s['stale'][j] else 'exp')); continue
            kept.append(j)
        with np.errstate(all='ignore'):
            ratio = px / cheap
        cmask = s['pool'] & np.isfinite(ratio) & (ratio <= 1.0)
        cand = np.where(cmask)[0]; cand = cand[np.argsort(ratio[cand])]
        new = []; nb = 0; hs = set(kept)
        for j in cand:
            if len(kept) + len(new) >= cap: break
            if j in hs: continue
            if GAP[ex, j]: nb += 1; continue
            new.append(j); entry[j] = d
        held = kept + new
        events[ex] = (np.array(kept, int), np.array(new, int))
        pool = s['pool']
        with np.errstate(all='ignore'):
            pr = px / fair
        diag.append({'d': str(d.date()), 'pool': int(pool.sum()), 'n_le_cheap': int(cmask.sum()), 'n_ge_exp_in_pool': int((pool & (px >= exp)).sum()),
                     'n_band_ok_in_pool': int((pool & np.isfinite(cheap)).sum()), 'held': len(held), 'new': len(new), 'sold': sold, 'gap_blocked': nb,
                     'median_px_over_fair_pool': round(float(np.nanmedian(pr[pool])), 3) if pool.any() else None,
                     'n_ratio_lt_0p4': int((cmask & (ratio < 0.4)).sum())})
    for j, d0 in entry.items(): spells.append((d0, None, 'open'))
    return events, diag, spells

E50n = L.E50.to_numpy()
def etf_rates(cost):
    if cost == 'gross': return 0.0, 0.0
    if cost == 'stress': return 0.005, 0.005
    return L.FEE + cost, L.FEE + 0.001 + cost

def sim(events, cost, t0, t1, wmode='eq'):
    """events={ex: (kept, new)}；eq＝持有名單等權；drift＝舊持股不加碼；無持股 → 0050。"""
    b, s = (0.0, 0.0) if cost == 'gross' else L.cost_rates(cost)
    eb, es = etf_rates(cost)
    u = np.zeros(N); z = 0.0; nav = np.empty(t1 - t0 + 1); tov = []; z_share = []
    for t in range(t0, t1 + 1):
        if t in events:
            kept, new = events[t]; Pz = L.AO[t]; Pe = E50n[t - 1]
            h = np.where(u > 0, u * np.nan_to_num(Pz), 0.0); he = z * Pe
            Vt = h.sum() + he; Vn = Vt; n = len(kept) + len(new)
            for _ in range(8):
                tgt = np.zeros(N)
                if n > 0:
                    if wmode == 'eq':
                        tgt[np.concatenate([kept, new])] = Vn / n
                    else:
                        tgt[new] = Vn / n
                        hk = h[kept].sum(); room = Vn - len(new) * Vn / n
                        if hk > 0: tgt[kept] = h[kept] * min(1.0, room / hk)
                te = max(Vn - tgt.sum(), 0.0)
                cost_ = b * np.clip(tgt - h, 0, None).sum() + s * np.clip(h - tgt, 0, None).sum() + eb * max(te - he, 0) + es * max(he - te, 0)
                Vn = Vt - cost_
            tov.append(np.clip(h - tgt, 0, None).sum() / Vt if Vt > 0 else 0)
            u = np.zeros(N); nz = tgt > 0; u[nz] = tgt[nz] / Pz[nz]; z = te / Pe; z_share.append(te / Vn if Vn > 0 else 0)
        elif t == t0:
            z = 1.0 / E50n[t]
        nav[t - t0] = np.nansum(u * L.AC[t]) + z * E50n[t]
    return pd.Series(nav, index=IDX[t0:t1 + 1]), float(np.mean(tov[1:])) if len(tov) > 1 else float('nan'), float(np.mean(z_share)) if z_share else float('nan')

T_END = len(IDX) - 1
WIN = {'select_price_2005-01_2019-04': ('2005-01-01', '2019-04-30'), 'valid_price_2019-05_2026-10': ('2019-05-01', '2026-10-31'),
       'select_2021_2023': ('2021-01-01', '2023-12-31'), 'valid_2024_2026-10': ('2024-01-01', '2026-10-31'),
       'y2024_2025': ('2024-01-01', '2025-12-31'), 'full': ('2005-01-01', '2026-10-31')}
E50 = L.E50
sl = pd.read_csv(SL, dtype={'code': str}).drop_duplicates('code').set_index('code')['name']
IND = M.IND.reindex(COLS).fillna('未分類')

def ew_nav(bs, t_start):
    ev = {}
    for d in SIGD:
        if d in EXI and d in bs and bs[d]['pool'].sum() > 0:
            ev[EXI[d]] = (np.where(bs[d]['pool'])[0], np.array([], int))
    nav, _, _ = sim(ev, 'gross', t_start, T_END)
    return nav, ev

BS = {'K10': base_signals(10, EV), 'K5': base_signals(5, EV), 'K10cap15': base_signals(10, EV[~(EV['yld'] > 0.15)])}
EWS = {}
for k, bs in BS.items():
    d0 = min(bs); EWS[k] = ew_nav(bs, pos(d0))
# 引擎檢查：同池等權 exlib vs 本 sim（滑價 0.1%）
evc = EWS['K10'][1]; t0c = min(evc) - 1
nA, _, _ = sim(evc, 0.001, t0c, T_END)
nB, _ = L.sim_portfolio({k: ('open', v[0]) for k, v in evc.items()}, *L.cost_rates(0.001), t0c, T_END)
_m = nA.index > IDX[min(evc)]   # 本 sim 起始日持有 0050、exlib 起始為現金 → 比對第一次換股後的日報酬
res['engine_check_max_abs_daily_ret_diff'] = float((nA.pct_change()[_m] - nB.pct_change()[_m]).abs().max())

def fA(Nyr, kind): return lambda d, s: band_price(d, Nyr, kind)
def fB(m1, m2):
    return lambda d, s: (s['DK'] * m1, s['DK'] * 20, s['DK'] * m2)

RUNS = [  # (name, base, bandf, wmode, main?)
    ('A|N7|raw|eq', 'K10', fA(7, 'raw'), 'eq', True),
    ('A|N10|raw|eq', 'K10', fA(10, 'raw'), 'eq', False),
    ('A|N7|adj|eq', 'K10', fA(7, 'adj'), 'eq', False),
    ('A|N7|raw|drift', 'K10', fA(7, 'raw'), 'drift', False),
    ('B|15/30|eq', 'K10', fB(15, 30), 'eq', True),
] + [(f'B|{a}/{b}|eq', 'K10', fB(a, b), 'eq', False) for a, b in [(15, 20), (15, 32), (16, 20), (16, 30), (16, 32), (20, 30), (20, 32)]] + [
    ('B|15/30|drift', 'K10', fB(15, 30), 'drift', False),
    ('B|15/30|cap15|eq', 'K10cap15', fB(15, 30), 'eq', False),
    ('L5|A|N5|raw|eq', 'K5', fA(5, 'raw'), 'eq', False),
    ('L5|A|N7|raw|eq', 'K5', fA(7, 'raw'), 'eq', False),
    ('L5|B|15/30|eq', 'K5', fB(15, 30), 'eq', False),
]
navs = {f'EW_{k}': v[0] for k, v in EWS.items()}
def ok(w): return bool(w is not None and (w['t_vs_EW_m'] >= 2) and (w['CAGR'] >= w['0050_CAGR']))
verd = {}
for name, bk, bandf, wmode, main in RUNS:
    bs = BS[bk]; events, diag, spells = build(bs, bandf)
    t_start = min(events) - 1; first_exec = IDX[min(events)]
    EWn = EWS[bk][0]
    R = {'base': bk, 'main': main, 'first_exec': str(first_exec.date())}
    for cost in ('gross', 0.001, 'stress'):
        nav, tov, zs = sim(events, cost, t_start, T_END, wmode)
        navs[f'{name}|{cost}'] = nav
        wins = {}
        for w, (a, b) in WIN.items():
            a2 = max(pd.Timestamp(a), IDX[t_start])
            wins[w] = M.stats(nav, a2, b, SIGD, bench=E50, ew=EWn) if a2 < pd.Timestamp(b) else None
        R[f'cost_{cost}'] = {'turnover_per_rebalance': round(tov, 3), 'mean_0050_share': round(zs, 3), 'windows': wins, 'yearly_pct': M.yearly(nav)}
    dg = pd.DataFrame(diag)
    durs = [((e if e is not None else SIGD[-1]) - s0).days / 30.44 for s0, e, _ in spells]
    R['holdings'] = {'mean_pool': round(float(dg['pool'].mean()), 1), 'mean_n_le_cheap': round(float(dg['n_le_cheap'].mean()), 1),
                     'mean_held': round(float(dg['held'].mean()), 1), 'months_at_30': int((dg['held'] == 30).sum()), 'months_all_0050': int((dg['held'] == 0).sum()),
                     'n_months': int(len(dg)), 'gap_blocked_total': int(dg['gap_blocked'].sum()), 'n_spells': len(spells),
                     'median_hold_months': round(float(np.median(durs)), 1) if durs else None,
                     'share_exit_exp': round(float(np.mean([x[2] == 'exp' for x in spells])), 3) if spells else None,
                     'share_still_open': round(float(np.mean([x[2] == 'open' for x in spells])), 3) if spells else None,
                     'signals_ratio_lt_0p4': int(dg['n_ratio_lt_0p4'].sum()),
                     'by_year_mean_held': dg.groupby(dg['d'].str[:4])['held'].mean().round(1).to_dict(),
                     'by_year_mean_pool': dg.groupby(dg['d'].str[:4])['pool'].mean().round(0).to_dict(),
                     'by_year_mean_n_le_cheap': dg.groupby(dg['d'].str[:4])['n_le_cheap'].mean().round(1).to_dict(),
                     'by_year_median_px_over_fair': dg.groupby(dg['d'].str[:4])['median_px_over_fair_pool'].median().round(3).to_dict()}
    if main or name.startswith('L5|B'):
        rows = []; exs = sorted(events)
        for a_, b_ in zip(exs, exs[1:] + [T_END]):
            kept, new = events[a_]
            for j in np.concatenate([kept, new]).astype(int):
                p0, p1 = L.AO[a_, j], (L.AO[b_, j] if b_ != T_END else L.AC[b_, j])
                rows.append({'code': COLS[j], 'name': sl.get(COLS[j], ''), 'industry': IND[COLS[j]], 'exec': str(IDX[a_].date()),
                             'gross': p1 / p0 - 1 if np.isfinite(p0) and np.isfinite(p1) else np.nan,
                             'flag345': bool(np.isin(FL[a_:b_ + 1, j], [3, 4, 5]).any())})
        hd = pd.DataFrame(rows)
        if main: hd.to_csv(HERE / f"holdings_{name.split('|')[0]}.csv.gz", index=False)
        R['holding_detail'] = {'n_stock_months': int(len(hd)), 'n_unique_codes': int(hd['code'].nunique()) if len(hd) else 0,
                               'n_flag345': int(hd['flag345'].sum()) if len(hd) else 0, 'n_gross_gt_50pct': int((hd['gross'] > 0.5).sum()) if len(hd) else 0,
                               'share_financial': round(float((hd['industry'] == '金融保險業').mean()), 3) if len(hd) else None,
                               'industry_mix_top8': hd['industry'].value_counts().head(8).to_dict() if len(hd) else {},
                               'most_held_top12': {f"{c}{sl.get(c, '')}": int(n) for c, n in hd['code'].value_counts().head(12).items()} if len(hd) else {}}
        R['diag_sample'] = diag[::12]
    res[name] = R
    mw = R['cost_0.001']['windows']
    p1, p2 = ok(mw['select_price_2005-01_2019-04']), ok(mw['valid_price_2019-05_2026-10'])
    q1, q2 = ok(mw['select_2021_2023']), ok(mw['valid_2024_2026-10'])
    verd[name] = {'price_select_pass': p1, 'price_valid_pass': p2, 'split2021_select_pass': q1, 'split2021_valid_pass': q2,
                  'verdict_price_split': '可用' if p1 and p2 else ('待觀察' if p1 or p2 else '不可用')}
    print(name, {k: (v['CAGR'], v['t_vs_EW_m'], v['t_vs_0050_m'], v['0050_CAGR'], v['EW_CAGR']) if v else None for k, v in mw.items()},
          R['holdings']['mean_held'], R['holdings']['months_all_0050'], flush=True)
for k, (nv, _) in EWS.items():
    res[f'same_pool_EW_{k}'] = {'yearly_pct': M.yearly(nv), 'windows': {w: M.stats(nv, max(pd.Timestamp(a), nv.index[0]), b, SIGD, bench=E50) for w, (a, b) in WIN.items()}}
res['0050_yearly_pct'] = M.yearly(E50.loc[EWS['K10'][0].index])
res['verdicts_all_variants'] = verd
res['verdict_main'] = {'A': verd['A|N7|raw|eq'], 'B': verd['B|15/30|eq']}
# 與 sun 題相關
try:
    sn = pd.read_parquet(HERE.parent / 'w4_sun_7pct_dividend_bible' / 'nav.parquet')['E1|hyst|idle_cash|0.001']
    a = M.month_rets(navs['B|15/30|eq|0.001'], SIGD); b = M.month_rets(sn, SIGD); c = M.month_rets(navs['A|N7|raw|eq|0.001'], SIGD)
    j = pd.concat([a, b, c], axis=1, keys=['B', 'sun', 'A']).dropna()
    res['corr_monthly'] = {'B_vs_sun': round(float(j['B'].corr(j['sun'])), 3), 'A_vs_sun': round(float(j['A'].corr(j['sun'])), 3), 'A_vs_B': round(float(j['A'].corr(j['B'])), 3), 'n': int(len(j))}
except Exception as e:
    res['corr_monthly'] = str(e)
# 價帶診斷：原始 vs 還原 便宜價比
dchk = {}
for d in SIGD[SIGD.month == 12]:
    if d not in BS['K10']: continue
    p = BS['K10'][d]['pool']; r1 = band_price(d, 7, 'raw'); r2 = band_price(d, 7, 'adj')
    if r1 is None: continue
    with np.errstate(all='ignore'):
        dchk[d.year] = {'median_adj_cheap_over_raw_cheap': round(float(np.nanmedian((r2[0] / r1[0])[p])), 3),
                        'median_exp_over_cheap_raw': round(float(np.nanmedian((r1[2] / r1[0])[p])), 3),
                        'median_DK30_over_DK15': 2.0}
res['band_check_dec'] = dchk
pd.DataFrame(navs).to_parquet(HERE / 'nav.parquet')
res['runtime_s'] = round(time.time() - T0, 1)

def conv(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    return str(o)
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=conv), encoding='utf-8')
print('done', res['runtime_s'], res['verdict_main'], 'engine diff', res['engine_check_max_abs_daily_ret_diff'], res['corr_monthly'])
