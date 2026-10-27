#!/usr/bin/env python3
"""Robeco 保守公式（van Vliet & Blitz 2018, "The Conservative Formula: Quantitative Investing Made Easy"）
台股上市簡化版：低波動（36 個月月報酬標準差）→ 淨配息率（只有現金殖利率）＋ 12-1 動能 → 前 20%，季換股。
執行：python3 -I run.py。PREREG 先寫入 results.json 才開始計算任何策略結果。
成交、成本、漲停判定沿用 momentum_family/mlib.py → exec_rebuild/exlib.py（只讀取、不修改）。
股利推估邏輯複製自 w3_dividend_lowvol/run.py（E3 面板推估 ＋ 2021-10-07 起 exdiv_5y；不 import 它）。"""
import json, sys, time, math, datetime
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'momentum_family'))
import mlib as M  # noqa: E402
L = M.L
T0 = time.time()
XR = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras')
SL = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv')
W3NAV = HERE.parent / 'w3_dividend_lowvol' / 'nav.parquet'

PREREG = {
 'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
 '寫於看結果前': True,
 'rule_name': 'Robeco 保守公式（van Vliet–Blitz 2018：低波動＋淨配息率＋動能）',
 'sources': ['https://alphaarchitect.com/the-conservative-formula-quantitative-investing-made-easy/',
             'https://robeco.com/es-latam/vision-de-mercado/2022/05/conservative-investing-stands-the-test-of-time',
             'https://en.wikipedia.org/wiki/Conservative_Formula_Investing',
             'https://marcellus.in/story/long-term-compounding-using-low-volatility-other-fundamental-factors/',
             'https://momentumlab.substack.com/p/exploring-the-conservative-formula',
             'https://blog.validea.com/long-term-compounding-using-low-volatility-other-fundamental-factors/'],
 'source_rule_verbatim': '股池：每季底（3、6、9、12 月最後交易日），流動性 ≥ NT$1,000 萬、價格 ≥ 10 元的上市股中，依成交值（代替市值）取前 1,000 檔（台股不足就全取）。'
                         '(1) 用過去 36 個月月報酬標準差算波動（論文預設；最少 24 個月），保留波動最低的 50%；'
                         '(2) 在這一半裡算兩個排名：淨配息率＝過去 12 個月現金股利 ÷ 現價（缺買回資料，只用現金殖利率），以及 12-1 動能（t-12～t-2 月累積報酬）；'
                         '(3) 兩個排名平均，取前 20%（對應原版 500 檔取 100 檔）；另報固定 30 檔版本；(4) 等權，季底收盤算訊號、隔日開盤成交，每季換股。',
 'deviation_from_paper': [
   '原論文：美國最大 1,000 檔（市值），net payout yield＝(現金股利＋淨買回)/市值。台股沒有庫藏股買回資料 → 淨配息率只剩「現金殖利率」這一半，這是殖利率版、不是完整 NPY。',
   '市值用 252 日日均成交值代替（股本只有 2025-09 起）。台股流動性合格上市股遠少於 1,000 檔 → 實際上是全取。',
   '論文換股頻率是季度、等權（Robeco 保守公式原文 100 檔等權、每季）；這裡照做，只做多。',
   '2021-10 前現金股利由面板除息缺口反推，無法區分現金與股票股利（2005～2012 配股盛行時殖利率偏高）。'],
 'type': '選股型（季換股）：組合報告（年化、日頻最大回撤、對 0050 月 t、對同池等權月 t、逐年、周轉）',
 'data': {
   'panel': 'scratchpad/panel 上市日線（含已下市；2019–2023 只有存活者）。有值檔數 < 300 的日子（2018-01-13）先剔除，才算月底日曆、252 日均量與任何回看。',
   'codes': '4 碼數字代碼、排除 91xx（存託憑證）。',
   'month_end': '剔除稀疏日後，每個日曆月最後一個交易日 ME（2024-01 無資料，因此 2023-12→2024-02 是一個兩個月的報酬，照實計算）。',
   'monthly_return': 'MR[m] = adj(ME[m]) / adj(ME[m-1]) − 1；任一端 adj 缺值 → 缺值。資料異常保護：MR ≥ +200% 或 ≤ −80% 視為缺值（事前固定，報告筆數）。',
 },
 'dividends': {
   '2004-01～2021-10-06（面板推估 E3，主規格）': '除權息日（flag 1 或 2、前一交易日有收盤）。每股股利 D = 前收 − 當日開盤／(1＋當日全市場開盤跳空中位數)；無開盤用 前收−收盤；D<0 記 0。（w4_sun_7pct_dividend_bible 以 exdiv_5y 驗證 MAE 約 1% 股價。）',
   '2021-10-07 起（exdiv_5y，TWSE）': 'kind=息 → 現金股利＝value；kind=權 → 不算；kind=權息 → value 全額當現金（上限）。',
   'flag 3/4/5': '不當作股利。',
   'TTM 現金殖利率 NPY': 'NPY(d) = Σ{除息日 ∈ (ME[m-12], d]} D ÷ close_raw(d)；沒有除息 → 0（論文中無配息者 NPY=0，一樣參與排名）。',
   'pe_5y 校對（2021-09 起，只報告）': '每個季底 d 的 NPY vs d 之後第一個 pe_5y 日期的 TWSE yield（官方：最近年度股利／價格），報 Spearman 與中位數比值。',
   '敏感度（事前固定，只報告）': ['E1：D = 前收 − 當日收盤', 'cap15：E3 下單一事件推估殖利率 > 15% 視為非現金（股票股利／公司行動），不計入'],
 },
 'signals_at_d（季底 d = ME[m]，m 為 3/6/9/12 月）': {
   'VOL36': 'std(MR[m-35..m], ddof=1)，有效月數 ≥ 24，否則無值。',
   'MOM12_1': 'Π(1+MR[m-11..m-1]) − 1（t-12～t-2，跳過最近一個月 MR[m]），11 個月都要有值。',
   'NPY': '見 dividends。',
 },
 'universe（d 收盤可知）': [
   '普通股、d 有收盤、flag(d) ∈ {0,1,2}',
   'mlib.V60(d) ≥ NT$1,000 萬、close_raw(d) ≥ 10 元',
   '上述股票依 ADV252（剔除稀疏日後 252 日日均成交值，min_periods 200）由大到小取前 1,000 檔（不足全取；ADV252 無值者排在最後仍可入選，若總數 ≤ 1,000）',
   'd+1 有開盤價與 adj（可成交）',
   'VOL36 與 MOM12_1 都有值 → 「排序股池 U」＝同池等權對手的股池。U < 50 檔該季不換股（持有舊組合）。'],
 'selection': {
   'step1': 'U 依 VOL36 由小到大，取前 floor(50%×|U|) 檔 = 低波動半 LV。',
   'step2': 'LV 內：NPY 百分位排名（高＝好，同值取平均排名）、MOM12_1 百分位排名（高＝好）。',
   'step3': '分數 = 兩個排名平均；由高到低排序（同分以 VOL36 較低者優先）。',
   'CF20（主規格、判定用）': '取前 k = floor(20%×|LV|) 檔，等權。',
   'CF30（另報）': '取前 30 檔，等權。',
   'tradable': 'd+1 開盤漲停（mlib.GAP_UP：2015-06-01 前 7%、之後 10%，容差 0.6%）且未持有 → 買不到，由排序下一名遞補，維持目標檔數；已持有者照常保留。',
 },
 'components（事前固定，只報告、不進判定）': {
   'LV_EW': 'LV 全部等權（只用低波動）',
   'LV_NPY20': 'LV 內只用 NPY 排名取前 20%',
   'LV_MOM20': 'LV 內只用 MOM 排名取前 20%',
   'ALL_NPYMOM': '不先篩低波動：U 全部用 NPY＋MOM 平均排名取同樣 k 檔',
   'CF20_E1 / CF20_cap15': '股利推估敏感度',
 },
 'execution': {
   'calendar': '季底 d 收盤算訊號 → d+1 開盤成交（exlib.sim_portfolio，AO 還原開盤）；持有到下一季底隔日開盤換股；期間不調整（權重漂移）。第一個訊號 2006-12-29（有 24 個月以上月報酬）→ 2007-01-02 開盤。最後一個訊號 2026-09-30，持有到面板最後一日 2026-10-22。',
   '停牌／下市': '依 exlib：最後成交價無限期 ffill，下次換股以該價賣出。',
   '面板缺漏': '2023-12-29 訊號的下一個面板交易日是 2024-02-15，就在該日開盤成交（所有對手同樣跨過缺口）。',
 },
 '成本': 'exlib.cost_rates：買 0.0855%＋滑價、賣 0.0855%＋證交稅 0.3%＋滑價。主＝滑價 0.1%（來回約 0.67%，≥ 題目 0.5%）；壓力＝買 0.5%＋賣 0.5%（來回 1%）；另報未扣成本。周轉以漂移後權重計。',
 '對手': {'0050': '含息日頻（finlab etf_0050.csv）', '同池等權 EW_U（判定用）': '排序股池 U 全部等權、同換股日、同引擎、不扣成本',
         'EW_LV（另報）': 'LV 等權（＝components.LV_EW 的未扣成本版）', 'w3_dividend_lowvol': '已測紅利低波主規格（年度、等權、E3、滑價 0.1%）日頻淨值，讀 w3_dividend_lowvol/nav.parquet（只讀）'},
 '統計': '日頻 NAV：年化（實際天數）、波動、日頻最大回撤、逐年。t＝扣成本後「月底收盤→下月底收盤」月超額報酬（每月一個觀察值＝按月份分群）對 0050、同池等權、w3。',
 '期間': {'價格類挑選期': ['2005-01-01', '2019-04-30'], '價格類驗證期': ['2019-05-01', '2026-10-31'], '2021 起切分挑選': ['2021-01-01', '2023-12-31'],
         '2021 起切分驗證': ['2024-01-01', '2026-10-31'], '另報': ['2024-01-01', '2025-12-31'], '全期': ['2005-01-01', '2026-10-31'],
         '說明': '價格類挑選期實際從 2007-01 起（先累積 36 個月月報酬，最少 24）。本規則只用價格與除息資料（2004 起都有）→ 判定以價格類切分為準；2021 起切分附註。面板 2019～2023 只有存活股（偏多）。'},
 '判定': '主規格 CF20、滑價 0.1%：價格類挑選期與驗證期都「對同池等權月 t ≥ 2 且年化 ≥ 0050 年化」→ 可用；只有一段過 → 待觀察；都沒過 → 不可用。資料不足 → 無法測試。壓力成本與其他版本只報告。',
 '資料檢查（事前列出）': ['每季：股池、LV、k、實選數、NPY=0 比例、漲停擋下數', 'MR 異常保護筆數', '持股產業分布（金融股比重）、最常入選名單、入選股 NPY／VOL36／MOM 中位數 vs U',
                    'NPY vs pe_5y 官方殖利率（2021-09 起）', '持有期 flag 3/4/5 與單期報酬 > 100%', 'U 規模 vs 1,000 上限'],
}
res = {'PREREG': PREREG}
OUT = HERE / 'results.json'
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')

IDX, COLS, N, pos = L.IDX, L.COLS, L.N, L.pos
C = L.C; Cn = L.CN; On = L.OPEN_RAW; adj = L.adj; V = L.V; RET = L.RET
FLAG = pd.read_parquet(L.P / 'flag.parquet').reindex(index=IDX, columns=COLS)
FL = FLAG.to_numpy()
CUT = pd.Timestamp('2021-10-07')

# ---------------- 股利事件（複製 w3_dividend_lowvol） ----------------
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
res['div_events'] = {'n_pre_cut': int(len(ev_pre)), 'n_post_cut_cash_or_mixed': int(len(ev_post)),
                     'pre_cut_share_E3_yield_gt15pct': round(float((ev_pre['yE3'] > .15).mean()), 4)}

def div_matrix(col, drop_gt=None):
    """日頻股利矩陣（除息日 × 股票，原始每股金額）。"""
    e = EV if drop_gt is None else EV[~(EV['yE3'] > drop_gt)]
    g = e.groupby(['date', 'j'])[col].sum()
    Dm = np.zeros((len(IDX), N))
    di = IDX.get_indexer(g.index.get_level_values(0))
    ok = di >= 0
    np.add.at(Dm, (di[ok], g.index.get_level_values(1)[ok]), g.values[ok])
    res.setdefault('div_events_not_on_panel_day', {})[f'{col}|{drop_gt}'] = int((~ok).sum())
    return np.cumsum(Dm, axis=0)       # 累積，方便取區間和

DCUM = {'E3': div_matrix('D_E3'), 'E1': div_matrix('D_E1'), 'cap15': div_matrix('D_E3', 0.15)}

# ---------------- 月底日曆與月報酬 ----------------
GOOD = adj.notna().sum(axis=1) >= 300
res['sparse_days_removed'] = [str(d.date()) for d in IDX[~GOOD]]
GI = IDX[GOOD.to_numpy()]
_s = pd.Series(GI, index=GI)
ME = pd.DatetimeIndex([g.iloc[-1] for _, g in _s.groupby([GI.year, GI.month])])
AM = adj.reindex(ME)
MR = AM / AM.shift(1) - 1
bad = (MR >= 2.0) | (MR <= -0.8)
res['MR_outlier_set_nan'] = int(bad.to_numpy().sum())
MR = MR.mask(bad)
MRn = MR.to_numpy()
VOL36 = MR.rolling(36, min_periods=24).std()
lg = np.log1p(MR)
MOM = np.expm1(lg.shift(1).rolling(11, min_periods=11).sum())     # MR[m-11..m-1]
ADV252 = V.loc[GOOD].rolling(252, min_periods=200).mean().reindex(IDX)
V60 = M.V60.reindex(IDX)
is_common = np.array([(len(c) == 4 and c.isdigit() and not c.startswith('91')) for c in COLS])

QE = [d for d in ME if d.month in (3, 6, 9, 12) and pd.Timestamp('2006-12-01') <= d < IDX[-1]]
assert QE[0] == pd.Timestamp('2006-12-29'), QE[0]
MEPOS = {d: k for k, d in enumerate(ME)}
GAP = M.GAP_UP

def npy_at(d, est):
    m = MEPOS[d]; i = pos(d); i0 = pos(ME[m - 12])
    D = DCUM[est][i] - DCUM[est][i0]          # 除息日 ∈ (ME[m-12], d]
    with np.errstate(all='ignore'):
        return D / Cn[i]

def pool_at(d):
    i = pos(d); ex = i + 1
    base = is_common & np.isfinite(Cn[i]) & np.isin(FL[i], [0, 1, 2]) & (V60.loc[d].to_numpy() >= 1e7) & (Cn[i] >= 10)
    adv = ADV252.loc[d].to_numpy()
    idx = np.where(base)[0]
    idx = idx[np.argsort(-np.nan_to_num(adv[idx], nan=-1), kind='stable')][:1000]
    top = np.zeros(N, bool); top[idx] = True
    trad = np.isfinite(L.OPEN_RAW[ex]) & np.isfinite(L.ADJ_RAW[ex])
    vol = VOL36.loc[d].to_numpy(); mom = MOM.loc[d].to_numpy()
    U = top & trad & np.isfinite(vol) & np.isfinite(mom)
    return {'ex': ex, 'U': U, 'vol': vol, 'mom': mom, 'n_base': int(base.sum()), 'n_top': int(top.sum()), 'n_trad': int((top & trad).sum())}

def pct_rank(x):
    return pd.Series(x).rank(pct=True, method='average').to_numpy()

def take(order, ex, held, n):
    pick = []; nb = 0
    for j in order:
        if GAP[ex, j] and j not in held: nb += 1; continue
        pick.append(j)
        if len(pick) == n: break
    return np.array(pick, int), nb

VARIANTS = ['CF20', 'CF30', 'LV_EW', 'LV_NPY20', 'LV_MOM20', 'ALL_NPYMOM', 'CF20_E1', 'CF20_cap15']

def build(variant):
    est = 'E1' if variant.endswith('_E1') else ('cap15' if variant.endswith('_cap15') else 'E3')
    events = {}; ewU = {}; ewLV = {}; diag = []; held = set(); plog = []
    for d in QE:
        p = pool_at(d); ex = p['ex']; U = p['U']
        uidx = np.where(U)[0]
        if len(uidx) < 50:
            diag.append({'d': str(d.date()), 'skipped': True, 'U': int(len(uidx))}); continue
        vol, mom = p['vol'], p['mom']; npy = npy_at(d, est); npy = np.where(np.isfinite(npy), npy, 0.0)
        o = uidx[np.argsort(vol[uidx], kind='stable')]
        nlv = int(math.floor(0.5 * len(uidx))); LV = o[:nlv]
        k = int(math.floor(0.2 * nlv))
        def score_order(S, use):
            rN, rM = pct_rank(npy[S]), pct_rank(mom[S])
            sc = (rN + rM) / 2 if use == 'both' else (rN if use == 'npy' else rM)
            return S[np.lexsort((vol[S], -sc))]        # 分數高者先、同分低波動先
        if variant in ('CF20', 'CF20_E1', 'CF20_cap15'): order, n = score_order(LV, 'both'), k
        elif variant == 'CF30': order, n = score_order(LV, 'both'), 30
        elif variant == 'LV_EW': order, n = LV, nlv
        elif variant == 'LV_NPY20': order, n = score_order(LV, 'npy'), k
        elif variant == 'LV_MOM20': order, n = score_order(LV, 'mom'), k
        elif variant == 'ALL_NPYMOM': order, n = score_order(uidx, 'both'), k
        pick, nb = take(order, ex, held, n)
        events[ex] = ('open', pick); ewU[ex] = ('open', uidx); ewLV[ex] = ('open', LV)
        held = set(pick.tolist())
        diag.append({'d': str(d.date()), 'exec': str(IDX[ex].date()), 'n_base': p['n_base'], 'n_top1000': p['n_top'], 'n_trad': p['n_trad'], 'U': int(len(uidx)),
                     'LV': nlv, 'k_target': n, 'n_pick': int(len(pick)), 'gap_blocked': nb,
                     'share_npy0_U': round(float((npy[uidx] <= 0).mean()), 3), 'share_npy0_pick': round(float((npy[pick] <= 0).mean()), 3) if len(pick) else None,
                     'med_npy_U': round(float(np.median(npy[uidx])), 4), 'med_npy_pick': round(float(np.median(npy[pick])), 4) if len(pick) else None,
                     'med_vol_U_ann': round(float(np.median(vol[uidx]) * math.sqrt(12)), 4), 'med_vol_pick_ann': round(float(np.median(vol[pick]) * math.sqrt(12)), 4) if len(pick) else None,
                     'med_mom_U': round(float(np.median(mom[uidx])), 4), 'med_mom_pick': round(float(np.median(mom[pick])), 4) if len(pick) else None})
        plog.append((d, ex, pick, npy[pick], vol[pick], mom[pick]))
    return events, ewU, ewLV, diag, plog

T_START = pos(QE[0]); T_END = len(IDX) - 1
def simc(events, cost):
    b, s = (0.0, 0.0) if cost == 'gross' else L.cost_rates(cost)
    return L.sim_portfolio(events, b, s, T_START, T_END)

# ---------------- 統計（與 w3 同口徑：月底→月底月超額） ----------------
E50 = L.E50.loc[IDX[T_START]:]
MEC = pd.DatetimeIndex([d for d in ME if d >= IDX[T_START]])
def mrets(nav):
    x = nav.reindex(MEC); r = x / x.shift(1) - 1
    r.index = r.index.to_period('M'); return r.iloc[1:]

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
        bp = mrets(bn).reindex(pr.index); ex = (pr - bp).dropna()
        if len(ex) < 6: continue
        o[f'excess_vs_{nm}_pct_m'] = round(float(ex.mean() * 100), 2); o[f't_vs_{nm}_m'] = L.tstat(ex); o[f'n_m_{nm}'] = int(len(ex))
        d2 = dstats(bn.dropna(), max(pd.Timestamp(a), bn.dropna().index[0]), b)
        if d2: o[f'{nm}_CAGR'] = d2['CAGR']; o[f'{nm}_maxDD'] = d2['maxDD']
    return o

def yearly(nav):
    r = nav.pct_change().dropna()
    return {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in r.groupby(r.index.year)}

# ---------------- 對手 ----------------
w3 = pd.read_parquet(W3NAV)['annual|ew|E3|pre|0.001'].reindex(IDX).loc[IDX[T_START]:]
w3 = w3 / w3.iloc[0]
sl = pd.read_csv(SL, dtype={'code': str}).drop_duplicates('code').set_index('code')['name']
IND = M.IND.reindex(COLS).fillna('未分類')

navs = {}
ev0, ewU, ewLV, diag0, _ = build('CF20')
EWU = simc(ewU, 'gross')[0]; EWLV = simc(ewLV, 'gross')[0]
navs['EW_U|gross'] = EWU; navs['EW_LV|gross'] = EWLV
benches = {'0050': E50, 'EW': EWU, 'EWLV': EWLV, 'w3': w3}
res['bench'] = {nm: {'yearly_pct': yearly(n), 'windows': {w: wstats(n, a, b, {'0050': E50}) for w, (a, b) in WIN.items()}}
                for nm, n in [('EW_U', EWU), ('EW_LV', EWLV), ('w3_dividend_lowvol_main_net', w3)]}
res['bench']['0050'] = {'yearly_pct': yearly(E50), 'windows': {w: dstats(E50, a, b) for w, (a, b) in WIN.items()}}

for var in VARIANTS:
    events, _, _, diag, plog = build(var)
    R = {}
    for cost in ('gross', 0.001, 'stress'):
        nav, tov = simc(events, cost); navs[f'{var}|{cost}'] = nav
        R[f'cost_{cost}'] = {'turnover_per_rebalance': round(tov, 3), 'windows': {w: wstats(nav, a, b, benches) for w, (a, b) in WIN.items()}, 'yearly_pct': yearly(nav)}
    dg = pd.DataFrame([x for x in diag if not x.get('skipped')])
    R['diag_summary'] = {'n_rebalances': int(len(dg)), 'n_skipped': int(sum(1 for x in diag if x.get('skipped'))),
                         'U_min_mean_max': [int(dg['U'].min()), round(float(dg['U'].mean()), 1), int(dg['U'].max())],
                         'n_top1000_max': int(dg['n_top1000'].max()), 'k_min_mean_max': [int(dg['k_target'].min()), round(float(dg['k_target'].mean()), 1), int(dg['k_target'].max())],
                         'min_pick': int(dg['n_pick'].min()), 'gap_blocked_total': int(dg['gap_blocked'].sum()),
                         'mean_share_npy0_U': round(float(dg['share_npy0_U'].mean()), 3), 'mean_share_npy0_pick': round(float(dg['share_npy0_pick'].mean()), 3),
                         'median_of_med_npy_U_vs_pick': [round(float(dg['med_npy_U'].median()), 4), round(float(dg['med_npy_pick'].median()), 4)],
                         'median_of_med_vol_U_vs_pick': [round(float(dg['med_vol_U_ann'].median()), 4), round(float(dg['med_vol_pick_ann'].median()), 4)],
                         'median_of_med_mom_U_vs_pick': [round(float(dg['med_mom_U'].median()), 4), round(float(dg['med_mom_pick'].median()), 4)]}
    if var in ('CF20', 'CF30'):
        R['diag'] = diag
        rows = []; exs = [x[1] for x in plog] + [T_END]
        for k, (d, ex, pk, nn, vv, mm) in enumerate(plog):
            nx = exs[k + 1]
            for j, n_, v_, m_ in zip(pk, nn, vv, mm):
                p0 = L.AO[ex, j]; p1 = L.AO[nx, j] if nx != T_END else L.AC[nx, j]
                rows.append({'signal': str(d.date()), 'code': COLS[j], 'name': sl.get(COLS[j], ''), 'industry': IND[COLS[j]],
                             'npy': round(float(n_), 4), 'vol36_ann': round(float(v_ * math.sqrt(12)), 4), 'mom12_1': round(float(m_), 4),
                             'gross': p1 / p0 - 1 if np.isfinite(p0) and np.isfinite(p1) else np.nan,
                             'flag345': bool(np.isin(FL[ex:nx + 1, j], [3, 4, 5]).any())})
        hd = pd.DataFrame(rows)
        if var == 'CF20': hd.to_csv(HERE / 'holdings_CF20.csv.gz', index=False)
        R['holding_detail'] = {'n_stock_periods': int(len(hd)), 'n_unique': int(hd['code'].nunique()), 'n_flag345': int(hd['flag345'].sum()),
                               'n_gross_gt_100pct': int((hd['gross'] > 1).sum()), 'n_gross_lt_m50pct': int((hd['gross'] < -0.5).sum()),
                               'mean_gross_flag345': round(float(hd.loc[hd['flag345'], 'gross'].mean()), 4) if hd['flag345'].any() else None,
                               'share_financial': round(float((hd['industry'] == '金融保險業').mean()), 3),
                               'share_financial_by_signal_year': hd.assign(f=hd['industry'] == '金融保險業').groupby(hd['signal'].str[:4])['f'].mean().round(2).to_dict(),
                               'industry_top10': hd['industry'].value_counts().head(10).to_dict(),
                               'most_held_top20': {f"{c}{sl.get(c, '')}": int(n) for c, n in hd['code'].value_counts().head(20).items()},
                               'npy_gt15pct_picks': int((hd['npy'] > 0.15).sum()),
                               'latest_picks': hd[hd['signal'] == hd['signal'].max()][['code', 'name', 'industry', 'npy', 'vol36_ann', 'mom12_1']].to_dict('records')}
    res[var] = R
    w = R['cost_0.001']['windows']
    print(var, {k: (v['CAGR'], v.get('t_vs_EW_m'), v.get('t_vs_0050_m'), v.get('0050_CAGR'), v.get('EW_CAGR')) if v else None for k, v in w.items()}, flush=True)

# 持股重疊：CF20 vs w3 主規格持股（同年 12 月訊號）
try:
    h3 = pd.read_csv(HERE.parent / 'w3_dividend_lowvol' / 'holdings_main.csv.gz', dtype={'code': str})
    hc = pd.read_csv(HERE / 'holdings_CF20.csv.gz', dtype={'code': str})
    ov = {}
    for sig, g in hc[hc['signal'].str[5:7] == '12'].groupby('signal'):
        g3 = h3[h3['signal'] == sig]
        if len(g3): ov[sig] = {'n_cf': int(len(g)), 'n_w3': int(len(g3)), 'overlap': int(len(set(g['code']) & set(g3['code'])))}
    res['overlap_with_w3_dec_signals'] = ov
except Exception as e:  # noqa
    res['overlap_with_w3_dec_signals'] = f'error {e}'

# 月報酬相關（CF20 淨 vs w3、EW_U、0050）
a_ = mrets(navs['CF20|0.001'])
res['corr_monthly_CF20'] = {nm: round(float(a_.corr(mrets(bn).reindex(a_.index))), 3) for nm, bn in benches.items()}

# pe_5y 校對（2021-09 起季底）
pe = pd.read_csv(XR / 'pe_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
pe = pe[(pe['market'] == 'TWSE') & pe['code'].isin(COLS)]
pdates = np.sort(pe['date'].unique())
xc = {}
for d in QE:
    if d < pd.Timestamp('2021-09-01'): continue
    nxt = pdates[pdates > np.datetime64(d)]
    if not len(nxt): continue
    g = pe[pe['date'] == nxt[0]]
    p = pool_at(d); j = np.array([L.CI[c] for c in g['code']]); U = p['U'][j]
    ours = npy_at(d, 'E3')[j]; tw = g['yield'].to_numpy() / 100
    ok = U & np.isfinite(ours) & np.isfinite(tw)
    pos2 = ok & (ours > 0) & (tw > 0)
    xc[str(d.date())] = {'pe_date': str(pd.Timestamp(nxt[0]).date()), 'n': int(ok.sum()),
                         'spearman_all': round(float(pd.Series(ours[ok]).rank().corr(pd.Series(tw[ok]).rank())), 3),
                         'median_ratio_pos': round(float(np.median(ours[pos2] / tw[pos2])), 3),
                         'zero_agree': round(float(((ours[ok] <= 0) == (tw[ok] <= 0)).mean()), 3)}
res['pe5y_crosscheck_npy'] = xc

pd.DataFrame(navs).to_parquet(HERE / 'nav.parquet')

def ok(w): return bool(w is not None and w.get('t_vs_EW_m', -9) >= 2 and w['CAGR'] >= w['0050_CAGR'])
verd = {}
for var in VARIANTS:
    mw = res[var]['cost_0.001']['windows']
    p1, p2 = ok(mw['select_price_2005-01_2019-04']), ok(mw['valid_price_2019-05_2026-10'])
    q1, q2 = ok(mw['select_2021_2023']), ok(mw['valid_2024_2026-10'])
    verd[var] = {'price_select_pass': p1, 'price_valid_pass': p2, 'split2021_select_pass': q1, 'split2021_valid_pass': q2,
                 'verdict_price_split': '可用' if p1 and p2 else ('待觀察' if p1 or p2 else '不可用')}
res['verdicts_all_variants'] = verd
res['verdict_main'] = {'spec': 'CF20 滑價 0.1%', **verd['CF20']}
res['runtime_s'] = round(time.time() - T0, 1)

def conv(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    return str(o)
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=conv), encoding='utf-8')
print('done', res['runtime_s'], res['verdict_main'])
