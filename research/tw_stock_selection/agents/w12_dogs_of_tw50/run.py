#!/usr/bin/env python3
"""台灣版道瓊狗股（Dogs of the Dow）：成交值前 50 大（0050 代理）中，過去 12 個月現金殖利率最高 10 檔等權，每年換一次。
執行：python3 -I run.py。PREREG 先寫入 results.json，才開始計算任何策略結果。
成交、成本、漲停判定沿用 momentum_family/mlib.py → exec_rebuild/exlib.py（只讀取、不修改）。
股利推估邏輯參考 w4_sun_7pct_dividend_bible/run.py（複製寫法，不 import）。"""
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

PREREG = {
 'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
 '寫於看結果前': True,
 'rule_name': '台灣版道瓊狗股（Dogs of the Dow）：大型股中殖利率最高 10 檔、每年換一次',
 'sources': ['https://learn.stocktrak.com/beginners/strategies-for-beginners/dogs-of-dow-strategy/', 'https://www.zacks.com/stock-education/18/',
             'https://dividend.watch/lists/dogs-dow-stocks', 'https://www.investmentexecutive.com/?p=71495', 'https://freedomisntfree.co.uk/articles/dogs-of-the-dow'],
 'source_rule_verbatim': '原版：每年最後一個交易日，在道瓊 30 檔成分股中挑出股利殖利率（過去 12 個月股利 ÷ 年底股價）最高的 10 檔，'
                         '新年度第一個交易日等金額買進，持有一年，隔年同一時點重新篩選換股。Small Dogs（Puppies）：這 10 檔中股價最低的 5 檔。',
 'tw_translation': {
   '股池（0050 代理）': '每年最後一個交易日 d：面板上市普通股（4 碼數字、排除 91xx）中，d 有收盤、close_raw(d) ≥ 10 元、mlib.V60(d) ≥ NT$1,000 萬（BRIEF 流動性），'
                      '依 250 日均成交值 V250 排名取前 50 名。V250 = 剔除稀疏日（2018-01-13 等「有值檔數 < 中位數一半」的日子，mlib.SPARSE）後 rolling(250, min_periods=200) 的平均成交值。'
                      '沒有長期股本 → 無法用市值；成交值前 50 名偏向高周轉股（事前已知偏誤）。2025-12 用 shares_hist 市值、2026-09 用 etf0050_w 實際 0050 成分股交叉驗證（只報告）。',
   '殖利率': 'y(d) = 除息日 ∈ (d − 365 天, d] 的每股現金股利合計 ÷ close_raw(d)。y ≤ 0（過去 12 個月沒配現金）不入選。',
   '現金股利（主規格 E1，題目指定的 adj／close_raw 反推）': {
       '2004-01～2021-10-06': '除權息日（flag 1 或 2）且前一交易日有收盤（排除「不比價」X 日）：D = 前收 ×(1+ret) − 收盤，也就是 ret（含息報酬）與原始價報酬之差 × 前收。'
                              '面板在這段把除權息日 ret 設為 0，所以 D = 前收 − 收盤。D < 0 記 0。',
       '2021-10-07 起（exdiv_5y 校對）': 'exdiv_5y（market=TWSE、代碼在面板）：kind=息 → value；kind=權（純股票股利）→ 0；kind=權息 → 用 exdiv.csv 的 cash_div（2024-08 起才有），沒有就 value 全額當現金（上限）。',
       'flag 3/4/5': '減資、分割、資料缺口／錯誤的日子不當股利（只採 flag 1/2）。',
   },
   '敏感度（事前固定，只報告，不進判定）': ['E3：D = 前收 − 當日開盤 ÷ (1 + 當日全市場開盤跳空中位數)（w4 以 exdiv_5y 驗證 MAE 約 1% 股價，E1 約 1.8%）',
                                            'cap15：E1 下單一事件推估殖利率 > 15% 視為股票股利／公司行動，不計入',
                                            'mixed0：2021-10 後權息事件若沒有 cash_div → 現金記 0（下限）'],
   '選股': 'V250 前 50 名中，y 由高到低；元月第一個交易日（= d 的下一個面板交易日）開盤等權買進 10 檔。'
           '開盤漲停（mlib.GAP_UP：2015-06 前 7%、之後 10%，含息跳空 ≥ 漲停 −0.6%）且未持有 → 買不到，由下一名遞補（exlib／mlib 慣例）；已持有者照常保留。'
           '隔日沒有開盤價或還原價 → 不能買，下一名遞補。',
   '持有與換股': '持有 12 個月，下一年最後一個交易日重新篩選，下一個交易日開盤換股（同時把留任股調回等權）。持有期間不調整、不停損。停牌／下市股以最後成交價計，換股時以該價賣出（exlib 規則）。'
                 '2024-01-02～02-14 面板缺漏 → 2023-12-29 的訊號在 2024-02-15 開盤執行（原持股抱到那天），事前寫明。',
   '股利再投入': '用 adj（含息還原）計價，等同股利再投入原股。',
 },
 'variants（事前固定）': {
   'main': 'V250 前 50 → 殖利率前 10 檔等權',
   'a_small_dogs': '主規格那 10 檔（遞補後）中 close_raw(d) 最低的 5 檔等權',
   'b_pool30': 'V250 前 30 → 殖利率前 10 檔等權',
 },
 'cost': 'exlib.cost_rates：買 0.0855%＋滑價、賣 0.0855%＋證交稅 0.3%＋滑價；主 = 滑價 0.1%（來回約 0.67%），壓力 = 買 0.5%＋賣 0.5%（來回 1%）；另報未扣成本。周轉用漂移後權重。',
 'benchmarks': {'0050': 'finlab etf_0050.csv 含息（日頻）',
                '同池等權': '同一年的 V250 前 50（b 版另用前 30）全部等權（不篩殖利率；隔日可成交者），同引擎、同時點每年換股；主比較用未扣成本（研究中心慣例，對策略較嚴），另報扣 0.1% 滑價版',
                '0056（參考）': 'finlab etf_0056.csv 含息（2007-12-26 起），只報告'},
 'statistics': '日頻 NAV：年化（實際天數）、日頻最大回撤、逐年。t（判定用）＝扣成本後「月底收盤 → 下月底收盤」月超額報酬（每月一個觀察值＝按月份分群）對 0050、對同池等權。'
               '因為一年才換一次，另報「年度超額」t（每年一個觀察值，挑選期 2005～2018 共 14 年、驗證期 2019～2026 共 8 年，檢定力很低）。',
 'periods': {'價格類挑選期': ['2005-01-01', '2019-04-30'], '價格類驗證期': ['2019-05-01', '2026-10-31'],
             '2021 起資料挑選期': ['2021-01-01', '2023-12-31'], '2021 起資料驗證期': ['2024-01-01', '2026-10-31'], '另報': ['2024-01-01', '2025-12-31'],
             '全期': ['2005-01-01', '2026-10-31'],
             '說明': '年頻換股：挑選期約 14 次換股、驗證期約 7 次。2021 起切分只是報告（股利 2021-10 起才用 exdiv_5y）。'},
 'verdict_rule': '只看主規格（E1、前 50、前 10、滑價 0.1%）在價格類兩段：兩段都「對同池等權月 t ≥ 2 且年化 ≥ 0050 年化」→ 可用；只一段過 → 待觀察；都不過 → 不可用；資料不足 → 無法測試。變體、敏感度、壓力成本只報告。',
 'data_checks（事前列出）': ['每年股池 50 檔名單、與前一年重疊數', '每年 10 狗名單、殖利率、產業、金融股比重',
                            '殖利率 > 15% 的入選數（股票股利污染指標）', '2021-12～2025-12：E1 推估（前收−收盤）vs exdiv_5y 真值的殖利率 Spearman 與前 10 名重疊（量化 2021 前選股雜訊）',
                            '2025-12 V250 前 50 vs shares_hist 市值前 50 重疊；2026-09-25 V250 前 50 vs etf0050_w 實際成分股重疊',
                            '持有期 flag 3/4/5、單年報酬 > 100% 或 < −60%、開盤漲停擋下數、周轉'],
}
res = {'PREREG': PREREG}
OUT = HERE / 'results.json'
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')

# ================= 資料 =================
IDX, COLS, N, pos = L.IDX, L.COLS, L.N, L.pos
C = L.C; Cn = L.CN; On = L.OPEN_RAW; RETn = L.RETN
FLAG = pd.read_parquet(L.P / 'flag.parquet').reindex(index=IDX, columns=COLS); FL = FLAG.to_numpy()
CUT = pd.Timestamp('2021-10-07')
is_common = np.array([(len(c) == 4 and c.isdigit() and not c.startswith('91')) for c in COLS])
sl = pd.read_csv(SL, dtype={'code': str}).drop_duplicates('code').set_index('code')['name']
IND = M.IND.reindex(COLS).fillna('未分類')

# ---- 股利事件 ----
Cprev = np.vstack([np.full((1, N), np.nan), Cn[:-1]])
normal = (FL == 0) & np.isfinite(Cprev) & np.isfinite(Cn)
with np.errstate(all='ignore'):
    mgap = np.nanmedian(np.where(normal & np.isfinite(On), On / Cprev - 1, np.nan), axis=1)
isX = np.isin(FL, [1, 2]) & np.isfinite(Cprev) & np.isfinite(Cn)
pre = (IDX < CUT)[:, None]
ii, jj = np.where(isX & pre)
cp, cc, oo, rr = Cprev[ii, jj], Cn[ii, jj], On[ii, jj], RETn[ii, jj]
D_E1 = cp * (1 + np.nan_to_num(rr)) - cc
D_E3 = np.where(np.isfinite(oo), cp - oo / (1 + mgap[ii]), cp - cc)
ev_pre = pd.DataFrame({'date': IDX[ii], 'j': jj, 'E1': np.maximum(D_E1, 0), 'E3': np.maximum(D_E3, 0), 'yE1': D_E1 / cp})
ev_pre['cap15'] = np.where(ev_pre['yE1'] > 0.15, 0.0, ev_pre['E1']); ev_pre['mixed0'] = ev_pre['E1']
exd = pd.read_csv(XR / 'exdiv_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
exd = exd[(exd['market'] == 'TWSE') & exd['code'].isin(COLS) & (exd['date'] >= CUT)].copy()
exn = pd.read_csv(XR / 'exdiv.csv', dtype={'code': str}, parse_dates=['date'])
exn = exn[exn['market'] == 'TWSE'][['date', 'code', 'cash_div', 'stock_ratio']]
exd = exd.merge(exn, on=['date', 'code'], how='left')
cash_main = np.where(exd['kind'] == '息', exd['value'], np.where(exd['kind'] == '權', 0.0, np.where(exd['cash_div'].notna(), exd['cash_div'], exd['value'])))
cash_low = np.where(exd['kind'] == '息', exd['value'], np.where(exd['kind'] == '權', 0.0, np.where(exd['cash_div'].notna(), exd['cash_div'], 0.0)))
ev_post = pd.DataFrame({'date': exd['date'].values, 'j': [L.CI[c] for c in exd['code']], 'E1': cash_main, 'E3': cash_main, 'cap15': cash_main, 'mixed0': cash_low,
                        'yE1': (exd['value'] / exd['prev_close']).values})
EV = pd.concat([ev_pre, ev_post], ignore_index=True).sort_values('date').reset_index(drop=True)
res['div_events'] = {'n_pre_cut': int(len(ev_pre)), 'n_post_cut': int(len(ev_post)), 'post_cut_kinds': exd['kind'].value_counts().to_dict(),
                     'post_cut_mixed_with_cash_div': int(((exd['kind'] == '權息') & exd['cash_div'].notna()).sum()),
                     'pre_cut_share_yE1_gt15pct': round(float((ev_pre['yE1'] > .15).mean()), 4)}

# ---- 訊號日（每年最後一個交易日）與執行日 ----
_s = pd.Series(IDX, index=IDX)
YE = [g.iloc[-1] for _, g in _s.groupby(IDX.year)]
YE = [d for d in YE if d.year <= 2025]          # 2026 年底尚未到
SIG = pd.DatetimeIndex(YE)
EXI = {d: pos(d) + 1 for d in SIG}
res['calendar'] = {str(d.date()): str(IDX[EXI[d]].date()) for d in SIG}

V250 = L.V.drop(M.SPARSE).rolling(250, min_periods=200).mean().reindex(IDX).ffill(limit=1).to_numpy()
V60 = M.V60.reindex(IDX).to_numpy()
GAP = M.GAP_UP

_sh = pd.read_csv(XR / 'shares_hist.csv', dtype={'code': str})
SH25 = np.full(N, np.nan)
for c, v in _sh[_sh['ym'] == '2025-12'].set_index('code')['shares'].items():
    if c in L.CI: SH25[L.CI[c]] = v

def pool_at(d, k, mode='v250'):
    i = pos(d)
    base = is_common & np.isfinite(Cn[i]) & (Cn[i] >= 10) & (V60[i] >= 1e7) & np.isfinite(V250[i]) & np.isin(FL[i], [0, 1, 2])
    key = V250[i] if mode == 'v250' else SH25 * Cn[i]     # 事後診斷：2025-12 股數 × 當時原始價（存活偏誤、股數不隨時間調整）
    base = base & np.isfinite(key)
    v = np.where(base, key, -np.inf)
    top = np.argsort(-v)[:k]
    return top[np.isfinite(v[top])]

def yields_at(d, col, ev=EV):
    m = (ev['date'] > d - pd.Timedelta(days=365)) & (ev['date'] <= d)
    D = np.zeros(N); np.add.at(D, ev.loc[m, 'j'].to_numpy(), ev.loc[m, col].to_numpy())
    with np.errstate(all='ignore'):
        return D / Cn[pos(d)]

def build(col='E1', k=50, n=10, small=None, pmode='v250'):
    events, pools, picks = {}, {}, []; held = set(); nblk = 0
    for d in SIG:
        i = pos(d); ex = EXI[d]
        pl = pool_at(d, k, pmode)
        trad = np.isfinite(L.OPEN_RAW[ex]) & np.isfinite(L.ADJ_RAW[ex])
        y = yields_at(d, col)
        cand = [j for j in pl[np.argsort(-y[pl])] if y[j] > 0 and trad[j]]
        pk = []
        for j in cand:
            if GAP[ex, j] and j not in held:
                nblk += 1; continue
            pk.append(j)
            if len(pk) == n: break
        if small:
            pk = sorted(pk, key=lambda j: Cn[i, j])[:small]
        events[ex] = ('open', np.array(pk)); pools[ex] = ('open', np.array([j for j in pl if trad[j]]))
        picks.append({'signal': str(d.date()), 'exec': str(IDX[ex].date()), 'n_pool': int(len(pl)),
                      'picks': [{'code': COLS[j], 'name': sl.get(COLS[j], ''), 'ind': IND[COLS[j]], 'y': round(float(y[j]), 4), 'px': float(Cn[i, j])} for j in pk]})
        held = set(pk)
    return events, pools, picks, nblk

t_start = pos(SIG[0]); T_END = len(IDX) - 1
def run(events, cost):
    b, s = (0.0, 0.0) if cost == 'gross' else L.cost_rates(cost)
    return L.sim_portfolio(events, b, s, t_start, T_END)

E50 = L.E50
e56 = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0056.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()
E56 = e56.reindex(IDX, method='ffill'); E56[E56.index < e56.index[0]] = np.nan
ME = [g.iloc[-1] for _, g in _s.groupby([IDX.year, IDX.month])]
ME = pd.DatetimeIndex([d for d in ME if d >= SIG[0]])
WIN = {'select_2005-01_2019-04': ('2005-01-01', '2019-04-30'), 'valid_2019-05_2026-10': ('2019-05-01', '2026-10-31'),
       'select_2021_2023': ('2021-01-01', '2023-12-31'), 'valid_2024_2026-10': ('2024-01-01', '2026-10-31'),
       'y2024_2025': ('2024-01-01', '2025-12-31'), 'full_2005_2026-10': ('2005-01-01', '2026-10-31')}

def yearly_from_ye(nav):
    """以年底收盤（=訊號日）切年：Y 年報酬 = nav(YE_Y)/nav(YE_{Y-1}) − 1；2026 到最後一日。"""
    pts = list(SIG) + [IDX[-1]]
    nv = nav.reindex(pd.DatetimeIndex(pts))
    return {int(b.year) if b.year <= 2025 else 2026: float(nv[b] / nv[a] - 1) for a, b in zip(pts[:-1], pts[1:])}

def annual_t(strat, bench, years):
    ex = pd.Series({y: strat[y] - bench[y] for y in years if y in strat and y in bench})
    return {'n_years': int(len(ex)), 'mean_excess_pct': round(float(ex.mean() * 100), 2), 't': L.tstat(ex), 'win': f'{int((ex > 0).sum())}/{len(ex)}'}

# ================= 執行 =================
EWN = {}
for k in (50, 30):
    _, pools, _, _ = build('E1', k)
    pe = {ex: v for ex, v in pools.items()}
    EWN[(k, 'gross')], _ = run(pe, 'gross'); EWN[(k, 0.001)], _ = run(pe, 0.001)

RUNS = [('main', 'E1', 50, 10, None), ('a_small_dogs', 'E1', 50, 10, 5), ('b_pool30', 'E1', 30, 10, None),
        ('sens_E3', 'E3', 50, 10, None), ('sens_cap15', 'cap15', 50, 10, None), ('sens_mixed0', 'mixed0', 50, 10, None)]
navs = {'EW50_gross': EWN[(50, 'gross')], 'EW50_net': EWN[(50, 0.001)], 'EW30_gross': EWN[(30, 'gross')], 'EW30_net': EWN[(30, 0.001)]}
y50, y56 = yearly_from_ye(E50), yearly_from_ye(E56)
SEL_Y, VAL_Y = list(range(2005, 2019)), list(range(2019, 2027))
for name, col, k, n, small in RUNS:
    events, pools, picks, nblk = build(col, k, n, small)
    ew, ewn = EWN[(k, 'gross')], EWN[(k, 0.001)]
    R = {'gap_blocked': nblk}
    yE = yearly_from_ye(ew)
    for cost in ('gross', 0.001, 'stress'):
        nav, tov = run(events, cost); navs[f'{name}|{cost}'] = nav
        W = {w: M.stats(nav, a, b, ME, bench=E50, ew=ew) for w, (a, b) in WIN.items()}
        Wn = {w: M.stats(nav, a, b, ME, ew=ewn) for w, (a, b) in WIN.items()}
        W56 = {w: M.stats(nav, max(pd.Timestamp(a), pd.Timestamp('2008-01-01')), b, ME, bench=E56) for w, (a, b) in WIN.items()}
        for w in W:
            if W[w] is None: continue
            W[w]['t_vs_EWnet_m'] = Wn[w]['t_vs_EW_m']; W[w]['EWnet_CAGR'] = Wn[w]['EW_CAGR']
            if W56[w]: W[w]['vs0056_from2008'] = {'CAGR': W56[w]['CAGR'], '0056_CAGR': W56[w]['0050_CAGR'], 't_vs_0056_m': W56[w]['t_vs_0050_m']}
        ys = yearly_from_ye(nav)
        R[f'cost_{cost}'] = {'turnover_per_rebalance': round(tov, 3), 'windows': W,
                             'yearly_pct': {y: round(v * 100, 1) for y, v in ys.items()},
                             'annual_t': {'select_2005_2018': {'vs_EW': annual_t(ys, yE, SEL_Y), 'vs_0050': annual_t(ys, y50, SEL_Y)},
                                          'valid_2019_2026ytd': {'vs_EW': annual_t(ys, yE, VAL_Y), 'vs_0050': annual_t(ys, y50, VAL_Y)}}}
    if name in ('main', 'a_small_dogs', 'b_pool30'):
        R['picks_by_year'] = picks
        rows = []; exs = sorted(events)
        for a_, b_ in zip(exs, exs[1:] + [T_END]):
            for j in events[a_][1]:
                p0 = L.AO[a_, j]; p1 = L.AO[b_, j] if b_ != T_END else L.AC[b_, j]
                if not np.isfinite(p1): p1 = L.AC[b_, j]
                rows.append({'exec': str(IDX[a_].date()), 'code': COLS[j], 'name': sl.get(COLS[j], ''), 'ind': IND[COLS[j]],
                             'y': yields_at(IDX[a_ - 1], col)[j], 'gross': p1 / p0 - 1, 'flag345': bool(np.isin(FL[a_:b_ + 1, j], [3, 4, 5]).any())})
        hd = pd.DataFrame(rows)
        if name == 'main': hd.to_csv(HERE / 'holdings_main.csv', index=False)
        R['holding_detail'] = {'n_stock_years': int(len(hd)), 'n_unique': int(hd['code'].nunique()), 'n_flag345': int(hd['flag345'].sum()),
                               'flag345_list': hd[hd['flag345']][['exec', 'code', 'name', 'gross']].round(3).to_dict('records'),
                               'n_gross_gt100pct': int((hd['gross'] > 1).sum()), 'n_gross_lt_m60pct': int((hd['gross'] < -.6).sum()),
                               'extreme_list': hd[(hd['gross'] > 1) | (hd['gross'] < -.6)][['exec', 'code', 'name', 'y', 'gross']].round(3).to_dict('records'),
                               'n_y_gt15pct': int((hd['y'] > .15).sum()), 'y_gt15_list': hd[hd['y'] > .15][['exec', 'code', 'name', 'y']].round(3).to_dict('records'),
                               'median_y_select(<=2019)': round(float(hd[hd['exec'] < '2019-05']['y'].median()), 4),
                               'median_y_valid': round(float(hd[hd['exec'] >= '2019-05']['y'].median()), 4),
                               'share_financial': round(float((hd['ind'] == '金融保險業').mean()), 3),
                               'industry_mix': hd['ind'].value_counts().head(10).to_dict(),
                               'most_held': {f"{c}{sl.get(c, '')}": int(v) for c, v in hd['code'].value_counts().head(12).items()}}
    res[name] = R
    w = R['cost_0.001']['windows']
    print(name, {k_: (v['CAGR'], v['EW_CAGR'], v['0050_CAGR'], v['t_vs_EW_m'], v['t_vs_0050_m']) for k_, v in w.items() if v}, flush=True)

res['bench_yearly_pct'] = {'0050': {y: round(v * 100, 1) for y, v in y50.items()}, '0056': {y: round(v * 100, 1) for y, v in y56.items() if np.isfinite(v)},
                           'EW50_gross': {y: round(v * 100, 1) for y, v in yearly_from_ye(EWN[(50, 'gross')]).items()},
                           'EW30_gross': {y: round(v * 100, 1) for y, v in yearly_from_ye(EWN[(30, 'gross')]).items()}}
res['bench_windows'] = {'EW50_gross': {w: M.stats(EWN[(50, 'gross')], a, b, ME, bench=E50) for w, (a, b) in WIN.items()},
                        'EW30_gross': {w: M.stats(EWN[(30, 'gross')], a, b, ME, bench=E50) for w, (a, b) in WIN.items()}}

# ================= 資料檢查 =================
# 股池穩定度
pl_prev = None; stab = {}
for d in SIG:
    pl = set(pool_at(d, 50))
    stab[d.year] = {'n': len(pl), 'overlap_prev': (len(pl & pl_prev) if pl_prev is not None else None)}
    pl_prev = pl
res['pool_stability'] = stab
# E1 推估 vs exdiv 真值：2021-12～2025-12 的殖利率排名
ii2, jj2 = np.where(isX & ~pre)
ev_e1post = pd.DataFrame({'date': IDX[ii2], 'j': jj2, 'raw': np.maximum(Cprev[ii2, jj2] - Cn[ii2, jj2], 0)})
cal = {}
for d in SIG[SIG >= '2021-12-01']:
    if d < pd.Timestamp('2022-10-07'): continue
    pl = pool_at(d, 50)
    yt = yields_at(d, 'E1')[pl]; ye = yields_at(d, 'raw', ev_e1post)[pl]
    t10 = set(pl[np.argsort(-yt)[:10]]); e10 = set(pl[np.argsort(-ye)[:10]])
    cal[d.year] = {'spearman_pool50': round(float(pd.Series(yt).rank().corr(pd.Series(ye).rank())), 3), 'top10_overlap': len(t10 & e10),
                   'median_abs_diff_pct': round(float(np.median(np.abs(yt - ye)) * 100), 2)}
res['calib_E1_raw_vs_exdiv'] = {'note': '2022-12～2025-12（完整 12 個月都在 exdiv_5y 期間）：用「前收−收盤」推估（2021 前的口徑）vs exdiv 真值，在 V250 前 50 股池內', **cal}
# 股池 vs 市值（shares_hist 2025-12）、vs 實際 0050（etf0050_w）
sh = pd.read_csv(XR / 'shares_hist.csv', dtype={'code': str})
d25 = SIG[-1]; s25 = sh[sh['ym'] == '2025-12'].set_index('code')['shares']
mc = pd.Series({c: s25[c] * Cn[pos(d25), L.CI[c]] for c in s25.index if c in L.CI and is_common[L.CI[c]] and np.isfinite(Cn[pos(d25), L.CI[c]])})
mc50 = set(mc.sort_values(ascending=False).index[:50]); v50 = set(COLS[pool_at(d25, 50)])
w0050 = pd.read_csv(XR / 'etf0050_w.csv', dtype={'code': str})
dw = pd.Timestamp('2026-09-25'); act = set(w0050[w0050['asof'] == '2026-09-25']['code'])
v50w = set(COLS[pool_at(IDX[IDX <= dw][-1], 50)])
main_p25 = [p['code'] for p in res['main']['picks_by_year'][-1]['picks']]
res['universe_check'] = {'2025-12_V250top50_vs_mcap_top50_overlap': len(v50 & mc50), 'n_mcap_ranked': int(len(mc)),
                         'in_V250_not_mcap': sorted(f"{c}{sl.get(c, '')}" for c in v50 - mc50), 'in_mcap_not_V250': sorted(f"{c}{sl.get(c, '')}" for c in mc50 - v50),
                         '2026-09-25_V250top50_vs_actual0050_overlap': len(v50w & act), 'actual0050_n': len(act),
                         'in_V250_not_0050': sorted(f"{c}{sl.get(c, '')}" for c in v50w - act), 'in_0050_not_V250': sorted(f"{c}{sl.get(c, '')}" for c in act - v50w),
                         '2026_dogs_in_mcap_top50': sum(c in mc50 for c in main_p25), '2026_dogs_in_actual0050': sum(c in act for c in main_p25)}

# ================= 事後診斷（看過主結果後加，只報告、不進判定） =================
# 成交值代理排除了大部分金融、電信、台塑化等低周轉高股息大型股（2025-12 只有 31/50 與市值前 50 重疊）。
# 用「2025-12 股數 × 當時原始收盤」當市值代理重跑：只含 2025 年還在的股票（存活偏誤），股數不隨配股／增資調整（早年越不準）。
ph = {}
for pm_col in ('E1', 'E3'):
    _, pools_m, _, _ = build(pm_col, 50, 10, None, 'mcap25')
    ewm, _ = run(pools_m, 'gross')
    ev_m, _, pk_m, _ = build(pm_col, 50, 10, None, 'mcap25')
    navm, tovm = run(ev_m, 0.001); navs[f'posthoc_mcap25_{pm_col}|0.001'] = navm; navs['posthoc_mcap25_EW_gross'] = ewm
    ys = yearly_from_ye(navm); ye_ = yearly_from_ye(ewm)
    ph[pm_col] = {'turnover': round(tovm, 3), 'windows': {w: M.stats(navm, a, b, ME, bench=E50, ew=ewm) for w, (a, b) in WIN.items()},
                  'yearly_pct': {y: round(v * 100, 1) for y, v in ys.items()}, 'EW_yearly_pct': {y: round(v * 100, 1) for y, v in ye_.items()},
                  'annual_t_vs_EW': {'2005_2018': annual_t(ys, ye_, SEL_Y), '2019_2026': annual_t(ys, ye_, VAL_Y)},
                  'picks_recent': {p['exec'][:4]: ' '.join(f"{q['name']}{q['y']*100:.1f}" for q in p['picks']) for p in pk_m},
                  'share_financial': round(float(np.mean([q['ind'] == '金融保險業' for p in pk_m for q in p['picks']])), 3)}
    w = ph[pm_col]['windows']
    print('posthoc', pm_col, {k_: (v['CAGR'], v['EW_CAGR'], v['0050_CAGR'], v['t_vs_EW_m'], v['t_vs_0050_m']) for k_, v in w.items() if v})
res['posthoc_mcap2025_pool'] = ph

pd.DataFrame(navs).to_parquet(HERE / 'nav.parquet')
def ok(w): return bool(w is not None and w['t_vs_EW_m'] >= 2 and w['CAGR'] >= w['0050_CAGR'])
verd = {}
for name, *_ in RUNS:
    W = res[name]['cost_0.001']['windows']
    p1, p2 = ok(W['select_2005-01_2019-04']), ok(W['valid_2019-05_2026-10'])
    verd[name] = {'select_pass': p1, 'valid_pass': p2, 'split2021_select_pass': ok(W['select_2021_2023']), 'split2021_valid_pass': ok(W['valid_2024_2026-10']),
                  'verdict': '可用' if p1 and p2 else ('待觀察' if p1 or p2 else '不可用')}
res['verdicts_all'] = verd
res['verdict_main'] = verd['main']
res['runtime_s'] = round(time.time() - T0, 1)

def conv(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.bool_): return bool(o)
    return str(o)
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=conv), encoding='utf-8')
print('done', res['runtime_s'], verd)
