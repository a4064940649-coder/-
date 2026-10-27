#!/usr/bin/env python3
"""商周版「營收 13 個月新高＋連 2 月年增＋營收 > 1 億，停損停利各 7%」長歷史可成交回測。
輸出 results.json（PREREG 先落地、看結果前固定）、nav.parquet、trades.parquet。執行：python3 -I run_r13s7.py
成交、成本、漲停判定、60 日均值修補沿用 momentum_family/mlib.py → exec_rebuild/exlib.py（只讀取、不修改）。
新增：逐筆部位停損停利引擎（收盤判斷、隔日開盤成交）與 30 檔上限的資金曲線。"""
import json, math, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'momentum_family'))
import mlib as M  # noqa: E402
L = M.L
T0 = time.time()
LONG = Path('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv')

PREREG = {
    '寫於看結果前': True,
    '來源': ['https://smart.businessweekly.com.tw/Reading/WebArticle.aspx?id=7009905'],
    '來源規則（教學）': '營收創 13 個月新高、連 2 個月年增、月營收 > 1 億；停損停利各 7%；教學稱約 7 年 7,000 多次交易、勝率約 6 成。',
    '股池': '面板上市普通股（排除 91xx 存託憑證）；訊號日 d 收盤：60 日均成交值 ≥ NT$1,000 萬（mlib.V60，2018-01-13 等有值檔數 < 300 的日子剔除後滾動）、收盤 ≥ 10 元；d+1 有開盤價與收盤值。不設前 300 名、不設處置濾網（教學沒有）。',
    '營收資料': 'NiklavsD/bottleneck-signals tw_revenue.csv（2005-01～2026-09，千元；sii 優先去重；只用面板內代碼）。營收 ≤ 0 視為缺值。',
    '訊號日': '每月第一個日期 ≥ 11 日的交易日 d（= 營收 10 日截止後第一個交易日）收盤；用 t = d 所在月前一月營收。長期檔無公告日，一律假設 10 日前已公告。',
    '訊號條件（三者同時）': {
        '(1) 營收門檻': 't 月營收 > 100,000 千元（= 1 億元）',
        '(2) 13 個月新高': 't 月營收 > t-1～t-12 每一個月營收（12 個月須全部有值）',
        '(3) 連 2 月年增': 't 與 t-1 月年增率都 > 0%（自算：營收／12 個月前營收 − 1；需 t-12、t-13 有值且 > 0）',
    },
    'IFRS 處理': '營收月份 2013-01～2014-01 一律不產生訊號（2013 合併口徑對 2012 個體口徑；2014-01 的 t-1＝2013-12 年增率仍跨斷點）；其間不新進場、既有部位照規則出場。挑選期統計剔除訊號日 2013-02～2014-02 的區段。',
    '進場': 'd+1 開盤買進（還原開盤價）；d+1 開盤漲停（還原開盤／前收 −1 ≥ 漲停幅 −0.6%；漲停幅 2015-06-01 前 7%、後 10%）不買、不遞補。同一檔持有中（含已觸發、將於 d+1 開盤出場者）再出現訊號不重複買。',
    '出場（逐筆、先觸發者）': {
        '(a) 停利': '任一收盤日 t ≥ 進場日：還原收盤／還原進場開盤 − 1 ≥ +7% → t+1 開盤賣出',
        '(b) 停損': '還原收盤／還原進場開盤 − 1 ≤ −7% → t+1 開盤賣出',
        '(c) 時間': '都沒觸發：進場後第 60 個交易日開盤賣出（持有 60 個交易日；教學沒寫，研究中心假設）',
        '說明': '面板無最低價，停損停利只能用收盤判斷、隔日開盤成交，比教學的盤中觸發保守（跳空會使實際出場價偏離 ±7%）。用還原價判斷（除息不會誤觸停損）。停牌／下市：價格沿用最後收盤（ffill），到期以該價出場。面板最後日 2026-10-22 仍未出場者以當日收盤計價。跌停賣不掉不模擬（面板無最低價）。',
    },
    '成本': 'exlib.cost_rates：買 0.0855%+滑價、賣 0.0855%+稅 0.3%+滑價；主＝滑價 0.1%（來回約 0.67%），壓力＝買 0.5%+賣 0.5%（來回 1%）；另報未扣成本。',
    '事件報酬（全部訊號，不受 30 檔限制）': '每筆＝進場開盤→出場開盤，扣成本；異常報酬＝淨報酬 − 同期（同進場日、同出場日、開盤→開盤）同池上市等權報酬（不扣成本）；t＝按進場月份分群（每月平均異常報酬一個觀察值）。勝率＝扣成本淨報酬 > 0 的比例（另報未扣成本勝率）。',
    '資金曲線': '起始 1；同時最多 30 檔；每筆部位金額 = 進場日開盤時 NAV/30（現金不足則用剩餘現金，低於 NAV/120 則不買）；訊號多於空位時依 t 月年增率由高到低優先（教學沒寫，事前定）；閒置現金報酬 0。日頻 NAV（收盤）。',
    '對手': '0050 含息（日頻）；同池等權＝每月同一股池全部等權、同引擎、d+1 開盤換股、不扣成本。',
    '組合統計': '日頻 NAV 算年化、最大回撤、逐年；t＝訊號日收盤→下一訊號日收盤的月窗超額（每月一個觀察值）。',
    '期間': {'挑選期': '2005-01～2019-04（實際首訊號約 2006-03），剔除 IFRS 區段',
             '驗證期': '2019-05～2026-10', '另報': ['2024-01～2025-12', 'A 乾淨段（～2013-01 訊號）', 'B 存活者段 2014-03～2019-04', '2021-03～2023-12／2024-01～2026-10', '官方價格來源段（排除 src3 2024-02-15～2025-09-23）']},
    '判定（滑價 0.1%）': '某期間「通過」＝ (i) 事件異常報酬月份分群 t ≥ 2，且 (ii) 資金曲線對同池等權月窗 t ≥ 2，且 (iii) 資金曲線年化 ≥ 0050 年化。挑選期與驗證期都通過 → 可用；只一段通過 → 待觀察；否則不可用。',
    '診斷（只報告、不改判定）': ['盤中停利版：最高價（還原）≥ 進場 ×1.07 當日以 max(開盤, 進場×1.07) 賣出；停損仍收盤判斷（面板無最低價）', '資金曲線改用隨機優先順序（20 個種子）', '不設 1 億門檻／不設連 2 月年增的訊號數', '持有期 flag 3/4/5 與單筆 >100% 報酬', '出場原因分布、平均持有天數'],
}
res = {'PREREG': PREREG}
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')

IDX, COLS, N, pos = L.IDX, L.COLS, L.N, L.pos
T = len(IDX)
AO, AC = L.AO, L.AC
# 還原最高價
AH = (L.H * (L.adj / L.C)).to_numpy()

# ---------------- 營收 ----------------
d = pd.read_csv(LONG, dtype={'code': str, 'market': str, 'name': str})
d['p'] = pd.PeriodIndex(pd.to_datetime(d['month']), freq='M')
d = d.sort_values(['p', 'market']).drop_duplicates(['p', 'code'], keep='first')
d = d[d['code'].isin(COLS)]
REV = d.pivot_table(index='p', columns='code', values='revenue_k_twd', aggfunc='last').sort_index()
REV = REV.reindex(pd.period_range(REV.index.min(), REV.index.max(), freq='M')).reindex(columns=COLS)
REV = REV.where(REV > 0)
hm12 = REV.rolling(12, min_periods=12).max().shift(1)
YOY = REV / REV.shift(12) - 1
c_amt = REV > 100000
c_high = REV > hm12
c_yoy = (YOY > 0) & (YOY.shift(1) > 0)
SIG = c_amt & c_high & c_yoy
IFRS = (REV.index >= pd.Period('2013-01', 'M')) & (REV.index <= pd.Period('2014-01', 'M'))
SIG_raw = SIG.copy()
SIG.loc[IFRS] = False

# ---------------- 日曆、股池 ----------------
D = [x for x in M.LDATES if (pd.Period(x, 'M') - 1) in REV.index]
DATES = pd.DatetimeIndex(D)
mm = [pd.Period(x, 'M') - 1 for x in D]
common = np.array([not c.startswith('91') for c in COLS])
v60 = M.V60.reindex(DATES); px = L.C.reindex(DATES)
liq = ((v60 >= 1e7) & (px >= 10)).to_numpy() & common[None, :]
SIGm = SIG.reindex(mm).fillna(False).to_numpy().astype(bool)
SIGm_raw = SIG_raw.reindex(mm).fillna(False).to_numpy().astype(bool)
YOYm = YOY.reindex(mm).to_numpy()
diag_counts = {'amt_high_yoy': [], 'high_only': [], 'high_yoy_no_amt': [], 'high_amt_no_yoy': []}
for nm, X in (('high_only', c_high), ('high_yoy_no_amt', c_high & c_yoy), ('high_amt_no_yoy', c_high & c_amt)):
    Xm = X.reindex(mm).fillna(False).to_numpy().astype(bool)
    diag_counts[nm] = Xm
months = []   # per signal month info
for k, dd in enumerate(D):
    i = pos(dd); ex = i + 1
    if ex >= T: break
    trad = np.isfinite(L.OPEN_RAW[ex]) & np.isfinite(L.ADJ_RAW[ex])
    pool = liq[k] & trad
    cand = np.where(pool & SIGm[k])[0]
    blk = M.GAP_UP[ex, cand]
    months.append({'k': k, 'd': dd, 'ex': ex, 'pool': np.where(pool)[0], 'cand': cand[~blk], 'n_sig': int(len(cand)), 'n_blocked': int(blk.sum()),
                   'n_sig_raw': int((pool & SIGm_raw[k]).sum()),
                   'n_high_only': int((pool & diag_counts['high_only'][k]).sum()), 'n_high_yoy_no_amt': int((pool & diag_counts['high_yoy_no_amt'][k]).sum()),
                   'n_high_amt_no_yoy': int((pool & diag_counts['high_amt_no_yoy'][k]).sum())})
EXDAYS = {m['ex']: m for m in months}

LIMIT_TP, LIMIT_SL, HOLD = 0.07, -0.07, 60

def exit_of(j, e, intraday_tp=False):
    """回傳 (出場日 x, 出場價, 原因)。進場 e 開盤，價 AO[e, j]。"""
    p0 = AO[e, j]
    last = min(e + HOLD, T - 1)
    for t in range(e, last):
        if intraday_tp:
            hi = AH[t, j]
            tgt = p0 * (1 + LIMIT_TP)
            if t > e:
                o = AO[t, j]
                if np.isfinite(o) and o >= tgt and np.isfinite(L.OPEN_RAW[t, j]):
                    return t, o, 'tp_intraday_gap'
            if np.isfinite(hi) and hi >= tgt:
                return t, tgt, 'tp_intraday'
        r = AC[t, j] / p0 - 1
        if r >= LIMIT_TP and not intraday_tp:
            return t + 1, AO[t + 1, j], 'tp'
        if r <= LIMIT_SL:
            return t + 1, AO[t + 1, j], 'sl'
    if e + HOLD <= T - 1:
        return e + HOLD, AO[e + HOLD, j], 'time'
    return T - 1, AC[T - 1, j], 'open_at_end'

# ---------------- 事件（全部訊號）----------------
def build_trades(intraday_tp=False):
    rows = []; open_until = np.full(N, -1)
    for m in months:
        e = m['ex']
        for j in m['cand']:
            if open_until[j] >= e:     # 持有中（含 e 開盤才出場者）→ 不重複買
                continue
            x, p1, why = exit_of(j, e, intraday_tp)
            open_until[j] = x
            rows.append({'k': m['k'], 'd': m['d'], 'e': e, 'x': x, 'j': j, 'code': COLS[j], 'p0': AO[e, j], 'p1': p1, 'why': why,
                         'yoy': YOYm[m['k'], j]})
    tr = pd.DataFrame(rows)
    tr['ret'] = tr.p1 / tr.p0 - 1
    tr['hold'] = tr.x - tr.e
    # 同池等權：同進場日 e、同出場日 x、開盤→開盤（最後一日用收盤）
    ewr = np.empty(len(tr))
    for i_, r in enumerate(tr.itertuples()):
        pl = months_by_k[r.k]['pool']
        px1 = AO[r.x, pl] if r.why != 'open_at_end' else AC[r.x, pl]
        if intraday_tp and r.why == 'tp_intraday':   # 盤中出場：對手用當日收盤
            px1 = AC[r.x, pl]
        ewr[i_] = np.nanmean(px1 / AO[r.e, pl] - 1)
    tr['ew'] = ewr
    return tr

months_by_k = {m['k']: m for m in months}
trades = build_trades(False)
print('trades', len(trades), round(time.time() - T0), flush=True)
trades_id = build_trades(True)
print('trades intraday', len(trades_id), round(time.time() - T0), flush=True)

d13 = DATES[DATES >= pd.Timestamp('2013-02-01')][0]; d14 = DATES[DATES >= pd.Timestamp('2014-03-01')][0]
first_sig = [m for m in months if len(m['cand'])][0]['d']
a0 = str(first_sig.date())
W = {
    'select_official': [(a0, str(d13.date())), (str(d14.date()), '2019-04-30')],
    'valid_2019-05_2026-10': [('2019-05-01', '2026-10-31')],
    'y2024-01_2025-12': [('2024-01-01', '2025-12-31')],
    'A_clean_to_2013-01sig': [(a0, str(d13.date()))],
    'B_survivor_2014-03_2019-04': [(str(d14.date()), '2019-04-30')],
    'valid_survivorPanel_2019-05_2023-12': [('2019-05-01', '2023-12-31')],
    'r21_select_2021-03_2023-12': [('2021-03-01', '2023-12-31')],
    'r21_valid_2024-01_2026-10': [('2024-01-01', '2026-10-31')],
    'valid_officialPx_only': [('2019-05-01', '2024-02-14'), ('2025-09-24', '2026-10-31')],
    'tutorial_like_2015-01_2021-12': [('2015-01-01', '2021-12-31')],
    'full_ex_IFRS': [(a0, str(d13.date())), (str(d14.date()), '2026-10-31')],
}

def event_stats(tr, cost):
    b, s = (0.0, 0.0) if cost == 'gross' else L.cost_rates(cost)
    net = (1 + tr.ret) * (1 - s) / (1 + b) - 1
    abn = net - tr.ew
    out = {}
    for kname, segs in W.items():
        m = np.zeros(len(tr), bool)
        for a, b_ in segs: m |= (tr.d >= pd.Timestamp(a)) & (tr.d < pd.Timestamp(b_))
        if m.sum() < 10: continue
        x = tr[m]; g = abn[m].groupby(x.d).mean()
        yrs = (x.d.max() - x.d.min()).days / 365.25
        out[kname] = {'n_trades': int(m.sum()), 'trades_per_year': round(float(m.sum() / max(yrs, 0.5)), 0), 'n_months': int(len(g)),
                      'win_rate_net': round(float((net[m] > 0).mean()), 3), 'win_rate_gross': round(float((x.ret > 0).mean()), 3),
                      'mean_ret_gross_pct': round(float(x.ret.mean() * 100), 2), 'mean_net_ret_pct': round(float(net[m].mean() * 100), 2),
                      'median_net_ret_pct': round(float(net[m].median() * 100), 2),
                      'mean_poolEW_same_window_pct': round(float(x.ew.mean() * 100), 2),
                      'mean_abn_pct': round(float(abn[m].mean() * 100), 2), 'month_cluster_mean_abn_pct': round(float(g.mean() * 100), 2),
                      't_month_cluster': L.tstat(g), 'hit_abn': round(float((abn[m] > 0).mean()), 3),
                      'mean_hold_days': round(float(x.hold.mean()), 1),
                      'exit_mix': {k_: round(float(v), 3) for k_, v in x.why.value_counts(normalize=True).items()}}
    return out

res['events'] = {str(c): event_stats(trades, c) for c in ('gross', 0.001, 'stress')}
res['events_intraday_tp_diag'] = {str(c): event_stats(trades_id, c) for c in ('gross', 0.001)}
res['events_by_year_main'] = {}
b1, s1 = L.cost_rates(0.001)
_net = (1 + trades.ret) * (1 - s1) / (1 + b1) - 1
for y, g in trades.groupby(trades.d.dt.year):
    nn = _net[g.index]
    res['events_by_year_main'][int(y)] = {'n': int(len(g)), 'win_net': round(float((nn > 0).mean()), 3), 'mean_net_pct': round(float(nn.mean() * 100), 2),
                                          'mean_abn_pct': round(float((nn - g.ew).mean() * 100), 2)}
print(json.dumps({k: res['events']['0.001'][k] for k in ('select_official', 'valid_2019-05_2026-10', 'y2024-01_2025-12') if k in res['events']['0.001']}, ensure_ascii=False), flush=True)

# ---------------- 資金曲線 ----------------
def sim_slots(buy, sell, order='yoy', seed=None, max_slots=30, t_start=None):
    rng = np.random.default_rng(seed) if seed is not None else None
    t0 = pos(first_sig) if t_start is None else t_start
    cash = 1.0; posl = []   # dict j,u,e,p0,x(None)
    nav = np.empty(T - t0); n_open = np.empty(T - t0); invested = np.empty(T - t0); nbuy = 0; nskip_slot = 0
    for t in range(t0, T):
        # 開盤：出場
        keep = []
        for p in posl:
            if p['x'] == t:
                cash += p['u'] * AO[t, p['j']] * (1 - sell)
            else:
                keep.append(p)
        posl = keep
        # 開盤：進場
        if t in EXDAYS:
            m = EXDAYS[t]
            held = {p['j'] for p in posl}
            cand = [j for j in m['cand'] if j not in held]
            if order == 'yoy':
                cand = sorted(cand, key=lambda j: -np.nan_to_num(YOYm[m['k'], j], nan=-9))
            else:
                cand = list(rng.permutation(cand))
            navo = cash + sum(p['u'] * AO[t, p['j']] for p in posl)
            slot = navo / max_slots
            for j in cand:
                if len(posl) >= max_slots:
                    nskip_slot += 1; continue
                amt = min(slot, cash)
                if amt < navo / 120:
                    nskip_slot += 1; continue
                u = amt / (1 + buy) / AO[t, j]
                cash -= amt; nbuy += 1
                posl.append({'j': j, 'u': u, 'e': t, 'p0': AO[t, j], 'x': None})
        # 收盤：觸發判斷
        val = 0.0
        for p in posl:
            c = AC[t, p['j']]; val += p['u'] * c
            if p['x'] is None and t + 1 < T:
                r = c / p['p0'] - 1
                if r >= LIMIT_TP or r <= LIMIT_SL or (t + 1 - p['e']) >= HOLD:
                    p['x'] = t + 1
        nav[t - t0] = cash + val; n_open[t - t0] = len(posl); invested[t - t0] = val / (cash + val)
    idx = IDX[t0:]
    return pd.Series(nav, index=idx), pd.Series(n_open, index=idx), pd.Series(invested, index=idx), nbuy, nskip_slot

def daily_stats(nav, segs, bench_nav=None, ew_nav=None):
    """同 w1_rev_record_high.daily_stats（月窗以本檔訊號日曆）。"""
    r = nav.pct_change(); mask = np.zeros(len(r), bool); span = 0.0
    for a, b in segs:
        m = (r.index > pd.Timestamp(a)) & (r.index <= pd.Timestamp(b)) & r.notna().to_numpy()
        if m.sum() == 0: continue
        first = np.argmax(m); last = len(m) - 1 - np.argmax(m[::-1])
        span += (r.index[last] - r.index[first - 1]).days / 365.25
        mask |= m
    rr = r[mask]
    if len(rr) < 20: return None
    eq = (1 + rr).cumprod()
    o = {'CAGR': round(float(eq.iloc[-1] ** (1 / span) - 1), 4), 'vol': round(float(rr.std() * math.sqrt(252)), 4),
         'maxDD': round(float((eq / eq.cummax() - 1).min()), 4), 'years': round(span, 2)}
    DD = DATES[(DATES >= nav.index[0])].append(pd.DatetimeIndex([IDX[-1]]))
    nd = nav.reindex(DD); pr = (nd.shift(-1) / nd - 1).iloc[:-1].dropna()
    inw = np.zeros(len(pr), bool)
    for a, b in segs: inw |= (pr.index >= pd.Timestamp(a)) & (pr.index < pd.Timestamp(b))
    pr = pr[inw]; o['n_months'] = int(len(pr))
    for nm, bn in (('0050', bench_nav), ('EW', ew_nav)):
        if bn is None: continue
        bd = bn.reindex(DD); bp = (bd.shift(-1) / bd - 1).iloc[:-1].reindex(pr.index); ex = pr - bp
        o[f'excess_vs_{nm}_pct_m'] = round(float(ex.mean() * 100), 2); o[f't_vs_{nm}_m'] = L.tstat(ex)
        o[f'hit_vs_{nm}_m'] = round(float((ex > 0).mean()), 3)
        b = bn.pct_change().reindex(rr.index); eb = (1 + b.fillna(0)).cumprod()
        o[f'{nm}_CAGR'] = round(float(eb.iloc[-1] ** (1 / span) - 1), 4); o[f'{nm}_maxDD'] = round(float((eb / eb.cummax() - 1).min()), 4)
    return o

def yearly(nav):
    r = nav.pct_change().dropna(); return {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in r.groupby(r.index.year)}

t0 = pos(first_sig) + 1
pools_ev = {m['ex']: ('open', m['pool']) for m in months if m['ex'] >= t0 and len(m['pool'])}
ew, _ = L.sim_portfolio(pools_ev, 0.0, 0.0, t0, T - 1)
ew = pd.concat([pd.Series([1.0], index=[IDX[t0 - 1]]), ew])
E50 = L.E50.ffill()
navs = {'EW_pool': ew}
port = {}
for cost in ('gross', 0.001, 'stress'):
    b, s = (0.0, 0.0) if cost == 'gross' else L.cost_rates(cost)
    nav, nop, inv, nbuy, nskip = sim_slots(b, s)
    navs[f'slots30_{cost}'] = nav
    port[str(cost)] = {'n_buys': nbuy, 'n_skipped_no_slot_or_cash': nskip,
                       'mean_open_positions': round(float(nop.mean()), 1), 'mean_invested_share': round(float(inv.mean()), 3),
                       'windows': {k: daily_stats(nav, sg, E50, ew) for k, sg in W.items()}, 'yearly_pct': yearly(nav)}
    if cost == 0.001:
        port[str(cost)]['open_positions_by_year'] = {int(y): round(float(g.mean()), 1) for y, g in nop.groupby(nop.index.year)}
        port[str(cost)]['invested_by_year'] = {int(y): round(float(g.mean()), 3) for y, g in inv.groupby(inv.index.year)}
    print('port', cost, {k: port[str(cost)]['windows'][k] for k in ('select_official', 'valid_2019-05_2026-10')}, round(time.time() - T0), flush=True)
res['portfolio_slots30'] = port
res['EW_pool_yearly'] = yearly(ew)
res['0050_yearly'] = yearly(E50[E50.index >= first_sig])
res['EW_pool_windows'] = {k: daily_stats(ew, sg, E50) for k, sg in W.items()}

# 隨機優先順序（診斷）
rnd = {}
for seed in range(20):
    nav, *_ = sim_slots(b1, s1, order='random', seed=seed)
    for k in ('select_official', 'valid_2019-05_2026-10', 'y2024-01_2025-12'):
        st = daily_stats(nav, W[k], E50, ew)
        rnd.setdefault(k, []).append((st['CAGR'], st['t_vs_EW_m'], st['t_vs_0050_m']))
res['portfolio_random_order_diag'] = {k: {'CAGR_mean': round(float(np.mean([x[0] for x in v])), 4), 'CAGR_min': round(float(np.min([x[0] for x in v])), 4),
                                          'CAGR_max': round(float(np.max([x[0] for x in v])), 4), 't_vs_EW_mean': round(float(np.mean([x[1] for x in v])), 2),
                                          't_vs_0050_mean': round(float(np.mean([x[2] for x in v])), 2)} for k, v in rnd.items()}
print('random', res['portfolio_random_order_diag'], round(time.time() - T0), flush=True)

# ---------------- 資料檢查 ----------------
mi = pd.DataFrame([{k: v for k, v in m.items() if k not in ('pool', 'cand')} | {'pool_n': len(m['pool'])} for m in months])
res['signals_per_month_by_year'] = {int(y): {'pool': round(float(g.pool_n.mean()), 0), 'n_sig': round(float(g.n_sig.mean()), 1), 'n_blocked': round(float(g.n_blocked.mean()), 2),
                                             'n_high_only': round(float(g.n_high_only.mean()), 1), 'n_high_yoy_no_amt': round(float(g.n_high_yoy_no_amt.mean()), 1),
                                             'n_high_amt_no_yoy': round(float(g.n_high_amt_no_yoy.mean()), 1)}
                                    for y, g in mi.groupby(pd.DatetimeIndex(mi.d).year)}
res['ifrs_block_raw_signals'] = {str(mm[int(r.k)]): int(r.n_sig_raw) for r in mi.itertuples() if pd.Timestamp('2012-08-01') <= r.d <= pd.Timestamp('2014-06-30')}
res['blocked_limit_up_total'] = int(mi.n_blocked.sum())
FL = pd.read_parquet(L.P / 'flag.parquet').reindex(index=IDX, columns=COLS).to_numpy()
nfl = sum(bool(np.isin(FL[r.e:r.x + 1, r.j], [3, 4, 5]).any()) for r in trades.itertuples())
res['trades_with_flag345'] = {'n': int(nfl), 'of': int(len(trades))}
big = trades[(trades.ret > 1.0) | (trades.ret < -0.6)]
res['extreme_trades'] = big.assign(d=big.d.astype(str))[['d', 'code', 'ret', 'why', 'hold']].round(3).to_dict('records')[:30]
src = pd.read_parquet(L.P / 'src.parquet').reindex(index=IDX, columns=COLS).to_numpy()
res['trades_entry_src3_share'] = round(float(np.mean([src[r.e, r.j] == 3 for r in trades.itertuples()])), 3)
res['gap_exit_diag'] = {'tp_mean_exit_ret_pct': round(float(trades[trades.why == 'tp'].ret.mean() * 100), 2),
                        'sl_mean_exit_ret_pct': round(float(trades[trades.why == 'sl'].ret.mean() * 100), 2),
                        'sl_worse_than_-10pct_share': round(float((trades[trades.why == 'sl'].ret < -0.10).mean()), 3),
                        'time_mean_ret_pct': round(float(trades[trades.why == 'time'].ret.mean() * 100), 2)}
res['last_signal'] = {'d': str(months[-1]['d'].date()), 'codes': [COLS[j] for j in months[-1]['cand']]}

# 判定
def passes(k):
    ev = res['events']['0.001'].get(k); pw = port['0.001']['windows'].get(k)
    if not ev or not pw: return None
    return {'event_t': ev['t_month_cluster'], 'port_t_vs_EW': pw['t_vs_EW_m'], 'port_CAGR': pw['CAGR'], '0050_CAGR': pw['0050_CAGR'],
            'pass': bool(ev['t_month_cluster'] >= 2 and pw['t_vs_EW_m'] >= 2 and pw['CAGR'] >= pw['0050_CAGR'])}
J = {k: passes(k) for k in ('select_official', 'valid_2019-05_2026-10')}
npass = sum(v['pass'] for v in J.values() if v)
J['verdict'] = '可用' if npass == 2 else ('待觀察' if npass == 1 else '不可用')
res['judgement'] = J
print('judgement', J, flush=True)
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
pd.DataFrame(navs).to_parquet(HERE / 'nav.parquet')
trades.drop(columns=['j']).to_parquet(HERE / 'trades.parquet')
print('done', round(time.time() - T0))
