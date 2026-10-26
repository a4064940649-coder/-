#!/usr/bin/env python3
"""exec_rebuild 共用：資料、主策略選股（複製自 bt_final_rule.py，不改原檔）、日頻模擬引擎、統計。

事前固定（見 FINDINGS）：
* 成本：買 = 手續費 0.1425%×0.6 + 滑價；賣 = 手續費 0.1425%×0.6 + 證交稅 0.3% + 滑價；滑價每邊 0.1%／0.3%；壓力情境 買 0.5%＋賣 0.5%（來回 1%）。
* 周轉用漂移後權重：換股時把現有持股（漂移後市值）調到新目標等權，買賣金額分別乘費率。
* 持有期間停止交易／下市：以最後成交價計（無限期 ffill），下次換股時以該價賣出。
"""
import math, sys
from pathlib import Path
import numpy as np, pandas as pd

P = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
X = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from btlib import period_returns, bench_returns, ann, yearly, tstat  # noqa: E402

FEE = 0.001425 * 0.6
TAX = 0.003
def cost_rates(slip):
    if slip == 'stress':
        return 0.005, 0.005
    return FEE + slip, FEE + TAX + slip

# ---------------- 資料 ----------------
adj = pd.read_parquet(P / 'adj.parquet'); C = pd.read_parquet(P / 'close_raw.parquet'); V = pd.read_parquet(P / 'value.parquet')
O = pd.read_parquet(P / 'open.parquet'); H = pd.read_parquet(P / 'high.parquet'); RET = pd.read_parquet(P / 'ret.parquet')
SRC = pd.read_parquet(P / 'src.parquet')
IDX = adj.index; COLS = adj.columns; N = len(COLS); CI = {c: j for j, c in enumerate(COLS)}
e50 = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()
E50 = e50.reindex(IDX, method='ffill')          # 面板日曆上的 0050 含息指數
r50 = E50.pct_change()

# ---------------- 主策略（複製 bt_final_rule.py 的邏輯） ----------------
rev = pd.read_csv(X / 'rev_5y.csv.gz', dtype={'code': str}); rev = rev[rev['code'].isin(COLS)]; rev['p'] = pd.PeriodIndex(rev['ym'], freq='M')
YOY = rev.pivot_table(index='p', columns='code', values='yoy', aggfunc='last').sort_index()
YOY = YOY.reindex(pd.period_range(YOY.index.min(), YOY.index.max(), freq='M')); YOY3 = YOY.rolling(3, min_periods=3).mean()
cnt = YOY.notna().sum(axis=1); last_full = cnt[cnt >= 0.8 * cnt.median()].index.max()
ii = IDX[IDX >= pd.Timestamp('2021-01-01')]; _s = pd.Series(ii, index=ii)
ds = [g[g.dt.day >= 11].iloc[0] for _, g in _s.groupby([ii.year, ii.month]) if (g.dt.day >= 11).any()]
keep = [d for d in ds if YOY3.index.min() <= (pd.Period(d, 'M') - 1) <= last_full and YOY3.reindex([pd.Period(d, 'M') - 1]).notna().sum(axis=1).iloc[0] > 100]
dates = keep + [d for d in ds if d > keep[-1]][:1]
DATES = pd.DatetimeIndex(dates)
R = period_returns(adj, dates)
B = bench_returns(e50, dates)
mm = [pd.Period(d, 'M') - 1 for d in dates]
def bym(M): return pd.DataFrame(M.reindex(mm).to_numpy(), index=DATES, columns=M.columns).reindex(columns=COLS)
yoy3 = bym(YOY3)
v60_all = V.rolling(60).mean(); s52_all = adj / adj.rolling(252).max()
dsp = pd.read_csv(X / 'disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['date', 'start', 'end'])
def in_disposal(days):
    m = pd.DataFrame(False, index=pd.DatetimeIndex(days), columns=COLS)
    for _, r in dsp.iterrows():
        if r['code'] in m.columns and pd.notna(r['start']) and pd.notna(r['end']):
            m.loc[(m.index >= r['start']) & (m.index <= r['end']), r['code']] = True
    return m

def pos(d): return IDX.get_loc(d)
prev_day = pd.DatetimeIndex([IDX[pos(d) - 1] for d in dates])
next_day = pd.DatetimeIndex([IDX[min(pos(d) + 1, len(IDX) - 1)] for d in dates])

def t300_at(days):
    v60 = v60_all.reindex(days); px = C.reindex(days); base = (v60 >= 1e7) & (px >= 10)
    out = base & (v60.where(base).rank(axis=1, ascending=False) <= 300)
    return pd.DataFrame(out.to_numpy(), index=DATES, columns=COLS)

def s52_at(days): return pd.DataFrame(s52_all.reindex(days).to_numpy(), index=DATES, columns=COLS)

t300_d = t300_at(DATES); t300_dm1 = t300_at(prev_day)
s52_d = s52_at(DATES); s52_dm1 = s52_at(prev_day)
inD_d = in_disposal(DATES)
inD_dp1 = pd.DataFrame(in_disposal(next_day).to_numpy(), index=DATES, columns=COLS)

# 價格陣列（numpy）
AC = adj.ffill().to_numpy()                     # 還原收盤（無限期 ffill）
_fac = (adj / C)
AO_raw = (O * _fac).to_numpy()                  # 還原開盤 = 開盤 × 當日還原因子
AO = np.where(np.isfinite(AO_raw), AO_raw, np.vstack([np.full((1, N), np.nan), AC[:-1]]))
ADJ_RAW = adj.to_numpy(); OPEN_RAW = O.to_numpy(); RETN = RET.to_numpy(); CN = C.to_numpy(); HN = H.to_numpy(); VN = V.to_numpy()
LIMIT = 0.10; LTOL = 0.006
LOCK_UP_CLOSE = (RETN >= LIMIT - LTOL) & (CN >= HN * 0.999)          # 收盤鎖漲停
AC_prev = np.vstack([np.full((1, N), np.nan), AC[:-1]])
GAP_UP_OPEN = (AO_raw / AC_prev - 1) >= LIMIT - LTOL                   # 開盤漲停（含息跳空 ≥ 9.4%）

def selection(variant, n):
    """回傳 events: {exec_day_index: (mode, [col idx])}、pools: {exec_day_index: [col idx]}、signal date of each event。"""
    events, pools = {}, {}
    held = set(); nblk = 0
    # 診斷用：M1_lagonly＝只用前一日訊號、不排除鎖漲停；M1_exclonly＝當日訊號、只排除鎖漲停
    if variant in ('M0', 'M2', 'M3', 'M1_exclonly'):
        el = t300_d & yoy3.notna() & s52_d.notna() & ~inD_d
        sc = 0.5 * s52_d.where(el).rank(axis=1, pct=True) + 0.5 * yoy3.where(el).rank(axis=1, pct=True)
    else:  # M1：s52、v60、股價用 d 前一交易日
        el = t300_dm1 & yoy3.notna() & s52_dm1.notna() & ~inD_d
        sc = 0.5 * s52_dm1.where(el).rank(axis=1, pct=True) + 0.5 * yoy3.where(el).rank(axis=1, pct=True)
    if variant == 'M3':
        el = el & ~inD_dp1
    for k, d in enumerate(dates[:-1]):
        i = pos(d)
        if variant == 'M0':
            ok = el.loc[d] & R.loc[d].notna()
            if ok.sum() < 50: continue
            pick = list(sc.loc[d][ok].sort_values(ascending=False).index[:n])
            ex, mode = i, 'close'; pool = list(ok[ok].index)
        else:
            ex, mode = (i, 'close') if variant.startswith('M1') else (i + 1, 'open')
            okb = el.loc[d].to_numpy()
            if variant.startswith('M1'):
                trad = np.isfinite(ADJ_RAW[ex])
                blocked = LOCK_UP_CLOSE[ex] if variant != 'M1_lagonly' else np.zeros(N, bool)
            else:
                trad = np.isfinite(OPEN_RAW[ex]) & np.isfinite(ADJ_RAW[ex])
                blocked = GAP_UP_OPEN[ex]
            ok = okb & trad
            if ok.sum() < 50: continue
            order = sc.loc[d].to_numpy().copy(); order[~ok] = np.nan
            cand = np.argsort(-np.nan_to_num(order, nan=-9))[:int(ok.sum())]
            pick = []
            for j in cand:
                if blocked[j] and j not in held:
                    nblk += 1; continue      # 未持有且買不到 → 下一名遞補
                pick.append(COLS[j])
                if len(pick) == n: break
            pool = list(COLS[ok])
        events[ex] = (mode, np.array([CI[c] for c in pick])); pools[ex] = (mode, np.array([CI[c] for c in pool]))
        held = set(CI[c] for c in pick)
    selection.nblk = nblk
    return events, pools

def sim_portfolio(events, buy, sell, t0, t1):
    """日頻 NAV（每日收盤）。events={day: (mode, idx array)}，換股日調回等權（漂移後權重計周轉）。"""
    u = np.zeros(N); cash = 1.0; nav = np.empty(t1 - t0 + 1); tov = []
    for t in range(t0, t1 + 1):
        if t in events:
            mode, pk = events[t]
            Pz = AO[t] if mode == 'open' else AC[t]
            h = np.where(u > 0, u * np.nan_to_num(Pz), 0.0)
            Vt = h.sum() + cash
            tgt = np.zeros(N); w = 1.0 / len(pk)
            Vn = Vt
            for _ in range(6):
                tgt[:] = 0; tgt[pk] = w * Vn
                cost = buy * np.clip(tgt - h, 0, None).sum() + sell * np.clip(h - tgt, 0, None).sum()
                Vn = Vt - cost
            tov.append(np.clip(h - tgt, 0, None).sum() / Vt if Vt > 0 else 0)
            u = np.zeros(N); u[pk] = tgt[pk] / Pz[pk]; cash = 0.0
        nav[t - t0] = np.nansum(u * AC[t]) + cash
    return pd.Series(nav, index=IDX[t0:t1 + 1]), float(np.mean(tov[1:])) if len(tov) > 1 else float('nan')

# ---------------- 統計 ----------------
WINDOWS = {
    'full_2021-03_2026-10': [('2021-03-01', '2026-10-31')],
    'select_2021-03_2023-12': [('2021-03-01', '2023-12-31')],
    'valid_2024-01_2026-10': [('2024-01-01', '2026-10-31')],
    'valid_ex2026_2024-01_2025-12': [('2024-01-01', '2025-12-31')],
    'src3_2024-02-15_2025-09-23': [('2024-02-15', '2025-09-23')],
    'official_only': [('2021-03-01', '2024-02-14'), ('2025-09-24', '2026-10-31')],
}

def daily_stats(nav, segs, bench_nav=None, ew_nav=None):
    """nav: 收盤 NAV 序列；segs: [(a,b)]。日頻年化、波動、最大回撤、日超額 t、月窗（換股日收盤→下一換股日收盤）t。"""
    r = nav.pct_change()
    mask = np.zeros(len(r), bool); span = 0.0
    for a, b in segs:
        m = (r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b)) & r.notna().to_numpy()
        if m.sum() == 0: continue
        first = np.argmax(m); last = len(m) - 1 - np.argmax(m[::-1])
        span += (r.index[last] - r.index[first - 1]).days / 365.25 if first > 0 else (r.index[last] - r.index[first]).days / 365.25
        mask |= m
    rr = r[mask]
    if len(rr) < 20: return None
    eq = (1 + rr).cumprod()
    o = {'CAGR': round(float(eq.iloc[-1] ** (1 / span) - 1), 4), 'vol': round(float(rr.std() * math.sqrt(252)), 4),
         'maxDD': round(float((eq / eq.cummax() - 1).min()), 4), 'days': int(len(rr)), 'years': round(span, 2)}
    o['yearly_pct'] = {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in rr.groupby(rr.index.year)}
    # 月窗
    nd = nav.reindex(DATES)
    pr = (nd.shift(-1) / nd - 1).iloc[:-1].dropna()
    inw = np.zeros(len(pr), bool)
    for a, b in segs: inw |= (pr.index >= pd.Timestamp(a)) & (pr.index <= pd.Timestamp(b))
    pr = pr[inw]
    o['n_months'] = int(len(pr))
    if bench_nav is not None:
        bd = bench_nav.reindex(DATES); bp = (bd.shift(-1) / bd - 1).iloc[:-1].reindex(pr.index)
        ex = pr - bp
        o['excess_vs_0050_pct_m'] = round(float(ex.mean() * 100), 2); o['t_vs_0050_m'] = tstat(ex)
        o['hit_vs_0050_m'] = round(float((ex > 0).mean()), 3)
        b = bench_nav.pct_change().reindex(rr.index); o['t_vs_0050_daily'] = tstat(rr - b)
        eb = (1 + b.fillna(0)).cumprod(); o['0050_CAGR'] = round(float(eb.iloc[-1] ** (1 / span) - 1), 4)
    if ew_nav is not None:
        ed = ew_nav.reindex(DATES); ep = (ed.shift(-1) / ed - 1).iloc[:-1].reindex(pr.index)
        ex = pr - ep
        o['excess_vs_EW_pct_m'] = round(float(ex.mean() * 100), 2); o['t_vs_EW_m'] = tstat(ex)
        e = ew_nav.pct_change().reindex(rr.index); o['t_vs_EW_daily'] = tstat(rr - e)
        ee = (1 + e.fillna(0)).cumprod(); o['EW_CAGR'] = round(float(ee.iloc[-1] ** (1 / span) - 1), 4)
    return o

def all_windows(nav, bench_nav=None, ew_nav=None, windows=WINDOWS):
    return {k: daily_stats(nav, s, bench_nav, ew_nav) for k, s in windows.items()}

E50_NAV = E50.copy()
