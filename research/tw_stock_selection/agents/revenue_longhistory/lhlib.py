#!/usr/bin/env python3
"""revenue_longhistory 共用：長期月營收（NiklavsD/bottleneck-signals 的 MOPS 彙總頁解析檔）＋主策略 M2 可成交版引擎。

選股與引擎邏輯複製自 agents/exec_rebuild/exlib.py（不 import 它，因為它在 import 時就把期間鎖在 2021 起）。
參數完全不改：60 日均成交值前 300 名（≥ NT$1,000 萬、股價 ≥ 10 元、非處置）、0.5×52 週高點百分位 + 0.5×近 3 月營收年增率平均百分位、
前 n 檔等權；訊號＝每月第一個 ≥ 11 日交易日收盤，隔日開盤成交，開盤漲停且未持有者跳過遞補。成本同 exlib。

唯一的「執行面」調整（事前定好）：漲跌幅上限 2015-06-01 前為 7%、之後 10%（exlib 只寫 10%，因為它只跑 2021 起）。
"""
import math, sys
from pathlib import Path
import numpy as np, pandas as pd

P = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
X = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras')
LONG = Path('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from btlib import tstat  # noqa: E402

FEE = 0.001425 * 0.6
TAX = 0.003
def cost_rates(slip):
    if slip == 'stress':
        return 0.005, 0.005
    return FEE + slip, FEE + TAX + slip

# ---------------- 資料 ----------------
adj = pd.read_parquet(P / 'adj.parquet'); C = pd.read_parquet(P / 'close_raw.parquet'); V = pd.read_parquet(P / 'value.parquet')
O = pd.read_parquet(P / 'open.parquet'); H = pd.read_parquet(P / 'high.parquet'); RET = pd.read_parquet(P / 'ret.parquet')
IDX = adj.index; COLS = adj.columns; N = len(COLS); CI = {c: j for j, c in enumerate(COLS)}
e50 = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()
E50 = e50.reindex(IDX, method='ffill')

def yoy_long():
    """長資料：營收水準 → 自算年增率（%）。sii 優先去重。"""
    d = pd.read_csv(LONG, dtype={'code': str, 'market': str, 'name': str})
    d['p'] = pd.PeriodIndex(pd.to_datetime(d['month']), freq='M')
    d = d.sort_values(['p', 'market']).drop_duplicates(['p', 'code'], keep='first')
    d = d[d['code'].isin(COLS)]
    L = d.pivot_table(index='p', columns='code', values='revenue_k_twd', aggfunc='last').sort_index()
    L = L.reindex(pd.period_range(L.index.min(), L.index.max(), freq='M'))
    prev = L.shift(12)
    return ((L / prev - 1) * 100).where(prev > 0)

def yoy_rev5y():
    rev = pd.read_csv(X / 'rev_5y.csv.gz', dtype={'code': str}); rev = rev[rev['code'].isin(COLS)]; rev['p'] = pd.PeriodIndex(rev['ym'], freq='M')
    Y = rev.pivot_table(index='p', columns='code', values='yoy', aggfunc='last').sort_index()
    return Y.reindex(pd.period_range(Y.index.min(), Y.index.max(), freq='M'))

def pos(d): return IDX.get_loc(d)

# 價格陣列
AC = adj.ffill().to_numpy()
_fac = (adj / C)
AO_raw = (O * _fac).to_numpy()
AO = np.where(np.isfinite(AO_raw), AO_raw, np.vstack([np.full((1, N), np.nan), AC[:-1]]))
ADJ_RAW = adj.to_numpy(); OPEN_RAW = O.to_numpy()
LTOL = 0.006
AC_prev = np.vstack([np.full((1, N), np.nan), AC[:-1]])
GAPRET = AO_raw / AC_prev - 1
LIM_ERA = np.where(IDX < pd.Timestamp('2015-06-01'), 0.07, 0.10)[:, None]
GAP_UP_OPEN_ERA = GAPRET >= LIM_ERA - LTOL          # 主設定：依當時漲跌幅
GAP_UP_OPEN_10 = GAPRET >= 0.10 - LTOL              # 診斷：exlib 原樣（全期 10%）
v60_raw = V.rolling(60).mean(); s52_raw = adj / adj.rolling(252).max()     # exlib 原式
# 面板缺漏修補（事前定好、只影響訊號計算）：2018-01-13（週六補班日）只有 46 檔有值，其餘約 880 檔缺，
# 使 rolling(252)/rolling(60) 在之後 252／60 個交易日全部變 NaN（2018 全年無法換股）。把「有值檔數 < 300」的日子當作不存在再算滾動值。
GOOD = adj.notna().sum(axis=1) >= 300
BAD_DAYS = [str(d.date()) for d in IDX[~GOOD]]
_ag = adj.loc[GOOD]; _vg = V.loc[GOOD]
s52_fix = (_ag / _ag.rolling(252).max()).reindex(IDX)
v60_fix = _vg.rolling(60).mean().reindex(IDX)
dsp = pd.read_csv(X / 'disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['date', 'start', 'end'])


class Setup:
    """給定月營收年增率表 YOY（index=月 Period），建立換股日與各橫截面。"""
    def __init__(self, YOY, start='2005-01-01', end=None, raw_panel=False):
        v60_all, s52_all = (v60_raw, s52_raw) if raw_panel else (v60_fix, s52_fix)
        self.YOY = YOY; YOY3 = YOY.rolling(3, min_periods=3).mean(); self.YOY3 = YOY3
        cnt = YOY.notna().sum(axis=1); last_full = cnt[cnt >= 0.8 * cnt.median()].index.max()
        ii = IDX[IDX >= pd.Timestamp(start)]
        if end is not None: ii = ii[ii <= pd.Timestamp(end)]
        s = pd.Series(ii, index=ii)
        ds = [g[g.dt.day >= 11].iloc[0] for _, g in s.groupby([ii.year, ii.month]) if (g.dt.day >= 11).any()]
        keep = [d for d in ds if YOY3.index.min() <= (pd.Period(d, 'M') - 1) <= last_full and YOY3.reindex([pd.Period(d, 'M') - 1]).notna().sum(axis=1).iloc[0] > 100]
        allds = [d for d in IDX[IDX >= pd.Timestamp(start)]]
        s2 = pd.Series(allds, index=pd.DatetimeIndex(allds))
        ds2 = [g[g.dt.day >= 11].iloc[0] for _, g in s2.groupby([s2.index.year, s2.index.month]) if (g.dt.day >= 11).any()]
        self.dates = keep + [d for d in ds2 if d > keep[-1]][:1]
        self.DATES = pd.DatetimeIndex(self.dates)
        mm = [pd.Period(d, 'M') - 1 for d in self.dates]
        self.yoy3 = pd.DataFrame(YOY3.reindex(mm).to_numpy(), index=self.DATES, columns=YOY3.columns).reindex(columns=COLS)
        v60 = v60_all.reindex(self.DATES); px = C.reindex(self.DATES); base = (v60 >= 1e7) & (px >= 10)
        self.t300 = base & (v60.where(base).rank(axis=1, ascending=False) <= 300)
        self.s52 = s52_all.reindex(self.DATES)
        m = pd.DataFrame(False, index=self.DATES, columns=COLS)
        for _, r in dsp.iterrows():
            if r['code'] in m.columns and pd.notna(r['start']) and pd.notna(r['end']):
                m.loc[(m.index >= r['start']) & (m.index <= r['end']), r['code']] = True
        self.inD = m
        # 已排序後的 R 只用來判斷「有沒有下一期報酬」— 與 exlib 一樣用 period_returns 的 notna 不需要；M2 以隔日開盤可成交判定

    def selection(self, n, w52=0.5, gap='era', need_rev=True):
        """M2：訊號日 d 收盤算分數，d+1 開盤成交；開盤漲停且未持有者跳過遞補。w52 只用於診斷（拆腿）。"""
        el = self.t300 & self.s52.notna() & ~self.inD
        if need_rev: el = el & self.yoy3.notna()
        r52 = self.s52.where(el).rank(axis=1, pct=True); rrv = self.yoy3.where(el).rank(axis=1, pct=True)
        if w52 == 1.0: sc = r52
        elif w52 == 0.0: sc = rrv
        else: sc = w52 * r52 + (1 - w52) * rrv
        GU = GAP_UP_OPEN_ERA if gap == 'era' else GAP_UP_OPEN_10
        events, pools = {}, {}; held = set(); nblk = 0
        for d in self.dates[:-1]:
            i = pos(d); ex = i + 1
            okb = el.loc[d].to_numpy()
            trad = np.isfinite(OPEN_RAW[ex]) & np.isfinite(ADJ_RAW[ex])
            ok = okb & trad & np.isfinite(sc.loc[d].to_numpy())
            if ok.sum() < 50: continue
            order = sc.loc[d].to_numpy().copy(); order[~ok] = np.nan
            cand = np.argsort(-np.nan_to_num(order, nan=-9))[:int(ok.sum())]
            pick = []
            for j in cand:
                if GU[ex, j] and j not in held:
                    nblk += 1; continue
                pick.append(j)
                if len(pick) == n: break
            events[ex] = ('open', np.array(pick)); pools[ex] = ('open', np.where(ok)[0])
            held = set(pick)
        self.nblk = nblk
        return events, pools


def sim_portfolio(events, buy, sell, t0, t1):
    """日頻 NAV（與 exlib.sim_portfolio 相同）。"""
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


def daily_stats(nav, segs, DATES, bench_nav=None, ew_nav=None):
    """與 exlib.daily_stats 相同，DATES 改為參數。"""
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
        o['0050_maxDD'] = round(float((eb / eb.cummax() - 1).min()), 4)
        o['0050_yearly_pct'] = {int(y): round(float((1 + g.fillna(0)).prod() - 1) * 100, 1) for y, g in b.groupby(b.index.year)}
    if ew_nav is not None:
        ed = ew_nav.reindex(DATES); ep = (ed.shift(-1) / ed - 1).iloc[:-1].reindex(pr.index)
        ex = pr - ep
        o['excess_vs_EW_pct_m'] = round(float(ex.mean() * 100), 2); o['t_vs_EW_m'] = tstat(ex)
        e = ew_nav.pct_change().reindex(rr.index); o['t_vs_EW_daily'] = tstat(rr - e)
        ee = (1 + e.fillna(0)).cumprod(); o['EW_CAGR'] = round(float(ee.iloc[-1] ** (1 / span) - 1), 4)
        o['EW_yearly_pct'] = {int(y): round(float((1 + g.fillna(0)).prod() - 1) * 100, 1) for y, g in e.groupby(e.index.year)}
    return o
