# -*- coding: utf-8 -*-
"""獨立複製員自寫：VCP 突破警示（XS 腳本）訊號、事件研究、組合引擎。
只依 FINDINGS.md 第 2 節規格與 raw/vcp_alert.xs 撰寫；沒有讀也沒有 import 研究員的程式。
一律 python3 -I -B。
"""
import numpy as np, pandas as pd

S = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/'
PANEL = S + 'panel/'
BASE = '/home/user/-/research/tw_stock_selection/agents/o_ext_vcp_xs/'
IDX = '/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv'
ETF = '/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv'
DISP = '/home/user/-/research/tw_stock_selection/agents/o_disposal_list_pit_audit_pkl18/data/universe_announce.csv'

DEFAULT = dict(SegLen=25, MaxLastDepth=10.0, MinFirstDepth=10.0, VolDryRatio=0.7, VolBreak=1.5,
               En_RS=1, RSMinDiff=0.0, MinVol=500.0, LowMult=1.3, HighPct=25.0, MA200UpBars=20)
START = pd.Timestamp('2005-01-31')


def tick_of(p):
    if p < 10: return 0.01
    if p < 50: return 0.05
    if p < 100: return 0.1
    if p < 500: return 0.5
    if p < 1000: return 1.0
    return 5.0


def floor_tick(p):
    t = tick_of(p)
    return np.floor(p / t + 1e-7) * t


def ceil_tick(p):
    t = tick_of(p)
    return np.ceil(p / t - 1e-7) * t


class Data:
    def __init__(self, low_override=None, vol_override=None):
        rd = lambda n: pd.read_parquet(PANEL + n + '.parquet')
        c = rd('close_raw')
        self.dates = c.index
        self.codes = np.array(c.columns)
        self.C = c.values
        self.O = rd('open').values
        self.H = rd('high').values
        self.A = rd('adj').values
        self.R = rd('ret').values
        self.FLAG = rd('flag').values
        self.SRC = rd('src').values
        self.VAL = rd('value').values
        self.L = pd.read_parquet(BASE + 'data/low.parquet').values.astype(float) if low_override is None else low_override
        self.V = pd.read_parquet(BASE + 'data/volume.parquet').values.astype(float) if vol_override is None else vol_override
        self.T, self.N = self.C.shape
        ix = pd.read_csv(IDX, parse_dates=['date']).set_index('date')['close']
        self.IDXC = ix.reindex(ix.index.union(self.dates)).ffill().reindex(self.dates).values
        ef = pd.read_csv(ETF, parse_dates=['date']).set_index('date')['adj']
        self.ETF = ef.reindex(ef.index.union(self.dates)).ffill().reindex(self.dates).values
        self.F = self.A / self.C                         # 還原因子
        self.AO = self.O * self.F
        self.AH = self.H * self.F
        self.AL = self.L * self.F
        self.lim = np.where(self.dates < pd.Timestamp('2015-06-01'), 0.07, 0.10)
        self._pack()
        self._aux()

    # ---- 把每檔自己的 K 棒往上壓實（bar 座標） ----
    def _pack(self):
        has = ~np.isnan(self.C)
        order = np.argsort(~has, axis=0, kind='stable')          # 有值的列在前，保持時間序
        self.nbar = has.sum(0)
        self.order = order
        self.barno = np.where(has, np.cumsum(has, 0), 0)          # 日曆格→第幾根（1 起）
        self.has = has
        rows = np.arange(self.T)[:, None]
        self.packmask = rows < self.nbar[None, :]

    def pack(self, X):
        P = np.take_along_axis(X, self.order, axis=0).astype(float)
        P[~self.packmask] = np.nan
        return P

    def unpack(self, P, fill=np.nan):
        X = np.full((self.T, self.N), fill, dtype=P.dtype if P.dtype != bool else bool)
        if P.dtype == bool: X[:] = False
        cols = np.broadcast_to(np.arange(self.N), (self.T, self.N))
        m = self.packmask
        X[self.order[m], cols[m]] = P[m]
        return X

    def _aux(self):
        # 上一根收盤（日曆座標，ffill 後 shift）
        Cf = pd.DataFrame(self.C).ffill().values
        prevC = np.vstack([np.full((1, self.N), np.nan), Cf[:-1]])
        ref = self.C / (1.0 + self.R)
        ref = np.where(self.FLAG == 1, prevC, ref)
        self.REF = ref
        up = ref * (1 + self.lim[:, None])
        fl = np.vectorize(floor_tick, otypes=[float])
        m = ~np.isnan(up)
        UP = np.full_like(up, np.nan)
        UP[m] = fl(up[m])
        self.UPLIM = UP
        self.open_limit_up = m & (self.O >= UP - 1e-9)
        # 全日鎖跌停：最高＝最低且日報酬 ≤ −(漲跌幅−0.6%)
        self.lock_dn = (self.H == self.L) & (self.R <= -(self.lim[:, None] - 0.006))
        self.Af = pd.DataFrame(self.A).ffill().values           # 計值用（停牌以最後收盤）
        # 還原收盤的 50 根均線（bar 座標）
        pa = pd.DataFrame(self.pack(self.A))
        ma50 = pa.rolling(50, min_periods=50).mean().values
        self.AMA50 = self.unpack(ma50)
        # 處置
        d = pd.read_csv(DISP, dtype={'code': str})
        pos = {c: i for i, c in enumerate(self.codes)}
        self.disp = {}
        for code, an, st, en in zip(d['code'], d['announce'], d['start'], d['end']):
            if code in pos:
                self.disp.setdefault(pos[code], []).append((pd.Timestamp(an), pd.Timestamp(st), pd.Timestamp(en)))

    def in_disposal(self, j, e_date):
        for an, st, en in self.disp.get(j, ()):
            if st <= e_date <= en and an < e_date:
                return True
        return False


def signals(D, P=None, mode='raw', ablate=(), min_bars=260, want_funnel=False):
    """回傳 DataFrame（每列一個訊號）與漏斗。ablate ⊂ {'trend','contract','voldry','breakvol','rs','minvol'}"""
    P = dict(DEFAULT, **(P or {}))
    Lg = int(P['SegLen'])
    if mode == 'raw':
        C, Hh, Ll = D.pack(D.C), D.pack(D.H), D.pack(D.L)
    else:
        C, Hh, Ll = D.pack(D.A), D.pack(D.AH), D.pack(D.AL)
    V = D.pack(D.V) / 1000.0
    cal = np.take_along_axis(np.broadcast_to(np.arange(D.T)[:, None], (D.T, D.N)), D.order, axis=0)
    IX = D.IDXC[cal]
    IX[~D.packmask] = np.nan
    dC, dH, dL, dV, dIX = map(pd.DataFrame, (C, Hh, Ll, V, IX))
    t = np.arange(1, D.T + 1)[:, None] * np.ones((1, D.N))
    ok = D.packmask & (t >= min_bars) & (t > 3 * Lg + 1)
    ma50 = dC.rolling(50, min_periods=50).mean()
    ma150 = dC.rolling(150, min_periods=150).mean()
    ma200 = dC.rolling(200, min_periods=200).mean()
    ma200p = ma200.shift(P['MA200UpBars']).values.copy()
    ma200p[(t - P['MA200UpBars']) < min_bars] = 0.0
    low52 = dL.rolling(250, min_periods=250).min().values
    high52 = dH.rolling(250, min_periods=250).max().values
    ma50, ma150, ma200 = ma50.values, ma150.values, ma200.values
    with np.errstate(invalid='ignore'):
        trend = ((C > ma150) & (C > ma200) & (ma150 > ma200) & (ma200 > ma200p) & (ma50 > ma150) & (ma50 > ma200)
                 & (C > ma50) & (low52 > 0) & (C >= low52 * P['LowMult']) & (high52 > 0)
                 & (C >= high52 * (1 - P['HighPct'] / 100.0)))
        h3 = dH.shift(1).rolling(Lg, min_periods=Lg).max().values
        l3 = dL.shift(1).rolling(Lg, min_periods=Lg).min().values
        h2 = dH.shift(Lg + 1).rolling(Lg, min_periods=Lg).max().values
        l2 = dL.shift(Lg + 1).rolling(Lg, min_periods=Lg).min().values
        h1 = dH.shift(2 * Lg + 1).rolling(Lg, min_periods=Lg).max().values
        l1 = dL.shift(2 * Lg + 1).rolling(Lg, min_periods=Lg).min().values
        d1 = (h1 - l1) / h1 * 100
        d2 = (h2 - l2) / h2 * 100
        d3 = (h3 - l3) / h3 * 100
        contract = (d1 >= P['MinFirstDepth']) & (d1 > d2) & (d2 > d3) & (d3 <= P['MaxLastDepth'])
        av10 = dV.shift(1).rolling(10, min_periods=10).mean().values
        av50 = dV.shift(1).rolling(50, min_periods=50).mean().values
        vratio = av10 / av50
        voldry = (av50 > 0) & (vratio < P['VolDryRatio'])
        pivot = h3
        pivp = pd.DataFrame(h3).shift(1).values.copy()
        pivp[(t - 1) < min_bars] = 0.0
        Cp = dC.shift(1).values
        cross = (pivot > 0) & (C > pivot) & (Cp <= pivp)
        stk = (C / dC.shift(250).values - 1) * 100
        ixr = (IX / dIX.shift(250).values - 1) * 100
        rs = ~(stk < ixr + P['RSMinDiff'])            # XS：stk<idx 才設 false；NaN 視為通過不了下面的 ok
        rs &= ~np.isnan(stk) & ~np.isnan(ixr)
        bvol = V > av50 * P['VolBreak']
        mvol = (V >= P['MinVol']) if P['MinVol'] > 0 else np.ones_like(ok)
    if not P['En_RS']: ablate = tuple(ablate) + ('rs',)
    conds = [('trend', trend), ('rs', rs), ('contract', contract), ('voldry', voldry), ('cross', cross),
             ('breakvol', bvol), ('minvol', mvol)]
    cur = ok.copy()
    funnel = {'computable': int(ok.sum())}
    single = {}
    for name, m in conds:
        single[name] = float((m & ok).sum() / ok.sum())
        if name not in ablate:
            cur = cur & m
        funnel[name] = int(cur.sum())
    ii, jj = np.nonzero(cur)
    sig = pd.DataFrame({
        'code': D.codes[jj], 'j': jj, 'di': cal[ii, jj], 'date': D.dates[cal[ii, jj]], 'bar': ii + 1,
        'close': C[ii, jj], 'prev_close': Cp[ii, jj], 'pivot': pivot[ii, jj], 'pivot_prev': pivp[ii, jj],
        'd1': d1[ii, jj], 'd2': d2[ii, jj], 'd3': d3[ii, jj], 'vdry': vratio[ii, jj],
        'vbreak': V[ii, jj] / av50[ii, jj], 'vol_lots': V[ii, jj],
        'ma50': ma50[ii, jj], 'ma150': ma150[ii, jj], 'ma200': ma200[ii, jj], 'ma200_20': ma200p[ii, jj],
        'low52': low52[ii, jj], 'high52': high52[ii, jj], 'stk_roc': stk[ii, jj], 'idx_roc': ixr[ii, jj]})
    # 樞紐換到還原價座標（E5 用）
    sig['pivot_adj'] = sig['pivot'].values * (D.F[sig['di'].values, sig['j'].values] if mode == 'raw' else 1.0)
    sig = sig.sort_values(['date', 'code']).reset_index(drop=True)
    if want_funnel:
        return sig, funnel, single
    return sig


def tradable(D, sig):
    """加上 e 日與可成交旗標。"""
    s = sig.copy()
    e = s['di'].values + 1
    okE = e < D.T
    ee = np.where(okE, e, D.T - 1)
    j = s['j'].values
    s['ei'] = ee
    s['no_open'] = ~okE | np.isnan(D.O[ee, j])
    s['lim_up'] = okE & D.open_limit_up[ee, j]
    s['lock_up'] = s['lim_up'] & (D.H[ee, j] == D.L[ee, j])
    s['disp'] = [okE[k] and D.in_disposal(j[k], D.dates[ee[k]]) for k in range(len(s))]
    s['ok'] = ~(s['no_open'] | s['lim_up'] | s['disp'])
    return s


def cluster_t(x, groups):
    x = np.asarray(x, float)
    n = len(x)
    if n < 3: return dict(n=n, mean=np.nan, t=np.nan, G=0, neff=np.nan)
    m = x.mean()
    e = x - m
    g = pd.Series(e).groupby(np.asarray(groups)).sum().values
    G = len(g)
    vc = (g ** 2).sum() / n ** 2 * G / (G - 1)
    vo = x.var(ddof=1) / n
    return dict(n=n, mean=float(m), t=float(m / np.sqrt(vc)), G=int(G), neff=float(n * vo / vc),
                t_naive=float(m / np.sqrt(vo)))


_POOL_CACHE = {}


def pool_ret(D, di, h):
    key = (di, h)
    if key in _POOL_CACHE: return _POOL_CACHE[key]
    e = di + 1
    x = e + h - 1
    if x >= D.T:
        _POOL_CACHE[key] = np.nan
        return np.nan
    m = (D.V[di] / 1000.0 >= 500) & (D.barno[di] >= 260) & D.has[di] & ~np.isnan(D.O[e])
    r = D.Af[x, m] / D.AO[e, m] - 1
    v = float(np.nanmean(r))
    _POOL_CACHE[key] = v
    return v


def event_study(D, sig, hs=(5, 10, 20, 60)):
    s = tradable(D, sig)
    s = s[s['ok']].reset_index(drop=True)
    j, di, e = s['j'].values, s['di'].values, s['ei'].values
    out = s[['code', 'date', 'j', 'di', 'ei']].copy()
    px = D.O[e, j]
    cost = 0.001425 * 0.28 * 2 + 0.003 + np.array([tick_of(p) for p in px]) / px   # 半跳×2
    for h in hs:
        x = e + h - 1
        ok = x < D.T
        xx = np.where(ok, x, D.T - 1)
        r = np.where(ok, D.Af[xx, j] / D.AO[e, j] - 1, np.nan)
        b = np.where(ok, D.ETF[np.minimum(di + h, D.T - 1)] / D.ETF[di] - 1, np.nan)
        p = np.array([pool_ret(D, a, h) for a in di])
        out['r%d' % h] = r
        out['net%d' % h] = (1 + r) * (1 - cost) - 1
        out['etf%d' % h] = b
        out['pool%d' % h] = p
    out['ym'] = out['date'].dt.strftime('%Y-%m')
    return out


def ev_summary(ev, h, mask=None):
    d = ev if mask is None else ev[mask]
    d = d[d['r%d' % h].notna()]
    r, b, p, nr = d['r%d' % h].values, d['etf%d' % h].values, d['pool%d' % h].values, d['net%d' % h].values
    a = cluster_t(r - b, d['ym'].values)
    c = cluster_t(r - p, d['ym'].values)
    cn = cluster_t(nr - p, d['ym'].values)
    return dict(n=int(len(d)), mean=float(r.mean()) if len(d) else np.nan, median=float(np.median(r)) if len(d) else np.nan,
                win=float((r > 0).mean()) if len(d) else np.nan, net=float(nr.mean()) if len(d) else np.nan,
                etf=float(b.mean()) if len(d) else np.nan,
                ex_etf=a['mean'], t_etf=a['t'], ex_pool=c['mean'], t_pool=c['t'], ex_pool_net=cn['mean'], t_pool_net=cn['t'],
                G=a['G'], neff_etf=a['neff'], neff_pool=c['neff'], t_etf_naive=a.get('t_naive'), t_pool_naive=c.get('t_naive'),
                med_ex_etf=float(np.median(r - b)) if len(d) else np.nan, med_ex_pool=float(np.median(r - p)) if len(d) else np.nan)


# ------------------------------------------------------------------ 組合引擎
def run_portfolio(D, sig, exit_rule='E3', idle='cash', K=10, capital=1_000_000.0, start=START, stress=0.0,
                  zero_cost=False, idle_mode='spec', etf_intraday=None, log=False, delist_exit=False):
    """idle: 'cash'|'etf'。idle_mode: 'spec'（照規格：前一日收盤現金賺當日 0050 全日報酬）
       'conservative'：當日開盤買股用掉的現金不賺當日 0050（只賺到前一日），開盤賣股得款也不賺當日。
       'intraday'：用 etf_intraday（當日 0050 開盤→收盤報酬）修正：買股用掉的錢只賺隔夜，開盤賣股得款補賺盤中。"""
    T = D.T
    si = int(np.searchsorted(D.dates.values, np.datetime64(start)))
    etf_r = np.zeros(T)
    etf_r[1:] = D.ETF[1:] / D.ETF[:-1] - 1
    FEE = 0.001425 * 0.28
    TAX = 0.003
    BUY_X = 0.0017 if idle == 'etf' else 0.0
    SELL_X = 0.0007 if idle == 'etf' else 0.0
    if zero_cost: FEE = TAX = BUY_X = SELL_X = 0.0
    by_e = {}
    for r in sig.itertuples(index=False):
        if r.di >= si:
            by_e.setdefault(r.di + 1, []).append(r)
    cash = capital
    pos = {}          # j -> dict
    nav = np.full(T, np.nan)
    nav[si] = capital
    invested = np.full(T, np.nan)
    nheld = np.full(T, np.nan)
    trades = []
    stats = dict(bought=0, held=0, lim_up=0, full=0, cant_afford=0, disp=0, no_open=0, cap5=0, before_start=0)
    lot_switch = np.datetime64('2020-10-26')
    hold_max = {'E1': 20, 'E2': 60}.get(exit_rule, 250)
    # 凍結後加的診斷選項（預設關閉，不影響凍結數字）：下市股在最後一根 K 棒收盤強制出場，避免永遠不出場的殭屍部位
    lastbar = T - 1 - np.argmax(D.has[::-1], axis=0)

    def can_sell(t, j):
        return (not np.isnan(D.C[t, j])) and (not D.lock_dn[t, j])

    def do_sell(t, j, raw_price, reason):
        nonlocal cash
        p = pos.pop(j)
        px = raw_price - (0.0 if zero_cost else tick_of(raw_price) / 2.0)
        amt = p['u'] * px * D.F[t, j]
        fee = 0.0 if zero_cost else max(20.0, amt * FEE)
        net = amt - fee - amt * TAX - amt * stress
        got = net * (1 - SELL_X)
        cash += got
        trades.append(dict(code=D.codes[j], sig_date=p['sig_date'], entry=D.dates[p['e']], exit=D.dates[t], entry_px=p['px'],
                           exit_px=raw_price, shares=p['shares'], cost=p['cost'], proceeds=got, ret=got / p['cost'] - 1,
                           reason=reason, hold=t - p['e'] + 1))
        return got

    for t in range(si + 1, T):
        nav_prev = nav[t - 1]
        day = D.dates.values[t]
        open_sell_proceeds = 0.0
        if idle == 'etf':
            cash *= (1 + etf_r[t])
        cash_after_accrual = cash
        # 2. 開盤賣出前一日收盤已判定者
        for j in [j for j, p in pos.items() if p['flag']]:
            if can_sell(t, j):
                open_sell_proceeds += do_sell(t, j, D.O[t, j], pos[j]['flag'])
        # 3. 開盤買進
        buy_spent = 0.0
        cands = by_e.get(t, [])
        if cands:
            cands = sorted(cands, key=lambda r: (-r.vbreak, r.code))
            for r in cands:
                j = r.j
                if j in pos: stats['held'] += 1; continue
                o = D.O[t, j]
                if np.isnan(o): stats['no_open'] += 1; continue
                if D.open_limit_up[t, j]: stats['lim_up'] += 1; continue
                if D.in_disposal(j, D.dates[t]): stats['disp'] += 1; continue
                if len(pos) >= K: stats['full'] += 1; continue
                budget = min(cash, nav_prev / K)
                cap = D.VAL[t, j] * 0.05
                capped = False
                if budget > cap: budget = cap; capped = True
                px = o + (0.0 if zero_cost else tick_of(o) / 2.0)
                lot = 1000 if day < lot_switch else 1
                unit_cost = px * (1 + FEE + stress) * (1 + BUY_X)
                sh = np.floor(budget / unit_cost / lot + 1e-9) * lot
                while sh > 0:
                    amt = sh * px
                    fee = 0.0 if zero_cost else max(20.0, amt * FEE)
                    tot = (amt + fee + amt * stress) * (1 + BUY_X)
                    if tot <= budget + 1e-6: break
                    sh -= lot
                if sh <= 0: stats['cant_afford'] += 1; continue
                if capped: stats['cap5'] += 1
                cash -= tot
                buy_spent += tot
                stats['bought'] += 1
                stop = floor_tick(o * 0.92) * D.F[t, j]
                tp = ceil_tick(o * 1.20) * D.F[t, j]
                pos[j] = dict(u=sh / D.F[t, j], e=t, px=o, shares=sh, cost=tot, stop=stop, tp=tp, flag=None,
                              sig_date=r.date, pivot_adj=r.pivot_adj, hi=o * D.F[t, j], entry_adj=o * D.F[t, j])
        # 0050 版口徑修正（診斷用）
        if idle == 'etf' and idle_mode != 'spec':
            if idle_mode == 'conservative':
                cash -= buy_spent * etf_r[t]                    # 買股用掉的錢當日完全不賺 0050
            elif idle_mode == 'intraday' and etf_intraday is not None and not np.isnan(etf_intraday[t]):
                cash -= buy_spent * etf_intraday[t]             # 買股用掉的錢不該賺當日盤中
                cash += open_sell_proceeds * etf_intraday[t]    # 開盤賣股得款補賺盤中
        # 4. 盤中停損／停利
        if exit_rule in ('E3', 'E4'):
            for j in list(pos.keys()):
                p = pos[j]
                if np.isnan(D.C[t, j]): continue
                f = D.F[t, j]
                if D.lock_dn[t, j]: continue
                entry_day = (p['e'] == t)
                if (not entry_day) and D.AO[t, j] <= p['stop'] + 1e-12:
                    do_sell(t, j, D.O[t, j], 'gap_stop'); continue
                if D.AL[t, j] <= p['stop'] + 1e-12:
                    do_sell(t, j, p['stop'] / f, 'stop'); continue
                if exit_rule == 'E4':
                    if (not entry_day) and D.AO[t, j] >= p['tp'] - 1e-12:
                        do_sell(t, j, D.O[t, j], 'gap_tp'); continue
                    if D.AH[t, j] >= p['tp'] - 1e-12:
                        do_sell(t, j, p['tp'] / f, 'tp'); continue
        # 5. 收盤
        for j in list(pos.keys()):
            p = pos[j]
            if delist_exit and t == lastbar[j] and t < T - 1:
                do_sell(t, j, D.C[t, j], 'delist'); continue
            hd = t - p['e'] + 1
            if hd >= hold_max:
                if can_sell(t, j):
                    do_sell(t, j, D.C[t, j], 'time')
                continue
            if np.isnan(D.C[t, j]) or p['flag']: continue
            a = D.A[t, j]
            if exit_rule == 'E3':
                m = D.AMA50[t, j]
                if not np.isnan(m) and a < m: p['flag'] = 'ma50'
            elif exit_rule == 'E5':
                if a < p['pivot_adj'] - 1e-12: p['flag'] = 'pivot'
            elif exit_rule == 'E6':
                p['hi'] = max(p['hi'], a)
                if a <= 0.90 * p['hi']: p['flag'] = 'trail'
        mv = sum(p['u'] * D.Af[t, j] for j, p in pos.items())
        nav[t] = cash + mv
        invested[t] = mv / nav[t]
        nheld[t] = len(pos)
    res = dict(nav=pd.Series(nav, index=D.dates).dropna(), invested=float(np.nanmean(invested[si + 1:])),
               nheld=float(np.nanmean(nheld[si + 1:])), stats=stats, trades=pd.DataFrame(trades),
               inv_series=pd.Series(invested, index=D.dates).dropna())
    # 凍結後加的診斷輸出（不影響任何數字）：資料終點仍未出場的部位
    res['open_pos'] = [dict(code=D.codes[j], entry=D.dates[p['e']], sig_date=p['sig_date'], cost=p['cost'], value=p['u'] * D.Af[T - 1, j],
                            last_bar=D.dates[np.nonzero(D.has[:, j])[0][-1]], flag=p['flag']) for j, p in pos.items()]
    return res


def ann(series, a, b):
    s = series[(series.index >= pd.Timestamp(a)) & (series.index <= pd.Timestamp(b))]
    days = (s.index[-1] - s.index[0]).days
    return (s.iloc[-1] / s.iloc[0]) ** (365.25 / days) - 1


def monthly_t(nav, bench, a=None, b=None, drop_partial_last=False):
    df = pd.DataFrame({'s': nav, 'b': bench}).dropna()
    me = df.groupby([df.index.year, df.index.month]).tail(1)
    if drop_partial_last:
        last = df.index[-1]
        # 最後一個月如果不是月底就丟掉
        if (last + pd.offsets.BDay(1)).month == last.month:
            me = me.iloc[:-1]
    r = me.pct_change().dropna()
    if a is not None: r = r[r.index >= pd.Timestamp(a)]
    if b is not None: r = r[r.index <= pd.Timestamp(b)]
    x = (r['s'] - r['b']).values
    return dict(n=int(len(x)), mean=float(x.mean()), sd=float(x.std(ddof=1)), t=float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))))


def maxdd(s):
    return float((s / s.cummax() - 1).min())


def perf(D, res, label=''):
    nav = res['nav']
    etf = pd.Series(D.ETF, index=D.dates)
    out = {}
    for nm, a in (('full', '2005-01-31'), ('2019', '2019-01-01')):
        sa = ann(nav, a, '2026-10-22'); ba = ann(etf, a, '2026-10-22')
        mt = monthly_t(nav, etf.reindex(nav.index), a=None if nm == 'full' else '2019-01-01')
        out[nm] = dict(ann=float(sa), etf=float(ba), ex=float(sa - ba), t=mt['t'], n=mt['n'], mean_m=mt['mean'], sd_m=mt['sd'])
    out['mdd'] = maxdd(nav)
    out['invested'] = res['invested']
    out['nheld'] = res['nheld']
    out['stats'] = res['stats']
    out['n_trades'] = int(len(res['trades']))
    return out
