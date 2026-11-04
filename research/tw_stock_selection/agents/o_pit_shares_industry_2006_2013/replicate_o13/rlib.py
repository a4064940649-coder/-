"""Independent replication of o13 (PIT shares / industry 2006-2013). Written from the frozen spec only."""
import os, glob, gzip, io, json, pickle
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TOP = os.path.dirname(HERE)
RAW = os.path.join(TOP, 'raw')
AG = os.path.dirname(TOP)
PANEL = os.path.join(AG, 'o_v7_listing_exec_engine', 'data')
DELIST = os.path.join(AG, 'o_infra_v3_rebuild', 'delist_prices.parquet')
CACHE = os.path.join(HERE, 'cache')
OUT = os.path.join(HERE, 'out')
os.makedirs(CACHE, exist_ok=True); os.makedirs(OUT, exist_ok=True)

PRIO = ['tejm', 'tejq', 'twq', 'gq', 'ga', 'o8']


def _cache(name, fn):
    p = os.path.join(CACHE, name + '.pkl')
    if os.path.exists(p):
        with open(p, 'rb') as f:
            return pickle.load(f)
    obj = fn()
    with open(p, 'wb') as f:
        pickle.dump(obj, f)
    return obj


def load_panel():
    def f():
        P = {}
        for k in ['close_raw', 'open', 'value', 'flag']:
            P[k] = pd.read_parquet(os.path.join(PANEL, k + '.parquet'))
        dp = pd.read_parquet(DELIST)
        dp['date'] = pd.to_datetime(dp['date'])
        dp['code'] = dp['code'].astype(str)
        for k, col in [('close_raw', 'close'), ('open', 'open'), ('value', 'value')]:
            w = dp.pivot_table(index='date', columns='code', values=col, aggfunc='last')
            base = P[k]
            idx = base.index.union(w.index)
            cols = base.columns.union(w.columns)
            base = base.reindex(index=idx, columns=cols)
            w = w.reindex(index=idx, columns=cols)
            P[k] = base.where(base.notna(), w)
        idx = P['close_raw'].index
        cols = P['close_raw'].columns
        for k in P:
            P[k] = P[k].reindex(index=idx, columns=cols)
        P['close_raw'].index = pd.DatetimeIndex(P['close_raw'].index).as_unit('ns')
        for k in P:
            P[k].index = P['close_raw'].index
        return P
    return _cache('panel', f)


def td_index():
    return load_panel()['close_raw'].index


def to_td(dates, TD=None):
    """map dates to last trading day <= date"""
    TD = td_index() if TD is None else TD
    d = pd.DatetimeIndex(pd.to_datetime(dates)).as_unit('ns')
    pos = TD.searchsorted(d, side='right') - 1
    out = pd.DatetimeIndex([TD[p] if p >= 0 else pd.NaT for p in pos])
    return out


def listing_mask():
    def f():
        P = load_panel(); TD = P['close_raw'].index; cols = P['close_raw'].columns
        lm = pd.read_csv(os.path.join(RAW, 'center', 'listing_master.csv'), dtype=str)
        M = pd.DataFrame(False, index=TD, columns=cols)
        for _, r in lm.iterrows():
            if r['code'] not in cols or not isinstance(r['segments'], str):
                continue
            for seg in r['segments'].split(';'):
                a, b = seg.split('~')
                M.loc[a:b, r['code']] = True
        return M
    return _cache('listing', f)


def survivors():
    """近 6 個交易日有成交、上市主檔為上市、非 0／91 開頭"""
    def f():
        P = load_panel(); v = P['value']
        traded = (v > 0).astype(float).rolling(6, min_periods=1).max() > 0
        L = listing_mask()
        ok = np.array([not (c.startswith('0') or c.startswith('91')) for c in v.columns])
        S = traded & L
        S.loc[:, ~ok] = False
        return S
    return _cache('surv', f)


# ---------------------------------------------------------------- anchors

def anchors_raw():
    def f():
        P = load_panel(); TD = P['close_raw'].index; cols = set(P['close_raw'].columns)
        close = P['close_raw']
        rows = []
        # tejm
        t = pd.read_csv(os.path.join(RAW, 'tej_ff5', 'shares_monthly.csv'), dtype={'code': str})
        t = t[t.market.isin(['TSE', 'REG'])]
        t['date'] = to_td(pd.to_datetime(t['month_end'], format='%Y/%m/%d'), TD)
        rows.append(pd.DataFrame({'code': t.code, 'date': t.date, 'src': 'tejm', 'shares': t.shares_k.astype(float) * 1000}))
        # tejq
        x = pd.read_excel(os.path.join(RAW, 'tej_jackwang', '市值.xlsx'))
        x['code'] = x['ticker'].astype(str)
        x = x[x.code.isin(cols)].copy()
        x['date'] = to_td(x['date'], TD)
        cl = close.stack(); cl.index.names = ['date', 'code']; cl = cl.rename('c').reset_index()
        x = x.merge(cl, on=['date', 'code'], how='left')
        x = x[(x.c > 0) & (x.mv > 0)]
        rows.append(pd.DataFrame({'code': x.code, 'date': x.date, 'src': 'tejq', 'shares': x.mv * 1000.0 / x.c}))
        # gq
        g = []
        for fn in glob.glob(os.path.join(RAW, 'goodinfo_q', '*_BzPerformance.csv')):
            d = pd.read_csv(fn, dtype=str)
            d = d[['sid', '季度', '股本(億)']].copy()
            d['v'] = pd.to_numeric(d['股本(億)'].str.replace(',', ''), errors='coerce')
            d = d[d.v > 0]
            q = d['季度'].str.extract(r'(\d{4})Q(\d)')
            d['qe'] = pd.PeriodIndex.from_fields(year=q[0].astype(int), quarter=q[1].astype(int), freq='Q').end_time.normalize()
            g.append(d)
        g = pd.concat(g)
        g['date'] = to_td(g['qe'], TD)
        rows.append(pd.DataFrame({'code': g.sid.astype(str), 'date': g.date, 'src': 'gq', 'shares': g.v * 1e7}))
        # ga
        a = pd.read_csv(os.path.join(RAW, 'goodinfo_a', 'equity_year.csv'), dtype={'code': str})
        a = a[a.equity_e8 > 0]
        a['date'] = to_td(pd.to_datetime(a.year.astype(str) + '-12-31'), TD)
        rows.append(pd.DataFrame({'code': a.code, 'date': a.date, 'src': 'ga', 'shares': a.equity_e8 * 1e7}))
        # twq
        w = []
        for fn in glob.glob(os.path.join(RAW, 'twsec_q', '*.csv')):
            with open(fn, encoding='utf-8-sig') as fh:
                first = fh.readline()
            code = first.split(',')[-1].strip()
            d = pd.read_csv(fn, skiprows=1, dtype=str)
            d['v'] = pd.to_numeric(d['股本'], errors='coerce')
            q = d['季度'].str.extract(r'(\d{4})\s*Q(\d)')
            d = d[q[0].notna() & (d.v > 0)]
            q = q.loc[d.index]
            d['qe'] = pd.PeriodIndex.from_fields(year=q[0].astype(int), quarter=q[1].astype(int), freq='Q').end_time.normalize()
            d = d[d.qe >= '2013-01-01']
            d['code'] = code
            w.append(d)
        w = pd.concat(w)
        w['date'] = to_td(w['qe'], TD)
        rows.append(pd.DataFrame({'code': w.code, 'date': w.date, 'src': 'twq', 'shares': w.v * 100.0}))
        # o8 2013Q1..2014Q4 quarter ends
        o8 = pd.read_parquet(os.path.join(RAW, 'center', 'o8_shares_pit.parquet'))
        o8.index = pd.DatetimeIndex(o8.index).as_unit('ns')
        qe = pd.period_range('2013Q1', '2014Q4', freq='Q').end_time.normalize()
        qd = to_td(qe, TD)
        s = o8.reindex(qd).stack(); s.index.names = ['date', 'code']; s = s.rename('shares').reset_index()
        s['src'] = 'o8'
        rows.append(s[['code', 'date', 'src', 'shares']])
        A = pd.concat(rows, ignore_index=True)
        A = A[A.code.isin(cols) & A.date.notna() & (A.shares > 0)]
        A['code'] = A.code.astype(str)
        return A
    return _cache('anchors_raw', f)


def anchors():
    """one anchor per (code, trading day) by priority"""
    def f():
        A = anchors_raw().copy()
        A['pr'] = A.src.map({s: i for i, s in enumerate(PRIO)})
        A = A.sort_values(['code', 'date', 'pr']).drop_duplicates(['code', 'date'], keep='first')
        return A.drop(columns='pr').reset_index(drop=True)
    return _cache('anchors', f)


# ---------------------------------------------------------------- events

def gap_factor():
    """f = 前收 × (1+當日非事件股開盤跳空中位數) ÷ 開盤，for flag 1/2/3 cells, 2005-06..2021-06"""
    def f():
        P = load_panel()
        c = P['close_raw']; o = P['open']; fl = P['flag']; v = P['value']
        pc = c.ffill().shift(1)
        gap = o / pc - 1
        ev = fl.isin([1, 2, 3, 4, 5])
        ok = (~ev) & (o > 0) & (pc > 0) & (v > 0)
        med = gap.where(ok).median(axis=1)
        F = pc.mul(1 + med, axis=0) / o
        E = fl.isin([1, 2, 3]) & (o > 0) & (pc > 0)
        E.loc[:'2005-05-31'] = False
        E.loc['2021-07-01':] = False
        s = F.where(E).stack().dropna()
        s.index.names = ['date', 'code']
        out = s.rename('f').reset_index()
        flg = fl.stack(); flg.index.names = ['date', 'code']
        out = out.merge(flg.rename('flag').reset_index(), on=['date', 'code'], how='left')
        return out
    return _cache('gapf', f)


def event_m(ev):
    """ev: DataFrame date, code, f, flag -> adds m per spec"""
    ev = ev.copy()
    ev['m'] = 1.0
    d12 = ev[ev.flag.isin([1, 2])].copy()
    y = d12.f - 1
    d12['inwin'] = (y > 0.005) & (y <= 0.08)
    mk = d12[d12.inwin].groupby(d12.date.dt.year)['f'].median() - 1
    yhat = np.full(len(d12), np.nan)
    d12 = d12.reset_index().rename(columns={'index': 'orig'})
    for code, g in d12.groupby('code'):
        w = g[g.inwin]
        if len(w) == 0:
            continue
        wd = w.date.values; wy = (w.f - 1).values
        for i, r in g.iterrows():
            lo = r.date - pd.DateOffset(years=3); hi = r.date + pd.DateOffset(years=3)
            sel = (wd >= np.datetime64(lo)) & (wd <= np.datetime64(hi))
            if sel.any():
                yhat[i] = np.median(wy[sel])
    yh = pd.Series(yhat)
    fill = d12.date.dt.year.map(mk)
    yh = yh.fillna(fill)
    m = d12.f.values / (1 + yh.values)
    m = np.where(m <= 1.02, 1.0, m)
    ev.loc[d12.orig.values, 'm'] = m
    d3 = ev.flag == 3
    ev.loc[d3, 'm'] = np.where((ev.loc[d3, 'f'] - 1).abs() > 0.02, ev.loc[d3, 'f'], 1.0)
    ev['m'] = ev['m'].clip(0.2, 3.0)
    return ev


def events():
    return _cache('events', lambda: event_m(gap_factor()))


# ---------------------------------------------------------------- construction

class Builder:
    """per-stock construction on trading-day positions"""

    def __init__(self, ev_df, version='V2'):
        self.TD = td_index()
        self.T = len(self.TD)
        self.version = version
        e = ev_df[ev_df.m != 1.0]
        pos = self.TD.get_indexer(pd.DatetimeIndex(e.date).as_unit('ns'))
        self.ev = {}
        for code, p, lm in zip(e.code.values, pos, np.log(e.m.values)):
            self.ev.setdefault(code, []).append((p, lm))
        self.cum = {}
        for code, lst in self.ev.items():
            lst.sort()
            ce = np.zeros(self.T); cn = np.zeros(self.T, dtype=int)
            for p, lm in lst:
                ce[p] += lm; cn[p] += 1
            self.cum[code] = (np.cumsum(ce), np.cumsum(cn), np.array([p for p, _ in lst]))

    def _ce(self, code):
        if code in self.cum:
            return self.cum[code]
        return (np.zeros(self.T), np.zeros(self.T, dtype=int), np.array([], dtype=int))

    def series(self, code, apos, aL):
        """full log-shares path over all TD; apos sorted unique positions, aL log values"""
        ce, cn, epos = self._ce(code)
        t = np.arange(self.T)
        out = np.full(self.T, np.nan)
        n = len(apos)
        if n == 0:
            return out
        p0 = apos[0]
        out[:p0] = aL[0] - (ce[p0] - ce[:p0])
        pn = apos[-1]
        out[pn:] = aL[-1] + ce[pn:] - ce[pn]
        for i in range(n - 1):
            a, b = apos[i], apos[i + 1]
            La, Lb = aL[i], aL[i + 1]
            tt = t[a:b + 1]
            tot = Lb - La
            ne = cn[b] - cn[a]
            if ne == 0 or self.version == 'V3':
                out[a:b + 1] = La + tot * (tt - a) / (b - a)
                continue
            M = ce[b] - ce[a]
            fallback = (M * tot < 0) or (abs(M) > abs(tot))
            if self.version == 'V1':
                fallback = True
            if fallback:
                e1 = epos[np.searchsorted(epos, a, side='right')]
                out[a:b + 1] = np.where(tt < e1, La, Lb)
            else:
                out[a:b + 1] = La + (ce[a:b + 1] - ce[a]) + (tot - M) * (tt - a) / (b - a)
        out[apos] = aL
        return out

    def value_at(self, code, pos, apos, aL):
        ce, cn, epos = self._ce(code)
        n = len(apos)
        if n == 0:
            return np.nan
        j = np.searchsorted(apos, pos)
        if j < n and apos[j] == pos:
            return aL[j]
        if j == 0:
            p0 = apos[0]
            return aL[0] - (ce[p0] - ce[pos])
        if j == n:
            pn = apos[-1]
            return aL[-1] + ce[pos] - ce[pn]
        a, b = apos[j - 1], apos[j]
        La, Lb = aL[j - 1], aL[j]
        tot = Lb - La
        ne = cn[b] - cn[a]
        if ne == 0 or self.version == 'V3':
            return La + tot * (pos - a) / (b - a)
        M = ce[b] - ce[a]
        fallback = (M * tot < 0) or (abs(M) > abs(tot)) or self.version == 'V1'
        if fallback:
            e1 = epos[np.searchsorted(epos, a, side='right')]
            return La if pos < e1 else Lb
        return La + (ce[pos] - ce[a]) + (tot - M) * (pos - a) / (b - a)


def anchor_arrays(A=None):
    A = anchors() if A is None else A
    TD = td_index()
    pos = TD.get_indexer(pd.DatetimeIndex(A.date).as_unit('ns'))
    A = A.assign(pos=pos, L=np.log(A.shares.values))
    d = {}
    for code, g in A.sort_values('pos').groupby('code'):
        d[code] = (g.pos.values, g.L.values, g.src.values, g.date.values)
    return d


def full_build(version='V2'):
    def f():
        B = Builder(events(), version)
        AA = anchor_arrays()
        cols = load_panel()['close_raw'].columns
        S = np.full((B.T, len(cols)), np.nan)
        for j, c in enumerate(cols):
            if c in AA:
                S[:, j] = np.exp(B.series(c, AA[c][0], AA[c][1]))
        return pd.DataFrame(S, index=B.TD, columns=cols)
    return _cache('full_' + version, f)


def adv(n=120):
    def f():
        v = load_panel()['value']
        return v.rolling(n, min_periods=1).mean()
    return _cache('adv%d' % n, f)


def impute_ratio(mcap_row, adv_row, surv_row):
    """median(mcap/ADV) over stocks with shares, ADV rank 30..200 among survivors"""
    ok = surv_row & adv_row.notna() & (adv_row > 0)
    r = adv_row[ok].rank(ascending=False, method='first')
    sel = r[(r >= 30) & (r <= 200)].index
    x = (mcap_row[sel] / adv_row[sel]).dropna()
    return x.median()


def month_last_td(start, end):
    TD = td_index()
    s = pd.Series(TD, index=TD).loc[start:end]
    return pd.DatetimeIndex(s.groupby(s.index.to_period('M')).max().values)
