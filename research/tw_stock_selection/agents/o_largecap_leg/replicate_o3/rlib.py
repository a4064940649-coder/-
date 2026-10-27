"""獨立複製 o_largecap_leg V01：資料、訊號、可成交引擎、統計（從頭寫，不 import 研究員程式）。"""
import json, math
import numpy as np, pandas as pd

P = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel/'
AG = '/home/user/-/research/tw_stock_selection/agents/'
EX = '/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/'
END = pd.Timestamp('2026-10-12')
GAP_DAYS = pd.to_datetime(['2017-08-03', '2018-07-12'])
GAP_RANGE = (pd.Timestamp('2024-02-15'), pd.Timestamp('2024-03-15'))

CB = 0.000855 + 0.001
CS = 0.000855 + 0.003 + 0.001
COSTS = {'normal': (CB, CS), 'stress3': (CB + 0.003, CS + 0.003), 'old05': (0.005, 0.005), 'zero': (0.0, 0.0)}


def load():
    D = {k: pd.read_parquet(P + k + '.parquet') for k in ['adj', 'close_raw', 'open', 'value', 'ret']}
    return D


def sig_dates(idx_valid):
    s = pd.Series(idx_valid, index=idx_valid)
    s = s[s.dt.day >= 11]
    g = s.groupby([s.index.year, s.index.month]).first()
    return [x for x in g if pd.Timestamp('2006-04-01') <= x <= END]


def gap_roll(df_valid, win, how):
    """df_valid：已剔除 <300 檔的日子。窗內缺值只豁免缺口日。"""
    ex = pd.Series(df_valid.index.isin(GAP_DAYS) | ((df_valid.index >= GAP_RANGE[0]) & (df_valid.index <= GAP_RANGE[1])), index=df_valid.index)
    na = df_valid.isna()
    exna = na.mul(ex, axis=0)
    ok = (df_valid.notna().astype(float).rolling(win).sum() + exna.astype(float).rolling(win).sum()) >= win
    r = df_valid.rolling(win, min_periods=1)
    v = r.max() if how == 'max' else r.mean()
    return v.where(ok)


def tick(p):
    return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)


def limit_up(ref, date):
    lim = 0.10 if date >= pd.Timestamp('2015-06-01') else 0.07
    raw = ref * (1 + lim)
    t = tick(raw)
    lu = np.floor(raw / t + 1e-9) * t
    # 漲停價跨越跳動點級距時，以較小級距再確認
    return lu


def foreign_codes(cols):
    sl = pd.read_csv(EX + 'stock_list.csv', dtype=str)
    names = dict(zip(sl.code, sl.name))
    tr = pd.read_csv('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv', dtype={'code': str}, usecols=['code', 'name'])
    for c, n in tr.drop_duplicates('code').values:
        names.setdefault(c, n)
    out = set()
    for c in cols:
        n = str(names.get(c, ''))
        if 'KY' in n.upper() or c.startswith('91') or n.endswith('DR'):
            out.add(c)
    return out


def fin_codes(cols):
    sl = pd.read_csv(EX + 'stock_list.csv', dtype=str)
    f = set(sl.code[sl.industry == '金融保險業'])
    return {c for c in cols if c in f or c.startswith('28') or c in ('5876', '5880')}


def rev_yoy():
    y = pd.read_parquet(AG + 'o_rev_survivorship/yoy_long_R_DEL.parquet')
    r = pd.read_csv(EX + 'extras/rev_5y.csv.gz', dtype={'code': str})
    r5 = r.pivot_table(index='ym', columns='code', values='yoy')
    return y, r5


def yoy3_at(t, y, r5, mode='mean'):
    """訊號日 t 的近 3 月（m−1～m−3）營收年增率平均；t<2021-03-01 用 R_DEL，否則 rev_5y。"""
    src = y if t < pd.Timestamp('2021-03-01') else r5
    m = pd.Period(t, 'M')
    ms = [str(m - k) for k in (1, 2, 3)]
    rows = [src.loc[x] if x in src.index else pd.Series(dtype=float) for x in ms]
    df = pd.concat(rows, axis=1)
    return df.mean(axis=1, skipna=False)


def disposal_set():
    d = pd.read_csv(EX + 'extras/disposal_5y.csv.gz', dtype={'code': str})
    d = d[d.flag.astype(str).str.contains('處置')]
    d['start'] = pd.to_datetime(d.start); d['end'] = pd.to_datetime(d.end)
    return d[['code', 'start', 'end']]


def in_disposal(dsp, day):
    m = (dsp.start <= day) & (dsp.end >= day)
    return set(dsp.code[m])


class Data:
    def __init__(self):
        D = load()
        self.adj, self.close, self.open, self.value, self.ret = D['adj'], D['close_raw'], D['open'], D['value'], D['ret']
        self.idx = self.adj.index
        cnt = self.adj.notna().sum(1)
        vidx = self.idx[cnt >= 300]
        a = self.adj.loc[vidx]
        self.r52 = a / gap_roll(a, 252, 'max')
        self.v60 = gap_roll(self.value.loc[vidx], 60, 'mean')
        self.sigs = sig_dates(vidx)
        self.y, self.r5 = rev_yoy()
        self.fin = fin_codes(self.adj.columns)
        self.foreign = foreign_codes(self.adj.columns)
        self.dsp = disposal_set()
        # 還原開盤
        self.adjopen = self.open * (self.adj / self.close)
        self.adjf = self.adj.ffill()
        self.nidx = self.idx[(self.idx >= self.sigs[0]) & (self.idx <= END)]

    def next_day(self, t):
        i = self.idx.get_loc(t)
        return self.idx[i + 1]

    def base_frame(self, t, disp_day='t'):
        nd = self.next_day(t)
        df = pd.DataFrame({'v60': self.v60.loc[t], 'close': self.close.loc[t], 'r52': self.r52.loc[t],
                           'nopen': self.open.loc[nd]})
        df['yoy3'] = yoy3_at(t, self.y, self.r5).reindex(df.index)
        dday = nd if disp_day == 'next' else t
        dset = in_disposal(self.dsp, dday) if t >= pd.Timestamp('2021-09-01') else set()
        df['disp'] = df.index.isin(dset)
        return df

    def main_pool(self, t):
        df = self.base_frame(t)
        p = df[(df.v60 >= 1e7) & (df.close >= 10)].sort_values('v60', ascending=False).head(300)
        p = p[~p.disp & (p.nopen > 0) & p.yoy3.notna() & p.r52.notna()]
        return p[~p.index.isin(self.fin)]

    def lc_pool(self, t, n=50, drop=()):
        df = self.base_frame(t)
        ok = (df.v60 >= 1e7) & (df.close >= 10) & df.yoy3.notna() & df.r52.notna() & ~df.disp & (df.nopen > 0)
        ok &= ~df.index.isin(self.fin) & ~df.index.isin(self.foreign) & ~df.index.isin(list(drop))
        p = df[ok].sort_values('v60', ascending=False).head(n).copy()
        return p

    @staticmethod
    def rank(p):
        p = p.copy()
        p['score'] = 0.5 * p.r52.rank(pct=True) + 0.5 * p.yoy3.rank(pct=True)
        p['score_r'] = p.score.round(10)
        p['code'] = p.index
        return list(p.sort_values(['score_r', 'yoy3', 'code'], ascending=[False, False, True]).index)

    def lu_blocked(self, code, d):
        """d 日開盤是否漲停（參考價＝收盤/(1+R)，R 缺時用前收）。"""
        o = self.open.at[d, code]
        if not (o > 0):
            return True
        c, r = self.close.at[d, code], self.ret.at[d, code]
        if pd.notna(c) and pd.notna(r):
            ref = c / (1 + r)
        else:
            i = self.idx.get_loc(d)
            ref = self.close[code].iloc[:i].dropna().iloc[-1]
        return bool(o >= limit_up(ref, d) - 1e-9)

    def pick(self, ranked, k, held, d):
        out = []
        for c in ranked:
            if len(out) >= k:
                break
            if c in held or not self.lu_blocked(c, d):
                out.append(c)
        return out

    def simulate(self, targets, cost='normal', delist='freeze', xrule=None):
        """targets：{sig_date: {code: w}}（w 合計 1，可含 'CASH'）。訊號日收盤決定，隔日開盤成交，換股日調回目標權重。"""
        cb, cs = COSTS[cost]
        nav = pd.Series(index=self.nidx, dtype=float)
        hold = {}  # code -> value at previous close
        cash = 1.0
        execs = {self.next_day(t): w for t, w in targets.items() if self.next_day(t) <= END}
        prev = None
        adjf, adjopen = self.adjf, self.adjopen
        turn = []
        for d in self.nidx:
            if prev is None:
                nav[d] = 1.0; prev = d; continue
            if d in execs:
                # 前收→開盤
                for c in list(hold):
                    o = adjopen.at[d, c]
                    if o > 0:
                        hold[c] *= o / adjf.at[prev, c]
                tot = cash + sum(hold.values())
                tgt = execs[d]
                frozen = {c: v for c, v in hold.items() if not (adjopen.at[d, c] > 0)}
                fz = sum(frozen.values())
                # 下市（之後永無資料）直接以最後價計入現金（扣賣出成本）
                for c in list(frozen):
                    if xrule is not None and c in xrule:
                        rel, mult = xrule[c]
                        if d >= rel:
                            cash += frozen[c] * mult; del hold[c]; fz -= frozen.pop(c)
                    elif delist == 'cash' and self.adj[c].loc[d:].notna().sum() == 0:
                        cash += frozen[c] * (1 - cs); del hold[c]; fz -= frozen.pop(c)
                tot = cash + sum(hold.values())
                free = tot - fz
                wt = {c: w for c, w in tgt.items() if c not in frozen}
                sw = sum(wt.values())
                tv = {c: free * w / sw for c, w in wt.items()} if sw > 0 else {}
                for _ in range(3):
                    buys = sum(max(tv.get(c, 0) - hold.get(c, 0), 0) for c in set(tv) | set(hold) if c not in frozen)
                    sells = sum(max(hold.get(c, 0) - tv.get(c, 0), 0) for c in set(tv) | set(hold) if c not in frozen)
                    fee = buys * cb + sells * cs
                    tv = {c: (free - fee) * w / sw for c, w in wt.items()}
                turn.append(sells / tot)
                newh = dict(frozen)
                for c, v in tv.items():
                    if c != 'CASH':
                        newh[c] = newh.get(c, 0) + v
                cash = tv.get('CASH', 0.0)
                hold = newh
                # 開盤→收盤
                for c in list(hold):
                    o = adjopen.at[d, c]
                    a = adjf.at[d, c]
                    if o > 0:
                        hold[c] *= a / o
            else:
                for c in list(hold):
                    a0, a1 = adjf.at[prev, c], adjf.at[d, c]
                    if a0 > 0 and a1 > 0:
                        hold[c] *= a1 / a0
            nav[d] = cash + sum(hold.values())
            prev = d
        return nav, float(np.mean(turn)) if turn else float('nan')


def bench0050(idx):
    b = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj']
    return b.reindex(idx, method='ffill')


# ---------- 統計 ----------
SEG = {'SEL': ('2006-04-01', '2018-12-31'), 'VAL': ('2019-01-01', '2026-09-30'), 'VAL_A': ('2019-01-01', '2023-12-31'),
       'HOLD': ('2024-01-01', '2026-09-30'), 'ALL': ('2006-04-01', '2026-09-30')}


def mret(nav, sigs):
    s = [x for x in sigs if x <= END]
    pts = list(s) + ([END] if s[-1] != END else [])
    v = nav.reindex(pts)
    r = v.shift(-1) / v - 1
    return r.iloc[:-1]


def seg_mask(r, seg):
    a, b = SEG[seg]
    return r[(r.index >= a) & (r.index <= b)]


NAVSEG = {'SEL': ('2006-04-11', '2019-01-11'), 'VAL': ('2019-01-11', '2026-10-12'), 'VAL_A': ('2019-01-11', '2023-12-31'),
          'HOLD': ('2023-12-29', '2026-10-12'), 'ALL': ('2006-04-11', '2026-10-12')}


def seg_nav_cagr(nav, sigs=None, seg='ALL'):
    a, e = NAVSEG[seg]
    a = nav.index[nav.index >= a][0]; e = nav.index[nav.index <= e][-1]
    yrs = (e - a).days / 365.25
    return float((nav[e] / nav[a]) ** (1 / yrs) - 1)


def seg_dd(nav, seg='ALL'):
    a, e = NAVSEG[seg]
    n = nav[(nav.index >= a) & (nav.index <= e)]
    return float((n / n.cummax() - 1).min())


def tstat(x):
    x = pd.Series(x).dropna()
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))


def maxdd(nav, a=None, b=None):
    n = nav
    if a is not None: n = n[n.index >= a]
    if b is not None: n = n[n.index <= b]
    return float((n / n.cummax() - 1).min())
