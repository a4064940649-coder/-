"""Independent replication o8: signals, universes, tilt, engine."""
import numpy as np, pandas as pd
from rdata import *

CB, CS = 0.000855 + 0.001, 0.000855 + 0.003 + 0.001
CST = 0.005
LAM = 1.5


class World:
    def __init__(self, fix4763=False):
        cl, op, rt, va = load_panel()
        self.days = cl.index
        cl = cl.loc['2012-01-01':]; op = op.loc['2012-01-01':]; rt = rt.loc['2012-01-01':]; va = va.loc['2012-01-01':]
        self.cl, self.op, self.rt, self.va = cl, op, rt, va
        self.days = cl.index
        self.clf = cl.ffill()
        sh = load_shares().reindex(index=self.days).ffill()
        sh = sh.reindex(columns=cl.columns)
        if fix4763:  # sensitivity: TDCC 2021 truth 58.6M vs PIT 741M; 2025-06-30 1:10 split
            a = sh.index < pd.Timestamp('2023-07-27'); b = (sh.index >= pd.Timestamp('2023-07-27')) & (sh.index < pd.Timestamp('2025-06-30'))
            c0 = sh.index >= pd.Timestamp('2019-09-04')
            sh.loc[a & c0, '4763'] = sh.loc[a & c0, '4763'] / 12.65
            sh.loc[b, '4763'] = sh.loc[b, '4763'] / 10
        self.sh = sh
        self.mcap = sh * self.clf
        # 52w ratio gapfix
        good = cl.notna().sum(1) >= 300
        adj = (1 + rt.fillna(0)).cumprod().where(self.clf.notna())
        ag = adj[good]
        r52 = ag / ag.rolling(252, min_periods=200).max()
        self.r52 = r52.reindex(self.days).ffill(limit=3)
        self.r52 = self.r52.where(cl.notna())
        self.rev = revenue()
        self.ind = industries()
        self.fin = {c: is_fin(c, self.ind) for c in cl.columns}
        self.dr = {c: c.startswith('91') or not c[:4].isdigit() or c.startswith('00') for c in cl.columns}
        self.traded6 = (va.fillna(0) > 0).rolling(6, min_periods=1).max() > 0
        self.b50 = bench0050()
        self.sched, self.first_eff, self.before_first = tw50_members()
        self.pre = None
        self.tx = taiex_days()

    # ---------- dates ----------
    def deadline(self, per):
        day = 15 if str(per) in ('2019-02', '2023-04') else 10
        t0 = pd.Timestamp(per.year, per.month, day)
        return self.tx[self.tx >= t0][0]

    def signal_days(self, j, start='2013-12', end='2026-10'):
        out = {}
        for per in pd.period_range(start, end, freq='M'):
            D = self.deadline(per)
            pos = self.days.searchsorted(D)
            if pos + j < len(self.days):
                out[per] = self.days[pos + j]
        return out

    def official_days(self, start='2013-12', end='2026-10'):
        out = {}
        for per in pd.period_range(start, end, freq='M'):
            t0 = pd.Timestamp(per.year, per.month, 11)
            pos = self.days.searchsorted(t0)
            if pos < len(self.days) and self.days[pos].to_period('M') == per:
                out[per] = self.days[pos]
        return out

    # ---------- membership ----------
    def build_pre(self):
        """Rule-based TW50 reconstruction 2013-12..2015-06 reviews using PIT mcap."""
        effs = [pd.Timestamp(x) for x in ['2013-12-20', '2014-03-21', '2014-06-20', '2014-09-19',
                                         '2014-12-19', '2015-03-20', '2015-06-19', '2015-09-18']]
        mem = None
        res = []
        for e in effs:
            per = (e.to_period('M') - 1)
            base = self.days[self.days.to_period('M') == per][-1]
            mc = self.mcap.loc[base]
            ok = [c for c in mc.index if not self.dr[c] and np.isfinite(mc[c]) and self.traded6.loc[base, c]]
            rk = mc[ok].sort_values(ascending=False)
            rank = pd.Series(np.arange(1, len(rk) + 1), index=rk.index)
            if mem is None:
                new = list(rk.index[:50])
            else:
                keep = [c for c in mem if c in rank.index and rank[c] <= 60]
                add = [c for c in rk.index[:40] if c not in mem]
                new = keep + add
                if len(new) > 50:
                    new = sorted(new, key=lambda c: rank[c])[:50]
                i = 0
                while len(new) < 50:
                    c = rk.index[i]; i += 1
                    if c not in new:
                        new.append(c)
            mem = new
            res.append((e, list(new)))
        self.pre = res
        return res

    def t1(self, d):
        if d > self.first_eff or (d == self.first_eff):
            cur = None
            for eff, m in self.sched:
                if eff <= d:
                    cur = m
            return cur
        cur = None
        for eff, m in self.pre:
            if eff <= d and eff < self.first_eff:
                cur = m
        return cur

    def universe(self, d, which):
        u = [c for c in self.t1(d) if c in self.cl.columns]
        if which == 'T2':
            mc = self.mcap.loc[d]
            ok = [c for c in mc.index if not self.dr[c] and np.isfinite(mc[c]) and mc[c] > 0 and self.traded6.loc[d, c]]
            top = mc[ok].sort_values(ascending=False).index[:100]
            u = list(dict.fromkeys(u + list(top)))
        return u

    # ---------- weights ----------
    def weights(self, d, per, which, lam=LAM, perms=0, seed=0):
        u = self.universe(d, which)
        mc = self.mcap.loc[d, u].dropna()
        mc = mc[mc > 0]
        wB = mc / mc.sum()
        rk = mc.rank(ascending=False, method='first')
        tiers = {}
        for c in mc.index:
            r = rk[c]
            if c == '2330':
                t = 'R1'
            elif r <= 5:
                t = 'R2_5'
            elif r <= 50:
                t = 'R6_50'
            else:
                t = 'R51'
            tiers[c] = t
        # revenue of month per-1 is filed up to midnight of deadline day D: known only for d > D
        lag = 1 if d > self.deadline(per) else 2
        months = [per - k for k in (lag, lag + 1, lag + 2)]
        rv = self.rev.reindex(index=months, columns=mc.index)
        revs = rv.mean().where(rv.notna().sum() >= 2)
        r52 = self.r52.loc[d, mc.index]
        wT = wB.copy()
        Z = {}
        elig_info = {}
        for t in ('R2_5', 'R6_50', 'R51'):
            mem = [c for c in mc.index if tiers[c] == t]
            el = [c for c in mem if not self.fin.get(c, False)]
            if len(el) < 2:
                continue
            ok = [c for c in el if np.isfinite(r52.get(c, np.nan)) and np.isfinite(revs.get(c, np.nan))]
            pool = mem if getattr(self, 'pct_all', False) else el
            p52 = r52[pool].rank(pct=True)[el]
            prv = revs[pool].rank(pct=True)[el]
            sc = 0.5 * p52 + 0.5 * prv
            z = pd.Series(0.0, index=el)
            n = len(ok)
            if n >= 2:
                order = sorted(ok, key=lambda c: (sc[c], revs[c], c))
                r = pd.Series(np.arange(1, n + 1), index=order)
                z[order] = (r - (n + 1) / 2) / np.sqrt((n * n - 1) / 12)
            Z.update(z.to_dict())
            w0 = wB[el]
            w1 = w0 * np.exp(lam * z[el])
            wT[el] = w1 / w1.sum() * w0.sum()
            elig_info[t] = (el, z[el].values, w0.values)
        res = {'B': wB, 'T': wT, 'tiers': tiers, 'z': Z}
        if perms:
            rng = np.random.default_rng(seed)
            wP = wB.copy()
            for t, (el, z, w0) in elig_info.items():
                zz = np.array(z)
                P = np.array([rng.permutation(zz) for _ in range(perms)])
                W = w0[None, :] * np.exp(lam * P)
                W = W / W.sum(1, keepdims=True) * w0.sum()
                wP[el] = W.mean(0)
            res['P'] = wP
        return res

    # ---------- engine ----------
    def window_static(self, w, d0, d1):
        """zero-cost static-weight return close d0 -> close d1"""
        cl = self.clf
        rr = (1 + self.rt.loc[d0:d1].iloc[1:].fillna(0)).prod() - 1
        return float((w * rr.reindex(w.index).fillna(0)).sum())

    def run(self, wlist, dates, cost=(CB, CS), end=None):
        """wlist: list of target weight Series at dates[t] (signal close). Trade at next-day open.
        Returns list of window returns (dates[t] close -> dates[t+1] close), turnover, cost."""
        days = self.days
        h = pd.Series(dtype=float)  # value holdings
        cash = 1.0
        out = []
        cb, cs = cost
        for t, d in enumerate(dates):
            nav0 = h.sum() + cash
            d1 = dates[t + 1] if t + 1 < len(dates) else (end if end is not None else days[-1])
            pos = days.get_loc(d)
            if pos + 1 >= len(days) or d1 <= d:
                break
            e = days[pos + 1]
            # overnight with old holdings
            if len(h):
                o = self.op.loc[e, h.index]; c = self.cl.loc[e, h.index]; r = self.rt.loc[e, h.index]
                on = (1 + r) * o / c
                # delisted: no price at e and none later -> cash at last close minus sell cost
                fut = self.cl.loc[e:, h.index].notna().any()
                dead = ~fut
                if dead.any():
                    cash += (h[dead] * (1 - cs)).sum()
                    h = h[~dead]; on = on[~dead]; o = o[~dead]
                on = on.where(np.isfinite(on), 1.0)
                h = h * on
            nav = h.sum() + cash
            w = wlist[t]
            ope = self.op.loc[e]
            tradable = lambda c: np.isfinite(ope.get(c, np.nan))
            frozen = [c for c in h.index if not tradable(c)]
            fval = h[frozen].sum() if frozen else 0.0
            wt = w[[c for c in w.index if tradable(c)]]
            wt = wt / wt.sum()
            target = wt * (nav - fval)
            allc = target.index.union(h.index.difference(frozen))
            cur = h.reindex(allc).fillna(0)
            tg = target.reindex(allc).fillna(0)
            diff = tg - cur
            buy = diff.clip(lower=0).sum(); sell = (-diff).clip(lower=0).sum()
            cst = buy * cb + sell * cs
            turn = (buy + sell) / 2 / nav
            newh = tg[tg > 0]
            if frozen:
                newh = pd.concat([newh, h[frozen]])
            h = newh * (1 - cst / (nav - 0))
            cash = 0.0 if True else cash
            # leftover: positions sum = nav - cst approx; cash residue
            cash = (nav - cst) - h.sum()
            # intraday e
            o = self.op.loc[e, h.index]; c = self.cl.loc[e, h.index]
            intr = (c / o).where(np.isfinite(c / o), 1.0)
            h = h * intr
            if d1 > e:
                rr = (1 + self.rt.loc[e:d1, h.index].iloc[1:].fillna(0)).prod()
                h = h * rr
            nav1 = h.sum() + cash
            out.append({'d': d, 'd1': d1, 'R': nav1 / nav0 - 1, 'turn': turn, 'cost': cst / nav})
            # renormalise
            h = h / nav1; cash = cash / nav1
        return out

    def r0050(self, d0, d1):
        b = self.b50
        return b.asof(d1) / b.asof(d0) - 1
