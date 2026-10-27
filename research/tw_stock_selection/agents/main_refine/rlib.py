#!/usr/bin/env python3
"""main_refine 共用：改良版選股計畫＋事件式日頻引擎（停損、換第 11 名、隔月換股、緩衝續抱）＋日曆月統計。
資料、股池、分數、漲停判定、成本、價格陣列全部 import revenue_longhistory/lhlib.py（不改原檔）。"""
import math, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'revenue_longhistory'))
sys.dont_write_bytecode = True
import lhlib as L  # noqa: E402
from btlib import tstat  # noqa: E402

IDX, COLS, N = L.IDX, L.COLS, L.N
LIMDN = L.GAPRET <= -L.LIM_ERA + L.LTOL          # 開盤跌停（賣不掉）


def rev_level_long():
    d = pd.read_csv(L.LONG, dtype={'code': str, 'market': str, 'name': str})
    d['p'] = pd.PeriodIndex(pd.to_datetime(d['month']), freq='M')
    d = d.sort_values(['p', 'market']).drop_duplicates(['p', 'code'], keep='first')
    d = d[d['code'].isin(COLS)]
    Lv = d.pivot_table(index='p', columns='code', values='revenue_k_twd', aggfunc='last').sort_index()
    return Lv.reindex(pd.period_range(Lv.index.min(), Lv.index.max(), freq='M'))


class Frames:
    """給定訊號日 dates（每個 d 用 Period(d)−1 月為止的營收），建立股池與各橫截面（邏輯同 lhlib.Setup）。"""
    def __init__(self, dates, YOY):
        self.dates = list(dates); self.DATES = pd.DatetimeIndex(self.dates)
        self.YOY = YOY; self.YOY3 = YOY.rolling(3, min_periods=3).mean()
        self.mm = [pd.Period(d, 'M') - 1 for d in self.dates]
        self.yoy3 = self.bym(self.YOY3)
        v60 = L.v60_fix.reindex(self.DATES); px = L.C.reindex(self.DATES); base = (v60 >= 1e7) & (px >= 10)
        self.t300 = base & (v60.where(base).rank(axis=1, ascending=False) <= 300)
        self.s52 = L.s52_fix.reindex(self.DATES)
        m = pd.DataFrame(False, index=self.DATES, columns=COLS)
        for _, r in L.dsp.iterrows():
            if r['code'] in m.columns and pd.notna(r['start']) and pd.notna(r['end']):
                m.loc[(m.index >= r['start']) & (m.index <= r['end']), r['code']] = True
        self.inD = m

    def bym(self, M):
        return pd.DataFrame(M.reindex(self.mm).to_numpy(), index=self.DATES, columns=M.columns).reindex(columns=COLS)

    def legs(self, LEV):
        Y, Y3 = self.YOY, self.YOY3
        out = {'yoy3': self.yoy3, 'yoy1': self.bym(Y), 'yoy6': self.bym(Y.rolling(6, min_periods=6).mean()),
               'accel': self.bym(Y3 - Y3.shift(3))}
        Lv = LEV.reindex(columns=Y.columns)
        s3 = Lv.rolling(3, min_periods=3).sum(); mx = s3.rolling(12, min_periods=12).max()
        out['hi12'] = self.bym((s3 / mx).where(mx > 0))
        return out

    def plan(self, leg, w52=0.5):
        """回傳 [(ex, cand_order)]：與 lhlib.Setup.selection 相同的合格判定與排序（含同分處理）。"""
        el = self.t300 & self.s52.notna() & ~self.inD & self.yoy3.notna() & leg.notna()
        r52 = self.s52.where(el).rank(axis=1, pct=True); rrv = leg.where(el).rank(axis=1, pct=True)
        sc = w52 * r52 + (1 - w52) * rrv
        SC = sc.to_numpy(); EL = el.to_numpy(); out = []
        for k, d in enumerate(self.dates):
            ex = L.pos(d) + 1
            if ex >= len(IDX): break
            trad = np.isfinite(L.OPEN_RAW[ex]) & np.isfinite(L.ADJ_RAW[ex])
            ok = EL[k] & trad & np.isfinite(SC[k])
            if ok.sum() < 50: continue
            order = SC[k].copy(); order[~ok] = np.nan
            cand = np.argsort(-np.nan_to_num(order, nan=-9))[:int(ok.sum())]
            out.append((ex, cand))
        return out


def base_dates(YOY, start='2005-01-01'):
    S = L.Setup(YOY, start=start); return S


def monthend_dates(YOY, start='2006-01-01'):
    """R5：每月最後一個有效交易日（有值檔數 ≥300）為訊號日；用 Period(d)−1 月的營收＝下個月換股時的『上上個月』。"""
    YOY3 = YOY.rolling(3, min_periods=3).mean()
    cnt = YOY.notna().sum(axis=1); last_full = cnt[cnt >= 0.8 * cnt.median()].index.max()
    ii = IDX[(IDX >= pd.Timestamp(start)) & L.GOOD.to_numpy()]
    s = pd.Series(ii, index=ii)
    ds = [g.iloc[-1] for _, g in s.groupby([ii.year, ii.month])]
    keep = []
    for d in ds:
        m = pd.Period(d, 'M') - 1
        if YOY3.index.min() <= m <= last_full and YOY3.reindex([m]).notna().sum(axis=1).iloc[0] > 100 and L.pos(d) + 1 < len(IDX):
            keep.append(d)
    return keep


def simulate(plan, buy, sell, t0, t1, n=10, every=1, buffer_k=None, stop=None, stop_mode='cash', log=None):
    """plan: [(ex, cand)]；every=2 隔月換股；buffer_k: 續抱排名門檻；stop: 停損幅度（0.15）；stop_mode: 'cash'／'next'。
    回傳 (nav Series（index 從 t0−1，起始 1.0）, info)。stop=None, every=1, buffer_k=None 時與 lhlib.sim_portfolio 完全相同。"""
    AO, AC, OR = L.AO, L.AC, L.OPEN_RAW
    GU = L.GAP_UP_OPEN_ERA
    reb = {}
    for q, (ex, cand) in enumerate([p for p in plan if t0 <= p[0] <= t1]):
        if q % every == 0: reb[ex] = cand
    u = np.zeros(N); cash = 1.0; nav = np.empty(t1 - t0 + 2); nav[0] = 1.0
    entry = np.full(N, np.nan); held = set(); last_cand = None; stopped = set(); pending = set()
    nstop = nrepl = 0; tov = []
    for t in range(t0, t1 + 1):
        if t in reb:
            cand = reb[t]
            if buffer_k is not None and held:
                rk = {j: r for r, j in enumerate(cand)}
                pick = [j for j in held if j in rk and rk[j] < buffer_k]
                for j in cand:
                    if len(pick) >= n: break
                    if j in pick: continue
                    if GU[t, j] and j not in held: continue
                    pick.append(j)
            else:
                pick = []
                for j in cand:
                    if GU[t, j] and j not in held: continue
                    pick.append(j)
                    if len(pick) == n: break
            pk = np.array(pick, dtype=int)
            Pz = AO[t]
            h = np.where(u > 0, u * np.nan_to_num(Pz), 0.0)
            Vt = h.sum() + cash
            tgt = np.zeros(N); w = 1.0 / len(pk); Vn = Vt
            for _ in range(6):
                tgt[:] = 0; tgt[pk] = w * Vn
                cost = buy * np.clip(tgt - h, 0, None).sum() + sell * np.clip(h - tgt, 0, None).sum()
                Vn = Vt - cost
            tov.append(np.clip(h - tgt, 0, None).sum() / Vt if Vt > 0 else 0)
            newe = np.full(N, np.nan)
            for j in pk:
                newe[j] = entry[j] if (u[j] > 0 and j not in pending and np.isfinite(entry[j])) else AO[t, j]
            entry = newe
            u = np.zeros(N); u[pk] = tgt[pk] / Pz[pk]; cash = 0.0
            held = set(int(j) for j in pk); last_cand = cand; stopped = set(); pending = set()
        elif pending:
            done = []
            for j in sorted(pending):
                if u[j] <= 0: done.append(j); continue
                if not np.isfinite(OR[t, j]) or LIMDN[t, j]: continue
                proceeds = u[j] * AO[t, j] * (1 - sell)
                if log is not None: log.append(('sell', t, int(j), float(entry[j]), float(AO[t, j])))
                u[j] = 0.0; entry[j] = np.nan; held.discard(j); done.append(j); nstop += 1
                if stop_mode == 'next' and last_cand is not None:
                    rep = None
                    for j2 in last_cand:
                        if u[j2] > 0 or j2 in stopped or j2 in held: continue
                        if not (np.isfinite(OR[t, j2]) and np.isfinite(L.ADJ_RAW[t, j2])) or GU[t, j2]: continue
                        rep = j2; break
                    if rep is not None:
                        u[rep] = proceeds / (1 + buy) / AO[t, rep]
                        if log is not None: log.append(('buy', t, int(rep), float(AO[t, rep])))
                        entry[rep] = AO[t, rep]; held.add(int(rep)); nrepl += 1
                    else:
                        cash += proceeds
                else:
                    cash += proceeds
            for j in done: pending.discard(j)
        nav[t - t0 + 1] = np.nansum(u * AC[t]) + cash
        if stop is not None:
            hj = np.where(u > 0)[0]
            for j in hj:
                if j in pending: continue
                if AC[t, j] <= entry[j] * (1 - stop):
                    pending.add(int(j)); stopped.add(int(j))
                    if log is not None: log.append(('trig', t, int(j), float(entry[j]), float(AC[t, j])))
    s = pd.Series(nav, index=IDX[t0 - 1:t1 + 1])
    return s, {'n_rebalances': len(reb), 'n_stops': nstop, 'n_replacements': nrepl,
               'turnover_per_rebalance': round(float(np.mean(tov[1:])), 3) if len(tov) > 1 else None}


# ---------------- 統計 ----------------
def mret(nav):
    """日曆月報酬（月底淨值比）。"""
    me = nav.groupby([nav.index.year, nav.index.month]).tail(1)
    r = me.pct_change(); r.iloc[0] = me.iloc[0] / nav.iloc[0] - 1
    r.index = pd.PeriodIndex(me.index, freq='M'); return r

E50 = L.E50
def seg_stats(nav, a, b, bench=E50):
    r = nav.pct_change().dropna(); r = r[(r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b))]
    if len(r) < 20: return None
    prev = nav.index[nav.index.get_loc(r.index[0]) - 1]
    span = (r.index[-1] - prev).days / 365.25
    eq = (1 + r).cumprod(); cagr = float(eq.iloc[-1] ** (1 / span) - 1); dd = float((eq / eq.cummax() - 1).min())
    m = mret(nav); m = m[(m.index >= pd.Period(a, 'M')) & (m.index <= pd.Period(b, 'M'))]
    bn = bench.reindex(nav.index); bm = mret(bn).reindex(m.index)
    o = {'CAGR': round(cagr, 4), 'maxDD': round(dd, 4), 'Calmar': round(cagr / abs(dd), 3) if dd < 0 else None,
         'vol': round(float(r.std() * math.sqrt(252)), 4), 'n_months': int(len(m)),
         't_vs_0050_m': tstat(m - bm), 'excess_vs_0050_pct_m': round(float((m - bm).mean() * 100), 3)}
    return o

def paired(nav, base, a, b):
    m = mret(nav); mb = mret(base).reindex(m.index); d = (m - mb)
    d = d[(d.index >= pd.Period(a, 'M')) & (d.index <= pd.Period(b, 'M'))].dropna()
    rd = (nav.pct_change() - base.reindex(nav.index).pct_change()).dropna()
    rd = rd[(rd.index >= pd.Timestamp(a)) & (rd.index <= pd.Timestamp(b) + pd.offsets.MonthEnd(0))]
    return {'diff_pct_m': round(float(d.mean() * 100), 3), 't_paired_m': tstat(d), 'n_months': int(len(d)),
            'hit': round(float((d > 0).mean()), 3), 't_paired_daily': tstat(rd)}

def yearly(nav):
    r = nav.pct_change().dropna()
    return {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in r.groupby(r.index.year)}
