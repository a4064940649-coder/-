#!/usr/bin/env python3
"""tpex_main 共用：上市／上櫃／合併三種股池的主策略 M2 引擎。

選股、成交、成本、統計邏輯逐行照抄 agents/revenue_longhistory/lhlib.py（它本身照抄 exec_rebuild/exlib.py），
只把「面板」與「流動性前 K 名」參數化；換股日一律沿用 lhlib.Setup（上市、長期營收）的日期。
"""
import math, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / 'revenue_longhistory'))
sys.path.insert(0, str(HERE.parents[1]))
import lhlib as L  # noqa: E402  （載入上市面板、0050、處置表、成本函式）
from btlib import tstat  # noqa: E402

TPP = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/tpex_panel')
SCR = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data')
IDX = L.IDX
E50 = L.E50
cost_rates = L.cost_rates


def load_tpex():
    return {k: pd.read_parquet(TPP / f'{k}.parquet').reindex(IDX) for k in ['adj', 'close_raw', 'value', 'open', 'high']}


def twse_panel():
    return {'adj': L.adj, 'close_raw': L.C, 'value': L.V, 'open': L.O, 'high': L.H}


def combine(tw, tp):
    """合併上市＋上櫃。轉上市股：上櫃時期接在上市時期之前，還原指數以「上市首日收盤／上櫃末日收盤」接續。"""
    both = [c for c in tp['adj'].columns if c in tw['adj'].columns]
    out = {}
    for k in tw:
        a = tw[k]; b = tp[k]
        only_b = b[[c for c in b.columns if c not in a.columns]]
        m = pd.concat([a, only_b], axis=1)
        out[k] = m
    for c in both:
        ta = tw['adj'][c]; first = ta.first_valid_index()
        pb = tp['adj'][c].loc[:first - pd.Timedelta(days=1)].dropna()
        if first is None or not len(pb): continue
        last_b = pb.index[-1]
        cr_b = tp['close_raw'].at[last_b, c]; cr_a = tw['close_raw'].at[first, c]
        k_ = ta[first] / (pb.iloc[-1] * cr_a / cr_b)
        mask = out['adj'].index < first
        for k in ('close_raw', 'value', 'open', 'high'):
            out[k].loc[mask, c] = tp[k].loc[mask, c]
        out['adj'].loc[mask, c] = tp['adj'].loc[mask, c] * k_
    return out


class Universe:
    def __init__(self, panel, topk, name, lenient=False, good=None):
        self.name = name; self.topk = topk
        adj = panel['adj']; C = panel['close_raw']; V = panel['value']; O = panel['open']
        self.COLS = adj.columns; self.N = len(self.COLS)
        self.C = C
        self.AC = adj.ffill().to_numpy()
        fac = adj / C
        AO_raw = (O * fac).to_numpy()
        self.AO = np.where(np.isfinite(AO_raw), AO_raw, np.vstack([np.full((1, self.N), np.nan), self.AC[:-1]]))
        self.ADJ_RAW = adj.to_numpy(); self.OPEN_RAW = O.to_numpy()
        AC_prev = np.vstack([np.full((1, self.N), np.nan), self.AC[:-1]])
        GAPRET = AO_raw / AC_prev - 1
        LIM = np.where(IDX < pd.Timestamp('2015-06-01'), 0.07, 0.10)[:, None]
        self.GU = GAPRET >= LIM - L.LTOL
        g = good if good is not None else (adj.notna().sum(axis=1) >= 300)
        ag = adj.loc[g]; vg = V.loc[g]
        if lenient:
            self.s52 = (ag / ag.rolling(252, min_periods=120).max()).reindex(IDX)
            self.v60 = vg.rolling(60, min_periods=20).mean().reindex(IDX)
        else:
            self.s52 = (ag / ag.rolling(252).max()).reindex(IDX)
            self.v60 = vg.rolling(60).mean().reindex(IDX)

    def setup(self, YOY, dates):
        self.dates = list(dates); self.DATES = pd.DatetimeIndex(self.dates)
        YOY3 = YOY.rolling(3, min_periods=3).mean()
        mm = [pd.Period(d, 'M') - 1 for d in self.dates]
        self.yoy3 = pd.DataFrame(YOY3.reindex(mm).to_numpy(), index=self.DATES, columns=YOY3.columns).reindex(columns=self.COLS)
        v60 = self.v60.reindex(self.DATES); px = self.C.reindex(self.DATES); base = (v60 >= 1e7) & (px >= 10)
        self.v60d = v60
        self.tk = base & (v60.where(base).rank(axis=1, ascending=False) <= self.topk)
        self.s52d = self.s52.reindex(self.DATES)
        m = pd.DataFrame(False, index=self.DATES, columns=self.COLS)
        for _, r in L.dsp.iterrows():
            if r['code'] in m.columns and pd.notna(r['start']) and pd.notna(r['end']):
                m.loc[(m.index >= r['start']) & (m.index <= r['end']), r['code']] = True
        self.inD = m
        return self

    def selection(self, n, w52=0.5):
        el = self.tk & self.s52d.notna() & ~self.inD & self.yoy3.notna()
        r52 = self.s52d.where(el).rank(axis=1, pct=True); rrv = self.yoy3.where(el).rank(axis=1, pct=True)
        sc = w52 * r52 + (1 - w52) * rrv
        events, pools = {}, {}; held = set(); nblk = 0; sig = {}
        for d in self.dates[:-1]:
            i = L.pos(d); ex = i + 1
            okb = el.loc[d].to_numpy()
            trad = np.isfinite(self.OPEN_RAW[ex]) & np.isfinite(self.ADJ_RAW[ex])
            ok = okb & trad & np.isfinite(sc.loc[d].to_numpy())
            if ok.sum() < 50: continue
            order = sc.loc[d].to_numpy().copy(); order[~ok] = np.nan
            cand = np.argsort(-np.nan_to_num(order, nan=-9))[:int(ok.sum())]
            pick = []
            for j in cand:
                if self.GU[ex, j] and j not in held:
                    nblk += 1; continue
                pick.append(j)
                if len(pick) == n: break
            events[ex] = ('open', np.array(pick)); pools[ex] = ('open', np.where(ok)[0]); sig[ex] = d
            held = set(pick)
        self.nblk = nblk; self.sig = sig
        return events, pools

    def sim(self, events, buy, sell, t0, t1):
        N = self.N; AO = self.AO; AC = self.AC
        u = np.zeros(N); cash = 1.0; nav = np.empty(t1 - t0 + 1); tov = []
        for t in range(t0, t1 + 1):
            if t in events:
                mode, pk = events[t]
                Pz = AO[t]
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


def yoy_long(cols):
    d = pd.read_csv(L.LONG, dtype={'code': str, 'market': str, 'name': str})
    d['p'] = pd.PeriodIndex(pd.to_datetime(d['month']), freq='M')
    d = d.sort_values(['p', 'market']).drop_duplicates(['p', 'code'], keep='first')
    d = d[d['code'].isin(cols)]
    Y = d.pivot_table(index='p', columns='code', values='revenue_k_twd', aggfunc='last').sort_index()
    Y = Y.reindex(pd.period_range(Y.index.min(), Y.index.max(), freq='M'))
    prev = Y.shift(12)
    return ((Y / prev - 1) * 100).where(prev > 0)


def fin_codes(cols):
    sl = pd.read_csv(SCR / 'stock_list.csv', dtype=str)
    return set(sl.loc[sl['industry'] == '金融保險業', 'code']) | {c for c in cols if c[:2] == '28' or c in ('5876', '5880')}


def period_returns(nav, DATES):
    nd = nav.reindex(DATES)
    return (nd.shift(-1) / nd - 1).iloc[:-1].dropna()


def in_segs(idx, segs):
    m = np.zeros(len(idx), bool)
    for a, b in segs: m |= (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))
    return m


def stats(nav, segs, DATES, ew=None, ref=None):
    """lhlib.daily_stats（日頻年化／回撤、月窗對 0050、對同池等權 t）＋ 對原上市版（ref）配對月 t。"""
    o = L.daily_stats(nav, segs, DATES, E50, ew)
    if o is None: return None
    keep = ('CAGR', 'vol', 'maxDD', 'years', 'n_months', 'excess_vs_0050_pct_m', 't_vs_0050_m', 'excess_vs_EW_pct_m', 't_vs_EW_m', 'EW_CAGR', '0050_CAGR', 'yearly_pct', 'EW_yearly_pct', '0050_yearly_pct')
    o = {k: o[k] for k in keep if k in o}
    if ref is not None:
        pr = period_returns(nav, DATES); rp = period_returns(ref, DATES)
        j = pd.concat([pr, rp], axis=1, join='inner').dropna()
        j = j[in_segs(j.index, segs)]
        d = j.iloc[:, 0] - j.iloc[:, 1]
        o['paired_vs_TWSE_pct_m'] = round(float(d.mean() * 100), 2); o['t_paired_vs_TWSE_m'] = tstat(d); o['paired_n'] = int(len(d))
        o['paired_hit'] = round(float((d > 0).mean()), 3)
        rr = ref.pct_change(); m = in_segs(rr.index, segs) & rr.notna().to_numpy()
        eq = (1 + rr[m]).cumprod()
        o['TWSE_CAGR'] = round(float(eq.iloc[-1] ** (1 / o['years']) - 1), 4) if len(eq) else None
    return o
